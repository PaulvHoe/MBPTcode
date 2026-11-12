"""The closed-shell parquet driver is the spin-orbital one, spin-adapted.

On a closed shell (water and ammonia 6-31G) the restricted driver's four
channels and the spin-orbital driver's two describe one two-body problem: the
spin-orbital eh, ee and hh spectra are the restricted singlets and three
copies of the triplets, and the spin-orbital EcRPA is the restricted
singlet + 3 triplet (`ParquetResult.ec_eh`, `ec_pp`), at every two-body
iteration with the same damping. The two self-energies are Quack's
R_Parquet_self_energy_diag and G_Parquet_self_energy_diag, other partial
fractions of the same terms under the one-body regularizer: at s_1b = 1e8
they coincide, so FLEX and osPA give the same quasiparticle energies, and
the linearized evPA, where both drivers run Quack's own route, the same
first iterations (the two regularized forms part by 2e-11 Ha at the first,
and every iteration feeds its energies back into the denominators); at
s_1b = 50 they part by meV (the restricted module docstring's FLEX RPA
numbers). The spin-orbital driver gets the RHF's alpha/beta pairs, as
`run_parquet_so` hands them for an RHF.

The two drivers' converged loops are not compared: each tests the change of
its own vertices (the four spin-adapted ones, or the two spin-orbital ones),
so they stop at other counts. Fixed counts are compared instead, with the
eh and pp Tamm-Dancoff switches together and apart (each reaches its own
problem), and the linearized evPA with and without one-body DIIS (whose
extrapolation both drivers apply alike).

Run: python tests/test_parquet_spin_adaptation.py, or under pytest.
"""
import functools
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import ao2mo, gto, scf

from src.Base.numerics.constants import HARTREE_TO_EV, PARQUET_VERTEX_DAMPING
from src.SingleReference.Parquet import restricted, spin_orbital
from tests.parquet_quack_newton import quack_newton

#: QuAcK's geometries (Angstrom), 6-31G.
GEOMETRIES = {
    'water': 'O 0 0 0; H 0.7571 0 0.5861; H -0.7571 0 0.5861',
    'ammonia': 'N 0 0 0; H 0.3816 0.9375 0; H 0.3816 -0.4687 0.8119; H 0.3816 -0.4687 -0.8119',
}
BASIS = '6-31g'
SCF_CONV_TOL = 1e-12
#: Ha. The same quantity in the two drivers (spectra, EcRPA, and at
#: s_1b = 1e8 eqp): 9.5e-13 at most measured (osPA RPA EcRPA, water);
#: other GEMM shapes in each, x10 for another BLAS.
CROSS_DRIVER = 1e-11
#: Z at s_1b = 1e8 there, absolute: 9.4e-12 measured (FLEX RPA, water, a Z of
#: 1.8 beside a pole).
CROSS_DRIVER_Z = 1e-10
#: Ha. The first two linearized evPA iterations at s_1b = 1e8: 1.1e-10 at most
#: measured (ammonia, iteration 2; iteration 1 1.7e-11, the regularized forms'
#: own difference), growing tenfold per iteration after.
CROSS_DRIVER_EV = 1e-9
#: One-body iterations of the linearized evPA held to CROSS_DRIVER_EV.
EV_ITERATIONS = 2
#: Two-body iterations per one-body iteration in the fixed-count comparisons.
TWO_BODY_CAP = 5
#: meV. The FLEX RPA (linearized) HOMO of the spin-orbital driver against the
#: restricted one at s_1b = 50, as the restricted module docstring states it:
#: water 2.4, ammonia 1.6 (2.425 and 1.566 measured).
HOMO_SPLIT_MEV = {'water': 2.4, 'ammonia': 1.6}
#: meV. How far the measured split may lie from the stated one (its rounding).
HOMO_SPLIT_ROUNDING_MEV = 0.05

T, F = True, False
#: Fixed-count runs of both drivers: (scheme, options).
RUNS = {
    'flex_tda': ('flex', dict(tda_eh=T, tda_pp=T, eta_1b=1e8, eta_2b=1.0,
                              qp_solver='linearized')),
    'flex_rpa': ('flex', dict(tda_eh=F, tda_pp=F, eta_1b=1e8, eta_2b=1.0,
                              qp_solver='linearized')),
    'os_tda_cap': ('os', dict(tda_eh=T, tda_pp=T, eta_1b=1e8, eta_2b=1.0,
                              qp_solver='linearized', max_it_2b=TWO_BODY_CAP, conv_2b=1e-12,
                              damping=PARQUET_VERTEX_DAMPING)),
    'os_rpa_cap': ('os', dict(tda_eh=F, tda_pp=F, eta_1b=1e8, eta_2b=1.0,
                              qp_solver='linearized', max_it_2b=TWO_BODY_CAP, conv_2b=1e-12,
                              damping=PARQUET_VERTEX_DAMPING)),
    'flex_ehtda_pprpa': ('flex', dict(tda_eh=T, tda_pp=F, eta_1b=1e8, eta_2b=1.0,
                                      qp_solver='linearized')),
    'flex_ehrpa_pptda': ('flex', dict(tda_eh=F, tda_pp=T, eta_1b=1e8, eta_2b=1.0,
                                      qp_solver='linearized')),
}
#: Ha. |EcRPA| of a channel under the TDA, rounding noise (1e-13 measured).
TDA_EC_NOISE = 1e-6
#: One-body DIIS length of the linearized evPA with extrapolation.
ONE_BODY_DIIS = 2
#: Ha. The extrapolated energies of that evPA's second iteration lie at least
#: this far from the linearized ones the iteration solved for (2.2e-3 and
#: 3.0e-4 measured, water and ammonia): the extrapolation acted.
DIIS_ACTED = 1e-6


@functools.lru_cache(maxsize=None)
def rhf(molecule):
    mol = gto.M(atom=GEOMETRIES[molecule], basis=BASIS, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = SCF_CONV_TOL
    mf.kernel()
    assert mf.converged, molecule
    return mf


@functools.lru_cache(maxsize=None)
def both_layouts(molecule):
    """((eps, eri, nO) closed shell, (eps, eri, nO, pairs) over interleaved
    spin orbitals) of the same RHF."""
    mf = rhf(molecule)
    n = mf.mo_coeff.shape[1]
    eri = ao2mo.full(mf.mol, mf.mo_coeff, compact=False).reshape((n,) * 4).transpose(0, 2, 1, 3)
    eri = np.ascontiguousarray(eri)
    eps = np.asarray(mf.mo_energy)
    nO = mf.mol.nelectron // 2
    spin = np.tile([0, 1], n)
    orbital = np.repeat(np.arange(n), 2)
    same = spin[:, None] == spin[None, :]
    eri_so = np.ascontiguousarray(eri[np.ix_(orbital, orbital, orbital, orbital)]
                                  * (same[:, None, :, None] & same[None, :, None, :]))
    return (eps, eri, nO), (np.repeat(eps, 2), eri_so, 2 * nO, np.arange(2 * n).reshape(n, 2))


def both_drivers(molecule, scheme, options):
    """(restricted result, spin-orbital result) of one run on both layouts."""
    (eps, eri, nO), (eps_so, eri_so, nO_so, pairs) = both_layouts(molecule)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = restricted.ev_parquet(eps, eri, nO, scheme, **options)
        g = spin_orbital.ev_parquet_so(eps_so, eri_so, nO_so, scheme, spin_pairs=pairs,
                                       **options)
    return r, g


def spin_adapted(spectra, channel):
    """The restricted channel as a spin-orbital one: its singlet eigenvalues
    and three copies of its triplet ones, ascending."""
    return np.sort(np.concatenate([getattr(spectra, f'{channel}_singlet').om]
                                  + [getattr(spectra, f'{channel}_triplet').om] * 3))


def assert_same_two_body(r, g, label, bar=CROSS_DRIVER):
    """The spin-orbital spectra and EcRPA are the restricted ones spin-adapted,
    within `bar` (Ha)."""
    for channel in ('eh', 'ee', 'hh'):
        om_r, om_g = spin_adapted(r.spectra, channel), np.sort(getattr(g.spectra, channel).om)
        assert om_r.shape == om_g.shape, (label, channel, om_r.shape, om_g.shape)
        assert np.abs(om_r - om_g).max() <= bar, (label, channel)
    assert abs(r.ec_eh - g.ec_eh) <= bar, (label, r.ec_eh, g.ec_eh)
    assert abs(r.ec_pp - g.ec_pp) <= bar, (label, r.ec_pp, g.ec_pp)


@pytest.mark.parametrize('molecule', sorted(GEOMETRIES))
@pytest.mark.parametrize('run', sorted(RUNS))
def test_one_closed_shell_two_drivers(molecule, run):
    """FLEX and osPA (fixed two-body counts) at s_1b = 1e8: the spectra, the
    EcRPA floats, eqp and Z of the two drivers, the alpha/beta copies equal
    to the bit."""
    scheme, options = RUNS[run]
    r, g = both_drivers(molecule, scheme, options)
    label = (molecule, run)
    assert r.n_it_2b == g.n_it_2b, label
    assert_same_two_body(r, g, label)
    assert np.array_equal(g.eqp[0::2], g.eqp[1::2]), label
    assert np.abs(r.eqp - g.eqp[0::2]).max() <= CROSS_DRIVER, label
    assert np.abs(r.z - g.z[0::2]).max() <= CROSS_DRIVER_Z, label
    # a full channel has a correlation energy, a TDA one rounding noise: the
    # comparison has teeth, and each switch reached its own problem
    if options['tda_eh']:
        assert abs(r.ec_eh) <= TDA_EC_NOISE, (label, r.ec_eh)
    else:
        assert r.ec_eh < -0.1, (label, r.ec_eh)
        singlet, triplet = r.ec_eh_spin
        assert r.ec_eh == singlet + 3.0 * triplet, label
    if options['tda_pp']:
        assert abs(r.ec_pp) <= TDA_EC_NOISE, (label, r.ec_pp)
    else:
        assert r.ec_pp < -0.05, (label, r.ec_pp)


@pytest.mark.parametrize('molecule', sorted(GEOMETRIES))
def test_the_linearized_evpa_of_both_drivers(molecule):
    """Quack's own route in both drivers, the linearized evPA at s_1b = 1e8
    with fixed two-body counts: the first EV_ITERATIONS iterations agree, and
    the spin-orbital pairs stay degenerate to the bit."""
    options = dict(tda_eh=T, tda_pp=T, eta_1b=1e8, eta_2b=1.0, qp_solver='linearized',
                   max_it_2b=3, conv_2b=1e-12, max_it_1b=EV_ITERATIONS, conv_1b=1e-12,
                   damping=PARQUET_VERTEX_DAMPING)
    r, g = both_drivers(molecule, 'ev', options)
    assert r.n_it_1b == g.n_it_1b == EV_ITERATIONS, molecule
    for k, (hr, hg) in enumerate(zip(r.history, g.history)):
        assert np.array_equal(hg['eqp'][0::2], hg['eqp'][1::2]), (molecule, k)
        gap = np.abs(hr['eqp'] - hg['eqp'][0::2]).max()
        assert gap <= CROSS_DRIVER_EV, (molecule, k, gap)
    # the last spectra stand on the previous iteration's energies, as close as those
    assert_same_two_body(r, g, molecule, CROSS_DRIVER_EV)


@pytest.mark.parametrize('molecule', sorted(GEOMETRIES))
def test_the_one_body_diis_of_both_drivers(molecule):
    """The same linearized evPA with one-body DIIS (error eQP - eOld): both
    drivers extrapolate alike, iteration by iteration, and the extrapolation
    moved the second iteration's energies off the linearized ones."""
    options = dict(tda_eh=T, tda_pp=T, eta_1b=1e8, eta_2b=1.0, qp_solver='linearized',
                   max_it_2b=3, conv_2b=1e-12, max_it_1b=EV_ITERATIONS, conv_1b=1e-12,
                   max_diis_1b=ONE_BODY_DIIS, damping=PARQUET_VERTEX_DAMPING)
    r, g = both_drivers(molecule, 'ev', options)
    assert r.n_it_1b == g.n_it_1b == EV_ITERATIONS, molecule
    for k, (hr, hg) in enumerate(zip(r.history, g.history)):
        assert np.array_equal(hg['eqp'][0::2], hg['eqp'][1::2]), (molecule, k)
        gap = np.abs(hr['eqp'] - hg['eqp'][0::2]).max()
        assert gap <= CROSS_DRIVER_EV, (molecule, k, gap)
    for result in (r, g):
        last = result.history[-1]
        assert np.abs(last['eqp'] - last['eqp_lin']).max() > DIIS_ACTED, molecule


@pytest.mark.parametrize('molecule', sorted(GEOMETRIES))
def test_the_self_energies_part_at_finite_s1b(molecule):
    """At s_1b = 50 the two regularized self-energies are other functions:
    on the same spectra and EcRPA, FLEX RPA's HOMO lies the restricted
    docstring's meV apart."""
    options = dict(tda_eh=F, tda_pp=F, eta_1b=50.0, eta_2b=1.0, qp_solver='linearized')
    r, g = both_drivers(molecule, 'flex', options)
    assert_same_two_body(r, g, molecule)
    homo = rhf(molecule).mol.nelectron // 2 - 1
    split = abs(g.eqp[2 * homo] - r.eqp[homo]) * HARTREE_TO_EV * 1e3
    assert abs(split - HOMO_SPLIT_MEV[molecule]) <= HOMO_SPLIT_ROUNDING_MEV, (molecule, split)


@pytest.mark.parametrize('molecule', sorted(GEOMETRIES))
def test_the_front_doors_agree_on_the_two_body_problem(molecule):
    """run_parquet and run_parquet_so on the same RHF: one FLEX RPA two-body
    problem, whatever their default dampings (which act after it)."""
    mf = rhf(molecule)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # Quack's Newton: the default root finder takes a hundredfold the evaluations here
        r = restricted.run_parquet(mf, 'flex', tda_eh=F, tda_pp=F, eta_1b=1e8,
                                   qp_solver=quack_newton)
        g = spin_orbital.run_parquet_so(mf, 'flex', tda_eh=F, tda_pp=F, eta_1b=1e8,
                                        qp_solver=quack_newton)
    assert g.reference == 'RHF' and g.options['spin_pairs'] is not None, molecule
    assert_same_two_body(r, g, molecule)
    assert np.abs(r.eqp - g.eqp[0::2]).max() <= CROSS_DRIVER, molecule


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
