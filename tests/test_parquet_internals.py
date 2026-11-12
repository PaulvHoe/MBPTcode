"""The parquet drivers' pieces that the comparisons with Quack do not isolate.

* The blocked contractions (`restricted_vertices.eh_vertex_from_weights`,
  `pp_vertex_from_weights`, `pp_effective_integrals`), which hold one block
  of a vertex at a time, are their whole contraction: against an einsum
  written here, and the pp effective integrals in blocks of one orbital
  against one block, to the relative cross-path bar (other GEMM shapes
  round otherwise), with more than one block taken; a closed shell with
  one occupied orbital (no triplet hh pair) included.
* `run_parquet` turns PySCF's chemist integrals into <pq|rs> in place, and
  hands qsPA <mu nu|la si>: the arrays the lower levels receive are
  np.ascontiguousarray of the transposed integrals, to the bit; and it hands
  qsPA the mean field's own core Hamiltonian, mf.get_hcore() (water in a
  field, spin-free X2C water), not T + V, so that qsPA starts from the
  reference's Fock matrix (as `run_parquet_so` does).
* qsPA diagonalizes its orthogonalizer's overlap and its Fock matrix by
  dsyev from the upper triangle (PARQUET_QS_EIGH_DRIVER, Quack's
  diagonalize_matrix), and X^T S X = 1.
* The spin-orbital pp problem with an empty branch (one occupied or one
  virtual spin orbital) and with none at all: arrays of the empty shapes.
* Under the TDA no B block is built, bare or kernel, eh or pp, in either
  driver (None, where Quack adds an uninitialized Gamma_B that its TDA
  solve never reads): the TDA solves would not notice a zero block, its
  memory would.
* The self-energy functions the method page maps to Quack's routines, which
  the drivers do not all call, give the drivers' numbers (water/6-31G FLEX,
  s_1b = 50, linearized: the run's SigC and Z are the self-energy's at
  eHF): the closed-shell `diagonal_self_energy` (R_Parquet_self_energy_diag),
  `self_energy_at` (R_Parquet_self_energy_diag_omega) at any w as the
  whole-set object's `at`, `self_energy_parts` (dump_RParquet_self_energy)
  six terms that sum to SigC, a `DiagonalSelfEnergy` of some orbitals the
  rows of the whole one; the spin-orbital `DiagonalSelfEnergySO.diagonal`
  and `diagonal_self_energy_so` with their parts, `at_each` with its parts
  (dump_GParquet_self_energy) summing to Sigma and its derivative, `at`
  over an array of frequencies, and `self_energy_at_so`; to the relative
  cross-path bar where the paths differ.
* The SRG static self-energy of qsPA on one worker (BLAS's own threads under
  its products) is the one over two (BLAS held at one thread), as
  `tests.pins.assert_same` holds two computations.

Run: python tests/test_parquet_internals.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import scipy.linalg as la
from pyscf import gto
from pyscf.scf import uhf

from src.Base.numerics.constants import (PARQUET_EIGH_DRIVER, PARQUET_ORTHOGONALIZATION_THRESHOLD,
                                         PARQUET_QS_EIGH_DRIVER)
from src.Base.pyscf.pyscf_interface import get_two_electron_integrals_chemist
from src.SingleReference.Parquet import (restricted, restricted_bse, restricted_vertices,
                                         spin_orbital, spin_orbital_two_body)
from src.SingleReference.Parquet.pairs import pp_pair_count
from src.SingleReference.Parquet.restricted_self_energy import (TERMS, DiagonalSelfEnergy,
                                                                diagonal_self_energy,
                                                                self_energy_at,
                                                                self_energy_parts)
from src.SingleReference.Parquet.restricted_static_self_energy import static_self_energy_srg
from src.SingleReference.Parquet.spectra import RestrictedVertices, SpinOrbitalVertices
from src.SingleReference.Parquet.spin_orbital_self_energy import (DiagonalSelfEnergySO,
                                                                  diagonal_self_energy_so,
                                                                  self_energy_at_so)
from tests.pins import CROSS_PATH_RTOL, assert_same, bitwise
from tests.test_parquet_restricted_vs_quack import mo_integrals, rhf
from tests.test_parquet_so_layout import keep_core_hamiltonians, mean_field
from tests.test_parquet_spin_adaptation import both_layouts

#: The seed of the random integrals and weights of the contraction checks.
SEED = 11
#: Orbitals, doubly occupied orbitals and eh pairs of the synthetic problems.
SIZES = ((13, 5), (13, 1), (24, 5))
#: X^T S X = 1 to this (2e-15 measured).
ORTHONORMAL = 1e-12
#: Ha. The field's and X2C's core Hamiltonians differ from T + V by more
#: than this somewhere (0.011 and 0.021 measured).
OTHER_HCORE = 1e-3


def relative(got, want):
    return float(np.abs(got - want).max() / np.abs(want).max())


@pytest.mark.parametrize('norb, nO', SIZES)
def test_the_eh_vertex_in_blocks_is_the_whole_contraction(norb, nO):
    rng = np.random.default_rng(SEED)
    nS = nO * (norb - nO)
    rho = rng.standard_normal((norb, norb, nS))
    w_res, w_anti = rng.standard_normal(nS), rng.standard_normal(nS)
    got = restricted_vertices.eh_vertex_from_weights(rho, w_res, w_anti)
    want = (np.einsum('prn,sqn,n->pqrs', rho, rho, w_res)
            + np.einsum('rpn,qsn,n->pqrs', rho, rho, w_anti))
    assert relative(got, want) <= CROSS_PATH_RTOL
    phi_bytes = 8 * norb**4
    block = min(restricted_vertices.PARQUET_EH_VERTEX_BLOCK_BYTES,
                phi_bytes // restricted_vertices.PARQUET_MIN_BLOCKS_PER_VERTEX)
    assert max(1, block // (8 * norb**3)) < norb          # more than one block


@pytest.mark.parametrize('norb, nO', SIZES)
def test_the_pp_vertex_in_blocks_is_the_whole_contraction(norb, nO):
    rng = np.random.default_rng(SEED)
    nVV, nOO = pp_pair_count(norb - nO, 'singlet'), pp_pair_count(nO, 'singlet')
    rho_ee = rng.standard_normal((norb, norb, nVV))
    rho_hh = rng.standard_normal((norb, norb, nOO))
    w_ee, w_hh = rng.standard_normal(nVV), rng.standard_normal(nOO)
    got = restricted_vertices.pp_vertex_from_weights(rho_ee, w_ee, rho_hh, w_hh)
    want = (np.einsum('pqm,rsm,m->pqrs', rho_ee, rho_ee, w_ee)
            + np.einsum('pqm,rsm,m->pqrs', rho_hh, rho_hh, w_hh))
    assert relative(got, want) <= CROSS_PATH_RTOL


@pytest.mark.parametrize('norb, nO', SIZES)
@pytest.mark.parametrize('spin', ['singlet', 'triplet'])
def test_the_pp_integrals_in_blocks_of_one_orbital(norb, nO, spin, monkeypatch):
    rng = np.random.default_rng(SEED)
    eri = rng.standard_normal((norb,) * 4)
    vertices = RestrictedVertices(*(rng.standard_normal((norb,) * 4) for _ in range(4)))
    nVV, nOO = pp_pair_count(norb - nO, spin), pp_pair_count(nO, spin)
    x1, y1 = rng.standard_normal((nVV, nVV)), rng.standard_normal((nOO, nVV))
    x2, y2 = rng.standard_normal((nVV, nOO)), rng.standard_normal((nOO, nOO))
    args = (eri, vertices, x1, y1, x2, y2, nO, spin)
    monkeypatch.setattr(restricted_vertices, 'PARQUET_PP_INTEGRAL_BLOCK_BYTES', 1)
    blocked = restricted_vertices.pp_effective_integrals(*args)
    monkeypatch.setattr(restricted_vertices, 'PARQUET_PP_INTEGRAL_BLOCK_BYTES', 2**40)
    monkeypatch.setattr(restricted_vertices, 'PARQUET_MIN_BLOCKS_PER_VERTEX', 1)
    whole = restricted_vertices.pp_effective_integrals(*args)
    for got, want, n in zip(blocked, whole, (nVV, nOO)):
        assert got.shape == want.shape == (norb, norb, n)
        if n:
            assert relative(got, want) <= CROSS_PATH_RTOL


def test_the_integrals_reach_the_lower_levels_as_transposed(monkeypatch):
    """<pq|rs> in place of PySCF's (pr|qs), and qsPA's <mu nu|la si>, to the
    bit of np.ascontiguousarray(chem.transpose(0, 2, 1, 3)): chem the MO
    integrals the driver was handed, a copy kept as they came back (another
    ao2mo transformation is another computation, `tests.pins.bitwise`), and
    the AO integrals computed again (libcint computes each integral alone,
    with no BLAS reduction to round otherwise)."""
    mf = rhf('water')
    mol = mf.mol
    n = mf.mo_coeff.shape[1]
    seen, handed = {}, {}

    def chemist(*args, **kwargs):
        got = get_two_electron_integrals_chemist(*args, **kwargs)
        handed['chem'] = np.array(got)          # transposed in place after this
        return got
    monkeypatch.setattr(restricted, 'get_two_electron_integrals_chemist', chemist)

    def ev_parquet(eps_hf, eri, nO, scheme, **kwargs):
        seen['eri'] = np.array(eri)
        raise StopIteration
    monkeypatch.setattr(restricted, 'ev_parquet', ev_parquet)
    with pytest.raises(StopIteration):
        restricted.run_parquet(mf, 'os')
    chem = handed['chem'].reshape((n,) * 4)
    assert bitwise(seen['eri'], np.ascontiguousarray(chem.transpose(0, 2, 1, 3)))

    def qs_parquet(eps_hf, c_hf, s, x, hc, eri_ao, nO, **kwargs):
        seen['eri_ao'] = np.array(eri_ao)
        raise StopIteration
    monkeypatch.setattr(restricted, 'qs_parquet', qs_parquet)
    with pytest.raises(StopIteration):
        restricted.run_parquet(mf, 'qs')
    ao = mol.intor('int2e', aosym='s1')
    assert seen['eri_ao'].tobytes() == np.ascontiguousarray(ao.transpose(0, 2, 1, 3)).tobytes()


@pytest.mark.parametrize('name', ['water_field', 'water_x2c'])
def test_qspa_takes_the_mean_fields_own_core_hamiltonian(name, monkeypatch):
    """run_parquet reads mf.get_hcore() once and hands qs_parquet what it
    returned, to the bit (a copy kept as it came back: a second call is
    another computation, X2C's a diagonalization and products), where T + V
    differs (a field, X2C)."""
    mf = mean_field(name)
    mol = mf.mol
    seen = {}
    kept = keep_core_hamiltonians(mf, monkeypatch)

    def qs_parquet(eps_hf, c_hf, s, x, hc, eri_ao, nO, **kwargs):
        seen['hc'] = np.array(hc)
        raise StopIteration
    monkeypatch.setattr(restricted, 'qs_parquet', qs_parquet)
    with pytest.raises(StopIteration):
        restricted.run_parquet(mf, 'qs')
    (own,) = kept
    assert bitwise(seen['hc'], own)
    t_plus_v = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    assert np.abs(own - t_plus_v).max() > OTHER_HCORE, name


def test_qspa_diagonalizes_by_dsyev(monkeypatch):
    """Every symmetric diagonalization of qsPA's orthogonalizer and Fock
    matrix is dsyev 'U' (PARQUET_QS_EIGH_DRIVER), the BSEs' the house's
    PARQUET_EIGH_DRIVER; X^T S X = 1 for both drivers' orthogonalizers."""
    mf = rhf('water')
    s = mf.mol.intor('int1e_ovlp')
    calls = []
    real = la.eigh

    def spy(a, *args, **kwargs):
        calls.append((np.shape(a)[0], kwargs.get('driver'), kwargs.get('lower')))
        return real(a, *args, **kwargs)
    monkeypatch.setattr(la, 'eigh', spy)
    for module in (restricted, spin_orbital):
        calls.clear()
        x = module.orthogonalization_matrix(s)
        assert calls == [(len(s), PARQUET_QS_EIGH_DRIVER, False)], calls
        assert np.abs(x.T @ s @ x - np.eye(x.shape[1])).max() <= ORTHONORMAL
        u = np.linalg.eigvalsh(s)
        assert x.shape[1] == np.count_nonzero(u > PARQUET_ORTHOGONALIZATION_THRESHOLD)
    eps, eri, nO = mo_integrals('water')
    norb = len(eps)
    calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        restricted.run_parquet(mf, 'qs', max_it_1b=1, max_it_2b=1)
    fock = [c for c in calls if c[0] == norb and c[1] == PARQUET_QS_EIGH_DRIVER]
    assert fock and all(c[2] is False for c in fock), calls
    others = {c[1] for c in calls if c[0] != norb}
    assert others <= {PARQUET_EIGH_DRIVER, PARQUET_QS_EIGH_DRIVER}, calls
    calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        spin_orbital.run_parquet_so(mf, 'qs', max_it_1b=1, max_it_2b=1)
    assert any(c[0] == 2 * norb and c[1] == PARQUET_QS_EIGH_DRIVER for c in calls), calls


@pytest.mark.parametrize('nO, nso', [(1, 6), (5, 6), (1, 2)])
def test_an_empty_spin_orbital_pp_branch(nO, nso):
    """One occupied spin orbital (no 2h pair), one virtual (no 2p pair), or
    both: the pp integrals, BSE and EcRPA of the empty shapes."""
    rng = np.random.default_rng(SEED)
    nVV, nOO = pp_pair_count(nso - nO, 'spin-orbital'), pp_pair_count(nO, 'spin-orbital')
    eri = rng.standard_normal((nso,) * 4)
    vertices = SpinOrbitalVertices(*(rng.standard_normal((nso,) * 4) for _ in range(2)))
    x1, y1 = rng.standard_normal((nVV, nVV)), rng.standard_normal((nOO, nVV))
    x2, y2 = rng.standard_normal((nVV, nOO)), rng.standard_normal((nOO, nOO))
    rho_ee, rho_hh = spin_orbital_two_body.pp_effective_integrals_so(eri, vertices, x1, y1, x2,
                                                                     y2, nO)
    assert rho_ee.shape == (nso, nso, nVV) and rho_hh.shape == (nso, nso, nOO)
    assert np.isfinite(rho_ee).all() and np.isfinite(rho_hh).all()
    # C's roots are the 2p energies, -D's the 2h ones (below zero, Quack's split)
    c = np.diag(np.arange(1.0, nVV + 1.0))
    d = np.diag(np.arange(1.0, nOO + 1.0))
    for tda in (True, False):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            solution = spin_orbital_two_body.solve_pp_bse_so(
                None if tda else np.zeros((nVV, nOO)), c, d, tda)
        assert solution.om_ee.shape == (nVV,) and solution.om_hh.shape == (nOO,), tda
        assert np.allclose(solution.om_ee, np.arange(1.0, nVV + 1.0)), tda
        assert np.allclose(solution.om_hh, -np.arange(nOO, 0.0, -1.0)), tda
        assert abs(solution.ec) <= ORTHONORMAL, (tda, solution.ec)


def test_a_one_electron_system_has_no_pp_pair():
    """H/STO-3G (two spin orbitals, one occupied): no pp pair at all, and the
    run is finite."""
    mol = gto.M(atom='H 0 0 0', basis='sto-3g', spin=1, verbose=0)
    mf = uhf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = spin_orbital.run_parquet_so(mf, 'os', eta_2b=1.0)
    assert result.spectra.ee.om.shape == (0,) and result.spectra.hh.om.shape == (0,)
    assert np.isfinite(result.eqp).all() and result.ec_pp == 0.0


def test_no_b_block_under_the_tda():
    """The block functions of both drivers return None for B (bare and
    kernel, eh and pp, both closed-shell spin channels) under the TDA, and a
    block otherwise."""
    eps, eri, nO = mo_integrals('water')
    zero = RestrictedVertices.zeros(len(eps))
    for spin in ('singlet', 'triplet'):
        for tda in (True, False):
            blocks = (restricted_bse.eh_bse_blocks(eps, eri, nO, spin, tda)[1],
                      restricted_bse.eh_kernel_blocks(zero, nO, spin, tda)[1],
                      restricted_bse.pp_bse_blocks(eps, eri, nO, spin, tda)[0],
                      restricted_bse.pp_kernel_blocks(zero, nO, spin, tda)[0])
            assert all((b is None) == tda for b in blocks), (spin, tda)
    eps, eri, nO, _ = both_layouts('water')[1]
    zero = SpinOrbitalVertices.zeros(len(eps))
    for tda in (True, False):
        blocks = (spin_orbital_two_body.eh_bse_blocks_so(eps, eri, nO, tda)[1],
                  spin_orbital_two_body.eh_kernel_blocks_so(zero, nO, tda)[1],
                  spin_orbital_two_body.pp_bse_blocks_so(eps, eri, nO, tda)[0],
                  spin_orbital_two_body.pp_kernel_blocks_so(zero, nO, tda)[0])
        assert all((b is None) == tda for b in blocks), tda


#: Ha^-2. The one-body regularization of the self-energy checks.
S1B = 50.0
#: Ha. Where the self-energies are evaluated off eHF: eHF + OFF_SHELL.
OFF_SHELL = 0.05
#: Orbitals of the subset checks (closed shell; over spin orbitals 2p, 2p + 1).
SOME = (1, 4, 9)


def flex_linearized(driver):
    """Water/6-31G FLEX (TDA, s_2b = 1, s_1b = S1B) by `driver`'s lower level,
    linearized: its sigc and z are the self-energy's at eHF; and its inputs."""
    (eps, eri, nO), (eps_so, eri_so, nO_so, pairs) = both_layouts('water')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if driver == 'restricted':
            return restricted.ev_parquet(eps, eri, nO, 'flex', eta_1b=S1B, eta_2b=1.0,
                                         qp_solver='linearized'), (eps, eri, nO)
        return spin_orbital.ev_parquet_so(eps_so, eri_so, nO_so, 'flex', eta_1b=S1B,
                                          eta_2b=1.0, qp_solver='linearized',
                                          spin_pairs=pairs), (eps_so, eri_so, nO_so)


def test_the_closed_shell_self_energy_functions_are_the_drivers():
    result, (eps, eri, nO) = flex_linearized('restricted')
    spectra = result.spectra
    sigc, z = diagonal_self_energy(S1B, eps, eri, nO, spectra)
    assert_same(sigc, result.sigc, label='diagonal_self_energy sigc')
    assert_same(z, result.z, unit='1', label='diagonal_self_energy z')
    parts = self_energy_parts(S1B, eps, eri, nO, spectra)
    assert set(parts) == set(TERMS)
    total = sum(parts[t] for t in TERMS)
    assert relative(total, sigc) <= CROSS_PATH_RTOL, relative(total, sigc)
    assert all(np.abs(parts[t]).max() > 1e-6 for t in TERMS), parts
    whole = DiagonalSelfEnergy(S1B, eps, eri, nO, spectra)
    some = DiagonalSelfEnergy(S1B, eps, eri, nO, spectra, orbitals=SOME)
    assert relative(some.diagonal()[0], sigc[list(SOME)]) <= CROSS_PATH_RTOL
    for p in SOME:
        w = eps[p] + OFF_SHELL
        got, want = np.array(self_energy_at(p, w, S1B, eps, eri, nO, spectra)), whole.at(p, w)
        assert relative(got, np.array(want)) <= CROSS_PATH_RTOL, (p, got, want)
        with pytest.raises(ValueError, match='not among the orbitals'):
            some.at(p + 1, w)


def test_the_spin_orbital_self_energy_functions_are_the_drivers():
    result, (eps, eri, nO) = flex_linearized('spin_orbital')
    se = DiagonalSelfEnergySO(S1B, eps, eri, nO, result.spectra)
    diagonal_parts, driver_parts = {}, {}
    sigc, z = se.diagonal(parts=diagonal_parts)
    assert relative(sigc, result.sigc) <= CROSS_PATH_RTOL, relative(sigc, result.sigc)
    assert relative(z, result.z) <= CROSS_PATH_RTOL, relative(z, result.z)
    diagonal_self_energy_so(S1B, eps, eri, nO, result.spectra, parts=driver_parts)
    assert set(driver_parts) == set(diagonal_parts), sorted(driver_parts)
    for name, part in diagonal_parts.items():
        assert relative(driver_parts[name], part) <= CROSS_PATH_RTOL, name
    w = eps + OFF_SHELL
    parts = {}
    sig, dsig = se.at_each(w, parts=parts)
    for name, want in (('sig', sig), ('dsig', dsig)):
        total = sum(parts[f'{name}_{t}'].sum(axis=1) for t in ('2', 'eh', 'pp'))
        assert relative(total, want) <= CROSS_PATH_RTOL, (name, relative(total, want))
    for p in [q for k in SOME for q in (2 * k, 2 * k + 1)]:
        want = np.array([sig[p], dsig[p]])
        assert relative(np.array(se.at(p, w[p])), want) <= CROSS_PATH_RTOL, p
        got = np.array(self_energy_at_so(p, w[p], S1B, eps, eri, nO, result.spectra))
        assert relative(got, want) <= CROSS_PATH_RTOL, (p, got, want)
        several = w[p] + np.array([[0.0, OFF_SHELL], [-OFF_SHELL, 2.0 * OFF_SHELL]])
        got = np.array(se.at(p, several))
        assert got.shape == (2,) + several.shape, got.shape
        want = np.array([se.at(p, x) for x in several.ravel()]).T.reshape(got.shape)
        assert relative(got, want) <= CROSS_PATH_RTOL, p


def test_the_srg_static_self_energy_on_one_worker():
    eps, eri, nO = mo_integrals('water')
    result, _ = flex_linearized('restricted')
    one = static_self_energy_srg(S1B, eps, eri, nO, result.spectra, n_workers=1)
    two = static_self_energy_srg(S1B, eps, eri, nO, result.spectra, n_workers=2)
    assert_same(one[0], two[0], label='SRG Sigma, one worker against two')
    assert_same(one[1], two[1], unit='1', label='SRG Z, one worker against two')


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
