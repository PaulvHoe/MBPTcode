"""Every named flag of the parquet drivers is raised on its event, warned with
its class, and listed in the run's result.numerical_flags.

The events come from the inputs that make them (docs/warnings.md):

* loops stopped short: a two-body or one-body iteration cap, a qp_solver
  function's verdict False (the tests' Quack Newton, `tests.
  parquet_quack_newton`, its NEWTON_MAX_IT patched to one step), a
  reference marked not converged;
* non-finite numbers: a vertex builder returning NaN (the vertex flag; the
  two-body change all NaN, which Quack's loop takes for convergence; with
  DIIS the history's Gram matrix; in evPA the next BSEs, self-energy, the
  energies Quack's Newton returns and one-body change), and AO integrals
  holding a NaN for qsPA (its Fock matrix, static self-energy, BSEs and
  vertex weights); the default
  builder's NaN vertices are flagged where they arise, never as the hook's
  'parquet_vertex_not_finite'; a NaN below the diagonal of a BSE matrix,
  which the Tamm-Dancoff eh problem never reads and the full one carries
  through A + B into the matrix it diagonalizes (both drivers), unless
  A - B is not positive definite, which alone is then flagged;
* unstable BSEs: H2 stretched to 1.5 and 2.0 Angstrom (the RHF triplet
  instability: imaginary excitations in the full eh BSE, negative ones in
  its Tamm-Dancoff form in both drivers, an A - B that is not positive
  definite over spin orbitals, where the spin-flip components carry it), H2
  just before that instability (a full eh root below the zero-mode
  threshold, where the self-energy grows as 1/Omega: flagged by both
  drivers), a vertex builder scaling Quack's vertices up (the pp problem's
  branch count, complex roots, Loewdin metric and EcRPA check, in both
  drivers), the open shells whose full BSEs leave NaN (OH and H2^- with the
  full eh or pp problem), the spin flip of the H atom in STO-3G at exactly
  zero (a zero mode of the full eh problem; under the TDA no negative
  excitation, whichever side of zero roundoff puts it), a NaN in the full pp
  matrix alone (flagged at its source), and LAPACK failures planted in the
  BSE solvers;
* quasiparticle roots: a linearized energy with Z outside (0, 1], and an
  equation without a root (a self-energy patched to a constant 1000 Ha).

`test_every_flag_has_its_test` maps each FLAG_* constant of `events` to the
test that raises it (DIIS's exactly singular system is in
`test_parquet_diis`; a qp_solver function's 'z_outside' in
`test_parquet_qp_solver`). The
names the house shares with other surfaces ('scf_not_converged',
'z_outside', 'no_root') are held by value, not only through the constants.

Run: python tests/test_parquet_flags.py, or under pytest.
"""
import copy
import functools
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import scipy.linalg as la
from pyscf import gto, scf
from pyscf.scf import uhf
from scipy.linalg import lapack

from src.Base.numerical_flags import collecting
from src.Base.numerics.constants import PARQUET_EH_ZERO_MODE_TOL, PARQUET_EIGH_DRIVER
from src.SingleReference.Parquet import (events, restricted, restricted_bse,
                                         restricted_self_energy, restricted_vertices,
                                         spin_orbital, spin_orbital_self_energy,
                                         spin_orbital_two_body)
from src.SingleReference.Parquet.events import (ParquetBSEUnstable, ParquetNotConverged,
                                                ParquetNotFinite)
from src.Solvers.qp_equation import QPNoRoot, QPWeightOutside
from tests import parquet_quack_newton
from tests.parquet_quack_newton import quack_newton
from tests.pins import assert_same
from tests.test_parquet_restricted_vs_quack import mo_integrals, qs_inputs, rhf
from tests.test_parquet_spin_adaptation import both_layouts

#: Ha. A self-energy constant this large puts the root of w = eHF + Sigma
#: beyond the finder's widest scan window (QP_SEARCH_WINDOW_TIERS).
NO_ROOT_SIGMA = 1e3
#: Scales of Quack's reducible vertices that make the BSEs unstable: x20
#: breaks the closed-shell TDA eh problem and both drivers' full problems,
#: x50 the closed-shell full pp problem's branch count (water/6-31G, three
#: two-body iterations).
UNSTABLE_SCALE = 20.0
PP_UNSTABLE_SCALE = 50.0
#: Ha. An eh excitation of the constructed BSE below PARQUET_EH_ZERO_MODE_TOL.
ZERO_MODE = 2e-4
#: Ha. H/STO-3G's spin flip lies at zero to this (2.2e-16 measured).
SPIN_FLIP_ROUNDOFF = 1e-12
#: Angstrom. H2/6-31G stretched past its RHF triplet instability: the full
#: eh problem's triplet (A + B) is indefinite at STRETCHED_RPA, its TDA A
#: at STRETCHED_TDA.
STRETCHED_RPA = 1.5
STRETCHED_TDA = 2.0
#: Angstrom. H2/6-31G just before its RHF triplet instability (RPA Omega^2
#: = 0 at 1.19134543165 Angstrom, bisected): the triplet's lowest full eh
#: root is 3.6e-4 Ha there, below PARQUET_EH_ZERO_MODE_TOL, and Sigma_c of
#: the HOMO -5346 eV (-66 eV at 1.1863 Angstrom, Omega 0.026 Ha).
EDGE_OF_INSTABILITY = 1.1913444316463324
#: Z of the HOMO there below this (0.0033 measured, linearized at s_1b =
#: 1e8): the self-energy's slope, 1/Omega too, has taken over.
EDGE_Z = 0.01


def recorded(fn, *args, **kwargs):
    """(fn(*args, **kwargs), {names of the warning classes it warned})."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fn(*args, **kwargs)
    return result, {w.category.__name__ for w in caught}


def names(result):
    return {entry['flag'] for entry in result.numerical_flags}


def assert_raised(result, categories, flags, classes):
    """Every flag of `flags` is in the result's numerical_flags and every
    warning class of `classes` was warned."""
    assert set(flags) <= names(result), (sorted(flags), sorted(names(result)))
    assert {c.__name__ for c in classes} <= categories, (classes, categories)


def nan_vertex(name, spectra, eta_2b):
    """A vertex builder returning NaN: a closed-shell or spin-orbital
    (n,)*4 array, n read off the spectra's first channel."""
    n = spectra[0].rho.shape[0]
    return np.full((n,) * 4, np.nan)


def scaled(builder, k):
    """A vertex builder returning k times Quack's vertex."""
    def build(name, spectra, eta_2b):
        return k * builder(name, spectra, eta_2b)
    return build


@functools.lru_cache(maxsize=None)
def water_so():
    """(eps, eri, nO, pairs) of water/6-31G over interleaved spin orbitals."""
    return both_layouts('water')[1]


def run_r(scheme, **options):
    """Water/6-31G through ev_parquet: the quasiparticle equation by Quack's
    Newton (`quack_newton`, a qp_solver function, whose verdicts the driver
    flags among those held), unless `options` names another solver."""
    eps, eri, nO = mo_integrals('water')
    options.setdefault('qp_solver', quack_newton)
    return recorded(restricted.ev_parquet, eps, eri, nO, scheme, **options)


def run_so(scheme, **options):
    """`run_r` over water's interleaved spin orbitals and their pairs."""
    eps, eri, nO, pairs = water_so()
    options.setdefault('qp_solver', quack_newton)
    return recorded(spin_orbital.ev_parquet_so, eps, eri, nO, scheme, spin_pairs=pairs,
                    **options)


@functools.lru_cache(maxsize=None)
def stretched_h2(distance):
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {distance}', basis='6-31g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged, distance
    return mf


@functools.lru_cache(maxsize=None)
def open_shell(atom, basis, charge, cls):
    """A UHF (conv_tol 1e-12) and its spin-orbital (eps, eri, nO)."""
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=1, verbose=0)
    mf = cls(mol)
    mf.conv_tol = 1e-12
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mf.kernel()
    ref = spin_orbital.spin_orbital_reference(mf)
    return ref.eps, spin_orbital.spin_orbital_eri(ref.eri_ao, ref.mo_coeff), ref.nO


OH = ('O 0 0 0; H 0 0 0.9697', '6-31g', 0, scf.UHF)
H2_ANION = ('H 0 0 0; H 0 0 0.74', 'sto-3g', -1, scf.UHF)
H_STO = ('H 0 0 0', 'sto-3g', 0, uhf.UHF)
#: Quack's FLEX of the open shells' NaN paths: one two-body iteration,
#: linearized.
FLEX_LINEARIZED = dict(eta_1b=50.0, eta_2b=1.0, max_it_2b=1, conv_2b=1e-6,
                       qp_solver='linearized')


# --------------------------------------------------------------------------
# loops stopped short


@pytest.mark.parametrize('run', [run_r, run_so])
def test_iteration_caps(run):
    result, warned = run('os', eta_2b=1.0, max_it_2b=2)
    assert_raised(result, warned, {events.FLAG_TWO_BODY_NOT_CONVERGED}, {ParquetNotConverged})
    assert result.converged is False and result.converged_2b == [False]
    result, warned = run('ev', eta_2b=1.0, max_it_2b=2, max_it_1b=2)
    assert_raised(result, warned, {events.FLAG_ONE_BODY_NOT_CONVERGED}, {ParquetNotConverged})
    assert result.converged_1b is False and result.n_it_1b == 2


def test_the_qspa_iteration_cap():
    result, warned = recorded(restricted.qs_parquet, **qs_inputs('water'), eta_2b=1.0,
                              max_it_1b=1, max_it_2b=2)
    assert_raised(result, warned, {events.FLAG_ONE_BODY_NOT_CONVERGED,
                                   events.FLAG_TWO_BODY_NOT_CONVERGED}, {ParquetNotConverged})
    assert result.converged_1b is False


def test_newton_capped(monkeypatch):
    """Quack's Newton allowed one step keeps every orbital at eQPlin, as
    R_Parquet_QP_graph does, with its verdict False, which the driver flags
    for a qp_solver function."""
    mo_integrals('water'), water_so()
    monkeypatch.setattr(parquet_quack_newton, 'NEWTON_MAX_IT', 1)
    result, warned = run_r('flex')
    assert_raised(result, warned, {events.FLAG_QP_NOT_CONVERGED}, {ParquetNotConverged})
    assert np.array_equal(result.eqp, result.eqp_lin) and not result.qp_converged.any()
    result, warned = run_so('flex')
    assert_raised(result, warned, {events.FLAG_QP_NOT_CONVERGED}, {ParquetNotConverged})
    # the second spin orbital of a pair takes the first's eQPlin
    assert np.array_equal(result.eqp[0::2], result.eqp_lin[0::2])
    assert np.array_equal(result.eqp[1::2], result.eqp[0::2])


@pytest.mark.parametrize('front_door', [restricted.run_parquet, spin_orbital.run_parquet_so])
def test_an_unconverged_reference(front_door):
    mf = copy.copy(rhf('water'))
    mf.converged = False
    result, warned = recorded(front_door, mf, 'flex', qp_solver=quack_newton)
    assert_raised(result, warned, {events.FLAG_SCF_NOT_CONVERGED}, {ParquetNotConverged})
    # the house's name, shared with the ADC, photophysics and GW-BSE surfaces
    assert 'scf_not_converged' in names(result)


# --------------------------------------------------------------------------
# non-finite numbers


@pytest.mark.parametrize('run', [run_r, run_so])
def test_a_nan_vertex_ends_the_two_body_loop(run):
    """A NaN vertex makes every change NaN, which ends Quack's loop as a
    success: here flagged, converged False, the first iteration's spectra
    kept (finite osPA energies); with DIIS its history is NaN."""
    result, warned = run('os', eta_2b=1.0, max_it_2b=3, vertex_builder=nan_vertex)
    assert_raised(result, warned, {events.FLAG_VERTEX_NOT_FINITE,
                                   events.FLAG_TWO_BODY_NOT_FINITE}, {ParquetNotFinite})
    assert result.converged is False and result.n_it_2b == [1]
    assert np.isfinite(result.eqp).all()
    result, warned = run('os', eta_2b=1.0, max_it_2b=3, max_diis_2b=3, vertex_builder=nan_vertex)
    assert_raised(result, warned, {events.FLAG_DIIS_NOT_FINITE}, {ParquetNotFinite})
    assert np.isnan(result.history[0]['two_body'][0]['diis_rcond'])


@pytest.mark.parametrize('run', [run_r, run_so])
def test_a_nan_vertex_reaches_the_one_body_step(run):
    """In evPA the next one-body iteration starts from the NaN vertices: its
    BSEs, self-energy, the energies Quack's Newton returns (flagged by the
    driver) and one-body change are NaN."""
    result, warned = run('ev', eta_2b=1.0, max_it_2b=3, max_it_1b=2, vertex_builder=nan_vertex)
    assert_raised(result, warned, {events.FLAG_BSE_NOT_FINITE,
                                   events.FLAG_SELF_ENERGY_NOT_FINITE,
                                   events.FLAG_QP_NOT_FINITE,
                                   events.FLAG_ONE_BODY_NOT_FINITE}, {ParquetNotFinite})
    assert np.isnan(result.eqp).all() and result.converged_1b is False


def test_a_nan_self_energy_leaves_no_root_and_no_branch():
    """With the root finder, the NaN vertices make the second evPA
    iteration's self-energy NaN: every equation is left without a root
    ('no_root', NaN), and none is reported as a followed branch the largest
    weight has left ('parquet_qp_branch_flipped' names a root kept, not a
    missing one)."""
    result, warned = run_r('ev', eta_2b=1.0, max_it_2b=3, max_it_1b=2, vertex_builder=nan_vertex,
                           qp_solver='find_qp_root')
    assert_raised(result, warned, {events.FLAG_NO_ROOT}, {QPNoRoot})
    assert events.FLAG_QP_BRANCH_FLIPPED not in names(result), sorted(names(result))
    second = result.history[1]
    assert np.isnan(second['eqp']).all() and not second['qp_flipped'].any()


def test_a_nan_one_body_diis_history():
    result, warned = run_r('ev', eta_2b=1.0, max_it_2b=3, max_it_1b=3, max_diis_1b=3,
                           vertex_builder=nan_vertex)
    assert_raised(result, warned, {events.FLAG_DIIS_NOT_FINITE}, {ParquetNotFinite})
    assert np.isnan(result.history[-1]['diis_rcond'])


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_a_nan_ao_integral_in_qspa(driver):
    """qsPA from AO integrals holding a NaN: its Fock matrix, static
    self-energy, BSEs and vertex weights are not finite, each flagged."""
    if driver == 'restricted':
        inputs = qs_inputs('water')
        inputs['eri_ao'] = inputs['eri_ao'].copy()
        inputs['eri_ao'][0, 0, 0, 0] = np.nan
        result, warned = recorded(restricted.qs_parquet, **inputs, max_it_1b=1)
    else:
        ref = spin_orbital.spin_orbital_reference(rhf('water'))
        eri_ao = ref.eri_ao.copy()
        eri_ao[0, 0, 0, 0] = np.nan
        x = spin_orbital.orthogonalization_matrix(ref.ovlp)
        result, warned = recorded(spin_orbital.qs_parquet_so, ref.eps, ref.mo_coeff, ref.ovlp, x,
                                  ref.hcore, eri_ao, ref.nO, max_it_1b=1, max_it_2b=2)
    assert_raised(result, warned, {events.FLAG_QS_FOCK_NOT_FINITE,
                                   events.FLAG_STATIC_SIGMA_NOT_FINITE,
                                   events.FLAG_BSE_NOT_FINITE, events.FLAG_VERTEX_POLE,
                                   events.FLAG_ONE_BODY_NOT_FINITE}, {ParquetNotFinite})
    assert np.isnan(result.eqp).all() and result.converged_1b is False
    assert events.FLAG_VERTEX_NOT_FINITE not in names(result)


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_a_nan_default_vertex_is_not_the_hooks_flag(driver):
    """A NaN orbital energy makes the BSEs, and so the default builder's
    vertices, NaN: flagged where it arises ('parquet_bse_not_finite'), never
    as 'parquet_vertex_not_finite', which names a builder other than the
    default (docs/warnings.md)."""
    if driver == 'restricted':
        eps, eri, nO = mo_integrals('water')
        eps = eps.copy()
        eps[-1] = np.nan
        result, _ = recorded(restricted.ev_parquet, eps, eri, nO, 'os', eta_2b=1.0, max_it_2b=2,
                             qp_solver=quack_newton)
    else:
        eps, eri, nO, _ = water_so()
        eps = eps.copy()
        eps[-2:] = np.nan       # no spin pairs: NaN energies are no pair's equal ones
        result, _ = recorded(spin_orbital.ev_parquet_so, eps, eri, nO, 'os', eta_2b=1.0,
                             max_it_2b=2, qp_solver=quack_newton)
    assert any(np.isnan(v).any() for v in result.vertices), sorted(names(result))
    assert events.FLAG_BSE_NOT_FINITE in names(result), sorted(names(result))
    assert events.FLAG_VERTEX_NOT_FINITE not in names(result), sorted(names(result))


# --------------------------------------------------------------------------
# unstable BSEs


def test_the_stretched_h2_triplet_instability():
    """The closed-shell driver takes |Omega^2| of the imaginary triplet
    excitations (phRLR) and keeps the negative TDA ones, flagging both; over
    spin orbitals the instability is an A - B that is not positive definite,
    which phGLR turns into NaN, flagged."""
    result, warned = recorded(restricted.run_parquet, stretched_h2(STRETCHED_RPA), 'flex',
                              tda_eh=False, tda_pp=False, qp_solver=quack_newton)
    assert_raised(result, warned, {events.FLAG_EH_NEGATIVE_EXCITATIONS}, {ParquetBSEUnstable})
    assert np.isfinite(result.spectra.eh_triplet.om).all()
    result, warned = recorded(restricted.run_parquet, stretched_h2(STRETCHED_TDA), 'flex',
                              qp_solver=quack_newton)
    assert_raised(result, warned, {events.FLAG_EH_TDA_NEGATIVE_EXCITATIONS},
                  {ParquetBSEUnstable})
    assert result.spectra.eh_triplet.om[0] < 0.0
    # the same three triplet roots over spin orbitals, flagged as the closed shell's
    result_so, warned = recorded(spin_orbital.run_parquet_so, stretched_h2(STRETCHED_TDA), 'flex',
                                 qp_solver=quack_newton)
    assert_raised(result_so, warned, {events.FLAG_EH_TDA_NEGATIVE_EXCITATIONS},
                  {ParquetBSEUnstable})
    assert result_so.spectra.eh.om[:3] == pytest.approx([result.spectra.eh_triplet.om[0]] * 3)
    assert result_so.spectra.eh.om[0] < -PARQUET_EH_ZERO_MODE_TOL
    result, warned = recorded(spin_orbital.run_parquet_so, stretched_h2(STRETCHED_RPA), 'flex',
                              tda_eh=False, tda_pp=False, qp_solver=quack_newton)
    assert_raised(result, warned, {events.FLAG_EH_A_MINUS_B_NOT_POSITIVE},
                  {ParquetBSEUnstable})
    assert np.isnan(result.spectra.eh.om).all()


def test_a_closed_shell_at_the_edge_of_its_triplet_instability():
    """A closed shell breaks no continuous symmetry, so it has no zero mode
    of its own, but just before its triplet instability its lowest full eh
    root goes to zero, X + Y grows as Omega^-1/2 and the self-energy as
    1/Omega: both drivers flag the root below PARQUET_EH_ZERO_MODE_TOL as a
    zero mode and keep the numbers (Z of the HOMO near zero)."""
    mf = stretched_h2(EDGE_OF_INSTABILITY)
    options = dict(tda_eh=False, tda_pp=False, eta_1b=1e8, qp_solver='linearized')
    result, warned = recorded(restricted.run_parquet, mf, 'flex', **options)
    assert_raised(result, warned, {events.FLAG_EH_ZERO_MODE}, {ParquetBSEUnstable})
    assert 0.0 < result.spectra.eh_triplet.om[0] < PARQUET_EH_ZERO_MODE_TOL
    assert events.FLAG_EH_NEGATIVE_EXCITATIONS not in names(result)
    assert 0.0 < result.z[result.nO - 1] < EDGE_Z, result.z
    result_so, warned = recorded(spin_orbital.run_parquet_so, mf, 'flex', **options)
    assert_raised(result_so, warned, {events.FLAG_EH_ZERO_MODE}, {ParquetBSEUnstable})


def test_imaginary_excitations_of_a_positive_a_minus_b():
    """A - B positive definite and A + B not (Omega^2 = 0.75 and -1.25):
    phRLR takes |Omega^2|, phGLR sqrt(Omega^2), NaN; both flag it."""
    a, b = np.diag([1.0, 1.0]), np.diag([-0.5, -1.5])
    (om_r, *_), warned = recorded(restricted_bse.solve_eh_bse, a, b, False)
    assert 'ParquetBSEUnstable' in warned
    assert om_r == pytest.approx([np.sqrt(1.25), np.sqrt(0.75)]), om_r
    solution, warned = recorded(spin_orbital_two_body.solve_eh_bse_so, a, b, False)
    assert 'ParquetBSEUnstable' in warned
    assert np.isnan(solution.om[0]) and solution.om[1] == pytest.approx(np.sqrt(0.75))
    flags = []
    with collecting(flags), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        restricted_bse.solve_eh_bse(a, b, False)
        spin_orbital_two_body.solve_eh_bse_so(a, b, False)
    assert [e['flag'] for e in flags] == [events.FLAG_EH_NEGATIVE_EXCITATIONS] * 2, flags


def test_scaled_vertices_break_the_bses():
    """Quack's vertices scaled up: the closed-shell TDA eh problem gets
    negative excitations, the full eh problems a non-positive A - B, the full
    pp problems a wrong branch count (complex roots over spin orbitals), a
    near-singular Loewdin metric and an inconsistent EcRPA."""
    result, warned = run_r('os', eta_2b=1.0, max_it_2b=3, qp_solver='linearized',
                           vertex_builder=scaled(restricted.static_reducible_vertex,
                                                 UNSTABLE_SCALE))
    assert_raised(result, warned, {events.FLAG_EH_TDA_NEGATIVE_EXCITATIONS},
                  {ParquetBSEUnstable})
    result, warned = run_r('os', tda_eh=False, tda_pp=False, eta_2b=1.0, max_it_2b=3,
                           qp_solver='linearized',
                           vertex_builder=scaled(restricted.static_reducible_vertex,
                                                 PP_UNSTABLE_SCALE))
    assert_raised(result, warned, {events.FLAG_EH_A_MINUS_B_NOT_POSITIVE,
                                   events.FLAG_PP_BRANCH_COUNT,
                                   events.FLAG_PP_LOWDIN_SMALL_EIGENVALUE,
                                   events.FLAG_PP_EC_INCONSISTENT}, {ParquetBSEUnstable})
    assert_raised(result, warned, {events.FLAG_VERTEX_POLE, events.FLAG_SELF_ENERGY_NOT_FINITE},
                  {ParquetNotFinite})
    result, warned = run_so('os', tda_eh=False, tda_pp=False, eta_2b=1.0, max_it_2b=3,
                            qp_solver='linearized',
                            vertex_builder=scaled(spin_orbital.static_reducible_vertex_so,
                                                  UNSTABLE_SCALE))
    assert_raised(result, warned, {events.FLAG_PP_COMPLEX_ROOTS, events.FLAG_PP_BRANCH_COUNT,
                                   events.FLAG_PP_LOWDIN_SMALL_EIGENVALUE,
                                   events.FLAG_PP_EC_INCONSISTENT,
                                   events.FLAG_EH_A_MINUS_B_NOT_POSITIVE}, {ParquetBSEUnstable})


def test_complex_pp_roots_of_a_two_by_two_problem():
    """C = 1 and D = -1 make [[C, B], [-B, -D]] = [[1, B], [-B, 1]], whose
    roots 1 +- iB are complex: the spin-orbital solver flags them (the
    closed-shell one has no such check, as Quack's ppRLR), both the branch
    count and the Loewdin metric of the padded branch."""
    args = (np.array([[2.0]]), np.array([[1.0]]), np.array([[-1.0]]), False)
    _, warned = recorded(spin_orbital_two_body.solve_pp_bse_so, *args)
    assert 'ParquetBSEUnstable' in warned
    flags = []
    with collecting(flags), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        spin_orbital_two_body.solve_pp_bse_so(*args)
        restricted_bse.solve_pp_bse(*args)
    found = [entry['flag'] for entry in flags]
    assert found.count(events.FLAG_PP_COMPLEX_ROOTS) == 1, found
    assert found.count(events.FLAG_PP_BRANCH_COUNT) == 2, found
    assert events.FLAG_PP_LOWDIN_SMALL_EIGENVALUE in found, found


def test_the_open_shell_nan_paths():
    """The open shells' NaN paths: OH with the full eh problem (its spin rotation
    below zero) and H2^- with the full pp problem (one virtual spin orbital,
    the 2h branch count) are NaN everywhere and flagged; their TDA twins are
    finite."""
    eps, eri, nO = open_shell(*OH)
    result, warned = recorded(spin_orbital.ev_parquet_so, eps, eri, nO, 'os', tda_eh=False,
                              tda_pp=False, **FLEX_LINEARIZED)
    assert_raised(result, warned, {events.FLAG_EH_A_MINUS_B_NOT_POSITIVE},
                  {ParquetBSEUnstable})
    assert np.isnan(result.eqp).all()
    twin, _ = recorded(spin_orbital.ev_parquet_so, eps, eri, nO, 'os', tda_eh=True,
                       tda_pp=False, **FLEX_LINEARIZED)
    assert np.isfinite(twin.eqp).all()
    eps, eri, nO = open_shell(*H2_ANION)
    result, warned = recorded(spin_orbital.ev_parquet_so, eps, eri, nO, 'os', tda_eh=True,
                              tda_pp=False, **FLEX_LINEARIZED)
    assert_raised(result, warned, {events.FLAG_PP_BRANCH_COUNT}, {ParquetBSEUnstable})
    assert np.isnan(result.eqp).all()
    twin, _ = recorded(spin_orbital.ev_parquet_so, eps, eri, nO, 'os', tda_eh=True,
                       tda_pp=True, **FLEX_LINEARIZED)
    assert np.isfinite(twin.eqp).all()


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_a_nan_in_the_full_pp_matrix(driver):
    """<aa|aa> of the first virtual set to NaN enters the pp C block alone:
    with the eh TDA and the full pp problem the eh spectra stay finite and
    the pp matrix is flagged at its source (over spin orbitals the Loewdin
    metric built from its NaN eigenpairs is flagged one step later too)."""
    if driver == 'restricted':
        eps, eri, nO = mo_integrals('water')
        eri = eri.copy()
        eri[nO, nO, nO, nO] = np.nan
        result, warned = recorded(restricted.ev_parquet, eps, eri, nO, 'flex', tda_eh=True,
                                  tda_pp=False, eta_2b=1.0, qp_solver=quack_newton)
        assert all(np.isfinite(c.om).all() for c in (result.spectra.eh_singlet,
                                                     result.spectra.eh_triplet))
        assert np.isnan(result.spectra.ee_singlet.om).all()
        source = 'the pp BSE matrix holds NaN or inf'
    else:
        eps, eri, nO, pairs = water_so()
        eri = eri.copy()
        eri[nO, nO + 1, nO, nO + 1] = np.nan       # <ab|ab> of the first virtual pair
        result, warned = recorded(spin_orbital.ev_parquet_so, eps, eri, nO, 'flex', tda_eh=True,
                                  tda_pp=False, eta_2b=1.0, spin_pairs=pairs,
                                  qp_solver=quack_newton)
        assert np.isfinite(result.spectra.eh.om).all()
        source = 'the pp matrix is not finite'
    assert_raised(result, warned, {events.FLAG_BSE_NOT_FINITE}, {ParquetNotFinite})
    messages = [e['message'] for e in result.numerical_flags
                if e['flag'] == events.FLAG_BSE_NOT_FINITE]
    assert any(source in m for m in messages), messages


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_a_nan_below_the_diagonal_is_never_read(driver):
    """Quack diagonalizes a symmetric BSE matrix by dsyev('V', 'U'), which
    reads its upper triangle only: a NaN below the diagonal (an element a
    vertex builder put there) leaves the Tamm-Dancoff eh solution finite and
    unflagged in both drivers, the eigenvalues those of the upper triangle;
    a NaN above it gives NaN eigenpairs, flagged 'parquet_bse_not_finite'."""
    rng = np.random.default_rng(1)
    m = rng.standard_normal((6, 6))
    a = m + m.T + 10.0 * np.eye(6)
    if driver == 'restricted':
        def solve(x):
            return restricted_bse.solve_eh_bse(x, None, True)[0]
    else:
        def solve(x):
            return spin_orbital_two_body.solve_eh_bse_so(x, None, True).om
    for (i, j), read in (((4, 1), False), ((1, 4), True)):
        b = a.copy()
        b[i, j] = np.nan
        flags = []
        with collecting(flags), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            om = solve(b)
        if read:
            assert np.isnan(om).all(), om
            assert [e['flag'] for e in flags] == [events.FLAG_BSE_NOT_FINITE], flags
        else:
            assert flags == [] and [w.category for w in caught] == [], (flags, caught)
            upper = la.eigh(a, lower=False, driver=PARQUET_EIGH_DRIVER)[0]
            assert_same(om, upper, quantity='root', label=(i, j))


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
@pytest.mark.parametrize('block', ['A', 'B'])
def test_a_nan_below_the_diagonal_reaches_the_full_eh_problem(driver, block):
    """Without the TDA the eh BSE reads A - B by its upper triangle and A + B
    whole, in the product (A - B)^1/2 (A + B) (A - B)^1/2 (Quack's phRLR and
    phGLR): a NaN below the diagonal of A or of B passes A - B and reaches
    that product, so both drivers give NaN eigenpairs and flag it once,
    'parquet_bse_not_finite' (the spin-orbital solver had passed it over
    silently, as an A - B that is not positive definite, which it is not);
    the same blocks without the NaN solve finite and unflagged. Where A - B
    is not positive definite (its upper triangle finite), the product is not
    diagonalized: that alone is flagged, 'parquet_eh_a_minus_b_not_positive',
    the NaN below the diagonal among the NaN it gives."""
    rng = np.random.default_rng(1)
    m, n = rng.standard_normal((6, 6)), rng.standard_normal((6, 6))
    blocks = {'A': m + m.T + 10.0 * np.eye(6), 'B': 0.1 * (n + n.T)}
    if driver == 'restricted':
        def solve(a, b):
            return restricted_bse.solve_eh_bse(a, b, False)[0]
    else:
        def solve(a, b):
            return spin_orbital_two_body.solve_eh_bse_so(a, b, False).om
    for nan in (False, True):
        given = {k: v.copy() for k, v in blocks.items()}
        if nan:
            given[block][4, 1] = np.nan
        flags = []
        with collecting(flags), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            om = solve(given['A'], given['B'])
        if nan:
            assert np.isnan(om).all(), om
            assert [e['flag'] for e in flags] == [events.FLAG_BSE_NOT_FINITE], flags
            assert {w.category for w in caught} == {ParquetNotFinite}, caught
        else:
            assert np.isfinite(om).all() and flags == [] and caught == [], (om, flags)
    given = {'A': m + m.T + np.eye(6), 'B': 2.0 * (n + n.T)}     # A - B: lowest -9.2
    given[block][4, 1] = np.nan
    flags = []
    with collecting(flags), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        om = solve(given['A'], given['B'])
    assert np.isnan(om).all(), om
    assert [e['flag'] for e in flags] == [events.FLAG_EH_A_MINUS_B_NOT_POSITIVE], flags
    assert {w.category for w in caught} == {ParquetBSEUnstable}, caught


def test_an_overflowing_denominator_is_no_event():
    """Past about 1e154 Ha, which only the eigenvalues of a diverging
    two-body loop reach, d^2 overflows in the regularized 1/d: exp(-inf) is
    0 and the weight is 1/d, finite, so neither driver's regularized_inverse
    warns (numpy's 'overflow encountered in multiply' came out by the
    thousand beside the loop's named flags). A zero denominator is still
    NaN, warned and flagged by the closed-shell one ('parquet_vertex_pole';
    the spin-orbital vertices count theirs)."""
    d = np.array([1e160, -1e160, 0.5])
    for regularized_inverse in (restricted_vertices.regularized_inverse,
                                spin_orbital_two_body.regularized_inverse):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            weights = regularized_inverse(d, 50.0)
        assert np.isfinite(weights).all(), weights
        assert weights[:2].tolist() == (1.0 / d[:2]).tolist(), weights
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert np.isnan(regularized_inverse(np.zeros(1), 50.0)).all()
    with pytest.warns(ParquetNotFinite, match='not finite'):
        restricted_vertices.regularized_inverse(np.zeros(1), 50.0)


def test_the_spin_flip_zero_mode():
    """An eh excitation below PARQUET_EH_ZERO_MODE_TOL (a constructed full
    BSE, A - B and A + B positive) is kept and flagged. H/STO-3G has one eh
    root, its spin flip, at zero to roundoff: which side of zero the last
    bit puts it on decides between the zero mode (kept, flagged) and Quack's
    NaN (flagged as the unstable BSE it then is), so either is held."""
    a = np.diag([ZERO_MODE, 1.0])
    solution, warned = recorded(spin_orbital_two_body.solve_eh_bse_so, a, np.zeros((2, 2)),
                                False)
    assert 'ParquetBSEUnstable' in warned
    assert solution.om == pytest.approx([ZERO_MODE, 1.0])
    flags = []
    with collecting(flags), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        spin_orbital_two_body.solve_eh_bse_so(a, np.zeros((2, 2)), False)
    assert [e['flag'] for e in flags] == [events.FLAG_EH_ZERO_MODE], flags
    # the closed-shell solver flags it too (a closed shell at the edge of an instability)
    (om_r, *_), warned = recorded(restricted_bse.solve_eh_bse, a, np.zeros((2, 2)), False)
    assert 'ParquetBSEUnstable' in warned and om_r == pytest.approx([ZERO_MODE, 1.0])
    flags = []
    with collecting(flags), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        restricted_bse.solve_eh_bse(a, np.zeros((2, 2)), False)
    assert [e['flag'] for e in flags] == [events.FLAG_EH_ZERO_MODE], flags
    eps, eri, nO = open_shell(*H_STO)
    # under the TDA the spin flip is an eigenvalue of A at zero to roundoff,
    # not a negative excitation, whichever side of zero it lands on
    result, _ = recorded(spin_orbital.ev_parquet_so, eps, eri, nO, 'os', tda_eh=True,
                         tda_pp=True, **FLEX_LINEARIZED)
    om, = result.spectra.eh.om
    assert abs(om) < SPIN_FLIP_ROUNDOFF, om
    assert events.FLAG_EH_TDA_NEGATIVE_EXCITATIONS not in names(result), names(result)
    result, warned = recorded(spin_orbital.ev_parquet_so, eps, eri, nO, 'os', tda_eh=False,
                              tda_pp=False, **FLEX_LINEARIZED)
    om, = result.spectra.eh.om
    if np.isfinite(om):
        assert_raised(result, warned, {events.FLAG_EH_ZERO_MODE}, {ParquetBSEUnstable})
        assert om < SPIN_FLIP_ROUNDOFF, om
    else:
        assert names(result) & {events.FLAG_EH_A_MINUS_B_NOT_POSITIVE,
                                events.FLAG_EH_NEGATIVE_EXCITATIONS}, names(result)


@pytest.mark.parametrize('run', [run_r, run_so])
def test_lapack_failures(run, monkeypatch):
    """A failed symmetric diagonalization gives NaN eigenpairs; a dgeev that
    does not converge keeps what it returned; both flagged."""
    mo_integrals('water'), water_so()

    def fails(*args, **kwargs):
        raise la.LinAlgError('planted failure')
    with monkeypatch.context() as patch:
        patch.setattr(la, 'eigh', fails)
        result, warned = run('flex')
    assert_raised(result, warned, {events.FLAG_BSE_DIAGONALIZATION_FAILED},
                  {ParquetBSEUnstable})
    real = lapack.dgeev

    def no_convergence(*args, **kwargs):
        return real(*args, **kwargs)[:-1] + (1,)
    monkeypatch.setattr(lapack, 'dgeev', no_convergence)
    result, warned = run('flex', tda_eh=False, tda_pp=False)
    assert_raised(result, warned, {events.FLAG_PP_DGEEV_FAILED}, {ParquetBSEUnstable})
    assert np.isfinite(result.eqp).all()


# --------------------------------------------------------------------------
# quasiparticle roots


def test_a_linearized_energy_outside_the_weights():
    """'linearized' keeps eQPlin whatever its diagonal Z, and flags the ones
    outside (0, 1]: water's FLEX TDA at s_1b = 1e8 (orbitals 7 and 10, Z
    -0.17 and -0.10, Quack's own Z there)."""
    result, warned = run_r('flex', eta_1b=1e8, eta_2b=1.0, qp_solver='linearized')
    assert_raised(result, warned, {events.FLAG_Z_OUTSIDE}, {QPWeightOutside})
    outside = np.flatnonzero(~((result.z > 0.0) & (result.z <= 1.0)))
    named = sorted({int(e['message'].split('orbital ')[1].split()[0])
                    for e in result.numerical_flags if e['flag'] == events.FLAG_Z_OUTSIDE})
    assert named == outside.tolist() == [7, 10], (named, result.z)
    assert 'z_outside' in names(result)          # the house's name, every QP route's


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_an_equation_without_a_root(driver, monkeypatch):
    """A self-energy of 1000 Ha leaves no root in the finder's windows: NaN,
    flagged for every orbital (over spin orbitals, the second of each pair
    too, from its partner)."""
    if driver == 'restricted':
        eps, eri, nO = mo_integrals('water')
        monkeypatch.setattr(restricted_self_energy.DiagonalSelfEnergy, 'at',
                            lambda self, p, w: (NO_ROOT_SIGMA, 0.0))
        result, warned = recorded(restricted.ev_parquet, eps, eri, nO, 'flex',
                                  qp_solver='find_qp_root')
        n = len(eps)
    else:
        mol = gto.M(atom='He 0 0 0', basis='6-31g', verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        monkeypatch.setattr(spin_orbital_self_energy.DiagonalSelfEnergySO, 'at',
                            lambda self, p, w: (NO_ROOT_SIGMA, 0.0))
        result, warned = recorded(spin_orbital.run_parquet_so, mf, 'flex',
                                  qp_solver='find_qp_root')
        n = 2 * mf.mo_coeff.shape[1]
    assert_raised(result, warned, {events.FLAG_NO_ROOT}, {QPNoRoot})
    assert 'no_root' in names(result)            # the house's root finder's name
    assert np.isnan(result.eqp).all() and not result.qp_converged.any()
    assert sum(e['flag'] == events.FLAG_NO_ROOT for e in result.numerical_flags) == n


# --------------------------------------------------------------------------
# the map

#: Each flag of `events` and the test that raises it.
FLAG_TESTS = {
    events.FLAG_TWO_BODY_NOT_CONVERGED: 'test_iteration_caps',
    events.FLAG_ONE_BODY_NOT_CONVERGED: 'test_iteration_caps',
    events.FLAG_QP_NOT_CONVERGED: 'test_newton_capped',
    events.FLAG_SCF_NOT_CONVERGED: 'test_an_unconverged_reference',
    events.FLAG_TWO_BODY_NOT_FINITE: 'test_a_nan_vertex_ends_the_two_body_loop',
    events.FLAG_ONE_BODY_NOT_FINITE: 'test_a_nan_vertex_reaches_the_one_body_step',
    events.FLAG_DIIS_NOT_FINITE: 'test_a_nan_vertex_ends_the_two_body_loop',
    events.FLAG_DIIS_SINGULAR: 'test_parquet_diis.py::test_an_exactly_singular_system',
    events.FLAG_QS_FOCK_NOT_FINITE: 'test_a_nan_ao_integral_in_qspa',
    events.FLAG_BSE_NOT_FINITE: 'test_a_nan_vertex_reaches_the_one_body_step',
    events.FLAG_SELF_ENERGY_NOT_FINITE: 'test_a_nan_vertex_reaches_the_one_body_step',
    events.FLAG_STATIC_SIGMA_NOT_FINITE: 'test_a_nan_ao_integral_in_qspa',
    events.FLAG_QP_NOT_FINITE: 'test_a_nan_vertex_reaches_the_one_body_step',
    events.FLAG_VERTEX_POLE: 'test_a_nan_ao_integral_in_qspa',
    events.FLAG_VERTEX_NOT_FINITE: 'test_a_nan_vertex_ends_the_two_body_loop',
    events.FLAG_EH_A_MINUS_B_NOT_POSITIVE: 'test_the_stretched_h2_triplet_instability',
    events.FLAG_EH_NEGATIVE_EXCITATIONS: 'test_the_stretched_h2_triplet_instability',
    events.FLAG_EH_TDA_NEGATIVE_EXCITATIONS: 'test_the_stretched_h2_triplet_instability',
    events.FLAG_PP_BRANCH_COUNT: 'test_scaled_vertices_break_the_bses',
    events.FLAG_PP_COMPLEX_ROOTS: 'test_scaled_vertices_break_the_bses',
    events.FLAG_PP_LOWDIN_SMALL_EIGENVALUE: 'test_scaled_vertices_break_the_bses',
    events.FLAG_PP_EC_INCONSISTENT: 'test_scaled_vertices_break_the_bses',
    events.FLAG_PP_DGEEV_FAILED: 'test_lapack_failures',
    events.FLAG_BSE_DIAGONALIZATION_FAILED: 'test_lapack_failures',
    events.FLAG_EH_ZERO_MODE: 'test_the_spin_flip_zero_mode',
    events.FLAG_Z_OUTSIDE: 'test_a_linearized_energy_outside_the_weights',
    events.FLAG_NO_ROOT: 'test_an_equation_without_a_root',
    events.FLAG_QP_BRANCH_FLIPPED:
        'test_parquet_qp_solver.py::test_a_branch_the_largest_weight_left_is_reported',
}


def test_every_flag_has_its_test():
    """Every FLAG_* constant of `events` is raised by a test named here."""
    declared = {value for name, value in vars(events).items() if name.startswith('FLAG_')}
    assert set(FLAG_TESTS) == declared, sorted(declared ^ set(FLAG_TESTS))
    here = globals()
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    for test in FLAG_TESTS.values():
        if '::' in test:
            path, name = test.split('::')
            with open(os.path.join(tests_dir, path)) as f:
                assert f'def {name}(' in f.read(), test
        else:
            assert callable(here.get(test)), test


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
