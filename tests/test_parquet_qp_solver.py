"""The parquet quasiparticle equation: qp_solver 'find_qp_root', 'linearized', a function.

FLEX, osPA and evPA hand w = eHF_p + Sigma_pp(w) to the house's root finder
('find_qp_root', the default: `restricted_self_energy.DiagonalSelfEnergy.
solve_qp_scan`, `spin_orbital.solve_qp_scan_so`), take eQPlin ('linearized',
Quack's lin_parquet and G_evParquet's only branch), or hand it to a function
(`options.checked_qp_solver`); Quack's own search, R_Parquet_QP_graph's
Newton from eQPlin, is such a function of the tests (`tests.
parquet_quack_newton.quack_newton`, 'Newton' below), not part of the
package. Held here, on water and ammonia 6-31G:

* the default is the house's finder, reached once per orbital solved in
  both drivers (one per spin pair); a function handed in is called as its
  contract says, with the driver's self-energy (in evPA the one each
  iteration built on the previous iteration's energies), its own copies of
  the arrays and the orbitals to solve (the first of each spin pair), and
  what it returns is kept, recorded under its label, counted and flagged (a
  verdict False, an energy not finite, a Z outside (0, 1]); a scan of the
  package's source finds no shape of a root search of its own that it
  knows (loops that can end early or test a magnitude but the parquet
  iterations, one lookup and the input checks; call cycles but the labels
  of a partial; scipy.optimize, a run-time import, a house import not
  listed), though no static scan can prove that there is none: searches
  planted in thirty shapes and the tests' Quack Newton are found, and five
  shapes it is known to miss are checked to pass it (others pass it too);

* where both reach one root, the finder's root is Newton's: the HOMO and LUMO
  to ROOT_EV, and at s_1b = 50, where no root sits beside a pole, every
  orbital to EVERY_ORBITAL_EV (Newton stops at |f| <= 1e-6 Ha at its last
  iterate but one, the finder at 1e-11 Ha); each route's residual |f|;
* the flags each route raises: Newton's 'z_outside' for the roots of Z -0.17
  and 1.54 that it keeps beside poles at s_1b = 1e8 (water FLEX, orbitals 7
  and 8; over spin orbitals both of each pair, 14 to 17), which the finder,
  keeping roots of valid weight, does not raise there; both routes'
  'z_outside' for the root of Z = 1.067 of osPA with two-body DIIS at
  s_1b = 50 (water orbital 11), and the spin-orbital finder's for both
  spin orbitals of the pair whose root it solves once (Z = 1.08, spin
  orbitals 22 and 23, the second through its partner);
* 'linearized' is eQPlin to the bit, and Quack's (the recorded Fortran
  numbers of `test_parquet_restricted_vs_quack` and
  `test_parquet_so_vs_quack`);
* the finder is reached once per orbital solved (one per spin pair), and in
  evPA (both drivers) it follows the previous iteration's roots from the
  second iteration on, where their Z lay in (0, 1] (`options.roots_to_follow`), and searches
  afresh where it did not: following its own roots gives them back at no
  more evaluations than the search (the window followed is searched whole
  for the branch report), an unfollowed (NaN) entry is the search without
  a follow, a branch whose largest weight has left is reported, and
  ammonia's evPA stays on Newton's branch (it cycled between three N 1s
  roots, 25 eV off at the third iteration, when every iteration re-decided
  by weight).

`test_parquet_qp_solver_slow` holds the longer runs (both drivers' evPA to
convergence, the spin-orbital osPA flags).

Run: python tests/test_parquet_qp_solver.py, or under pytest.
"""
import ast
import functools
import os
import pathlib
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import ao2mo, gto, scf

import src.Solvers.qp_equation as qp_equation
from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.Parquet import restricted, spin_orbital
from src.SingleReference.Parquet.options import roots_to_follow
from src.SingleReference.Parquet.restricted_self_energy import DiagonalSelfEnergy
from src.SingleReference.Parquet.spin_orbital_self_energy import DiagonalSelfEnergySO
from tests.parquet_quack_newton import quack_newton
from tests.pins import assert_same
from tests.test_parquet_restricted_vs_quack import (EXACT_EQP, EXACT_Z_REL, QUACK, mo_integrals,
                                                    rhf)
from tests.test_parquet_so_vs_quack import QUACK as QUACK_SO
from tests.test_parquet_so_vs_quack import open_shell_spin_orbitals

#: eV. The HOMO and LUMO of the two routes where they reach one root: 1.5e-15
#: measured (restricted, s_1b 50 and 1e8; spin-orbital 5.4e-12).
ROOT_EV = 1e-8
#: eV. Every orbital at s_1b = 50: 4.9e-6 measured (restricted), 1.9e-5
#: (spin-orbital, ammonia): Newton's tolerance, |f| <= 1e-6 Ha at its
#: last iterate but one, divided by Z.
EVERY_ORBITAL_EV = 5e-5
#: Ha. The finder's residual |f(eqp)| at its roots: 2.9e-12 measured
#: (QP_ROOT_TOL 1e-11).
SCAN_RESIDUAL = 1e-10
#: Ha. Newton's residual at the energy it returns: 3e-7 measured (its test
#: is R_Parquet_QP_graph's thresh, 1e-6, at the iterate before the last step).
NEWTON_RESIDUAL = 1e-6
#: eV. Every orbital of ammonia's evPA (s_1b = 50), finder against Newton, at
#: every one of EV_ITERATIONS iterations: 1.0e-5 measured.
EVPA_BRANCH_EV = 3e-5
#: One-body iterations of the evPA comparisons.
EV_ITERATIONS = 4
#: Rounding of the Z values the module docstring states.
Z_DECIMALS = 2

T = True
#: evPA as the follow rule was measured on: TDA, s_2b = 1, s_1b = 50.
EVPA = dict(tda_eh=T, tda_pp=T, eta_1b=50.0, eta_2b=1.0, conv_2b=1e-8, max_it_2b=400,
            conv_1b=1e-7)


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def flagged_orbitals(flags, name='z_outside'):
    """The 0-based orbitals the messages of the flags `name` give."""
    found = set()
    for entry in flags:
        if entry['flag'] == name:
            text = entry['message']
            marker = 'spin orbital ' if 'spin orbital ' in text else 'orbital '
            found.add(int(text.split(marker, 1)[1].split()[0]))
    return sorted(found)


@functools.lru_cache(maxsize=None)
def flex_self_energy(s1b):
    """Water's FLEX (TDA, s_2b = 1) by Newton and the DiagonalSelfEnergy of its
    spectra."""
    eps, eri, nO = mo_integrals('water')
    result = quiet(restricted.ev_parquet, eps, eri, nO, 'flex', eta_1b=s1b, eta_2b=1.0,
                   qp_solver=quack_newton)
    return result, DiagonalSelfEnergy(s1b, eps, eri, nO, result.spectra)


def newton_on(se, eps_hf, eqp_lin, z_lin):
    """((eqp, z, converged), {'n_iter': steps per orbital}): Quack's Newton on
    the equations of the DiagonalSelfEnergy `se`, its self-energy handed in
    as the restricted driver hands it (one evaluation per step)."""
    steps = np.zeros(len(eps_hf), dtype=int)

    def sigma(p, w):
        steps[p] += 1
        sig, dsig = se.at(p, w)
        return np.float64(sig), np.float64(dsig)
    solution = quack_newton(sigma, eps_hf, eqp_lin, z_lin, np.arange(len(eps_hf)))
    return solution, {'n_iter': steps}


@functools.lru_cache(maxsize=None)
def both_routes(s1b):
    """(newton, its steps, scan, its out) on water's FLEX self-energy at s1b,
    as arrays (eqp, z, verdict)."""
    eps, _, _ = mo_integrals('water')
    _, se = flex_self_energy(s1b)
    sigc, z = se.diagonal()
    scan_out = {}
    newton, newton_out = newton_on(se, eps, eps + z * sigc, z)
    scan = quiet(se.solve_qp_scan, eps, out=scan_out)
    return newton, newton_out, scan, scan_out


def residual(se, eps_hf, w):
    """|w_p - eHF_p - Sigma_pp(w_p)| of every orbital."""
    return np.array([abs(w[p] - eps_hf[p] - se.at(p, w[p])[0]) for p in range(len(w))])


@pytest.mark.parametrize('s1b', [50.0, 1e8])
def test_the_finder_reaches_newtons_root(s1b):
    """HOMO and LUMO of both routes; at s_1b = 50 every orbital; the
    residuals; the finder's evaluations at least Newton's steps; its record."""
    eps, _, nO = mo_integrals('water')
    _, se = flex_self_energy(s1b)
    (e_n, z_n, ok_n), out_n, (e_s, z_s, found), out_s = both_routes(s1b)
    gap = np.abs(e_n - e_s) * HARTREE_TO_EV
    assert gap[nO - 1] <= ROOT_EV and gap[nO] <= ROOT_EV, (s1b, gap)
    if s1b == 50.0:
        assert gap.max() <= EVERY_ORBITAL_EV, gap
    assert found.all() and ok_n.all(), s1b
    assert residual(se, eps, e_s).max() <= SCAN_RESIDUAL, s1b
    assert residual(se, eps, e_n).max() <= NEWTON_RESIDUAL, s1b
    assert np.all(out_s['n_eval'] >= out_n['n_iter']), s1b
    assert {'n_eval', 'no_root', 'z_outside', 'followed', 'flipped',
            'numerical_flags'} <= set(out_s), sorted(out_s)
    assert not out_s['followed'].any() and not out_s['no_root'].any(), s1b


def test_newton_keeps_and_flags_the_roots_beside_poles():
    """At s_1b = 1e8 Newton keeps water orbitals 7 and 8 at roots of
    Z = -0.17 and 1.54, and the driver flags them 'z_outside' (in the run's
    numerical_flags); the finder keeps roots of valid weight there, unflagged."""
    result, _ = flex_self_energy(1e8)
    assert flagged_orbitals(result.numerical_flags) == [7, 8], result.numerical_flags
    assert np.round(result.z[[7, 8]], Z_DECIMALS).tolist() == [-0.17, 1.54], result.z
    (_, z_n, _), _, (_, z_s, _), out_s = both_routes(1e8)
    assert np.flatnonzero(~((z_n > 0.0) & (z_n <= 1.0))).tolist() == [7, 8], z_n
    assert out_s['numerical_flags'] == [] and not out_s['z_outside'].any(), out_s
    assert np.all((z_s[[7, 8]] > 0.0) & (z_s[[7, 8]] <= 1.0)), z_s


def test_spin_orbital_newton_keeps_and_flags_the_roots_beside_poles():
    """The same over spin orbitals (run_parquet_so on the RHF, its pairs):
    spin orbitals 14 to 17, the pairs of orbitals 7 and 8, at Z = -0.17 and
    1.54, each flagged (the second of a pair too)."""
    result = quiet(spin_orbital.run_parquet_so, rhf('water'), 'flex', eta_1b=1e8, eta_2b=1.0,
                   qp_solver=quack_newton)
    outside = np.flatnonzero(~((result.z > 0.0) & (result.z <= 1.0))).tolist()
    assert outside == [14, 15, 16, 17], result.z
    assert np.round(result.z[outside], Z_DECIMALS).tolist() == [-0.17, -0.17, 1.54, 1.54]
    assert flagged_orbitals(result.numerical_flags) == outside, result.numerical_flags


@pytest.mark.parametrize('qp_solver', [quack_newton, 'find_qp_root'], ids=['newton', 'finder'])
def test_a_root_of_weight_above_one_is_flagged_on_both_routes(qp_solver):
    """osPA with two-body DIIS 5 at s_1b = 50 keeps water orbital 11 at a
    root of Z = 1.067 on both routes and flags it 'z_outside'."""
    eps, eri, nO = mo_integrals('water')
    result = quiet(restricted.ev_parquet, eps, eri, nO, 'os', eta_1b=50.0, eta_2b=1.0,
                   max_diis_2b=5, qp_solver=qp_solver)
    assert flagged_orbitals(result.numerical_flags) == [11], result.numerical_flags
    assert round(float(result.z[11]), 3) == 1.067, result.z[11]


def test_the_spin_orbital_finder_flags_both_spin_orbitals_of_a_pair():
    """run_parquet_so osPA (s_2b = 1, s_1b = 50) on the water RHF: the finder
    solves the pair of spin orbitals 22 and 23 once, at a root of Z = 1.08,
    and flags both, the second as taken from its spin partner."""
    result = quiet(spin_orbital.run_parquet_so, rhf('water'), 'os', eta_1b=50.0, eta_2b=1.0,
                   max_it_2b=20, conv_2b=1e-6, qp_solver='find_qp_root')
    assert flagged_orbitals(result.numerical_flags) == [22, 23], result.numerical_flags
    assert np.round(result.z[[22, 23]], Z_DECIMALS).tolist() == [1.08, 1.08], result.z
    partner = [e['message'] for e in result.numerical_flags
               if e['flag'] == 'z_outside' and 'spin partner' in e['message']]
    assert len(partner) == 1 and 'spin orbital 23 ' in partner[0], partner


def test_following_its_own_roots_gives_them_back():
    """The finder handed its own roots as the ones to follow returns them
    (refined from the same brackets: the same roots and Z, held as
    `tests.pins.assert_same` holds two computations), followed where their Z
    lies in (0, 1], at no more evaluations per orbital than the search: the
    window followed is searched whole for the branch report, no cell twice
    (measured: the search's own counts, 154 to 4 044 per orbital); NaN
    entries are solved as without a follow; a follow of another shape is
    refused."""
    eps, _, _ = mo_integrals('water')
    _, se = flex_self_energy(50.0)
    _, _, (e_s, z_s, _), out_s = both_routes(50.0)
    out = {}
    e_f, z_f, _ = quiet(se.solve_qp_scan, eps, follow=e_s, out=out)
    assert_same(e_f, e_s, label='followed eqp')
    assert_same(z_f, z_s, unit='1', label='followed Z')
    assert np.array_equal(out['followed'], (z_s > 0.0) & (z_s <= 1.0)), out['followed']
    assert np.all(out['n_eval'] <= out_s['n_eval']), (out['n_eval'], out_s['n_eval'])
    out = {}
    e_nan, z_nan, _ = quiet(se.solve_qp_scan, eps, follow=np.full_like(eps, np.nan), out=out)
    assert_same(e_nan, e_s, label='eqp without a follow')
    assert_same(z_nan, z_s, unit='1', label='Z without a follow')
    assert not out['followed'].any()
    with pytest.raises(ValueError, match='follow has shape'):
        se.solve_qp_scan(eps, follow=e_s[:-1])


def test_the_follow_rule():
    """roots_to_follow: the previous energies where the previous Z lies in
    (0, 1], NaN where it lies outside or is NaN."""
    e = np.array([-1.0, -0.5, 0.2, 0.7, 1.1])
    z = np.array([0.8, 1.0, 1.067, -0.17, np.nan])
    follow = roots_to_follow(e, z)
    assert follow[:2].tolist() == [-1.0, -0.5] and np.isnan(follow[2:]).all(), follow
    with pytest.raises(ValueError):
        roots_to_follow(e, z[:-1])


def test_linearized_is_eqp_lin_and_quacks():
    """'linearized' returns eQPlin itself, with no Newton verdict or count,
    and it is Quack's linearized energy (R_evParquet's, G_evParquet's)."""
    eps, eri, nO = mo_integrals('water')
    r = quiet(restricted.ev_parquet, eps, eri, nO, 'flex', tda_eh=True, tda_pp=True,
              eta_1b=1e8, eta_2b=1.0, qp_solver='linearized')
    assert np.array_equal(r.eqp, r.eqp_lin)
    assert r.qp_converged is None and r.history[0]['qp_n_iter'] is None
    assert r.history[0]['qp_solver'] == r.options['qp_solver'] == 'linearized'
    quack = QUACK['water/flex_tda_lin']
    assert np.abs(r.eqp - quack['eqp']).max() <= EXACT_EQP
    assert (np.abs(r.z - quack['z']) / np.abs(quack['z'])).max() <= EXACT_Z_REL
    eps_so, eri_so, nO_so = open_shell_spin_orbitals('oh')
    g = quiet(spin_orbital.ev_parquet_so, eps_so, eri_so, nO_so, 'os', tda_eh=True, tda_pp=True,
              eta_1b=50.0, eta_2b=1.0, max_it_2b=1, conv_2b=1e-6, qp_solver='linearized')
    assert np.array_equal(g.eqp, g.eqp_lin)
    assert g.qp_converged is None and g.history[0]['qp_n_iter'] is None
    assert np.abs(g.eqp - QUACK_SO['oh/flex_tda_s1b_50']['eqp']).max() <= EXACT_EQP


def counting_finder(monkeypatch):
    """Count the calls of the house's find_qp_root (the finder the parquet
    scans reach through solve_qp_equation_scan)."""
    calls = []
    real = qp_equation.find_qp_root

    def counted(source, a, **kwargs):
        calls.append(a)
        return real(source, a, **kwargs)
    monkeypatch.setattr(qp_equation, 'find_qp_root', counted)
    return calls


def test_the_restricted_finder_is_called_once_per_orbital(monkeypatch):
    calls = counting_finder(monkeypatch)
    eps, _, _ = mo_integrals('water')
    _, se = flex_self_energy(50.0)
    quiet(se.solve_qp_scan, eps)
    assert len(calls) == len(eps)


def test_the_spin_orbital_finder_is_called_once_per_spin_pair(monkeypatch):
    """run_parquet_so on an RHF solves the first of each alpha/beta pair and
    copies its root to the second (He/6-31G: two pairs, two calls)."""
    mol = gto.M(atom='He 0 0 0', basis='6-31g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    calls = counting_finder(monkeypatch)
    g = quiet(spin_orbital.run_parquet_so, mf, 'flex', qp_solver='find_qp_root')
    assert len(calls) == 2, calls
    assert np.array_equal(g.eqp[0::2], g.eqp[1::2])
    assert g.history[0]['qp_n_iter'][1::2].tolist() == [0, 0]


def test_the_driver_follows_the_previous_iteration(monkeypatch):
    """evPA hands the finder no follow at its first iteration and
    roots_to_follow(eOld, previous Z) from the second on; FLEX and osPA hand
    none."""
    seen = []
    real = DiagonalSelfEnergy.solve_qp_scan

    def spy(self, eps_hf, follow=None, out=None):
        seen.append(None if follow is None else np.array(follow))
        return real(self, eps_hf, follow=follow, out=out)
    monkeypatch.setattr(DiagonalSelfEnergy, 'solve_qp_scan', spy)
    eps, eri, nO = mo_integrals('water')
    result = quiet(restricted.ev_parquet, eps, eri, nO, 'ev', max_it_1b=3,
                   qp_solver='find_qp_root', **EVPA)
    assert len(seen) == result.n_it_1b == 3 and seen[0] is None
    for k in (1, 2):
        want = roots_to_follow(result.history[k - 1]['eqp'], result.history[k - 1]['z'])
        assert np.array_equal(seen[k], want, equal_nan=True), k
    seen.clear()
    for scheme in ('flex', 'os'):
        quiet(restricted.ev_parquet, eps, eri, nO, scheme, eta_2b=1.0, qp_solver='find_qp_root')
    assert seen == [None, None]


def test_the_spin_orbital_driver_follows_the_previous_iteration(monkeypatch):
    """The same of the spin-orbital evPA (run_parquet_so on He/6-31G, its
    pairs): no follow at the first iteration, roots_to_follow(eOld, previous
    Z) over the spin orbitals from the second on; FLEX and osPA hand none."""
    seen = []
    real = spin_orbital.solve_qp_scan_so

    def spy(*args, follow=None, **kwargs):
        seen.append(None if follow is None else np.array(follow))
        return real(*args, follow=follow, **kwargs)
    monkeypatch.setattr(spin_orbital, 'solve_qp_scan_so', spy)
    mol = gto.M(atom='He 0 0 0', basis='6-31g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    result = quiet(spin_orbital.run_parquet_so, mf, 'ev', eta_2b=1.0, max_it_1b=3,
                   conv_1b=1e-12, qp_solver='find_qp_root')
    assert len(seen) == result.n_it_1b == 3 and seen[0] is None
    for k in (1, 2):
        want = roots_to_follow(result.history[k - 1]['eqp'], result.history[k - 1]['z'])
        assert np.array_equal(seen[k], want, equal_nan=True), k
    seen.clear()
    for scheme in ('flex', 'os'):
        quiet(spin_orbital.run_parquet_so, mf, scheme, eta_2b=1.0, qp_solver='find_qp_root')
    assert seen == [None, None]


@functools.lru_cache(maxsize=None)
def water_minimal():
    """Water's RHF in STO-3G (conv_tol 1e-12): a small evPA that flips."""
    mol = gto.M(atom='O 0 0 0; H 0.7571 0 0.5861; H -0.7571 0 0.5861', basis='sto-3g',
                verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


#: evPA where the house finder reports a followed branch flipped at the second
#: iteration (water/STO-3G, s_1b = 1e8, two-body DIIS 5): orbital 1, over spin
#: orbitals its pair 2 and 3.
FLIPS = dict(eta_1b=1e8, eta_2b=1.0, max_diis_2b=5, conv_2b=1e-8, max_it_2b=400, max_it_1b=2,
             conv_1b=1e-7)
FLIPPED = {'run_parquet': [1], 'run_parquet_so': [2, 3]}


@pytest.mark.parametrize('front', [restricted.run_parquet, spin_orbital.run_parquet_so])
def test_a_branch_the_largest_weight_left_is_reported(front):
    """Where the house finder reports that the largest weight has left a
    root followed from the previous evPA iteration, the drivers say so as
    the house does for a followed root: a RuntimeWarning in
    qp_equation.branch_warning's words, the named flag
    'parquet_qp_branch_flipped' (for both spin orbitals of a pair, the
    second through its partner), the history's per-orbital 'qp_followed'
    and 'qp_flipped', and the result's for the last iteration. The first
    iteration follows nothing; Newton and the linearized energy follow
    nothing and report nothing. The report is the whole of it: the search
    without a follow on the same self-energy finds a root of larger weight
    for the orbital flipped and, for every other, the root followed."""
    mf = water_minimal()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = front(mf, 'ev', qp_solver='find_qp_root', **FLIPS)
    first, second = result.history
    assert not first['qp_followed'].any() and not first['qp_flipped'].any()
    assert second['qp_followed'].all(), second['qp_followed']
    flipped = np.flatnonzero(second['qp_flipped']).tolist()
    assert flipped == FLIPPED[front.__name__], flipped
    assert flagged_orbitals(result.numerical_flags, 'parquet_qp_branch_flipped') == flipped
    said = [str(w.message) for w in caught if w.category is RuntimeWarning]
    assert len(said) == len(flipped) and all('the branch is kept' in s for s in said), said
    assert np.array_equal(result.qp_followed, second['qp_followed'])
    assert np.array_equal(result.qp_flipped, second['qp_flipped'])
    if front is restricted.run_parquet:
        eps, eri, nO = mo_integrals_of(mf)
        se = DiagonalSelfEnergy(FLIPS['eta_1b'], first['eqp'], eri, nO, result.spectra)
        w_largest, z_largest, _ = quiet(se.solve_qp_scan, eps)
        assert np.all(z_largest[flipped] > second['z'][flipped]), (z_largest, second['z'])
        kept = np.flatnonzero(~second['qp_flipped'])
        assert_same(second['eqp'][kept], w_largest[kept], label='eqp not flipped')
        assert_same(second['z'][kept], z_largest[kept], unit='1', label='Z not flipped')
    for qp_solver in (quack_newton, 'linearized'):
        other = quiet(front, mf, 'ev', qp_solver=qp_solver, **FLIPS)
        assert other.qp_followed is None and other.qp_flipped is None
        assert all(h['qp_followed'] is None and h['qp_flipped'] is None for h in other.history)
        assert 'parquet_qp_branch_flipped' not in {e['flag'] for e in other.numerical_flags}


def mo_integrals_of(mf):
    """(eps, <pq|rs>, nO) of an RHF."""
    n = mf.mo_coeff.shape[1]
    eri = ao2mo.full(mf.mol, mf.mo_coeff, compact=False).reshape((n,) * 4)
    return (np.asarray(mf.mo_energy), np.ascontiguousarray(eri.transpose(0, 2, 1, 3)),
            mf.mol.nelectron // 2)


@functools.lru_cache(maxsize=None)
def ammonia_evpa(qp_solver):
    eps, eri, nO = mo_integrals('ammonia')
    return quiet(restricted.ev_parquet, eps, eri, nO, 'ev', max_it_1b=EV_ITERATIONS,
                 qp_solver=qp_solver, **EVPA)


def test_evpa_stays_on_newtons_branch():
    """Ammonia's evPA at s_1b = 50: at every iteration the finder's energies
    are Newton's on every orbital (its N 1s cycled between three roots, 25 eV
    off at the third iteration, when every iteration re-decided by weight)."""
    newton, scan = ammonia_evpa(quack_newton), ammonia_evpa('find_qp_root')
    assert newton.n_it_2b == scan.n_it_2b
    for k, (hn, hs) in enumerate(zip(newton.history, scan.history)):
        gap = np.abs(hn['eqp'] - hs['eqp']).max() * HARTREE_TO_EV
        assert gap <= EVPA_BRANCH_EV, (k, gap)


def test_the_spin_orbital_finder_reaches_newtons_root():
    """run_parquet_so (RHF, pairs) FLEX on water at s_1b = 50: the finder's
    HOMO and LUMO are Newton's, every spin orbital within EVERY_ORBITAL_EV,
    at no fewer evaluations than Newton's steps."""
    mf = rhf('water')
    newton = quiet(spin_orbital.run_parquet_so, mf, 'flex', eta_1b=50.0, eta_2b=1.0,
                   qp_solver=quack_newton)
    scan = quiet(spin_orbital.run_parquet_so, mf, 'flex', eta_1b=50.0, eta_2b=1.0,
                 qp_solver='find_qp_root')
    gap = np.abs(newton.eqp - scan.eqp) * HARTREE_TO_EV
    nO = newton.nO
    assert gap[nO - 1] <= ROOT_EV and gap[nO] <= ROOT_EV, gap
    assert gap.max() <= EVERY_ORBITAL_EV, gap
    assert np.all(scan.history[0]['qp_n_iter'] >= newton.history[0]['qp_n_iter'])
    assert np.array_equal(scan.eqp[0::2], scan.eqp[1::2])
    assert scan.numerical_flags == newton.numerical_flags == []


# --------------------------------------------------------------------------
# the default, a function handed in, and the scan for a search of the package's own


def test_the_default_reaches_the_one_finder(monkeypatch):
    """With no qp_solver, both drivers, at the front door and below, solve
    FLEX by the house's find_qp_root, once per orbital solved (over spin
    orbitals once per alpha/beta pair at the front door, every spin orbital
    below, where no pairs are given), and record it by name."""
    mf = water_minimal()
    eps, eri, nO = mo_integrals_of(mf)
    norb = len(eps)
    ref = spin_orbital.spin_orbital_reference(mf)
    eri_so = spin_orbital.spin_orbital_eri(ref.eri_ao, ref.mo_coeff)
    runs = {
        'run_parquet': (lambda: restricted.run_parquet(mf, 'flex', eta_2b=1.0), norb),
        'ev_parquet': (lambda: restricted.ev_parquet(eps, eri, nO, 'flex', eta_2b=1.0), norb),
        'run_parquet_so': (lambda: spin_orbital.run_parquet_so(mf, 'flex', eta_2b=1.0), norb),
        'ev_parquet_so': (lambda: spin_orbital.ev_parquet_so(ref.eps, eri_so, ref.nO, 'flex',
                                                             eta_2b=1.0), 2 * norb),
    }
    calls = counting_finder(monkeypatch)
    for name, (run, solved) in runs.items():
        calls.clear()
        result = quiet(run)
        assert len(calls) == solved, (name, len(calls))
        assert result.options['qp_solver'] == result.history[0]['qp_solver'] == 'find_qp_root'


#: The shift `shifted_solver` gives eQPlin, Ha.
SHIFT = 1e-3
#: What `shifted_solver` was handed, one dict per call.
HANDED = []


def shifted_solver(sigma, eps_hf, eqp_lin, z_lin, orbitals):
    """A qp_solver function for the contract tests: it records what it is
    handed and one evaluation of the self-energy per orbital at its bare
    level, writes over the arrays handed in (its own copies), and returns
    eQPlin - SHIFT with the diagonal Z, but a verdict False for the first
    orbital solved, a Z of 2 for the second and a NaN energy for the last."""
    HANDED.append(dict(eps_hf=eps_hf.copy(), eqp_lin=eqp_lin.copy(), z_lin=z_lin.copy(),
                       orbitals=orbitals.copy(),
                       at_bare={int(p): sigma(p, eps_hf[p]) for p in orbitals}))
    eqp, z = eqp_lin - SHIFT, z_lin.copy()
    converged = np.ones(len(eqp), dtype=bool)
    converged[orbitals[0]] = False
    z[orbitals[1]] = 2.0
    eqp[orbitals[-1]] = np.nan
    for handed in (eps_hf, eqp_lin, z_lin):
        handed[:] = 0.0
    return eqp, z, converged


@pytest.mark.parametrize('front', [restricted.run_parquet, spin_orbital.run_parquet_so])
def test_a_function_handed_in_is_called_and_kept(front):
    """A function as qp_solver is called once per one-body step as its
    contract says: the driver's self-energy at w (two float64: Sigma_pp, at
    the bare level the run's SigC as `tests.pins.assert_same` holds two
    computations, and its derivative, the diagonal Z's),
    the bare levels and eQPlin of the result, the orbitals to solve (the
    first of each spin pair). Its arrays are its own: writing over them
    changes nothing of the run. What it returns is kept as returned (the
    second of a pair takes the first's), its evaluations are counted per
    orbital, it is recorded under its module and qualified name, and the
    driver flags it: the verdict False 'parquet_qp_not_converged', the NaN
    energy 'parquet_qp_not_finite', the Z of 2 'z_outside', each with its
    warning class. It follows nothing."""
    mf = water_minimal()
    HANDED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = front(mf, 'flex', eta_2b=1.0, qp_solver=shifted_solver)
    (handed,) = HANDED
    n = len(result.eqp)
    pairs = front is spin_orbital.run_parquet_so
    solved = np.arange(0, n, 2) if pairs else np.arange(n)
    source = np.arange(n) - np.arange(n) % 2 if pairs else np.arange(n)
    assert np.array_equal(handed['orbitals'], solved), handed['orbitals']
    assert np.array_equal(handed['eps_hf'], result.eps_hf)
    assert np.array_equal(handed['eqp_lin'], result.eqp_lin)
    assert all(type(x) is np.float64 for pair in handed['at_bare'].values() for x in pair)
    sig = np.array([handed['at_bare'][p][0] for p in solved])
    dsig = np.array([handed['at_bare'][p][1] for p in solved])
    assert_same(sig, result.sigc[solved], label='Sigma_pp at the bare level')
    assert np.allclose(handed['z_lin'][solved], 1.0 / (1.0 - dsig), rtol=1e-12, atol=0.0)
    first, second, last = solved[0], solved[1], solved[-1]
    eqp = handed['eqp_lin'] - SHIFT
    eqp[last] = np.nan
    z = handed['z_lin'].copy()
    z[second] = 2.0
    ok = np.ones(n, dtype=bool)
    ok[first] = False
    eqp, z, ok = eqp[source], z[source], ok[source]
    assert np.array_equal(result.eqp, eqp, equal_nan=True), (result.eqp, eqp)
    assert np.array_equal(result.z, z) and np.array_equal(result.qp_converged, ok)
    assert np.array_equal(result.history[0]['eqp'], eqp, equal_nan=True)
    counted = np.zeros(n, dtype=int)
    counted[solved] = 1
    assert np.array_equal(result.history[0]['qp_n_iter'], counted)
    label = f'{shifted_solver.__module__}.{shifted_solver.__qualname__}'
    assert result.options['qp_solver'] == result.history[0]['qp_solver'] == label
    assert result.qp_followed is None and result.qp_flipped is None
    named = {}
    for entry in result.numerical_flags:
        named.setdefault(entry['flag'], []).append(entry['message'])
    partners = (lambda p: [int(p), int(p) + 1]) if pairs else (lambda p: [int(p)])
    assert set(named) == {'parquet_qp_not_converged', 'parquet_qp_not_finite', 'z_outside'}
    assert len(named['parquet_qp_not_converged']) == len(named['parquet_qp_not_finite']) == 1
    assert f'{partners(first)} (0-based)' in named['parquet_qp_not_converged'][0]
    assert f'{partners(last)} (0-based)' in named['parquet_qp_not_finite'][0]
    outside = np.flatnonzero(np.isfinite(eqp) & ~((z > 0.0) & (z <= 1.0))).tolist()
    assert set(partners(second)) <= set(outside)
    assert flagged_orbitals(result.numerical_flags) == outside, result.numerical_flags
    assert all(label in m for messages in named.values() for m in messages)
    classes = {type(w.message).__name__ for w in caught}
    assert {'ParquetNotConverged', 'ParquetNotFinite', 'QPWeightOutside'} <= classes, classes


#: What `recording_solver` was handed: the self-energy of each call.
SIGMAS = []
#: Ha. At the second evPA iteration, how far from the self-energy on the
#: previous energies one on eHF lies at w = eOld_p, on every orbital solved
#: (water/STO-3G: 1.2e-3 at least, 0.23 at most measured): what a function
#: handed the wrong one would see.
NOT_ON_EHF = 1e-4


def recording_solver(sigma, eps_hf, eqp_lin, z_lin, orbitals):
    """A qp_solver function that keeps the self-energy it is handed and
    returns the linearized solution, converged."""
    SIGMAS.append(sigma)
    return eqp_lin.copy(), z_lin.copy(), np.ones(len(eqp_lin), dtype=bool)


@pytest.mark.parametrize('front', [restricted.run_parquet, spin_orbital.run_parquet_so])
def test_a_function_sees_each_iterations_self_energy(front):
    """evPA hands a qp_solver function, at each one-body iteration, the
    self-energy that iteration built on the previous iteration's energies
    eOld (eHF at the first), as the run's SigC is built: at w = eOld_p it is
    that iteration's SigC_p (`tests.pins.assert_same`), over spin orbitals
    too, where the driver builds a function's self-energy one spin orbital
    at a time, apart from SigC (water/STO-3G, two iterations, damping 0.75).
    At the second iteration a self-energy built on eHF lies NOT_ON_EHF or
    more away on every orbital solved."""
    mf = water_minimal()
    SIGMAS.clear()
    result = quiet(front, mf, 'ev', eta_2b=1.0, max_it_1b=2, conv_1b=1e-12, damping=0.75,
                   qp_solver=recording_solver)
    assert result.n_it_1b == len(SIGMAS) == 2 and all(result.converged_2b)
    n = len(result.eqp)
    pairs = front is spin_orbital.run_parquet_so
    solved = np.arange(0, n, 2) if pairs else np.arange(n)
    e_old = (result.eps_hf, result.history[0]['eqp'])
    for k, sigma in enumerate(SIGMAS):
        at_e_old = [sigma(p, e_old[k][p])[0] for p in solved]
        assert_same(at_e_old, result.history[k]['sigc'][solved], label=('SigC', k))
    eta = result.options['eta_1b']
    if pairs:
        ref = spin_orbital.spin_orbital_reference(mf)
        eri = spin_orbital.spin_orbital_eri(ref.eri_ao, ref.mo_coeff)
        on_hf = [DiagonalSelfEnergySO(eta, result.eps_hf, eri, result.nO, result.spectra,
                                      orbitals=[p]).at(p, e_old[1][p])[0] for p in solved]
    else:
        _, eri, nO = mo_integrals_of(mf)
        se = DiagonalSelfEnergy(eta, result.eps_hf, eri, nO, result.spectra)
        on_hf = [se.at(p, e_old[1][p])[0] for p in solved]
    gap = np.abs(np.array(on_hf) - result.history[1]['sigc'][solved])
    assert gap.min() > NOT_ON_EHF, gap


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
#: The try statements of the Pythons this runs on (3.11 added except*).
_TRIES = tuple(t for t in (ast.Try, getattr(ast, 'TryStar', None)) if t is not None)
#: What leaves a loop through an exception.
_RAISES = (ast.Raise, ast.Assert)


def _own(nodes):
    """The nodes under `nodes`, not descending into a nested function, class
    or lambda."""
    stack = list(nodes)
    while stack:
        node = stack.pop()
        if isinstance(node, _FUNCTIONS + (ast.ClassDef, ast.Lambda)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _ends_early(loop):
    """Whether a for loop can end before its iterable does: a break of its
    own body, or a return, a raise or an assert anywhere in it."""
    stack = list(loop.body)
    while stack:
        node = stack.pop()
        if isinstance(node, _FUNCTIONS + (ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(node, (ast.Break, ast.Return) + _RAISES):
            return True
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            # a break there is that loop's; a return or a raise still leaves this one
            if any(isinstance(n, (ast.Return,) + _RAISES) for n in _own(node.body + node.orelse)):
                return True
            continue
        stack.extend(ast.iter_child_nodes(node))
    return any(isinstance(n, (ast.Return,) + _RAISES) for n in _own(loop.orelse))


#: A test of a magnitude: a comparison holding a call of the first four, or
#: a call of the last two anywhere; a comparison holding a name with a mark
#: of a tolerance, or a nonzero number of this size or less.
MAGNITUDES = ('abs', 'fabs', 'absolute', 'norm', 'isclose', 'allclose')
TOLERANCE_MARKS = ('tol', 'conv', 'thresh')
SMALL_CONSTANT = 1e-3


def _called(node):
    """The name a call calls: a function's, or an attribute's (a method, a
    module's function)."""
    f = node.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ''


def _tests_a_magnitude(loop):
    """Whether a for loop's own body (not a nested function) tests a
    magnitude (MAGNITUDES, TOLERANCE_MARKS, SMALL_CONSTANT)."""
    for node in _own(loop.body + loop.orelse):
        if isinstance(node, ast.Call) and _called(node) in MAGNITUDES[4:]:
            return True
        if not isinstance(node, ast.Compare):
            continue
        for x in ast.walk(node):
            name = (x.id if isinstance(x, ast.Name) else
                    x.attr if isinstance(x, ast.Attribute) else '')
            if ((isinstance(x, ast.Call) and _called(x) in MAGNITUDES[:4])
                    or any(mark in name.lower() for mark in TOLERANCE_MARKS)
                    or (isinstance(x, ast.Constant) and type(x.value) in (int, float)
                        and 0 < abs(x.value) <= SMALL_CONSTANT)):
                return True
    return False


def loop_census(source):
    """{function: [while loops, for loops a search could be]} of a module's
    source, each loop under the innermost function holding it, by qualified
    name ('<module>' outside any). A for loop is counted where it can end
    early (`_ends_early`), where it lies in the body of a try with an except
    clause of its own function (an exception of a call in it can end it
    there), or where it tests a magnitude (`_tests_a_magnitude`: a fixed
    number of steps whose test masks or skips the step)."""
    loops = {}

    def visit(nodes, scope, prefix, caught):
        for child in nodes:
            if isinstance(child, ast.ClassDef):
                visit(child.body, scope, prefix + child.name + '.', False)
            elif isinstance(child, _FUNCTIONS):
                visit(child.body, prefix + child.name, prefix + child.name + '.', False)
            elif isinstance(child, _TRIES) and child.handlers:
                visit(child.body, scope, prefix, True)
                visit(child.handlers + child.orelse + child.finalbody, scope, prefix, caught)
            else:
                if isinstance(child, ast.While):
                    loops.setdefault(scope, [0, 0])[0] += 1
                elif isinstance(child, (ast.For, ast.AsyncFor)) and (
                        caught or _ends_early(child) or _tests_a_magnitude(child)):
                    loops.setdefault(scope, [0, 0])[1] += 1
                visit(ast.iter_child_nodes(child), scope, prefix, caught)
    visit(ast.parse(source).body, '<module>', '', False)
    return loops


def call_cycles(source):
    """The functions of a module's source on a cycle of references. A
    function, a method or a lambda bound to a name refers to what its own
    body reads by name (to call it, bind it or hand it on), resolved as
    Python resolves it: its own definitions, then those of the functions
    around it, then the module's (a local variable of that name hides
    them); a method's, or a function's inside one, self.name, cls.name or
    class.name to a method of its class. A function on a cycle reaches
    itself (one calling itself among them)."""
    tree = ast.parse(source)
    bound = {id(n.value): n.targets[0].id for n in ast.walk(tree)
             if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Lambda)}

    def name_of(node):
        return bound[id(node)] if id(node) in bound else node.name

    def split(body):
        """(definitions, reads, names bound) of a body, its own: the
        functions, classes and bound lambdas it defines; the names and
        attributes it reads (an unbound lambda's too, and a definition's
        decorators and defaults); the names it binds."""
        defs, reads, stores, stack = [], [], set(), list(body)
        while stack:
            node = stack.pop()
            if isinstance(node, _FUNCTIONS + (ast.ClassDef,)) or id(node) in bound:
                defs.append(node)
                stack.extend(getattr(node, 'decorator_list', []))
                if not isinstance(node, ast.ClassDef):
                    stack.extend(node.args.defaults + [d for d in node.args.kw_defaults if d])
                continue
            if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
                reads.append(node)
            elif isinstance(node, ast.Name):
                stores.add(node.id)
            elif isinstance(node, ast.arg):
                stores.add(node.arg)
            stack.extend(ast.iter_child_nodes(node))
        return defs, reads, stores

    edges = {}

    def visit(defs, prefix, scopes, methods):
        for d in defs:
            if isinstance(d, ast.ClassDef):
                own = [m for m in d.body if isinstance(m, _FUNCTIONS + (ast.ClassDef,))]
                visit(own, prefix + d.name + '.', scopes,
                      (d.name, {m.name: prefix + d.name + '.' + m.name for m in own
                                if isinstance(m, _FUNCTIONS)}))
                continue
            name = prefix + name_of(d)
            inner, reads, stores = split(d.body if isinstance(d.body, list) else [d.body])
            stores |= {a.arg for a in ast.walk(d.args) if isinstance(a, ast.arg)}
            local = {name_of(n): name + '.' + name_of(n) for n in inner
                     if not isinstance(n, ast.ClassDef)}
            around, refs = [local] + scopes, set()
            for x in reads:
                if isinstance(x, ast.Name):
                    if x.id not in stores or x.id in local:
                        refs.update(next(([s[x.id]] for s in around if x.id in s), []))
                elif (methods and isinstance(x.value, ast.Name)
                      and x.value.id in ('self', 'cls', methods[0]) and x.attr in methods[1]):
                    refs.add(methods[1][x.attr])
            edges[name] = refs
            visit(inner, name + '.', around, methods)

    top, _, _ = split(tree.body)
    visit(top, '', [{name_of(n): name_of(n) for n in top if not isinstance(n, ast.ClassDef)}],
          None)
    on_cycle = set()
    for start, out in edges.items():
        seen, stack = set(), list(out)
        while stack and start not in seen:
            name = stack.pop()
            if name not in seen:
                seen.add(name)
                stack.extend(edges.get(name, ()))
        if start in seen:
            on_cycle.add(start)
    return on_cycle


#: The package, where its relative imports start.
PACKAGE = ('src', 'SingleReference', 'Parquet')
#: What the package imports from the house outside itself, by module: the
#: names (None: any, the house's constants). The house's root finder only
#: through solve_qp_equation_scan.
HOUSE_IMPORTS = {
    'src.Base.environment_seam': {'NoEnvironment', 'environment_of'},
    'src.Base.numerical_flags': {'collecting', 'flag', 'flags_into_out', 'gathered'},
    'src.Base.numerics.constants': None,
    'src.Base.pyscf.pyscf_interface': {'get_two_electron_integrals_chemist',
                                       'require_closed_shell_or_unrestricted'},
    'src.Base.utils.parallel.threads': {'blas_single_threaded', 'resolve_workers', 'row_map'},
    'src.Base.utils.timing': {'stage'},
    'src.Solvers.qp_equation': {'QPNoRoot', 'QPRoot', 'QPWeightOutside', 'branch_warning',
                                'quasiparticle_weight', 'solve_qp_equation_scan'},
}


def outside_searches(source):
    """Where a module's source can reach a root search that is not a loop of
    its own: scipy.optimize (an import, an attribute named optimize, a
    string naming it), a module imported by a name computed at run time
    (importlib, __import__, a string naming a module of the house), and an
    import from the house outside the package that HOUSE_IMPORTS does not
    list (a module of it imported whole among them)."""
    found, inside = [], '.'.join(PACKAGE)

    def house(module):
        return module.split('.')[0] == 'src' and not (module + '.').startswith(inside + '.')
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [f'import {a.name}' for a in node.names
                      if a.name.startswith(('scipy.optimize', 'importlib')) or house(a.name)]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if node.level:
                module = '.'.join(PACKAGE[:len(PACKAGE) + 1 - node.level] + (module,)).strip('.')
            names = [a.name for a in node.names]
            if module.startswith(('scipy.optimize', 'importlib')) or (
                    module == 'scipy' and 'optimize' in names):
                found.append(f'from {module} import ' + ', '.join(names))
            elif house(module):
                allowed = HOUSE_IMPORTS.get(module, set())
                extra = [n for n in names if allowed is not None and n not in allowed]
                found += [f'from {module} import ' + ', '.join(extra)] if extra else []
        elif isinstance(node, ast.Attribute) and node.attr in ('optimize', 'import_module'):
            found.append('.' + node.attr)
        elif isinstance(node, ast.Name) and node.id == '__import__':
            found.append('__import__')
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and (
                node.value == 'optimize' or node.value.startswith('scipy.optimize')
                or house(node.value)):
            found.append(repr(node.value))
    return found


def unlisted(source, loops=None, cycles=()):
    """What in a module's source a root search of its own could be, beyond
    what is listed: every function whose loops a search could be differ from
    `loops` ({function: [while loops, for loops]} as `loop_census` counts
    them; a loop at module level is never listed), a function listed and not
    found, one on a cycle of references that `cycles` does not name
    (`call_cycles`), and every way to a search elsewhere
    (`outside_searches`)."""
    found = loop_census(source)
    listed = loops or {}
    out = [f'{name}: {got[0]} while, {got[1]} for' for name, got in sorted(found.items())
           if listed.get(name) != got]
    out += [f'{name}: listed, not found' for name in sorted(set(listed) - set(found))]
    out += [f'{name}: on a call cycle' for name in sorted(call_cycles(source) - set(cycles))]
    return out + outside_searches(source)


#: The loops of the package a search could be, by module and function:
#: [while loops, for loops]. The two-body and one-body iterations of both
#: drivers, the lookup in the SRG static self-energy of the term pair a term
#: joins, and the input checks: for loops over the inputs that raise
#: ValueError at the first wrong one.
LISTED_LOOPS = {
    'restricted.py': {'_two_body_loop': [1, 0], '_ev_loops': [1, 0], '_qs_loops': [1, 0],
                      'qs_parquet': [0, 1]},
    'spin_orbital.py': {'two_body_loop_so': [1, 0], '_ev_loops': [1, 0], '_qs_loops': [1, 0]},
    'spin_orbital_self_energy.py': {'static_self_energy_srg_so': [0, 1], '_checked': [0, 1]},
    'diis.py': {'QuackDIIS.push': [0, 1], 'QuackDIIS._outputs': [0, 1]},
    'options.py': {'checked_qp_solution': [0, 1], 'loop_options': [0, 3]},
    'restricted_bse.py': {'_vertex_tensors': [0, 1]},
    'restricted_static_self_energy.py': {'_checked': [0, 1]},
    'restricted_vertices.py': {'_tensors': [0, 1]},
    'spin_orbital_two_body.py': {'_orbital_count': [0, 1], '_vertex_arrays': [0, 1],
                                 'pp_effective_integrals_so': [0, 1]},
}
#: The package's functions on a cycle of references: the labels of a
#: functools.partial, which name the function it wraps.
LISTED_CYCLES = {'options.py': ('qp_solver_label', 'builder_name')}
#: Root searches in shapes the scan finds, most of them Quack's Newton,
#: w <- w - f / f' on f = w - eHF - Sigma(w) (the one beside a listed loop
#: inside a function LISTED_LOOPS lists).
PLANTED = {
    'while on abs': ('def seeded(w, a, sigma, slope):\n'
                     '    f = 1.0\n'
                     '    while abs(f) > 1e-8:\n'
                     '        f = w - a - sigma(w)\n'
                     '        w = w - f / slope(w)\n'
                     '    return w\n'),
    'for with a break': ('def seeded(w, a, sigma, slope):\n'
                         '    for _ in range(64):\n'
                         '        f = w - a - sigma(w)\n'
                         '        if abs(f) < 1e-8:\n'
                         '            break\n'
                         '        w = w - f / slope(w)\n'
                         '    return w\n'),
    'while True with a return': ('def seeded(w, a, sigma, slope):\n'
                                 '    while True:\n'
                                 '        f = w - a - sigma(w)\n'
                                 '        if abs(f) <= 1e-8:\n'
                                 '            return w\n'
                                 '        w -= f / slope(w)\n'),
    'np.abs and a step through a temporary': ('import numpy as np\n'
                                              'def seeded(w, a, sigma, slope):\n'
                                              '    f = 1.0\n'
                                              '    while np.abs(f) > 1e-8:\n'
                                              '        f = w - a - sigma(w)\n'
                                              '        dw = f / slope(w)\n'
                                              '        w = w - dw\n'
                                              '    return w\n'),
    'a residual helper and a squared test': ('def seeded(w, a, sigma, slope):\n'
                                             '    def residual(x):\n'
                                             '        return x - a - sigma(x)\n'
                                             '    f = residual(w)\n'
                                             '    while f * f > 1e-16:\n'
                                             '        w = w - f / slope(w)\n'
                                             '        f = residual(w)\n'
                                             '    return w\n'),
    'a flag, per orbital': ('def seeded(w, a, sigma, slope, p):\n'
                            '    done = False\n'
                            '    while not done:\n'
                            '        f = w[p] - a[p] - sigma(p, w[p])\n'
                            '        done = abs(f) < 1e-8\n'
                            '        w[p] = w[p] - f / slope(p, w[p])\n'
                            '    return w\n'),
    'a stop helper': ('def seeded(w, a, sigma, slope, small):\n'
                      '    for _ in range(64):\n'
                      '        f = w - a - sigma(w)\n'
                      '        if small(f):\n'
                      '            break\n'
                      '        w = w - f / slope(w)\n'
                      '    return w\n'),
    'a raise to leave a for, caught here': ('class Converged(Exception):\n'
                                            '    pass\n'
                                            'def seeded(w, a, sigma, slope, small):\n'
                                            '    try:\n'
                                            '        for _ in range(64):\n'
                                            '            f = w - a - sigma(w)\n'
                                            '            if small(f):\n'
                                            '                raise Converged\n'
                                            '            w = w - f / slope(w)\n'
                                            '    except Converged:\n'
                                            '        pass\n'
                                            '    return w\n'),
    'a raise to leave a for, caught by the caller': ('def walk(w, a, sigma, slope, small):\n'
                                                     '    for _ in range(64):\n'
                                                     '        f = w - a - sigma(w)\n'
                                                     '        if small(f):\n'
                                                     '            raise StopIteration(w)\n'
                                                     '        w = w - f / slope(w)\n'),
    "a call's raise, caught around the loop": ('def step(w, a, sigma, slope, small):\n'
                                               '    f = w - a - sigma(w)\n'
                                               '    if small(f):\n'
                                               '        raise StopIteration(w)\n'
                                               '    return w - f / slope(w)\n'
                                               'def seeded(w, a, sigma, slope, small):\n'
                                               '    try:\n'
                                               '        for _ in range(64):\n'
                                               '            w = step(w, a, sigma, slope, small)\n'
                                               '    except StopIteration as stop:\n'
                                               '        w = stop.value\n'
                                               '    return w\n'),
    'fixed steps over the orbitals, a mask on the step': (
        'import numpy as np\n'
        'def seeded(sigma, eps_hf, w, p, eps):\n'
        '    for _ in range(64):\n'
        '        sig, dsig = sigma(p, w)\n'
        '        f = w - eps_hf[p] - sig\n'
        '        active = np.abs(f) > eps\n'
        '        w[active] -= f[active] / (1.0 - dsig[active])\n'
        '    return w\n'),
    'fixed steps and a converged flag': ('import numpy as np\n'
                                         'def seeded(w, a, sigma, slope, eps):\n'
                                         '    done = False\n'
                                         '    for _ in range(64):\n'
                                         '        if done:\n'
                                         '            continue\n'
                                         '        f = w - a - sigma(w)\n'
                                         '        done = np.isclose(f, 0.0, atol=eps)\n'
                                         '        w = w - f / slope(w)\n'
                                         '    return w\n'),
    'fixed steps and a squared test': ('def seeded(w, a, sigma, slope):\n'
                                       '    for _ in range(64):\n'
                                       '        f = w - a - sigma(w)\n'
                                       '        if f * f > 1e-16:\n'
                                       '            w = w - f / slope(w)\n'
                                       '    return w\n'),
    'fixed steps and a named tolerance': ('def seeded(w, a, sigma, slope, qp_tol):\n'
                                          '    for _ in range(64):\n'
                                          '        f = w - a - sigma(w)\n'
                                          '        if not -qp_tol < f < qp_tol:\n'
                                          '            w = w - f / slope(w)\n'
                                          '    return w\n'),
    'recursion': ('def seeded(w, a, sigma, slope, n=0):\n'
                  '    f = w - a - sigma(w)\n'
                  '    if abs(f) < 1e-8 or n > 64:\n'
                  '        return w\n'
                  '    return seeded(w - f / slope(w), a, sigma, slope, n + 1)\n'),
    'a method calling itself': ('class Solver:\n'
                                '    def seeded(self, w, a, n=0):\n'
                                '        f = w - a - self.sigma(w)\n'
                                '        if abs(f) < 1e-8 or n > 64:\n'
                                '            return w\n'
                                '        return self.seeded(w - f / self.slope(w), a, n + 1)\n'),
    'mutual recursion': ('def seeded(w, a, sigma, slope, n=0):\n'
                         '    f = w - a - sigma(w)\n'
                         '    if abs(f) < 1e-8 or n > 64:\n'
                         '        return w\n'
                         '    return again(w - f / slope(w), a, sigma, slope, n + 1)\n'
                         'def again(w, a, sigma, slope, n):\n'
                         '    return seeded(w, a, sigma, slope, n)\n'),
    'recursion through a nested function': ('def seeded(w, a, sigma, slope, n=0):\n'
                                            '    def again(x):\n'
                                            '        return seeded(x, a, sigma, slope, n + 1)\n'
                                            '    f = w - a - sigma(w)\n'
                                            '    if abs(f) < 1e-8 or n > 64:\n'
                                            '        return w\n'
                                            '    return again(w - f / slope(w))\n'),
    'recursion through a lambda': ('def seeded(a, sigma, slope):\n'
                                   '    step = lambda w, n=0: (\n'
                                   '        w if abs(w - a - sigma(w)) < 1e-8 or n > 64\n'
                                   '        else step(w - (w - a - sigma(w)) / slope(w), n + 1))\n'
                                   '    return step(a)\n'),
    'recursion through an alias of a method': ('class Solver:\n'
                                               '    def seeded(self, w, a):\n'
                                               '        again = self.seeded\n'
                                               '        f = w - a - self.sigma(w)\n'
                                               '        if abs(f) < 1e-8:\n'
                                               '            return w\n'
                                               '        return again(w - f / self.slope(w), a)\n'),
    'scipy.optimize.newton': ('from scipy.optimize import newton\n'
                              'def seeded(w, a, sigma, slope):\n'
                              '    return newton(lambda x: x - a - sigma(x), w,\n'
                              '                  fprime=lambda x: 1.0 - slope(x))\n'),
    'scipy.optimize by attribute': ('import scipy\n'
                                    'def seeded(w, a, sigma):\n'
                                    '    return scipy.optimize.brentq(\n'
                                    '        lambda x: x - a - sigma(x), w - 1.0, w + 1.0)\n'),
    'scipy.optimize by a name': ('import scipy\n'
                                 'def seeded(w, a, sigma):\n'
                                 "    return getattr(scipy, 'optimize').brentq(\n"
                                 '        lambda x: x - a - sigma(x), w - 1.0, w + 1.0)\n'),
    'scipy.optimize through importlib': ('import importlib\n'
                                         'def seeded(w, a, sigma, name):\n'
                                         '    brentq = importlib.import_module(name).brentq\n'
                                         '    return brentq(lambda x: x - sigma(x), w, w + 1)\n'),
    'scipy.optimize through __import__': ('def seeded(w, a, sigma, name):\n'
                                          '    brentq = __import__(name, None, None, [0]).brentq\n'
                                          '    return brentq(lambda x: x - sigma(x), w, w + 1)\n'),
    "the house finder's own Newton": ('from src.Solvers.qp_equation import _bracketed_newton\n'
                                      'def seeded(f, l, r):\n'
                                      '    return _bracketed_newton(f, l, r, -1.0, 1e-12, 100)\n'),
    "the house finder's module, relative": ('from ...Solvers import qp_equation\n'
                                            'def seeded(source, a):\n'
                                            '    return qp_equation.find_qp_root(source, a)\n'),
    "another house module's search": ('from src.SingleReference.Periodic.pbc_occupations import '
                                      'fermi_level\n'
                                      'def seeded(*args):\n'
                                      '    return fermi_level(*args)\n'),
    'a loop at module level': ('w, a, f = 0.0, 1.0, 1.0\n'
                               'while abs(f) > 1e-8:\n'
                               '    f = w - a - 0.1 * w\n'
                               '    w = w - f / 0.9\n'),
    'beside a listed loop': ('def _ev_loops(w, a, sigma, slope, n):\n'
                             '    while n > 0:\n'
                             '        n -= 1\n'
                             '        f = 1.0\n'
                             '        while abs(f) > 1e-8:\n'
                             '            f = w - a - sigma(w)\n'
                             '            w = w - f / slope(w)\n'
                             '    return w\n'),
}
#: Newtons in shapes the scan does not see (the test's docstring names them).
NOT_SEEN = {
    'a fixed number of steps with no test': ('def seeded(w, a, sigma, slope):\n'
                                             '    for _ in range(3):\n'
                                             '        f = w - a - sigma(w)\n'
                                             '        w = w - f / slope(w)\n'
                                             '    return w\n'),
    'fixed steps, the step decided by a helper': ('def seeded(w, a, sigma, slope, small):\n'
                                                  '    for _ in range(64):\n'
                                                  '        f = w - a - sigma(w)\n'
                                                  '        if not small(f):\n'
                                                  '            w = w - f / slope(w)\n'
                                                  '    return w\n'),
    'an iterator and a comprehension': (
        'import itertools\n'
        'def seeded(w, a, sigma, slope, small):\n'
        '    steps = itertools.accumulate(itertools.repeat(None, 64),\n'
        '                                 lambda x, _: x - (x - a - sigma(x)) / slope(x),\n'
        '                                 initial=w)\n'
        '    return next(x for x in steps if small(x - a - sigma(x)))\n'),
    "a call's raise, caught by the loop's caller": (
        'def step(w, a, sigma, slope, small):\n'
        '    f = w - a - sigma(w)\n'
        '    if small(f):\n'
        '        raise StopIteration(w)\n'
        '    return w - f / slope(w)\n'
        'def walk(w, a, sigma, slope, small):\n'
        '    for _ in range(64):\n'
        '        w = step(w, a, sigma, slope, small)\n'
        '    return w\n'
        'def seeded(w, a, sigma, slope, small):\n'
        '    try:\n'
        '        return walk(w, a, sigma, slope, small)\n'
        '    except StopIteration as stop:\n'
        '        return stop.value\n'),
    'a method looked up by its name': ('class Solver:\n'
                                       '    def seeded(self, w, a, small, n=0):\n'
                                       '        f = w - a - self.sigma(w)\n'
                                       '        if small(f) or n > 64:\n'
                                       '            return w\n'
                                       "        again = getattr(self, 'seeded')\n"
                                       '        w = w - f / self.slope(w)\n'
                                       '        return again(w, a, small, n + 1)\n'),
}


def test_the_scan_finds_no_root_search_in_the_package():
    """The house's find_qp_root is the package's one root search: a scan of
    every module of the package (`unlisted`) finds no shape of a search of
    its own beyond what is listed. A static scan cannot prove that there is
    none; it shows that none of the shapes it knows is there.

    It knows: a while loop; a for loop that a break of its own, or a
    return, a raise or an assert in it, can end early, that lies in a try
    with an except clause of its own function, or that tests a magnitude,
    as a fixed number of steps does that masks or skips its step (a
    comparison holding abs, fabs, absolute or norm, a name with tol, conv
    or thresh in it or a nonzero number of size 1e-3 or less; a call of
    isclose or allclose): each where LISTED_LOOPS does not list it, by count
    per function (the parquet iterations, one lookup, the input checks); a
    function on a cycle of references within its module (calling itself, or
    through a nested function, a lambda bound to a name, an alias, a
    callback or another function; a method through self, cls or its class)
    that LISTED_CYCLES does not list; scipy.optimize by import, attribute or
    string; importlib, __import__, a string naming a module of the house;
    an import from the house outside the package that HOUSE_IMPORTS does not
    list. Each search of PLANTED is found, and so is the tests' Newton
    (`tests.parquet_quack_newton.quack_newton`).

    It does not see, as NOT_SEEN shows: a fixed number of steps with no
    test of a magnitude (none, or a helper's); an iteration an iterator or a
    comprehension drives; a loop left by an exception raised in a function
    it calls and caught by its own function's caller; a reference made at
    run time (getattr). Nor a cycle across modules, a listed loop whose own
    body became a search, or a root found another way (an eigenvalue
    problem, numpy.roots). These lists are not complete: searches through
    an alias at module or class level, a dispatch table, a self(...) or
    type(self) call, a loop inside contextlib.suppress, or a magnitude test
    with no abs or tol, conv or thresh name pass it too."""
    for name, source in PLANTED.items():
        listed = LISTED_LOOPS['restricted.py'] if name == 'beside a listed loop' else None
        assert unlisted(source, listed), name
    for name, source in NOT_SEEN.items():
        assert unlisted(source) == [], name
    here = pathlib.Path(__file__).resolve().parent
    assert unlisted((here / 'parquet_quack_newton.py').read_text()) == [
        'quack_newton: 1 while, 1 for']
    package = here.parent / 'src' / 'SingleReference' / 'Parquet'
    modules = {p.name: p.read_text() for p in sorted(package.glob('*.py'))}
    assert len(modules) > 10 and set(LISTED_LOOPS) | set(LISTED_CYCLES) <= set(modules)
    found = {name: unlisted(source, LISTED_LOOPS.get(name), LISTED_CYCLES.get(name, ()))
             for name, source in modules.items()}
    assert {name: f for name, f in found.items() if f} == {}, found


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
