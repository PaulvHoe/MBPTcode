"""The parquet quasiparticle equation: the long runs of `test_parquet_qp_solver`.

* The restricted evPA (water and ammonia 6-31G, TDA, s_2b = 1, s_1b = 50) to
  convergence with 'find_qp_root' and with Quack's Newton (`tests.
  parquet_quack_newton.quack_newton`, a qp_solver function): the same
  iteration count, both converged, every orbital within EVPA_BRANCH_EV.
* The spin-orbital evPA (water, RHF pairs) at s_1b = 50, damped as the
  restricted driver damps: at each of three iterations the finder's
  energies are Newton's.
* The spin-orbital osPA with two-body DIIS 5 at s_1b = 50 (water, ammonia):
  Quack's Newton and 'find_qp_root' flag 'z_outside' on the same spin orbitals
  (water's 2b1-like virtual pair, ammonia's three pairs with Z = 1.04).
* The finder handed a root of Z outside (0, 1] to follow cannot follow it
  (it follows valid weights only): ammonia's spin orbitals 22-27 come back
  unfollowed and flipped, at the same root and Z; the drivers
  therefore do not hand such roots on (`options.roots_to_follow`).

Run: python tests/test_parquet_qp_solver_slow.py, or under pytest.
"""
import functools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.numerics.constants import HARTREE_TO_EV, PARQUET_VERTEX_DAMPING
from src.SingleReference.Parquet import restricted, spin_orbital
from tests.parquet_quack_newton import quack_newton
from tests.pins import assert_same
from tests.test_parquet_qp_solver import (EVPA, EVPA_BRANCH_EV, EVERY_ORBITAL_EV,
                                          flagged_orbitals, quiet)
from tests.test_parquet_restricted_vs_quack import mo_integrals, rhf

#: Ha. The evPA one-body threshold of the convergence comparison.
CONV_1B = 1e-4
#: One-body iterations the restricted evPA takes to CONV_1B on either route,
#: as measured.
EVPA_ITERATIONS = {'water': 11, 'ammonia': 12}
#: One-body iterations of the spin-orbital evPA comparison.
SO_EV_ITERATIONS = 3
#: The spin orbitals whose osPA root (s_1b = 50, two-body DIIS 5) has Z
#: outside (0, 1] on both routes.
Z_OUTSIDE = {'water': [22, 23], 'ammonia': [22, 23, 24, 25, 26, 27]}
#: osPA as those flags were measured: TDA, s_2b = 1, two-body DIIS 5.
OSPA_D5 = dict(tda_eh=True, tda_pp=True, eta_1b=50.0, eta_2b=1.0, max_diis_2b=5,
               conv_2b=1e-10)
#: The two routes compared: Quack's Newton, handed in as a qp_solver
#: function, and the house's root finder.
ROUTES = {'newton': quack_newton, 'find_qp_root': 'find_qp_root'}


@pytest.mark.parametrize('molecule', sorted(EVPA_ITERATIONS))
def test_the_restricted_evpa_converges_on_newtons_branch(molecule):
    eps, eri, nO = mo_integrals(molecule)
    runs = {qp: quiet(restricted.ev_parquet, eps, eri, nO, 'ev', qp_solver=ROUTES[qp],
                      **dict(EVPA, conv_1b=CONV_1B, max_it_1b=50))
            for qp in ROUTES}
    for qp, result in runs.items():
        assert result.converged_1b and result.n_it_1b == EVPA_ITERATIONS[molecule], (
            molecule, qp, result.n_it_1b)
    gap = np.abs(runs['newton'].eqp - runs['find_qp_root'].eqp).max() * HARTREE_TO_EV
    assert gap <= EVPA_BRANCH_EV, (molecule, gap)


def test_the_spin_orbital_evpa_stays_on_newtons_branch():
    mf = rhf('water')
    runs = {qp: quiet(spin_orbital.run_parquet_so, mf, 'ev', qp_solver=ROUTES[qp],
                      damping=PARQUET_VERTEX_DAMPING, **dict(EVPA, max_it_1b=SO_EV_ITERATIONS))
            for qp in ROUTES}
    newton, scan = runs['newton'], runs['find_qp_root']
    assert newton.n_it_1b == scan.n_it_1b == SO_EV_ITERATIONS
    for k, (hn, hs) in enumerate(zip(newton.history, scan.history)):
        gap = np.abs(hn['eqp'] - hs['eqp']).max() * HARTREE_TO_EV
        assert gap <= EVERY_ORBITAL_EV, (k, gap)
        assert np.array_equal(hs['eqp'][0::2], hs['eqp'][1::2]), k


@functools.lru_cache(maxsize=None)
def ospa(molecule, qp_solver):
    return quiet(spin_orbital.run_parquet_so, rhf(molecule), 'os', qp_solver=qp_solver,
                 **OSPA_D5)


@pytest.mark.parametrize('molecule', sorted(Z_OUTSIDE))
def test_both_routes_flag_the_same_roots(molecule):
    for qp in ROUTES:
        result = ospa(molecule, ROUTES[qp])
        assert flagged_orbitals(result.numerical_flags) == Z_OUTSIDE[molecule], (
            molecule, qp, result.numerical_flags)
        z = result.z[Z_OUTSIDE[molecule]]
        assert np.all((z <= 0.0) | (z > 1.0)), (molecule, qp, z)


def test_a_root_outside_the_weights_is_not_followed():
    """Ammonia's osPA spectra: the finder handed its own roots as the ones to
    follow returns every root and Z (`tests.pins.assert_same`), the spin
    orbitals of Z outside (0, 1] unfollowed and flipped (the search without a
    follow gave them)."""
    result = ospa('ammonia', 'find_qp_root')
    ref = spin_orbital.spin_orbital_reference(rhf('ammonia'))
    eri = spin_orbital.spin_orbital_eri(ref.eri_ao, ref.mo_coeff)
    pairs = np.array(result.options['spin_pairs'])
    out = {}
    eqp, z, _ = quiet(spin_orbital.solve_qp_scan_so, 50.0, ref.eps, ref.eps, eri, ref.nO,
                      result.spectra, follow=result.eqp, spin_pairs=pairs, out=out)
    assert_same(eqp, result.eqp, label='followed eqp')
    assert_same(z, result.z, unit='1', label='followed Z')
    outside = Z_OUTSIDE['ammonia']
    assert not out['followed'][outside].any() and out['flipped'][outside].all(), out
    inside = np.setdiff1d(np.arange(len(eqp)), outside)
    assert out['followed'][inside].all(), out['followed']


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
