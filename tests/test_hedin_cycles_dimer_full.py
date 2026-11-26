"""The cycle-based schemes on the half-filled Hubbard dimer (t = 1): TEST_PLAN's Tier 4 exactness of cycle 2.

As tests/test_hedin_cycles_dimer.py, for cycle-based Gamma/W with the kernel at fixed form K[G^(1), W[G^(1)]]
(the strict kernel, equal to cycle-based I/L by tests/test_hedin_cycles_matsubara.py 7). The error of
[G^(2)]^-1 in the bonding orbital against exact diagonalization scales as U^3 from (H) and as U^4 from (HF).
These need the larger grid of the SG checks (beta = 20, 64 fermionic and 128 bosonic frequencies, a vertex box
of 16 by 65): on a quarter of it the boson grid's U^2 remainder (7e-8 at U = 0.05) bends the (HF) slope to
3.68.

Measured on one thread, each point run alone (35 s from (H), 70 s from (HF)):

    (H)   2.624e-5 at U = 0.1, 3.782e-6 at 0.05: slope 2.79, 0.026 U^3
    (HF)  2.113e-6 at U = 0.1, 1.398e-7 at 0.05: slope 3.92, 0.021 U^4

The dimer report's cycle-based scheme uses the kernel K[G^(1), W[h]] instead, the cycle-1 formula
differentiated with the start line's W in its W lines. Built here from the algebra (a level whose W carries
W[h]'s values and the recipe screen(P[G]), so that its tangent is W[h] delta P[G] W[h], not offered in the
API), it gives 3.785e-6 at U = 0.05 from (H), 0.0303 U^3 against the report's large box 0.030 U^3, and differs
from the strict kernel only from U^4 on (8e-8 at U = 0.1, 3e-9 at 0.05). The report's small-box 2.28e-5 at
U = 0.1 is 13 % below the converged 2.63e-5, so it is not compared.

Run: python tests/test_hedin_cycles_dimer_full.py, or under pytest (about 4 minutes on one thread, so on
ADA).
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np

from test_hedin_cycles_dimer import GRIDS, check, exact_inverse
from src.Base.numerical_flags import collecting
from src.SingleReference.HedinCycles.matsubara.backend import Boson, MatsubaraBackend
from src.SingleReference.HedinCycles.matsubara.systems import hubbard_dimer
from src.SingleReference.HedinCycles.schemes import gw_level, run_cycles, start_level

#: the slope may miss the first wrong order by this much: measured 2.79 from (H), whose U^4 term bends it as
#: it bends SG-Gamma/W's 2.81 in the fast file, and 3.92 from (HF), deviations of 0.21 and 0.08
SLOPE_TOL = 0.3

#: the report's cycle-based coefficient at U = 0.05 with its large box (M2 = 256, MK = 512, NBG = 128), and the
#: relative agreement asked (measured 0.9 %)
REPORT_WH = 0.030
REPORT_WH_TOL = 0.03


def cycle_two_error(start, U):
    alg = MatsubaraBackend(hubbard_dimer(1.0, U), start, GRIDS)
    nn = np.arange(4)
    exact = exact_inverse(alg, 1j * GRIDS.nu(nn) + alg.start.mu)
    hist = run_cycles(alg, 'cb_gw', start, 2)
    line = hist.lines[2]
    ours = [np.linalg.inv(G) for G in line.at(nn)]
    return (max(abs(a[0, 0] - b[0, 0]) for a, b in zip(ours, exact)), [f['flag'] for f in hist.numerical_flags])


def wh_level(alg):
    """The cycle-1 level of the (H) start with the start line's W in its W lines: S(G) = -G W' Gamma, W' with
    the values of W[h] and the recipe screen(P[G]), so that its tangent is W[h] delta P[G] W[h]."""
    S0 = start_level(alg, 'H')
    G0 = alg.line0
    Wh = alg.screen(alg.polarizability(G0, alg.solve_vertex(G0, alg.kernel(S0, G0))))

    def S(G):
        vertex = alg.solve_vertex(G, alg.kernel(S0, G))
        P = alg.polarizability(G, vertex)
        return alg.sigma_gw(G, Boson(alg.ctx, 'M', Wh.data, ('screen', P)), vertex)
    return S


def check_wh_variant():
    U = 0.05
    alg = MatsubaraBackend(hubbard_dimer(1.0, U), 'H', GRIDS)
    G1 = run_cycles(alg, 'sg_il', 'H', 1).lines[1]
    flags = []
    with collecting(flags):
        G2 = alg.dyson(alg.sigma_h(G1) + gw_level(alg, wh_level(alg))(G1) - alg.sigma_h(alg.line0))
    nn = np.arange(4)
    exact = exact_inverse(alg, 1j * GRIDS.nu(nn) + alg.start.mu)
    err = max(abs(np.linalg.inv(a)[0, 0] - b[0, 0]) for a, b in zip(G2.at(nn), exact))
    ours = err / U ** 3
    return check(abs(ours / REPORT_WH - 1) < REPORT_WH_TOL and not flags,
                 '(H) cycle-based Gamma/W with the kernel K[G^(1), W[h]] agrees with the dimer report',
                 f'{err:.3e} at U = 0.05, {ours:.4f} U^3 against {REPORT_WH} U^3, flags {[f["flag"] for f in flags]}')


def check_cycle_based_exactness():
    ok = True
    for start, power in (('H', 3), ('HF', 4)):
        (e1, f1), (e2, f2) = cycle_two_error(start, 0.1), cycle_two_error(start, 0.05)
        slope = np.log2(e1 / e2)
        ok &= check(abs(slope - power) < SLOPE_TOL and not f1 and not f2,
                    f'({start}) cycle-based Gamma/W cycle 2: the error scales as U^{power}, GMRES converged',
                    f'{e1:.3e} at U = 0.1, {e2:.3e} at 0.05, slope {slope:.2f}, flags {f1 + f2}')
    return ok


def run():
    warnings.simplefilter('ignore')
    print('-- exactness of the cycle-based schemes (beta = 20)')
    ok = check_cycle_based_exactness()
    print('\n-- the dimer report\'s kernel K[G^(1), W[h]]')
    ok &= check_wh_variant()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_hedin_cycles_dimer_full():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
