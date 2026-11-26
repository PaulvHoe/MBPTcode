"""TEST_PLAN's Tier 1 of the Hedin cycles on H2/STO-3G (milestone 2).

The checks of tests/test_hedin_cycles_matsubara.py on a molecule (4 spin orbitals, 3 Coulomb factors, beta =
20 / Ha), whose factors are not diagonal in one basis, so that the tails of the frequency sums do not cancel
as on the dimer. Measured on one thread:

1. The derivative along the map of Sigma^(1), both routes and both starts: central finite differences at
   Omega = 0 (4e-10 to 3e-9) and gauge covariance at every transfer (2e-16). The tangent carries the tails
   of its own boson sums, so it is the derivative of the discrete self-energy, tails included.
2. SG cycle 2: SG-I/L minus SG-Gamma/W equals -G^(1) (W^(1) - W^(2)) Gamma^(2) (3.7e-6 from (H), 2.8e-5
   from (HF)), SG-Gamma/W without its lag equals SG-I/L with W(4,1) (1.5e-7 from (H), 2.9e-6 from (HF)), and
   Gamma^(2) does not depend on w_order (1e-16) while the printed W(1,4) breaks the lag identity (1.0e-4
   and 2.0e-3).
3. The kernels at fixed form on G^(1) against finite differences (2e-9 to 3.3e-8).
4. Cycle 2 of the cycle-based schemes: Gamma/W equals I/L (1.5e-6 from (H), 1.9e-5 from (HF)), apart from
   SG-I/L (0.16 and 0.08), GMRES converged.

The transfer box must stay well inside the boson grid for the routes to agree: on the larger transfer box
of the dimer's checks (8 transfers each side with a boson grid of 32) the SG difference is 1.2e-3 from
(HF), falling as the cube of the boson grid.

Run: python tests/test_hedin_cycles_h2.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from test_hedin_cycles_matsubara import (check, cycle_based_routes, fixed_form_errors, h2, lag_identity,
                                         sg_difference_error, tangent_errors)
from src.SingleReference.HedinCycles.matsubara.grids import Grids

#: the grids of the tangents, of the SG checks and of the kernel and cycle-based checks
TANGENT_GRIDS = Grids(beta=20.0, n_fermion=64, n_vertex=8, m_boson=64, m_vertex=3)
SG_GRIDS = Grids(beta=20.0, n_fermion=32, n_vertex=16, m_boson=64, m_vertex=4)
KERNEL_GRIDS = Grids(beta=20.0, n_fermion=32, n_vertex=8, m_boson=32, m_vertex=4)
#: the SG cycle-2 difference against -G^(1) (W^(1) - W^(2)) Gamma^(2), relative: measured 3.7e-6 from (H)
#: and 2.8e-5 from (HF), and the bar is three times the largest
SG_DIFFERENCE_TOL = 1e-4
#: SG-Gamma/W without its lag against SG-I/L, relative: measured 1.5e-7 from (H) and 2.9e-6 from (HF), and the
#: bar is three times the largest
LAG_TOL = 1e-5
#: Kernel.apply against central differences (step 1e-5), relative: measured 2e-9 to 1.1e-8 (static) and
#: 2.8e-9 to 3.3e-8 (vertex box), and the bars are three times the largest
FIXED_FORM_FD_TOL = {'static': 4e-8, 'box': 1e-7}
#: cycle-based Gamma/W against I/L at cycle 2, relative: measured 1.5e-6 from (H) and 1.9e-5 from (HF) with
#: m_boson = 32 (2.1e-7 and 2.5e-6 at 64), and the bar is three times the largest
CYCLE_BASED_ROUTE_TOL = 6e-5


def check_tangents(mol):
    ok = True
    for start in ('H', 'HF'):
        for route, (err, gauge) in tangent_errors(mol, start, TANGENT_GRIDS).items():
            ok &= check(err < 1e-8 and gauge < 1e-12,
                        f'({start}) {route} route: the tangent of Sigma^(1) against finite differences and gauge covariance',
                        f'FD {err:.1e} relative, gauge {gauge:.1e}')
    return ok


def check_sg_cycle_two(mol):
    ok = True
    for start in ('H', 'HF'):
        err, size = sg_difference_error(mol, start, SG_GRIDS)
        ok &= check(err < SG_DIFFERENCE_TOL,
                    f'({start}) cycle 2: SG-I/L - SG-Gamma/W = -G^(1) (W^(1) - W^(2)) Gamma^(2)',
                    f'{err:.1e} relative on a difference of {size:.1e}')
    for start in ('H', 'HF'):
        lag, printed, same_vertex, asym = lag_identity(mol, SG_GRIDS, start)
        ok &= check(lag < LAG_TOL, f'({start}) SG-Gamma/W without its lag = SG-I/L', f'{lag:.1e} relative')
        ok &= check(same_vertex < 1e-13, f'({start}) Gamma^(2) is the same with w_order = 1,4', f'{same_vertex:.1e}')
        ok &= check(printed > 10 * lag,
                    f'({start}) P^(2) is not symmetric, and the printed W(1,4) breaks the lag identity',
                    f'asymmetry of M {asym:.1e}, difference {printed:.1e} relative')
    return ok


def check_kernels_at_fixed_form(mol):
    ok = True
    for start in ('H', 'HF'):
        for (route, kind), err in fixed_form_errors(mol, start, KERNEL_GRIDS).items():
            ok &= check(err < FIXED_FORM_FD_TOL[kind],
                        f'({start}) {route} route: the kernel at fixed form on G^(1), {kind} direction, '
                        f'against finite differences', f'{err:.1e} relative')
    return ok


def check_cycle_based_cycle_two(mol):
    ok = True
    for start in ('H', 'HF'):
        r = cycle_based_routes(mol, start, KERNEL_GRIDS, orders=False)
        ok &= check(r['route'] < CYCLE_BASED_ROUTE_TOL and r['apart'] > 1e-3 and r['converged'],
                    f'({start}) cycle 2: cycle-based Gamma/W = cycle-based I/L, apart from SG-I/L',
                    f"{r['route']:.1e} relative, {r['apart']:.1e} from SG-I/L, GMRES converged: {r['converged']}")
    return ok


def run():
    warnings.simplefilter('ignore')
    mol = h2()
    print('-- 1. the derivative along the map')
    ok = check_tangents(mol)
    print('\n-- 2. cycle 2 of the SG schemes')
    ok &= check_sg_cycle_two(mol)
    print('\n-- 3. kernels at fixed form')
    ok &= check_kernels_at_fixed_form(mol)
    print('\n-- 4. cycle 2 of the cycle-based schemes')
    ok &= check_cycle_based_cycle_two(mol)
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_hedin_cycles_h2():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
