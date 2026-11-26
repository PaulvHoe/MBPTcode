"""The Hedin cycles on one-electron systems from (HF) (milestone 3, slow): TEST_PLAN's Tier 5, one electron.

With a single electron in the occupied spin orbital o of the HF start, the exact ground state is that orbital.
So o can take no second electron, the exact G has no element between o and any other spin orbital, and its oo
element is the start's single pole: the occupied row and column of [G_exact]^-1 are those of [G^(0)]^-1
(asserted against `ExactSelfEnergy`), and there the exact Sigma_xc cancels Sigma_H. The rest of G is not the
start's, since an added electron makes a correlated two-electron state (H^-, H2). With e^(n) the largest
element of the occupied row and column of [G^(n)]^-1 - [G^(0)]^-1 over the four lowest Matsubara frequencies,
any nonzero e^(n) is the self-screening error of cycle n.

1. Cycle 1 (GW with the TDHF vertex and screening, the same in every scheme) has none. For one electron the
   TDHF coupling block <oo||ab> vanishes, so TDHF is exact. e^(1) is the cut of the boson grid alone (it does not
   change with the vertex box) and falls as m_boson^-3, and its Aitken limit over m_boson = 128, 192 and 288
   (e_inf = e_3 - d_2^2 / (d_1 - d_2), with d_1 = e_1 - e_2 and d_2 = e_2 - e_3) must be below 1 % of e^(1) at
   128.
2. Cycle 2 has some, reported and not asserted, on the boxes M = 16, 24 and 36 (n_vertex = m_vertex = M,
   m_boson = 8 M). SG-I/L converges smoothly. SG-Gamma/W, through its one-cycle Hartree lag, moves with the box
   (H2+: the shift of the occupied level at nu_0 is +4.0e-6, -2.1e-4, -1.4e-4 and -5.3e-5 for M = 16, 24, 36 and
   54, while SG-Gamma/W without its lag gives +1.04e-5, +1.38e-5 and +1.49e-5 for 16 to 36). Its cycle-2
   screening W^(2) is what moves: the identity SG-I/L - SG-Gamma/W = -G^(1) (W^(1) - W^(2)) Gamma^(2) holds on
   every box, and is asserted.

beta = 90 / Ha: the start gap of 0.59 Ha (H) and 0.67 Ha (H2+) keeps exp(-beta gap / 2) below 3e-12. The
gap is the occupied level against the same orbital with the other spin and closes with the interaction, so
these systems are taken at full coupling and not scaled. Measured on one thread (54 s per system):

| | H/6-31G | H2+/STO-3G |
|---|---|---|
| exact against start, occupied row and column (elsewhere) | 1.4e-16 (4.8e-2) | 6.2e-17 (5.1e-2) |
| cycle 1, m_boson = 128, 192, 288 | 1.96e-5, 5.84e-6, 1.74e-6 | 2.62e-6, 7.76e-7, 2.30e-7 |
| cycle 1, exponent and limit | 2.98, -1.6e-8 | 3.00, -3.8e-10 |
| SG-I/L cycle 2, M = 16, 24, 36 (limit) | 3.92e-5, 5.21e-5, 5.66e-5 (5.91e-5) | 1.04e-5, 1.37e-5, 1.48e-5 (1.54e-5) |
| SG-Gamma/W cycle 2, M = 16, 24, 36 | 4.72e-4, 1.27e-4, 5.39e-5 | 8.92e-5, 2.12e-4, 1.39e-4 |
| the identity, relative | 3.6e-3, 3.0e-3, 2.6e-3 | 1.2e-3, 1.1e-3, 7.2e-4 |

Run: python tests/test_hedin_cycles_one_electron.py (about 2 minutes), or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from pyscf import gto, scf

from test_hedin_cycles_dimer import exact_inverse
from test_hedin_cycles_matsubara import check
from src.SingleReference.HedinCycles.matsubara.backend import Boson, MatsubaraBackend
from src.SingleReference.HedinCycles.matsubara.grids import Grids
from src.SingleReference.HedinCycles.matsubara.systems import from_pyscf
from src.SingleReference.HedinCycles.schemes import run_cycles, start_level

BETA = 90.0
#: the boson grids of the cycle-1 extrapolation (vertex box 16), and the vertex boxes of cycle 2
BOSON_GRIDS = (128, 192, 288)
BOXES = (16, 24, 36)
#: the Aitken limit of e^(1) relative to e^(1) at the first grid (measured 8e-4 and 1.5e-4), and the window
#: of its exponent (measured 2.98 and 3.00)
CYCLE_ONE_TOL = 1e-2
CYCLE_ONE_EXPONENT = (2.5, 3.5)
#: SG-I/L - SG-Gamma/W against -G^(1) (W^(1) - W^(2)) Gamma^(2), relative: the routes' cut, measured 7.2e-4 to
#: 3.6e-3 on the three boxes, and the bar is three times the largest
IDENTITY_TOL = 1e-2
NN = np.arange(4)


def one_electron(atom, basis, charge, name):
    return from_pyscf(scf.ROHF(gto.M(atom=atom, basis=basis, charge=charge, spin=1, verbose=0)).run(), name)


#: (name, system)
CASES = (('H/6-31G', lambda: one_electron('H 0 0 0', '6-31g', 0, 'H/6-31G')),
         ('H2+/STO-3G', lambda: one_electron('H 0 0 0; H 0 0 1.06', 'sto-3g', 1, 'H2+/STO-3G')))


def occupied_error(alg, line):
    """The largest element of the occupied rows and columns of [G]^-1 - [G^(0)]^-1 at the four lowest
    frequencies."""
    occ = np.arange(alg.start.nocc)
    d = np.linalg.inv(line.at(NN)) - np.linalg.inv(alg.line0.at(NN))
    return max(np.abs(d[:, occ, :]).max(), np.abs(d[:, :, occ]).max())


def check_exact_structure(name, system):
    alg = MatsubaraBackend(system, 'HF', Grids(beta=BETA, n_fermion=128, n_vertex=16, m_boson=128, m_vertex=16))
    occ = np.arange(alg.start.nocc)
    exact = np.array(exact_inverse(alg, 1j * alg.grids.nu(NN) + alg.start.mu))
    d = exact - np.linalg.inv(alg.line0.at(NN))
    row = max(np.abs(d[:, occ, :]).max(), np.abs(d[:, :, occ]).max())
    rest = np.abs(np.delete(np.delete(d, occ, axis=1), occ, axis=2)).max()
    return check(row < 1e-12 and rest > 1e-3,
                 f'{name}: the exact [G]^-1 equals the start\'s in the occupied row and column, not elsewhere',
                 f'{row:.1e} there, {rest:.1e} in the rest')


def check_cycle_one(name, system):
    e = []
    for mb in BOSON_GRIDS:
        alg = MatsubaraBackend(system, 'HF', Grids(beta=BETA, n_fermion=128, n_vertex=16, m_boson=mb, m_vertex=16))
        e.append(occupied_error(alg, run_cycles(alg, 'sg_il', 'HF', 1).lines[1]))
    d1, d2 = e[0] - e[1], e[1] - e[2]
    limit = e[2] - d2 ** 2 / (d1 - d2)
    exponent = np.log(d1 / d2) / np.log(BOSON_GRIDS[1] / BOSON_GRIDS[0])
    return check(abs(limit) < CYCLE_ONE_TOL * e[0] and CYCLE_ONE_EXPONENT[0] < exponent < CYCLE_ONE_EXPONENT[1],
                 f'{name}: cycle 1 has no self-screening error (the cut of the boson grid falls as its cube)',
                 ', '.join(f'{x:.2e}' for x in e) + f' for m_boson = {BOSON_GRIDS}, exponent {exponent:.2f}, '
                 f'limit {limit:+.1e}')


def cycle_two(system, M):
    """e^(2) of SG-I/L and SG-Gamma/W on the box M, the identity's residual (relative), and the flags."""
    alg = MatsubaraBackend(system, 'HF', Grids(beta=BETA, n_fermion=128, n_vertex=M, m_boson=8 * M, m_vertex=M))
    il, gw = run_cycles(alg, 'sg_il', 'HF', 2), run_cycles(alg, 'sg_gw', 'HF', 2)
    G0 = alg.line0
    W1 = alg.screen(alg.polarizability(G0, alg.solve_vertex(G0, alg.kernel(start_level(alg, 'HF'), G0))))
    eye = np.eye(alg.ctx.naux)
    diff = Boson(alg.ctx, 'M', W1.data - gw.screened[2].data + eye[None], ('test',))
    one = Boson(alg.ctx, 'M', np.broadcast_to(eye, W1.data.shape).copy(), ('test',))
    formula = alg.sigma_gw(gw.lines[1], diff, gw.vertices[2]).dyn - alg.sigma_gw(gw.lines[1], one, gw.vertices[2]).dyn
    delta = il.sigmas[2].at_grid() - gw.sigmas[2].at_grid()
    identity = np.abs(delta - formula).max() / np.abs(delta).max()
    return (occupied_error(alg, il.lines[2]), occupied_error(alg, gw.lines[2]), identity,
            set(il.numerical_flags) | set(gw.numerical_flags))


def check_cycle_two(name, system):
    rows = [cycle_two(system, M) for M in BOXES]
    il, gw, identity = ([r[i] for r in rows] for i in range(3))
    flags = set().union(*(r[3] for r in rows))
    d1, d2 = il[0] - il[1], il[1] - il[2]
    print(f'  [report] {name} cycle 2, self-screening error on the boxes {BOXES}: SG-I/L '
          + ', '.join(f'{x:.2e}' for x in il) + f' (Aitken limit {il[2] - d2 ** 2 / (d1 - d2):.2e}), SG-Gamma/W '
          + ', '.join(f'{x:.2e}' for x in gw))
    ok = check(max(identity) < IDENTITY_TOL,
               f'{name}: SG-I/L - SG-Gamma/W = -G^(1) (W^(1) - W^(2)) Gamma^(2) on every box',
               ', '.join(f'{x:.1e}' for x in identity) + ' relative')
    return check(not flags, f'{name}: no numerical flag in any run', ', '.join(sorted(flags)) or 'none') and ok


def run():
    warnings.simplefilter('ignore')
    ok = True
    for name, system in CASES:
        print(f'-- {name} from (HF), beta = {BETA:g} / Ha')
        mol = system()
        ok &= check_exact_structure(name, mol)
        ok &= check_cycle_one(name, mol)
        ok &= check_cycle_two(name, mol)
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_hedin_cycles_one_electron():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
