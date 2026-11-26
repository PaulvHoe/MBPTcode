"""TEST_PLAN's Tier 1 beyond zero dimensions: the SG shortcut against the BSE with the kernel along the map.

At cycle 2 the I/L route closes on K^(2) L^(2), with L^(2) = L_0 + L_0 K^(2) L^(2) and the kernel along the
map K^(2) = delta Sigma^(1) / delta G^(1) = (delta Sigma^(1) / delta G^(0)) J^-1. J maps a direction of the
start line to the cycle-1 line it moves: J[Y] = Y + delta(Sigma^(1) - Sigma^(0))[Y], the Dyson rule of the
tangents. SG-I/L replaces the solved BSE by the shortcut K^(2) L^(2) = (delta Sigma^(1) / delta G^(0)) L^(1).

1. The dimer (asymmetric, U = t, both starts), the I/L route: the BSE Gamma = 1 + K^(2) G^(1) Gamma G^(1) solved by
   GMRES, every application of K^(2) with its own GMRES for J^-1, and K^(2) Gamma against the shortcut
   (measured 7e-14 and 8e-14). At cycle 2 this reduces exactly, with Z = J^-1 Gamma, to the start's BSE
   Z = 1 + (delta Sigma^(0) / delta G^(0))[Z]: J and delta Sigma^(1) / delta G^(0) cancel, so the nested
   solve checks the start's L^(1) against the kernel as the tangents apply it, and the GMRES machinery.
2. The dimer, the Gamma/W route: Gamma^(2) of SG-Gamma/W without its lag (lag=False) against the GMRES solution
   of Gamma = 1 + (delta Sigma_xc^(1) / delta G^(1)) G^(1) Gamma G^(1) with the same nested J^-1, Sigma_xc^(1) =
   Sigma^(1) - Sigma_H[G^(1)] (measured 1.3e-13 and 1.2e-13). Here J, the Hartree response Lambda_H (through
   the Dyson rule) and the factor (1 + D)^-1 of the SG vertex do not cancel: the check holds the SG vertex
   formula itself.
3. H2/STO-3G (both starts), where the nested solve takes minutes: the reduced form of 1, the residual of the
   start's BSE at the L^(1) of `solve_l` (the pair-space solve with the static kernel), measured 4e-16 and
   5e-16.

The grids are small (the identities are algebraic): beta = 5, n_fermion = m_boson = 16, n_vertex = 4 and
m_vertex = 2.

Run: python tests/test_hedin_cycles_explicit_bse.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from test_hedin_cycles_matsubara import check, h2
from src.SingleReference.HedinCycles.matsubara.backend import MatsubaraBackend, ThreePoint
from src.SingleReference.HedinCycles.matsubara.grids import Grids
from src.SingleReference.HedinCycles.matsubara.systems import hubbard_dimer
from src.SingleReference.HedinCycles.matsubara.tangents import Seed, Tangents
from src.SingleReference.HedinCycles.schemes import run_cycles

GRIDS = Grids(beta=5.0, n_fermion=16, n_vertex=4, m_boson=16, m_vertex=2)
#: the explicit BSEs against the shortcut and the SG vertex, and the start's BSE residual, relative (measured
#: 4e-16 to 1.3e-13)
EXPLICIT_TOL = 1e-11


class Directions:
    """Directions along the map as flat vectors: a static part on the boson grid and a part on the vertex
    box, one per Coulomb factor."""

    def __init__(self, alg):
        self.alg, ctx, g = alg, alg.ctx, alg.grids
        self.s_shape = (len(g.boson_m), ctx.naux, ctx.N, ctx.N)
        self.d_shape = (len(g.vertex_m), ctx.naux, 2 * g.n_vertex, ctx.N, ctx.N)
        self.ns, self.nd = int(np.prod(self.s_shape)), int(np.prod(self.d_shape))
        self.size = self.ns + self.nd

    def three_point(self, v):
        return ThreePoint(self.alg.ctx, False, v[:self.ns].reshape(self.s_shape), v[self.ns:].reshape(self.d_shape))

    def flat(self, t):
        out = np.zeros(self.size, dtype=complex)
        if t is not None and t.static is not None:
            out[:self.ns] = t.static.ravel()
        if t is not None and t.dyn is not None:
            out[self.ns:] = t.dyn.ravel()
        return out

    def bare(self):
        out = np.zeros(self.size, dtype=complex)
        out[:self.ns] = np.broadcast_to(self.alg.ctx.B, self.s_shape).ravel()
        return out

    def tangent(self, obj, v):
        """(delta obj / delta G^(0)) in the direction v."""
        return self.flat(Tangents(self.alg, Seed(self.alg.line0, self.three_point(v)))(obj))


def explicit_bse_error(system, start):
    """K^(2) Gamma from the BSE with the kernel along the map, solved by nested GMRES, against the shortcut."""
    alg = MatsubaraBackend(system, start, GRIDS)
    hist = run_cycles(alg, 'sg_il', start, 1)
    S0, S1 = hist.sigmas[0], hist.sigmas[1]
    d = Directions(alg)
    J = LinearOperator((d.size, d.size), dtype=complex, matvec=lambda v: v + d.tangent(S1 - S0, v))

    def kernel(v):
        y0, info = gmres(J, v, rtol=1e-13, atol=0.0, restart=60, maxiter=20)
        if info != 0:
            raise RuntimeError('the GMRES of J did not converge')
        return d.tangent(S1, y0)

    B = d.bare()
    A = LinearOperator((d.size, d.size), dtype=complex, matvec=lambda x: x - kernel(x))
    x, info = gmres(A, kernel(B), rtol=1e-12, atol=0.0, restart=60, maxiter=20)
    explicit = kernel(B + x)
    shortcut = d.flat(alg.along_map(S1, hist.l1))
    return np.abs(explicit - shortcut).max() / np.abs(shortcut).max(), info == 0


def explicit_gw_vertex_error(system, start):
    """Gamma^(2) of SG-Gamma/W without its lag (lag=False, the Hartree term of G^(1) itself) against the GMRES
    solution of Gamma = 1 + (delta Sigma_xc^(1) / delta G^(1)) G^(1) Gamma G^(1), Sigma_xc^(1) = Sigma^(1) -
    Sigma_H[G^(1)], the kernel along the map applied as (delta Sigma_xc^(1) / delta G^(0)) J^-1, with
    Sigma_H[G^(1)] differentiated through the Dyson rule. Here J, Lambda_H and the factor (1 + D)^-1 of
    sg_vertex do not cancel, unlike the I/L closure at cycle 2."""
    alg = MatsubaraBackend(system, start, GRIDS)
    hist = run_cycles(alg, 'sg_gw', start, 2, lag=False)
    S0, S1, G1 = hist.sigmas[0], hist.sigmas[1], hist.lines[1]
    Sxc = S1 - alg.sigma_h(G1)
    d = Directions(alg)
    J = LinearOperator((d.size, d.size), dtype=complex, matvec=lambda v: v + d.tangent(S1 - S0, v))

    def kernel(v):
        y0, info = gmres(J, v, rtol=1e-13, atol=0.0, restart=60, maxiter=20)
        if info != 0:
            raise RuntimeError('the GMRES of J did not converge')
        return d.tangent(Sxc, y0)

    B = d.bare()
    A = LinearOperator((d.size, d.size), dtype=complex, matvec=lambda x: x - kernel(x))
    x, info = gmres(A, kernel(B), rtol=1e-12, atol=0.0, restart=60, maxiter=20)
    v = hist.vertices[2]
    g = alg.grids
    sg = d.flat(ThreePoint(alg.ctx, False, np.stack([v.static_at(m) for m in g.boson_m]), v.dyn))
    return np.abs(B + x - sg).max() / np.abs(sg - B).max(), info == 0


def start_bse_residual(system, start):
    """|Z - 1 - (delta Sigma^(0) / delta G^(0))[Z]| at Z = the start's L^(1) vertex, relative to |Z - 1|."""
    alg = MatsubaraBackend(system, start, GRIDS)
    hist = run_cycles(alg, 'sg_il', start, 1)
    d = Directions(alg)
    v = hist.l1.vertex
    g = alg.grids
    Z = d.flat(ThreePoint(alg.ctx, False, np.stack([v.static_at(m) for m in g.boson_m]), v.dyn))
    B = d.bare()
    return np.abs(Z - B - d.tangent(hist.sigmas[0], Z)).max() / np.abs(Z - B).max()


def check_dimer():
    ok = True
    for start in ('H', 'HF'):
        err, converged = explicit_bse_error(hubbard_dimer(1.0, 1.0, eps=(0.0, 0.4)), start)
        ok &= check(err < EXPLICIT_TOL and converged,
                    f'dimer ({start}): the explicitly solved BSE with the kernel along the map gives the SG shortcut',
                    f'{err:.1e} relative')
    return ok


def check_dimer_gamma_w():
    ok = True
    for start in ('H', 'HF'):
        err, converged = explicit_gw_vertex_error(hubbard_dimer(1.0, 1.0, eps=(0.0, 0.4)), start)
        ok &= check(err < EXPLICIT_TOL and converged,
                    f'dimer ({start}): Gamma^(2) of SG-Gamma/W without its lag solves the vertex BSE with the '
                    'kernel along the map', f'{err:.1e} relative')
    return ok


def check_h2():
    ok = True
    mol = h2()
    for start in ('H', 'HF'):
        res = start_bse_residual(mol, start)
        ok &= check(res < EXPLICIT_TOL, f'H2/STO-3G ({start}): the start\'s L^(1) solves its BSE as the tangents '
                    'apply the kernel, so the explicit BSE at cycle 2 gives the shortcut', f'{res:.1e} relative')
    return ok


def run():
    warnings.simplefilter('ignore')
    print('-- 1. the dimer: the BSE with the kernel along the map, solved')
    ok = check_dimer()
    print('\n-- 2. the dimer: the vertex BSE of SG-Gamma/W without its lag, solved')
    ok &= check_dimer_gamma_w()
    print('\n-- 3. H2/STO-3G: the start\'s BSE in the tangents\' form')
    ok &= check_h2()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_hedin_cycles_explicit_bse():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
