"""The Hedin cycles on the Matsubara axis (src/SingleReference/HedinCycles/matsubara, milestone 2).

The backend runs the cycle loop of schemes.py on a finite spin-orbital basis at the inverse
temperature beta, in the orbitals of the start line. The reference systems are the half-filled Hubbard
dimer (t = 1, built from its integrals) and H2/STO-3G.

1. Cycle 1 against the house GW code, which sums the same self-energies over Casida poles
   (`BSE.second_order_qp.cycle_self_energy`): G0W0 on the Hartree levels from (H), and GWGammaInf at the
   TDHF screening, Sigma_TDHF@L_TDHF, on the HF levels from (HF). Sigma_c(i nu_n), n = 0 .. 19, of every
   spatial orbital, on its diagonal only (the house gives diagonal poles, and on these systems the
   off-diagonal elements vanish by symmetry), and its convergence with the grids (the error falls about
   8x when n_fermion and m_boson double).
2. Cycle 1 is the same in all four schemes, to rounding from both starts: the Gamma/W route (Hedin's vertex
   BSE, the polarizability, the screening and -G W Gamma) and the I/L route (the BSE for L and the closure
   -v G K L) are separate code paths, whose summands and tails agree transfer by transfer.
3. Refusals: the Hartree start of a one-electron system (the H atom), whose level is degenerate and half
   filled, has no gap for the chemical potential. An unknown w_order, a start other than the backend's and
   a hedin section or Grids with a bad temperature, count or box are refused.
4. The derivative along the map (tangents.py) of Sigma^(1), both routes and both starts, on an
   asymmetric dimer: against central finite differences at Omega = 0 (the start line's h shifted by a
   random symmetric matrix), and gauge covariance at Omega != 0: in the direction Y = i Omega (a uniform
   time-dependent phase of the start line) every diagrammatic functional obeys
   delta Sigma(nu; Omega) = Sigma(nu) - Sigma(nu + Omega). In that direction every triangle vanishes, so
   the gauge check holds the line term with its tail and that the boson and vertex tangents vanish. The
   frequency routing of the triangles at Omega != 0 is check 9, and that of the other rules is held by the
   cross-route identities of checks 5 and 7.
5. Cycle 2 of the SG schemes: SG-I/L minus SG-Gamma/W equals -G^(1) (W^(1) - W^(2)) Gamma^(2) (it holds in
   either reading of W, as long as both sides read it alike), and the lag identity, SG-Gamma/W without its
   lag equal to SG-I/L, from both starts on an asymmetric dimer and at half filling. The lag identity
   separates the readings: it holds with W(4,1), and the cycle-2 polarizability P^(2) = G^(1) G^(1)
   Gamma^(2) is not symmetric in general, so the printed W(1,4) breaks it where P^(2) is not symmetric (and
   agrees on the particle-hole symmetric half-filled dimer), which the test records.
6. Kernels at fixed form (the cycle-based schemes' K = delta S_1[G] / delta G at G^(1), a line with a
   frequency-dependent self-energy), both routes and both starts: Kernel.apply at Omega = 0 against
   central finite differences of the level functional, for a Hermitian static direction (h shifted) and a
   direction on the vertex box (a self-energy with Y(-nu) = Y(nu)^dagger added there).
7. Cycle 2 of the cycle-based schemes (the vertex BSE of the kernel at fixed form solved by GMRES): with
   W(4,1), cycle-based Gamma/W equals cycle-based I/L beyond 0-D, from both starts on the asymmetric dimer.
   With w_order = '1,4' the kernel at fixed form is the same (the cycle-1 level is the same functional in
   both orders), and from (HF), whose kernel is not a second derivative, cycle-based Gamma/W leaves
   cycle-based I/L. From (H) the two readings differ only by the vertex box's truncation (not asserted).
8. Numerics: without a hedin section the backend takes beta = HEDIN_BETA_GAP over the start's gap and the
   default grids, and a BSE of a kernel at fixed form that stops short (bse_max_iter = 1) keeps its result
   and raises the flag 'hedin_bse_not_converged', which run_cycles collects into the history.
9. The triangles at Omega != 0 on the cycle-1 line, for a complex non-Hermitian static direction and a
   direction on the vertex box, against a brute-force sum over 6000 fermionic frequencies.

Exactness against exact diagonalization on the dimer is tests/test_hedin_cycles_dimer.py.

Run: python tests/test_hedin_cycles_matsubara.py, or under pytest.
"""
import itertools
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.numerics.constants import HEDIN_BETA_GAP, HEDIN_M_BOSON, HEDIN_N_FERMION
from src.Base.numerics.numerics import Numerics
from src.SingleReference.BSE.second_order_qp import cycle_self_energy
from src.SingleReference.HedinCycles.matsubara.backend import Boson, Line, MatsubaraBackend, ThreePoint
from src.SingleReference.HedinCycles.matsubara.grids import Grids
from src.SingleReference.HedinCycles.matsubara.systems import from_pyscf, hubbard_dimer
from src.SingleReference.HedinCycles.matsubara.tangents import LineTangent, Seed, Tangents
from src.SingleReference.HedinCycles.schemes import SCHEMES, gw_level, il_level, run_cycles, start_level

#: Sigma_c against the house pole sums at n_fermion = m_boson = 256, beta = 40 (measured 1.0e-5 for the
#: dimer from (H), 5.2e-6 from (HF), 1.6e-6 and 2.0e-6 for H2), and the bar is three times the largest. The error falls
#: about 8x per doubling of the grids (1.3e-6 at 512 for the dimer from (H)).
HOUSE_TOL = 3e-5
#: the two SG schemes' cycle-2 difference against -G^(1) (W^(1) - W^(2)) Gamma^(2), relative: the two routes
#: truncate the internal boson sums of their tangents differently, most for transfers near the boson grid's
#: edge (measured 2.4e-6 from (H) and 9.5e-6 from (HF) at beta = 5 with m_boson = 64 and m_vertex = 8, 8.6e-5
#: from (HF) with m_boson = 32), and the bar is three times the largest
SG_DIFFERENCE_TOL = 3e-5
#: SG-Gamma/W without its lag against SG-I/L, relative: measured 6.9e-8 from (H) and 7.0e-7 from (HF) on the
#: asymmetric dimer, 3e-16 and 4e-15 at half filling, and the bar is three times the largest
LAG_TOL = 3e-6
#: Tangents.triangle at Omega != 0 against a brute-force sum over 6000 fermionic frequencies, relative: the
#: static direction is limited by the brute force's window (measured 7.8e-10 to 9.3e-10), the box direction
#: is exact (6.4e-16 to 7.5e-16), and the bars are three times the largest and 1e-14
TRIANGLE_TOL = {'static': 3e-9, 'box': 1e-14}
#: Kernel.apply against central differences (step 1e-5) on the cycle-1 line, relative: a static direction
#: (measured 4.2e-10 to 1.2e-9) and a direction on the vertex box (4.3e-10 to 1.4e-9)
FIXED_FORM_FD_TOL = {'static': 1e-8, 'box': 1e-8}
#: cycle-based Gamma/W against cycle-based I/L at cycle 2, relative: the routes truncate their tangents'
#: internal boson sums differently (measured 1.0e-6 from (H) and 6.5e-6 from (HF) with m_boson = 32), and
#: the bar is three times the largest
CYCLE_BASED_ROUTE_TOL = 2e-5


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def h2():
    return from_pyscf(scf.RHF(gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)).run())


def grids(beta, n):
    return Grids(beta=beta, n_fermion=n, n_vertex=16, m_boson=n, m_vertex=8)


def house_difference(system, start, g):
    """max over n = 0 .. 19 and the spatial orbitals of |Sigma_c^(1)(i nu_n) - house|."""
    alg = MatsubaraBackend(system, start, g)
    sigma = run_cycles(alg, 'sg_il', start, 1).sigmas[1].dyn
    n = system.norb // 2
    C, e = alg.start.C, alg.e
    up = [k for k in range(2 * n) if np.abs(C[:n, k]).sum() > 1e-8]
    B = np.einsum('ip,Pij,jq->Ppq', C[:n, up], system.B[:, :n, :n], C[:n, up])
    poles_of, _ = cycle_self_energy('G0W0' if start == 'H' else 'Sigma_TDHF@L_TDHF', e[up], B,
                                    alg.start.nocc // 2)
    nn = np.arange(20)
    z = 1j * g.nu(nn) + alg.start.mu
    worst = 0.0
    for k, p in enumerate(up):
        r, P = poles_of(k)
        ref = (r[None, :] / (z[:, None] - P[None, :])).sum(axis=1)
        worst = max(worst, np.abs(sigma[nn + g.n_fermion, p, p] - ref).max())
    return worst


def check_cycle_one_against_the_house():
    ok = True
    dimer, mol = hubbard_dimer(1.0, 1.0), h2()
    for name, system, start, beta in [('dimer U=1', dimer, 'H', 40.0), ('dimer U=1', dimer, 'HF', 40.0),
                                      ('H2/STO-3G', mol, 'HF', 40.0), ('H2/STO-3G', mol, 'H', 80.0)]:
        coarse = house_difference(system, start, grids(beta, 128))
        fine = house_difference(system, start, grids(beta, 256))
        ok &= check(fine < HOUSE_TOL and coarse / fine > 5,
                    f'{name} ({start}), beta = {beta:g}: cycle 1 = the house pole sums, converging with the grids',
                    f'{coarse:.1e} at 128, {fine:.1e} at 256')
    return ok


def check_four_schemes_share_cycle_one():
    """The routes are algebraically equal, and their summands agree transfer by transfer. The tails of
    their sums are linear in the summand at the two ends of the grid, so the routes agree to rounding from
    both starts (measured 1e-16 to 2e-15)."""
    ok = True
    for system in (hubbard_dimer(1.0, 1.5, eps=(0.0, 0.3)), h2()):
        for start in ('H', 'HF'):
            spread = []
            for n in (128, 256):
                alg = MatsubaraBackend(system, start, grids(10.0, n))
                sigmas = [run_cycles(alg, s, start, 1).sigmas[1].at_grid() for s in SCHEMES]
                spread.append(max(np.abs(s - sigmas[0]).max() for s in sigmas))
            same = max(spread) < 1e-12
            ok &= check(same, f'{system.name[:40]} ({start}): cycle 1 is the same in all four schemes',
                        f'{spread[0]:.1e} at 128, {spread[1]:.1e} at 256')
    return ok


def check_refusals():
    system = from_pyscf(scf.ROHF(gto.M(atom='H 0 0 0', basis='sto-3g', spin=1, verbose=0)).run())
    try:
        MatsubaraBackend(system, 'H', grids(40.0, 64))
        refused = False
    except ValueError:
        refused = True
    alg = MatsubaraBackend(system, 'HF', grids(40.0, 64))
    try:
        MatsubaraBackend(system, 'HF', grids(40.0, 64), w_order='2,1')
        order_refused = False
    except ValueError:
        order_refused = True
    ok = (check(refused, 'the Hartree start of the H atom (a half-filled degenerate level) is refused')
          & check(alg.start.gap > 0.5, 'its HF start has a gap', f'{alg.start.gap:.3f} Ha')
          & check(order_refused, "a w_order other than '4,1' and '1,4' is refused"))
    try:
        run_cycles(MatsubaraBackend(hubbard_dimer(1.0, 1.0), 'HF', grids(10.0, 16)), 'sg_il', 'H', 1)
        start_refused = False
    except ValueError:
        start_refused = True
    ok &= check(start_refused, 'run_cycles refuses a start other than the one the backend was built on')
    good = dict(beta=10.0, n_fermion=16, n_vertex=4, m_boson=16, m_vertex=2)
    bad = [('beta', 0.0), ('beta', -1.0), ('n_fermion', 0), ('m_boson', -2), ('n_vertex', True),
           ('m_vertex', 1.5), ('n_vertex', 32), ('m_vertex', 32)]
    for name, value in bad:
        caught = []
        for build in (lambda kw: Numerics(hedin=kw), lambda kw: Grids(**kw)):
            try:
                build(dict(good, **{name: value}))
                caught.append(False)
            except ValueError:
                caught.append(True)
        ok &= check(all(caught), f'a hedin section and a Grids with {name} = {value!r} are refused')
    return ok


# ------------------------------------------------------------ Tier 1 measurements, for any system
def tangent_errors(system, start, g, seed=1):
    """{route: (finite-difference error, gauge error)} of the tangent of Sigma^(1) along the map: the start
    line's h shifted by a random real symmetric matrix at Omega = 0 (relative), and the gauge direction
    Y = i Omega at every transfer of the vertex box (absolute)."""
    rng = np.random.default_rng(seed)
    alg = MatsubaraBackend(system, start, g)
    S0 = start_level(alg, start)
    N = alg.ctx.N
    box = g.vertex_n + g.n_fermion
    out = {}
    for route, level in (('Gamma/W', gw_level), ('I/L', il_level)):
        def full(G):
            return alg.sigma_h(G) + level(alg, S0)(G)
        sigma = full(alg.line0)
        Y = rng.normal(size=(N, N))
        Y = Y + Y.T
        Ys = np.zeros((len(g.boson_m), 1, N, N), dtype=complex)
        Ys[g.boson_index(0), 0] = Y
        tan = Tangents(alg, Seed(alg.line0, ThreePoint(alg.ctx, False, Ys)))(sigma)
        eta = 1e-5
        fd = [full(Line(alg.ctx, np.diag(alg.e) + s * eta * Y)).at_grid() for s in (1, -1)]
        fd = (fd[0] - fd[1]) / (2 * eta)
        ours = tan.static[g.boson_index(0), 0][None] + tan.dyn[g.m_vertex, 0]
        err = np.abs(ours - fd[box]).max() / np.abs(fd[box]).max()
        Yg = np.stack([1j * g.omega(m) * np.eye(N) for m in g.boson_m])[:, None]
        tg = Tangents(alg, Seed(alg.line0, ThreePoint(alg.ctx, False, Yg)))(sigma)
        val = sigma.at_grid()
        gauge = max(np.abs(tg.dyn[k, 0] + (0 if tg.static is None else tg.static[g.boson_index(m), 0][None])
                           - (val[box] - val[box + m])).max() for k, m in enumerate(g.vertex_m))
        out[route] = (err, gauge)
    return out


def sg_difference_error(system, start, g):
    """SG-I/L minus SG-Gamma/W at cycle 2 against -G^(1) (W^(1) - W^(2)) Gamma^(2), relative to the
    difference, and the difference itself."""
    alg = MatsubaraBackend(system, start, g)
    il, gw = run_cycles(alg, 'sg_il', start, 2), run_cycles(alg, 'sg_gw', start, 2)
    G0 = alg.line0
    W1 = alg.screen(alg.polarizability(G0, alg.solve_vertex(G0, alg.kernel(start_level(alg, start), G0))))
    naux = alg.ctx.naux
    diff = Boson(alg.ctx, 'M', W1.data - gw.screened[2].data + np.eye(naux)[None], ('test',))
    one = Boson(alg.ctx, 'M', np.broadcast_to(np.eye(naux), W1.data.shape).copy(), ('test',))
    formula = (alg.sigma_gw(gw.lines[1], diff, gw.vertices[2]).dyn
               - alg.sigma_gw(gw.lines[1], one, gw.vertices[2]).dyn)
    delta = il.sigmas[2].at_grid() - gw.sigmas[2].at_grid()
    return np.abs(delta - formula).max() / np.abs(delta).max(), np.abs(delta).max()


def lag_identity(system, g, start='H'):
    """SG-Gamma/W without its lag against SG-I/L with W(4,1) and with the printed W(1,4) (relative to
    Sigma^(2) - Sigma^(0)), the largest change of Gamma^(2) between the orders, and the asymmetry of M^(2)."""
    alg = MatsubaraBackend(system, start, g)
    il, un = run_cycles(alg, 'sg_il', start, 2), run_cycles(alg, 'sg_gw', start, 2, lag=False)
    un14 = run_cycles(MatsubaraBackend(system, start, g, w_order='1,4'), 'sg_gw', start, 2, lag=False)
    W = un.screened[2]
    s_il = il.sigmas[2].at_grid()
    scale = np.abs(s_il - il.sigmas[0].at_grid()).max()
    lag = np.abs(un.sigmas[2].at_grid() - s_il).max() / scale
    printed = np.abs(un14.sigmas[2].at_grid() - s_il).max() / scale
    same_vertex = max(np.abs(un.vertices[2].static - un14.vertices[2].static).max(),
                      np.abs(un.vertices[2].dyn - un14.vertices[2].dyn).max())
    asym = max(np.abs(W.at(m) - W.at(-m).T).max() for m in g.boson_m)
    return lag, printed, same_vertex, asym


def fixed_form_errors(system, start, g, seed=3):
    """{(route, kind): error} of Kernel.apply at Omega = 0 on G^(1) against central finite differences of the
    level functional, relative: a Hermitian static direction (h shifted) and a direction on the vertex box
    (a self-energy with Y(-nu) = Y(nu)^dagger added there)."""
    rng = np.random.default_rng(seed)
    box = slice(g.n_fermion - g.n_vertex, g.n_fermion + g.n_vertex)
    inbox = g.vertex_n + g.n_fermion
    alg = MatsubaraBackend(system, start, g)
    G1 = run_cycles(alg, 'sg_il', start, 1).lines[1]
    S0 = start_level(alg, start)
    N, nb = alg.ctx.N, len(g.boson_m)
    il = il_level(alg, S0)
    out = {}
    for route, S1 in (('Gamma/W', gw_level(alg, S0)), ('I/L', lambda G: alg.sigma_h(G) + il(G))):
        K = alg.kernel(S1, G1)
        for kind in ('static', 'box'):
            if kind == 'static':
                Y = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
                Y = Y + Y.conj().T
                static = np.zeros((nb, 1, N, N), dtype=complex)
                static[g.boson_index(0), 0] = Y
                direction = ThreePoint(alg.ctx, False, static, None)

                def line(s, Y=Y):
                    return Line(alg.ctx, G1.h + s * Y, G1.dyn)
            else:
                Yn = rng.normal(size=(2 * g.n_vertex, N, N)) + 1j * rng.normal(size=(2 * g.n_vertex, N, N))
                Yn = Yn + np.conj(np.swapaxes(Yn[::-1], 1, 2))
                dyn = np.zeros((len(g.vertex_m), 1, 2 * g.n_vertex, N, N), dtype=complex)
                dyn[g.m_vertex, 0] = Yn
                direction = ThreePoint(alg.ctx, False, None, dyn)

                def line(s, Yn=Yn):
                    d = G1.dyn.copy()
                    d[box] += s * Yn
                    return Line(alg.ctx, G1.h, d)
            tan = K.apply(direction)
            eta = 1e-5
            fd = (S1(line(eta)).at_grid() - S1(line(-eta)).at_grid())[inbox] / (2 * eta)
            ours = (0 if tan.static is None else tan.static[g.boson_index(0), 0][None]) + tan.dyn[g.m_vertex, 0]
            out[route, kind] = np.abs(ours - fd).max() / np.abs(fd).max()
    return out


def cycle_based_routes(system, start, g, orders=True, seed=5):
    """Cycle 2 of the cycle-based schemes: {'route': CB-Gamma/W against CB-I/L, 'apart': CB-Gamma/W against
    SG-I/L (both relative to Sigma^(2) - Sigma^(0)), 'converged': no BSE flag}, and with orders also
    {'kernel_orders': the kernel at fixed form with W(1,4) against W(4,1) on a random direction at every
    transfer (relative), 'printed': CB-Gamma/W with W(1,4) against CB-I/L (relative)}."""
    alg = MatsubaraBackend(system, start, g)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hist = {s: run_cycles(alg, s, start, 2) for s in ('cb_gw', 'cb_il', 'sg_il')}
        if orders:
            alg14 = MatsubaraBackend(system, start, g, w_order='1,4')
            cb14 = run_cycles(alg14, 'cb_gw', start, 2)
    s = {k: h.sigmas[2].at_grid() for k, h in hist.items()}
    scale = np.abs(s['cb_il'] - hist['cb_il'].sigmas[0].at_grid()).max()
    out = {'route': np.abs(s['cb_gw'] - s['cb_il']).max() / scale,
           'apart': np.abs(s['cb_gw'] - s['sg_il']).max() / scale,
           'converged': not any('BSE' in str(w.message) for w in caught)}
    if not orders:
        return out
    rng = np.random.default_rng(seed)
    N, nb = alg.ctx.N, len(g.boson_m)
    Y = (rng.normal(size=(nb, 1, N, N)) + 0j, rng.normal(size=(len(g.vertex_m), 1, 2 * g.n_vertex, N, N)) + 0j)
    k41 = alg.kernel(gw_level(alg, start_level(alg, start)), hist['cb_gw'].lines[1]).apply(
        ThreePoint(alg.ctx, False, *Y))
    k14 = alg14.kernel(gw_level(alg14, start_level(alg14, start)), cb14.lines[1]).apply(
        ThreePoint(alg14.ctx, False, *Y))
    out['kernel_orders'] = max(np.abs(k41.static - k14.static).max(),
                               np.abs(k41.dyn - k14.dyn).max()) / np.abs(k41.dyn).max()
    out['printed'] = np.abs(cb14.sigmas[2].at_grid() - s['cb_il']).max() / scale
    return out


# ------------------------------------------------------------------------- the dimer's checks
def check_tangents():
    ok = True
    g = Grids(beta=10.0, n_fermion=128, n_vertex=8, m_boson=128, m_vertex=3)
    for start in ('H', 'HF'):
        for route, (err, gauge) in tangent_errors(hubbard_dimer(1.0, 1.0, eps=(0.0, 0.4)), start, g).items():
            ok &= check(err < 1e-8 and gauge < 1e-12,
                        f'({start}) {route} route: the tangent of Sigma^(1) against finite differences and gauge covariance',
                        f'FD {err:.1e} relative, gauge {gauge:.1e}')
    return ok


def check_sg_cycle_two():
    ok = True
    g = Grids(beta=5.0, n_fermion=32, n_vertex=16, m_boson=64, m_vertex=8)
    for start in ('H', 'HF'):
        err, size = sg_difference_error(hubbard_dimer(1.0, 1.0, eps=(0.0, 0.4)), start, g)
        ok &= check(err < SG_DIFFERENCE_TOL,
                    f'({start}) cycle 2: SG-I/L - SG-Gamma/W = -G^(1) (W^(1) - W^(2)) Gamma^(2)',
                    f'{err:.1e} relative on a difference of {size:.1e}')
    for (system, symmetric), start in itertools.product(
            ((hubbard_dimer(1.0, 1.0, eps=(0.0, 0.4)), False), (hubbard_dimer(1.0, 1.0), True)), ('H', 'HF')):
        lag, printed, same_vertex, asym = lag_identity(system, g, start)
        label = f'half filling ({start})' if symmetric else f'eps = (0, 0.4) ({start})'
        ok &= check(lag < LAG_TOL, f'dimer {label}: SG-Gamma/W without its lag = SG-I/L', f'{lag:.1e} relative')
        ok &= check(same_vertex < 1e-13, f'dimer {label}: Gamma^(2) is the same with w_order = 1,4 '
                    '(cycle 1 has a symmetric P for every line, so its tangent does not depend on the order)',
                    f'{same_vertex:.1e}')
        if symmetric:
            ok &= check(printed < LAG_TOL and asym < 1e-5,
                        f'dimer {label}: P^(2) is symmetric, so the printed W(1,4) gives the same',
                        f'{printed:.1e} relative, asymmetry {asym:.1e}')
        else:
            ok &= check(printed > 10 * lag,
                        f'dimer {label}: P^(2) is not symmetric, and the printed W(1,4) breaks the identity',
                        f'asymmetry of M {asym:.1e}, difference {printed:.1e} relative')
    return ok


def check_kernels_at_fixed_form():
    ok = True
    g = Grids(beta=5.0, n_fermion=32, n_vertex=8, m_boson=32, m_vertex=4)
    for start in ('H', 'HF'):
        for (route, kind), err in fixed_form_errors(hubbard_dimer(1.0, 1.0, eps=(0.0, 0.4)), start, g).items():
            ok &= check(err < FIXED_FORM_FD_TOL[kind],
                        f'({start}) {route} route: the kernel at fixed form on G^(1), {kind} direction, '
                        f'against finite differences', f'{err:.1e} relative')
    return ok


def check_cycle_based_cycle_two():
    ok = True
    g = Grids(beta=5.0, n_fermion=32, n_vertex=8, m_boson=32, m_vertex=4)
    for start in ('H', 'HF'):
        r = cycle_based_routes(hubbard_dimer(1.0, 1.0, eps=(0.0, 0.4)), start, g)
        ok &= check(r['route'] < CYCLE_BASED_ROUTE_TOL and r['apart'] > 1e-3 and r['converged'],
                    f'({start}) cycle 2: cycle-based Gamma/W = cycle-based I/L, apart from SG-I/L',
                    f"{r['route']:.1e} relative, {r['apart']:.1e} from SG-I/L, GMRES converged: {r['converged']}")
        ok &= check(r['kernel_orders'] < 1e-13, f'({start}) the kernel at fixed form is the same with w_order = 1,4',
                    f"{r['kernel_orders']:.1e} relative")
        if start == 'HF':
            ok &= check(r['printed'] > 10 * r['route'],
                        f'({start}) cycle 2 with the printed W(1,4): cycle-based Gamma/W leaves cycle-based I/L '
                        'by more than ten times the routes\' own difference', f"{r['printed']:.1e} relative")
    return ok


def triangle_errors(system, start, g, seed=7):
    """Tangents.triangle on the cycle-1 line (a frequency-dependent self-energy) at transfers Omega != 0, for a
    complex, non-Hermitian static direction and a direction on the vertex box, against a brute-force sum over
    2 L fermionic frequencies with line.at: {(kind, m): relative error}. The gauge direction of check 4 makes
    every triangle vanish, and the finite differences of check 6 see Omega = 0 only, so this is the direct
    check of the triangles' frequency routing at Omega != 0."""
    alg = MatsubaraBackend(system, start, g)
    G1 = run_cycles(alg, 'sg_il', start, 1).lines[1]
    N, B = alg.ctx.N, alg.ctx.B
    rng = np.random.default_rng(seed)
    Ys = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    Yn = 0.3 * (rng.normal(size=(2 * g.n_vertex, N, N)) + 1j * rng.normal(size=(2 * g.n_vertex, N, N)))
    gam = np.broadcast_to(B, (len(g.boson_m),) + B.shape).astype(complex)
    L = 3000
    nn = np.arange(-L, L)
    out = {}
    for kind, m in itertools.product(('static', 'box'), (2, -1)):
        static = np.zeros((len(g.boson_m), 1, N, N), dtype=complex)
        dyn = None
        if kind == 'static':
            static[g.boson_index(m), 0] = Ys
        else:
            dyn = np.zeros((len(g.vertex_m), 1, 2 * g.n_vertex, N, N), dtype=complex)
            dyn[m + g.m_vertex, 0] = Yn
        Y = ThreePoint(alg.ctx, False, static, dyn)
        T = Tangents(alg, Seed(G1, Y)).triangle(LineTangent(G1, Y), gam, m)               # (w, R, Q, N, N)

        def direction(k):
            out_ = np.broadcast_to(Ys if kind == 'static' else np.zeros((N, N)), (len(k), N, N)).astype(complex)
            if kind == 'box':
                inside = (k >= -g.n_vertex) & (k < g.n_vertex)
                out_[inside] = Yn[k[inside] + g.n_vertex]
            return out_
        worst = 0.0
        for w in (0, 3, -5):
            ref = (np.einsum('nab,nbc,ncd,Qde,nef->Qaf', G1.at(nn + w + m), direction(nn + w), G1.at(nn + w), B,
                             G1.at(nn), optimize=True)
                   + np.einsum('nab,Qbc,ncd,nde,nef->Qaf', G1.at(nn + w + m), B, G1.at(nn + m), direction(nn),
                               G1.at(nn), optimize=True)) / g.beta
            worst = max(worst, np.abs(T[g.boson_index(w), 0] - ref).max() / np.abs(ref).max())
        out[kind, m] = worst
    return out


def check_triangles_at_nonzero_transfer():
    ok = True
    g = Grids(beta=5.0, n_fermion=32, n_vertex=8, m_boson=32, m_vertex=4)
    for start in ('H', 'HF'):
        for (kind, m), err in triangle_errors(hubbard_dimer(1.0, 1.0, eps=(0.0, 0.4)), start, g).items():
            ok &= check(err < TRIANGLE_TOL[kind],
                        f'({start}) the triangle on G^(1) at Omega_{m}, {kind} direction, against a brute-force sum',
                        f'{err:.1e} relative')
    return ok


def check_numerics():
    alg = MatsubaraBackend(hubbard_dimer(1.0, 1.0), 'H')
    g = alg.grids
    ok = check(abs(g.beta * alg.start.gap - HEDIN_BETA_GAP) < 1e-12 and g.n_fermion == HEDIN_N_FERMION
               and g.m_boson == HEDIN_M_BOSON, 'without a hedin section: beta = HEDIN_BETA_GAP / gap, default grids',
               f'beta {g.beta:.3f} for the gap {alg.start.gap:.3f}')
    short = Numerics(hedin=dict(beta=5.0, n_fermion=32, n_vertex=8, m_boson=32, m_vertex=4, bse_max_iter=1))
    alg = MatsubaraBackend(hubbard_dimer(1.0, 1.0, eps=(0.0, 0.4)), 'HF', short)
    hist = run_cycles(alg, 'cb_gw', 'HF', 2)
    names = [f['flag'] for f in hist.numerical_flags]
    ok &= check(names == ['hedin_bse_not_converged'] and np.isfinite(hist.sigmas[2].at_grid()).all(),
                'a BSE of a kernel at fixed form stopped short keeps its result and flags it in the history',
                f'{names}')
    return ok


def run():
    warnings.simplefilter('ignore')
    print('-- 1. cycle 1 against the house GW code')
    ok = check_cycle_one_against_the_house()
    print('\n-- 2. cycle 1 in all four schemes')
    ok &= check_four_schemes_share_cycle_one()
    print('\n-- 3. refusals')
    ok &= check_refusals()
    print('\n-- 4. the derivative along the map')
    ok &= check_tangents()
    print('\n-- 5. cycle 2 of the SG schemes')
    ok &= check_sg_cycle_two()
    print('\n-- 6. kernels at fixed form')
    ok &= check_kernels_at_fixed_form()
    print('\n-- 7. cycle 2 of the cycle-based schemes')
    ok &= check_cycle_based_cycle_two()
    print('\n-- 8. numerics')
    ok &= check_numerics()
    print('\n-- 9. the triangles at Omega != 0')
    ok &= check_triangles_at_nonzero_transfer()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_hedin_cycles_matsubara():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
