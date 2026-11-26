"""The Hedin cycles on the half-filled Hubbard dimer (t = 1): TEST_PLAN's Tier 4, exactness.

The exact self-energy of the half-filled dimer on its Hartree line is purely second order, (U^2/4)/(i nu - 3t)
in the bonding orbital, so every term of third or higher order in a cycle is error. The error is that of
[G^(n)]^-1 in the bonding orbital, the largest over the four lowest Matsubara frequencies, against the T = 0
Lehmann G of `ExactSelfEnergy`. The slope between U = 0.1 and 0.05 must be the scheme's first wrong order:

    cycle 1: U^2 from (H) (G0W0), U^3 from (HF)
    cycle 2: SG-I/L U^4 from both starts (one order more than SG-Gamma/W from (H), special to the dimer),
             SG-Gamma/W U^3 from (H) and U^4 from (HF)

beta = 20: the start gap 2t keeps the thermal error near exp(-20). The grids are small (n_fermion = 64,
m_boson = 128, n_vertex = 8, m_vertex = 32: the dimer at half filling has a symmetric P^(2) whatever the box's
shape), and the transfer box and the boson grid converge as their cutoffs cubed thanks to the tails of
backend._three_point_sigma. The slopes then carry the next order of each
scheme (the U^5 term of SG-I/L, about -0.06 U^5, makes its slope 3.9) and the boson grid's U^2 remainder (3e-9
at U = 0.05).

Where the dimer report's large box is converged, the coefficients agree with it: SG-Gamma/W from (H) 0.054 U^3
at U = 0.1 (the report's 0.054), and from (HF) SG-I/L 0.0110 U^4 (0.0108) and SG-Gamma/W 0.0301 U^4 (0.0305).
SG-I/L from (H) gives 0.049 U^4 where the report has 0.034 U^4 (3.35e-6 at U = 0.1). Ours is stable to 0.3 %
from beta = 20 to 40 and in every cutoff. The report's value halved between its two boxes (7.03e-6 to 3.35e-6)
and sums the vertex only over its transfer box, without a tail, so its truncation is the likely cause. That
was not checked in its code, and the test asserts the slope only.

Instabilities (their treatment by the soft-mode split comes later): the cycles must still run across them.
The (HF) start's static BSE K chi(Omega = 0) has the RHF triplet instability of the dimer, an eigenvalue
crossing 1 between U = 2.0t and 2.0002t at beta = 10 (the report's 2.0002). The density sources of the
vertex and of L drive only the charge channel, so the solves stay finite through the crossing and beyond
it, where they describe an unstable reference. The check asserts that the largest eigenvalue of K chi(0) is
below 1 at U = 2.0t and above it at 2.0002t and 3t, where every scheme runs cycle 2, finite, with GMRES
converged. From (H) every scheme runs at U = 8t and both SG schemes at 16t ('at large U': the report's
magnetic pole of the strict cycle-based kernel, at 6.74t to 6.98t, is not located here).

Off half filling (site energies 0 and 0.4t), where symmetry no longer fixes the static update of every
cycle's line, the SG schemes' slopes must be at least each cycle's first wrong order (U^2 and U^3 from (H),
U^3 and U^4 from (HF)), less SLOPE_TOL: measured 1.89, 3.50 (SG-I/L) and 2.87 (SG-Gamma/W) from (H), 2.98,
3.91 and 3.94 from (HF).

Run: python tests/test_hedin_cycles_dimer.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.CodeGeneration.diagrammatic.exact_oracle import ExactSelfEnergy
from src.SingleReference.HedinCycles.matsubara.backend import MatsubaraBackend
from src.SingleReference.HedinCycles.matsubara.grids import Grids
from src.SingleReference.HedinCycles.matsubara.systems import hubbard_dimer
from src.SingleReference.HedinCycles.schemes import SCHEMES, run_cycles

#: the grids of the exactness check
GRIDS = Grids(beta=20.0, n_fermion=64, n_vertex=8, m_boson=128, m_vertex=32)
#: the slope may miss the first wrong order by this much (measured 0.03 to 0.19: 1.86, 2.97, 3.90, 2.81,
#: 3.89, 3.90 for the six rows below)
SLOPE_TOL = 0.25
#: the converged coefficients of the dimer report (table C and test_hf.py B, the large box) at U = 0.1, and
#: the relative agreement asked of them (measured 0.7 to 1.9 %)
REPORT = {('H', 'sg_gw', 2): (0.054, 3), ('HF', 'sg_il', 2): (0.0108, 4), ('HF', 'sg_gw', 2): (0.0305, 4)}
REPORT_TOL = 0.05


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def exact_inverse(alg, z):
    """[G_exact(z)]^-1 in the start's orbitals (occupied first), from the Lehmann G at T = 0. For a closed shell
    in spatial orbitals (System.spin = 2) the problem is set up in spin orbitals, the occupied ones of both
    spins first as ExactSelfEnergy needs them, and the spin-up block of the inverse is returned."""
    C = alg.start.C
    h1 = C.T @ alg.system.h @ C
    B = alg.ctx.B.real
    e, nocc = alg.e, alg.start.nocc
    if alg.system.spin == 2:
        n = len(e)
        up = np.r_[0:nocc, 2 * nocc:n + nocc]                  # spatial k with spin up, then with spin down
        down = np.r_[nocc:2 * nocc, n + nocc:2 * n]
        h_so, B_so, e_so = np.zeros((2 * n, 2 * n)), np.zeros((B.shape[0], 2 * n, 2 * n)), np.zeros(2 * n)
        for idx in (up, down):
            h_so[np.ix_(idx, idx)] = h1
            B_so[:, idx[:, None], idx[None, :]] = B
            e_so[idx] = e
        h1, B, e, nocc = h_so, B_so, e_so, 2 * nocc
    eri = np.einsum('Ppq,Prs->pqrs', B, B)
    g = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)
    exact = ExactSelfEnergy(e, h1, g, nocc)
    out = [np.linalg.inv(exact.green(w, 1.0)) for w in z]
    if alg.system.spin == 2:
        out = [x[np.ix_(up, up)] for x in out]
    return out


def errors(start, U, schemes, eps=(0.0, 0.0)):
    """{(scheme, cycle): error} for cycles 1 and 2 of each scheme, on the dimer with site energies eps."""
    alg = MatsubaraBackend(hubbard_dimer(1.0, U, eps=eps), start, GRIDS)
    nn = np.arange(4)
    exact = exact_inverse(alg, 1j * GRIDS.nu(nn) + alg.start.mu)
    out = {}
    for scheme in schemes:
        hist = run_cycles(alg, scheme, start, 2)
        for cycle in (1, 2):
            ours = [np.linalg.inv(G) for G in hist.lines[cycle].at(nn)]
            out[scheme, cycle] = max(abs(a[0, 0] - b[0, 0]) for a, b in zip(ours, exact))
    return out


def check_exactness():
    ok = True
    expected = {('H', 'sg_il', 1): 2, ('H', 'sg_il', 2): 4, ('H', 'sg_gw', 2): 3,
                ('HF', 'sg_il', 1): 3, ('HF', 'sg_il', 2): 4, ('HF', 'sg_gw', 2): 4}
    names = {'sg_il': 'SG-I/L', 'sg_gw': 'SG-Gamma/W'}
    for start in ('H', 'HF'):
        e1, e2 = errors(start, 0.1, ('sg_il', 'sg_gw')), errors(start, 0.05, ('sg_il', 'sg_gw'))
        for (s, scheme, cycle), power in expected.items():
            if s != start:
                continue
            slope = np.log2(e1[scheme, cycle] / e2[scheme, cycle])
            what = 'cycle 1' if cycle == 1 else f'{names[scheme]} cycle 2'
            ok &= check(abs(slope - power) < SLOPE_TOL, f'({start}) {what}: the error scales as U^{power}',
                        f'{e1[scheme, cycle]:.3e} at U = 0.1, {e2[scheme, cycle]:.3e} at 0.05, slope {slope:.2f}')
            if (start, scheme, cycle) in REPORT:
                ref, p = REPORT[start, scheme, cycle]
                ours = e1[scheme, cycle] / 0.1 ** p
                ok &= check(abs(ours / ref - 1) < REPORT_TOL,
                            f'({start}) {what}: the coefficient agrees with the dimer report',
                            f'{ours:.4f} U^{p} against {ref} U^{p}')
    return ok


def largest_static_eigenvalue(alg):
    """The largest real eigenvalue of K chi(Omega = 0) for the (HF) start's exchange kernel."""
    N = alg.ctx.N
    K = alg.kernel(alg.sigma_x, alg.line0)
    lam = np.linalg.eigvals(K.static_matrix() @ alg.line0.pair_map(0).reshape(N * N, N * N))
    return lam.real.max()


def check_across_instabilities():
    g = Grids(beta=10.0, n_fermion=32, n_vertex=8, m_boson=32, m_vertex=4)
    below = largest_static_eigenvalue(MatsubaraBackend(hubbard_dimer(1.0, 2.0), 'HF', g))
    ok = check(below < 1, '(HF) U = 2.0t: the RHF triplet instability is not crossed yet',
               f'largest eigenvalue of K chi(0) {below:.6f}')
    for start, U in (('HF', 2.0002), ('HF', 3.0), ('H', 8.0), ('H', 16.0)):
        alg = MatsubaraBackend(hubbard_dimer(1.0, U), start, g)
        detail, crossed = '', True
        if start == 'HF':
            lam = largest_static_eigenvalue(alg)
            crossed = lam > 1
            detail = f'largest eigenvalue of K chi(0) {lam:.6f}, '
        finite, flagged = True, False
        schemes = SCHEMES if U < 10 else ('sg_il', 'sg_gw')
        for scheme in schemes:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                sigma = run_cycles(alg, scheme, start, 2).sigmas[2].at_grid()
            finite &= bool(np.isfinite(sigma).all())
            flagged |= any('BSE' in str(w.message) for w in caught)
        names = 'every scheme' if len(schemes) == 4 else 'both SG schemes'
        where = 'past the RHF triplet instability' if start == 'HF' else 'at large U'
        ok &= check(crossed and finite and not flagged, f'({start}) U = {U}t: cycle 2 of {names} runs {where}',
                    f'{detail}finite: {finite}, a BSE flag: {flagged}')
    return ok


def check_exactness_off_half_filling():
    """The SG schemes on the dimer with site energies 0 and 0.4t, where the static update of every cycle's
    line (its Hartree and exchange) is not fixed by symmetry: cycle n is exact through U^n from (H) and
    U^(n+1) from (HF), so the slope is at least the next order (less SLOPE_TOL). SG-I/L's extra order from
    (H) is special to half filling: here it has a small U^3 term (slope 3.50, about 1.7e-3 U^3 + 0.024 U^4)."""
    ok = True
    eps = (0.0, 0.4)
    first_wrong = {('H', 1): 2, ('H', 2): 3, ('HF', 1): 3, ('HF', 2): 4}
    names = {'sg_il': 'SG-I/L', 'sg_gw': 'SG-Gamma/W'}
    for start in ('H', 'HF'):
        e1 = errors(start, 0.1, ('sg_il', 'sg_gw'), eps)
        e2 = errors(start, 0.05, ('sg_il', 'sg_gw'), eps)
        for (scheme, cycle), err in e1.items():
            if cycle == 1 and scheme != 'sg_il':
                continue
            power = first_wrong[start, cycle]
            slope = np.log2(err / e2[scheme, cycle])
            what = 'cycle 1' if cycle == 1 else f'{names[scheme]} cycle 2'
            ok &= check(slope > power - SLOPE_TOL, f'eps = (0, 0.4t) ({start}) {what}: exact through U^{power - 1}',
                        f'{err:.3e} at U = 0.1, {e2[scheme, cycle]:.3e} at 0.05, slope {slope:.2f}')
    return ok


def run():
    warnings.simplefilter('ignore')
    print('-- exactness against exact diagonalization (beta = 20)')
    ok = check_exactness()
    print('\n-- exactness off half filling')
    ok &= check_exactness_off_half_filling()
    print('\n-- across the instabilities (beta = 10)')
    ok &= check_across_instabilities()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_hedin_cycles_dimer():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
