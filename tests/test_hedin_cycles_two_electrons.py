"""The Hedin cycles on two-electron molecules against exact diagonalization (milestone 3, slow).

TEST_PLAN's Tier 5 for two electrons, read as the dimer's Tier 4 on molecules. The interaction is scaled,
v -> lambda v (System.scaled), the start is rerun at each lambda = 0.1, 0.05 and 0.025, and the error of a cycle
is the Frobenius norm of [G^(n)]^-1 - [G_exact]^-1 over the four lowest Matsubara frequencies and all orbitals,
with G_exact the T = 0 Lehmann G of `ExactSelfEnergy` in the start's orbitals. The molecules are closed shells
and run in spatial orbitals (System.spin = 2, equal to the spin-orbital runs to 1e-12, tests/
test_hedin_cycles_spin_adapted.py), and exact_inverse sets up the exact problem in spin orbitals and returns one
spin block (it matches the spin-orbital reference to 4e-15). (The largest element
is not smooth in lambda: on H2/6-31G the error of cycle 1 from (H) moves from element (6,6) to (0,0) between
lambda = 0.05 and 0.025, and its slopes 2.10 and 1.98 extrapolate to 1.85, where the norm's 2.03 and 2.02 give
2.00.) The error must scale as the cycle's first wrong
order:

    cycle 1: lambda^2 from (H) (G0W0 misses the second-order exchange), lambda^3 from (HF)
    cycle 2: lambda^3 from (H), lambda^4 from (HF), in every scheme

read, as in tests/test_hedin_cycles_tier2.py, from the extrapolated slope 2 s_2 - s_1, where s_1 and s_2 are the
slopes of log error against log lambda between 0.1 and 0.05 and between 0.05 and 0.025. The thermal error
exp(-beta Delta / 2), with Delta the start's gap (0.63 to 2.0 Ha), stays below 2e-10.

The transfer box decides cycle 2 from (HF). The dynamic part of the cycle-2 vertex is second order, so dropping
it beyond the box of M transfers (m_vertex = M) is a lambda^3 error, the order this cycle must get right. On
H2/STO-3G at beta = 72 / Ha its share of the error at lambda = 0.1 falls from 61 % at M = 16 (the default box,
transfers up to 1.4 Ha) as M^-2.7, so that cycle 2 from (HF) shows lambda^3 at the default box (extrapolated
slopes 2.84 and 2.94 with the largest element) and lambda^4 once extrapolated to an infinite box. The test
extrapolates it:

1. H2/STO-3G at beta = 72 / Ha and the default box: cycles 1 and 2 from (H) (SG-I/L, SG-Gamma/W and cycle-based
   Gamma/W, which equals cycle-based I/L at cycle 2) and cycle 1 from (HF).
2. H2/STO-3G, cycle 2 from (HF) (the SG schemes), extrapolated in the box: with X_k = [G^(2)]^-1 at the boxes
   M_k = 24, 36, 54 (n_vertex = m_vertex = M_k, m_boson = 8 M_k, a geometric sequence of ratio r = 1.5),
   r^a = ||X_1 - X_2|| / ||X_2 - X_3|| and X_inf = X_3 + (X_3 - X_2) / (r^a - 1), the Richardson step with the
   measured exponent a, which must lie between 2 and 3.5.
3. H2/6-31G (beta = 72 / Ha, the default box) and He/cc-pVDZ (beta = 30 / Ha, n_fermion = 64, m_boson = 128,
   n_vertex = m_vertex = 8): cycles 1 and 2 from (H) (the SG schemes) and cycle 1 from (HF). He needs the
   larger boson grid: with m_boson = 64 (transfers up to 13 Ha) cycle 1 from (HF) carries a lambda^2 cut of 1.2 %
   of its second-order self-energy (extrapolated slope 2.41 where 3 is due), and with 128 it gives 3.01. The
   fermion grid does not matter there (64 and 128 agree to every printed digit).

Measured on one thread, the extrapolated slopes:

| | cycle 1 (H) | SG-I/L 2 (H) | SG-Gamma/W 2 (H) | cycle-based 2 (H) | cycle 1 (HF) | cycle 2 (HF), infinite box |
|---|---|---|---|---|---|---|
| H2/STO-3G | 2.00 | 2.95 | 3.01 | 3.05 | 2.97 | SG-I/L 4.02, SG-Gamma/W 4.01 |
| H2/6-31G | 2.00 | 2.99 | 3.02 | | 2.94 | |
| He/cc-pVDZ | 2.00 | 3.00 | 3.02 | | 3.01 | |

The box's exponent was 2.65 to 2.73.

Run: python tests/test_hedin_cycles_two_electrons.py (about 7 minutes, so on a cluster), or under pytest.
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
from test_hedin_cycles_tier2 import LAMBDAS, NAMES
from src.SingleReference.HedinCycles.matsubara.backend import MatsubaraBackend
from src.SingleReference.HedinCycles.matsubara.grids import Grids
from src.SingleReference.HedinCycles.matsubara.systems import from_pyscf
from src.SingleReference.HedinCycles.schemes import run_cycles

#: the default box at beta = 72 / Ha (the hedin section's defaults), and He's grids
GRIDS = Grids(beta=72.0, n_fermion=128, n_vertex=16, m_boson=128, m_vertex=16)
HE_GRIDS = Grids(beta=30.0, n_fermion=64, n_vertex=8, m_boson=128, m_vertex=8)
#: the transfer boxes of the extrapolation, a geometric sequence
BOXES = (24, 36, 54)
#: the window of the measured exponent a of the box (measured 2.7 at lambda = 0.1)
BOX_EXPONENT = (2.0, 3.5)
#: the extrapolated slope may miss the first wrong order by this much (measured within 0.06, and the bar is
#: three times that)
SLOPE_TOL = 0.18


def molecule(atom, basis, name):
    """A closed-shell molecule in spatial orbitals."""
    mf = scf.RHF(gto.M(atom=atom, basis=basis, verbose=0)).run(conv_tol=1e-12)
    return from_pyscf(mf, name, spin_adapted=True)


def box_grids(M):
    return Grids(beta=72.0, n_fermion=128, n_vertex=M, m_boson=8 * M, m_vertex=M)


def inverse_lines(system, start, scheme, grids, lam, cycles):
    """[G^(n)]^-1 for n = 1 .. cycles and [G_exact]^-1 at the four lowest frequencies, each (4, N, N), and the
    run's numerical flags."""
    nn = np.arange(4)
    alg = MatsubaraBackend(system.scaled(lam), start, grids)
    exact = np.array(exact_inverse(alg, 1j * grids.nu(nn) + alg.start.mu))
    hist = run_cycles(alg, scheme, start, cycles)
    return [np.linalg.inv(hist.lines[n].at(nn)) for n in range(1, cycles + 1)], exact, hist.numerical_flags


def slope_check(label, v, p):
    s1, s2 = (np.log2(v[i] / v[i + 1]) for i in (0, 1))
    return check(abs(2 * s2 - s1 - p) < SLOPE_TOL, f'{label}: the error scales as lambda^{p}',
                 ', '.join(f'{x:.2e}' for x in v) + f', slopes {s1:.2f}, {s2:.2f}, extrapolated {2 * s2 - s1:.2f}')


def check_on_one_box(name, system, grids, rows):
    """rows: (start, scheme, cycles), every cycle of each row checked (cycle 1 once per start)."""
    ok, flags, seen = True, set(), set()
    for start, scheme, cycles in rows:
        errors = [[], []]
        for lam in LAMBDAS:
            ours, exact, f = inverse_lines(system, start, scheme, grids, lam, cycles)
            flags.update(f)
            for n in range(cycles):
                errors[n].append(np.linalg.norm(ours[n] - exact))
        for n in range(cycles):
            if n == 0 and start in seen:
                continue                                     # cycle 1 is the same in every scheme
            what = 'cycle 1' if n == 0 else f'{NAMES[scheme]} cycle 2'
            ok &= slope_check(f'{name} ({start}) {what}', errors[n], n + 2 + (start == 'HF'))
        seen.add(start)
    return check(not flags, f'{name}: no numerical flag in any run', ', '.join(sorted(flags)) or 'none') and ok


def box_extrapolated(system, start, scheme, lam):
    """The cycle-2 error extrapolated to an infinite transfer box, the measured exponent a, the error on the
    largest box and the numerical flags."""
    X, flags = [], set()
    for M in BOXES:
        ours, exact, f = inverse_lines(system, start, scheme, box_grids(M), lam, 2)
        X.append(ours[1])
        flags.update(f)
    ra = np.linalg.norm(X[0] - X[1]) / np.linalg.norm(X[1] - X[2])
    X_inf = X[2] + (X[2] - X[1]) / (ra - 1)
    return np.linalg.norm(X_inf - exact), np.log(ra) / np.log(BOXES[1] / BOXES[0]), np.linalg.norm(X[2] - exact), flags


def check_box_extrapolated(name, system, schemes):
    ok, flags = True, set()
    for scheme in schemes:
        errors, exponents, largest = [], [], []
        for lam in LAMBDAS:
            err, a, last, f = box_extrapolated(system, 'HF', scheme, lam)
            errors.append(err)
            exponents.append(a)
            largest.append(last)
            flags.update(f)
        ok &= check(all(BOX_EXPONENT[0] < a < BOX_EXPONENT[1] for a in exponents),
                    f'{name} (HF) {NAMES[scheme]} cycle 2: the box truncation falls as M^-a',
                    'a = ' + ', '.join(f'{a:.2f}' for a in exponents) + ', error on the box of '
                    f'{BOXES[-1]} ' + ', '.join(f'{x:.2e}' for x in largest))
        ok &= slope_check(f'{name} (HF) {NAMES[scheme]} cycle 2, on an infinite box', errors, 4)
    return check(not flags, f'{name} (HF), the boxes: no numerical flag in any run',
                 ', '.join(sorted(flags)) or 'none') and ok


def run():
    warnings.simplefilter('ignore')
    mol = molecule('H 0 0 0; H 0 0 0.74', 'sto-3g', 'H2/STO-3G')
    print(f'-- 1. H2/STO-3G, {GRIDS}')
    ok = check_on_one_box('H2/STO-3G', mol, GRIDS, [('H', 'sg_il', 2), ('H', 'sg_gw', 2), ('H', 'cb_gw', 2),
                                                    ('HF', 'sg_il', 1)])
    print(f'\n-- 2. H2/STO-3G, cycle 2 from (HF) on the boxes {BOXES}')
    ok &= check_box_extrapolated('H2/STO-3G', mol, ('sg_il', 'sg_gw'))
    print('\n-- 3. H2/6-31G and He/cc-pVDZ')
    rows = [('H', 'sg_il', 2), ('H', 'sg_gw', 2), ('HF', 'sg_il', 1)]
    ok &= check_on_one_box('H2/6-31G', molecule('H 0 0 0; H 0 0 0.74', '6-31g', 'H2/6-31G'), GRIDS, rows)
    ok &= check_on_one_box('He/cc-pVDZ', molecule('He 0 0 0', 'cc-pvdz', 'He/cc-pVDZ'), HE_GRIDS, rows)
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_hedin_cycles_two_electrons():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
