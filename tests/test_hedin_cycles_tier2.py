"""TEST_PLAN's Tier 2 on small molecules: the order of exactness by coupling scaling (milestone 3, slow).

The two-electron integrals are scaled, v -> lambda v (System.scaled), and the start is rerun at each lambda =
0.1, 0.05 and 0.025. ||X|| is the Frobenius norm of X over the four lowest Matsubara frequencies and all spin
orbitals. (The largest element is not smooth in lambda once the element that holds it changes: on H2/6-31G the
error of cycle 1 against exact diagonalization moves from (6,6) to (0,0) between lambda = 0.05 and 0.025, which
bends its slopes to 2.10 and 1.98.) s_1 and s_2 are the slopes of log ||X|| against log lambda between 0.1 and
0.05 and between 0.05 and 0.025. If ||X|| = C lambda^p (1 + a lambda + O(lambda^2)), then
s_i = p + a lambda_i / (2 ln 2) + O(lambda^2), with lambda_i the larger lambda of the pair, so that the
extrapolated slope 2 s_2 - s_1 equals p + O(lambda^2). No reference is needed:

1. Successive cycles: ||Sigma^(n) - Sigma^(n-1)|| scales as lambda^n from (H) and as lambda^(n+1) from (HF), for
   n = 1, 2 and every scheme, where Sigma^(0) is the start's static self-energy.
2. Scheme differences at cycle 2: from (H) every scheme's cycle 2 is exact through v^2, so two schemes differ
   from v^3 on and their difference scales as lambda^3. From (HF) every cycle 2 is exact through v^3 (the
   dimer's Tier 4), so the difference scales as lambda^4. The two cycle-based schemes are equal at cycle 2
   (Tier 1) and are not paired.

Successive cycles are read from 2 s_2 - s_1. Scheme differences are read from s_2 alone, with a bar that still
separates their power from the one below: the two routes' cut enters a difference one order early. From (H)
SG-I/L minus the cycle-based schemes has slopes 2.84 and 2.71, a lambda^2 part 1.6e-3 of the cycle's own
lambda^2 change beside a lambda^3 coefficient 9.2e-3, which bends 2 s_2 - s_1 to 2.59.

H2/STO-3G runs the four schemes, and H2/6-31G and He/cc-pVDZ (4 and 5 orbitals) the SG schemes: a cycle-based
run takes 1 to 2 minutes there, and those runs were made on the cluster. Every molecule runs in spatial orbitals
(System.spin = 2), which equal the spin-orbital runs to 1e-12 (tests/test_hedin_cycles_spin_adapted.py). The grids are fixed for each
molecule (beta = 20 / Ha, n_fermion = m_boson = 32, n_vertex = 8, m_vertex = 4), so lambda changes only the
integrals and the start line. Every run must end without a numerical flag.

Measured on one thread, with the Frobenius norm, in spatial orbitals (about a minute in all): on H2/STO-3G,
H2/6-31G and He/cc-pVDZ every successive cycle within 0.01 of its power (2 s_2 - s_1), the scheme differences'
s_2 from 2.71 to 2.96 (H) and from 3.93 to 4.01 (HF), except one row. The (HF) difference of the SG schemes on He
carries a lambda^3 part (slopes 3.93 and 3.88, extrapolated 3.83), where the 0-D counts of both schemes agree
with the exact ones through v^3. That row is reported, not asserted. The dynamic part of the cycle-2 vertex is
second order, so the transfer box cuts a lambda^3 term from both schemes (measured on H2/STO-3G in
tests/test_hedin_cycles_two_electrons.py), and He's box of 4 transfers reaches 1.26 Ha, below its lowest
excitation at 1.96 Ha. On the cluster (largest element) boxes of 4, 8 and 12 (m_boson eight times the box) took
the extrapolated slope from 3.72 to 3.91 and 3.96.

Run: python tests/test_hedin_cycles_tier2.py (about a minute), or under pytest.
"""
import itertools
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from pyscf import gto, scf

from test_hedin_cycles_matsubara import check
from src.SingleReference.HedinCycles.matsubara.backend import MatsubaraBackend
from src.SingleReference.HedinCycles.matsubara.grids import Grids
from src.SingleReference.HedinCycles.matsubara.systems import from_pyscf
from src.SingleReference.HedinCycles.schemes import SCHEMES, run_cycles

#: the grids of the runs, and the coupling scales
GRIDS = Grids(beta=20.0, n_fermion=32, n_vertex=8, m_boson=32, m_vertex=4)
LAMBDAS = (0.1, 0.05, 0.025)
#: successive cycles: the extrapolated slope 2 s_2 - s_1 may miss its power by this much (measured within 0.01
#: on H2/STO-3G, and the bar is the old one, three times the largest of the earlier largest-element norm)
SLOPE_TOL = 0.12
#: scheme differences: the plain slope s_2 may miss its power by this much, which still separates it from the
#: power below. The routes' cut enters them one order early (from (H) at lambda^2 with 1.6e-3 of the cycle's
#: own lambda^2 change on these grids), so their extrapolated slope is not taken (measured s_2 within 0.29 on
#: H2/STO-3G)
PAIR_SLOPE_TOL = 0.35
NAMES = {'sg_il': 'SG-I/L', 'sg_gw': 'SG-Gamma/W', 'cb_gw': 'cycle-based Gamma/W', 'cb_il': 'cycle-based I/L'}


def molecule(atom, basis, name):
    """A closed-shell molecule in spatial orbitals (System.spin = 2, equal to the spin-orbital runs to 1e-12)."""
    return from_pyscf(scf.RHF(gto.M(atom=atom, basis=basis, verbose=0)).run(conv_tol=1e-12), name, spin_adapted=True)


#: (name, system, schemes, rows reported and not asserted, as (start, key))
CASES = (('H2/STO-3G', lambda: molecule('H 0 0 0; H 0 0 0.74', 'sto-3g', 'H2/STO-3G'), SCHEMES, ()),
         ('H2/6-31G', lambda: molecule('H 0 0 0; H 0 0 0.74', '6-31g', 'H2/6-31G'), ('sg_il', 'sg_gw'), ()),
         ('He/cc-pVDZ', lambda: molecule('He 0 0 0', 'cc-pvdz', 'He/cc-pVDZ'), ('sg_il', 'sg_gw'),
          (('HF', ('sg_il', 'sg_gw')),)))


def scaling_norms(system, start, schemes, grids=GRIDS, lambdas=LAMBDAS):
    """The norms at each lambda, {(scheme, n): ||Sigma^(n) - Sigma^(n-1)||} for n = 1, 2 and
    {(a, b): ||Sigma_a^(2) - Sigma_b^(2)||} for the pairs of schemes, and the numerical flags of every run."""
    low = slice(grids.n_fermion, grids.n_fermion + 4)
    norms, flags = {}, set()
    for lam in lambdas:
        alg = MatsubaraBackend(system.scaled(lam), start, grids)
        sig = {}
        for s in schemes:
            hist = run_cycles(alg, s, start, 2)
            flags.update(hist.numerical_flags)
            sig[s] = [x.at_grid()[low] for x in hist.sigmas]
        for s in schemes:
            for n in (1, 2):
                norms.setdefault((s, n), []).append(np.linalg.norm(sig[s][n] - sig[s][n - 1]))
        for a, b in itertools.combinations(schemes, 2):
            if {a, b} != {'cb_gw', 'cb_il'}:
                norms.setdefault((a, b), []).append(np.linalg.norm(sig[a][2] - sig[b][2]))
    return norms, sorted(flags)


def expected_power(key, start):
    """lambda^n from (H) and lambda^(n+1) from (HF) for successive cycles, lambda^3 and lambda^4 for pairs."""
    return (key[1] if isinstance(key[1], int) else 3) + (start == 'HF')


def check_scaling(name, system, start, schemes, reported=()):
    norms, flags = scaling_norms(system, start, schemes)
    ok = check(not flags, f'{name} ({start}): no numerical flag in any run', ', '.join(flags) or 'none')
    for key, v in norms.items():
        p = expected_power(key, start)
        s1, s2 = (np.log2(v[i] / v[i + 1]) for i in (0, 1))
        what = (f'{NAMES[key[0]]}, ||Sigma^({key[1]}) - Sigma^({key[1] - 1})||' if isinstance(key[1], int)
                else f'cycle 2, {NAMES[key[0]]} - {NAMES[key[1]]}')
        detail = ', '.join(f'{x:.2e}' for x in v) + f', slopes {s1:.2f}, {s2:.2f}, extrapolated {2 * s2 - s1:.2f}'
        if (start, key) in reported:
            print(f'  [report] {name} ({start}) {what}, lambda^{p} expected   ({detail})')
            continue
        good = abs(2 * s2 - s1 - p) < SLOPE_TOL if isinstance(key[1], int) else abs(s2 - p) < PAIR_SLOPE_TOL
        ok &= check(good, f'{name} ({start}) {what} scales as lambda^{p}', detail)
    return ok


def run():
    warnings.simplefilter('ignore')
    ok = True
    print(f'grids: {GRIDS}')
    for name, system, schemes, reported in CASES:
        print(f'-- {name}')
        mol = system()
        for start in ('H', 'HF'):
            ok &= check_scaling(name, mol, start, schemes, reported)
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_hedin_cycles_tier2():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
