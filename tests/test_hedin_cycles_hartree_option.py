"""The Hartree term on the linearized density, run_cycles(..., hartree='linearized') (milestone 3, fast).

The Dyson lines of the cycles keep the start's chemical potential and need not hold the electron count: from
'H' the cycle-1 line of H2/STO-3G holds +0.019 electrons too many, cycle 2's Hartree term carries them, and two
molecules R apart feel each other's as dN / R. A chemical potential cannot remove them (the extra charge sits
in satellite weight of the empty orbitals below mu: inside the gap the count of the line does not move, and a
level shift against the fixed Sigma^(1) reaches 2 electrons only at -0.43 Ha). The option builds the Hartree
term of cycle n >= 2 on the linearized density, the start's density plus its first-order response to
Sigma^(n-1) - Sigma^(0). On H2/STO-3G:

1. The density holds the count: its trace moves by below 1e-9 at beta = 80 / Ha (measured 1e-11, the thermal
   response of the start's levels gives 1.9e-5 at beta = 40), where the line's moves by +0.019 (above 1e-2).
2. Size consistency without any correction: two H2 50 bohr apart (beta = 40 / Ha), SG-I/L from 'H' at cycle 2
   on one molecule equals the single molecule's to below 1e-6 (measured 2.7e-7), against above 1e-4 with the
   line (measured 3.8e-4).
3. Cycle 2 keeps its order of exactness: from 'H' the error against exact diagonalization (the Frobenius norm
   of [G^(2)]^-1 - [G_exact]^-1 at the four lowest frequencies, beta = 72 / Ha, the default box) scales as
   lambda^3 for SG-I/L and SG-Gamma/W (extrapolated slopes 2 s_2 - s_1 measured 3.06 and 3.01).
4. The option changes cycle 2 at lambda^3 from 'H' and lambda^5 from 'HF' (the Frobenius norm of the difference
   of the two Sigma^(2) at the four lowest frequencies, extrapolated slopes measured 3.00 and 5.00), and cycle 1
   not at all.
5. With hartree='line' the source of each Hartree term is the previous line itself, and the refusals: an
   unknown hartree, lag=False with 'linearized' (its tangent is not written), and a backend without
   linearized_line (the 0-D one).

Run: python tests/test_hedin_cycles_hartree_option.py (about 40 s), or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np

import test_hedin_cycles_size_consistency as sc
from test_hedin_cycles_dimer import exact_inverse
from test_hedin_cycles_matsubara import check, h2
from test_hedin_cycles_tier2 import LAMBDAS, NAMES
from src.SingleReference.HedinCycles import run_cycles, zero_d
from src.SingleReference.HedinCycles.matsubara.backend import MatsubaraBackend
from src.SingleReference.HedinCycles.matsubara.grids import Grids

COLD = Grids(beta=80.0, n_fermion=128, n_vertex=8, m_boson=128, m_vertex=8)
EXACT = Grids(beta=72.0, n_fermion=128, n_vertex=16, m_boson=128, m_vertex=16)
SMALL = Grids(beta=20.0, n_fermion=32, n_vertex=8, m_boson=32, m_vertex=4)
#: the extrapolated slope may miss its power by this much (measured within 0.06)
SLOPE_TOL = 0.15
NN = np.arange(4)


def slopes(v):
    s1, s2 = np.log2(v[0] / v[1]), np.log2(v[1] / v[2])
    return s1, s2, 2 * s2 - s1


def check_count():
    alg = MatsubaraBackend(h2(), 'H', COLD)
    line = run_cycles(alg, 'sg_il', 'H', 2).diagnostics
    lin = run_cycles(alg, 'sg_il', 'H', 2, hartree='linearized').diagnostics
    ok = check(abs(lin[2]['hartree_charge']) < 1e-9 and line[2]['hartree_charge'] > 1e-2,
               '(H) the linearized density holds the electron count, the cycle-1 line does not',
               f"{lin[2]['hartree_charge']:+.1e} against {line[2]['hartree_charge']:+.4f}")
    ok &= check(line[2]['hartree_charge'] == line[1]['excess_charge'],
                "(H) with hartree='line' the Hartree term of cycle 2 is built on G^(1) itself",
                f"{line[2]['hartree_charge']:+.6f} and {line[1]['excess_charge']:+.6f}")
    return ok


def check_size_consistency():
    single, pair, A, B, _ = sc.systems(50.0)
    out = {}
    for hartree in ('line', 'linearized'):
        a1, a2 = MatsubaraBackend(single, 'H', sc.GRIDS), MatsubaraBackend(pair, 'H', sc.GRIDS)
        h1, h2_ = run_cycles(a1, 'sg_il', 'H', 2, hartree=hartree), run_cycles(a2, 'sg_il', 'H', 2, hartree=hartree)
        i1 = np.linalg.inv(sc.in_system_basis(a1, h1.lines[2].at(NN)))
        i2 = np.linalg.inv(sc.in_system_basis(a2, h2_.lines[2].at(NN)))
        out[hartree] = np.abs(i2[:, A][:, :, A] - i1).max()
    return check(out['linearized'] < 1e-6 and out['line'] > 1e-4,
                 '(H) SG-I/L cycle 2 of two H2 50 bohr apart equals one H2\'s without correction',
                 f"{out['linearized']:.1e} with the linearized Hartree term, {out['line']:.1e} with the line's")


def check_exactness():
    ok, mol = True, h2()
    for scheme in ('sg_il', 'sg_gw'):
        err = []
        for lam in LAMBDAS:
            alg = MatsubaraBackend(mol.scaled(lam), 'H', EXACT)
            exact = np.array(exact_inverse(alg, 1j * EXACT.nu(NN) + alg.start.mu))
            h = run_cycles(alg, scheme, 'H', 2, hartree='linearized')
            err.append(np.linalg.norm(np.linalg.inv(h.lines[2].at(NN)) - exact))
        s1, s2, p = slopes(err)
        ok &= check(abs(p - 3) < SLOPE_TOL, f'(H) {NAMES[scheme]} cycle 2 with the linearized Hartree term is exact '
                    'through v^2', ', '.join(f'{e:.2e}' for e in err) + f', slopes {s1:.2f}, {s2:.2f}, '
                    f'extrapolated {p:.2f}')
    return ok


def check_change():
    ok, mol = True, h2()
    low = slice(SMALL.n_fermion, SMALL.n_fermion + 4)
    for start, power in (('H', 3), ('HF', 5)):
        d, first = [], 0.0
        for lam in LAMBDAS:
            alg = MatsubaraBackend(mol.scaled(lam), start, SMALL)
            a = run_cycles(alg, 'sg_il', start, 2)
            b = run_cycles(alg, 'sg_il', start, 2, hartree='linearized')
            d.append(np.linalg.norm(b.sigmas[2].at_grid()[low] - a.sigmas[2].at_grid()[low]))
            first = max(first, np.abs(b.sigmas[1].at_grid() - a.sigmas[1].at_grid()).max())
        s1, s2, p = slopes(d)
        ok &= check(abs(p - power) < SLOPE_TOL and first == 0.0,
                    f'({start}) the option changes cycle 2 at lambda^{power} and cycle 1 not at all',
                    ', '.join(f'{x:.2e}' for x in d) + f', slopes {s1:.2f}, {s2:.2f}, extrapolated {p:.2f}; '
                    f'cycle 1 {first:.0e}')
    return ok


def refused(fn, text):
    try:
        fn()
    except (ValueError, TypeError) as err:
        return text in str(err)
    return False


def check_refusals():
    alg = MatsubaraBackend(h2(), 'H', SMALL)
    ok = check(refused(lambda: run_cycles(alg, 'sg_il', 'H', 2, hartree='density'), 'unknown hartree'),
               'an unknown hartree is refused')
    ok &= check(refused(lambda: run_cycles(alg, 'sg_gw', 'H', 2, lag=False, hartree='linearized'), 'lag=False'),
                "lag=False with hartree='linearized' is refused")
    ok &= check(refused(lambda: run_cycles(zero_d.ZeroDBackend(), 'sg_il', 'H', 2, hartree='linearized'),
                        'linearized_line'), "the 0-D backend refuses hartree='linearized'")
    return ok


def run():
    warnings.simplefilter('ignore')
    print('-- 1. the electron count')
    ok = check_count()
    print('\n-- 2. size consistency')
    ok &= check_size_consistency()
    print('\n-- 3. the order of exactness')
    ok &= check_exactness()
    print('\n-- 4. what the option changes')
    ok &= check_change()
    print('\n-- 5. refusals')
    ok &= check_refusals()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_hedin_cycles_hartree_option():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
