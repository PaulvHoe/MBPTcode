"""The per-cycle diagnostics of the Hedin cycles (TEST_PLAN section 8), CycleHistory.diagnostics (fast).

The diagnostics are reported and not asserted by the runs. This file checks what they report, against
independent numbers, on H2/STO-3G (beta = 20 / Ha, n_fermion = m_boson = 32, n_vertex = 8, m_vertex = 4):

1. One dict per line, with the keys of MatsubaraBackend.cycle_diagnostics for each scheme and cycle, and empty
   dicts from the 0-D backend.
2. The excess charge of the cycle-1 line: none from 'HF' (7.8e-10 here, the thermal occupation of its levels at
   beta = 20, below 1e-8) and +0.042 from 'H' (above 1e-2), the particle number the Dyson lines do not conserve
   (tests/test_hedin_cycles_size_consistency.py).
3. The start's Gamma/W BSE from 'HF': start_bse_min equals 1 - lambda_max of K chi(0) as the dimer test
   computes it (largest_static_eigenvalue), to 1e-12. From 'H' its kernel vanishes and the value is 1.
4. SG-Gamma/W's chain-rule factor: exactly 1 at cycle 2 with the lag (Lambda_H vanishes there) and not 1
   without it.
5. The charge SG-Gamma/W's cycle-2 bubble induces: above 1e-3 from both starts (measured 1.5e-2 from 'H' and
   9.2e-3 from 'HF' here, 1.1e-2 and 4.0e-3 at beta = 40 in the size-consistency test), where the start's
   bubble induces none.
6. The change of Sigma is positive and every Dyson denominator keeps its distance from singular in (0, 2).

Run: python tests/test_hedin_cycles_diagnostics.py (2 s), or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from test_hedin_cycles_dimer import largest_static_eigenvalue
from test_hedin_cycles_matsubara import check, h2
from src.SingleReference.HedinCycles import zero_d
from src.SingleReference.HedinCycles.matsubara.backend import MatsubaraBackend
from src.SingleReference.HedinCycles.matsubara.grids import Grids
from src.SingleReference.HedinCycles.schemes import run_cycles

GRIDS = Grids(beta=20.0, n_fermion=32, n_vertex=8, m_boson=32, m_vertex=4)
KEYS = {0: {'excess_charge', 'start_bse_min'}, 1: {'excess_charge', 'hartree_charge', 'sigma_change', 'dyson_min'}}
SG_GW_KEYS = KEYS[1] | {'screening_min', 'chain_min', 'bubble_charge'}


def check_keys(runs):
    ok = True
    for (start, scheme), hist in runs.items():
        expected = [KEYS[0], KEYS[1], SG_GW_KEYS if scheme == 'sg_gw' else KEYS[1]]
        got = [set(d) for d in hist.diagnostics]
        ok &= check(got == expected, f'({start}) {scheme}: one dict of diagnostics per line, with its keys',
                    f'{[sorted(k) for k in got]}')
    hist = run_cycles(zero_d.ZeroDBackend(), 'sg_gw', 'H', 2)
    ok &= check(hist.diagnostics == [{}, {}, {}], 'the 0-D backend reports empty dicts', f'{hist.diagnostics}')
    return ok


def check_values(runs, algs):
    ok = True
    hf, h = runs['HF', 'sg_gw'].diagnostics, runs['H', 'sg_gw'].diagnostics
    ok &= check(abs(hf[1]['excess_charge']) < 1e-8 and h[1]['excess_charge'] > 1e-2,
                'the excess charge of the cycle-1 line: none from HF, the Dyson line\'s from H',
                f"{hf[1]['excess_charge']:+.1e} and {h[1]['excess_charge']:+.4f}")
    ref = 1 - largest_static_eigenvalue(algs['HF'])
    ok &= check(abs(hf[0]['start_bse_min'][1] - ref) < 1e-12 and h[0]['start_bse_min'][1] == 1.0,
                'the start\'s Gamma/W BSE: 1 - lambda_max of K chi(0) from HF, 1 from H',
                f"{hf[0]['start_bse_min'][1]:.12f} against {ref:.12f}")
    unlagged = run_cycles(algs['HF'], 'sg_gw', 'HF', 2, lag=False).diagnostics[2]
    ok &= check(hf[2]['chain_min'] == 1.0 and h[2]['chain_min'] == 1.0 and unlagged['chain_min'] != 1.0,
                'SG-Gamma/W\'s chain-rule factor is 1 at cycle 2 with the lag, not without it',
                f"{hf[2]['chain_min']}, {h[2]['chain_min']}, unlagged {unlagged['chain_min']:.6f}")
    ok &= check(min(hf[2]['bubble_charge'], h[2]['bubble_charge']) > 1e-3,
                'SG-Gamma/W\'s cycle-2 bubble induces a charge', f"{h[2]['bubble_charge']:.1e} (H), "
                f"{hf[2]['bubble_charge']:.1e} (HF)")
    worst = [(d['sigma_change'], d['dyson_min']) for hist in runs.values() for d in hist.diagnostics[1:]]
    ok &= check(all(s > 0 and 0 < m < 2 for s, m in worst),
                'the change of Sigma is positive, every Dyson denominator is away from singular',
                f'dyson_min from {min(m for _, m in worst):.3f} to {max(m for _, m in worst):.3f}')
    return ok


def run():
    warnings.simplefilter('ignore')
    algs = {start: MatsubaraBackend(h2(), start, GRIDS) for start in ('H', 'HF')}
    runs = {(start, scheme): run_cycles(algs[start], scheme, start, 2)
            for start in ('H', 'HF') for scheme in ('sg_il', 'sg_gw')}
    print('-- the keys')
    ok = check_keys(runs)
    print('\n-- the values')
    ok &= check_values(runs, algs)
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_hedin_cycles_diagnostics():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
