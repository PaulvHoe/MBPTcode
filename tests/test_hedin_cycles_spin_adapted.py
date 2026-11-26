"""Spin adaptation of the Hedin cycles' Matsubara backend: a closed shell in spatial orbitals (fast).

TEST_PLAN's Tier 5, spin. Every source of the cycles is a charge density, so for a closed shell every object
of the cycles is the same for both spins (the singlet channel). The spatial form (System.spin = 2,
from_pyscf(..., spin_adapted=True), hubbard_dimer(..., spin_adapted=True)) holds one spin block, with a factor
2 on every closed fermion loop: the Hartree term (Sigma_H, the Hartree part of a static kernel and its
tangent) and the polarizability (and its tangent). The checks:

1. Sigma^(n), n = 0, 1, 2, of every scheme from both starts on H2/STO-3G and the asymmetric Hubbard dimer
   (U = t, site energies 0 and 0.4t): the spatial run equals both spin blocks of the spin-orbital run, taken
   to the system's basis, and nothing couples the spins (measured 2e-16 to 1.3e-12, the dimer from 'H' at
   the start's SCF tolerance, and 0 to 2e-17 between the spins).
2. The same with hartree='linearized' (SG-I/L and SG-Gamma/W, both starts).
3. The diagnostics: the excess charge, the Hartree term's charge and the bubble charge are the same, and the
   spin-orbital start_bse_min (every spin channel) is the smaller of the spatial singlet and triplet ones.
4. The refusals: an odd electron count in spatial orbitals, a spin other than 1 or 2.

The grids are small (beta = 10 / Ha, n_fermion = m_boson = 16, n_vertex = 4, m_vertex = 2): the identities are
exact at any grid. On H2/6-31G at beta = 20 / Ha (n_fermion = m_boson = 32, n_vertex = 8, m_vertex = 4) a run of
SG-Gamma/W takes 0.6 s (H) and 0.9 s (HF) in spatial orbitals against 3.5 s and 6.1 s in spin orbitals.

Run: python tests/test_hedin_cycles_spin_adapted.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from pyscf import gto, scf

from test_hedin_cycles_matsubara import check
from src.SingleReference.HedinCycles import SCHEMES, run_cycles
from src.SingleReference.HedinCycles.matsubara.backend import MatsubaraBackend
from src.SingleReference.HedinCycles.matsubara.grids import Grids
from src.SingleReference.HedinCycles.matsubara.systems import System, from_pyscf, hubbard_dimer

GRIDS = Grids(beta=10.0, n_fermion=16, n_vertex=4, m_boson=16, m_vertex=2)
#: spatial against the spin blocks (measured 1.3e-12 at most), and between the spins (measured 2e-17)
BLOCK_TOL, CROSS_TOL = 1e-11, 1e-14


def cases():
    mf = scf.RHF(gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)).run(conv_tol=1e-12)
    return [('H2/STO-3G', from_pyscf(mf, 'H2/STO-3G'), from_pyscf(mf, 'H2/STO-3G', spin_adapted=True)),
            ('dimer', hubbard_dimer(1.0, 1.0, eps=(0.0, 0.4)), hubbard_dimer(1.0, 1.0, eps=(0.0, 0.4), spin_adapted=True))]


def in_system_basis(alg, X):
    C = alg.start.C
    return np.einsum('pk,...kl,ql->...pq', C, X, C)


def compare(so, sp, scheme, start, hartree='line'):
    """The largest difference between the spatial Sigma^(n) and each spin block of the spin-orbital one, the
    largest element between the spins, n = 0, 1, 2, and both runs."""
    a_so, a_sp = MatsubaraBackend(so, start, GRIDS), MatsubaraBackend(sp, start, GRIDS)
    h_so = run_cycles(a_so, scheme, start, 2, hartree=hartree)
    h_sp = run_cycles(a_sp, scheme, start, 2, hartree=hartree)
    n = sp.norb
    block = cross = 0.0
    for c in range(3):
        X_so = in_system_basis(a_so, h_so.sigmas[c].at_grid())
        X_sp = in_system_basis(a_sp, h_sp.sigmas[c].at_grid())
        block = max(block, np.abs(X_so[:, :n, :n] - X_sp).max(), np.abs(X_so[:, n:, n:] - X_sp).max())
        cross = max(cross, np.abs(X_so[:, :n, n:]).max(), np.abs(X_so[:, n:, :n]).max())
    return block, cross, h_so, h_sp


def check_schemes():
    ok = True
    for name, so, sp in cases():
        for start in ('H', 'HF'):
            rows = [compare(so, sp, scheme, start)[:2] for scheme in SCHEMES]
            ok &= check(max(r[0] for r in rows) < BLOCK_TOL and max(r[1] for r in rows) < CROSS_TOL,
                        f'{name} ({start}): every scheme in spatial orbitals equals both spin blocks, n = 0, 1, 2',
                        ', '.join(f'{r[0]:.0e}' for r in rows) + f' for {SCHEMES}, between spins '
                        f'{max(r[1] for r in rows):.0e}')
    return ok


def check_linearized():
    ok = True
    name, so, sp = cases()[0]
    for start in ('H', 'HF'):
        rows = [compare(so, sp, scheme, start, hartree='linearized')[:2] for scheme in ('sg_il', 'sg_gw')]
        ok &= check(max(r[0] for r in rows) < BLOCK_TOL and max(r[1] for r in rows) < CROSS_TOL,
                    f"{name} ({start}): hartree='linearized' in spatial orbitals equals both spin blocks",
                    ', '.join(f'{r[0]:.0e}' for r in rows))
    return ok


def check_diagnostics():
    ok = True
    name, so, sp = cases()[0]
    for start in ('H', 'HF'):
        _, _, h_so, h_sp = compare(so, sp, 'sg_gw', start)
        d_so, d_sp = h_so.diagnostics, h_sp.diagnostics
        same = max(abs(d_so[n][k] - d_sp[n][k]) for n in (1, 2) for k in ('excess_charge', 'hartree_charge'))
        same = max(same, abs(d_so[2]['bubble_charge'] - d_sp[2]['bubble_charge']))
        channels = [min(s, t) for s, t in zip(d_sp[0]['start_bse_min'], d_sp[0]['start_bse_min_triplet'])]
        bse = max(abs(a - b) for a, b in zip(d_so[0]['start_bse_min'], channels))
        ok &= check(same < 1e-12 and bse < 1e-12,
                    f'{name} ({start}): the charges agree, and the spin-orbital start BSE is the smaller of the '
                    'singlet and triplet', f"charges {same:.0e}, BSE {bse:.0e} (singlet {d_sp[0]['start_bse_min'][0]:.4f}, "
                    f"triplet {d_sp[0]['start_bse_min_triplet'][0]:.4f})")
    return ok


def refused(fn, text):
    try:
        fn()
    except ValueError as err:
        return text in str(err)
    return False


def check_refusals():
    h, B = np.eye(2), np.zeros((1, 2, 2))
    ok = check(refused(lambda: System(h, B, 1, spin=2), 'even number of electrons'),
               'an odd electron count in spatial orbitals is refused')
    ok &= check(refused(lambda: System(h, B, 2, spin=3), 'spin must be 1'), 'a spin other than 1 or 2 is refused')
    return ok


def run():
    warnings.simplefilter('ignore')
    print('-- 1. every scheme, both starts')
    ok = check_schemes()
    print("\n-- 2. hartree='linearized'")
    ok &= check_linearized()
    print('\n-- 3. the diagnostics')
    ok &= check_diagnostics()
    print('\n-- 4. refusals')
    ok &= check_refusals()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_hedin_cycles_spin_adapted():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
