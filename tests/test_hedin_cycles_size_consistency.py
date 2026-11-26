"""Size consistency of the Hedin cycles: two H2/STO-3G molecules 50 bohr apart against one (milestone 3, slow).

TEST_PLAN's Tier 5. The pair is built in the basis of each molecule's own RHF orbitals (orthogonal to 3e-16 at
this distance, so that A's and B's spin orbitals are the system's), and [G^(n)]^-1 of the pair, taken back to
that basis from the start's orbitals, is compared with the single molecule's at the four lowest frequencies.

1. Between the molecules [G^(n)]^-1 vanishes (n = 0, 1, 2, both starts, SG-I/L and SG-Gamma/W).
2. On A it equals the single molecule's once the Hartree potential of B's density change is taken off:
   [G^(n)]^-1_AA(pair) - [G^(n)]^-1(single) = -J_AA[rho_B^(n-1) - rho_B^(0)], with rho^(n-1) the density of the
   line the Hartree term of cycle n is built on and J[D]_pq = sum_P B^P_pq tr(B^P D). Asserted for every cycle
   of SG-I/L and for cycles 0 and 1 of SG-Gamma/W (cycle 1 is common), both starts.

Three terms that break size consistency, two of them asserted as measured and one reported:
- The Dyson lines of the cycles keep the start's chemical potential and do not conserve the particle number:
  from (H) the cycle-1 line of H2/STO-3G holds 2.0189 electrons at beta = 40 / Ha (2.0186 at 80, converged in
  the grids, and 2.0422 at 20, where its lowest empty level, 0.2 Ha above the start's chemical potential, is
  thermally occupied). From (HF) it holds 2.0000. The Hartree term of cycle 2 then feels the other molecule's
  excess charge dN as a uniform potential dN / R on A (-3.8e-4 at 50 bohr): from (H), cycle 2 of every scheme
  is block-diagonal but size consistent only after this term, which grows with the number of molecules. The
  excess charge is reported.
- The cycle-2 polarizability of SG-Gamma/W, P^(2) = G^(1) G^(1) Gamma^(2), does not conserve charge: the charge
  that a source Q induces, tr(G Gamma^Q G) at Omega != 0, is zero for the start's bubble but 1.1e-2 (H) and
  4.0e-3 (HF) for SG-Gamma/W's (4.1e-2 and 2.1e-2 with the bare vertex on G^(1)). Each molecule's charge then
  fluctuates in W^(2), and the monopole couplings 1/R at both ends give A a term of order 1/R^2: 1.69e-6 (H)
  and 1.08e-6 (HF) at 50 bohr, 4.22e-7 and 2.71e-7 at 100 bohr (exponents 2.00). Reported with its exponent,
  and the induced charges asserted (the start's zero, SG-Gamma/W's nonzero). SG-I/L keeps the start's L^(1)
  and has no such term.
- At finite temperature each molecule's electron number fluctuates (the grand-canonical f (1 - f) of every
  level) and couples the same way, as 1/R^2: at beta = 20 / Ha (gap 0.76 Ha from (H)) it reaches 1.9e-6 at
  50 bohr. beta = 40 / Ha takes it below 1e-9.

Grids: beta = 40 / Ha, n_fermion = m_boson = 64, n_vertex = m_vertex = 8. The cycle-based schemes run on the
pair in over 90 s and are left to the cluster. Measured on one thread (about a minute): between the molecules
2e-16 (H) and 1e-17 (HF), on A after the Hartree term 1.0e-9, 2.1e-9, 2.7e-9 (H) and 1.0e-9, 9.8e-10,
9.7e-10 (HF) for SG-I/L's cycles 0, 1, 2.

Run: python tests/test_hedin_cycles_size_consistency.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import scipy.linalg
from pyscf import gto, scf

from test_hedin_cycles_matsubara import check, h2
from test_hedin_cycles_tier2 import NAMES
from src.SingleReference.HedinCycles.matsubara.backend import Legs, MatsubaraBackend, ThreePoint
from src.SingleReference.HedinCycles.matsubara.grids import Grids
from src.SingleReference.HedinCycles.matsubara.starts import hartree
from src.SingleReference.HedinCycles.matsubara.systems import from_pyscf
from src.SingleReference.HedinCycles.schemes import run_cycles

GRIDS = Grids(beta=40.0, n_fermion=64, n_vertex=8, m_boson=64, m_vertex=8)
#: the distances of the two molecules (bohr) and the bond length (bohr, 0.74 A)
DISTANCES = (50.0, 100.0)
BOND = 0.74 / 0.52917721092
#: between the molecules (measured 2e-16), and on A after the Hartree potential of B's density change
#: (measured 2.7e-9 at most, and the bar is three times that)
OFF_TOL = 1e-12
BLOCK_TOL = 1e-8
#: the induced charge of a bubble at Omega != 0: zero for the start's, and SG-Gamma/W's above this
CHARGE_ZERO, CHARGE_FLOOR = 1e-12, 1e-3
NN = np.arange(4)


def systems(distance):
    """The single molecule and the pair, in the basis of each molecule's own RHF orbitals, the pair's spin
    orbitals on A and on B (each spin block holds A's orbitals, then B's), and the orbitals' overlap error."""
    mf = scf.RHF(gto.M(atom=[('H', (0, 0, 0)), ('H', (0, 0, BOND))], basis='sto-3g', unit='Bohr', verbose=0))
    mf.run(conv_tol=1e-12)
    mol = gto.M(atom=[('H', (0, 0, 0)), ('H', (0, 0, BOND)), ('H', (distance, 0, 0)), ('H', (distance, 0, BOND))],
                basis='sto-3g', unit='Bohr', verbose=0)
    pair = scf.RHF(mol)
    pair.mo_coeff = scipy.linalg.block_diag(mf.mo_coeff, mf.mo_coeff)
    overlap = np.abs(pair.mo_coeff.T @ mol.intor('int1e_ovlp') @ pair.mo_coeff - np.eye(4)).max()
    n = mf.mo_coeff.shape[1]
    A = [*range(n), *range(2 * n, 3 * n)]
    B = [*range(n, 2 * n), *range(3 * n, 4 * n)]
    return from_pyscf(mf, 'H2/STO-3G'), from_pyscf(pair, 'two H2/STO-3G'), A, B, overlap


def in_system_basis(alg, X):
    """A matrix (or a stack of them) in the start's orbitals taken to the system's basis."""
    C = alg.start.C
    return np.einsum('pk,...kl,ql->...pq', C, X, C)


def compare(single, pair, A, B, scheme, start):
    """For n = 0, 1, 2: the largest element between the molecules, the largest on A after the Hartree potential
    of B's density change, and the single molecule's excess charge; the flags of both runs."""
    a1, a2 = MatsubaraBackend(single, start, GRIDS), MatsubaraBackend(pair, start, GRIDS)
    h1, h2 = run_cycles(a1, scheme, start, 2), run_cycles(a2, scheme, start, 2)
    rho = [in_system_basis(a2, line.density()) for line in h2.lines]
    rows = []
    for n in range(3):
        i1 = np.linalg.inv(in_system_basis(a1, h1.lines[n].at(NN)))
        i2 = np.linalg.inv(in_system_basis(a2, h2.lines[n].at(NN)))
        shift = np.zeros_like(rho[0])
        if n == 2:
            dB = np.zeros_like(rho[0])
            dB[np.ix_(B, B)] = (rho[1] - rho[0])[np.ix_(B, B)]
            shift = hartree(pair.B, dB)
        rows.append((np.abs(i2[:, A][:, :, B]).max(),
                     np.abs(i2[:, A][:, :, A] - i1 + shift[np.ix_(A, A)][None]).max(),
                     np.trace(h1.lines[n].density()).real - single.nelec))
    return rows, set(h1.numerical_flags) | set(h2.numerical_flags)


def check_start(start):
    ok = True
    single, pair, A, B, overlap = systems(DISTANCES[0])
    ok &= check(overlap < 1e-12, f'the two molecules\' orbitals are orthonormal at {DISTANCES[0]:g} bohr',
                f'{overlap:.1e}')
    rows, flags = compare(single, pair, A, B, 'sg_il', start)
    ok &= check(max(r[0] for r in rows) < OFF_TOL and max(r[1] for r in rows) < BLOCK_TOL and not flags,
                f'({start}) SG-I/L, n = 0, 1, 2: nothing between the molecules, and on A the single molecule\'s '
                'after the Hartree potential of B\'s density change',
                ', '.join(f'{r[0]:.0e}' for r in rows) + ' between, ' + ', '.join(f'{r[1]:.1e}' for r in rows) + ' on A')
    print(f'  [report] ({start}) excess charge of the single molecule\'s G^(0), G^(1), G^(2): '
          + ', '.join(f'{r[2]:+.5f}' for r in rows) + f', its potential on the other molecule in cycle 2, '
          f'-dN / R at {DISTANCES[0]:g} bohr: {-rows[1][2] / DISTANCES[0]:+.2e}')
    residual = []
    for distance in DISTANCES:
        single, pair, A, B, _ = systems(distance)
        rows, flags = compare(single, pair, A, B, 'sg_gw', start)
        if distance == DISTANCES[0]:
            ok &= check(max(r[0] for r in rows) < OFF_TOL and max(r[1] for r in rows[:2]) < BLOCK_TOL and not flags,
                        f'({start}) SG-Gamma/W: nothing between the molecules (n = 0, 1, 2), on A the single '
                        'molecule\'s at n = 0, 1', ', '.join(f'{r[0]:.0e}' for r in rows) + ' between, '
                        + ', '.join(f'{r[1]:.1e}' for r in rows[:2]) + ' on A')
        residual.append(rows[2][1])
    exponent = np.log(residual[0] / residual[1]) / np.log(DISTANCES[1] / DISTANCES[0])
    print(f'  [report] ({start}) SG-Gamma/W cycle 2 on A, after the Hartree term: '
          + ', '.join(f'{x:.2e}' for x in residual) + f' at {DISTANCES} bohr, as R^-{exponent:.2f}')
    return ok


def check_induced_charge():
    """tr(G Gamma^Q G) at Omega_m != 0 on the single molecule: zero for the start's bubble, not for SG-Gamma/W's
    cycle-2 bubble."""
    ok = True
    for start in ('H', 'HF'):
        alg = MatsubaraBackend(h2(), start, GRIDS)
        hist = run_cycles(alg, 'sg_gw', start, 2)

        def charge(line, vertex):
            legs = Legs(line, vertex)
            return max(np.abs(np.trace(legs.pair_density(m), axis1=-2, axis2=-1)).max() for m in (1, 2, 4, 8))

        start_bubble = charge(hist.lines[0], ThreePoint(alg.ctx, True))
        cycle_two = charge(hist.lines[1], hist.vertices[2])
        bare_one = charge(hist.lines[1], ThreePoint(alg.ctx, True))
        ok &= check(start_bubble < CHARGE_ZERO and cycle_two > CHARGE_FLOOR,
                    f'({start}) the start\'s bubble conserves charge, SG-Gamma/W\'s cycle-2 bubble does not',
                    f'induced charge {start_bubble:.0e} and {cycle_two:.1e} ({bare_one:.1e} with the bare vertex '
                    'on G^(1))')
    return ok


def run():
    warnings.simplefilter('ignore')
    ok = True
    for start in ('H', 'HF'):
        print(f'-- the {start} start')
        ok &= check_start(start)
    print('\n-- the induced charge of the bubbles (single molecule)')
    ok &= check_induced_charge()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_hedin_cycles_size_consistency():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
