"""The exact FCI vertex split of G0W quasiparticle energies (`fci_vertex_split`,
src/SingleReference/GW/fci_vertex_split.py): what is held.

1. The FCI rung is the exact W_c: at full Lanczos depth its Sigma^c_pp(w) equals the dense Lehmann
   sum over every FCI state (pyscf direct_spin1 with nroots = every determinant, trans_rdm1, as
   tests/fci_screened_gw.py) at five frequencies LEHMANN_GAP or more from any pole, to
   CROSS_PATH_RTOL (He/cc-pVDZ 4.0e-16, H2/6-31G** 2.3e-16 relative measured); its quasiparticle
   root equals `fci_screened_g0w_ip(mf, eta=0)` to IP_ORACLE_BAR, its linearized solution
   e_p + Z(e_p) Sigma(e_p) the dense poles' to CROSS_PATH_RTOL (3.0e-16 measured), and its lowest
   excitation the lowest state the start vectors couple to, to OMEGA_BAR. At the default depth
   that lowest excitation is the lowest coupled singlet although rounding seeds a lower triplet
   into the Lanczos runs: H2/cc-pVDZ 13.9137 eV (triplet 10.6757) and LiH/cc-pVDZ at 3.0 A
   1.5689 eV (triplet 0.7921), against the dense sums.
2. The RPA rung is the house G0W0@HF: on H2O/6-31G (full space, ncore = 0) the HOMO and HOMO-1
   roots equal `QPqb.solve_diag`'s to CROSS_PATH_RTOL with the same Z (5.6e-16 relative), and
   `calc_qp_energy` (df=False, eta=0) to HOUSE_SCAN_BAR (2.4e-8 eV measured); through the driver
   on H2O/STO-3G (ncore = 0) its three roots and linearized solutions equal those of QPqb's poles
   to CROSS_PATH_RTOL.
3. The TDHF rung: its excitation energies are pyscf's TDHF singlets (nstates = every pair) to the
   class bar of a root (2e-10 Ha; 4.8e-14 Ha measured), its residues obey the sum rules
   sum_k Omega_k (w^k_pq)^2 = 2 v (A - B) v and sum_k (w^k_pq)^2 / Omega_k = 2 v (A + B)^-1 v
   (v_ia = (pq|ia)) to CROSS_PATH_RTOL for RPA and TDHF (4.4e-16 measured), and its roots equal
   calc_qp_energy(selfenergy='GW@TDHF', polarizability='TDHF') to HOUSE_SCAN_BAR (8.9e-9 eV).
4. The static-BSE rung: its excitation energies equal `solve_bse` (dense, full integrals, qp=None,
   singlet) to the root class bar (9.8e-14 Ha measured) and those of an independent einsum (A, B)
   built from the four-index W_RPA(0) (7.6e-14 Ha), and its roots equal
   calc_qp_energy(selfenergy='GW@BSE', polarizability='BSE') to HOUSE_SCAN_BAR (7.8e-8 eV).
5. The exact hole against a dense diagonalization of the (m - 1, m) sector: on H2O/STO-3G
   (ncore = 0, 735 determinants; HOMO, HOMO-1, HOMO-2) and N2/STO-3G's inner-valence 2sigma_g
   (ncore = 2, 3920 determinants) the main pole is the state of largest |<s|a_p,alpha|Psi0>|^2
   (degenerate states summed), its IP to HOLE_BAR and its weight to HOLE_Z_BAR, with a residual
   bound below FCI_SPLIT_HOLE_RESIDUAL_BAR and weights summing to D_pp / 2 to WEIGHT_SUM_BAR.
   N2's 2sigma_g: the main pole 39.6451 eV (Z 0.5629) with the second 42.1253 eV (0.2631) within
   its residual bound of the dense second state, the window weight below the ambiguity bar; its
   G0W_FCI root (41.9550 eV, Z 0.4244) is no quasiparticle and flagged
   ('fci_split_rung_no_quasiparticle'), the other rungs' roots are, and the FCI rung's
   conservative half-depth check flags it ('fci_split_depth_unconverged', 4.2e-5 Ha from depth
   20 to 40, with the depth-40 root 1.5e-12 eV from depth 60). Its depth-20 root is the 20-node
   Gauss quadrature, a Lanczos with full reorthogonalization kept off Psi0 (written out here, from
   the module's start vectors and operator), to GAUSS_BAR: Q H Q holds Psi0 at eigenvalue 0, above
   the whole spectrum, and a recurrence that let rounding seed it misses by 2.2e-3 eV.
6. The pilot's numbers (independent code; tests/fci_vertex_split_references.py, full precision
   in Hartree with its SOURCES): HF/6-31G and LiH/cc-pVDZ, ncore = 1, HOMO: IP_exact, G0W0,
   G0W_TDHF, G0W_FCI and its Z, the lowest FCI-rung excitation, the W and Sigma sides,
   dHx_lin_scr and its IP, and the oracle density IPs IP[G0W0] - dH and IP[G0W0] - dHx, each to
   its class bar (PILOT_HF_BAR, PILOT_FCI_BAR, PILOT_ORACLE_BAR, PILOT_Z_BAR); the split adds up to
   minus the G0W0 error. The density side against independent code: dD_lin is the house's
   unrelaxed GW density (`GWDensityMatrixSolver`, RPA) on H2O/6-31G to CROSS_PATH_RTOL, and T_c
   is the pilot's screened_hartree_shift (re-typed below) to CROSS_PATH_RTOL, on HF/6-31G and LiH
   (ncore = 1).
7. Refusals before any work (ValueError naming the argument): a UHF, ROHF, Kohn-Sham,
   density-fitted, solvated, environment-carrying or unconverged mean field (a gas phase marked by
   `NoEnvironment` is accepted), an open shell, a non-aufbau occupation, a basis without virtuals;
   ncore, orbitals and depth outside their ranges, an ncore that splits a degenerate occupied level
   (HF/6-31G's pi pair); an active space past the memory this job may use. An FCI rung whose every
   start vector falls below ACTIVE_SPACE_VERTEX_TOL (the bar raised to 1e10) has no pole: its root
   is e_p with Z = 1, every start counted as skipped, no lowest excitation (None) and no flag.
   Flags, each with its `FCISplitShort` warning and its entry in numerical_flags: depth 4, a
   hole depth of 4 and a zero residual bar ('fci_split_depth_unconverged'), one Davidson cycle
   ('fci_split_ground_not_converged'), the ambiguity bars ('fci_split_main_pole_ambiguous'),
   QP_WINDOW_Z_MIN at 0.99 (that and 'fci_split_rung_no_quasiparticle'), a Fock and a spin bar
   below the measured misses ('fci_split_fock_not_diagonal', 'fci_split_ground_not_singlet'); and
   three real instabilities ('fci_split_rung_unstable', the rung NaN, the others finite):
   MgO/STO-3G (its TDHF A - B indefinite, as MgO/6-31G's), N2/STO-3G at 1.6 A (TDHF and BSE, and
   the TDHF A + B, so dhx_lin_scr is NaN too; its RHF pinned to the default guess's solution) and
   H2/6-31G at 2.5 A (static BSE).
8. Nothing existing changes: the mean field's arrays are bitwise their stored copies after a run;
   the default orbitals are the HOMO and the next level below its degenerate set (HF/6-31G: the
   3 sigma, not the pi partner; NH3/6-31G: the e pair's upper orbital); the record survives a
   JSON round trip.

Run: python tests/test_fci_vertex_split.py, or under pytest.
"""
import functools
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import dft, fci, gto, scf, tdscf

from src.Base.environment_seam import NoEnvironment, attach_environment
from src.Base.numerics.constants import FCI_CONV_TOL, HARTREE_TO_EV
from src.Base.numerics.numerics import Numerics
from src.Base.utils.parallel.memory import allocation_max_memory_mb
import src.SingleReference.GW.fci_vertex_split as FVS
from src.SingleReference.GW.fci_vertex_split import FCISplitShort, fci_vertex_split
from src.SingleReference.BSE.solve import solve_bse
from src.SingleReference.common.eri_blocks import MOEriBlocks
from src.SingleReference.DensityMatrix.density_matrix import GWDensityMatrixSolver
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.quasi_boson import QPqb
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import (LinearResponseSolver,
                                                                static_screened_coulomb_chemist)
from src.Solvers.qp_equation import solve_qp_equation_poles
from src.environments.environment import PointCharges
from tests.fci_screened_gw import fci_screened_g0w_ip
from tests.fci_vertex_split_references import GEOMETRIES, PILOT
from tests.pins import CROSS_PATH_RTOL, assert_within_class, bitwise_arrays
from tests.reference_scf import rhf_conv_1e12

#: eV. The FCI rung's root against `fci_screened_g0w_ip`, whose brentq stops at its default
#: xtol (2e-12 Ha): 1.1e-14 eV measured (He/cc-pVDZ), 0 (H2/6-31G**).
IP_ORACLE_BAR = 1e-8
#: Ha. The Lehmann frequencies' distance from every pole, Lanczos and dense: 0.0915 Ha measured
#: at the closest (H2/6-31G**; 1.18 Ha for He/cc-pVDZ).
LEHMANN_GAP = 0.05
#: eV. The FCI rung's lowest excitation against the dense lowest coupled state, a converged Ritz
#: value against an eigenvalue: 2.1e-13 eV measured at worst (He/cc-pVDZ at full depth; 0 for
#: H2/cc-pVDZ and stretched LiH at the default depth).
OMEGA_BAR = 1e-9
#: eV. A rung's root against `calc_qp_energy`'s Casida route, whose graphical scan stops at
#: QP_GRAPHICAL_TOL = 1e-8 Ha (2.7e-7 eV): 2.4e-8 (GW), 8.9e-9 (GW@TDHF), 7.8e-8 eV (GW@BSE)
#: measured, H2O/6-31G HOMO and HOMO-1.
HOUSE_SCAN_BAR = 1e-6
#: eV. The exact main pole against the dense cation: 1.1e-10 eV measured at worst (N2/STO-3G's
#: 2sigma_g, its residual bound 2.4e-6 Ha; 2.7e-12 eV for H2O/STO-3G).
HOLE_BAR = 1e-9
#: The exact main pole's weight against the dense overlap: 7.4e-11 measured at worst
#: (N2/STO-3G's 2sigma_g; 4.1e-15 for H2O/STO-3G).
HOLE_Z_BAR = 1e-8
#: Ha. Cation states closer than this are one degenerate level, their weights summed.
DEGENERATE_BAR = 1e-8
#: The hole's weights against D_pp / 2, an identity of the Lanczos quadrature (1.0e-15 measured),
#: the trace of dD_lin (1.5e-17), and Z = 1 of an FCI rung without a pole (0).
WEIGHT_SUM_BAR = 1e-12
#: eV. The W and Sigma sides adding up to minus the G0W0 error, IP[G0W0] - T_c, and an FCI rung
#: without a pole at -e_p, identities: 0 measured.
SPLIT_SUM_BAR = 1e-12
#: eV. The pilot's numbers that read the HF reference alone (IP_G0W0, IP_G0Wtdhf, dHx_lin_scr and
#: its IP), two codes on one orbital set: 1.8e-14 eV measured at worst (H2O, slow twin).
PILOT_HF_BAR = 1e-9
#: eV. Those that read the FCI ground state (IP_fci, IP_G0Wex, dP_ex, dSig_ex, minOm_ex). Both
#: Davidsons stop on pyscf's residual tolerance, the square root of the energy tolerance: 3.2e-6 Ha
#: in the pilot (conv_tol 1e-11), 3.2e-7 Ha here (FCI_CONV_TOL = 1e-13). For HF and H2O in 6-31G
#: the pilot's settings leave vector errors of 2.0e-7 and 6.8e-7, the module's 2.6e-8 and 6.9e-8,
#: so most of the miss is the pilot's Psi0 and part of it the module's own: 5.2e-7 eV measured at
#: worst (NH3/6-31G's IP_G0Wex, slow twin; HF/6-31G 1.8e-8).
PILOT_FCI_BAR = 2e-6
#: eV. The oracle IPs (dH_or, dHx_or), whose pilot reference is its own SCF in the active space
#: (its G0W0 there 5.7e-7 eV from the canonical one, NH3): 2.4e-6 eV measured at worst (NH3, slow
#: twin; HF/6-31G 1.2e-6).
PILOT_ORACLE_BAR = 1e-5
#: Z_ex: 3.5e-8 measured at worst (NH3, slow twin; HF/6-31G 3.4e-9).
PILOT_Z_BAR = 2e-7
#: eV. A Lanczos run settled: the FCI rung's change from D/2 to D, 3.8e-9 eV measured at worst
#: (H2O/6-31G, slow twin; HF/6-31G 1.9e-10).
SETTLED_EV = 1e-5
#: eV. The FCI rung's D/2 root against the D/2-node Gauss quadrature of a full-reorthogonalization
#: Lanczos kept off Psi0: 1.1e-13 eV measured (N2/STO-3G's 2sigma_g, D/2 = 20; 2.2e-3 eV for a
#: recurrence that lets rounding seed Psi0).
GAUSS_BAR = 1e-9
#: Ha. N2/STO-3G at 1.6 A has two RHF solutions, -107.1848465 Ha (pyscf's default minao guess, the
#: one the unstable rungs below are measured on) and -106.9520629 Ha (huckel or 1e guess).
N2_STRETCHED_E_TOT = -107.18484646
#: Ha. That solution's energy, converged to 1e-12 Ha (-107.184846460802 measured), against the
#: other's 0.23 Ha away.
SCF_ENERGY_BAR = 1e-7

#: The pilot's quantities by class (PILOT's keys).
PILOT_HF_KEYS = ('IP_G0W0', 'IP_G0Wtdhf', 'dHx_lin_scr', 'IP_G0W0_dHx_lin')
PILOT_FCI_KEYS = ('IP_fci', 'IP_G0Wex', 'dP_ex', 'dSig_ex', 'minOm_ex')
PILOT_ORACLE_KEYS = ('dH_or', 'dHx_or')
#: The pilot cases of this file; the slow twin runs the rest of PILOT.
FAST_PILOT = (('HF', '6-31g'), ('LiH', 'cc-pvdz'))


@functools.lru_cache(maxsize=None)
def rhf(name, basis):
    """The pilot's SCF (`rhf_conv_1e12`) on GEOMETRIES[name]."""
    return rhf_conv_1e12(gto.M(atom=GEOMETRIES[name], basis=basis, verbose=0))


def quiet_run(mf, ncore, **kw):
    """(record, the warnings raised) of fci_vertex_split(mf, ncore, **kw)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        rec = fci_vertex_split(mf, ncore, **kw)
    return rec, caught


@functools.lru_cache(maxsize=None)
def pilot_run(name, basis):
    """The default run on a pilot system (ncore = 1), its mean field's arrays copied before."""
    mf = rhf(name, basis)
    kept = [np.array(mf.mo_energy), np.array(mf.mo_coeff), np.array(mf.mo_occ),
            np.array(mf.e_tot)]
    rec, caught = quiet_run(mf, 1)
    return rec, caught, kept


def patched(patch, call):
    """call() with the module constants in patch set on `SingleReference.GW.fci_vertex_split`."""
    saved = {k: getattr(FVS, k) for k in patch}
    for k, v in patch.items():
        setattr(FVS, k, v)
    try:
        return call()
    finally:
        for k, v in saved.items():
            setattr(FVS, k, v)


def flag_names(rec):
    return [f['flag'] for f in rec['numerical_flags']]


def relative(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))


def linearized_ip(num, pol, e_p):
    """-(e_p + Z Sigma(e_p)) (eV) of the pole sum Sigma(w) = sum num / (w - pol), Z =
    1 / (1 - Sigma'(e_p)), written out here."""
    sigma = float(np.sum(num / (e_p - pol)))
    slope = -float(np.sum(num / (e_p - pol) ** 2))
    return -(e_p + sigma / (1.0 - slope)) * HARTREE_TO_EV


# -- 1. the FCI rung against a dense Lehmann sum ----------------------------------------------

def dense_lehmann(h1, eri, e, p):
    """(num, pol, omega, coupling) of Sigma^c_pp over every FCI excited state of a two-electron
    active space: the pole sum, and each state's excitation energy and summed weight."""
    n = len(e)
    solver = fci.direct_spin1.FCI()
    solver.conv_tol, solver.verbose = FCI_CONV_TOL, 0
    E, C = solver.kernel(h1, eri, n, (1, 1), nroots=n * n, max_space=n * n + 10)
    num, pol = [], []
    for en, cn in zip(E[1:], C[1:]):
        rho = solver.trans_rdm1(C[0], cn, n, (1, 1))
        rho_t = solver.trans_rdm1(cn, C[0], n, (1, 1))
        num.append(np.einsum('qrs,rs->q', eri[p], rho) * np.einsum('qrs,rs->q', eri[p], rho_t))
        pol.append(np.where(np.arange(n) < 1, e - (en - E[0]), e + (en - E[0])))
    num = np.asarray(num)
    return num.ravel(), np.ravel(pol), np.asarray(E[1:]) - E[0], num.sum(axis=1)


def lowest_coupled(omega, coupling):
    """(the lowest excitation energy (Ha) of a state carrying weight, the lowest of any state):
    weight above FCI_SPLIT_LOWEST_POLE_WEIGHT of the total, as the module reads it."""
    return (float(omega[coupling > FVS.FCI_SPLIT_LOWEST_POLE_WEIGHT * coupling.sum()].min()),
            float(omega.min()))


@pytest.mark.parametrize('name,basis', [('He', 'cc-pvdz'), ('H2', '6-31g**')])
def test_the_fci_rung_at_full_depth_is_the_dense_lehmann_sum(name, basis):
    mf = rhf(name, basis)
    h1, eri, _, e, m, _ = FVS.active_hamiltonian(mf, 0)
    n, p = len(e), 0
    e0, c0, _, converged, _ = FVS.ground_state(h1, eri, n, m)
    assert converged
    depth = n * n + (n * n) % 2                     # past every Krylov dimension
    (num, pol), _, omega_min, _, _ = FVS._fci_rung(FVS._Screening(h1, eri, n, m, c0), e, m, e0,
                                                   p, depth)
    num_d, pol_d, omega_d, coupling_d = dense_lehmann(h1, eri, e, p)
    sigma = lambda w, nu, po: float(np.sum(nu / (w - po)))
    ws = e[p] + np.array([-0.31, -0.07, 0.013, 0.11, 0.37])
    assert min(min(np.abs(w - pol_d).min(), np.abs(w - pol).min()) for w in ws) > LEHMANN_GAP
    for w in ws:
        assert relative(sigma(w, num, pol), sigma(w, num_d, pol_d)) < CROSS_PATH_RTOL, w
    w_l, z_l = solve_qp_equation_poles(num, pol, e[p], label='Lanczos')
    w_d, z_d = solve_qp_equation_poles(num_d, pol_d, e[p], label='dense')
    assert relative(w_l, w_d) < CROSS_PATH_RTOL and abs(z_l - z_d) < CROSS_PATH_RTOL
    assert abs(-w_l * HARTREE_TO_EV - fci_screened_g0w_ip(mf, eta=0.0)) < IP_ORACLE_BAR
    assert abs(omega_min - lowest_coupled(omega_d, coupling_d)[0]) * HARTREE_TO_EV < OMEGA_BAR
    rec, _ = quiet_run(mf, 0, depth=depth)
    fci_rung = rec['orbitals'][0]['rungs']['fci']
    assert relative(fci_rung['ip_ev'], -w_d * HARTREE_TO_EV) < CROSS_PATH_RTOL
    assert relative(fci_rung['ip_linearized_ev'], linearized_ip(num_d, pol_d, e[p])) \
        < CROSS_PATH_RTOL
    assert fci_rung['quasiparticle'] is True
    assert rec['numerical_flags'] == []


@pytest.mark.parametrize('name,basis,ncore', [('H2', 'cc-pvdz', 0), ('LiH@3.0', 'cc-pvdz', 1)])
def test_the_lowest_fci_excitation_is_a_coupled_state(name, basis, ncore):
    mf = rhf(name, basis)
    h1, eri, _, e, m, _ = FVS.active_hamiltonian(mf, ncore)
    rec, _ = quiet_run(mf, ncore)
    orb = rec['orbitals'][0]
    _, _, omega_d, coupling_d = dense_lehmann(h1, eri, e, orb['active_orbital'])
    coupled, lowest = lowest_coupled(omega_d, coupling_d)
    assert lowest * HARTREE_TO_EV < coupled * HARTREE_TO_EV - 0.5      # a triplet lies below
    assert abs(orb['rungs']['fci']['min_omega_ev'] - coupled * HARTREE_TO_EV) < OMEGA_BAR
    assert rec['parameters']['screening_depth'] == FVS.FCI_SPLIT_SCREENING_DEPTH
    assert rec['numerical_flags'] == []


# -- 2-4. the Casida rungs against the house routes -------------------------------------------

@functools.lru_cache(maxsize=None)
def h2o_rungs():
    """H2O/6-31G, full space: (mf, e, eri, m, the module's Casida rungs)."""
    mf = rhf('H2O', '6-31g')
    _, eri, _, e, m, _ = FVS.active_hamiltonian(mf, 0)
    return mf, e, eri, m, FVS.casida_rungs(e, eri, m)


def rung_root(rung, p):
    _, e, eri, m, rungs = h2o_rungs()
    omega, xpy = rungs[rung]
    num, pol = FVS.pole_arrays(e, m, FVS.casida_residues(eri, m, xpy, [p])[0], omega)
    return solve_qp_equation_poles(num, pol, e[p], label=f'{rung} orbital {p}')


@pytest.mark.parametrize('rung,selfenergy,polarizability',
                         [('rpa', 'GW', 'RPA'), ('tdhf', 'GW@TDHF', 'TDHF'),
                          ('bse', 'GW@BSE', 'BSE')])
@pytest.mark.parametrize('below_homo', [0, 1])
def test_each_casida_rung_is_the_house_route(rung, selfenergy, polarizability, below_homo):
    mf, _, _, m, _ = h2o_rungs()
    p = m - 1 - below_homo
    w, z = rung_root(rung, p)
    house = float(np.ravel(calc_qp_energy(mf, selfenergy=selfenergy,
                                          polarizability=polarizability, df=False, eta=0.0,
                                          state=p))[0])
    assert abs(-w * HARTREE_TO_EV + house) < HOUSE_SCAN_BAR
    assert 0.0 < z <= 1.0
    if rung == 'rpa':
        qp = QPqb(mf.mo_energy, MOEriBlocks.from_mol(mf.mol, mf.mo_coeff, m), m)
        wq, zq = qp.solve_diag(p)
        assert relative(w, wq) < CROSS_PATH_RTOL and abs(z - zq) < CROSS_PATH_RTOL


def test_the_casida_spectra_are_pyscf_tdhf_and_the_house_bse():
    mf, e, eri, m, rungs = h2o_rungs()
    n = len(e)
    td = tdscf.TDHF(mf)
    td.nstates, td.verbose = m * (n - m), 0
    td.kernel()
    assert all(td.converged)
    assert_within_class(np.sort(td.e), rungs['tdhf'][0], quantity='root', label='TDHF')
    bse = solve_bse(mf, solver='dense', integrals='full', qp=None, spin='singlet',
                    numerics=Numerics(bse={'nroots': None}))
    assert_within_class(np.sort(np.asarray(bse[0])), rungs['bse'][0], quantity='root',
                        label='BSE')
    # an independent einsum static BSE from the four-index W_RPA(0)
    o, v, nov = slice(0, m), slice(m, n), m * (n - m)
    W4 = static_screened_coulomb_chemist(e, eri, m)
    dE = (e[v][None, :] - e[o][:, None]).ravel()
    A = (np.diag(dE) + 2 * eri[o, v, o, v].reshape(nov, nov)
         - W4[o, o, v, v].transpose(0, 2, 1, 3).reshape(nov, nov))
    B = (2 * eri[o, v, o, v].reshape(nov, nov)
         - W4[o, v, o, v].transpose(0, 3, 2, 1).reshape(nov, nov))
    assert_within_class(np.asarray(CasidaSolver(A, B).solve()[0]), rungs['bse'][0],
                        quantity='root', label='einsum BSE')


@pytest.mark.parametrize('rung,lbse', [('rpa', False), ('tdhf', True)])
def test_the_casida_residues_obey_both_sum_rules(rung, lbse):
    _, e, eri, m, rungs = h2o_rungs()
    n = len(e)
    A, B = LinearResponseSolver(e, eri_chemist=eri, spin_mode='restricted',
                                eta=0.0).build_casida_matrices(m, lBSE=lbse)
    omega, xpy = rungs[rung]
    p = m - 1
    w = FVS.casida_residues(eri, m, xpy, [p])[0]
    v = eri[p][:, :m, m:].reshape(n, -1)
    assert relative((w ** 2) @ omega, 2 * np.einsum('qi,ij,qj->q', v, A - B, v)) \
        < CROSS_PATH_RTOL
    assert relative((w ** 2) @ (1 / omega),
                    2 * np.einsum('qi,qi->q', v, np.linalg.solve(A + B, v.T).T)) \
        < CROSS_PATH_RTOL


# -- 5. the exact hole against the dense cation; the RPA rung through the driver -------------

def dense_cation(mf, ncore):
    """(E0, D, IP (Ha) and eigenvectors of every (m - 1, m) state, c0, n, m): the cation of the
    active space diagonalized densely, its Hamiltonian's columns built by pyscf's contract_2e."""
    h1, eri, _, e, m, _ = FVS.active_hamiltonian(mf, ncore)
    n = len(e)
    solver = fci.direct_spin1.FCI()
    solver.conv_tol, solver.verbose = FCI_CONV_TOL, 0
    e0, c0 = solver.kernel(h1, eri, n, (m, m))
    nelec = (m - 1, m)
    h2 = fci.direct_spin1.absorb_h1e(h1, eri, n, nelec, 0.5)
    shape = (fci.cistring.num_strings(n, m - 1), fci.cistring.num_strings(n, m))
    dim = shape[0] * shape[1]
    H = np.empty((dim, dim))
    for j in range(dim):
        unit = np.zeros(dim)
        unit[j] = 1.0
        H[:, j] = fci.direct_spin1.contract_2e(h2, unit.reshape(shape), n, nelec).ravel()
    E, V = np.linalg.eigh(0.5 * (H + H.T))
    return e0, solver.make_rdm1(c0, n, (m, m)), E - e0, V, c0, n, m


def dense_poles(cation, p):
    """[(IP (Ha), weight)] of active orbital p's one-hole spectrum, by decreasing weight, the
    weights |<s|a_p,alpha|Psi0>|^2 of degenerate states summed."""
    _, _, ips, V, c0, n, m = cation
    weights = (V.T @ fci.addons.des_a(c0, n, (m, m), p).ravel()) ** 2
    levels, start = [], 0
    for k in range(1, len(ips) + 1):
        if k == len(ips) or ips[k] - ips[start] > DEGENERATE_BAR:
            levels.append((float(ips[start]), float(weights[start:k].sum())))
            start = k
    return sorted(levels, key=lambda level: -level[1])


def qpqb_linearized_ip(qp, p):
    """The linearized G0W0 IP (eV) of orbital p from QPqb's own poles and couplings."""
    nocc, eps, om = qp.nocc, qp.eps, qp.omega
    num = np.concatenate([(qp.Wnu[p, :nocc, :] ** 2).ravel(), (qp.Wnu[p, nocc:, :] ** 2).ravel()])
    pol = np.concatenate([(eps[:nocc, None] - om[None, :]).ravel(),
                          (eps[nocc:, None] + om[None, :]).ravel()])
    return linearized_ip(num, pol, eps[p])


def test_the_exact_hole_is_the_dense_cation():
    mf = rhf('H2O', 'sto-3g')
    nocc = mf.mol.nelectron // 2
    rec, _ = quiet_run(mf, 0, orbitals=[nocc - 1, nocc - 2, nocc - 3])
    assert rec['numerical_flags'] == []
    cation = dense_cation(mf, 0)
    D = cation[1]
    qp = QPqb(mf.mo_energy, MOEriBlocks.from_mol(mf.mol, mf.mo_coeff, nocc), nocc)
    for orb in rec['orbitals']:
        p, exact = orb['orbital'], orb['exact']
        (ip, z) = dense_poles(cation, p)[0]
        assert abs(exact['ip_ev'] - ip * HARTREE_TO_EV) < HOLE_BAR, p
        assert abs(exact['z'] - z) < HOLE_Z_BAR, p
        assert exact['residual_ha'] <= FVS.FCI_SPLIT_HOLE_RESIDUAL_BAR
        assert abs(exact['weight_sum'] - D[p, p] / 2) < WEIGHT_SUM_BAR
        assert exact['second_z'] < exact['z'] and exact['window_weight'] == 0.0
        wq, _ = qp.solve_diag(p)
        assert relative(orb['rungs']['rpa']['ip_ev'], -wq * HARTREE_TO_EV) < CROSS_PATH_RTOL
        assert relative(orb['rungs']['rpa']['ip_linearized_ev'], qpqb_linearized_ip(qp, p)) \
            < CROSS_PATH_RTOL


def gauss_root(mf, ncore, p_mo, nodes):
    """The G0W_FCI IP (eV) of orbital p_mo from the `nodes`-node Gauss quadratures of its FCI
    rung: a Lanczos with full reorthogonalization, every vector kept off Psi0, from the module's
    own start vectors and operator (`_Screening`), written out here."""
    h1, eri, _, e, m, _ = FVS.active_hamiltonian(mf, ncore)
    n, p = len(e), p_mo - ncore
    e0, c0, _, _, _ = FVS.ground_state(h1, eri, n, m)
    screening = FVS._Screening(h1, eri, n, m, c0)
    num, pol = [], []
    for q in range(n):
        v = screening.start(p, q)
        norm2 = float(v @ v)
        if np.sqrt(norm2) < FVS.ACTIVE_SPACE_VERTEX_TOL:
            continue
        basis, a, b = [screening.c0f], [], []
        v = v / np.sqrt(norm2)
        for _ in range(nodes):
            w = screening.matvec(v)
            a.append(float(v @ w))
            basis.append(v)
            V = np.asarray(basis)
            for _ in range(2):
                w -= V.T @ (V @ w)
            b.append(float(np.linalg.norm(w)))
            v = w / b[-1]
        theta, U = np.linalg.eigh(np.diag(a) + np.diag(b[:-1], 1) + np.diag(b[:-1], -1))
        num.append(norm2 * U[0] ** 2)
        pol.append(e[q] + (-1.0 if q < m else 1.0) * (theta - e0))
    w, _ = solve_qp_equation_poles(np.concatenate(num), np.concatenate(pol), e[p], label='Gauss')
    return -w * HARTREE_TO_EV


def test_an_inner_valence_hole_and_a_rung_without_a_quasiparticle():
    mf = rhf('N2', 'sto-3g')
    rec, caught = quiet_run(mf, 2, orbitals=[2])
    orb = rec['orbitals'][0]
    exact, rungs = orb['exact'], orb['rungs']
    half = rungs['fci']['depth'] // 2
    assert abs(rungs['fci']['ip_half_depth_ev'] - gauss_root(mf, 2, 2, half)) < GAUSS_BAR
    assert abs(rungs['fci']['half_depth_change_ev']) > FVS.FCI_SPLIT_HALF_DEPTH_BAR * HARTREE_TO_EV
    poles = dense_poles(dense_cation(mf, 2), 0)
    assert abs(exact['ip_ev'] - poles[0][0] * HARTREE_TO_EV) < HOLE_BAR
    assert abs(exact['z'] - poles[0][1]) < HOLE_Z_BAR
    assert exact['residual_ha'] <= FVS.FCI_SPLIT_HOLE_RESIDUAL_BAR
    assert abs(exact['second_ip_ev'] - poles[1][0] * HARTREE_TO_EV) \
        <= exact['second_residual_ha'] * HARTREE_TO_EV
    assert exact['window_weight'] < FVS.FCI_SPLIT_SECOND_POLE_RATIO * exact['z']
    assert [rungs[r]['quasiparticle'] for r in FVS.RUNGS] == [True, True, True, False]
    assert rungs['fci']['z'] <= FVS.QP_WINDOW_Z_MIN
    assert sorted(flag_names(rec)) == ['fci_split_depth_unconverged',
                                       'fci_split_rung_no_quasiparticle']
    assert all('G0W_FCI' in f['message'] for f in rec['numerical_flags'])
    assert any(issubclass(w.category, FCISplitShort) for w in caught)


# -- 6. the pilot's numbers and the density side --------------------------------------------

def pilot_quantities(orbital):
    """An orbital's record under the pilot's names (PILOT's keys), eV (Z_ex dimensionless)."""
    rungs, splits, dens = orbital['rungs'], orbital['splits_ev'], orbital['density']
    return dict(IP_fci=orbital['exact']['ip_ev'], IP_G0W0=rungs['rpa']['ip_ev'],
                IP_G0Wtdhf=rungs['tdhf']['ip_ev'], IP_G0Wex=rungs['fci']['ip_ev'],
                Z_ex=rungs['fci']['z'], minOm_ex=rungs['fci']['min_omega_ev'],
                dP_ex=splits['w_side'], dSig_ex=splits['sigma_side'],
                dHx_lin_scr=dens['dhx_lin_scr_ev'], IP_G0W0_dHx_lin=dens['ip_g0w0_dhx_lin_scr_ev'],
                dH_or=dens['ip_g0w0_dh_oracle_ev'], dHx_or=dens['ip_g0w0_dhx_oracle_ev'])


def assert_pilot(orbital, pilot):
    """An orbital's record against the pilot's numbers (PILOT, Ha), each class to its bar, with
    its split adding up to minus the G0W0 error and both Lanczos runs settled."""
    got = pilot_quantities(orbital)
    for keys, bar in ((PILOT_HF_KEYS, PILOT_HF_BAR), (PILOT_FCI_KEYS, PILOT_FCI_BAR),
                      (PILOT_ORACLE_KEYS, PILOT_ORACLE_BAR)):
        for key in keys:
            want = pilot[key] * HARTREE_TO_EV
            assert abs(got[key] - want) < bar, (key, got[key], want)
    assert abs(got['Z_ex'] - pilot['Z_ex']) < PILOT_Z_BAR, (got['Z_ex'], pilot['Z_ex'])
    splits = orbital['splits_ev']
    assert abs(splits['w_side'] + splits['sigma_side'] + splits['g0w0_minus_exact']) \
        < SPLIT_SUM_BAR
    assert abs(orbital['rungs']['fci']['half_depth_change_ev']) < SETTLED_EV
    assert orbital['exact']['residual_ha'] <= FVS.FCI_SPLIT_HOLE_RESIDUAL_BAR
    assert all(orbital['rungs'][r]['quasiparticle'] for r in FVS.RUNGS)


@pytest.mark.parametrize('name,basis', FAST_PILOT)
def test_the_pilot_numbers_are_reproduced(name, basis):
    rec, caught, _ = pilot_run(name, basis)
    assert rec['numerical_flags'] == [] and not caught
    assert_pilot(rec['orbitals'][0], PILOT[(name, basis)])


def pilot_lin_gw_density(e, m, Om, w):
    """The pilot's linearized-GW density (pilot/practitioner_pilot/pilot.py, lin_gw_density),
    re-typed: w[s, p, q] with the sqrt2, loops as there."""
    n = len(e)
    o, v = np.arange(m), np.arange(m, n)
    g = np.zeros((n, n))

    def sigc(p, q, om):
        return (np.sum(w[:, p, o] * w[:, q, o] / (om - e[None, o] + Om[:, None]))
                + np.sum(w[:, p, v] * w[:, q, v] / (om - e[None, v] - Om[:, None])))
    for i in o:
        for j in o:
            g[i, j] = -np.sum(w[:, i, v] * w[:, j, v] / ((e[None, v] + Om[:, None] - e[i])
                                                         * (e[None, v] + Om[:, None] - e[j])))
    for a in v:
        for b in v:
            g[a, b] = np.sum(w[:, a, o] * w[:, b, o] / ((e[None, o] - Om[:, None] - e[a])
                                                        * (e[None, o] - Om[:, None] - e[b])))
    for i in o:
        for a in v:
            val = sigc(i, a, e[i]) / (e[i] - e[a])
            val += np.sum(w[:, i, o] * w[:, a, o] / ((e[None, o] - Om[:, None] - e[i])
                                                     * (e[None, o] - Om[:, None] - e[a])))
            g[i, a] = g[a, i] = val
    return 2 * g


def pilot_screened_hartree_shift(e, eri, m, dD, p):
    """The pilot's screened_hartree_shift (pilot.py), re-typed: J[dD]_pp plus its static RPA
    (Hartree-kernel) screening."""
    n = len(e)
    o, v = np.arange(m), np.arange(m, n)
    dE = (e[v][None, :] - e[o][:, None]).ravel()
    K = eri[np.ix_(o, v, o, v)].reshape(len(dE), len(dE))
    J = np.einsum('pqrs,rs->pq', eri, dD)
    y = -2 * np.linalg.solve(np.diag(dE) + 4 * K, J[np.ix_(v, o)].T.ravel())
    return J[p, p] + 2 * np.dot(eri[p, p][np.ix_(o, v)].ravel(), y)


@pytest.mark.parametrize('name,basis', FAST_PILOT)
def test_the_screened_tadpole_is_the_pilots(name, basis):
    mf = rhf(name, basis)
    rec, _, _ = pilot_run(name, basis)
    _, eri, _, e, m, _ = FVS.active_hamiltonian(mf, 1)
    qp = QPqb(e, eri, m)
    dD = pilot_lin_gw_density(e, m, qp.omega, np.transpose(qp.Wnu, (2, 0, 1)))
    for orb in rec['orbitals']:
        p = orb['active_orbital']
        tc = pilot_screened_hartree_shift(e, eri, m, dD, p) * HARTREE_TO_EV
        bare = np.einsum('rs,rs->', eri[p, p], dD) * HARTREE_TO_EV
        assert relative(orb['density']['tc_ev'], tc) < CROSS_PATH_RTOL
        assert relative(orb['density']['tc_bare_ev'], bare) < CROSS_PATH_RTOL
        assert abs(orb['density']['trace_dd_lin']) < WEIGHT_SUM_BAR
        assert abs(orb['density']['ip_g0w0_tc_ev']
                   - (orb['rungs']['rpa']['ip_ev'] - orb['density']['tc_ev'])) < SPLIT_SUM_BAR


def test_the_linearized_density_is_the_house_gw_density():
    mf, e, eri, m, rungs = h2o_rungs()
    omega, xpy = rungs['rpa']
    w = FVS.casida_residues(eri, m, xpy, list(range(len(e))))
    dD = FVS.linearized_gw_density(e, m, omega, w)
    oo, ov, vv = GWDensityMatrixSolver(e, eri_chemist=eri, eta=0.0).compute_unrelaxed_blocks(
        m, mf, mf.mol, 'RPA')
    for got, want in ((dD[:m, :m], oo), (dD[:m, m:], ov), (dD[m:, m:], vv)):
        assert relative(got, want) < CROSS_PATH_RTOL


# -- 7. refusals and flags --------------------------------------------------------------------

def copied(mf):
    """A fresh RHF object holding mf's converged state."""
    new = scf.RHF(mf.mol)
    new.__dict__.update(mf.__dict__)
    return new


def refusal_cases():
    """(label, mean field, ncore, keywords, the message's start) of each refusal."""
    hf = rhf('HF', '6-31g')
    mol = hf.mol
    unconverged = scf.RHF(mol)
    triplet = scf.hf.RHF(gto.M(atom='O 0 0 0; O 0 0 1.21', basis='sto-3g', spin=2, verbose=0))
    triplet.converged = True
    shuffled = copied(hf)
    shuffled.mo_occ = np.array(hf.mo_occ)
    shuffled.mo_occ[[4, 5]] = shuffled.mo_occ[[5, 4]]
    charged = attach_environment(copied(hf), PointCharges([[0.0, 0.0, 5.0]], [0.4]))
    helium = rhf('He', 'sto-3g')
    return [('UHF', scf.UHF(mol), 1, {}, r'^mf = UHF'),
            ('ROHF', scf.ROHF(mol), 1, {}, r'^mf = ROHF'),
            ('RKS', dft.RKS(mol), 1, {}, r'^mf\.xc = '),
            ('density fitted', scf.RHF(mol).density_fit(), 1, {}, r'^mf\.with_df = '),
            ('solvated', scf.RHF(mol).ddCOSMO(), 1, {}, r'^mf\.with_solvent = '),
            ('point charges', charged, 1, {}, r'^mf\.with_screening = PointCharges: '),
            ('unconverged', unconverged, 1, {}, r'^mf\.converged = '),
            ('open shell', triplet, 1, {}, r'^mf\.mol\.spin = '),
            ('non-aufbau', shuffled, 1, {}, r'^mf\.mo_occ = \[2, 2, 2, 2, 0, 2, 0, \.\.\.\]: '),
            ('no virtual', helium, 0, {}, r'^mf\.mo_occ = \[2\]: '),
            ('ncore = nocc', hf, 5, {}, r'^ncore = '),
            ('ncore < 0', hf, -1, {}, r'^ncore = '),
            ('ncore float', hf, 1.0, {}, r'^ncore = '),
            ('ncore bool', hf, True, {}, r'^ncore = '),
            ('ncore splits the pi pair', hf, 4, {},
             r'^ncore = 4: it splits the degenerate occupied level \[3, 4\]'),
            ('a virtual orbital', hf, 1, dict(orbitals=[5]), r'^orbitals = '),
            ('a virtual orbital as an int', hf, 1, dict(orbitals=5), r'^orbitals = 5: '),
            ('a core orbital', hf, 1, dict(orbitals=[0]), r'^orbitals = '),
            ('twice the HOMO', hf, 1, dict(orbitals=[4, 4]), r'^orbitals = '),
            ('no orbital', hf, 1, dict(orbitals=[]), r'^orbitals = '),
            ('orbital float', hf, 1, dict(orbitals=[4.0]), r'^orbitals = '),
            ('odd depth', hf, 1, dict(depth=3), r'^depth = '),
            ('depth 0', hf, 1, dict(depth=0), r'^depth = '),
            ('depth float', hf, 1, dict(depth=4.0), r'^depth = '),
            ('depth bool', hf, 1, dict(depth=True), r'^depth = ')]


def no_work(*args, **kwargs):
    raise AssertionError('work started')


def test_every_wrong_input_is_refused_before_any_work(monkeypatch):
    cases = refusal_cases()
    gas = attach_environment(copied(rhf('HF', '6-31g')), NoEnvironment())
    monkeypatch.setattr(FVS, 'active_hamiltonian', no_work)
    for label, mf, ncore, kw, start in cases:
        with pytest.raises(ValueError, match=start):
            fci_vertex_split(mf, ncore, **kw)
    with pytest.raises(AssertionError, match='work started'):      # the gas phase marked
        fci_vertex_split(gas, 1)


def test_an_active_space_past_the_memory_allowance_is_refused(monkeypatch):
    hf = rhf('HF', '6-31g')
    rec, _, _ = pilot_run('HF', '6-31g')
    sizes = rec['sizes']
    peak = max(FVS.FCI_SPLIT_DAVIDSON_VECTORS * sizes['dets_n'],
               FVS.FCI_SPLIT_HOLE_COPIES * FVS.FCI_SPLIT_HOLE_DEPTH * sizes['dets_n_minus_1'])
    assert sizes['memory_estimate_mb'] == pytest.approx(peak * 8 / 1e6, rel=1e-12)
    assert rec['parameters']['max_memory_mb'] == allocation_max_memory_mb(
        default=FVS.FCI_SPLIT_MAX_MEMORY_MB)
    monkeypatch.setattr(FVS, 'active_hamiltonian', no_work)
    monkeypatch.setattr(FVS, 'allocation_max_memory_mb', lambda default=None: 30.0)
    with pytest.raises(ValueError, match=r'^ncore = 1: .* needs about 36 MB .* above the 30 MB'):
        fci_vertex_split(hf, 1)


def flagged(rec, caught, name):
    """Whether `name` is in the record's numerical_flags and an FCISplitShort warned."""
    return (name in flag_names(rec)
            and any(issubclass(w.category, FCISplitShort) for w in caught))


def test_a_short_screening_depth_is_flagged():
    hf = rhf('HF', '6-31g')
    rec, caught = quiet_run(hf, 1, depth=4)
    assert flag_names(rec) == ['fci_split_depth_unconverged'] * 2 and flagged(
        rec, caught, 'fci_split_depth_unconverged')
    assert all('G0W_FCI' in f['message'] for f in rec['numerical_flags'])
    for orb in rec['orbitals']:
        assert abs(orb['rungs']['fci']['half_depth_change_ev']) \
            > FVS.FCI_SPLIT_HALF_DEPTH_BAR * HARTREE_TO_EV


def test_an_fci_rung_without_a_start_vector_has_no_pole():
    mf = rhf('He', 'cc-pvdz')
    rec, caught = patched({'ACTIVE_SPACE_VERTEX_TOL': 1e10}, lambda: quiet_run(mf, 0))
    orb = rec['orbitals'][0]
    fci_rung = orb['rungs']['fci']
    assert fci_rung['skipped_starts'] == rec['sizes']['n_act'] and fci_rung['steps'] == 0
    for key in ('ip_ev', 'ip_linearized_ev', 'ip_half_depth_ev'):
        assert abs(fci_rung[key] + orb['eps_hf_ev']) < SPLIT_SUM_BAR, key
    assert abs(fci_rung['z'] - 1.0) < WEIGHT_SUM_BAR and abs(fci_rung['z_linearized'] - 1.0) \
        < WEIGHT_SUM_BAR
    assert fci_rung['quasiparticle'] is True and fci_rung['stable'] is True
    assert fci_rung['min_omega_ev'] is None and fci_rung['half_depth_change_ev'] == 0.0
    assert rec['numerical_flags'] == []
    assert not any(issubclass(w.category, FCISplitShort) for w in caught)
    assert json.loads(json.dumps(rec))['orbitals'][0]['rungs']['fci']['min_omega_ev'] is None


@pytest.mark.parametrize('patch,names', [
    ({'FCI_SPLIT_HOLE_DEPTH': 4}, ('fci_split_depth_unconverged',)),
    ({'FCI_SPLIT_HOLE_RESIDUAL_BAR': 0.0}, ('fci_split_depth_unconverged',)),
    ({'FCI_SPLIT_MAX_CYCLE': 1}, ('fci_split_ground_not_converged',)),
    ({'FCI_SPLIT_SECOND_POLE_RATIO': 0.01, 'FCI_SPLIT_SECOND_POLE_WINDOW': 3.0},
     ('fci_split_main_pole_ambiguous',)),
    ({'QP_WINDOW_Z_MIN': 0.99},
     ('fci_split_main_pole_ambiguous', 'fci_split_rung_no_quasiparticle')),
    ({'FCI_SPLIT_FOCK_BAR': 0.0}, ('fci_split_fock_not_diagonal',)),
    ({'FCI_SPLIT_SPIN_SQUARE_BAR': -1.0}, ('fci_split_ground_not_singlet',))])
def test_each_registry_bar_raises_its_flag(patch, names):
    hf = rhf('HF', '6-31g')
    plain, _ = quiet_run(hf, 1, depth=4)
    rec, caught = patched(patch, lambda: quiet_run(hf, 1, depth=4))
    assert all(flagged(rec, caught, name) for name in names)
    if names == ('fci_split_depth_unconverged',):
        assert flag_names(plain) == list(names) * 2 and flag_names(rec) == list(names) * 4
        assert sum('exact main pole' in f['message'] for f in rec['numerical_flags']) == 2
        assert rec['parameters']['hole_depth'] == patch.get('FCI_SPLIT_HOLE_DEPTH',
                                                            FVS.FCI_SPLIT_HOLE_DEPTH)
    else:
        assert not set(names) & set(flag_names(plain))
        assert set(flag_names(rec)) == set(names) | {'fci_split_depth_unconverged'}
    if 'fci_split_ground_not_converged' in names:
        assert rec['ground_state']['converged'] is False
    if 'fci_split_rung_no_quasiparticle' in names:
        assert not any(o['rungs'][r]['quasiparticle'] for o in rec['orbitals'] for r in FVS.RUNGS)


@pytest.mark.parametrize('name,basis,ncore,unstable,dhx_finite', [
    ('MgO', 'sto-3g', 6, ('tdhf',), True),
    ('N2@1.6', 'sto-3g', 2, ('tdhf', 'bse'), False),
    ('H2@2.5', '6-31g', 0, ('bse',), True)])
def test_an_unstable_rung_is_nan_and_flagged(name, basis, ncore, unstable, dhx_finite):
    mf = rhf(name, basis)
    if name == 'N2@1.6':
        assert abs(mf.e_tot - N2_STRETCHED_E_TOT) < SCF_ENERGY_BAR
    rec, caught = quiet_run(mf, ncore)
    entries = [f for f in rec['numerical_flags'] if f['flag'] == 'fci_split_rung_unstable']
    assert flagged(rec, caught, 'fci_split_rung_unstable')
    assert len(entries) == len(unstable) + (not dhx_finite)
    for orb in rec['orbitals']:
        for rung in FVS.RUNGS:
            r = orb['rungs'][rung]
            assert r['stable'] is (rung not in unstable)
            assert bool(np.isfinite(r['ip_ev'])) is (rung not in unstable)
            assert bool(np.isfinite(r['ip_linearized_ev'])) is (rung not in unstable)
        assert bool(np.isfinite(orb['density']['dhx_lin_scr_ev'])) is dhx_finite
        assert np.isfinite(orb['density']['tc_ev'])
    json.loads(json.dumps(rec))


# -- 8. nothing existing changes; the defaults ------------------------------------------------

def test_the_mean_field_is_left_as_it_was():
    for key in FAST_PILOT:
        mf = rhf(*key)
        _, _, kept = pilot_run(*key)
        assert bitwise_arrays([mf.mo_energy, mf.mo_coeff, mf.mo_occ, np.array(mf.e_tot)], kept)


def test_the_default_orbitals_are_the_homo_and_the_level_below():
    rec, _, _ = pilot_run('HF', '6-31g')
    assert rec['parameters']['orbitals'] == [4, 2]
    assert [o['level'] for o in rec['orbitals']] == [[3, 4], [2]]
    assert rec['parameters']['screening_depth'] == FVS.FCI_SPLIT_SCREENING_DEPTH
    mf = rhf('NH3', '6-31g')
    e = np.asarray(mf.mo_energy)[1:]
    assert FVS.default_orbitals(e, 4, 1) == [4, 3]
    assert FVS.level_members(e, 4, 2) == [1, 2]
    rec, _, _ = pilot_run('LiH', 'cc-pvdz')
    assert rec['parameters']['orbitals'] == [1]


def test_the_record_is_json():
    rec, _, _ = pilot_run('HF', '6-31g')
    back = json.loads(json.dumps(rec))
    assert {k: back['sizes'][k] for k in ('nmo', 'nocc', 'n_act', 'nelec_act', 'n_ov', 'dets_n',
                                           'dets_n_minus_1')} == {
        'nmo': 11, 'nocc': 5, 'n_act': 10, 'nelec_act': 8, 'n_ov': 24, 'dets_n': 44100,
        'dets_n_minus_1': 25200}
    exact = back['orbitals'][0]['exact']
    assert exact['steps'] == FVS.FCI_SPLIT_HOLE_DEPTH and exact['breakdown'] is False
    assert set(back['orbitals'][0]['rungs']['fci']) == {
        'ip_ev', 'z', 'quasiparticle', 'ip_linearized_ev', 'z_linearized', 'min_omega_ev',
        'stable', 'ip_half_depth_ev', 'half_depth_change_ev', 'depth', 'steps',
        'skipped_starts'}


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
