"""The soft-mode split of a closed shell's TDHF triplet instability (`LinearResponse.soft_mode`)
and the PSD self-energies on its levels (`GW.soft_mode_qp`): what is held.

1. The triplet matrices built from MO blocks are the house's (build_casida_matrices with
   triplet=True) to CROSS_PATH_RTOL: H2 and H2O, cc-pVDZ.
2. The split is exact: mu / M = Omega_s^2 to 1e-12 of the largest Omega_n^2 and d^T e = 1 to
   1e-12, and the soft mode put back at its harmonic level is the house Casida's root 0,
   (Omega, X, Y) to 1e-10 (H2 at 1.0 A; N2, whose soft direction spans two degenerate pi pairs).
3. The RHF->UHF path: V''(0) = mu to 1e-5 relative (h and 2h Richardson, h = 4e-3; 1.3e-6
   measured), D(Q) =
   sin Q for one occupied orbital to 1e-12, and N2's half-period pi / sigma_1 = 1.42 pi with
   sigma_1 = sigma_2 to 1e-9 (its two pi pairs).
4. The collective solvers on a harmonic V give Omega and <1|Q|0>^2 = 1/(2 M Omega) to 1e-5
   relative: 1D periodic, 1D between walls, 3D (5e-7, 5e-7, 5.4e-6 measured).
5. PSD2 with the soft mode at its harmonic level is PSD2@TDHF: the pole sums agree to
   CROSS_PATH_RTOL at five frequencies (H2/cc-pVDZ, 1.0 A), and with model=None the driver gives
   `calc_qp_energy`'s PSD2@TDHF HOMO to 1e-6 eV (2.6e-8 eV measured).
6. Through the instability (H2/cc-pVDZ, R_CF = 1.2104 A): Omega_T^2 > 0 at 1.20 A and < 0 at
   1.212 A, where the house triplet Casida refuses (CasidaInstability) and the split gives a
   finite collective triplet.
7. Against FCI (the house's PyscfFCISolver; H2/cc-pVDZ at 0.75, 1.0, 1.212, 1.5 and 2.0 A): the
   collective 1D triplet within 1.0 eV (0.83 eV the worst, at 1.5 A), and closer than TDA from
   1.0 A on (TDA 0.8 to 1.4 eV low, negative at 2.0 A).
8. PSD2 through the instability: PSD2@TDHF has no quasiparticle at 1.19 A (Z = 0.36, IP 26 eV)
   and is refused at 1.212 A; with the soft mode split off Z > 0.7 at 0.75 to 1.212 A, the IP
   moves by 0.03 eV from 1.19 to 1.212 A, and it lies closer to FCI than PSD2@TDHF at 0.75, 1.0
   and 1.19 A.
9. Molecules at their MP2/cc-pVDZ minima, against the recorded EOM-CCSD of
   tests/soft_mode_references.py: the collective 1D triplet of the soft modes of N2, H2CO,
   ethylene and O3 within 1.2 eV (-0.48, +0.69, +0.52, +1.15 eV measured) and closer than the TDA
   root of the same mode (TDA misses by -1.60, -1.20, -0.93 eV, and is negative for O3); H2O, whose
   triplet is stiff (TDHF 8.09 eV), is the control: within 1.1 eV (+1.00 measured), where TDHF is
   closer (+0.65).
10. Refusals before any work: a UHF, a DFT, a density-fitted and an unconverged reference; an
    unknown model, boundary or self-energy; an A - B that is not positive definite
    (CasidaInstability).
11. Flags: O3's three soft modes raise 'soft_modes_coupled' (SoftModesCoupled); an edge-weight bar
    of zero raises 'soft_mode_edge_weight' (SoftModeEdgeWeight); both reach numerical_flags.

The relaxed path (path='relaxed', `LinearResponse.soft_mode.relaxed_path`):

12. Its harmonic limit is TDHF's. On the RHF quadratic model of the energy the constrained minimum
    is the straight path, kappa = Q d to 1e-12 (0 measured) and V = mu Q^2 / 2 to 1e-12 relative
    (9e-16), so M(Q) = M (H2 at 1.0 A, N2); the variable-mass solver on a harmonic V gives Omega
    and 1/(2 M Omega) to 1e-5 (periodic and walls, 5e-7 measured), and with a constant mass it
    gives `collective_levels`' levels to 1e-5 eV (7e-11 periodic, 1.4e-6 walls, H2 at 1.5 A). On
    the real energy V''(0) = mu to 1e-5 relative (Richardson from fresh constrained minima at
    h = 4e-3 and 2h: 1.2e-9 H2, 1.2e-7 ethylene twisted by 60 degrees, an unstable mode, and
    -4.7e-7 to +7.3e-7 for it over four RHF initial guesses) and the path inertia M(0) = M to
    1e-10 (9e-16, 3e-13): V''(0) / M(0) = Omega_s^2. At Q = 4e-3 and 8e-3 the straight point Q d is
    already the constrained minimum to the gradient bar (the gradient across the plane is O(Q^3)
    there), so this part holds the RHF's stationarity and the split's conjugate e, not the
    relaxation; claims 14 to 16 hold what the relaxation can get wrong.
13. One pair (H2/STO-3G at 1.0 and 2.5 A) has nothing to relax: the relaxed path is the straight
    one, V to 1e-12 Ha (7e-16), M = M to 1e-12 (4e-16), Q* = pi / sigma_1, and its level is the
    straight path's to 1e-5 eV (1.2e-6 measured, the two 1D solvers).
14. Ethylene twisted by 60 degrees: V_relaxed <= V_straight at every common sample up to Q*
    (1e-10 Ha; 0 at Q = 0, below elsewhere) and below it by more than 10 mHa at Q = L / 2 (24.4
    mHa measured), e^T kappa = Q to 1e-12, and the relaxed 1D triplet within 0.05 eV of spin-flip
    EOM-CCSD (-0.0075 measured, where the straight path is +0.371); O3's lowest mode holds its two
    other soft modes, e_t^T kappa = 0 to 1e-12 (7e-15).
15. Constrained saddles: H2CO's path turns into one from 0.9 L along its second triplet (3.37 eV)
    and N2's from 0.69 L along its 3Delta_u pair (5.01 eV); a warm start falls toward the
    wrap-around points of the plane e^T kappa = Q (V = 0), and the fresh start from kappa = Q d,
    which keeps the symmetry of d, stays on the saddle. Without hold rounds H2CO's probe finds a
    negative curvature (-0.66 in units of the RHF's; it also takes five samples from fresh starts at
    1 to 4 threads, where they fall depending on the thread count), flagged 'soft_mode_path_branch'
    (SoftModePathShort). With them H2CO holds its mode 1 and N2 its modes 1 and 2 (and the modes
    that turn negative past the antipode: H2CO's 2 and 4, N2's 3 to 5), after which neither path
    is frozen or takes a sample from a fresh start and the lowest curvature probed is positive;
    H2CO's level is the one without the holds to 1e-5 eV (2e-8 measured), and N2's that of the
    continuous symmetric branch, 6.57186 eV, to 1e-4 eV (2e-6 measured; frozen at 0.76 L without
    the holds, it was 6.57225).
16. The relaxed levels of the 14 small systems (H2/cc-pVDZ at 0.75, 1.0, 1.2104, 1.5, 2.0 and
    3.0 A, N2, H2CO, H2O, ethylene, O3 and ethylene twisted by 45, 60 and 70 degrees) are this
    code's recorded ones (RELAXED_LEVELS_EV) to 1e-4 eV (3e-9 at most at 1, 2 and 4 threads;
    within 5e-4 eV of the tweak study's audit-corrected targets); every path closes (the alpha
    determinants at -+Q* overlap to 1e-3 of 1 and |D(Q*)| < 1e-3: 4e-13 and 3.8e-8 measured) and
    none is frozen, unconverged or restarted; and at the relaxed sample nearest Q* the constrained
    Hessian of H2CO (with its held modes) and O3 is positive definite, built dense from central
    differences of the analytic gradient across the plane (lowest +0.082 and +0.042 Ha).
17. `GW.soft_mode_qp` passes path through (H2 at 1.0 A: Z > 0.7), and refuses an unknown path and
    the relaxed path outside the 1D model before any work (a UHF reference is refused for its
    path first), as `collective_levels` refuses a mass profile in 3D.
18. Flags: one minimiser step per sample raises 'soft_mode_path_not_converged', a branch test
    that rejects every step freezes the path at Q = 0 ('soft_mode_path_frozen', the straight
    level to 1e-5 eV), and a closure bar above 1 raises 'soft_mode_path_closure', all
    SoftModePathShort.

The generator-coordinate model (model='gcm', `LinearResponse.soft_mode.gcm_levels`):

19. Its closed-form kernels are the generic non-orthogonal (Loewdin) ones: N, H - E_RHF N, D_alpha
    and S^2 between the path's determinants, by `loewdin_kernels` in the RHF MO basis, at eight
    random (Q, Q') to 1e-10 Ha for H (1.7e-13 measured) and 1e-12 for the others (3e-15): H2O and
    N2, and H2O/STO-3G, whose five occupied orbitals outnumber its two virtuals; their diagonal is
    `path_samples`' V and D to 1e-10 (H2O; 3.4e-13 measured).
20. One pair: H2/STO-3G at 0.75, 1.0, 1.5 and 2.5 A gives the FCI triplet (the house's
    PyscfFCISolver) to 1e-8 eV (3e-14 measured), as one level; H2/cc-pVDZ at 0.75 to 2.0 A gives
    the CI in the two orbitals {g, v_d} (`casci_two_orbitals`), its gap and coupling to 1e-10
    (5e-14 eV, 6e-16), with spin-pure states (<S^2> = 0 and 2 to 1e-12) whatever the grid (12
    points against 32: 1e-14 eV).
21. H2/cc-pVDZ against FCI: within 0.25 eV from 0.75 to 2.0 A (-0.227 measured, at 0.75 A), and
    closer than the 1D model from 1.0 A on.
22. Convergence in the number of generator points: 24 and 48 against 32 within 1e-3 eV at the same
    ranks (N2 5.4e-4 and 3.8e-4 measured, H2O 3e-5).
23. Its harmonic limit is not TDHF's: on the exact kernels V''(0) = mu, the overlap is
    exp(-s^2 / 4) at small s = Q - Q', and h2(0) = d^T (A - B) d / 4, all to 1e-5 relative
    (Richardson at s = 4e-3 and 8e-3; 2.5e-7 at most, H2O), so its small-amplitude inertia is the
    Peierls-Yoccoz mass 1/d^T (A - B) d and a harmonic mode comes out (M d^T (A - B) d)^1/2 times
    Omega_s: 1 for H2/STO-3G (to 1e-12; d is an eigenvector of A - B), 1.024 for H2/cc-pVDZ at
    1.0 A and 1.027 for N2.
24. Molecules: the first level of N2, H2CO, H2O, ethylene and O3 is the tweak study's audited
    prototype's (`GCM_AUDITED`) to 1e-6 eV (7e-13 measured) with its ranks; N2's degenerate pi
    pairs raise 'soft_mode_gcm_degenerate_pairs' (<S^2> = 0.184 in its even ground state, the
    audit's independent value to 1e-3) and the others nothing but O3's 'soft_modes_coupled';
    H2CO and ethylene lie within 0.25 eV of EOM-CCSD (-0.232 and -0.144 measured, where the 1D
    model of claim 9 is +0.69 and +0.52).
25. `GW.soft_mode_qp` passes model='gcm' through (H2 at 1.0 A: Z > 0.7), whose poles carry the
    model's level and coupling (X + Y = sqrt2 c d to 1e-12); the relaxed path with model='gcm' is
    refused with NotImplementedError before any work in both drivers (a UHF reference is refused
    for its path), and model='gcm' in `collective_levels` with ValueError, whose refusal of an
    unknown model names only the models it solves ('1d', '3d').
26. Flags: N2 raises 'soft_mode_gcm_degenerate_pairs', and an overlap cut of 0.5, which keeps one
    state per block, 'soft_mode_gcm_few_states', both SoftModeGCMShort.

The reference numbers come from pyscf FCI run here, from EOM-CCSD recorded on a workstation and
from the spin-flip EOM-CCSD of twisted ethylene that the heavy file computed on the ADA cluster
(job 1238268; EOM_CCSD and SPIN_FLIP_EV, provenance in tests/soft_mode_references.py); N2's
symmetric branch and the audited GCM levels come from the soft-mode tweak study
(N2_SYMMETRIC_BRANCH_EV and GCM_AUDITED), and RELAXED_LEVELS_EV from this code (a regression
record); the bars hold measured misses at 1 to 4 threads.

Run: python tests/test_soft_mode_triplet.py, or under pytest.
"""
import functools
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.linalg
from pyscf import ao2mo, gto, scf

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.common.eri_blocks import mo_eri
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.GW.soft_mode_qp import soft_mode_qp
from src.SingleReference.LinearResponse import soft_mode as SM
from src.SingleReference.LinearResponse.casida import CasidaInstability, CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.Solvers.active_space_solver import PyscfFCISolver
from tests.pins import CROSS_PATH_RTOL
from tests.soft_mode_references import EOM_CCSD, GEOMETRIES, SPIN_FLIP_EV, geometry, rhf

#: Bar of the split's exact identities: mu / M = Omega_s^2 relative to the largest |Omega_n^2|,
#: the scale the rounding of T = (A - B)^1/2 (A + B) (A - B)^1/2 goes with (332 Ha^2 for N2,
#: whose Omega_s^2 is 5.1e-3: relative to Omega_s^2 itself N2 misses by up to 4e-11), and
#: d^T e = 1. Measured over 40 orders of the pairs, six OpenBLAS kernels and 1 to 4 threads on a
#: workstation: at most 6e-16 (N2) and 7e-16.
SPLIT_RTOL = 1e-12
#: Ha. The harmonic collective level against the house Casida root: 3e-13 (N2) measured, 4e-12
#: at most over six OpenBLAS kernels and 1 to 4 threads.
HARMONIC_ROOT_TOL = 1e-10
#: N2's sigma_1 = sigma_2 (its two pi pairs): 2e-13 measured, 9e-12 at most over 40 orders of the
#: pairs, six OpenBLAS kernels and 1 to 4 threads.
SIGMA_PAIR_TOL = 1e-9
#: Relative bar of V''(0) = mu, Richardson from h and 2h (h = 4e-3, V(0) = 0 and V even); a
#: single difference at h = 1e-3 loses 1e-5 to rounding for N2, whose V(h) is 6e-9 Ha. Measured:
#: 4e-10 (H2) and 1.3e-6 (N2).
CURVATURE_RTOL = 1e-5
#: D(Q) = sin Q for one occupied orbital: 1e-16 measured.
SIN_TOL = 1e-12
#: Relative bar of the collective solvers on a harmonic V: 5.4e-6 measured (3D).
SOLVER_RTOL = 1e-5
#: eV. model=None against calc_qp_energy's PSD2@TDHF: 2.6e-8 measured (bisection stop 1e-8 Ha).
DEFAULT_ROUTE_TOL = 1e-6
#: eV. Collective 1D triplet against FCI for H2: 0.83 measured at 1.5 A.
FCI_TRIPLET_TOL = 1.0
#: eV. The soft-mode PSD2 IP between 1.19 and 1.212 A: 0.034 measured.
ACROSS_TOL = 0.1
#: Pole strength the soft-mode PSD2 HOMO keeps from 0.75 to 1.212 A: 0.84 measured.
Z_MIN_SOFT = 0.7
#: eV. Collective 1D triplet against EOM-CCSD for the soft modes: 1.15 measured (O3).
EOM_TRIPLET_TOL = 1.2
#: eV. The same for H2O's stiff triplet: 1.00 measured.
EOM_STIFF_TOL = 1.1
#: The relaxed path on the RHF quadratic model: |kappa - Q d| (0 measured) and |V - mu Q^2 / 2|
#: relative to its largest value (9e-16 measured), H2 and N2.
QUADRATIC_PATH_TOL = 1e-12
#: Relative bar of the path inertia M(0) = M: 9e-16 (H2) and 3e-13 (ethylene@60) measured.
INERTIA_RTOL = 1e-10
#: eV. The variable-mass solver with a constant mass against `collective_levels`: 7e-11 periodic,
#: 1.4e-6 between walls (H2 at 1.5 A); H2/STO-3G's relaxed level against the straight one: 1.2e-6.
SOLVER_LEVEL_TOL = 1e-5
#: The constraints e^T kappa = Q and e_t^T kappa = 0 along a relaxed path: 9e-16 and 7e-15 (O3).
CONSTRAINT_TOL = 1e-12
#: Ha. V_relaxed - V_straight at the common samples: 0 measured (at Q = 0; negative elsewhere).
BOUND_TOL = 1e-10
#: Ha. How far the relaxation lowers ethylene@60's V at Q = L / 2 at least: 24.4 mHa measured (H2
#: at 1.0 A 6.8, H2CO 7.5, O3 11.9, N2 18.2, BF 3.8 to 3.9 mHa).
MID_PATH_DROP = 1e-2
#: Ha. H2/STO-3G's relaxed V against the straight path's: 7e-16 measured.
ONE_PAIR_V_TOL = 1e-12
#: eV. Relaxed 1D triplet against spin-flip EOM-CCSD for ethylene twisted by 60 degrees: -0.0075
#: measured (the straight path +0.371).
SPIN_FLIP_TOL = 0.05
#: eV. N2's relaxed level against its continuous symmetric branch: 2e-6 measured at 1 to 4 threads
#: (6.571862 against the audit's 6.57186, printed to 1e-5 eV).
N2_BRANCH_TOL = 1e-4
#: eV. H2CO's relaxed level with its saddle modes held against the one without hold rounds: 2e-8
#: measured at 1, 2 and 4 threads (the held modes are off the symmetric branch the fresh starts
#: land on).
HELD_LEVEL_TOL = 1e-5
#: eV. The 14 relaxed levels against RELAXED_LEVELS_EV: 3e-9 at most at 1, 2 and 4 threads.
RELAXED_PIN_TOL = 1e-4
#: The closure of a relaxed loop: 1 minus the overlap of the alpha determinants at -+Q*, and
#: |D(Q*)|; 4e-13 and 3.8e-8 at most measured over the 14 small systems (O3 at 1 to 4 threads).
CLOSURE_TOL = 1e-3
#: Step (in |kappa|) of the central differences of the analytic gradient that build the dense
#: constrained Hessian: the result is symmetric to 4e-9 Ha (H2CO, O3).
HESSIAN_STEP = 1e-4
#: Ha. The GCM's closed-form H kernel against `loewdin_kernels` (1.7e-13 measured, O3 and ethylene
#: too besides the test's H2O, N2 and H2O/STO-3G) and its diagonal against `path_samples` (3.4e-13,
#: H2O); the overlap, D and S^2 kernels take GCM_OVERLAP_TOL (3e-15 measured).
GCM_KERNEL_TOL = 1e-10
GCM_OVERLAP_TOL = 1e-12
#: eV. H2/STO-3G's GCM level against FCI: 3e-14 measured.
GCM_ONE_PAIR_TOL = 1e-8
#: H2/cc-pVDZ's GCM gap (eV) and coupling against the two-orbital CI, and the grid's 12 points
#: against 32: 5e-14 eV, 6e-16 and 1e-14 eV measured.
GCM_CI_TOL = 1e-10
#: <S^2> of H2's GCM states against 0 and 2: 2e-15 measured.
SPIN_PURE_TOL = 1e-12
#: eV. GCM triplet against FCI for H2/cc-pVDZ: -0.227 measured at 0.75 A.
GCM_FCI_TOL = 0.25
#: eV. 24 and 48 generator points against 32: 5.4e-4 measured (N2).
GCM_GRID_TOL = 1e-3
#: Relative bars of V''(0) = mu, the overlap's width 1/2 and h2(0) = d^T (A - B) d / 4 on the GCM
#: kernels (Richardson at s = 4e-3 and 8e-3): 2.5e-7 measured (H2O; the RHF's own convergence).
GCM_SMALL_AMPLITUDE_RTOL = 1e-5
#: eV. The GCM's first level against the audited prototype's: 7e-13 measured here; the audit's
#: independent kernels and solver agree with the prototype to 2e-9.
GCM_AUDITED_TOL = 1e-6
#: N2's even GCM ground state against the audit's independent <S^2> (printed to four digits).
GCM_SPIN_TOL = 1e-3
#: eV. GCM triplet against EOM-CCSD for H2CO and ethylene: -0.232 measured (H2CO).
GCM_EOM_TOL = 0.25
#: H2/STO-3G's (M d^T (A - B) d)^1/2 against 1, d an eigenvector of A - B: 0 measured; and the
#: least excess of a harmonic GCM level over Omega_s for a multi-orbital mode: 1.023 (H2O).
ONE_PAIR_MASS_TOL = 1e-12
PEIERLS_YOCCOZ_MIN = 1.02
#: The GCM's ladder pole against its level and X + Y = sqrt2 c d: 0 measured.
GCM_POLE_TOL = 1e-12

#: eV. N2's relaxed level along the continuous symmetric branch, from constrained minima started
#: at kappa = Q d at every sample (scipy L-BFGS-B, gradient 1e-9) with the antipode at the root of
#: cos(theta_1): the audit of the soft-mode tweak study (18 November 2026, cc-pVDZ, MP2 geometry).
N2_SYMMETRIC_BRANCH_EV = 6.57186
#: eV. The relaxed 1D level of the lowest soft mode (`relaxed_path` quantized by
#: `collective_levels`, as soft_mode_triplets(mf, '1d', 'relaxed') does): this code's own, recorded
#: on a workstation (macOS arm64, OpenBLAS 0.3.29, pyscf 2.13.0; 1, 2 and 4 threads agree to 3e-9
#: eV), cc-pVDZ at the geometries of tests/soft_mode_references.py, as a regression record. They lie
#: within 5e-4 eV of the tweak study's audit-corrected targets (H2 10.433 / 7.431 / 5.423 / 3.362 /
#: 1.217 / 0.033, N2 6.572, H2CO 6.448, H2O 8.399, ethylene 4.786, O3 2.128, twisted 2.815 / 1.686 /
#: 0.895).
RELAXED_LEVELS_EV = {'H2@0.75': 10.433259396, 'H2@1.0': 7.431148553, 'H2@1.2104': 5.423362843,
                     'H2@1.5': 3.362220483, 'H2@2.0': 1.217470329, 'H2@3.0': 0.032527948,
                     'N2': 6.571862162, 'H2CO': 6.447809713, 'H2O': 8.398823050,
                     'ethylene': 4.786455298, 'O3': 2.128051723, 'ethylene@45': 2.814563421,
                     'ethylene@60': 1.685533379, 'ethylene@70': 0.895330015}
#: Ha. The first GCM level (E_odd,0 - E_even,0) and the (even, odd) ranks of the tweak study's
#: exact-kernel prototype (gcm_exact/gcm.py: n = 32, cut 1e-8; cc-pVDZ, the MP2 geometries of
#: tests/soft_mode_references.py, RHF conv_tol 1e-10), whose audit (18 November 2026) found the
#: same levels with its own non-orthogonal kernels and solver to 2e-9 eV and <S^2> = 0.1840 for
#: N2's even ground state.
GCM_AUDITED = {'N2': (0.205921901460687, (5, 5)), 'H2CO': (0.20865924587723392, (4, 4)),
               'H2O': (0.30387773966615844, (4, 3)), 'ethylene': (0.15702348679272685, (4, 4)),
               'O3': (0.053414165075516995, (4, 4))}
GCM_AUDITED_SPIN_N2 = 0.1840

H2_DISTANCES = (0.75, 1.0, 1.212, 1.5, 2.0)
SOFT_MOLECULES = ('N2', 'H2CO', 'ethylene', 'O3')


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def relative(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(1.0, np.max(np.abs(b))))


def h2(R, basis='cc-pvdz'):
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {R}', basis=basis, unit='Angstrom', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def fci_h2(mf):
    """(triplet - singlet gap, IP) of H2 by the house's FCI solver on the RHF MO integrals (Ha)."""
    C, n = mf.mo_coeff, mf.mo_coeff.shape[1]
    h1 = C.T @ mf.get_hcore() @ C
    eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, C), n)
    fci = PyscfFCISolver()
    e_s = fci.roots(h1, eri, n, (1, 1), spin=0)[0][0]
    e_t = fci.roots(h1, eri, n, (1, 1), spin=2)[0][0]
    e_c = fci.roots(h1, eri, n, (1, 0), spin=None)[0][0]
    return e_t - e_s, e_c - e_s


def triplet_casida(mf):
    nocc = mf.mol.nelectron // 2
    lr = LinearResponseSolver(mf.mo_energy, eri_chemist=mo_eri(mf, mf.mol), spin_mode='restricted')
    return lr.build_casida_matrices(nocc, lBSE=True, triplet=True)


def check_matrices():
    ok = True
    for label, mf in (('H2/cc-pVDZ 1.0 A', h2(1.0)), ('H2O/cc-pVDZ', rhf(GEOMETRIES['H2O']))):
        A, B = SM.triplet_tdhf_matrices(mf)
        Ah, Bh = triplet_casida(mf)
        miss = max(relative(A, Ah), relative(B, Bh))
        ok &= check(miss < CROSS_PATH_RTOL, f'{label}: triplet A, B from MO blocks = the house builder',
                    f'{miss:.1e}')
    return ok


def check_split_exact():
    ok = True
    for label, mf in (('H2 1.0 A', h2(1.0)), ('N2', rhf(GEOMETRIES['N2']))):
        A, B = SM.triplet_tdhf_matrices(mf)
        split = SM.soft_split(A, B)
        mode = split['modes'][0]
        miss = (abs(mode['stiffness'] / mode['mass'] - mode['omega2'])
                / np.max(np.abs(split['omega2'])))
        ok &= check(miss < SPLIT_RTOL and abs(mode['direction'] @ mode['conjugate'] - 1) < SPLIT_RTOL,
                    f'{label}: mu / M = Omega_s^2 and d^T e = 1', f'{miss:.1e}')
        om = np.sqrt(mode['omega2'])
        c1 = np.array([1.0 / np.sqrt(2 * mode['mass'] * om)])
        dE, X, Y = SM.collective_modes(np.array([om]), c1, mode['mass'], mode['direction'],
                                       mode['conjugate'])
        w, Xh, Yh = (np.asarray(v) for v in CasidaSolver(*triplet_casida(mf)).solve())
        sign = np.sign((X[:, 0] + Y[:, 0]) @ (Xh[:, 0] + Yh[:, 0]))
        miss = max(abs(dE[0] - w[0]), np.abs(sign * X[:, 0] - Xh[:, 0]).max(),
                   np.abs(sign * Y[:, 0] - Yh[:, 0]).max())
        ok &= check(miss < HARMONIC_ROOT_TOL,
                    f'{label}: the harmonic level of the soft mode = the house Casida root 0',
                    f'{miss:.1e}')
    return ok


def check_path():
    ok = True
    h = 4e-3
    for label, mf in (('H2 1.0 A', h2(1.0)), ('N2', rhf(GEOMETRIES['N2']))):
        nocc = mf.mol.nelectron // 2
        mode = SM.soft_split(*SM.triplet_tdhf_matrices(mf))['modes'][0]
        q, V, D = SM.path_samples(mf, mode['direction'], h * (SM.SOFT_MODE_PATH_POINTS - 1))
        curv = (4 * (2 * (V[1] - V[0]) / h ** 2) - (V[2] - V[0]) / (2 * h ** 2)) / 3
        miss = abs(curv - mode['stiffness']) / abs(mode['stiffness'])
        ok &= check(miss < CURVATURE_RTOL, f"{label}: V''(0) = mu", f'{miss:.1e}')
        L, sigma = SM.path_period(mode['direction'], nocc)
        if nocc == 1:
            q, V, D = SM.path_samples(mf, mode['direction'], L)
            ok &= check(np.abs(D - np.sin(q)).max() < SIN_TOL, f'{label}: D(Q) = sin Q, period 2 pi',
                        f'{np.abs(D - np.sin(q)).max():.1e}')
        else:
            ok &= check(abs(sigma[0] - sigma[1]) < SIGMA_PAIR_TOL and abs(L / np.pi - 1.4227) < 1e-3,
                        f'{label}: half-period pi / sigma_1 with sigma_1 = sigma_2 (two pi pairs)',
                        f'L = {L / np.pi:.4f} pi, |sigma_1 - sigma_2| = {abs(sigma[0] - sigma[1]):.1e}')
    return ok


def check_solvers():
    ok = True
    M, mu = 2.0, 2.0
    q = np.linspace(0.0, np.pi, SM.SOFT_MODE_PATH_POINTS)
    for model, boundary in (('1d', 'periodic'), ('1d', 'walls'), ('3d', 'periodic')):
        dE, cm, _ = SM.collective_levels(M, q, 0.5 * mu * q ** 2, q, np.pi, model, boundary)
        miss = max(abs(dE[0] - 1.0), abs(cm[0] ** 2 - 0.25) / 0.25)
        ok &= check(miss < SOLVER_RTOL, f'harmonic V, {model} {boundary}: Omega and 1/(2 M Omega)',
                    f'{miss:.1e}')
    return ok


def psd2_poles(mf, triplets, p):
    """(res, pol) of PSD2@TDHF for orbital p with the given triplet list (omega, X, Y)."""
    nocc = mf.mol.nelectron // 2
    eps = mf.mo_energy
    eri = mo_eri(mf, mf.mol)
    lr = LinearResponseSolver(eps, eri_chemist=eri, spin_mode='restricted')
    om_s, X_s, Y_s = (np.asarray(v) for v in CasidaSolver(*lr.build_casida_matrices(nocc, lBSE=True)).solve())
    om_t, X_t, Y_t = triplets
    se = SelfEnergySolver(eps, eri_chemist=eri, spin_mode='restricted', eta=0.0)
    ev = se.self_energy_evaluator(p, nocc, om_s, se.get_chi_a(nocc, X_s, Y_s, p_state=p),
                                  se.get_chi_b_vertex(nocc, X_s, Y_s, eri_w=None, p_state=p),
                                  eigenvalues_casida_t=om_t,
                                  chiXYb_t=se.get_chi_b_vertex(nocc, X_t, Y_t, eri_w=None, p_state=p),
                                  vertex_mode='PSD2')
    return ev.poles()


def check_psd_identity():
    ok = True
    mf = h2(1.0)
    split = SM.soft_split(*SM.triplet_tdhf_matrices(mf))
    mode = split['modes'][0]
    om = np.sqrt(mode['omega2'])
    harmonic = SM.collective_modes(np.array([om]), np.array([1.0 / np.sqrt(2 * mode['mass'] * om)]),
                                   mode['mass'], mode['direction'], mode['conjugate'])
    swapped = tuple(np.concatenate([r, h], axis=-1) if r.ndim == 1 else np.hstack([r, h])
                    for r, h in zip(split['regular'], harmonic))
    plain = tuple(np.asarray(v) for v in CasidaSolver(*triplet_casida(mf)).solve())
    w = mf.mo_energy[0] + np.array([-0.3, -0.1, 0.05, 0.2, 0.6])
    sig = []
    for trip in (swapped, plain):
        res, pol = psd2_poles(mf, trip, 0)
        sig.append(np.array([np.sum(res / (x - pol)) for x in w]))
    ok &= check(relative(sig[0], sig[1]) < CROSS_PATH_RTOL,
                'H2 1.0 A: PSD2 with the soft mode at its harmonic level = PSD2@TDHF (pole sums)',
                f'{relative(sig[0], sig[1]):.1e}')
    ref = -calc_qp_energy(mf, selfenergy='PSD2', polarizability='TDHF', mode='casida', df=False,
                          eta=0.0, state='homo')
    got = -soft_mode_qp(mf, 'PSD2', model=None)['eps'][0] * HARTREE_TO_EV
    ok &= check(abs(got - ref) < DEFAULT_ROUTE_TOL, "model=None: calc_qp_energy's PSD2@TDHF HOMO",
                f'{got:.8f} vs {ref:.8f} eV')
    return ok


def check_instability():
    ok = True
    om2 = [SM.soft_split(*SM.triplet_tdhf_matrices(h2(R)))['omega2'][0] for R in (1.20, 1.212)]
    ok &= check(om2[0] > 0 > om2[1], 'H2: Omega_T^2 changes sign between 1.20 and 1.212 A',
                f'{om2[0]:.2e}, {om2[1]:.2e} Ha^2')
    mf = h2(1.212)
    try:
        CasidaSolver(*triplet_casida(mf)).solve()
        refused = False
    except CasidaInstability:
        refused = True
    rec = SM.soft_mode_triplets(mf, '1d')
    exc = rec['soft'][0]['excitation'] * HARTREE_TO_EV
    ok &= check(refused and np.isfinite(exc) and exc > 0,
                'H2 1.212 A: the house triplet Casida refuses, the collective triplet is finite',
                f'{exc:.3f} eV')
    return ok


def check_fci():
    ok = True
    for R in H2_DISTANCES:
        mf = h2(R)
        gap, _ = fci_h2(mf)
        rec = SM.soft_mode_triplets(mf, '1d')
        e1 = rec['soft'][0]['excitation']
        tda = rec['omega_tda_soft']
        miss = (e1 - gap) * HARTREE_TO_EV
        line = f'1D {e1 * HARTREE_TO_EV:.3f}, TDA {tda * HARTREE_TO_EV:.3f}, FCI {gap * HARTREE_TO_EV:.3f} eV'
        ok &= check(abs(miss) < FCI_TRIPLET_TOL, f'H2 {R} A: collective 1D triplet within '
                    f'{FCI_TRIPLET_TOL} eV of FCI', line)
        if R >= 1.0:
            ok &= check(abs(e1 - gap) < abs(tda - gap) or tda < 0, f'H2 {R} A: closer than TDA')
    return ok


def check_psd_through():
    ok = True
    ips = {}
    for R in (0.75, 1.0, 1.19, 1.212):
        mf = h2(R)
        _, ip_fci = fci_h2(mf)
        soft = soft_mode_qp(mf, 'PSD2', model='1d')
        ip_soft, z_soft = -soft['eps'][0], soft['Z'][0]
        try:
            plain = soft_mode_qp(mf, 'PSD2', model=None)
            ip_plain, z_plain = -plain['eps'][0], plain['Z'][0]
        except CasidaInstability:
            ip_plain = z_plain = None
        ips[R] = ip_soft
        ok &= check(z_soft > Z_MIN_SOFT, f'H2 {R} A: soft-mode PSD2 has a quasiparticle',
                    f'Z = {z_soft:.2f}')
        if R == 1.19:
            ok &= check(z_plain is not None and z_plain < 0.5,
                        'H2 1.19 A: PSD2@TDHF has no quasiparticle', f'Z = {z_plain:.2f}')
        if R == 1.212:
            ok &= check(ip_plain is None, 'H2 1.212 A: PSD2@TDHF is refused (CasidaInstability)')
        else:
            ok &= check(abs(ip_soft - ip_fci) < abs(ip_plain - ip_fci),
                        f'H2 {R} A: soft-mode PSD2 IP closer to FCI than PSD2@TDHF',
                        f'{ip_soft * HARTREE_TO_EV:.3f} / {ip_plain * HARTREE_TO_EV:.3f} / FCI '
                        f'{ip_fci * HARTREE_TO_EV:.3f} eV')
    moved = abs(ips[1.212] - ips[1.19]) * HARTREE_TO_EV
    ok &= check(moved < ACROSS_TOL, 'soft-mode PSD2 IP continuous across R_CF', f'{moved:.3f} eV')
    return ok


def check_molecules():
    ok = True
    flags = {}
    for name in SOFT_MOLECULES + ('H2O',):
        mf = rhf(GEOMETRIES[name])
        rec = SM.soft_mode_triplets(mf, '1d')
        flags[name] = [f['flag'] for f in rec['numerical_flags']]
        eom = EOM_CCSD[name][0]
        e1, tda = rec['soft'][0]['excitation'], rec['omega_tda_soft']
        miss = (e1 - eom) * HARTREE_TO_EV
        line = (f'1D {e1 * HARTREE_TO_EV:.3f}, TDA {tda * HARTREE_TO_EV:.3f}, '
                f'EOM-CCSD {eom * HARTREE_TO_EV:.3f} eV')
        if name == 'H2O':
            ok &= check(abs(miss) < EOM_STIFF_TOL, 'H2O (stiff control): collective 1D triplet '
                        f'within {EOM_STIFF_TOL} eV of EOM-CCSD', line)
        else:
            ok &= check(abs(miss) < EOM_TRIPLET_TOL and (abs(e1 - eom) < abs(tda - eom) or tda < 0),
                        f'{name}: collective 1D triplet within {EOM_TRIPLET_TOL} eV of EOM-CCSD '
                        'and closer than TDA', line)
    ok &= check('soft_modes_coupled' in flags['O3'] and not flags['N2'],
                "O3's three soft modes are flagged 'soft_modes_coupled', N2 raises nothing",
                f"O3 {flags['O3']}")
    return ok


def check_refusals():
    ok = True
    mol = gto.M(atom='H 0 0 0; H 0 0 1.0', basis='cc-pvdz', verbose=0)
    uhf = scf.UHF(mol).run()
    rks = scf.RKS(mol).run(xc='pbe')
    dfr = scf.RHF(mol).density_fit().run()
    raw = scf.RHF(mol)
    cases = [('UHF', lambda: SM.soft_mode_triplets(uhf), NotImplementedError),
             ('RKS', lambda: SM.soft_mode_triplets(rks), ValueError),
             ('density fitting', lambda: SM.soft_mode_triplets(dfr), ValueError),
             ('unconverged', lambda: SM.soft_mode_triplets(raw), ValueError),
             ("model '2d'", lambda: SM.soft_mode_triplets(h2(1.0), '2d'), ValueError),
             ("boundary 'open'", lambda: SM.collective_levels(1.0, np.linspace(0, 1, 5),
                                                             np.zeros(5), np.zeros(5), 1.0, '1d',
                                                             'open'), ValueError),
             ("selfenergy 'GW'", lambda: soft_mode_qp(h2(1.0), 'GW'), ValueError),
             ('A - B indefinite', lambda: SM.soft_split(np.diag([1.0, -0.5]), np.zeros((2, 2))),
              CasidaInstability)]
    for label, call, kind in cases:
        try:
            call()
            raised = None
        except Exception as exc:   # noqa: BLE001 -- the class is what is checked
            raised = exc
        ok &= check(isinstance(raised, kind), f'refused: {label}', type(raised).__name__)
    return ok


def check_edge_flag():
    saved = SM.SOFT_MODE_EDGE_WEIGHT_MAX
    SM.SOFT_MODE_EDGE_WEIGHT_MAX = 0.0
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            rec = SM.soft_mode_triplets(h2(1.0), '1d')
    finally:
        SM.SOFT_MODE_EDGE_WEIGHT_MAX = saved
    names = [f['flag'] for f in rec['numerical_flags']]
    warned = any(issubclass(w.category, SM.SoftModeEdgeWeight) for w in caught)
    return check(names == ['soft_mode_edge_weight'] and warned,
                 "an edge-weight bar of zero raises 'soft_mode_edge_weight'", f'{names}')


def soft_setup(mf):
    """(nocc, A, B, mode, held) of the lowest soft mode, its other soft modes held."""
    A, B = SM.triplet_tdhf_matrices(mf)
    split = SM.soft_split(A, B)
    s = split['soft'][0]
    held = [split['modes'][t]['conjugate'] for t in split['soft'] if t != s]
    return mf.mol.nelectron // 2, A, B, split['modes'][s], held


@functools.lru_cache(maxsize=None)
def relaxed_lowest(label):
    """(mf, its `soft_setup`, the `relaxed_path` record of its lowest soft mode, that mode's first
    relaxed level (Ha) as soft_mode_triplets(mf, '1d', 'relaxed') quantizes it) for 'H2@R'
    (cc-pVDZ) or a label of tests/soft_mode_references.py."""
    mf = h2(float(label[3:])) if label.startswith('H2@') else rhf(geometry(label))
    setup = soft_setup(mf)
    nocc, A, B, mode, held = setup
    rel = SM.relaxed_path(mf, A, B, mode, held)
    level = SM.collective_levels(mode['mass'], rel['q'], rel['V'], rel['D'], rel['antipode'], '1d',
                                 mass_profile=(rel['q_mass'], rel['mass']))[0][0]
    return mf, setup, rel, level


def patched_relaxed(mf, patch):
    """(the record of soft_mode_triplets(mf, '1d', 'relaxed') with the module constants in patch
    set, the `relaxed_path` record of its lowest soft mode, the warnings raised)."""
    saved = {k: getattr(SM, k) for k in patch}
    paths, original = [], SM.relaxed_path

    def kept(*args):
        paths.append(original(*args))
        return paths[-1]
    for k, v in patch.items():
        setattr(SM, k, v)
    SM.relaxed_path = kept
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            rec = SM.soft_mode_triplets(mf, '1d', 'relaxed')
    finally:
        SM.relaxed_path = original
        for k, v in saved.items():
            setattr(SM, k, v)
    return rec, paths[0], caught


def lowest_constrained_curvature(mf, setup, rel):
    """The lowest eigenvalue (Ha) of Z^T H Z at the relaxed sample nearest Q*: H from central
    differences (HESSIAN_STEP) of the analytic gradient of `_TripletEnergy`, Z an orthonormal basis
    of the rotations off e, off the other soft modes and off the TDHF modes rel['held_modes']
    (their conjugates (A - B)^-1 (A - B)^1/2 t_n from this function's own eigenvectors t_n of
    (A - B)^1/2 (A + B) (A - B)^1/2)."""
    nocc, A, B, mode, held = setup
    w, U = np.linalg.eigh(A - B)
    S = (U * np.sqrt(w)) @ U.T
    t = np.linalg.eigh(S @ (A + B) @ S)[1]
    extra = [np.linalg.solve(A - B, S @ t[:, n]) for n in rel['held_modes']]
    rows = np.column_stack([mode['conjugate']] + held + extra)
    Z = np.linalg.qr(rows, mode='complete')[0][:, rows.shape[1]:]
    kappa = rel['kappa'][int(np.argmin(np.abs(rel['q'][:rel['relaxed']] - rel['antipode'])))]
    energy = SM._TripletEnergy(mf, nocc)
    H = np.array([Z.T @ (energy(kappa + HESSIAN_STEP * z)[1] - energy(kappa - HESSIAN_STEP * z)[1])
                  for z in Z.T]) / (2.0 * HESSIAN_STEP)
    return float(np.linalg.eigvalsh(0.5 * (H + H.T))[0])


def check_relaxed_harmonic():
    ok = True
    for label, mf in (('H2 1.0 A', h2(1.0)), ('N2', rhf(GEOMETRIES['N2']))):
        nocc, A, B, mode, held = soft_setup(mf)
        K = A + B
        minima = SM._ConstrainedMinima(lambda k: (0.5 * k @ K @ k, K @ k), mode['direction'],
                                       mode['conjugate'], held, K)
        L = SM.path_period(mode['direction'], nocc)[0]
        path = SM._relaxed_samples(minima, nocc, L / (SM.SOFT_MODE_RELAXED_POINTS - 1))
        harmonic = 0.5 * mode['stiffness'] * path['q'] ** 2
        bent = np.abs(path['kappa'] - np.outer(path['q'], mode['direction'])).max()
        miss = np.abs(path['V'] - harmonic).max() / np.abs(harmonic).max()
        ok &= check(bent < QUADRATIC_PATH_TOL and miss < QUADRATIC_PATH_TOL,
                    f'{label}: on the quadratic model the relaxed path is kappa = Q d, '
                    'V = mu Q^2 / 2', f'{bent:.1e}, {miss:.1e}')
    M, mu = 2.0, 2.0
    q = np.linspace(0.0, np.pi, SM.SOFT_MODE_PATH_POINTS)
    for boundary in SM.BOUNDARIES:
        dE, cm, _ = SM.collective_levels(M, q, 0.5 * mu * q ** 2, q, np.pi, '1d', boundary,
                                         mass_profile=(q, np.full(q.size, M)))
        miss = max(abs(dE[0] - 1.0), abs(cm[0] ** 2 - 0.25) / 0.25)
        ok &= check(miss < SOLVER_RTOL, f'harmonic V, variable-mass solver {boundary}: Omega and '
                    '1/(2 M Omega)', f'{miss:.1e}')
    mf = h2(1.5)
    nocc, A, B, mode, held = soft_setup(mf)
    L = SM.path_period(mode['direction'], nocc)[0]
    q, V, D = SM.path_samples(mf, mode['direction'], L)
    for boundary in SM.BOUNDARIES:
        plain = SM.collective_levels(mode['mass'], q, V, D, L, '1d', boundary)[0][0]
        flat = SM.collective_levels(mode['mass'], q, V, D, L, '1d', boundary,
                                    mass_profile=(q, np.full(q.size, mode['mass'])))[0][0]
        miss = abs(flat - plain) * HARTREE_TO_EV
        ok &= check(miss < SOLVER_LEVEL_TOL, f'H2 1.5 A: a constant mass profile gives '
                    f'collective_levels ({boundary})', f'{miss:.1e} eV')
    h = 4e-3
    for label in ('H2@1.0', 'ethylene@60'):
        mf, _, rel, _ = relaxed_lowest(label)
        nocc, A, B, mode, held = soft_setup(mf)
        minima = SM._ConstrainedMinima(SM._TripletEnergy(mf, nocc), mode['direction'],
                                       mode['conjugate'], held, A + B)
        V = [0.0] + [minima.minimum(Q, np.zeros(minima.size), SM.SOFT_MODE_RELAXED_GRADIENT)[1]
                     for Q in (h, 2 * h)]
        curv = (4 * (2 * (V[1] - V[0]) / h ** 2) - (V[2] - V[0]) / (2 * h ** 2)) / 3
        curv_miss = abs(curv / mode['stiffness'] - 1)
        mass_miss = abs(rel['mass'][0] / mode['mass'] - 1)
        ok &= check(curv_miss < CURVATURE_RTOL and mass_miss < INERTIA_RTOL,
                    f"{label}: relaxed V''(0) = mu and M(0) = M, so V''(0) / M(0) = Omega_s^2",
                    f'{curv_miss:.1e}, {mass_miss:.1e}')
    return ok


def check_relaxed_one_pair():
    ok = True
    for R in (1.0, 2.5):
        mf = h2(R, 'sto-3g')
        nocc, A, B, mode, held = soft_setup(mf)
        rel = SM.relaxed_path(mf, A, B, mode, held)
        L = SM.path_period(mode['direction'], nocc)[0]
        V = SM.path_samples(mf, mode['direction'], L)[1]
        v_miss = np.abs(rel['V'][:2 * V.size - 1:2] - V).max()
        m_miss = np.abs(rel['mass'] / mode['mass'] - 1).max()
        q_miss = abs(rel['antipode'] / L - 1)
        levels = [SM.soft_mode_triplets(mf, '1d', p)['soft'][0]['excitation'] for p in SM.PATHS]
        e_miss = abs(levels[1] - levels[0]) * HARTREE_TO_EV
        ok &= check(max(v_miss, m_miss, q_miss) < ONE_PAIR_V_TOL and e_miss < SOLVER_LEVEL_TOL,
                    f'H2/STO-3G {R} A: the relaxed path is the straight one (V, M, Q*) and so is '
                    'its level', f'{v_miss:.1e} Ha, {m_miss:.1e}, {q_miss:.1e}; {e_miss:.1e} eV')
    return ok


def check_relaxed_molecules():
    ok = True
    mf, (nocc, A, B, mode, held), rel, level = relaxed_lowest('ethylene@60')
    L = SM.path_period(mode['direction'], nocc)[0]
    V = SM.path_samples(mf, mode['direction'], L)[1]
    common = np.arange(0, rel['q'].size, 2)
    common = common[rel['q'][common] <= min(L, rel['antipode'])]
    rise = (rel['V'][common] - V[common // 2]).max()
    mid = (SM.SOFT_MODE_PATH_POINTS - 1) // 2
    drop = V[mid] - rel['V'][2 * mid]
    plane = np.abs(rel['kappa'] @ mode['conjugate'] - rel['q']).max()
    ok &= check(rise < BOUND_TOL and drop > MID_PATH_DROP and plane < CONSTRAINT_TOL,
                'ethylene@60: V_relaxed <= V_straight up to Q*, lower by more than '
                f'{MID_PATH_DROP * 1e3:.0f} mHa at L / 2, e^T kappa = Q',
                f'{rise:.1e} Ha, {drop * 1e3:.1f} mHa, {plane:.1e}')
    e1 = level * HARTREE_TO_EV
    e_straight = SM.soft_mode_triplets(mf, '1d')['soft'][0]['excitation'] * HARTREE_TO_EV
    sf = SPIN_FLIP_EV['ethylene@60']
    ok &= check(abs(e1 - sf) < SPIN_FLIP_TOL and abs(e1 - sf) < abs(e_straight - sf),
                f'ethylene@60: relaxed 1D triplet within {SPIN_FLIP_TOL} eV of spin-flip EOM-CCSD, '
                'closer than the straight path',
                f"relaxed {e1:.4f}, straight {e_straight:.3f}, SF {sf:.3f} eV; Q*/L "
                f"{rel['antipode'] / L:.4f}, M(Q*)/M {rel['mass_antipode'] / mode['mass']:.3f}")
    mf, (nocc, A, B, mode, held), rel, _ = relaxed_lowest('O3')
    plane = np.abs(rel['kappa'] @ mode['conjugate'] - rel['q']).max()
    others = max(np.abs(rel['kappa'] @ e).max() for e in held)
    ok &= check(len(held) == 2 and max(plane, others) < CONSTRAINT_TOL,
                "O3's lowest mode: e^T kappa = Q, e_t^T kappa = 0 for its two other soft modes",
                f'{plane:.1e}, {others:.1e}')
    return ok


def check_relaxed_saddles():
    ok = True
    mf = rhf(GEOMETRIES['H2CO'])
    rec, bare, caught = patched_relaxed(mf, {'SOFT_MODE_RELAXED_HOLD_ROUNDS': 0})
    L = rec['soft'][0]['half_period']
    names = [f['flag'] for f in rec['numerical_flags']]
    warned = any(issubclass(w.category, SM.SoftModePathShort) for w in caught)
    ok &= check(bare['curvature'] < -SM.SOFT_MODE_RELAXED_SADDLE_BAR and not bare['held_modes']
                and names == ['soft_mode_path_branch'] and warned,
                "H2CO without hold rounds: a negative curvature at its probes, flagged "
                "'soft_mode_path_branch'",
                f"fresh starts at Q/L {np.round(bare['restarted'] / L, 4).tolist()}, curvature "
                f"{bare['curvature']:+.3f}, flags {names}")
    for label, modes in (('H2CO', {1}), ('N2', {1, 2})):
        _, (nocc, A, B, mode, held), rel, level = relaxed_lowest(label)
        clean = (modes <= set(rel['held_modes']) and not rel['restarted'].size
                 and rel['frozen_from'] is None and rel['curvature'] > 0)
        if label == 'H2CO':
            miss = abs(level - rec['soft'][0]['excitation']) * HARTREE_TO_EV
            ok &= check(clean and miss < HELD_LEVEL_TOL,
                        'H2CO: its mode 1 held, a constrained minimum through Q*, at the level '
                        'without the holds', f"held {rel['held_modes']}, curvature "
                        f"{rel['curvature']:+.3f}; {level * HARTREE_TO_EV:.6f} eV ({miss:.1e})")
        else:
            miss = abs(level * HARTREE_TO_EV - N2_SYMMETRIC_BRANCH_EV)
            ok &= check(clean and miss < N2_BRANCH_TOL,
                        "N2: its 3Delta_u pair held, unfrozen through Q*, at the symmetric "
                        f"branch's level ({N2_SYMMETRIC_BRANCH_EV} eV)",
                        f"held {rel['held_modes']}, curvature {rel['curvature']:+.3f}; "
                        f"{level * HARTREE_TO_EV:.6f} eV ({miss:.1e})")
    return ok


def check_relaxed_levels():
    ok = True
    for label, want in RELAXED_LEVELS_EV.items():
        _, _, rel, level = relaxed_lowest(label)
        miss = abs(level * HARTREE_TO_EV - want)
        shut = max(1.0 - rel['closure_overlap'], abs(rel['coupling_antipode']))
        clean = (rel['frozen_from'] is None and rel['converged'] and not rel['restarted'].size
                 and rel['curvature'] > 0)
        ok &= check(miss < RELAXED_PIN_TOL and shut < CLOSURE_TOL and clean,
                    f'{label}: the recorded relaxed level, a closed path, unfrozen, converged, no '
                    'fresh start', f"{level * HARTREE_TO_EV:.6f} eV ({miss:.1e}), closure "
                    f"{shut:.1e}, curvature {rel['curvature']:+.3f}")
    for label in ('H2CO', 'O3'):
        mf, setup, rel, _ = relaxed_lowest(label)
        lowest = lowest_constrained_curvature(mf, setup, rel)
        ok &= check(lowest > 0, f'{label}: the constrained Hessian at the relaxed sample nearest '
                    'Q* is positive definite (a true minimum)',
                    f'lowest eigenvalue {lowest:+.4f} Ha')
    return ok


def check_relaxed_qp_and_refusals():
    ok = True
    res = soft_mode_qp(h2(1.0), 'PSD2', path='relaxed')
    ok &= check(res['Z'][0] > Z_MIN_SOFT and res['triplets']['soft'][0]['path'] == 'relaxed',
                "H2 1.0 A: soft_mode_qp passes path='relaxed' through",
                f"IP {-res['eps'][0] * HARTREE_TO_EV:.3f} eV, Z = {res['Z'][0]:.2f}")
    uhf = scf.UHF(gto.M(atom='H 0 0 0; H 0 0 1.0', basis='cc-pvdz', verbose=0)).run()
    mf, q = h2(1.0), np.linspace(0.0, 1.0, 5)
    cases = [("path 'curved' (a UHF reference)",
              lambda: SM.soft_mode_triplets(uhf, '1d', 'curved')),
             ("path 'relaxed' with model '3d'", lambda: SM.soft_mode_triplets(mf, '3d', 'relaxed')),
             ("soft_mode_qp: path 'curved'", lambda: soft_mode_qp(mf, 'PSD2', path='curved')),
             ("soft_mode_qp: path 'relaxed' with model None",
              lambda: soft_mode_qp(mf, 'PSD2', model=None, path='relaxed')),
             ('a mass profile in the 3D model',
              lambda: SM.collective_levels(1.0, q, q, q, 1.0, '3d', mass_profile=(q, 1.0 + q)))]
    for label, call in cases:
        try:
            call()
            raised = None
        except Exception as exc:   # noqa: BLE001 -- the class is what is checked
            raised = exc
        ok &= check(isinstance(raised, ValueError)
                    and str(raised).startswith(('path = ', 'model = ')),
                    f'refused: {label}', type(raised).__name__)
    return ok


def check_relaxed_flags():
    ok = True
    mf = h2(1.0)
    for name, patch, want in (
            ('one minimiser step per sample', {'SOFT_MODE_RELAXED_ITERATIONS': 1},
             'soft_mode_path_not_converged'),
            ('a branch test that rejects every step',
             {'SOFT_MODE_RELAXED_JUMP_RATIO': 0.0, 'SOFT_MODE_RELAXED_JUMP_SLACK': 0.0},
             'soft_mode_path_frozen'),
            ('a closure bar above 1', {'SOFT_MODE_RELAXED_CLOSURE_MIN': 1.1},
             'soft_mode_path_closure')):
        rec, _, caught = patched_relaxed(mf, patch)
        names = [f['flag'] for f in rec['numerical_flags']]
        warned = any(issubclass(w.category, SM.SoftModePathShort) for w in caught)
        soft = rec['soft'][0]
        if want == 'soft_mode_path_frozen':
            straight = SM.soft_mode_triplets(mf, '1d')['soft'][0]['excitation']
            done = (soft['frozen_from'] == 0.0
                    and abs(soft['excitation'] - straight) * HARTREE_TO_EV < SOLVER_LEVEL_TOL)
        elif want == 'soft_mode_path_not_converged':
            done = not soft['converged']
        else:
            done = (abs(soft['excitation'] * HARTREE_TO_EV - RELAXED_LEVELS_EV['H2@1.0'])
                    < RELAXED_PIN_TOL)
        ok &= check(names == [want] and warned and done, f"{name} raises '{want}'", f'{names}')
    return ok


@functools.lru_cache(maxsize=None)
def gcm_run(label, basis='cc-pvdz'):
    """(mf, the record of soft_mode_triplets(mf, 'gcm')) for 'H2@R' or a label of
    tests/soft_mode_references.py."""
    mf = h2(float(label[3:]), basis) if label.startswith('H2@') else rhf(geometry(label), basis)
    return mf, SM.soft_mode_triplets(mf, 'gcm')


def lowest_direction(mf):
    """(nocc, the lowest soft mode's record from `soft_split`, A, B)."""
    A, B = SM.triplet_tdhf_matrices(mf)
    split = SM.soft_split(A, B)
    return mf.mol.nelectron // 2, split['modes'][split['soft'][0]], A, B


def loewdin_kernels(mf, direction, pairs):
    """(N, H - E_RHF N, <D_alpha>, <S^2>) between the straight path's determinants at the (Q, Q')
    pairs, by the generic non-orthogonal rules in the RHF MO basis (rotations by expm, as in
    `path_samples`): transition densities G = y (x^T y)^-1 x^T per spin, the two-body energy from
    them by Wick's theorem, S^2 = N_beta - tr(G_alpha G_beta) at M_s = 0."""
    nocc, C = mf.mol.nelectron // 2, mf.mo_coeff
    n = C.shape[1]
    h = C.T @ mf.get_hcore() @ C
    eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, C), n)
    dm = np.reshape(direction, (nocc, n - nocc))
    Dmo = np.zeros((n, n))
    Dmo[:nocc, nocc:], Dmo[nocc:, :nocc] = dm, dm.T

    def occupied(Q):
        K = np.zeros((n, n))
        K[nocc:, :nocc], K[:nocc, nocc:] = 0.5 * Q * dm.T, -0.5 * Q * dm
        U = scipy.linalg.expm(K)
        return U[:, :nocc], U.T[:, :nocc]
    out = []
    for q1, q2 in pairs:
        N, G = 1.0, []
        for x, y in zip(occupied(q1), occupied(q2)):
            S = x.T @ y
            N *= np.linalg.det(S)
            G.append(y @ np.linalg.solve(S, x.T))
        Gt = G[0] + G[1]
        E = (mf.energy_nuc() + np.einsum('pq,qp', h, Gt)
             + 0.5 * np.einsum('pqrs,qp,sr', eri, Gt, Gt)
             - 0.5 * sum(np.einsum('pqrs,sp,qr', eri, g, g) for g in G))
        out.append((N, N * (E - mf.e_tot), N * np.einsum('pq,qp', Dmo, G[0]),
                    N * (nocc - np.trace(G[0] @ G[1]))))
    return np.array(out).T


def casci_two_orbitals(mf, direction):
    """(triplet - singlet gap (Ha), |<T0|D_alpha|S>|) of H2 in the two orbitals {g, v_d},
    v_d = sum_a d_a phi_a: the triplet h_gg + h_vv + (gg|vv) - (gv|vg) above the lowest root
    (vector c) of [[2 h_gg + (gg|gg), (gv|gv)], [(gv|gv), 2 h_vv + (vv|vv)]], coupled by
    |c_g - c_v| / sqrt2."""
    C = mf.mo_coeff
    orb = np.stack([C[:, 0], C[:, 1:] @ direction], axis=1)
    h = orb.T @ mf.get_hcore() @ orb
    g = ao2mo.restore(1, ao2mo.kernel(mf.mol, orb), 2)
    w, c = np.linalg.eigh(np.array([[2 * h[0, 0] + g[0, 0, 0, 0], g[0, 1, 0, 1]],
                                    [g[0, 1, 0, 1], 2 * h[1, 1] + g[1, 1, 1, 1]]]))
    triplet = h[0, 0] + h[1, 1] + g[0, 0, 1, 1] - g[0, 1, 1, 0]
    return triplet - w[0], abs(c[0, 0] - c[1, 0]) / np.sqrt(2.0)


def check_gcm_kernels():
    ok = True
    rng = np.random.default_rng(7)
    for label, basis in (('H2O', 'sto-3g'), ('H2O', 'cc-pvdz'), ('N2', 'cc-pvdz')):
        mf = gcm_run(label, basis)[0]
        nocc, mode, _, _ = lowest_direction(mf)
        L, sigma = SM.path_period(mode['direction'], nocc)
        pairs = rng.uniform(-L, L, (8, 2))
        want = loewdin_kernels(mf, mode['direction'], pairs)
        theta = 0.5 * pairs[:, :, None] * sigma
        got = np.array(SM._GCMKernels(mf, mode['direction'], nocc).elements(theta[:, 0],
                                                                             theta[:, 1]))
        miss = np.abs(got - want).max(axis=1)
        ok &= check(miss[1] < GCM_KERNEL_TOL and max(miss[[0, 2, 3]]) < GCM_OVERLAP_TOL,
                    f'{label}/{basis} ({nocc} occupied, {sigma.size} pairs): the closed-form GCM '
                    'kernels = the generic non-orthogonal ones',
                    f'N {miss[0]:.1e}, H {miss[1]:.1e} Ha, D {miss[2]:.1e}, S^2 {miss[3]:.1e}')
    mf = gcm_run('H2O')[0]
    nocc, mode, _, _ = lowest_direction(mf)
    L, sigma = SM.path_period(mode['direction'], nocc)
    q, V, D = SM.path_samples(mf, mode['direction'], L)
    theta = 0.5 * np.outer(q, sigma)
    _, H, Dk, _ = SM._GCMKernels(mf, mode['direction'], nocc).elements(theta, theta)
    miss = max(np.abs(H - V).max(), np.abs(Dk - D).max())
    ok &= check(miss < GCM_KERNEL_TOL, "H2O: the kernels' diagonal = path_samples' V and D",
                f'{miss:.1e}')
    return ok


def check_gcm_h2():
    ok = True
    for R in (0.75, 1.0, 1.5, 2.5):
        mf, rec = gcm_run(f'H2@{R}', 'sto-3g')
        gap = fci_h2(mf)[0]
        miss = abs(rec['soft'][0]['excitation'] - gap) * HARTREE_TO_EV
        ok &= check(miss < GCM_ONE_PAIR_TOL and rec['omega'].size == 1,
                    f'H2/STO-3G {R} A: the GCM triplet = FCI, one level', f'{miss:.1e} eV')
    saved = SM.SOFT_MODE_GCM_POINTS
    for R in H2_DISTANCES:
        mf, rec = gcm_run(f'H2@{R}')
        soft = rec['soft'][0]
        d = lowest_direction(mf)[1]['direction']
        gap, c = casci_two_orbitals(mf, d)
        SM.SOFT_MODE_GCM_POINTS = 12
        try:
            coarse = SM.gcm_levels(mf, d)['excitations'][0]
        finally:
            SM.SOFT_MODE_GCM_POINTS = saved
        e_miss = max(abs(soft['excitation'] - gap),
                     abs(coarse - soft['excitation'])) * HARTREE_TO_EV
        spin = max(abs(soft['spin_square'][0]), abs(soft['spin_square'][1] - 2.0))
        ok &= check(e_miss < GCM_CI_TOL and abs(soft['coupling'] - c) < GCM_CI_TOL
                    and spin < SPIN_PURE_TOL and soft['ranks'] == (2, 1),
                    f'H2/cc-pVDZ {R} A: the GCM = the CI in {{g, v_d}} (gap, coupling), spin-pure, '
                    'on any grid', f'{e_miss:.1e} eV, {abs(soft["coupling"] - c):.1e}, {spin:.1e}')
        fci = fci_h2(mf)[0]
        miss = (soft['excitation'] - fci) * HARTREE_TO_EV
        flat = SM.soft_mode_triplets(mf, '1d')['soft'][0]['excitation']
        closer = R < 1.0 or abs(soft['excitation'] - fci) < abs(flat - fci)
        ok &= check(abs(miss) < GCM_FCI_TOL and closer,
                    f'H2/cc-pVDZ {R} A: the GCM triplet within {GCM_FCI_TOL} eV of FCI'
                    + (', closer than the 1D model' if R >= 1.0 else ''),
                    f'GCM {soft["excitation"] * HARTREE_TO_EV:.3f}, 1D {flat * HARTREE_TO_EV:.3f}, '
                    f'FCI {fci * HARTREE_TO_EV:.3f} eV')
    return ok


def check_gcm_grid_and_harmonic():
    ok = True
    saved = SM.SOFT_MODE_GCM_POINTS
    for label in ('N2', 'H2O'):
        mf, rec = gcm_run(label)
        d = lowest_direction(mf)[1]['direction']
        moved, ranks = [], []
        for n in (24, 48):
            SM.SOFT_MODE_GCM_POINTS = n
            try:
                g = SM.gcm_levels(mf, d)
            finally:
                SM.SOFT_MODE_GCM_POINTS = saved
            moved.append(abs(g['excitations'][0] - rec['soft'][0]['excitation']) * HARTREE_TO_EV)
            ranks.append(g['ranks'])
        ok &= check(max(moved) < GCM_GRID_TOL and ranks == [rec['soft'][0]['ranks']] * 2,
                    f'{label}: 24 and 48 generator points give the GCM level of 32',
                    f'{moved[0]:.1e}, {moved[1]:.1e} eV, ranks {ranks}')
    for label, mf in (('H2/STO-3G 1.0 A', h2(1.0, 'sto-3g')), ('H2/cc-pVDZ 1.0 A', h2(1.0)),
                      ('H2O', gcm_run('H2O')[0]), ('N2', gcm_run('N2')[0])):
        nocc, mode, A, B = lowest_direction(mf)
        d, sigma = mode['direction'], SM.path_period(mode['direction'], nocc)[1]
        kernels = SM._GCMKernels(mf, d, nocc)
        V, width, h2_ = [], [], []
        for s in (4e-3, 8e-3):
            N, H, _, _ = kernels.elements(np.outer([0.5 * s, s], 0.5 * sigma),
                                          np.outer([-0.5 * s, s], 0.5 * sigma))
            V.append(2 * H[1] / s ** 2)
            width.append(-2 * np.log(N[0]) / s ** 2)
            h2_.append(-2 * H[0] / N[0] / s ** 2)
        amb = d @ (A - B) @ d
        misses = [abs((4 * V[0] - V[1]) / 3 / mode['stiffness'] - 1),
                  abs((4 * width[0] - width[1]) / 3 / 0.5 - 1),
                  abs((4 * h2_[0] - h2_[1]) / 3 / (amb / 4) - 1)]
        ratio = np.sqrt(mode['mass'] * amb)
        one_pair = nocc * (mf.mo_coeff.shape[1] - nocc) == 1
        ok &= check(max(misses) < GCM_SMALL_AMPLITUDE_RTOL
                    and (abs(ratio - 1) < ONE_PAIR_MASS_TOL if one_pair
                         else ratio > PEIERLS_YOCCOZ_MIN),
                    f"{label}: GCM kernels at small amplitude: V''(0) = mu, width 1/2, h2(0) = "
                    'd^T(A-B)d/4 (the Peierls-Yoccoz mass)',
                    f'{max(misses):.1e}; harmonic level / Omega_s = {ratio:.4f}')
    return ok


def check_gcm_molecules():
    ok = True
    for name, (level, ranks) in GCM_AUDITED.items():
        mf, rec = gcm_run(name)
        soft = rec['soft'][0]
        names = [f['flag'] for f in rec['numerical_flags']]
        miss = abs(soft['excitation'] - level) * HARTREE_TO_EV
        want = {'N2': ['soft_mode_gcm_degenerate_pairs'],
                'O3': ['soft_modes_coupled']}.get(name, [])
        line = (f"{soft['excitation'] * HARTREE_TO_EV:.5f} eV, ranks {soft['ranks']}, <S^2> "
                f"{soft['spin_square'][0]:.4f}, flags {names}")
        good = miss < GCM_AUDITED_TOL and soft['ranks'] == ranks and names == want
        if name == 'N2':
            good &= abs(soft['spin_square'][0] - GCM_AUDITED_SPIN_N2) < GCM_SPIN_TOL
        ok &= check(good, f'{name}: the GCM level = the audited prototype, its ranks and flags',
                    line)
        if name in ('H2CO', 'ethylene'):
            eom = EOM_CCSD[name][0]
            ok &= check(abs(soft['excitation'] - eom) * HARTREE_TO_EV < GCM_EOM_TOL,
                        f'{name}: the GCM triplet within {GCM_EOM_TOL} eV of EOM-CCSD',
                        f'{(soft["excitation"] - eom) * HARTREE_TO_EV:+.3f} eV')
    return ok


def check_gcm_qp_refusals_flags():
    ok = True
    res = soft_mode_qp(h2(1.0), 'PSD2', model='gcm')
    trip = res['triplets']
    soft = trip['soft'][0]
    xpy = trip['X'][:, -1] + trip['Y'][:, -1]
    d = lowest_direction(h2(1.0))[1]['direction']
    pole = max(abs(trip['omega'][-1] - soft['excitation']),
               np.abs(np.abs(xpy) - np.sqrt(2.0) * soft['coupling'] * np.abs(d)).max())
    ok &= check(res['Z'][0] > Z_MIN_SOFT and pole < GCM_POLE_TOL,
                "H2 1.0 A: soft_mode_qp passes model='gcm' through, its pole at the GCM level "
                'with X + Y = sqrt2 c d', f"IP {-res['eps'][0] * HARTREE_TO_EV:.3f} eV, "
                f"Z = {res['Z'][0]:.2f}, pole {pole:.1e}")
    uhf = scf.UHF(gto.M(atom='H 0 0 0; H 0 0 1.0', basis='cc-pvdz', verbose=0)).run()
    mf, q = h2(1.0), np.linspace(0.0, 1.0, 5)
    cases = [("path 'relaxed' with model 'gcm' (a UHF reference)",
              lambda: SM.soft_mode_triplets(uhf, 'gcm', 'relaxed'), NotImplementedError,
              "path = 'relaxed' with model = 'gcm'"),
             ("soft_mode_qp: path 'relaxed' with model 'gcm'",
              lambda: soft_mode_qp(uhf, 'PSD2', model='gcm', path='relaxed'), NotImplementedError,
              "path = 'relaxed' with model = 'gcm'"),
             ("collective_levels with model 'gcm'",
              lambda: SM.collective_levels(1.0, q, q, q, 1.0, 'gcm'), ValueError, 'model = '),
             ("collective_levels with model '2d', naming only the models it solves",
              lambda: SM.collective_levels(1.0, q, q, q, 1.0, '2d'), ValueError,
              "model = '2d': one of ('1d', '3d')")]
    for label, call, kind, start in cases:
        try:
            call()
            raised = None
        except Exception as exc:   # noqa: BLE001 -- the class is what is checked
            raised = exc
        ok &= check(isinstance(raised, kind) and str(raised).startswith(start),
                    f'refused: {label}', type(raised).__name__)
    saved = SM.SOFT_MODE_GCM_OVERLAP_CUT
    for label, mf, cut, want in (('N2', gcm_run('N2')[0], saved, 'soft_mode_gcm_degenerate_pairs'),
                                 ('H2 1.0 A, an overlap cut of 0.5', h2(1.0), 0.5,
                                  'soft_mode_gcm_few_states')):
        SM.SOFT_MODE_GCM_OVERLAP_CUT = cut
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                rec = SM.soft_mode_triplets(mf, 'gcm')
        finally:
            SM.SOFT_MODE_GCM_OVERLAP_CUT = saved
        names = [f['flag'] for f in rec['numerical_flags']]
        warned = any(issubclass(w.category, SM.SoftModeGCMShort) for w in caught)
        ok &= check(names == [want] and warned, f"{label}: raises '{want}'",
                    f"{names}, ranks {rec['soft'][0]['ranks']}")
    return ok


def run():
    warnings.simplefilter('ignore')
    ok = True
    for title, fn in (('1. triplet matrices', check_matrices), ('2. the split is exact', check_split_exact),
                      ('3. the RHF->UHF path', check_path), ('4. collective solvers', check_solvers),
                      ('5. PSD2 identities', check_psd_identity),
                      ('6. through the instability', check_instability),
                      ('7. H2 triplet against FCI', check_fci),
                      ('8. PSD2 through the instability', check_psd_through),
                      ('9. molecules against EOM-CCSD', check_molecules),
                      ('10. refusals', check_refusals), ('11. flags', check_edge_flag),
                      ('12. the relaxed path: harmonic limit and solver', check_relaxed_harmonic),
                      ('13. the relaxed path: one pair', check_relaxed_one_pair),
                      ('14. the relaxed path: ethylene@60 and O3', check_relaxed_molecules),
                      ('15. the relaxed path: constrained saddles (H2CO, N2)',
                       check_relaxed_saddles),
                      ('16. the relaxed path: the 14 levels, closure, true minima',
                       check_relaxed_levels),
                      ('17. the relaxed path: soft_mode_qp and refusals',
                       check_relaxed_qp_and_refusals),
                      ('18. the relaxed path: flags', check_relaxed_flags),
                      ('19. the GCM: kernels', check_gcm_kernels),
                      ('20-21. the GCM: one pair, H2 against FCI', check_gcm_h2),
                      ('22-23. the GCM: grid and harmonic limit', check_gcm_grid_and_harmonic),
                      ('24. the GCM: molecules', check_gcm_molecules),
                      ('25-26. the GCM: soft_mode_qp, refusals and flags',
                       check_gcm_qp_refusals_flags)):
        print(f'\n-- {title}')
        ok &= fn()
    relaxed_lowest.cache_clear()   # its mean fields hold pyscf's temporary chkfiles
    gcm_run.cache_clear()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_soft_mode_triplet():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
