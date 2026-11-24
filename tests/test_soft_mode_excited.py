"""The sign-aware (Krein) split of `LinearResponse.soft_mode` for excited (non-aufbau) closed-shell
references, reference='excited', and the singlet block, `soft_mode_singlets`: what is held.

1. The singlet matrices built from MO blocks (`singlet_tdhf_matrices`) are the house's
   (build_casida_matrices with lBSE=True) to CROSS_PATH_RTOL: H2/cc-pVDZ at 1.0 A and H2O; on
   H2/6-31G sigma_u^2 (orbitals reordered by `excited_reference`) too, and its A - B is the
   triplet block's to 1e-14.
2. Where A - B is positive definite the sign-aware split is the default's: `krein_split` is
   `soft_split` to the bit (H2O, both blocks), and soft_mode_triplets(..., reference='excited')
   gives the default's omega, X and Y to 1e-12 (H2/cc-pVDZ at 1.0 and 1.5 A, models '1d', '3d' and
   'gcm'; 0 to 3e-16 measured). The singlet block's soft mode put back at its harmonic level is
   the house singlet Casida root 0 to 1e-10 (H2 at 1.0 A, H2O; 1.7e-14 measured).
3. The split of H2/6-31G sigma_u^2 (a symmetry-constrained RHF; A - B has a negative eigenvalue at
   every R): mu / M = Omega^2 to 1e-12 of the largest |Omega^2| and d^T e = 1 to 1e-12; the regular
   modes solve the RPA equations to 1e-10 with X'X - Y'Y = 1 and omega = sign(M) |Omega|; the soft
   mode put back at its harmonic level is the positive-norm eigenpair of the dense RPA matrix
   [[A, B], [-B, -A]] to 1e-10, at Omega < 0 for the singlet's negative transition at 1.2 A (M < 0,
   mu < 0). The classification along R: the singlet soft mode (sigma_u -> sigma_g, sigma_g') has
   M < 0 at 1.2 to 3.0 A, mu < 0 at 1.2 and 1.3 A and mu > 0 from 1.34 A on (the excited-state
   instability: Omega^2 changes sign between 1.33 and 1.34 A); the triplet's (sigma_u ->
   sigma_g') has M > 0 at 0.2 to 0.34 A, mu < 0 at 0.2 and 0.3 A (the short-bond instability) and
   mu > 0 at 0.34 A.
4. The singlet path (`path_samples` with channel='singlet', both spins turned by exp(K)):
   V''(0) = mu to 1e-5 relative (Richardson at h = 4e-3 and 2h; sigma_u^2 at 1.5 A, M < 0, and the
   aufbau H2O), D(Q) = sin Q for one occupied orbital to 1e-12, and V even to 1e-12 of max |V|.
   Its closed-form GCM kernels (`_GCMKernels` with channel='singlet', the beta pairs turned as the
   alpha ones) are the generic non-orthogonal ones at eight random (Q, Q') to 1e-10 Ha for H and
   1e-12 for N, D_alpha and S^2 (= 0): H2O/STO-3G (five occupied, two pairs), H2O/cc-pVDZ and
   H2/6-31G sigma_u^2; their diagonal is `path_samples`' V and D to 1e-10.
5. H2 sigma_u^2's symmetry breaking toward H+H- against FCI (6-31G at 1.2, 1.3, 1.34, 1.4, 1.5, 2.0
   and 3.0 A; FCI: from the singlet of largest sigma_u^2 weight to the singlet it couples to most
   through D, -6.04 to -0.07 eV). TDHF is imaginary from 1.34 A on and TDA has the wrong sign from
   2.0 A on, while the sign-aware levels of soft_mode_singlets(..., reference='excited') are real
   and negative at every R and rise toward 0 with R: the 1D model within 0.7 eV of FCI (-0.63 eV
   at 1.5 A the worst, every miss but 3.0 A's too negative), the GCM within 0.4 eV (+0.33 eV at
   1.2 A) and equal, gap and coupling, to the CI in the two orbitals {sigma_u, v_d}
   (`two_orbital_ci`) taken from its top even state, to 1e-10.
6. The short-bond triplet instability of sigma_u^2 against FCI (6-31G at 0.2 and 0.3 A, and
   0.34 A just past it): TDHF is imaginary at 0.2 and 0.3 A and TDA negative at 0.2 A; the mass is
   positive, so the reference is the bottom level, uninverted (in the GCM by the rule of a block
   whose A - B is indefinite, level 'overlap', which keeps the even ground state here: it holds
   0.87 to 0.96 of the determinant), and soft_mode_triplets(...,
   reference='excited') gives positive transitions: the GCM within 0.1 eV of FCI (+0.02 to
   +0.05 eV), the 1D model within 0.7 eV (+0.54 to +0.60 eV); the GCM is the two-orbital CI from
   its bottom even state to 1e-10 (1e-11 measured, at 0.2 A).
7. A maximum-overlap reference (a plain RHF whose get_occ keeps the orbital of largest overlap):
   at 0.3 A its sigma_u lies above sigma_g (mo_occ = [0, 2, 0, 0]), which the default refuses,
   and reference='excited' gives the symmetry-constrained levels to 1e-5 eV (3e-6 measured, the
   two RHFs' own agreement) with orbitals [1, 0, 2, 3]; at 1.5 A the sigma_u^2 state is aufbau in
   its own Fock spectrum, and the default refuses it at the split (CasidaInstability).
8. The singlet block of an aufbau reference (H2/cc-pVDZ at 0.75, 1.0, 1.5 and 2.0 A, against the
   FCI singlet it couples to most through D, B 1Sigma_u+): the GCM within 0.3 eV (+0.23 eV at
   0.75 A the worst); the 1D model, on this stiff mode, is 1.6 to 4.5 eV low (printed, not
   held).
9. Refusals before any work, each by its class and a fragment of its own message: an unknown
   reference; reference='excited' with path='relaxed' (NotImplementedError); soft_mode_singlets
   with model '3d' (ValueError) and with path 'relaxed' (NotImplementedError) on an aufbau
   H2/cc-pVDZ, which the singlet driver would otherwise split (on the straight path for the
   relaxed one); an open-shell occupation; a non-stationary one (H2's ground-state
   orbitals with sigma_u doubly occupied, 6.5e-2 Ha in the Fock matrix's occupied-virtual block);
   gcm_levels with an unknown level. After the split: a complex Omega^2 (the triplet at 1.0 A,
   inside its collision window) with CasidaInstability, and a totally symmetric singlet soft mode
   (every mode made soft) with NotImplementedError.
10. Flags: the GCM singlet at 3.0 A, whose two even states overlap the reference determinant by
    0.498 (the top one, kept) and 0.502, raises 'soft_mode_reference_overlap'; the singlet at
    0.76 A, 0.029 from a mode of the other Krein signature, raises 'soft_mode_krein_neighbour';
    both SoftModeExcitedShort, and both reach numerical_flags; every other H2 row of claims 5 and
    6, in the 1D model and the GCM, raises no flag (relative gaps 0.97 and more).
11. A degenerate soft pair with A - B indefinite: N2/6-31G at 1.1 A on the doubly excited closed
    shell 3sigma_g^2 -> 3sigma_u^2 ({A1g: 4, A1u: 6, E1u: 4}), singlet block, whose lowest
    Omega^2 (-2.69e-4 Ha^2, a pi pair) LAPACK can return as a conjugate pair with imaginary
    parts of rounding: every mode of `krein_split` solves (A - B)(A + B) d = Omega^2 d to
    SPLIT_RTOL of the largest |Omega^2| (2.2e-3 to 1.7e-2 when each vector of the pair was made
    real alone), the regular columns solve the RPA equations to RPA_TOL with X'X - Y'Y = 1, and
    the two soft modes have one mass and, through soft_mode_singlets, one level (to
    DEGENERATE_TOL; -12.02 Ha^-1 and -1.35 eV, against -1.20 Ha^-1 and -16.47 eV for the noise
    vector before).
12. The reference of a negative mass in a multi-pair GCM (`gcm_levels`, level='top'): LiH/6-31G at
    1.6 A on the excited closed shell 1s^2 1pi_x^2, singlet block, a soft mode of M < 0 whose
    path keeps three even states (overlaps with the reference determinant 0.27, 0.73 and 5e-5,
    ascending in energy): the kept state is the one of largest overlap above the even ground
    state (the second), not the top root (5e-5, which gave +0.87 eV, a transition of the wrong
    sign), its first transition is a de-excitation, the same at every cut of the scan to
    CUT_SCAN_TOL, and no 'soft_mode_reference_overlap' is raised.
13. The reference of a positive mass in a multi-pair GCM on a block whose A - B is indefinite
    (`gcm_levels`, level='overlap'; H4/6-31G excited closed shells): on square H4 at 1.0 A on
    b2u^2 b3u^2 (triplet), whose even ground state holds none of the reference determinant (a
    near-null root of the cut) and the next one 0.996, the kept state is that next one, its first
    transition positive and the same over the cut scan to OVERLAP_SCAN_TOL (+3.36 eV; the bottom
    gave -6.24 eV, and 3.35 to 11.76 eV over the cuts), with no 'soft_mode_reference_overlap'; on
    linear H4 at 1.5 A on 1sigma_g^2 2sigma_g^2 (singlet), whose even ground state holds the
    determinant (0.95) with an odd state below it, the first transition is the nearest odd state
    above (+5.55 eV, where the bottom's ascending order gave -0.04 eV), the one below after it.
    On rectangular H4 at 1.5 x 0.72 A on b1u^2 b3g^2 (singlet), whose two even states above the
    ground state share the determinant (0.4361 / 0.4356, and the kept level jumps from -4.98 to
    -0.41 eV by 0.73 A), 'soft_mode_reference_tie' is raised (SoftModeExcitedShort).

The FCI is the house's PyscfFCISolver on H2's 6-31G and cc-pVDZ orbitals; the bars hold measured
misses. Run: python tests/test_soft_mode_excited.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.linalg
from pyscf import ao2mo, gto, scf
from pyscf.fci import spin_op

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.common.eri_blocks import mo_eri
from src.SingleReference.GW.soft_mode_qp import soft_mode_qp
from src.SingleReference.LinearResponse import soft_mode as SM
from src.SingleReference.LinearResponse.casida import CasidaInstability, CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.Solvers.active_space_solver import PyscfFCISolver
from tests.pins import CROSS_PATH_RTOL
from tests.soft_mode_references import GEOMETRIES, rhf

#: A - B of the singlet and the triplet block of H2/6-31G sigma_u^2: 0 measured.
CHANNEL_AMB_TOL = 1e-14
#: The excited route against the default on an aufbau reference: 0 (1d, 3d) and 3e-16 (gcm)
#: measured; the singlet harmonic level against the house Casida root: 1.7e-14 (H2O).
SAME_ROUTE_TOL = 1e-12
HARMONIC_ROOT_TOL = 1e-10
#: The split's identities (relative to the largest |Omega^2|, and d^T e = 1): 1.1e-16 measured.
SPLIT_RTOL = 1e-12
#: The regular modes' RPA residual and norm, the harmonic soft pole against the dense RPA
#: eigenpair: 2.2e-16 measured.
RPA_TOL = 1e-10
#: Relative bar of V''(0) = mu on the singlet path (Richardson at h = 4e-3 and 2h): 1.4e-9
#: (sigma_u^2) and 2.5e-7 (H2O) measured.
CURVATURE_RTOL = 1e-5
#: D(Q) = sin Q, and V even relative to max |V|: 3e-16 and 2.4e-15 measured.
SIN_TOL = 1e-12
PARITY_TOL = 1e-12
#: Ha. The singlet GCM kernels against `loewdin_kernels` (1.4e-13 measured), the overlap, D and
#: S^2 kernels (9e-16), and their diagonal against `path_samples` (4e-13).
GCM_KERNEL_TOL = 1e-10
GCM_OVERLAP_TOL = 1e-12
#: eV. H2 sigma_u^2's singlet level against FCI: the 1D model -0.63 eV at worst (1.5 A), the GCM
#: +0.33 eV (1.2 A).
SINGLET_1D_TOL = 0.7
SINGLET_GCM_TOL = 0.4
#: eV. The GCM's gap and coupling against the two-orbital CI: 3e-15 to 2e-14 measured for the
#: singlet, 1e-11 for the triplet at 0.2 A.
TWO_ORBITAL_TOL = 1e-10
#: eV. The short-bond triplet against FCI: the GCM +0.05 eV at worst, the 1D model +0.60 eV.
TRIPLET_GCM_TOL = 0.1
TRIPLET_1D_TOL = 0.7
#: eV. The maximum-overlap reference's levels against the symmetry-constrained one's: 2.9e-6 eV
#: (0.3 A) and 2.5e-7 eV (1.5 A) measured (their RHFs agree to the SCF's convergence).
MOM_TOL = 1e-5
#: eV. The aufbau singlet's GCM level against FCI's B state: +0.23 eV at worst (0.75 A).
AUFBAU_SINGLET_GCM_TOL = 0.3
#: The two soft modes of a degenerate pair (N2's excited closed shell): their masses (relative)
#: and their levels (eV), 6.6e-14 and 2e-12 measured.
DEGENERATE_TOL = 1e-8
#: eV. A negative mass's GCM level over the overlap cuts of SOFT_MODE_GCM_CUT_SCAN (LiH 1pi_x^2):
#: 0 measured.
CUT_SCAN_TOL = 1e-3
#: eV. A positive mass's GCM level by overlap over the cuts of SOFT_MODE_GCM_CUT_SCAN (square H4
#: b2u^2 b3u^2 triplet): 7e-3 measured (3.349 to 3.356 eV).
OVERLAP_SCAN_TOL = 0.02

SINGLET_DISTANCES = (1.2, 1.3, 1.34, 1.4, 1.5, 2.0, 3.0)
TRIPLET_DISTANCES = (0.2, 0.3, 0.34)
#: The records of claims 5 and 6, {(channel, R, model): record}, which claim 10 reads for its flags.
ROW_RECORDS = {}


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


def sigma_u2(R, basis='6-31g'):
    """H2's sigma_u^2 RHF by a symmetry-constrained SCF (both electrons in A1u)."""
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {R}', basis=basis, unit='Angstrom', verbose=0,
                symmetry=True)
    mf = scf.RHF(mol)
    mf.irrep_nelec = {'A1u': 2}
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def sigma_u2_mom(R, basis='6-31g'):
    """H2's sigma_u^2 RHF by maximum overlap on a plain RHF: get_occ fills the orbital that
    overlaps the last occupied one most, started from the ground state's sigma_u."""
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {R}', basis=basis, unit='Angstrom', verbose=0)
    ground = scf.RHF(mol).run(conv_tol=1e-12)
    S = ground.get_ovlp()
    last = [ground.mo_coeff[:, [1]]]
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10

    def get_occ(mo_energy=None, mo_coeff=None):
        mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
        occ = np.zeros(mo_coeff.shape[1])
        occ[int(np.argmax(np.abs(last[0].T @ S @ mo_coeff)))] = 2.0
        last[0] = mo_coeff[:, occ > 0]
        return occ
    mf.get_occ = get_occ
    mf.kernel(dm0=2.0 * last[0] @ last[0].T)
    return mf


def lowest_mode(ref, channel):
    """(A, B, the split, the lowest soft mode's record) of the channel's block of a reference whose
    occupied orbitals come first."""
    A, B = (SM.singlet_tdhf_matrices if channel == 'singlet' else SM.triplet_tdhf_matrices)(ref)
    split = SM.krein_split(A, B)
    return A, B, split, split['modes'][split['soft'][0]]


def mo_integrals(mf, C):
    n = C.shape[1]
    return C.T @ mf.get_hcore() @ C, ao2mo.restore(1, ao2mo.kernel(mf.mol, C), n)


def fci_transition(ref, direction, channel):
    """(E_target - E_reference (Ha), the two state indices) by the house FCI in the (1, 1) sector
    of H2 on ref's orbitals (sigma_u first): the reference is the singlet of largest sigma_u^2
    weight, the target the state of the channel's spin it couples to most through
    D = sum_a d_a (a+_a u + u+ a), alpha plus (singlet) or minus (triplet) beta."""
    C = ref.mo_coeff
    n = C.shape[1]
    h1, eri = mo_integrals(ref, C)
    e, vecs = PyscfFCISolver().roots(h1, eri, n, (1, 1), nroots=n * n, spin=None)
    s2 = np.array([spin_op.spin_square(v, n, (1, 1))[0] for v in vecs])
    singlet = s2 < 1.0
    iref = int(np.argmax(np.where(singlet, [v[0, 0] ** 2 for v in vecs], -1.0)))
    T = np.zeros((n, n))
    T[1:, 0] = T[0, 1:] = direction
    dv = T @ vecs[iref] + (1.0 if channel == 'singlet' else -1.0) * vecs[iref] @ T.T
    coupling = np.array([abs(np.sum(v * dv)) for v in vecs])
    want = singlet if channel == 'singlet' else np.abs(s2 - 2.0) < 1e-6
    target = int(np.argmax(np.where(want, coupling, -1.0)))
    return e[target] - e[iref], (iref, target)


def two_orbital_ci(ref, direction, channel, level):
    """(transition (Ha), coupling) of the CI in the two orbitals {u, v_d} (u the occupied orbital,
    v_d = sum_a d_a phi_a): the even states (u^2, v^2) from [[2 h_uu + (uu|uu), (uv|uv)],
    [(uv|uv), 2 h_vv + (vv|vv)]] (the bottom or the top one, vector c), the odd one the open-shell
    singlet h_uu + h_vv + (uu|vv) + (uv|vu) or triplet (- (uv|vu)), coupled through D_alpha by
    |c_u +- c_v| / sqrt2."""
    C = ref.mo_coeff
    orb = np.stack([C[:, 0], C[:, 1:] @ direction], axis=1)
    h, g = mo_integrals(ref, orb)
    w, c = np.linalg.eigh(np.array([[2 * h[0, 0] + g[0, 0, 0, 0], g[0, 1, 0, 1]],
                                    [g[0, 1, 0, 1], 2 * h[1, 1] + g[1, 1, 1, 1]]]))
    k = 0 if level == 'bottom' else 1
    sign = 1.0 if channel == 'singlet' else -1.0
    odd = h[0, 0] + h[1, 1] + g[0, 0, 1, 1] + sign * g[0, 1, 1, 0]
    return odd - w[k], abs(c[0, k] + sign * c[1, k]) / np.sqrt(2.0)


def loewdin_kernels(mf, direction, pairs, channel):
    """(N, H - E_ref N, <D_alpha>, <S^2>) between the channel's straight-path determinants at the
    (Q, Q') pairs by the generic non-orthogonal rules in mf's MO basis (rotations by expm as in
    `path_samples`, the occupied orbitals first)."""
    nocc, C = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0)), mf.mo_coeff
    n = C.shape[1]
    h, eri = mo_integrals(mf, C)
    dm = np.reshape(direction, (nocc, n - nocc))
    Dmo = np.zeros((n, n))
    Dmo[:nocc, nocc:], Dmo[nocc:, :nocc] = dm, dm.T

    def occupied(Q):
        K = np.zeros((n, n))
        K[nocc:, :nocc], K[:nocc, nocc:] = 0.5 * Q * dm.T, -0.5 * Q * dm
        U = scipy.linalg.expm(K)
        return U[:, :nocc], (U.T if channel == 'triplet' else U)[:, :nocc]
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


def house_singlet(mf):
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
    lr = LinearResponseSolver(mf.mo_energy, eri_chemist=mo_eri(mf, mf.mol), spin_mode='restricted')
    return lr.build_casida_matrices(nocc, lBSE=True)


def check_matrices():
    ok = True
    excited = SM.excited_reference(sigma_u2(1.5))[0]
    for label, mf in (('H2/cc-pVDZ 1.0 A', h2(1.0)), ('H2O/cc-pVDZ', rhf(GEOMETRIES['H2O'])),
                      ('H2/6-31G sigma_u^2 1.5 A', excited)):
        A, B = SM.singlet_tdhf_matrices(mf)
        Ah, Bh = house_singlet(mf)
        miss = max(relative(A, Ah), relative(B, Bh))
        ok &= check(miss < CROSS_PATH_RTOL,
                    f'{label}: singlet A, B from MO blocks = the house builder', f'{miss:.1e}')
    A, B = SM.singlet_tdhf_matrices(excited)
    At, Bt = SM.triplet_tdhf_matrices(excited)
    miss = np.abs((A - B) - (At - Bt)).max()
    ok &= check(miss < CHANNEL_AMB_TOL, 'H2/6-31G sigma_u^2: A - B is the same in both blocks',
                f'{miss:.1e}')
    return ok


def check_aufbau_limit():
    ok = True
    mf = rhf(GEOMETRIES['H2O'])
    for name, build in (('triplet', SM.triplet_tdhf_matrices), ('singlet', SM.singlet_tdhf_matrices)):
        A, B = build(mf)
        a, b = SM.soft_split(A, B), SM.krein_split(A, B)
        same = (np.array_equal(a['omega2'], b['omega2']) and a['soft'] == b['soft']
                and all(np.array_equal(x, y) for x, y in zip(a['regular'], b['regular']))
                and all(np.array_equal(a['modes'][s][k], b['modes'][s][k])
                        for s in a['soft'] for k in ('direction', 'conjugate', 'mass')))
        ok &= check(same and np.all(b['signature'] == 1),
                    f'H2O {name} block: krein_split = soft_split to the bit, every signature +1')
    for R in (1.0, 1.5):
        mf = h2(R)
        for model in ('1d', '3d', 'gcm'):
            a = SM.soft_mode_triplets(mf, model)
            b = SM.soft_mode_triplets(mf, model, reference='excited')
            miss = max(np.abs(a[k] - b[k]).max() for k in ('omega', 'X', 'Y'))
            ok &= check(miss < SAME_ROUTE_TOL and np.all(b['signature'] == 1),
                        f"H2/cc-pVDZ {R} A {model}: reference='excited' = the default", f'{miss:.1e}')
    for label, mf in (('H2 1.0 A', h2(1.0)), ('H2O', rhf(GEOMETRIES['H2O']))):
        A, B = SM.singlet_tdhf_matrices(mf)
        mode = SM.soft_split(A, B)['modes'][0]
        om = np.sqrt(mode['omega2'])
        dE, X, Y = SM.collective_modes(np.array([om]), np.array([1.0 / np.sqrt(2 * mode['mass'] * om)]),
                                       mode['mass'], mode['direction'], mode['conjugate'])
        w, Xh, Yh = (np.asarray(v) for v in CasidaSolver(*house_singlet(mf)).solve())
        sign = np.sign((X[:, 0] + Y[:, 0]) @ (Xh[:, 0] + Yh[:, 0]))
        miss = max(abs(dE[0] - w[0]), np.abs(sign * X[:, 0] - Xh[:, 0]).max(),
                   np.abs(sign * Y[:, 0] - Yh[:, 0]).max())
        ok &= check(miss < HARMONIC_ROOT_TOL,
                    f'{label}: the singlet soft mode at its harmonic level = the house Casida root 0',
                    f'{miss:.1e}')
    return ok


def check_split():
    ok = True
    ref = SM.excited_reference(sigma_u2(1.2))[0]
    A, B, split, mode = lowest_mode(ref, 'singlet')
    scale = np.max(np.abs(split['omega2']))
    miss = max(abs(mode['stiffness'] / mode['mass'] - mode['omega2']) / scale,
               abs(mode['direction'] @ mode['conjugate'] - 1.0))
    ok &= check(miss < SPLIT_RTOL and np.linalg.eigvalsh(A - B)[0] < 0,
                'sigma_u^2 1.2 A singlet (A - B indefinite): mu / M = Omega^2 and d^T e = 1',
                f'{miss:.1e}')
    om, X, Y = split['regular']
    res = max(np.abs(A @ X + B @ Y - X * om).max(), np.abs(B @ X + A @ Y + Y * om).max(),
              np.abs(np.sum(X * X - Y * Y, axis=0) - 1.0).max())
    regular_mass = np.einsum('pn,pq,qn->n', X + Y, np.linalg.inv(A - B), X + Y)
    ok &= check(res < RPA_TOL and np.all(np.sign(om) == np.sign(regular_mass)),
                "the regular modes solve the RPA equations, X'X - Y'Y = 1, omega = sign(M) |Omega|",
                f'{res:.1e}, omega {np.round(om * HARTREE_TO_EV, 2).tolist()} eV')
    M = mode['mass']
    omega = np.sign(M) * np.sqrt(mode['omega2'])
    dE, Xs, Ys = SM.collective_modes(np.array([omega]), np.array([1.0 / np.sqrt(2 * M * omega)]), M,
                                     mode['direction'], mode['conjugate'])
    n = A.shape[0]
    w, V = np.linalg.eig(np.block([[A, B], [-B, -A]]))
    k = int(np.argmin(np.abs(w - omega)))
    x, y = V[:n, k].real, V[n:, k].real
    norm = x @ x - y @ y
    scale = np.sign(x @ Xs[:, 0]) / np.sqrt(abs(norm))
    x, y = x * scale, y * scale
    miss = max(abs(w[k].real - omega), np.abs(x - Xs[:, 0]).max(), np.abs(y - Ys[:, 0]).max())
    ok &= check(miss < RPA_TOL and norm > 0 and omega < 0 and M < 0 and mode['stiffness'] < 0,
                'its soft mode (M < 0, mu < 0) at its harmonic level = the positive-norm RPA '
                'eigenpair at Omega < 0', f'{miss:.1e}; Omega {omega * HARTREE_TO_EV:+.3f} eV')
    signs = {}
    for R in SINGLET_DISTANCES + (1.33,):
        mode = lowest_mode(SM.excited_reference(sigma_u2(R))[0], 'singlet')[3]
        signs[R] = (np.sign(mode['mass']), np.sign(mode['stiffness']), mode['omega2'])
    want = all(signs[R][0] < 0 for R in signs) and all(
        (signs[R][1] < 0) == (R < 1.34) and (signs[R][2] < 0) == (R >= 1.34) for R in signs)
    ok &= check(want, 'singlet soft mode: M < 0 from 1.2 to 3.0 A; mu < 0 before 1.34 A, mu > 0 '
                'and Omega^2 < 0 from 1.34 A on (the instability between 1.33 and 1.34 A)',
                f"Omega^2 {signs[1.33][2]:+.1e} / {signs[1.34][2]:+.1e} Ha^2 at 1.33 / 1.34 A")
    tsigns = {}
    for R in TRIPLET_DISTANCES:
        mode = lowest_mode(SM.excited_reference(sigma_u2(R))[0], 'triplet')[3]
        tsigns[R] = (mode['mass'], mode['stiffness'])
    want = (all(tsigns[R][0] > 0 for R in tsigns) and tsigns[0.2][1] < 0 and tsigns[0.3][1] < 0
            and tsigns[0.34][1] > 0)
    ok &= check(want, 'triplet soft mode: M > 0 from 0.2 to 0.34 A, mu < 0 at 0.2 and 0.3 A, '
                '> 0 at 0.34 A', f'M {[round(v[0], 2) for v in tsigns.values()]}')
    return ok


def check_singlet_path():
    ok = True
    h = 4e-3
    aufbau = rhf(GEOMETRIES['H2O'])
    for label, ref, split in (
            ('sigma_u^2 1.5 A (M < 0)', SM.excited_reference(sigma_u2(1.5))[0], SM.krein_split),
            ('H2O (aufbau)', aufbau, SM.soft_split)):
        nocc = int(np.count_nonzero(np.asarray(ref.mo_occ) > 0))
        mode = split(*SM.singlet_tdhf_matrices(ref))['modes'][0]
        q, V, D = SM.path_samples(ref, mode['direction'], h * (SM.SOFT_MODE_PATH_POINTS - 1),
                                  'singlet')
        curv = (4 * (2 * (V[1] - V[0]) / h ** 2) - (V[2] - V[0]) / (2 * h ** 2)) / 3
        miss = abs(curv - mode['stiffness']) / abs(mode['stiffness'])
        ok &= check(miss < CURVATURE_RTOL, f"{label}: V''(0) = mu on the singlet path", f'{miss:.1e}')
        if nocc == 1:
            L = SM.path_period(mode['direction'], nocc)[0]
            q, V, D = SM.path_samples(ref, mode['direction'], L, 'singlet')
            Vm = SM.path_samples(ref, -mode['direction'], L, 'singlet')[1]
            par = np.abs(Vm - V).max() / np.abs(V).max()
            ok &= check(np.abs(D - np.sin(q)).max() < SIN_TOL and par < PARITY_TOL,
                        f'{label}: D(Q) = sin Q and V even', f'{np.abs(D - np.sin(q)).max():.1e}, '
                        f'{par:.1e}')
    rng = np.random.default_rng(7)
    cases = (('H2O/STO-3G', rhf(GEOMETRIES['H2O'], 'sto-3g'), SM.soft_split),
             ('H2O/cc-pVDZ', aufbau, SM.soft_split),
             ('H2/6-31G sigma_u^2 1.5 A', SM.excited_reference(sigma_u2(1.5))[0], SM.krein_split))
    for label, ref, split in cases:
        nocc = int(np.count_nonzero(np.asarray(ref.mo_occ) > 0))
        mode = split(*SM.singlet_tdhf_matrices(ref))['modes'][0]
        L, sigma = SM.path_period(mode['direction'], nocc)
        pairs = rng.uniform(-L, L, (8, 2))
        want = loewdin_kernels(ref, mode['direction'], pairs, 'singlet')
        theta = 0.5 * pairs[:, :, None] * sigma
        kernels = SM._GCMKernels(ref, mode['direction'], nocc, 'singlet')
        got = np.array(kernels.elements(theta[:, 0], theta[:, 1]))
        miss = np.abs(got - want).max(axis=1)
        q, V, D = SM.path_samples(ref, mode['direction'], L, 'singlet')
        th = 0.5 * np.outer(q, sigma)
        _, H, Dk, _ = kernels.elements(th, th)
        diag = max(np.abs(H - V).max(), np.abs(Dk - D).max())
        ok &= check(miss[1] < GCM_KERNEL_TOL and max(miss[[0, 2, 3]]) < GCM_OVERLAP_TOL
                    and diag < GCM_KERNEL_TOL,
                    f'{label} ({nocc} occupied, {sigma.size} pairs): the singlet GCM kernels = the '
                    'generic non-orthogonal ones, their diagonal = path_samples',
                    f'N {miss[0]:.1e}, H {miss[1]:.1e} Ha, D {miss[2]:.1e}, S^2 {miss[3]:.1e}; '
                    f'{diag:.1e}')
    return ok


def check_singlet_fci():
    ok = True
    rows = []
    for R in SINGLET_DISTANCES:
        mf = sigma_u2(R)
        ref = SM.excited_reference(mf)[0]
        A, B, split, mode = lowest_mode(ref, 'singlet')
        fci, _ = fci_transition(ref, mode['direction'], 'singlet')
        recs = {m: SM.soft_mode_singlets(mf, m, reference='excited') for m in ('1d', 'gcm')}
        ROW_RECORDS.update({('singlet', R, m): rec for m, rec in recs.items()})
        one, gcm = (recs[m]['soft'][0] for m in ('1d', 'gcm'))
        wa, ua = np.linalg.eigh(A)
        tda = wa[np.argmax(np.abs(ua.T @ mode['direction']))]
        gap, c = two_orbital_ci(ref, mode['direction'], 'singlet', 'top')
        ci_miss = max(abs(gcm['excitation'] - gap) * HARTREE_TO_EV, abs(gcm['coupling'] - c))
        miss1, missg = ((x['excitation'] - fci) * HARTREE_TO_EV for x in (one, gcm))
        rows.append((R, one['excitation'], gcm['excitation']))
        tdhf = (f"{np.sign(mode['mass']) * np.sqrt(mode['omega2']) * HARTREE_TO_EV:+.3f}"
                if mode['omega2'] > 0 else f"{np.sqrt(-mode['omega2']) * HARTREE_TO_EV:.3f}i")
        ok &= check(one['excitation'] < 0 and gcm['excitation'] < 0 and abs(miss1) < SINGLET_1D_TOL
                    and abs(missg) < SINGLET_GCM_TOL and ci_miss < TWO_ORBITAL_TOL
                    and one['level'] == gcm['level'] == 'top' and (mode['omega2'] < 0) == (R >= 1.34)
                    and (tda > 0) == (R >= 2.0),
                    f'sigma_u^2 {R} A singlet: real negative levels, 1D within {SINGLET_1D_TOL} and '
                    f'GCM within {SINGLET_GCM_TOL} eV of FCI, GCM = the two-orbital CI (top)',
                    f'1D {one["excitation"] * HARTREE_TO_EV:+.3f}, GCM '
                    f'{gcm["excitation"] * HARTREE_TO_EV:+.3f}, FCI {fci * HARTREE_TO_EV:+.3f}, '
                    f'TDHF {tdhf}, TDA {tda * HARTREE_TO_EV:+.3f} eV; CI {ci_miss:.0e}')
    rising = all(b[1] > a[1] and b[2] > a[2] for a, b in zip(rows, rows[1:]))
    ok &= check(rising, 'sigma_u^2 singlet: both levels rise toward 0 with R')
    return ok


def check_triplet_fci():
    ok = True
    for R in TRIPLET_DISTANCES:
        mf = sigma_u2(R)
        ref = SM.excited_reference(mf)[0]
        A, B, split, mode = lowest_mode(ref, 'triplet')
        fci, _ = fci_transition(ref, mode['direction'], 'triplet')
        recs = {m: SM.soft_mode_triplets(mf, m, reference='excited') for m in ('1d', 'gcm')}
        ROW_RECORDS.update({('triplet', R, m): rec for m, rec in recs.items()})
        one, gcm = (recs[m]['soft'][0] for m in ('1d', 'gcm'))
        wa, ua = np.linalg.eigh(A)
        tda = wa[np.argmax(np.abs(ua.T @ mode['direction']))]
        gap, c = two_orbital_ci(ref, mode['direction'], 'triplet', 'bottom')
        ci_miss = max(abs(gcm['excitation'] - gap) * HARTREE_TO_EV, abs(gcm['coupling'] - c))
        miss1, missg = ((x['excitation'] - fci) * HARTREE_TO_EV for x in (one, gcm))
        ok &= check(one['excitation'] > 0 and gcm['excitation'] > 0
                    and abs(miss1) < TRIPLET_1D_TOL and abs(missg) < TRIPLET_GCM_TOL
                    and ci_miss < TWO_ORBITAL_TOL and one['level'] == 'bottom'
                    and gcm['level'] == 'overlap' and gcm['reference'] == 0
                    and (mode['omega2'] < 0) == (R < 0.34) and (tda < 0) == (R < 0.3),
                    f'sigma_u^2 {R} A triplet: M > 0, positive levels, GCM within '
                    f'{TRIPLET_GCM_TOL} and 1D within {TRIPLET_1D_TOL} eV of FCI',
                    f'1D {one["excitation"] * HARTREE_TO_EV:+.3f}, GCM '
                    f'{gcm["excitation"] * HARTREE_TO_EV:+.3f}, FCI {fci * HARTREE_TO_EV:+.3f}, '
                    f'Omega^2 {mode["omega2"]:+.1e} Ha^2, TDA {tda * HARTREE_TO_EV:+.3f} eV; '
                    f'CI {ci_miss:.0e}')
    return ok


def check_mom():
    ok = True
    mom = sigma_u2_mom(0.3)
    try:
        SM.soft_mode_triplets(mom)
        refused = None
    except Exception as exc:   # noqa: BLE001 -- the class is what is checked
        refused = exc
    sym = sigma_u2(0.3)
    miss = []
    for fn in (SM.soft_mode_singlets, SM.soft_mode_triplets):
        a = fn(sym, '1d', reference='excited')
        b = fn(mom, '1d', reference='excited')
        miss.append(abs(a['soft'][0]['excitation'] - b['soft'][0]['excitation']) * HARTREE_TO_EV)
        orbitals = b['orbitals']
    ok &= check(isinstance(refused, ValueError) and max(miss) < MOM_TOL
                and list(mom.mo_occ) == [0, 2, 0, 0] and list(orbitals) == [1, 0, 2, 3],
                'MOM sigma_u^2 at 0.3 A (mo_occ [0, 2, 0, 0]): refused by the default, the '
                "symmetry-constrained levels with reference='excited'",
                f'{type(refused).__name__}; {max(miss):.1e} eV; orbitals {orbitals.tolist()}')
    mom = sigma_u2_mom(1.5)
    try:
        SM.soft_mode_triplets(mom)
        refused = None
    except Exception as exc:   # noqa: BLE001 -- the class is what is checked
        refused = exc
    a = SM.soft_mode_singlets(sigma_u2(1.5), '1d', reference='excited')['soft'][0]['excitation']
    b = SM.soft_mode_singlets(mom, '1d', reference='excited')['soft'][0]['excitation']
    ok &= check(isinstance(refused, CasidaInstability) and list(mom.mo_occ) == [2, 0, 0, 0]
                and abs(a - b) * HARTREE_TO_EV < MOM_TOL,
                'MOM sigma_u^2 at 1.5 A (aufbau in its own Fock spectrum): refused by the default '
                'at the split, the same levels', f'{type(refused).__name__}; '
                f'{abs(a - b) * HARTREE_TO_EV:.1e} eV')
    return ok


def check_aufbau_singlet():
    ok = True
    for R in (0.75, 1.0, 1.5, 2.0):
        mf = h2(R)
        n = mf.mo_coeff.shape[1]
        h1, eri = mo_integrals(mf, mf.mo_coeff)
        e, vecs = PyscfFCISolver().roots(h1, eri, n, (1, 1), nroots=12, spin=0)
        T = np.zeros((n, n))
        mode = SM.soft_split(*SM.singlet_tdhf_matrices(mf))['modes'][0]
        T[1:, 0] = T[0, 1:] = mode['direction']
        dv = T @ vecs[0] + vecs[0] @ T.T
        target = int(np.argmax([abs(np.sum(v * dv)) for v in vecs]))
        fci = e[target] - e[0]
        one, gcm = (SM.soft_mode_singlets(mf, m)['soft'][0]['excitation'] for m in ('1d', 'gcm'))
        ok &= check(abs(gcm - fci) * HARTREE_TO_EV < AUFBAU_SINGLET_GCM_TOL,
                    f'H2/cc-pVDZ {R} A singlet (aufbau): the GCM within {AUFBAU_SINGLET_GCM_TOL} eV '
                    "of FCI's B state",
                    f'GCM {gcm * HARTREE_TO_EV:.3f}, 1D {one * HARTREE_TO_EV:.3f} (printed), TDHF '
                    f'{np.sqrt(mode["omega2"]) * HARTREE_TO_EV:.3f}, FCI {fci * HARTREE_TO_EV:.3f} eV')
    return ok


def check_refusals():
    ok = True
    mf = sigma_u2(1.5)
    ground = h2(1.5, '6-31g')
    aufbau = h2(1.0)
    swapped = ground.copy()
    swapped.mo_occ = np.array([0.0, 2.0, 0.0, 0.0])
    open_shell = ground.copy()
    open_shell.mo_occ = np.array([1.0, 1.0, 0.0, 0.0])
    cases = [("reference 'ground'", lambda: SM.soft_mode_triplets(mf, reference='ground'),
              ValueError, 'one of'),
             ("reference 'excited' with path 'relaxed'",
              lambda: SM.soft_mode_triplets(swapped, '1d', 'relaxed', 'excited'),
              NotImplementedError, "reference = 'excited' with path = 'relaxed'"),
             ("soft_mode_singlets: model '3d' (an aufbau H2, which it would split otherwise)",
              lambda: SM.soft_mode_singlets(aufbau, '3d'), ValueError,
              'a singlet soft mode has one'),
             ("soft_mode_singlets: path 'relaxed' (an aufbau H2, which it would split on the "
              'straight path otherwise)', lambda: SM.soft_mode_singlets(aufbau, '1d', 'relaxed'),
              NotImplementedError, 'for a singlet soft mode'),
             ('an open-shell occupation',
              lambda: SM.soft_mode_singlets(open_shell, reference='excited'), ValueError,
              'closed-shell occupation'),
             ('a non-stationary occupation (ground-state orbitals, sigma_u^2)',
              lambda: SM.soft_mode_singlets(swapped, reference='excited'), ValueError,
              'not stationary'),
             ('the MOM reference in soft_mode_qp', lambda: soft_mode_qp(sigma_u2_mom(0.3)),
              ValueError, 'aufbau'),
             ("gcm_levels: level 'middle'",
              lambda: SM.gcm_levels(ground, np.array([1.0, 0.0, 0.0]), 'triplet', 'middle'),
              ValueError, "level = 'middle'"),
             ('a complex Omega^2 (the triplet at 1.0 A)',
              lambda: SM.soft_mode_triplets(sigma_u2(1.0), reference='excited'),
              CasidaInstability, 'complex eigenvalue')]
    for label, call, kind, fragment in cases:
        try:
            call()
            raised = None
        except Exception as exc:   # noqa: BLE001 -- the class and message are what is checked
            raised = exc
        ok &= check(isinstance(raised, kind) and fragment in str(raised), f'refused: {label}',
                    f'{type(raised).__name__}: {str(raised)[:70]}')
    saved = SM.SOFT_MODE_OMEGA_CUT
    SM.SOFT_MODE_OMEGA_CUT = 10.0
    try:
        SM.soft_mode_singlets(mf, reference='excited')
        raised = None
    except Exception as exc:   # noqa: BLE001 -- the class is what is checked
        raised = exc
    finally:
        SM.SOFT_MODE_OMEGA_CUT = saved
    ok &= check(isinstance(raised, NotImplementedError) and 'not even' in str(raised),
                'refused: a totally symmetric singlet soft mode (every mode made soft)',
                f'{type(raised).__name__}')
    return ok


def check_flags():
    ok = True
    for label, call, want in (
            ('GCM singlet at 3.0 A',
             lambda: SM.soft_mode_singlets(sigma_u2(3.0), 'gcm', reference='excited'),
             'soft_mode_reference_overlap'),
            ('1D singlet at 0.76 A',
             lambda: SM.soft_mode_singlets(sigma_u2(0.76), '1d', reference='excited'),
             'soft_mode_krein_neighbour')):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            rec = call()
        names = [f['flag'] for f in rec['numerical_flags']]
        warned = any(issubclass(w.category, SM.SoftModeExcitedShort) for w in caught)
        soft = rec['soft'][0]
        detail = (f"overlaps {np.round(soft['reference_overlaps'], 3).tolist()}"
                  if 'reference_overlaps' in soft else f"gap {soft['krein_gap']:.3f}")
        ok &= check(want in names and warned, f"{label}: raises '{want}' (SoftModeExcitedShort)",
                    f'{names}; {detail}')
    clean, gaps = [], []
    for channel, rows in (('singlet', SINGLET_DISTANCES), ('triplet', TRIPLET_DISTANCES)):
        drive = SM.soft_mode_singlets if channel == 'singlet' else SM.soft_mode_triplets
        for R in rows:
            for model in ('1d', 'gcm'):
                rec = ROW_RECORDS.get((channel, R, model))
                if rec is None:
                    rec = drive(sigma_u2(R), model, reference='excited')
                gaps += [s['krein_gap'] for s in rec['soft']]
                if (channel, R, model) != ('singlet', 3.0, 'gcm'):
                    clean += [(channel, R, model, f['flag']) for f in rec['numerical_flags']]
    ok &= check(not clean, 'every other row of claims 5 and 6 (singlet 1.2 to 3.0 A, triplet 0.2 '
                'to 0.34 A, 1D and GCM) raises no flag', f'{clean}; krein_gap {min(gaps):.4f} at '
                'least')
    return ok


def excited_closed_shell(atom, basis, irrep_nelec, symmetry=True):
    """A symmetry-constrained RHF with the given irreducible-representation occupation."""
    mol = gto.M(atom=atom, basis=basis, unit='Angstrom', verbose=0, symmetry=symmetry)
    mf = scf.RHF(mol)
    mf.irrep_nelec = irrep_nelec
    mf.conv_tol = 1e-12
    mf.max_cycle = 300
    mf.kernel()
    return mf


def check_degenerate():
    ok = True
    mf = excited_closed_shell('N 0 0 0; N 0 0 1.1', '6-31g',
                              {'A1g': 4, 'A1u': 6, 'E1ux': 2, 'E1uy': 2})
    ref = SM.excited_reference(mf)[0]
    A, B = SM.singlet_tdhf_matrices(ref)
    split = SM.krein_split(A, B)
    product = (A - B) @ (A + B)
    scale = np.max(np.abs(split['omega2']))
    om, X, Y = split['regular']
    d = (X + Y) / np.linalg.norm(X + Y, axis=0)
    eigen = max(np.max(np.linalg.norm(product @ d - d * om ** 2, axis=0)),
                max(np.linalg.norm(product @ m['direction'] - m['omega2'] * m['direction'])
                    for m in split['modes'].values())) / scale
    rpa = max(np.abs(A @ X + B @ Y - X * om).max(), np.abs(B @ X + A @ Y + Y * om).max(),
              np.abs(np.sum(X * X - Y * Y, axis=0) - 1.0).max())
    lam = scipy.linalg.eigvals(product)
    rounding = int(np.count_nonzero((lam.imag != 0) & (np.abs(lam.imag)
                                                       < SM.SOFT_MODE_KREIN_IMAG_TOL
                                                       * np.max(np.abs(lam)))))
    masses = [m['mass'] for m in split['modes'].values()]
    levels = [s['excitation'] * HARTREE_TO_EV
              for s in SM.soft_mode_singlets(mf, '1d', reference='excited')['soft']]
    same = (len(masses) == 2 and len(levels) == 2
            and abs(masses[0] - masses[1]) < DEGENERATE_TOL * abs(masses[0])
            and abs(levels[0] - levels[1]) < DEGENERATE_TOL)
    ok &= check(mf.converged and eigen < SPLIT_RTOL and rpa < RPA_TOL and same,
                "N2/6-31G 3sigma_g^2 -> 3sigma_u^2 singlet (A - B indefinite, a degenerate soft pi "
                'pair): every mode an eigenvector, the RPA equations held, one mass and one level',
                f'{eigen:.1e}, {rpa:.1e}; M {np.round(masses, 4).tolist()} Ha^-1, levels '
                f'{np.round(levels, 4).tolist()} eV; {rounding} eigenvalues with imaginary parts of '
                'rounding')
    return ok


def check_top_rule():
    ok = True
    mf = excited_closed_shell('Li 0 0 0; H 0 0 1.6', '6-31g', {'A1': 2, 'E1x': 2})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        rec = SM.soft_mode_singlets(mf, 'gcm', reference='excited')
    names = [f['flag'] for f in rec['numerical_flags']]
    soft = [s for s in rec['soft'] if s['mass'] < 0][0]
    overlaps = soft['reference_overlaps']
    scan = max(abs(v[0] - soft['excitation']) for v in soft['cut_scan'].values()) * HARTREE_TO_EV
    above = 1 + int(np.argmax(overlaps[1:]))
    ok &= check(mf.converged and soft['level'] == 'top' and overlaps.size > 2
                and soft['reference'] == above and overlaps[-1] < overlaps[above]
                and soft['excitation'] < 0 and scan < CUT_SCAN_TOL
                and 'soft_mode_reference_overlap' not in names
                and not any(issubclass(w.category, SM.SoftModeExcitedShort) for w in caught),
                'LiH/6-31G 1s^2 1pi_x^2 singlet, M < 0 over three even states: the kept state is '
                'the one of largest overlap above the ground state, not the top root; a '
                'de-excitation, the same at every cut, no overlap flag',
                f"overlaps {np.round(overlaps, 5).tolist()}, kept {soft['reference']}; "
                f"{soft['excitation'] * HARTREE_TO_EV:+.4f} eV (cut scan within {scan:.1e}); {names}")
    return ok


def gcm_soft(mf, channel):
    """(the GCM record of channel's driver with reference='excited', its first soft mode's record,
    that mode's direction, the excited reference, the warnings raised)."""
    drive = SM.soft_mode_singlets if channel == 'singlet' else SM.soft_mode_triplets
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        rec = drive(mf, 'gcm', reference='excited')
    ref = SM.excited_reference(mf)[0]
    A, B = (SM.singlet_tdhf_matrices if channel == 'singlet' else SM.triplet_tdhf_matrices)(ref)
    soft = rec['soft'][0]
    return rec, soft, SM.krein_split(A, B)['modes'][soft['index']]['direction'], ref, caught


def check_overlap_rule():
    ok = True
    ev = HARTREE_TO_EV
    rect = 'H 0 0 0; H 0 0 {0}; H 0 {1} 0; H 0 {1} {0}'
    mf = excited_closed_shell(rect.format(1.0, 1.0), '6-31g', {'B2u': 2, 'B3u': 2}, 'D2h')
    rec, soft, d, ref, _ = gcm_soft(mf, 'triplet')
    names = [f['flag'] for f in rec['numerical_flags']]
    overlaps = soft['reference_overlaps']
    scan = max(abs(v[0] - soft['excitation']) for v in soft['cut_scan'].values()) * ev
    bottom = SM.gcm_levels(ref, d, 'triplet', 'bottom')['excitations'][0] * ev
    ok &= check(mf.converged and soft['mass'] > 0 and soft['level'] == 'overlap'
                and soft['reference'] == int(np.argmax(overlaps)) and soft['reference'] > 0
                and soft['excitation'] > 0 and scan < OVERLAP_SCAN_TOL and bottom < 0
                and 'soft_mode_reference_overlap' not in names
                and 'soft_mode_reference_tie' not in names,
                'square H4/6-31G b2u^2 b3u^2 triplet, M > 0: the even state of largest overlap is '
                'kept, not the bottom one; a positive transition, the same over the cuts',
                f"overlaps {np.round(overlaps, 4).tolist()}, kept {soft['reference']}; "
                f"{soft['excitation'] * ev:+.3f} eV (cut scan within {scan:.1e}; bottom "
                f'{bottom:+.3f}); {names}')
    linear = '; '.join(f'H 0 0 {k * 1.5}' for k in range(4))
    mf = excited_closed_shell(linear, '6-31g', {'A1g': 4, 'A1u': 0})
    rec, soft, d, ref, _ = gcm_soft(mf, 'singlet')
    gcm = SM.gcm_levels(ref, d, 'singlet', 'overlap')
    exc = gcm['excitations']
    bottom = SM.gcm_levels(ref, d, 'singlet', 'bottom')['excitations'][0] * ev
    ok &= check(mf.converged and soft['level'] == 'overlap' and soft['reference'] == 0
                and exc[0] > 0 and exc[0] == np.min(exc[exc > 0]) and np.any(exc < 0)
                and abs(soft['excitation'] - exc[0]) < SAME_ROUTE_TOL and bottom < 0,
                'linear H4/6-31G 1sigma_g^2 2sigma_g^2 singlet, M > 0: the even ground state '
                'kept, its first transition the nearest odd state above it, the one below after',
                f"overlap {soft['reference_overlaps'][0]:.3f}; {exc[0] * ev:+.3f} eV, below "
                f'{exc[exc < 0].max() * ev:+.3f}; bottom {bottom:+.3f} eV')
    mf = excited_closed_shell(rect.format(1.5, 0.72), '6-31g', {'B1u': 2, 'B3g': 2}, 'D2h')
    rec, soft, d, ref, caught = gcm_soft(mf, 'singlet')
    names = [f['flag'] for f in rec['numerical_flags']]
    tied = [s for s in rec['soft'] if s['level'] == 'top']
    rivals = np.sort(tied[0]['reference_overlaps'][1:])[::-1][:2] if tied else []
    ok &= check(mf.converged and 'soft_mode_reference_tie' in names
                and any(issubclass(w.category, SM.SoftModeExcitedShort) for w in caught),
                "rectangular H4/6-31G 1.5 x 0.72 A b1u^2 b3g^2 singlet: two even states above the "
                "ground state share the determinant, 'soft_mode_reference_tie'",
                f'{np.round(rivals, 4).tolist()}; {names}')
    return ok


def run():
    warnings.simplefilter('ignore')
    ok = True
    for title, fn in (('1. singlet matrices', check_matrices),
                      ('2. the aufbau limit', check_aufbau_limit),
                      ('3. the sign-aware split of sigma_u^2', check_split),
                      ('4. the singlet path and its GCM kernels', check_singlet_path),
                      ('5. sigma_u^2 singlet symmetry breaking against FCI', check_singlet_fci),
                      ('6. sigma_u^2 short-bond triplet against FCI', check_triplet_fci),
                      ('7. a maximum-overlap reference', check_mom),
                      ('8. the aufbau singlet block against FCI', check_aufbau_singlet),
                      ('9. refusals', check_refusals),
                      ('10. flags', check_flags),
                      ('11. a degenerate soft pair with A - B indefinite', check_degenerate),
                      ('12. the reference of a negative mass in a multi-pair GCM',
                       check_top_rule),
                      ('13. the reference of a positive mass on an indefinite block, and a near tie',
                       check_overlap_rule)):
        print(f'\n-- {title}')
        ok &= fn()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_soft_mode_excited():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
