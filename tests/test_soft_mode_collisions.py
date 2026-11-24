"""The two-mode generator-coordinate model of a Krein collision in the sign-aware soft-mode split of
an excited closed-shell reference, collisions='gcm' (`LinearResponse.soft_mode.krein_pairs`,
`collision_levels`, `collision_modes`): what is held.

1. Without a collision the option changes nothing: `krein_pairs` is `krein_split` to the bit with
   pairs = [] (H2O, both blocks, an aufbau A - B; H2/6-31G sigma_u^2's singlet at 1.5 A and triplet
   at 0.3 A, A - B indefinite), and soft_mode_singlets / soft_mode_triplets(...,
   reference='excited', collisions='gcm') give the collisions='refuse' record to the bit there
   (omega, X, Y, the soft records), with pairs = [].
2. The plane of a pair (H2/6-31G sigma_u^2: the triplet quartet at 0.98 A and its neighbours at
   0.94 A, the singlet quartet at 0.765 A and its neighbours at 0.76 A): invariant under
   (A - B)(A + B), its 2 x 2 restriction carrying the pair's Omega^2 (the complex-conjugate pair of
   a quartet) to PLANE_TOL relative, orthonormal, the metric (A - B)^-1 diagonal on it with one
   mass of each sign; the mode left outside the pair (sigma_u -> sigma_u') is regular and solves
   the RPA equations with X'X - Y'Y = 1; the default refuses the quartets (CasidaInstability) and
   flags the neighbours ('soft_mode_krein_neighbour').
3. The kernels (`_CollisionKernels`): for one occupied orbital (H2/6-31G sigma_u^2, both
   channels) against the exact two-electron vectors c_pq = a_p b_q of the determinants through
   pyscf's FCI contractions, at eight random pairs of rotations and at four pairs of orthogonal
   determinants, where the overlap is zero and the Hamiltonian is not, to KERNEL_TOL; with more
   occupied orbitals (H2O/STO-3G and H2O/cc-pVDZ, five; linear H4/6-31G's collision plane, two)
   against the generic non-orthogonal rules with an inverse at random rotations; their diagonal
   along e_1 is `path_samples`' V and D.
4. One occupied orbital spans {u^2, v_1^2, v_2^2, v_1 v_2} and {u v_1, u v_2} of the channel's
   spin: the model's even and odd levels and its couplings are the CI in {u, v_1, v_2} to
   CI_TOL (its reference the even state of largest u^2 weight), its odd levels the TDA roots of
   the plane, e^T A e, measured from the reference determinant (to TDA_TOL, the reference's own
   convergence), and its states spin-pure; on any grid (4 or 8 directions, 16 or 32 points per
   direction).
5. Against FCI around the collision windows the paper finds (H2/6-31G sigma_u^2, the windows
   0.7625 to 0.77 A and 0.945 to 1.01 A and rows past their edges; FCI's transitions from the
   singlet of largest sigma_u^2 weight to the states of the channel's spin, weighted by
   |<m|D_1|ref>|^2 + |<m|D_2|ref>|^2, D_k the one-body operators along the plane; the strongest
   state on each side, and their weighted centroid): TDHF is complex inside both windows and
   collisions='refuse' refuses it, while collisions='gcm' gives real poles of positive norm on
   every row. Triplet (0.94, 0.945, 0.95, 0.98, 1.0, 1.01, 1.015 A): the de-excitation within
   TRIPLET_DOWN_TOL and the excitation within TRIPLET_UP_TOL of FCI's strongest states. Singlet
   (0.755, 0.76, 0.7625, 0.765, 0.77, 0.7725, 0.775 A): the de-excitation within SINGLET_DOWN_TOL
   of FCI's; FCI splits the excitation between two states (8.5 and 18 eV), whose centroid the
   model's one pole is within SINGLET_UP_TOL of (the strongest state, printed, lies 3.8 to 4.2 eV
   below it). Printed beside: TDA, TDHF and the one-mode route (refused inside the windows, 1.6 to
   6.5 eV from FCI's strongest state at the neighbour rows).
6. Smooth through the triplet window's edges, where TDHF's two real roots meet and turn complex
   (a square-root branch point): the levels' second differences on 0.94, 0.945, 0.95 A and on
   1.005, 1.01, 1.015 A stay below SMOOTH_TOL.
7. The reference picked by overlap: in the singlet window the reference determinant is shared by
   two even states of the model (0.529 and 0.459 at 0.765 A), the kept one switches between
   0.765 and 0.766 A and the de-excitation jumps by 0.34 eV there; 'soft_mode_collision_reference'
   is raised on both rows (SoftModeCollisionShort) and on none of the triplet rows (overlap 0.956
   to 0.961).
8. Two occupied orbitals (linear H4/6-31G on 1sigma_g^2 2sigma_g^2, the singlet quartet at 0.8 A
   and the triplet one at 1.3 A): more states than one pair's span, finite real levels, and the
   reference determinant shared (overlap below SOFT_MODE_COLLISION_OVERLAP_MIN, flagged), where
   FCI's state of largest reference weight holds only 0.55 and 0.35 of it too; the levels against
   FCI are printed, not held (0.1 and 0.6 eV off at 0.8 A, 2.9 and 2.2 eV at 1.3 A; FCI's
   strongest states and centroid over all 784 states of the (2, 2) sector, a dense solve). The
   soft mode's krein_gap leaves the quartet's modes out (None: every mode of the other signature is
   the quartet's), and 'soft_modes_coupled' counts the colliding pair beside the soft set.
9. The poles (`collision_modes`): X + Y in the plane, X - Y = dE (A - B)^-1 (X + Y),
   X'X - Y'Y = 2 dE c^T diag(masses) c, to POLE_TOL; with one direction they are
   `collective_modes`' to POLE_TOL (6e-17 measured: the two take different floating-point routes).
10. Refusals: an unknown collisions option; collisions='gcm' on an aufbau reference, in both
    drivers (ValueError), and with the relaxed path (NotImplementedError, before any work); a
    quartet degenerate with another (a doubled block, NotImplementedError); a singlet plane no
    symmetry takes to minus itself (two totally symmetric directions of H2O, NotImplementedError);
    a TDHF quartet under the default (CasidaInstability). Flags: 'soft_mode_collision_few_states'
    when the overlap cut keeps fewer than four even and two odd states.
11. The pairing rule next to a degenerate set (`krein_pairs` on a synthetic block: A - B =
    diag(PAIRING_AMB) and (A - B)(A + B) = diag(PAIRING_OMEGA2) in a random orthonormal basis):
    the soft mode's nearest mode of the other signature is a degenerate pair (0.048 from it), so
    it is left unpaired, and so is the lone mode of that signature nearest to it (0.074; paired
    with it before); the degenerate pair's masses come out equal; without the pair the two modes
    pair as neighbours.
12. A near-collision of three modes (`krein_pairs` on synthetic blocks as in claim 11): with
    signatures +, -, + at Omega^2 1.00, 1.06 and 1.07e-3 Ha^2 (NEAR_THREE), the last two, each
    the other's nearest, pair as neighbours (the first two, 0.057 apart, paired before) and the
    first stays soft, its krein_gap counting the pair's mode of the other signature (0.057, below
    SOFT_MODE_KREIN_GAP: flagged 'soft_mode_krein_neighbour', SoftModeExcitedShort); in the
    mirrored order +, +, - (1.00, 1.01 and 1.07e-3) the same, at 0.065. With four, -, +, -, + at
    1.00, 1.01, 1.05 and 1.12e-3 (NEAR_FOUR), the first two pair, and the last two, each the
    other's nearest among the modes left but the third's nearest the pair's second, stay unpaired
    and flagged (0.038 and 0.062), the pairs taken over every mode, not over those left.

The FCI is the house's PyscfFCISolver on the reference's orbitals; the bars hold measured misses.
Run: python tests/test_soft_mode_collisions.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.linalg
from pyscf import ao2mo, fci, gto, scf
from pyscf.fci import spin_op

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.LinearResponse import soft_mode as SM
from src.SingleReference.LinearResponse.casida import CasidaInstability
from src.Solvers.active_space_solver import PyscfFCISolver
from tests.soft_mode_references import GEOMETRIES, rhf

#: The plane's invariance, its restriction's eigenvalues, orthonormality and diagonal metric,
#: relative to the largest |Omega^2| (or 1): 4e-16 measured.
PLANE_TOL = 1e-12
#: Ha. The kernels against exact two-electron vectors (8e-14 measured) and the generic rules
#: (3e-13, H2O/cc-pVDZ), and their diagonal against `path_samples` (3e-13).
KERNEL_TOL = 1e-11
#: Ha. The model against the CI in {u, v_1, v_2} (1e-14 measured) and over the grids (4e-15).
CI_TOL = 1e-12
#: Ha. Its odd levels against the TDA roots of the plane: 1.5e-10 measured, which is the CI's own
#: distance from them, the reference's convergence (its orbital gradient 7e-11 Ha at 0.765 A).
TDA_TOL = 1e-9
#: eV. The triplet against FCI: de-excitation -0.083 to -0.087, excitation -0.246 to -0.295.
TRIPLET_DOWN_TOL = 0.1
TRIPLET_UP_TOL = 0.35
#: eV. The singlet de-excitation against FCI, +0.80 to +1.06 (0.77 A the worst); its excitation
#: against FCI's centroid, -0.20 to +0.07.
SINGLET_DOWN_TOL = 1.1
SINGLET_UP_TOL = 0.25
#: eV. Second differences of the triplet levels across the window's edges: 1e-3 measured.
SMOOTH_TOL = 5e-3
#: The poles' identities: 1e-15 measured.
POLE_TOL = 1e-12
#: The synthetic pairing block (claim 11): its A - B per mode (the signs of the masses), its
#: Omega^2 (Ha^2: a soft mode, a degenerate pair of the other signature 0.048 from it, a lone mode
#: of that signature 0.074 from it, a stiff mode), and the seed of its random orthonormal basis.
PAIRING_AMB = (1.0, -1.0, -1.0, -1.0, 1.0)
PAIRING_OMEGA2 = (1.0e-3, 1.05e-3, 1.05e-3, 1.08e-3, 1.0)
PAIRING_SEED = 3
#: The synthetic blocks of a near-collision of three modes (claim 12), as (A - B, Omega^2) per mode
#: like PAIRING_AMB and PAIRING_OMEGA2: signatures +, -, + with relative gaps 0.057 and 0.0093,
#: and the mirrored order +, +, - with gaps 0.065 and 0.056 (each with a stiff mode).
NEAR_THREE = (((1.0, -1.0, 1.0, 1.0), (1.00e-3, 1.06e-3, 1.07e-3, 1.0)),
              ((1.0, 1.0, -1.0, 1.0), (1.00e-3, 1.01e-3, 1.07e-3, 1.0)))
#: Four near-colliding modes, signatures -, +, -, + (relative gaps 0.0099, 0.038 and 0.062 between
#: neighbours), and a stiff mode.
NEAR_FOUR = ((-1.0, 1.0, -1.0, 1.0, 1.0), (1.00e-3, 1.01e-3, 1.05e-3, 1.12e-3, 1.0))

TRIPLET_ROWS = (0.94, 0.945, 0.95, 0.98, 1.0, 1.01, 1.015)
SINGLET_ROWS = (0.755, 0.76, 0.7625, 0.765, 0.77, 0.7725, 0.775)


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def sigma_u2(R, basis='6-31g'):
    """H2's sigma_u^2 RHF by a symmetry-constrained SCF (both electrons in A1u)."""
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {R}', basis=basis, unit='Angstrom', verbose=0,
                symmetry=True)
    mf = scf.RHF(mol)
    mf.irrep_nelec = {'A1u': 2}
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def h4(R):
    """Linear H4/6-31G (spacing R) on 1sigma_g^2 2sigma_g^2, a symmetry-constrained excited RHF."""
    mol = gto.M(atom='; '.join(f'H 0 0 {k * R}' for k in range(4)), basis='6-31g',
                unit='Angstrom', verbose=0, symmetry=True)
    mf = scf.RHF(mol)
    mf.irrep_nelec = {'A1g': 4, 'A1u': 0}
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def matrices(ref, channel):
    return (SM.singlet_tdhf_matrices if channel == 'singlet' else SM.triplet_tdhf_matrices)(ref)


def driver(channel):
    return SM.soft_mode_singlets if channel == 'singlet' else SM.soft_mode_triplets


def first_pair(ref, channel):
    A, B = matrices(ref, channel)
    return A, B, SM.krein_pairs(A, B)['pairs'][0]


def mo_integrals(mf, C):
    n = C.shape[1]
    return C.T @ mf.get_hcore() @ C, ao2mo.restore(1, ao2mo.kernel(mf.mol, C), n)


def occupied(mf, plane, coeff, channel):
    """The occupied orbitals (alpha, beta) in mf's MO basis at kappa = a e_1 + b e_2."""
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
    n = mf.mo_coeff.shape[1]
    kappa = (coeff[0] * plane[:, 0] + coeff[1] * plane[:, 1]).reshape(nocc, n - nocc)
    K = np.zeros((n, n))
    K[nocc:, :nocc], K[:nocc, nocc:] = 0.5 * kappa.T, -0.5 * kappa
    U = scipy.linalg.expm(K)
    return U[:, :nocc], (U.T if channel == 'triplet' else U)[:, :nocc]


def operators(mf, plane):
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
    n = mf.mo_coeff.shape[1]
    out = []
    for m in range(2):
        D = np.zeros((n, n))
        dm = plane[:, m].reshape(nocc, n - nocc)
        D[:nocc, nocc:], D[nocc:, :nocc] = dm, dm.T
        out.append(D)
    return out


def exact_kernels(mf, plane, pairs, channel):
    """(N, H - E_ref N, <D_1>, <D_2>, <S^2>) of one occupied orbital from the two-electron vectors
    c_pq = a_p b_q through pyscf's FCI contractions: no overlap is inverted."""
    n = mf.mo_coeff.shape[1]
    h1, eri = mo_integrals(mf, mf.mo_coeff)
    h2 = fci.direct_spin1.absorb_h1e(h1, eri, n, (1, 1), 0.5)
    ops = operators(mf, plane)
    out = []
    for c1, c2 in pairs:
        a1, b1 = occupied(mf, plane, c1, channel)
        a2, b2 = occupied(mf, plane, c2, channel)
        v1, v2 = np.outer(a1[:, 0], b1[:, 0]), np.outer(a2[:, 0], b2[:, 0])
        N = np.sum(v1 * v2)
        H = np.sum(v1 * fci.direct_spin1.contract_2e(h2, v2, n, (1, 1))) + mf.energy_nuc() * N
        out.append((N, H - mf.e_tot * N, np.sum(v1 * (ops[0] @ v2)), np.sum(v1 * (ops[1] @ v2)),
                    np.sum(v1 * spin_op.contract_ss(v2, n, (1, 1)))))
    return np.array(out).T


def loewdin_kernels(mf, plane, pairs, channel):
    """The same by the generic non-orthogonal rules with an inverse (no overlap may vanish)."""
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
    h, eri = mo_integrals(mf, mf.mo_coeff)
    ops = operators(mf, plane)
    out = []
    for c1, c2 in pairs:
        N, G = 1.0, []
        for x, y in zip(occupied(mf, plane, c1, channel), occupied(mf, plane, c2, channel)):
            S = x.T @ y
            N *= np.linalg.det(S)
            G.append(y @ np.linalg.solve(S, x.T))
        Gt = G[0] + G[1]
        E = (mf.energy_nuc() + np.einsum('pq,qp', h, Gt)
             + 0.5 * np.einsum('pqrs,qp,sr', eri, Gt, Gt)
             - 0.5 * sum(np.einsum('pqrs,sp,qr', eri, g, g) for g in G))
        out.append((N, N * (E - mf.e_tot), N * np.einsum('pq,qp', ops[0], G[0]),
                    N * np.einsum('pq,qp', ops[1], G[0]), N * (nocc - np.trace(G[0] @ G[1]))))
    return np.array(out).T


def model_kernels(mf, plane, pairs, channel):
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
    kernels = SM._CollisionKernels(mf, plane, nocc, channel)
    bra = kernels.orbitals(pairs[:, 0])
    ket = kernels.orbitals(pairs[:, 1])
    return np.array(kernels.elements(bra, ket))


def three_orbital_ci(ref, plane, channel):
    """(even levels, odd levels, |couplings|) of the CI in {u, v_1, v_2} (v_m = sum_a (e_m)_a
    phi_a), relative to the energy of |u u~>: its even states the singlets of even v-number, its
    odd ones the states of the channel's spin with one electron in v, the couplings
    |<odd|D_m,alpha|ref>| from the even state of largest u^2 weight."""
    C = ref.mo_coeff
    orb = np.stack([C[:, 0], C[:, 1:] @ plane[:, 0], C[:, 1:] @ plane[:, 1]], axis=1)
    h, g = mo_integrals(ref, orb)
    eye = np.eye(3)
    H = (np.einsum('pr,qs->pqrs', h, eye) + np.einsum('pr,qs->pqrs', eye, h)
         + g.transpose(0, 2, 1, 3)).reshape(9, 9)
    e, v = np.linalg.eigh(H)
    vecs = list(v.T.reshape(9, 3, 3))
    e0 = 2.0 * h[0, 0] + g[0, 0, 0, 0]
    parity = np.outer([1.0, -1.0, -1.0], [1.0, -1.0, -1.0])
    vparity = np.array([np.sum(c * c * parity) for c in vecs])
    exchange = np.array([np.sum(c * c.T) for c in vecs])    # +1 singlet, -1 triplet (M_s = 0)
    even = (vparity > 0) & (exchange > 0)
    odd = (vparity < 0) & ((exchange > 0) if channel == 'singlet' else (exchange < 0))
    iref = int(np.argmax(np.where(even, [v[0, 0] ** 2 for v in vecs], -1.0)))
    couplings = []
    for m in np.flatnonzero(odd):
        row = []
        for k in (1, 2):
            D = np.zeros((3, 3))
            D[0, k] = D[k, 0] = 1.0
            row.append(abs(np.sum(vecs[m] * (D @ vecs[iref]))))
        couplings.append(row)
    return e[even] - e0, e[odd] - e0, np.array(couplings)


def fci_strength(ref, plane, channel, dense=False):
    """FCI's transitions from the singlet of largest weight on the reference determinant to the
    states of the channel's spin: (dE (Ha), weight |<m|D_1|ref>|^2 + |<m|D_2|ref>|^2 with
    D_k = (D_k,alpha +- D_k,beta) / sqrt2, the reference weight). The lowest n^2 states by the
    house's PyscfFCISolver (every state for H2/6-31G), or with dense every state of the (nocc,
    nocc) sector by a dense diagonalization of the Hamiltonian over all its determinants (pyscf's
    pspace), so that a centroid sees the whole spectrum (linear H4/6-31G: 784 states)."""
    nocc = int(np.count_nonzero(np.asarray(ref.mo_occ) > 0))
    n = ref.mo_coeff.shape[1]
    h1, eri = mo_integrals(ref, ref.mo_coeff)
    if dense:
        size = fci.cistring.num_strings(n, nocc)
        addr, h0 = fci.direct_spin1.pspace(h1, eri, n, (nocc, nocc), np=size * size)
        e, c = np.linalg.eigh(h0)
        vecs = []
        for k in range(e.size):
            v = np.zeros(size * size)
            v[addr] = c[:, k]
            vecs.append(v.reshape(size, size))
    else:
        e, vecs = PyscfFCISolver().roots(h1, eri, n, (nocc, nocc), nroots=n * n, spin=None)
        vecs = [np.asarray(v) for v in vecs]
    s2 = np.array([spin_op.spin_square(v, n, (nocc, nocc))[0] for v in vecs])
    singlet = s2 < 1.0
    w_ref = np.array([v[0, 0] ** 2 for v in vecs])
    iref = int(np.argmax(np.where(singlet, w_ref, -1.0)))
    sign = 1.0 if channel == 'singlet' else -1.0
    weight = np.zeros(len(e))
    for D in operators(ref, plane):
        dv = fci.direct_uhf.contract_1e((D, sign * D), vecs[iref], n, (nocc, nocc))
        weight += np.array([np.sum(v * dv) ** 2 for v in vecs]) / 2.0
    want = singlet if channel == 'singlet' else np.abs(s2 - 2.0) < 1e-4
    return e - e[iref], np.where(want, weight, 0.0), w_ref[iref]


def strongest(dE, weight, side):
    m = (side * dE > 0) & (weight > 1e-6)
    return dE[m][np.argmax(weight[m])], np.sum(dE[m] * weight[m]) / np.sum(weight[m])


def identical(a, b):
    """True if a and b agree to the bit, through dicts, lists and tuples."""
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(identical(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return (isinstance(b, (list, tuple)) and len(a) == len(b)
                and all(identical(x, y) for x, y in zip(a, b)))
    if a is None or b is None:
        return a is b
    return np.array_equal(np.asarray(a), np.asarray(b))


def check_no_collision():
    ok = True
    mf = rhf(GEOMETRIES['H2O'])
    for name in ('triplet', 'singlet'):
        A, B = matrices(mf, name)
        a, b = SM.krein_split(A, B), SM.krein_pairs(A, B)
        ok &= check(b.pop('pairs') == [] and identical(a, b),
                    f'H2O {name} block (A - B positive definite): krein_pairs = krein_split '
                    'to the bit, pairs = []')
    for channel, R in (('singlet', 1.5), ('triplet', 0.3)):
        mf = sigma_u2(R)
        A, B = matrices(SM.excited_reference(mf)[0], channel)
        a, b = SM.krein_split(A, B), SM.krein_pairs(A, B)
        same = b.pop('pairs') == [] and identical(a, b)
        refuse = driver(channel)(mf, 'gcm', reference='excited')
        gcm = driver(channel)(mf, 'gcm', reference='excited', collisions='gcm')
        ok &= check(same and gcm.pop('pairs') == [] and identical(refuse, gcm),
                    f"sigma_u^2 {R} A {channel} (no collision): krein_pairs = krein_split, the "
                    "record with collisions='gcm' = collisions='refuse' to the bit")
    return ok


def check_plane():
    ok = True
    for channel, R, kind in (('triplet', 0.98, 'quartet'), ('triplet', 0.94, 'neighbours'),
                             ('singlet', 0.765, 'quartet'), ('singlet', 0.76, 'neighbours')):
        mf = sigma_u2(R)
        A, B, pair = first_pair(SM.excited_reference(mf)[0], channel)
        P = pair['plane']
        product = (A - B) @ (A + B)
        scale = np.max(np.abs(np.linalg.eigvals(product)))
        reduced = P.T @ product @ P
        eig2 = np.sort_complex(np.linalg.eigvals(reduced))
        metric = P.T @ np.linalg.inv(A - B) @ P
        miss = max(np.abs(product @ P - P @ reduced).max() / scale,
                   np.abs(eig2 - np.sort_complex(pair['omega2'])).max() / scale,
                   np.abs(P.T @ P - np.eye(2)).max(), abs(metric[0, 1]),
                   np.abs(np.diag(metric) - pair['masses']).max())
        om, X, Y = SM.krein_pairs(A, B)['regular']
        rpa = max(np.abs(A @ X + B @ Y - X * om).max(), np.abs(B @ X + A @ Y + Y * om).max(),
                  np.abs(np.sum(X * X - Y * Y, axis=0) - 1.0).max())
        try:
            driver(channel)(mf, 'gcm', reference='excited')
            default = 'kept'
        except CasidaInstability:
            default = 'refused'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            flags = [f['flag'] for f in driver(channel)(mf, '1d', reference='excited')[
                'numerical_flags']] if default == 'kept' else []
        want_default = ('refused' if kind == 'quartet' else 'kept')
        masses = pair['masses']
        ok &= check(miss < PLANE_TOL and rpa < PLANE_TOL and om.size == 1
                    and pair['kind'] == kind and masses[0] > 0 > masses[1]
                    and default == want_default
                    and (kind == 'quartet' or 'soft_mode_krein_neighbour' in flags),
                    f'sigma_u^2 {R} A {channel} {kind}: the plane is invariant, carries the '
                    "pair's Omega^2, is orthonormal with a diagonal metric of masses of either "
                    f'sign; the regular mode left solves the RPA equations; the default '
                    f'{want_default} it', f'{miss:.1e}, RPA {rpa:.1e}; Omega^2 '
                    f'{np.round(pair["omega2"], 5).tolist()}, masses '
                    f'{np.round(masses, 3).tolist()}')
    return ok


def check_kernels():
    ok = True
    rng = np.random.default_rng(11)
    for channel, R in (('singlet', 0.765), ('triplet', 0.98)):
        ref = SM.excited_reference(sigma_u2(R))[0]
        plane = first_pair(ref, channel)[2]['plane']
        pairs = np.concatenate([rng.uniform(-np.pi, np.pi, (8, 2, 2)),
                                [[[0.0, 0.0], [np.pi, 0.0]], [[0.0, 0.0], [0.0, np.pi]],
                                 [[0.6 * np.pi, 0.8 * np.pi], [0.0, 0.0]],
                                 [[np.pi, 0.0], [0.0, np.pi]]]])
        want = exact_kernels(ref, plane, pairs, channel)
        got = model_kernels(ref, plane, pairs, channel)
        miss = np.abs(got - want).max()
        ok &= check(miss < KERNEL_TOL and np.abs(want[0, -4:]).max() < 1e-12
                    and np.abs(want[1, -4:]).min() > 1e-2,
                    f'sigma_u^2 {R} A {channel}: the kernels = the exact two-electron ones, four '
                    'orthogonal pairs (zero overlap, nonzero H) included', f'{miss:.1e} Ha')
    cases = (('H2O/STO-3G', rhf(GEOMETRIES['H2O'], 'sto-3g'), 'triplet', None),
             ('H2O/cc-pVDZ', rhf(GEOMETRIES['H2O']), 'singlet', None),
             ('linear H4/6-31G 0.8 A', SM.excited_reference(h4(0.8))[0], 'singlet', 'pair'))
    for label, mf, channel, plane in cases:
        if plane is None:
            w, v = np.linalg.eigh(matrices(mf, channel)[0])
            plane = np.linalg.qr(v[:, :2] + 0.3 * v[:, 2:4])[0]
        else:
            plane = first_pair(mf, channel)[2]['plane']
        pairs = rng.uniform(-2.0, 2.0, (8, 2, 2))
        miss = np.abs(model_kernels(mf, plane, pairs, channel)
                      - loewdin_kernels(mf, plane, pairs, channel)).max()
        nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
        L = SM.path_period(plane[:, 0], nocc)[0]
        q, V, D = SM.path_samples(mf, plane[:, 0], L, channel)
        kernels = SM._CollisionKernels(mf, plane, nocc, channel)
        orbs = kernels.orbitals(np.stack([q, np.zeros_like(q)], axis=1))
        values = kernels.elements(orbs, orbs)
        diag = max(np.abs(values[1] - V).max(), np.abs(values[2] - D).max())
        ok &= check(miss < KERNEL_TOL and diag < KERNEL_TOL,
                    f'{label} {channel} ({nocc} occupied): the kernels = the generic '
                    'non-orthogonal ones, their diagonal along e_1 = path_samples',
                    f'{miss:.1e}, {diag:.1e} Ha')
    return ok


def check_one_pair():
    ok = True
    saved = (SM.SOFT_MODE_COLLISION_DIRECTIONS, SM.SOFT_MODE_GCM_POINTS)
    for channel, R in (('triplet', 0.98), ('singlet', 0.765), ('triplet', 0.94)):
        ref = SM.excited_reference(sigma_u2(R))[0]
        A, B, pair = first_pair(ref, channel)
        even, odd, couplings = three_orbital_ci(ref, pair['plane'], channel)
        gcm = SM.collision_levels(ref, pair['plane'], channel)
        tda = np.linalg.eigvalsh(pair['plane'].T @ A @ pair['plane'])
        k = gcm['reference']
        miss = max(np.abs(gcm['even_levels'] - even).max(),
                   np.abs(gcm['excitations'] + gcm['even_levels'][k] - odd).max(),
                   np.abs(np.abs(gcm['couplings']) - couplings).max())
        tda_miss = np.abs(gcm['excitations'] + gcm['even_levels'][k] - tda).max()
        pure = (abs(gcm['spin_square'][0]) < CI_TOL and np.allclose(
            gcm['spin_square'][1], 2.0 if channel == 'triplet' else 0.0, atol=CI_TOL))
        grids = []
        for dirs, points in ((4, 32), (8, 16)):
            SM.SOFT_MODE_COLLISION_DIRECTIONS, SM.SOFT_MODE_GCM_POINTS = dirs, points
            try:
                other = SM.collision_levels(ref, pair['plane'], channel)
            finally:
                SM.SOFT_MODE_COLLISION_DIRECTIONS, SM.SOFT_MODE_GCM_POINTS = saved
            grids.append(max(np.abs(other['excitations'] - gcm['excitations']).max(),
                             np.abs(other['even_levels'] - gcm['even_levels']).max()))
        ok &= check(miss < CI_TOL and tda_miss < TDA_TOL and pure and gcm['ranks'] == (4, 2)
                    and max(grids) < CI_TOL,
                    f'sigma_u^2 {R} A {channel}: the model = the CI in {{u, v_1, v_2}}, its odd '
                    'levels the TDA roots of the plane, spin-pure, ranks (4, 2), on any grid',
                    f'{miss:.1e} Ha; TDA {tda_miss:.1e} Ha; grids {max(grids):.1e} Ha')
    return ok


def fci_rows(channel, rows):
    out = {}
    for R in rows:
        mf = sigma_u2(R)
        ref = SM.excited_reference(mf)[0]
        A, B, pair = first_pair(ref, channel)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rec = driver(channel)(mf, 'gcm', reference='excited', collisions='gcm')
            try:
                one = driver(channel)(mf, 'gcm', reference='excited')['soft'][0]['excitation']
            except CasidaInstability:
                one = None
        dE, weight, w_ref = fci_strength(ref, pair['plane'], channel)
        if pair['kind'] == 'quartet':
            om = np.sqrt(complex(pair['omega2'][1])) * HARTREE_TO_EV
            tdhf = f'{om.real:.2f} +- {abs(om.imag):.2f}i'
        else:
            signature = SM.krein_split(A, B)['signature'][pair['index']]
            omega = signature * np.sqrt(pair['omega2'].real) * HARTREE_TO_EV
            tdhf = ', '.join(f'{x:+.2f}' for x in omega)
        out[R] = dict(rec=rec, pair=rec['pairs'][0], plane=pair, one=one, w_ref=w_ref, tdhf=tdhf,
                      fci_down=strongest(dE, weight, -1), fci_up=strongest(dE, weight, 1))
    return out


def check_fci(rows, channel):
    ok = True
    ev = HARTREE_TO_EV
    for R, row in rows.items():
        p, rec = row['pair'], row['rec']
        down, up = p['deexcitation'], p['excitation']
        miss_down = (down - row['fci_down'][0]) * ev
        miss_up = (up - row['fci_up'][0]) * ev
        miss_centroid = (up - row['fci_up'][1]) * ev
        X, Y = rec['X'], rec['Y']
        norms = np.sum(X * X - Y * Y, axis=0)
        real = (np.all(np.isfinite(rec['omega'])) and np.isrealobj(rec['omega'])
                and np.all(norms > 0))
        complex_tdhf = np.max(np.abs(np.imag(p['omega2']))) > 0
        inside = p['kind'] == 'quartet'
        if channel == 'triplet':
            held = abs(miss_down) < TRIPLET_DOWN_TOL and abs(miss_up) < TRIPLET_UP_TOL
        else:
            held = abs(miss_down) < SINGLET_DOWN_TOL and abs(miss_centroid) < SINGLET_UP_TOL
        one = f"{row['one'] * ev:+.2f}" if row['one'] is not None else 'refused'
        bars = (f'within {TRIPLET_DOWN_TOL} / {TRIPLET_UP_TOL} eV of FCI' if channel == 'triplet'
                else f'within {SINGLET_DOWN_TOL} eV of FCI below, {SINGLET_UP_TOL} eV of its '
                'centroid above')
        tda = p['omega_tda'] * ev
        ok &= check(real and held and complex_tdhf == inside and (row['one'] is None) == inside,
                    f'sigma_u^2 {R} A {channel} ({p["kind"]}): real poles of positive norm, {bars}',
                    f'GCM {down * ev:+.3f} {up * ev:+.3f}; FCI {row["fci_down"][0] * ev:+.3f} '
                    f'{row["fci_up"][0] * ev:+.3f} (centroid {row["fci_up"][1] * ev:+.3f}); TDHF '
                    f'{row["tdhf"]}; TDA {tda[0]:+.3f} {tda[1]:+.3f}; one-mode {one}; overlap '
                    f'{p["reference_overlap"]:.3f}')
    return ok


def check_smooth():
    ok = True
    ev = HARTREE_TO_EV
    for edge in ((0.94, 0.945, 0.95), (1.005, 1.01, 1.015)):
        levels = []
        for R in edge:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p = SM.soft_mode_triplets(sigma_u2(R), 'gcm', reference='excited',
                                          collisions='gcm')['pairs'][0]
            levels.append((p['deexcitation'], p['excitation'], p['kind']))
        second = max(abs(levels[0][k] - 2 * levels[1][k] + levels[2][k]) * ev for k in (0, 1))
        kinds = [x[2] for x in levels]
        ok &= check(second < SMOOTH_TOL and len(set(kinds)) == 2,
                    f'triplet levels smooth across the window edge at {edge[1]} A ({kinds})',
                    f'second difference {second:.1e} eV')
    return ok


def check_reference(singlet, triplet):
    ok = True
    flagged = {R: [f['flag'] for f in row['rec']['numerical_flags']] for R, row in singlet.items()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        next_row = SM.soft_mode_singlets(sigma_u2(0.766), 'gcm', reference='excited',
                                         collisions='gcm')
    warned = any(issubclass(w.category, SM.SoftModeCollisionShort) for w in caught)
    a, b = singlet[0.765]['pair'], next_row['pairs'][0]
    jump = abs(b['deexcitation'] - a['deexcitation']) * HARTREE_TO_EV
    clean = [f['flag'] for row in triplet.values() for f in row['rec']['numerical_flags']]
    following = [f['flag'] for f in next_row['numerical_flags']]
    switched = int(np.argmax(a['reference_overlaps'])) != int(np.argmax(b['reference_overlaps']))
    kept = [r['pair']['reference_overlap'] for r in triplet.values()]
    ok &= check('soft_mode_collision_reference' in flagged[0.765] and warned
                and 'soft_mode_collision_reference' in following and switched and not clean,
                'the singlet window shares the reference determinant: the kept even state switches '
                "between 0.765 and 0.766 A, both flagged 'soft_mode_collision_reference'; the "
                'triplet rows raise no flag',
                f'overlaps {np.round(np.sort(a["reference_overlaps"])[-2:], 3).tolist()} -> '
                f'{np.round(np.sort(b["reference_overlaps"])[-2:], 3).tolist()}; jump '
                f'{jump:.2f} eV; triplet overlaps {min(kept):.3f} to {max(kept):.3f}')
    return ok


def check_two_occupied():
    ok = True
    ev = HARTREE_TO_EV
    for channel, R in (('singlet', 0.8), ('triplet', 1.3)):
        mf = h4(R)
        ref = SM.excited_reference(mf)[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rec = driver(channel)(mf, 'gcm', reference='excited', collisions='gcm')
        p = rec['pairs'][0]
        pair = first_pair(ref, channel)[2]
        dE, weight, w_ref = fci_strength(ref, pair['plane'], channel, dense=True)
        down, up = strongest(dE, weight, -1), strongest(dE, weight, 1)
        names = [f['flag'] for f in rec['numerical_flags']]
        ok &= check(p['kind'] == 'quartet' and p['ranks'][0] > 4 and p['ranks'][1] > 2
                    and np.all(np.isfinite(rec['omega']))
                    and p['reference_overlap'] < SM.SOFT_MODE_COLLISION_OVERLAP_MIN
                    and 'soft_mode_collision_reference' in names and w_ref < 0.6,
                    f'linear H4 {R} A {channel} (two occupied): finite levels, ranks {p["ranks"]}, '
                    'the reference shared (flagged) as in FCI',
                    f'overlap {p["reference_overlap"]:.3f}, FCI reference weight {w_ref:.3f}; GCM '
                    f'{p["deexcitation"] * ev:+.3f} {p["excitation"] * ev:+.3f}, FCI '
                    f'{down[0] * ev:+.3f} {up[0] * ev:+.3f} (centroid {up[1] * ev:+.3f}, all '
                    f'{dE.size} states) eV (printed)')
        gaps = [s['krein_gap'] for s in rec['soft']]
        signs = sorted({s['signature'] for s in rec['soft']})
        coupled = [f['message'] for f in rec['numerical_flags']
                   if f['flag'] == 'soft_modes_coupled']
        ok &= check(bool(rec['soft']) and all(g is None for g in gaps)
                    and any('1 colliding pairs' in m for m in coupled),
                    f"linear H4 {R} A {channel}: krein_gap leaves the quartet's modes out (None: "
                    "no other mode of the other signature); 'soft_modes_coupled' counts the pair",
                    f'soft signatures {signs}, gaps {gaps}')
    return ok


def check_poles():
    ok = True
    ref = SM.excited_reference(sigma_u2(0.98))[0]
    A, B, pair = first_pair(ref, 'triplet')
    gcm = SM.collision_levels(ref, pair['plane'], 'triplet')
    dE, X, Y = SM.collision_modes(gcm['excitations'], gcm['couplings'], pair['plane'],
                                  pair['conjugates'])
    xpy, xmy = X + Y, X - Y
    P = pair['plane']
    miss = max(np.abs(xpy - P @ (P.T @ xpy)).max(),
               np.abs(xmy - np.linalg.solve(A - B, xpy) * dE).max(),
               np.abs(np.sum(X * X - Y * Y, axis=0)
                      - 2 * dE * np.einsum('mk,k,mk->m', gcm['couplings'], pair['masses'],
                                           gcm['couplings'])).max())
    c = gcm['couplings'][:, :1]
    one = SM.collision_modes(dE, np.hstack([c, np.zeros_like(c)]), P, pair['conjugates'])
    ref_modes = SM.collective_modes(dE, c[:, 0], pair['masses'][0], P[:, 0],
                                    pair['conjugates'][:, 0] / pair['masses'][0])
    same = max(np.abs(x - y).max() for x, y in zip(one, ref_modes))
    ok &= check(miss < POLE_TOL and same < POLE_TOL,
                "the poles: X + Y in the plane, X - Y = dE (A - B)^-1 (X + Y), X'X - Y'Y = "
                "2 dE c^T diag(M) c; one direction = collective_modes", f'{miss:.1e}, {same:.1e}')
    return ok


def check_refusals():
    ok = True
    mf = sigma_u2(0.98)
    aufbau = rhf(GEOMETRIES['H2O'])
    ref = SM.excited_reference(mf)[0]
    A, B = matrices(ref, 'triplet')
    doubled = (scipy.linalg.block_diag(A, A), scipy.linalg.block_diag(B, B))
    sym = gto.M(atom=GEOMETRIES['H2O'], basis='cc-pvdz', verbose=0, symmetry=True)
    sym_mf = scf.RHF(sym).run(conv_tol=1e-10)
    occ = np.flatnonzero(sym_mf.mo_occ > 0)
    nocc, nvirt = occ.size, sym_mf.mo_coeff.shape[1] - occ.size
    orbsym = np.asarray(sym_mf.orbsym)
    total = [i * nvirt + a for i in range(nocc) for a in range(nvirt)
             if orbsym[i] == orbsym[nocc + a]]
    flat = np.zeros((nocc * nvirt, 2))
    flat[total[0], 0] = flat[total[1], 1] = 1.0
    cases = [("collisions 'pairs'", lambda: SM.soft_mode_triplets(mf, reference='excited',
                                                                  collisions='pairs'), ValueError),
             ("collisions 'gcm' on an aufbau reference (triplets)",
              lambda: SM.soft_mode_triplets(aufbau, collisions='gcm'), ValueError),
             ("collisions 'gcm' on an aufbau reference (singlets)",
              lambda: SM.soft_mode_singlets(aufbau, 'gcm', collisions='gcm'), ValueError),
             ("collisions 'gcm' with path 'relaxed'",
              lambda: SM.soft_mode_triplets(mf, '1d', 'relaxed', 'excited', 'gcm'),
              NotImplementedError),
             ('a quartet degenerate with another (a doubled block)',
              lambda: SM.krein_pairs(*doubled), NotImplementedError),
             ('a totally symmetric singlet plane (H2O)',
              lambda: SM.collision_levels(sym_mf, flat, 'singlet'), NotImplementedError),
             ("the quartet under collisions='refuse'",
              lambda: SM.soft_mode_triplets(mf, 'gcm', reference='excited'), CasidaInstability)]
    for label, call, kind in cases:
        try:
            call()
            raised = None
        except Exception as exc:   # noqa: BLE001 -- the class is what is checked
            raised = exc
        ok &= check(isinstance(raised, kind), f'refused: {label}',
                    f'{type(raised).__name__}: {str(raised)[:70]}')
    saved = SM.SOFT_MODE_GCM_OVERLAP_CUT
    SM.SOFT_MODE_GCM_OVERLAP_CUT = 0.3
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            rec = SM.soft_mode_triplets(mf, 'gcm', reference='excited', collisions='gcm')
    finally:
        SM.SOFT_MODE_GCM_OVERLAP_CUT = saved
    names = [f['flag'] for f in rec['numerical_flags']]
    warned = any(issubclass(w.category, SM.SoftModeCollisionShort) for w in caught)
    ok &= check('soft_mode_collision_few_states' in names and warned,
                "an overlap cut of 0.3 keeps too few states: 'soft_mode_collision_few_states'",
                f'ranks {rec["pairs"][0]["ranks"]}')
    return ok


def synthetic_block(amb, omega2):
    """(A, B) whose A - B is diag(amb) and (A - B)(A + B) diag(omega2) in a random orthonormal basis
    (PAIRING_SEED): mode k has Omega^2 omega2[k] and mass 1 / amb[k]."""
    amb, omega2 = np.asarray(amb), np.asarray(omega2)
    apb = omega2 / amb
    rng = np.random.default_rng(PAIRING_SEED)
    O = np.linalg.qr(rng.standard_normal((amb.size, amb.size)))[0]
    A, B = np.diag((apb + amb) / 2), np.diag((apb - amb) / 2)
    return O.T @ A @ O, O.T @ B @ O


def check_pairing():
    ok = True
    A, B = synthetic_block(PAIRING_AMB, PAIRING_OMEGA2)
    split = SM.krein_pairs(A, B)
    keep = [0, 3, 4]
    control = SM.krein_pairs(*synthetic_block(np.take(PAIRING_AMB, keep),
                                              np.take(PAIRING_OMEGA2, keep)))
    paired = [(p['kind'], [int(j) for j in p['index']]) for p in control['pairs']]
    masses = sorted(round(m['mass'], 6) for m in split['modes'].values())
    ok &= check(split['pairs'] == [] and split['soft'] == [0, 1, 2, 3]
                and paired == [('neighbours', [0, 1])] and masses == [-1.0, -1.0, -1.0, 1.0],
                'a soft mode whose nearest mode of the other signature is a degenerate set is left '
                'unpaired, and so is the soft mode nearest to it; without the set the two pair',
                f'pairs {split["pairs"]}, soft masses {masses}; control {paired}')
    return ok


def check_three_modes():
    ok = True
    for amb, omega2 in NEAR_THREE:
        split = SM.krein_pairs(*synthetic_block(amb, omega2))
        paired = [(p['kind'], [int(j) for j in p['index']]) for p in split['pairs']]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            gap = SM._krein_gap('soft mode 0', split['modes'][0], split['signature'],
                                split['omega2'], 0) if split['soft'] == [0] else None
        signs = ''.join('+' if a > 0 else '-' for a in amb[:3])
        ok &= check(paired == [('neighbours', [1, 2])] and split['soft'] == [0]
                    and gap is not None and gap < SM.SOFT_MODE_KREIN_GAP
                    and any(issubclass(w.category, SM.SoftModeExcitedShort) for w in caught),
                    f'three near-colliding modes ({signs}): the mutual nearest neighbours pair, the '
                    "third stays soft, flagged 'soft_mode_krein_neighbour' against the pair",
                    f'pairs {paired}, soft {split["soft"]}, krein_gap '
                    f'{"-" if gap is None else f"{gap:.4f}"}')
    split = SM.krein_pairs(*synthetic_block(*NEAR_FOUR))
    paired = [(p['kind'], [int(j) for j in p['index']]) for p in split['pairs']]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gaps = [SM._krein_gap(f'soft mode {s}', split['modes'][s], split['signature'],
                              split['omega2'], s) for s in split['soft']]
    ok &= check(paired == [('neighbours', [0, 1])] and split['soft'] == [2, 3]
                and all(g is not None and g < SM.SOFT_MODE_KREIN_GAP for g in gaps)
                and sum(issubclass(w.category, SM.SoftModeExcitedShort) for w in caught) == 2,
                'four near-colliding modes (-+-+): the first two pair; the last two, the third '
                "one's nearest in that pair, stay unpaired, flagged",
                f'pairs {paired}, soft {split["soft"]}, krein_gap '
                f'{[round(g, 4) for g in gaps if g is not None]}')
    return ok


def run():
    warnings.simplefilter('ignore')
    ok = True
    print('\n-- 1. no collision, no change')
    ok &= check_no_collision()
    print('\n-- 2. the plane of a pair')
    ok &= check_plane()
    print('\n-- 3. the kernels')
    ok &= check_kernels()
    print('\n-- 4. one occupied orbital: the CI in {u, v_1, v_2}')
    ok &= check_one_pair()
    print('\n-- 5. against FCI on the collision windows')
    triplet, singlet = fci_rows('triplet', TRIPLET_ROWS), fci_rows('singlet', SINGLET_ROWS)
    ok &= check_fci(triplet, 'triplet')
    ok &= check_fci(singlet, 'singlet')
    print('\n-- 6. smooth through the triplet window')
    ok &= check_smooth()
    print('\n-- 7. the reference picked by overlap')
    ok &= check_reference(singlet, triplet)
    print('\n-- 8. two occupied orbitals (linear H4)')
    ok &= check_two_occupied()
    print('\n-- 9. the poles')
    ok &= check_poles()
    print('\n-- 10. refusals and flags')
    ok &= check_refusals()
    print('\n-- 11. the pairing rule next to a degenerate set')
    ok &= check_pairing()
    print('\n-- 12. a near-collision of three modes')
    ok &= check_three_modes()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_soft_mode_collisions():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
