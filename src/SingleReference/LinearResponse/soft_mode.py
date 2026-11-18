"""The soft-mode treatment of a closed shell's TDHF triplet instability: the soft triplet modes
split off the TDHF ladder, quantized along the RHF->UHF path, and put back as poles.

At the RHF->UHF (triplet) instability the lowest eigenvalue of the TDHF triplet problem

    A = De - (ij|ab),   B = -(ib|ja)

goes through zero, and a self-energy that sums the triplet ladder over its poles diverges as
1/Omega_s there: the triplet sums of PSD2, PSD4, PSD7 and PSD9
(`GW.self_energy.TRIPLET_VERTEX_MODES`), on the way to the dense Casida refusal
(`LinearResponse.casida.CasidaInstability`) once Omega_s^2 < 0. The Sachdev-Ye-Kitaev model's
four-point ladder fails the same way at its reparametrization modes, and Maldacena and Stanford
(arXiv:1604.07818) take those modes out of the ladder and give them an action of their own. Here
each soft mode gets a collective coordinate Q with the Thouless-Valatin mass:

    T = (A - B)^1/2 (A + B) (A - B)^1/2 = sum_n Omega_n^2 t_n t_n^T      (Omega_n^2 of either sign)
    d = (A - B)^1/2 t_s / |(A - B)^1/2 t_s|,   M = d^T (A - B)^-1 d,   e = (A - B)^-1 d / M,
    mu = d^T (A + B) d = M Omega_s^2,
    H = P^2 / 2M + V(Q),   V(Q) = E_HF[alpha orbitals by exp(K), beta by exp(-K)] - E_RHF,
    K_ai = -K_ia = Q d_ia / 2,

so that V''(0) = mu and the harmonic H has the frequency Omega_s whatever its sign; past the
instability V is a double well. Its levels m (odd under Q -> -Q in 1D, l = 1 in the 3D model of
the three triplet components) re-enter the ladder as poles at dE_m with

    X_m + Y_m = sqrt2 <m|D|0> d,   X_m - Y_m = sqrt2 M dE_m <m|D|0> e,
    D(Q) = 2 sum_ia d_ia P^alpha_ai(Q)  (= Q to first order; sin Q for one occupied orbital),

which for a harmonic V are the removed TDHF eigenpair to rounding, in the house's Casida layout
(pair index i * nvirt + a, X'X - Y'Y = 1). Read as the nocc x nvirt matrix d_ia, d turns pair k
of its singular-value decomposition by sigma_k Q / 2; at Q = +-pi / sigma_1 the leading pair has
turned by +-90 degrees, one and the same determinant, so the 1D model is periodic on
[-pi / sigma_1, pi / sigma_1): exactly so when d has rank one (H2) or its leading pairs are
degenerate (N2's two pi pairs), up to the turn of the minor pairs otherwise. The same model with
walls at +-pi / sigma_1 measures how much the levels owe to that closure, and a difference above
SOFT_MODE_BOUNDARY_SHIFT_MAX is flagged. Modes are treated one at a time: a second soft set gets a
model of its own along its own path, and the couplings between them are left out (flagged).

Model choices on purpose: the straight path in d (no relaxation of the other rotations), the mass
M held at its Q = 0 value and the flat quantization P^2 / 2M are the default, measured against
path='relaxed' and model='gcm' below; not checked against alternatives are D(Q) as the transition
density that couples back, the flat 3D model (a ball of radius pi / sigma_1 rather than the curved
manifold of rotated determinants), and the closure of a straight path whose minor pairs have not
come back.

path='relaxed' (`relaxed_path`, 1D model only) relaxes every other triplet rotation along the way,
in the Thouless-Valatin form: with kappa the full real triplet rotation (alpha by exp(K(kappa)),
beta by exp(-K(kappa)), K_ai = kappa_ia / 2),

    kappa(Q) = argmin { E(kappa) : e^T kappa = Q,  e_t^T kappa = 0 for the other soft modes t },
    V(Q) = E(kappa(Q)) - E_RHF,   M(Q) = kappa'(Q)^T (A - B)^-1 kappa'(Q)   (the RHF A - B),
    H = -1/2 d/dQ M(Q)^-1 d/dQ + V(Q),   periodic on [-Q*, Q*),

with D(Q) as above and the poles built with the Q = 0 mass and e as before. Because
(A + B) d = Omega_s^2 M e, the minimum of the quadratic energy on e^T kappa = Q is kappa = Q d,
so that V''(0) = mu, M(0) = M and the harmonic level is still Omega_s. The path is followed by
warm-started minimisation on Q_k = k L / (SOFT_MODE_RELAXED_POINTS - 1), every step of the
minimiser capped (the plane e^T kappa = Q also holds wrap-around points, where pair angles reach
180 degrees and the determinant is the RHF again); a point that leaves the branch is tried again
from kappa = Q d, and if that leaves it too the relaxation is frozen there (kappa continued along
d, flagged) before the antipode, or the path ends after it. Q* is the antipode of the relaxed
loop, where its leading pair has turned by 90 degrees (the largest singular value of kappa_ia is
pi), and the inertia there comes from fresh minima at Q* -+ delta when the path ends within
SOFT_MODE_RELAXED_TRUST samples past it. The other soft modes are held at zero, as their models
are kept apart on the straight path; free, a nearly soft one turns the path into a saddle
(O3/cc-pVDZ's third mode, Omega^2 = 8e-5 Ha^2, between 0.95 and 1.0 L). A stiffer mode of another
symmetry can do the same (H2CO/cc-pVDZ from 0.9 L along its second triplet, 3.4 eV; N2 from
0.69 L along its 3Delta_u pair), where a warm start falls toward a wrap-around point and the fresh
start, which keeps the symmetry of d, stays on the saddle. So the lowest curvature across the
constraint plane is probed at every sample accepted from a fresh start and at the last relaxed
sample up to the antipode (Lanczos on the constrained Hessian whitened by the RHF's, from gradient
differences); the TDHF triplet modes that carry a negative curvature are held at zero too, and the
path is followed again, up to SOFT_MODE_RELAXED_HOLD_ROUNDS times. A fresh start still accepted or
a negative curvature still found is flagged, and so is a loop that does not close at Q* (the alpha
determinants at -+Q* overlap by less than SOFT_MODE_RELAXED_CLOSURE_MIN, or |D(Q*)| exceeds
SOFT_MODE_RELAXED_COUPLING_MAX).

model='gcm' (`gcm_levels`, straight path only) quantizes the soft mode by the generator-coordinate
method instead of P^2 / 2M: |Psi> = sum_j f_j |Phi(Q_j)> over the path's determinants, with the
exact Hill-Wheeler equation (H - E N) f = 0 on their overlap and energy kernels, which close in
the pair orbitals of d (`_GCMKernels`), solved in the even and odd blocks of Q -> -Q (Phi(-Q) is
Phi(Q) with its spins flipped: even spins and odd spins). Its levels are the odd states above the
even ground state, dE_m = E_odd,m - E_even,0, coupled by c_m = |<odd m|D_alpha|even 0>|, and they
enter the ladder through `collective_modes` with the Thouless-Valatin mass and e of the split. For
one pair (H2) the path spans {|g g~>, |v v~>} and T0 and the model is the CI in {g, v_d},
H2/STO-3G's FCI. A harmonic mode, however, comes out at sqrt(mu d^T (A - B) d) =
Omega_s (M / M_PY)^1/2, above TDHF: the model's inertia at small amplitude is the Peierls-Yoccoz
mass M_PY = 1/d^T (A - B) d <= M (real generator states carry no momentum), while X - Y is closed
with M. The straight path turns degenerate leading pairs together (N2), where the model is neither
size-consistent nor spin-pure (flagged).

`soft_mode_triplets` is the driver; `GW.soft_mode_qp` puts its spectrum into the house's PSD
self-energies. Energies are in Hartree.
"""
import numpy as np
from pyscf import ao2mo, scf
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.linalg import (cho_factor, cho_solve, cholesky, eigh, eigh_tridiagonal, expm,
                          expm_frechet, solve_triangular)
from scipy.optimize import brentq

from src.Base.numerical_flags import collecting, flag, gathered
from src.Base.numerics.constants import (SOFT_MODE_BOUNDARY_SHIFT_MAX, SOFT_MODE_DEGENERACY_TOL,
                                         SOFT_MODE_EDGE_FRACTION, SOFT_MODE_EDGE_WEIGHT_MAX,
                                         SOFT_MODE_GCM_CUT_SCAN, SOFT_MODE_GCM_OVERLAP_CUT,
                                         SOFT_MODE_GCM_PAIR_RATIO, SOFT_MODE_GCM_POINTS,
                                         SOFT_MODE_GRID_1D, SOFT_MODE_GRID_3D, SOFT_MODE_LEVELS,
                                         SOFT_MODE_OMEGA_CUT, SOFT_MODE_PATH_POINTS,
                                         SOFT_MODE_RELAXED_ARMIJO,
                                         SOFT_MODE_RELAXED_CLOSURE_MIN,
                                         SOFT_MODE_RELAXED_COUPLING_MAX,
                                         SOFT_MODE_RELAXED_CURVATURE_FLOOR,
                                         SOFT_MODE_RELAXED_ENERGY_SLACK,
                                         SOFT_MODE_RELAXED_GRADIENT,
                                         SOFT_MODE_RELAXED_HOLD_FRACTION,
                                         SOFT_MODE_RELAXED_HOLD_ROUNDS,
                                         SOFT_MODE_RELAXED_ITERATIONS,
                                         SOFT_MODE_RELAXED_JUMP_RATIO, SOFT_MODE_RELAXED_JUMP_SLACK,
                                         SOFT_MODE_RELAXED_LINE_FLOOR, SOFT_MODE_RELAXED_MEMORY,
                                         SOFT_MODE_RELAXED_PAST, SOFT_MODE_RELAXED_POINTS,
                                         SOFT_MODE_RELAXED_PROBE_SEED, SOFT_MODE_RELAXED_PROBE_STEP,
                                         SOFT_MODE_RELAXED_PROBE_VECTORS, SOFT_MODE_RELAXED_REACH,
                                         SOFT_MODE_RELAXED_SADDLE_BAR, SOFT_MODE_RELAXED_STEP_CAP,
                                         SOFT_MODE_RELAXED_TANGENT_GRADIENT,
                                         SOFT_MODE_RELAXED_TANGENT_SLACK,
                                         SOFT_MODE_RELAXED_TANGENT_STEP, SOFT_MODE_RELAXED_TRUST,
                                         SOFT_MODE_WEIGHT_CUT)
from src.Base.orbital_spaces import ov_energies
from src.Base.utils.linear_algebra.matrix_functions import eigh_sym, invsqrtm_sym, sqrtm_sym
from src.SingleReference.common.eri_blocks import MOEriBlocks
from src.SingleReference.LinearResponse.casida import CasidaInstability

#: The collective models `soft_mode_triplets` quantizes a soft mode in: the 1D and 3D collective
#: Hamiltonians `collective_levels` solves, and the generator-coordinate model of `gcm_levels`
#: ('gcm', on the straight path).
MODELS = ('1d', '3d', 'gcm')
#: The boundaries of the 1D model.
BOUNDARIES = ('periodic', 'walls')
#: The collective paths `soft_mode_triplets` quantizes along: the straight RHF->UHF rotation
#: along d, or the relaxed path of `relaxed_path` (1D model only).
PATHS = ('straight', 'relaxed')


class SoftModeEdgeWeight(RuntimeWarning):
    """The collective ground state reaches the edge of its domain, or its first excitation
    depends on how the path is closed there: the levels are kept."""


class SoftModesCoupled(RuntimeWarning):
    """More than one set of soft modes: each gets its own collective model along its own path, and
    the couplings between them are left out. The levels are kept."""


class SoftModePathShort(RuntimeWarning):
    """The relaxed collective path stopped short: a constrained minimisation ended above its
    gradient bar (its last iterate is kept), or the path left its branch before the antipode and
    the relaxation is frozen from there (the rotation continued along d), or a sample was taken
    from a fresh start or is a constrained saddle once the hold rounds are spent, or the loop does
    not close at its antipode. The levels are kept."""


class SoftModeGCMShort(RuntimeWarning):
    """The generator-coordinate model of a soft mode falls short: the two leading pairs of its
    direction are degenerate, so that the straight path turns them together (where the model is
    neither size-consistent nor spin-pure), or the canonical orthogonalization keeps fewer states
    than one pair's exact span. The levels are kept."""


def check_reference(mf):
    """Refuse a mean field the soft-mode split is not defined for: it needs a converged
    closed-shell RHF with exact integrals and an aufbau occupation."""
    if not isinstance(mf, scf.hf.RHF) or isinstance(mf, scf.rohf.ROHF):
        raise NotImplementedError(f'{type(mf).__name__}: the soft-mode split is implemented for '
                                  'closed-shell RHF references only')
    if getattr(mf, 'xc', None) is not None:
        raise ValueError(f'mf.xc = {mf.xc!r}: the soft-mode split needs a Hartree-Fock reference, '
                         'whose energy defines the RHF->UHF path')
    if getattr(mf, 'with_df', None) is not None:
        raise ValueError('mf.with_df is set: the soft-mode split builds its triplet matrices and '
                         'its path energies from exact integrals, which a density-fitted '
                         'reference does not make stationary')
    if not getattr(mf, 'converged', False):
        raise ValueError('mf.converged = False: the RHF->UHF path starts at a converged RHF')
    occ = np.asarray(mf.mo_occ)
    nocc = int(np.count_nonzero(occ > 0))
    if not (np.all(occ[:nocc] == 2) and np.all(occ[nocc:] == 0)):
        raise ValueError('mf.mo_occ: the soft-mode split needs an aufbau closed-shell occupation')
    return nocc


def check_path(path, model):
    """Refuse a collective path that is not one of `PATHS`, the relaxed path with the
    generator-coordinate model ('gcm', whose closed-form kernels hold on the straight path only:
    NotImplementedError), and the relaxed path in any other model but the 1D one ('1d', the only
    one with a coordinate-dependent mass)."""
    if path not in PATHS:
        raise ValueError(f'path = {path!r}: one of {PATHS}')
    if path == 'relaxed' and model == 'gcm':
        raise NotImplementedError("path = 'relaxed' with model = 'gcm': the generator-coordinate "
                                  'model is built on the closed-form kernels of the straight path; '
                                  'over the relaxed path it is not implemented')
    if path == 'relaxed' and model != '1d':
        raise ValueError(f"path = 'relaxed' with model = {model!r}: the relaxed path is quantized "
                         "in the 1D model ('1d') only")


def triplet_tdhf_matrices(mf):
    """(A, B) of the closed-shell TDHF triplet problem, A = De - (ij|ab) and B = -(ib|ja)
    (Hartree; pair index i * nvirt + a), from the three MO blocks, never forming nao^4.

    The same matrices as `LinearResponse.linear_response.LinearResponseSolver`'s
    build_casida_matrices(nocc, lBSE=True, triplet=True) on the full integrals.
    """
    nocc = check_reference(mf)
    blk = MOEriBlocks.from_mol(mf.mol, mf.mo_coeff, nocc)
    de = ov_energies(mf.mo_energy, nocc)
    n = de.size
    A = np.diag(de) - blk.oovv.transpose(0, 2, 1, 3).reshape(n, n)
    B = -blk.ovov.transpose(0, 3, 2, 1).reshape(n, n)
    return A, B


def soft_split(A, B):
    """The TDHF triplet spectrum with its soft modes split off: a dict with

    omega2   all Omega_n^2 (Ha^2), ascending, negative past an instability
    soft     the indices of the soft set (the lowest mode, every mode below
             SOFT_MODE_OMEGA_CUT, and the rest of their degenerate sets)
    modes    for each soft index: dict(direction d, conjugate e, mass M (Ha^-1),
             stiffness mu = d^T (A + B) d (Ha), omega2)
    regular  (Omega, X, Y) of the other modes, in the house's Casida normalization

    A - B must be positive definite, or the mass is undefined: refused with
    `LinearResponse.casida.CasidaInstability`, as the dense Casida solvers refuse it.
    """
    w_amb, v_amb = eigh_sym(A - B)
    if w_amb[0] <= 0:
        raise CasidaInstability(f'A - B is not positive definite (lowest eigenvalue '
                                f'{w_amb[0]:.3e} Ha): the soft mode has no Thouless-Valatin mass')
    S = sqrtm_sym(A - B, eig=(w_amb, v_amb))
    S_inv = invsqrtm_sym(A - B, eig=(w_amb, v_amb))
    omega2, t = eigh_sym(S @ (A + B) @ S)
    soft = {0} | set(np.flatnonzero(omega2 < SOFT_MODE_OMEGA_CUT ** 2).tolist())
    for s in list(soft):
        soft |= set(np.flatnonzero(np.abs(omega2 - omega2[s]) < SOFT_MODE_DEGENERACY_TOL).tolist())
    soft = sorted(soft)
    modes = {}
    for s in soft:
        d = S @ t[:, s]
        d /= np.linalg.norm(d)
        d *= np.sign(d[np.argmax(np.abs(d))])
        y = np.linalg.solve(A - B, d)
        M = float(d @ y)
        modes[s] = dict(direction=d, conjugate=y / M, mass=M, stiffness=float(d @ (A + B) @ d),
                        omega2=float(omega2[s]))
    reg = np.setdiff1d(np.arange(omega2.size), soft)
    omega = np.sqrt(omega2[reg])
    xpy = S @ t[:, reg] / np.sqrt(omega)
    xmy = S_inv @ t[:, reg] * np.sqrt(omega)
    return dict(omega2=omega2, soft=soft, modes=modes,
                regular=(omega, (xpy + xmy) / 2, (xpy - xmy) / 2))


def path_period(direction, nocc):
    """(pi / sigma_1, sigma): the half-period of the RHF->UHF path along the soft direction, where
    the leading pair of d_ia (an nocc x nvirt matrix) has turned by 90 degrees, and the singular
    values sigma of d_ia (sigma_1 = 1 when d has rank one)."""
    sigma = np.linalg.svd(direction.reshape(nocc, -1), compute_uv=False)
    return np.pi / sigma[0], sigma


def path_samples(mf, direction, q_end):
    """(Q, V, D) on SOFT_MODE_PATH_POINTS points of [0, q_end]: V(Q) = E_HF - E_RHF (Ha) of the
    RHF determinant with its alpha orbitals rotated by exp(K) and its beta orbitals by exp(-K),
    K_ai = -K_ia = Q d_ia / 2, and D(Q) = 2 sum_ia d_ia P^alpha_ai(Q), the alpha density's
    occupied-virtual block along d. V is even and D odd in Q."""
    nocc = check_reference(mf)
    C = mf.mo_coeff
    n = C.shape[1]
    dm = direction.reshape(nocc, n - nocc)
    uhf = scf.UHF(mf.mol)
    q = np.linspace(0.0, q_end, SOFT_MODE_PATH_POINTS)
    V, D = np.empty_like(q), np.empty_like(q)
    for k, qk in enumerate(q):
        K = np.zeros((n, n))
        K[nocc:, :nocc] = 0.5 * qk * dm.T
        K[:nocc, nocc:] = -0.5 * qk * dm
        U = expm(K)
        ca, cb = (C @ U)[:, :nocc], (C @ U.T)[:, :nocc]
        V[k] = uhf.energy_tot(dm=(ca @ ca.T, cb @ cb.T)) - mf.e_tot
        D[k] = 2.0 * np.sum(dm.T * (U[nocc:, :nocc] @ U[:nocc, :nocc].T))
    return q, V, D


def _rotation(kappa, nocc, nmo):
    """K(kappa), nmo x nmo: K_ai = kappa_ia / 2 = -K_ia (kappa in the pair index i * nvirt + a)."""
    K = np.zeros((nmo, nmo))
    X = 0.5 * np.reshape(kappa, (nocc, nmo - nocc))
    K[nocc:, :nocc] = X.T
    K[:nocc, nocc:] = -X
    return K


def _largest_angle(kappa, nocc):
    """sigma_1 of kappa read as the nocc x nvirt matrix kappa_ia: its leading pair turns by
    sigma_1 / 2 (sigma_1 = pi at the 90-degree point; Q sigma_1(d) on the straight path)."""
    return float(np.linalg.svd(np.reshape(kappa, (nocc, -1)), compute_uv=False)[0])


def _coupling(kappa, direction, nocc, nmo):
    """D = 2 sum_ia d_ia P^alpha_ai of the alpha orbitals rotated by exp(K(kappa)), as in
    `path_samples` (on the triplet manifold P^beta_ai = -P^alpha_ai)."""
    U = expm(_rotation(kappa, nocc, nmo))
    return float(2.0 * np.sum(np.reshape(direction, (nocc, -1)).T
                              * (U[nocc:, :nocc] @ U[:nocc, :nocc].T)))


class _TripletEnergy:
    """E(kappa) - E_RHF (Ha) of the RHF determinant with its alpha orbitals rotated by exp(K(kappa))
    and its beta orbitals by exp(-K(kappa)), the energy of `path_samples` for any triplet rotation,
    with its gradient in kappa (through the Frechet derivative of the exponential); `calls` counts
    the evaluations."""

    def __init__(self, mf, nocc):
        self.mol, self.C, self.nocc = mf.mol, mf.mo_coeff, nocc
        self.nmo = self.C.shape[1]
        self.uhf = scf.UHF(mf.mol)
        self.hcore = self.uhf.get_hcore()
        self.e_elec = mf.e_tot - mf.mol.energy_nuc()
        self.calls = 0

    def __call__(self, kappa):
        o, C = self.nocc, self.C
        self.calls += 1
        K = _rotation(kappa, o, self.nmo)
        U = expm(K)
        ca, cb = (C @ U)[:, :o], (C @ U.T)[:, :o]
        da, db = ca @ ca.T, cb @ cb.T
        va, vb = self.uhf.get_veff(self.mol, np.array([da, db]))
        E = (np.einsum('ij,ji->', self.hcore, da + db)
             + 0.5 * (np.einsum('ij,ji->', va, da) + np.einsum('ij,ji->', vb, db)) - self.e_elec)
        grad = []
        for Ks, c, v in ((K, ca, va), (-K, cb, vb)):
            dU = np.zeros((self.nmo, self.nmo))
            dU[:, :o] = 2.0 * C.T @ ((self.hcore + v) @ c)
            dK = expm_frechet(Ks.T, dU, compute_expm=False)
            grad.append(0.5 * (dK[o:, :o].T - dK[:o, o:]).ravel())
        return float(E), grad[0] - grad[1]


def _capped_lbfgs(fun, w, step_norm, bar):
    """(w, f, converged) of min f(w) by L-BFGS: fun(w) gives (f, gradient), every step is cut to
    SOFT_MODE_RELAXED_STEP_CAP in step_norm (so that a warm start stays on its branch) and
    backtracked to Armijo's condition; converged when |gradient| < bar within
    SOFT_MODE_RELAXED_ITERATIONS steps, else the last iterate."""
    f, g = fun(w)
    S, Y = [], []
    for _ in range(SOFT_MODE_RELAXED_ITERATIONS):
        if np.linalg.norm(g) < bar:
            return w, f, True
        p = g.copy()
        alphas = []
        for s, y in zip(reversed(S), reversed(Y)):
            alphas.append((s @ p) / (y @ s))
            p -= alphas[-1] * y
        p *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1]) if S else 1.0
        for (s, y), a in zip(zip(S, Y), reversed(alphas)):
            p += s * (a - (y @ p) / (y @ s))
        p = -p
        if p @ g >= 0:
            p, S, Y = -g, [], []
        length = step_norm(p)
        if length > SOFT_MODE_RELAXED_STEP_CAP:
            p *= SOFT_MODE_RELAXED_STEP_CAP / length
        t = 1.0
        while True:
            fn, gn = fun(w + t * p)
            if (fn <= f + SOFT_MODE_RELAXED_ARMIJO * t * (p @ g) + SOFT_MODE_RELAXED_ENERGY_SLACK
                    or t < SOFT_MODE_RELAXED_LINE_FLOOR):
                break
            t *= 0.5
        s, y = t * p, gn - g
        if s @ y > SOFT_MODE_RELAXED_CURVATURE_FLOOR * np.linalg.norm(s) * np.linalg.norm(y):
            S.append(s)
            Y.append(y)
            if len(S) > SOFT_MODE_RELAXED_MEMORY:
                S.pop(0)
                Y.pop(0)
        w, f, g = w + t * p, fn, gn
    return w, f, bool(np.linalg.norm(g) < bar)


class _ConstrainedMinima:
    """Minimisation of energy(kappa) over kappa = Q d + Z u at fixed Q: Z an orthonormal basis of
    the rotations with e^T kappa = 0 and e_t^T kappa = 0 for the held modes, so that e^T kappa = Q
    and e_t^T kappa = 0 (e^T d = 1, e_t^T d = 0), and u in variables w = L^T u whitened by the RHF
    Hessian, L L^T = Z^T (A + B) Z (positive definite once every other soft mode is held). What it
    reaches is a stationary point: a start that keeps a symmetry the minimum breaks (kappa = Q d)
    stays on a saddle, which `_curvature_probe` finds."""

    def __init__(self, energy, direction, conjugate, held, apb):
        rows = np.column_stack([conjugate] + list(held))
        self.Z = np.linalg.qr(rows, mode='complete')[0][:, rows.shape[1]:]
        self.size = self.Z.shape[1]
        self.chol = cholesky(self.Z.T @ apb @ self.Z, lower=True) if self.size else None
        self.energy, self.direction = energy, direction

    def u(self, w):
        return solve_triangular(self.chol.T, w, lower=False) if self.size else w

    def kappa(self, Q, w):
        return Q * self.direction + self.Z @ self.u(w)

    def whitened(self, Q, kappa):
        """w of a rotation kappa with e^T kappa = Q (and e_t^T kappa = 0)."""
        return self.chol.T @ (self.Z.T @ (kappa - Q * self.direction)) if self.size else np.zeros(0)

    def value_and_gradient(self, Q, w):
        """(energy, its gradient in w) at Q and w."""
        E, g = self.energy(self.kappa(Q, w))
        return E, solve_triangular(self.chol, self.Z.T @ g, lower=True)

    def minimum(self, Q, w, bar):
        """(w, V, converged) at Q from the start w."""
        if not self.size:
            return w, self.energy(Q * self.direction)[0], True
        return _capped_lbfgs(lambda x: self.value_and_gradient(Q, x), w,
                             lambda p: float(np.linalg.norm(self.u(p))), bar)


def _curvature_probe(minima, Q, w):
    """(theta, v): Ritz values (ascending) and vectors of the constrained Hessian at the relaxed
    sample (Q, w) in the whitened variables of `_ConstrainedMinima` (where the RHF's Hessian is 1),
    over a Krylov space of SOFT_MODE_RELAXED_PROBE_VECTORS vectors (Lanczos with full
    reorthogonalization from a start drawn with SOFT_MODE_RELAXED_PROBE_SEED), each Hessian-vector
    product a central difference of the whitened gradient over SOFT_MODE_RELAXED_PROBE_STEP. The
    lowest Ritz value lies above the lowest eigenvalue, so a negative one is a direction across the
    constraint plane along which the sample is a saddle."""
    n = min(SOFT_MODE_RELAXED_PROBE_VECTORS, minima.size)
    basis, image = np.zeros((minima.size, n)), np.zeros((minima.size, n))
    v = np.random.default_rng(SOFT_MODE_RELAXED_PROBE_SEED).standard_normal(minima.size)
    for j in range(n):
        for _ in range(2):
            v = v - basis[:, :j] @ (basis[:, :j].T @ v)
        length = np.linalg.norm(v)
        if length == 0.0:
            n = j
            break
        basis[:, j] = v / length
        step = SOFT_MODE_RELAXED_PROBE_STEP * basis[:, j]
        image[:, j] = ((minima.value_and_gradient(Q, w + step)[1]
                        - minima.value_and_gradient(Q, w - step)[1])
                       / (2.0 * SOFT_MODE_RELAXED_PROBE_STEP))
        v = image[:, j].copy()
    theta, s = eigh_sym(basis[:, :n].T @ image[:, :n])
    return theta, basis[:, :n] @ s


def _regular_modes(A, B):
    """(index, omega2, omega, X - Y) of the TDHF triplet modes outside the soft set: their
    indices into `soft_split`'s omega2, Omega_n^2 (Ha^2), Omega_n (Ha) and their X - Y columns
    (house Casida normalization), so that a rotation kappa off every soft mode is
    sum_n c_n (X + Y)_n with c_n = (X - Y)_n^T kappa, and
    kappa^T (A + B) kappa = sum_n Omega_n c_n^2."""
    split = soft_split(A, B)
    index = np.setdiff1d(np.arange(split['omega2'].size), split['soft'])
    omega, X, Y = split['regular']
    return index, split['omega2'][index], omega, X - Y


def _saddle_modes(minima, directions, regular):
    """The TDHF triplet modes outside the soft set (positions in the arrays of `_regular_modes`)
    that carry the negative curvatures `directions` (whitened unit vectors of `_curvature_probe`):
    each mode with SOFT_MODE_RELAXED_HOLD_FRACTION of one of them, its fraction Omega_n c_n^2 in
    the RHF Hessian's metric (they add up to 1), and the rest of its degenerate set."""
    index, omega2, omega, xmy = regular
    fractions = omega[:, None] * (xmy.T @ (minima.Z @ minima.u(directions))) ** 2
    picked = set(np.flatnonzero(np.any(fractions >= SOFT_MODE_RELAXED_HOLD_FRACTION, axis=1)))
    for j in list(picked):
        picked |= set(np.flatnonzero(np.abs(omega2 - omega2[j]) < SOFT_MODE_DEGENERACY_TOL))
    return sorted(picked)


def _relaxed_samples(minima, nocc, h):
    """The relaxed path on Q_k = k h, continued from Q = 0: a dict with q, V (Ha), kappa, w (the
    whitened rotations of the relaxed samples), relaxed (their count: the samples 0 .. relaxed - 1
    are constrained stationary points, minima unless `_curvature_probe` finds a negative curvature
    there), frozen_from (None, or the Q from which kappa is continued along d), passed (the first
    sample past the 90-degree point of the leading pair), converged, cold_restarts (the fresh
    starts tried) and restarted (the samples accepted from one).

    Each Q_k starts from the quadratic extrapolation of the last three minima; a minimum whose step
    from the last one exceeds SOFT_MODE_RELAXED_JUMP_RATIO times the larger of the step before and
    the extrapolated step, plus SOFT_MODE_RELAXED_JUMP_SLACK, has left the branch and is tried
    again from kappa = Q d. If that leaves it too, the relaxation is frozen before the antipode,
    or the path ends after it. The path runs SOFT_MODE_RELAXED_PAST samples past passed, the first
    sample beyond the antipode (so SOFT_MODE_RELAXED_PAST + 1 beyond it); one that has not reached
    the antipode by SOFT_MODE_RELAXED_REACH half-periods is frozen there.
    """
    d = minima.direction
    reach = SOFT_MODE_RELAXED_REACH * path_period(d, nocc)[0]
    q, V, kap, ws = [0.0], [0.0], [np.zeros_like(d)], [np.zeros(minima.size)]
    converged, cold, frozen, passed, restarted = True, 0, None, None, []
    k = 0
    while passed is None or k < passed + SOFT_MODE_RELAXED_PAST:
        k += 1
        Q = k * h
        if frozen is None and passed is None and Q > reach:
            frozen = q[-1]
        accepted = None
        if frozen is None:
            start = (3 * ws[-1] - 3 * ws[-2] + ws[-3] if len(ws) >= 3
                     else 2 * ws[-1] - ws[-2] if len(ws) == 2 else ws[-1])
            step = max(np.linalg.norm(kap[-1] - kap[-2]) if len(kap) >= 2 else h,
                       np.linalg.norm(minima.kappa(Q, start) - kap[-1]))
            for attempt, w0 in enumerate((start, np.zeros(minima.size))):
                cold += attempt
                w, E, ok = minima.minimum(Q, w0, SOFT_MODE_RELAXED_GRADIENT)
                kq = minima.kappa(Q, w)
                if (np.linalg.norm(kq - kap[-1])
                        <= SOFT_MODE_RELAXED_JUMP_RATIO * step + SOFT_MODE_RELAXED_JUMP_SLACK):
                    accepted = (w, E, ok, kq)
                    if attempt:
                        restarted.append(k)
                    break
                if not minima.size:
                    break
            if accepted is None:
                if passed is not None:
                    break
                frozen = q[-1]
        if accepted is None:
            kq = kap[-1] + (Q - q[-1]) * d
            E = minima.energy(kq)[0]
        else:
            w, E, ok, kq = accepted
            converged &= ok
            ws.append(w)
        q.append(Q)
        V.append(E)
        kap.append(kq)
        if passed is None and _largest_angle(kq, nocc) >= np.pi:
            passed = k
    return dict(q=np.array(q), V=np.array(V), kappa=np.array(kap), w=ws, relaxed=len(ws),
                frozen_from=frozen, passed=passed, converged=converged, cold_restarts=cold,
                restarted=restarted)


def relaxed_path(mf, A, B, mode, held):
    """The relaxed collective path of one soft mode (the module docstring's path='relaxed'): a dict
    with

    q, V, D            samples Q_k = k L / (SOFT_MODE_RELAXED_POINTS - 1), L = pi / sigma_1, from 0
                       to SOFT_MODE_RELAXED_PAST samples past the first one beyond the antipode (or
                       to where the path left its branch after it): V(Q) = E(kappa(Q)) - E_RHF (Ha)
                       and D(Q)
    kappa              the triplet rotation at each sample (pair index i * nvirt + a)
    relaxed            the number of relaxed samples (0 .. relaxed - 1; any later ones are
                       continued along d)
    antipode           Q*, where the leading pair of kappa_ia has turned by 90 degrees
    q_mass, mass       the path inertia M(Q) = kappa'^T (A - B)^-1 kappa' (Ha^-1) on [0, Q*], from
                       the cubic spline of kappa at samples with SOFT_MODE_RELAXED_TRUST relaxed
                       samples beyond them, M at Q* from that spline or (path ending sooner)
                       from fresh minima at Q* -+ SOFT_MODE_RELAXED_TANGENT_STEP h, and M on a
                       frozen part
    mass_antipode, coupling_antipode   M(Q*) (Ha^-1) and D(Q*)
    closure_overlap    |<alpha determinant at -Q*|alpha determinant at Q*>|: 1 for a closed loop
    tangent            where M(Q*) comes from: 'spline', 'minima' or 'frozen'
    held_modes         the TDHF triplet modes outside the soft set (indices into `soft_split`'s
                       omega2) held at zero besides held, for the negative curvatures they carried
    curvature          the lowest Ritz value of `_curvature_probe` over the probed samples (in units
                       of the RHF's curvature; None when no rotation is free): negative, a saddle
    restarted          the Q of the samples accepted from a fresh start at kappa = Q d
    converged, cold_restarts, frozen_from, energy_calls

    A and B are `triplet_tdhf_matrices`, mode the soft mode's record from `soft_split`, held the
    conjugates e_t of the soft modes kept at zero along the path. The rotation is minimised over
    the full real triplet space. The curvature across the constraint plane is probed at every
    sample accepted from a fresh start and at the last relaxed sample up to the antipode; where it
    is below -SOFT_MODE_RELAXED_SADDLE_BAR, the modes that carry it (`_saddle_modes`) are held at
    zero too and the path is followed again, up to SOFT_MODE_RELAXED_HOLD_ROUNDS times, the last
    path's samples and probes kept (converged, cold_restarts, restarted and curvature are its own;
    energy_calls counts every round). The 1D model of `collective_levels` quantizes the path with
    mass_profile=(q_mass, mass) on [-Q*, Q*).
    """
    nocc = check_reference(mf)
    nmo = mf.mo_coeff.shape[1]
    d, M0 = mode['direction'], mode['mass']
    h = path_period(d, nocc)[0] / (SOFT_MODE_RELAXED_POINTS - 1)
    energy = _TripletEnergy(mf, nocc)
    regular, extra = None, []
    for rounds in range(SOFT_MODE_RELAXED_HOLD_ROUNDS + 1):
        held_now = list(held) + [regular[3][:, j] for j in extra]
        minima = _ConstrainedMinima(energy, d, mode['conjugate'], held_now, A + B)
        path = _relaxed_samples(minima, nocc, h)
        curvature, negative = None, []
        if minima.size:
            for k in sorted(set(path['restarted']) | {min(path['passed'], path['relaxed'] - 1)}):
                theta, vectors = _curvature_probe(minima, path['q'][k], path['w'][k])
                curvature = theta[0] if curvature is None else min(curvature, theta[0])
                negative.append(vectors[:, theta < -SOFT_MODE_RELAXED_SADDLE_BAR])
        if rounds == SOFT_MODE_RELAXED_HOLD_ROUNDS or not any(v.shape[1] for v in negative):
            break
        if regular is None:
            regular = _regular_modes(A, B)
        new = [j for j in _saddle_modes(minima, np.hstack(negative), regular) if j not in extra]
        if not new:
            break
        extra += new
    q, kap, m, frozen = path['q'], path['kappa'], path['relaxed'], path['frozen_from']
    spline = (CubicSpline(np.r_[-q[1:m][::-1], q[:m]], np.r_[-kap[1:m][::-1], kap[:m]], axis=0)
              if m > 1 else None)

    def kappa_at(Q):
        return kap[m - 1] + (Q - frozen) * d if frozen is not None and Q >= frozen else spline(Q)
    j = path['passed']
    q_star = brentq(lambda x: _largest_angle(kappa_at(x), nocc) - np.pi, q[j - 1], q[j])
    factor = cho_factor(A - B)

    def inertia(t):
        return float(t @ cho_solve(factor, t))
    trusted = m - 1 - SOFT_MODE_RELAXED_TRUST
    table = [k for k in np.flatnonzero(q < q_star) if k == 0 or k <= trusted or k >= m]
    q_mass = [q[k] for k in table]
    mass = [inertia(spline(q[k], 1)) if k < m and m > 1 else M0 for k in table]
    converged = path['converged']
    if frozen is not None and q_star > frozen:
        m_star, tangent = M0, 'frozen'
    elif q_star <= trusted * h:
        m_star, tangent = inertia(spline(q_star, 1)), 'spline'
    else:
        delta = SOFT_MODE_RELAXED_TANGENT_STEP * h
        ends, on_branch = [], True
        for Q in (q_star + delta, q_star - delta):
            guess = spline(Q)
            w, _, ok = minima.minimum(Q, minima.whitened(Q, guess),
                                      SOFT_MODE_RELAXED_TANGENT_GRADIENT)
            ends.append(minima.kappa(Q, w))
            converged &= ok
            on_branch &= np.linalg.norm(ends[-1] - guess) <= SOFT_MODE_RELAXED_TANGENT_SLACK
        if on_branch:
            m_star, tangent = inertia((ends[0] - ends[1]) / (2 * delta)), 'minima'
        else:
            m_star, tangent, converged = inertia(spline(q_star, 1)), 'spline', False
    k_star = kappa_at(q_star)
    U = expm(_rotation(k_star, nocc, nmo))
    overlap = abs(np.linalg.det(U[:, :nocc].T @ U.T[:, :nocc]))
    D = np.array([_coupling(x, d, nocc, nmo) for x in kap])
    return dict(q=q, V=path['V'], D=D, kappa=kap, relaxed=m, antipode=float(q_star),
                q_mass=np.array(q_mass + [q_star]), mass=np.array(mass + [m_star]),
                mass_antipode=m_star, coupling_antipode=_coupling(k_star, d, nocc, nmo),
                closure_overlap=float(overlap), tangent=tangent,
                held_modes=sorted(int(regular[0][j]) for j in extra),
                curvature=None if curvature is None else float(curvature),
                restarted=q[path['restarted']], converged=bool(converged),
                cold_restarts=path['cold_restarts'], frozen_from=frozen,
                energy_calls=energy.calls)


def _splines(q, V, D):
    """Cubic splines of the even V and the odd D on [-q_end, q_end]."""
    qf = np.r_[-q[:0:-1], q]
    return CubicSpline(qf, np.r_[V[:0:-1], V]), CubicSpline(qf, np.r_[-D[:0:-1], D])


def _parity_blocks(n, periodic):
    """(even, odd) orthonormal bases of the reflection Q -> -Q on the 1D grid: n points of
    [-L, L) with the point -L (periodic, n even), or n interior points of (-L, L) (walls, n odd)."""
    if periodic:
        pairs = [(j, n - j) for j in range(1, n // 2)]
        fixed = [0, n // 2]
    else:
        pairs = [(j, n - 1 - j) for j in range(n // 2)]
        fixed = [n // 2]
    even = np.zeros((n, len(fixed) + len(pairs)))
    odd = np.zeros((n, len(pairs)))
    for k, j in enumerate(fixed):
        even[j, k] = 1.0
    r = np.sqrt(0.5)
    for k, (i, j) in enumerate(pairs):
        even[i, len(fixed) + k] = even[j, len(fixed) + k] = r
        odd[i, k], odd[j, k] = r, -r
    return even, odd


def _levels_1d(mass, Vs, Ds, L, periodic):
    """Odd excitations dE_m (Ha), <m|D|0> and the edge weight of |0>, 1D model on [-L, L]."""
    n = SOFT_MODE_GRID_1D if periodic else SOFT_MODE_GRID_1D + 1
    h = 2 * L / (n if periodic else n + 1)
    q = -L + h * (np.arange(n) if periodic else np.arange(1, n + 1))
    c = np.array([30.0, -16.0, 1.0]) / (24.0 * mass * h * h)
    H = np.diag(Vs(q) + c[0])
    for s in (1, 2):
        H += np.diag(np.full(n - s, c[s]), s) + np.diag(np.full(n - s, c[s]), -s)
        if periodic:
            H[np.arange(n - s, n), np.arange(s)] = c[s]
            H[np.arange(s), np.arange(n - s, n)] = c[s]
    even, odd = _parity_blocks(n, periodic)
    e0, v0 = eigh(even.T @ H @ even, subset_by_index=[0, 0])
    e1, v1 = eigh(odd.T @ H @ odd, subset_by_index=[0, SOFT_MODE_LEVELS - 1])
    psi0 = even @ v0[:, 0]
    cm = (odd @ v1).T @ (Ds(q) * psi0)
    edge = float(np.sum(psi0[np.abs(q) > (1 - SOFT_MODE_EDGE_FRACTION) * L] ** 2))
    return e1 - e0[0], cm, edge


def _levels_3d(mass, Vs, Ds, L):
    """l = 1 excitations dE_m (Ha) from the l = 0 ground state of the isotropic 3D model (the
    three triplet components), <m 1 0|D(r) cos(theta)|0 0 0>, and the edge weight of |0>; radial
    equation on (0, L) with nodes at both ends."""
    n = SOFT_MODE_GRID_3D
    h = L / (n + 1)
    r = h * np.arange(1, n + 1)
    off = np.full(n - 1, -1.0 / (2 * mass * h * h))
    e0, u0 = eigh_tridiagonal(Vs(r) + 1.0 / (mass * h * h), off, select='i', select_range=(0, 0))
    e1, u1 = eigh_tridiagonal(Vs(r) + 1.0 / (mass * h * h) + 1.0 / (mass * r * r), off,
                              select='i', select_range=(0, SOFT_MODE_LEVELS - 1))
    cm = (u1.T @ (Ds(r) * u0[:, 0])) / np.sqrt(3.0)
    edge = float(np.sum(u0[r > (1 - SOFT_MODE_EDGE_FRACTION) * L, 0] ** 2))
    return e1 - e0[0], cm, edge


def _staggered_derivative(n, h, periodic):
    """The 4th-order first derivative (1, -27, 27, -1) / (24 h) from the n nodes of the 1D grid to
    the half-nodes between them: n half-nodes j + 1/2 (periodic) or the n + 1 from -1/2 to n - 1/2
    (walls, the wave function zero beyond the n interior nodes)."""
    rows = np.arange(n if periodic else n + 1)
    j = rows if periodic else rows - 1
    out = np.zeros((rows.size, n))
    for c, offset in zip(np.array([1.0, -27.0, 27.0, -1.0]) / (24.0 * h), (-1, 0, 1, 2)):
        i = j + offset
        if periodic:
            out[rows, i % n] = c
        else:
            inside = (i >= 0) & (i < n)
            out[rows[inside], i[inside]] = c
    return out


def _levels_1d_mass(Ms, Vs, Ds, L, periodic):
    """Odd excitations dE_m (Ha), <m|D|0> and the edge weight of |0> of H = -1/2 d/dQ M(Q)^-1 d/dQ
    + V(Q) on the grid of `_levels_1d`: the kinetic term D^T M^-1 D / 2 with D the staggered
    derivative (`_staggered_derivative`) and M at the half-nodes, symmetric, and commuting with
    Q -> -Q for an even M; a constant M gives the levels of `_levels_1d` to the order of the
    grid."""
    n = SOFT_MODE_GRID_1D if periodic else SOFT_MODE_GRID_1D + 1
    h = 2 * L / (n if periodic else n + 1)
    q = -L + h * (np.arange(n) if periodic else np.arange(1, n + 1))
    half = q + 0.5 * h if periodic else np.r_[q[0] - 0.5 * h, q + 0.5 * h]
    Dq = _staggered_derivative(n, h, periodic)
    H = 0.5 * Dq.T @ (Dq / Ms(half)[:, None]) + np.diag(Vs(q))
    even, odd = _parity_blocks(n, periodic)
    e0, v0 = eigh(even.T @ H @ even, subset_by_index=[0, 0])
    e1, v1 = eigh(odd.T @ H @ odd, subset_by_index=[0, SOFT_MODE_LEVELS - 1])
    psi0 = even @ v0[:, 0]
    cm = (odd @ v1).T @ (Ds(q) * psi0)
    edge = float(np.sum(psi0[np.abs(q) > (1 - SOFT_MODE_EDGE_FRACTION) * L] ** 2))
    return e1 - e0[0], cm, edge


def collective_levels(mass, q, V, D, L, model='1d', boundary='periodic', mass_profile=None):
    """(dE, <m|D|0>, edge weight) of the collective Hamiltonian P^2 / 2M + V(Q) of one soft mode,
    from the path samples (q, V, D) of `path_samples` on [0, L]: its excitations that couple to
    the ground state (Ha), their transition matrix elements of D, and the ground state's weight on
    the outer SOFT_MODE_EDGE_FRACTION of the domain.

    model '1d' (the collinear M_s = 0 component, boundary 'periodic' on [-L, L) or 'walls' at
    +-L) or '3d' (three isotropic triplet components, radial equation with nodes at r = 0 and
    r = L; boundary is not read); 'gcm' is refused, its levels come from `gcm_levels`.

    mass_profile: None for the constant mass, or samples (q_m, M) of a coordinate-dependent mass
    M(Q) (Ha^-1) on [0, L], interpolated as an even function (PCHIP) and quantized as
    -1/2 d/dQ M(Q)^-1 d/dQ (`relaxed_path`; model '1d' only, mass is then not read). The samples
    (q, V, D) may run past L.
    """
    if model == 'gcm':
        raise ValueError("model = 'gcm': the generator-coordinate model is solved from its own "
                         'kernels (`gcm_levels`), not from path samples')
    if model not in MODELS:
        raise ValueError(f"model = {model!r}: one of {tuple(m for m in MODELS if m != 'gcm')}")
    if boundary not in BOUNDARIES:
        raise ValueError(f'boundary = {boundary!r}: one of {BOUNDARIES}')
    if mass_profile is not None and model != '1d':
        raise ValueError(f'model = {model!r} with a mass_profile: a coordinate-dependent mass is '
                         "quantized in the 1D model ('1d') only")
    Vs, Ds = _splines(q, V, D)
    if mass_profile is not None:
        qm, Mm = (np.asarray(x, dtype=float) for x in mass_profile)
        Ms = PchipInterpolator(np.r_[-qm[:0:-1], qm], np.r_[Mm[:0:-1], Mm])
        return _levels_1d_mass(Ms, Vs, Ds, L, boundary == 'periodic')
    if model == '1d':
        return _levels_1d(mass, Vs, Ds, L, boundary == 'periodic')
    return _levels_3d(mass, Vs, Ds, L)


def collective_modes(dE, cm, mass, direction, conjugate):
    """(Omega, X, Y) of the collective levels as ladder poles: X + Y = sqrt2 c_m d and
    X - Y = sqrt2 M dE_m c_m e, columns in the house's Casida layout; levels whose c_m^2 is below
    SOFT_MODE_WEIGHT_CUT are dropped."""
    keep = cm ** 2 > SOFT_MODE_WEIGHT_CUT
    dE, cm = dE[keep], cm[keep]
    xpy = np.sqrt(2.0) * np.outer(direction, cm)
    xmy = np.sqrt(2.0) * mass * np.outer(conjugate, dE * cm)
    return dE, (xpy + xmy) / 2, (xpy - xmy) / 2


class _GCMKernels:
    """The kernels of the generator-coordinate model (`gcm_levels`) between determinants of the
    straight RHF->UHF path along a soft direction d (those of `path_samples`), in closed form.

    With d_ia = sum_k sigma_k U_ik V_ak (k = 1 .. r, r = min(nocc, nvirt)) and the pair orbitals
    u_k = sum_i U_ik phi_i, v_k = sum_a V_ak phi_a, Phi(Q) turns pair k by theta_k = sigma_k Q / 2
    (alpha_k = cos theta_k u_k + sin theta_k v_k, beta_k = cos theta_k u_k - sin theta_k v_k), so
    that two determinants of the path are already in corresponding orbitals, pair k overlapping as
    t_k = cos(theta_k - theta'_k) in both spins, and around the RHF (its Fock matrix f; exact for
    the two-body Hamiltonian, and free of divisions, so that t_k = 0 needs no special case)

        N(Q, Q')             = prod_k t_k^2
        H(Q, Q') - E_RHF N   = sum_k [f(M_k) t_k + S_k + J(Ma_k, Mb_k)] P_k
                               + sum_{k<l} [J(M_k, M_l) - sum_s K(Ms_k, Ms_l)] t_k t_l P_kl
        <Phi(Q)|D_alpha|Phi(Q')> = sum_k sigma_k sin(theta_k + theta'_k) t_k P_k
        <Phi(Q)|S^2|Phi(Q')>     = sum_k sin(2 theta_k) sin(2 theta'_k) P_k

    P_k = prod_{j != k} t_j^2 and P_kl = prod_{j != k, l} t_j^2. Ms_k = x_s y_s^T - t_k e_u e_u^T is
    the unnormalized transition density of pair k in spin s less its RHF part, in the block
    (u_k, v_k) (x the bra orbital, y the ket's; with c, s = cos, sin theta_k and c', s' those of
    theta'_k, Ma_k = [[-s s', c s'], [s c', s s']], Mb_k the same with its off-diagonal signs
    flipped, M_k = Ma_k + Mb_k), S_k = 1/2 sum_s (J - K)(Ms_k, Ms_k), J(a, b) = sum (pq|rs)
    a_pq b_rs and K(a, b) = sum (ps|rq) a_pq b_rs over the 2 r pair orbitals. D_alpha = sum_ia d_ia
    (a+_a a_i + a+_i a_a)_alpha, whose diagonal is `path_samples`' D(Q); S^2 on the M_s = 0
    determinants is N_beta - tr(gamma_alpha gamma_beta). The occupied orbitals the pairs leave out
    (nocc > nvirt) stay in f.
    """

    def __init__(self, mf, direction, nocc):
        C = mf.mo_coeff
        U, sigma, Vt = np.linalg.svd(np.reshape(direction, (nocc, -1)), full_matrices=False)
        r = sigma.size
        act = np.hstack([C[:, :nocc] @ U, C[:, nocc:] @ Vt.T])
        fock = act.T @ (mf.get_hcore() + mf.get_veff(mf.mol, mf.make_rdm1())) @ act
        eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, act), 2 * r).reshape((2, r) * 4)
        self.sigma = sigma
        self.fock = np.ascontiguousarray(np.einsum('akbk->kab', fock.reshape(2, r, 2, r)))
        self.coulomb = np.ascontiguousarray(np.einsum('akbkcldl->klabcd', eri))
        self.exchange = np.ascontiguousarray(np.einsum('akblcldk->klabcd', eri))
        self.own = np.ascontiguousarray(np.einsum('kkabcd->kabcd', self.coulomb))

    def elements(self, bra, ket):
        """(N, H - E_RHF N, <D_alpha>, <S^2>) between the determinants whose pairs have turned by
        the angles bra and ket (arrays of shape (P, r), theta_k = sigma_k Q / 2 on the path)."""
        cb, sb, ck, sk = np.cos(bra), np.sin(bra), np.cos(ket), np.sin(ket)
        t = np.cos(bra - ket)
        P, r = t.shape
        alpha = np.empty((P, r, 2, 2))
        alpha[..., 0, 0] = -sb * sk
        alpha[..., 0, 1] = cb * sk
        alpha[..., 1, 0] = sb * ck
        alpha[..., 1, 1] = sb * sk
        beta = alpha * np.array([[1.0, -1.0], [-1.0, 1.0]])
        both = alpha + beta
        one = np.einsum('kab,pkab->pk', self.fock, both)
        own = np.einsum('kabcd,pkab,pkcd->pk', self.own, alpha, beta)
        cross = np.einsum('pklcd,plcd->pkl', np.einsum('klabcd,pkab->pklcd', self.coulomb, both),
                          both)
        for m in (alpha, beta):
            own += 0.5 * (np.einsum('kabcd,pkab,pkcd->pk', self.own, m, m)
                          - np.einsum('kabcd,pkad,pkcb->pk', self.own, m, m))
            cross -= np.einsum('pklbc,plcb->pkl', np.einsum('klabcd,pkad->pklbc', self.exchange, m),
                               m)
        t2 = t * t
        before = np.ones((P, r + 1))
        before[:, 1:] = np.cumprod(t2, axis=1)
        after = np.ones((P, r + 1))
        after[:, :-1] = np.cumprod(t2[:, ::-1], axis=1)[:, ::-1]
        rest = before[:, :-1] * after[:, 1:]
        H = np.sum((one * t + own) * rest, axis=1)
        for k in range(r - 1):
            between = np.ones((P, r - k - 1))
            between[:, 1:] = np.cumprod(t2[:, k + 1:r - 1], axis=1)
            H += np.sum(cross[:, k, k + 1:] * t[:, k, None] * t[:, k + 1:]
                        * (before[:, k, None] * between * after[:, k + 2:]), axis=1)
        D = np.sum(self.sigma * np.sin(bra + ket) * t * rest, axis=1)
        S2 = np.sum(np.sin(2.0 * bra) * np.sin(2.0 * ket) * rest, axis=1)
        return before[:, -1], H, D, S2

    def matrices(self, q):
        """(N, H - E_RHF N, D_alpha, S^2) on the path points q, as len(q) x len(q) matrices."""
        theta = 0.5 * np.outer(q, self.sigma)
        i, j = np.triu_indices(q.size)
        out = []
        for x in self.elements(theta[i], theta[j]):
            m = np.zeros((q.size, q.size))
            m[i, j] = x
            m[j, i] = x
            out.append(m)
        return out

    def potential(self, q):
        """V(Q) = E_HF - E_RHF (Ha) of the path's determinants at q, the kernels' diagonal."""
        theta = 0.5 * np.outer(q, self.sigma)
        return self.elements(theta, theta)[1]


def _reflection_blocks(n):
    """(even, odd) orthonormal bases of Q -> -Q on n (even) points symmetric about 0, point j
    mirrored by n - 1 - j."""
    even, odd = np.zeros((n, n // 2)), np.zeros((n, n // 2))
    r = np.sqrt(0.5)
    for k in range(n // 2):
        even[k, k] = even[n - 1 - k, k] = r
        odd[k, k], odd[n - 1 - k, k] = r, -r
    return even, odd


def gcm_levels(mf, direction):
    """The generator-coordinate (Hill-Wheeler) model of one soft mode on its straight path: a dict
    with

    excitations    dE_m = E_odd,m - E_even,0 (Ha) of every odd state kept, ascending
    couplings      c_m = |f_odd,m^T D f_even,0|, the matrix elements of D_alpha they couple with
    ranks          (even, odd): the states the canonical orthogonalization keeps
    spin_square    <S^2> of the even ground state and of the lowest odd state
    cut_scan       {cut: (dE_0, (even, odd) ranks)} at the cuts of SOFT_MODE_GCM_CUT_SCAN
    q, V           V(Q) (Ha) of the straight path on SOFT_MODE_PATH_POINTS points of [0, L], from
                   the kernels' diagonal

    |Psi> = sum_j f_j |Phi(Q_j)> over the SOFT_MODE_GCM_POINTS midpoints Q_j of [-L, L]
    (L = pi / sigma_1), (H - E N) f = 0 with the kernels of `_GCMKernels`, solved in the even and
    the odd block of Q -> -Q (Phi(-Q) is Phi(Q) with its spins flipped, so the even block holds the
    even spins and the odd block the odd ones, M_s = 0) by canonical orthogonalization: the
    overlap's eigenvalues above SOFT_MODE_GCM_OVERLAP_CUT times its largest are kept, block by
    block, and f^T N f = 1.
    """
    nocc = check_reference(mf)
    L = path_period(direction, nocc)[0]
    kernels = _GCMKernels(mf, direction, nocc)
    n = SOFT_MODE_GCM_POINTS
    N, H, D, S2 = kernels.matrices(-L + (np.arange(n) + 0.5) * 2.0 * L / n)
    blocks = []
    for basis in _reflection_blocks(n):
        w, U = eigh_sym(basis.T @ N @ basis)
        Hb = basis.T @ H @ basis
        solved = []
        for cut in (SOFT_MODE_GCM_OVERLAP_CUT,) + tuple(SOFT_MODE_GCM_CUT_SCAN):
            keep = w > cut * w[-1]
            X = U[:, keep] / np.sqrt(w[keep])
            E, c = eigh_sym(X.T @ Hb @ X)
            solved.append((E, basis @ X @ c, int(np.count_nonzero(keep))))
        blocks.append(solved)
    (E0, F0, r0), (E1, F1, r1) = blocks[0][0], blocks[1][0]
    q = np.linspace(0.0, L, SOFT_MODE_PATH_POINTS)
    return dict(excitations=E1 - E0[0], couplings=np.abs(F1.T @ D @ F0[:, 0]), ranks=(r0, r1),
                spin_square=(float(F0[:, 0] @ S2 @ F0[:, 0]), float(F1[:, 0] @ S2 @ F1[:, 0])),
                cut_scan={cut: (float(o[0][0] - e[0][0]), (e[2], o[2])) for cut, e, o in
                          zip(SOFT_MODE_GCM_CUT_SCAN, blocks[0][1:], blocks[1][1:])},
                q=q, V=kernels.potential(q))


def _degenerate_groups(soft, omega2):
    """The soft indices grouped into sets of equal Omega^2."""
    groups = []
    for s in soft:
        for g in groups:
            if abs(omega2[s] - omega2[g[0]]) < SOFT_MODE_DEGENERACY_TOL:
                g.append(s)
                break
        else:
            groups.append([s])
    return groups


def _gcm_mode(mf, s, mode, L, sigma, path):
    """(record, poles) of soft mode s in the generator-coordinate model (`gcm_levels`), with its
    flags: the record as `soft_mode_triplets` lists it, the poles as `collective_modes` builds them
    (with the Thouless-Valatin mass and e of the split)."""
    gcm = gcm_levels(mf, mode['direction'])
    ranks, V = gcm['ranks'], gcm['V']
    info = dict(index=s, omega2=mode['omega2'], mass=mode['mass'], stiffness=mode['stiffness'],
                half_period=L, sigma=sigma, path=path, v_min=float(V.min()),
                q_min=float(gcm['q'][np.argmin(V)]), excitation=float(gcm['excitations'][0]),
                coupling=float(gcm['couplings'][0]), ranks=ranks, spin_square=gcm['spin_square'],
                cut_scan=gcm['cut_scan'])
    if sigma.size > 1 and sigma[1] >= SOFT_MODE_GCM_PAIR_RATIO * sigma[0]:
        flag('soft_mode_gcm_degenerate_pairs',
             f'soft triplet mode {s}: the two leading pairs of its direction are degenerate '
             f'(sigma = {sigma[0]:.6f}, {sigma[1]:.6f}), so that the straight path turns them '
             'together; there the generator-coordinate model is neither size-consistent nor '
             f'spin-pure (<S^2> = {gcm["spin_square"][0]:.3f} in its even ground state)',
             SoftModeGCMShort)
    if ranks[0] < 2 or ranks[1] < 1 or ranks[0] < ranks[1]:
        flag('soft_mode_gcm_few_states',
             f'soft triplet mode {s}: the canonical orthogonalization keeps {ranks[0]} even and '
             f'{ranks[1]} odd states, short of one pair\'s exact span (two even, one odd) or with '
             'fewer even states than odd ones', SoftModeGCMShort)
    return info, collective_modes(gcm['excitations'], gcm['couplings'], mode['mass'],
                                  mode['direction'], mode['conjugate'])


def soft_mode_triplets(mf, model='1d', path='straight'):
    """The TDHF triplet spectrum of an RHF with its soft modes replaced by their collective levels:
    a dict with

    omega, X, Y        the spectrum (Ha) and its Casida vectors, regular modes first, ready
                       for the triplet sums of the PSD self-energies (`GW.soft_mode_qp`)
    omega2             every TDHF triplet Omega_n^2 (Ha^2), ascending
    omega_tda          the TDA triplet energies (Ha), ascending
    omega_tda_soft     the TDA energy of the root that overlaps the lowest soft direction most
    soft               per soft mode: index, omega2, mass, stiffness, half_period (pi / sigma_1),
                       sigma (singular values of d_ia), path, v_min and q_min (the well depth,
                       Ha, and where it lies, on [0, the end of the periodic domain]),
                       excitation (its first collective level, Ha), excitation_walls (1D, the
                       path cut by walls), edge_weight; with path='relaxed' also antipode (Q*),
                       mass_ratio_antipode (M(Q*) / M), coupling_antipode (D(Q*)),
                       closure_overlap, converged (every constrained minimisation),
                       cold_restarts, restarted, held_modes, curvature, frozen_from and
                       energy_calls (`relaxed_path`); with model='gcm' coupling (c_1), ranks,
                       spin_square and cut_scan (`gcm_levels`) in place of excitation_walls and
                       edge_weight
    numerical_flags    'soft_mode_edge_weight', 'soft_modes_coupled', with path='relaxed'
                       'soft_mode_path_not_converged', 'soft_mode_path_frozen',
                       'soft_mode_path_branch', 'soft_mode_path_closure', and with model='gcm'
                       'soft_mode_gcm_degenerate_pairs', 'soft_mode_gcm_few_states', each raised
                       here from the records of `relaxed_path` and `gcm_levels`

    model is '1d', '3d' (`collective_levels`) or 'gcm' (`gcm_levels`, the straight path only), see
    `MODELS`; path is 'straight' (kappa = Q d, `path_samples`) or 'relaxed' (`relaxed_path`, model
    '1d' only), see `PATHS`. A wrong reference is refused (`check_reference`), so are a wrong model
    or path (`check_path`; the relaxed path with model 'gcm' with NotImplementedError); an A - B
    that is not positive definite is refused with `LinearResponse.casida.CasidaInstability`.
    """
    if model not in MODELS:
        raise ValueError(f'model = {model!r}: one of {MODELS}')
    check_path(path, model)
    nocc = check_reference(mf)
    A, B = triplet_tdhf_matrices(mf)
    split = soft_split(A, B)
    w_tda, u_tda = eigh_sym(A)
    flags = []
    soft_info = []
    parts = [split['regular']]
    with collecting(flags):
        for s in split['soft']:
            mode = split['modes'][s]
            d = mode['direction']
            L, sigma = path_period(d, nocc)
            if model == 'gcm':
                info, poles = _gcm_mode(mf, s, mode, L, sigma, path)
                soft_info.append(info)
                parts.append(poles)
                continue
            if path == 'relaxed':
                held = [split['modes'][t]['conjugate'] for t in split['soft'] if t != s]
                rel = relaxed_path(mf, A, B, mode, held)
                q, V, D, end = rel['q'], rel['V'], rel['D'], rel['antipode']
                profile = (rel['q_mass'], rel['mass'])
            else:
                q, V, D = path_samples(mf, d, L)
                end, profile = L, None
            dE, cm, edge = collective_levels(mode['mass'], q, V, D, end, model,
                                             mass_profile=profile)
            inside = q <= end
            info = dict(index=s, omega2=mode['omega2'], mass=mode['mass'],
                        stiffness=mode['stiffness'], half_period=L, sigma=sigma, path=path,
                        v_min=float(V[inside].min()), q_min=float(q[inside][np.argmin(V[inside])]),
                        excitation=float(dE[0]), edge_weight=edge)
            moved = 0.0
            if model == '1d':
                dE_walls = collective_levels(mode['mass'], q, V, D, end, model, 'walls',
                                             mass_profile=profile)[0]
                info['excitation_walls'] = float(dE_walls[0])
                moved = abs(dE_walls[0] - dE[0])
            if path == 'relaxed':
                info.update(antipode=rel['antipode'],
                            mass_ratio_antipode=rel['mass_antipode'] / mode['mass'],
                            coupling_antipode=rel['coupling_antipode'],
                            closure_overlap=rel['closure_overlap'], converged=rel['converged'],
                            cold_restarts=rel['cold_restarts'], restarted=rel['restarted'],
                            held_modes=rel['held_modes'], curvature=rel['curvature'],
                            frozen_from=rel['frozen_from'], energy_calls=rel['energy_calls'])
                if not rel['converged']:
                    flag('soft_mode_path_not_converged',
                         f'soft triplet mode {s}: a constrained minimisation of its relaxed path '
                         'ended above its gradient bar, or the minima for the inertia at the '
                         'antipode left the branch (the spline tangent is kept there); the last '
                         'iterates are kept', SoftModePathShort)
                if rel['frozen_from'] is not None:
                    flag('soft_mode_path_frozen',
                         f'soft triplet mode {s}: its relaxed path is frozen (continued along d) '
                         f'from Q / L = {rel["frozen_from"] / L:.3f}, before the antipode: the '
                         'constrained minimum left its branch there and a fresh start from '
                         'kappa = Q d did not bring it back, or the antipode lay beyond '
                         'SOFT_MODE_RELAXED_REACH half-periods', SoftModePathShort)
                curvature = rel['curvature']
                saddle = curvature is not None and curvature < -SOFT_MODE_RELAXED_SADDLE_BAR
                if rel['restarted'].size or saddle:
                    flag('soft_mode_path_branch',
                         f'soft triplet mode {s}: its relaxed path is not a constrained minimum '
                         f'throughout: {rel["restarted"].size} samples (Q / L = '
                         f'{np.round(rel["restarted"] / L, 3).tolist()}) were taken from a fresh '
                         'start at kappa = Q d after the warm start left the branch, and the '
                         f'lowest curvature probed across the constraint plane is {curvature:+.3e} '
                         '(in units of the RHF curvature) with the TDHF triplet modes '
                         f'{rel["held_modes"]} held besides the soft ones; the samples are kept',
                         SoftModePathShort)
                if (rel['closure_overlap'] < SOFT_MODE_RELAXED_CLOSURE_MIN
                        or abs(rel['coupling_antipode']) > SOFT_MODE_RELAXED_COUPLING_MAX):
                    flag('soft_mode_path_closure',
                         f'soft triplet mode {s}: its relaxed loop does not close at the antipode '
                         f'Q* = {rel["antipode"] / L:.4f} L: the alpha determinants at -+Q* '
                         f'overlap by {rel["closure_overlap"]:.4f} and D(Q*) = '
                         f'{rel["coupling_antipode"]:+.1e}; the periodic levels are kept',
                         SoftModePathShort)
            if edge > SOFT_MODE_EDGE_WEIGHT_MAX or moved > SOFT_MODE_BOUNDARY_SHIFT_MAX:
                flag('soft_mode_edge_weight',
                     f'soft triplet mode {s}: the collective ground state has {edge:.1e} of its '
                     f'weight near the ends of its path, and its first excitation moves by '
                     f'{moved:.1e} Ha when the periodic path is cut by walls; its levels depend '
                     'on how the path closes', SoftModeEdgeWeight)
            soft_info.append(info)
            parts.append(collective_modes(dE, cm, mode['mass'], d, mode['conjugate']))
        groups = _degenerate_groups(split['soft'], split['omega2'])
        if len(groups) > 1:
            flag('soft_modes_coupled',
                 f'{len(groups)} sets of soft triplet modes ({groups}): each has its own collective '
                 'model, and the couplings between them are left out', SoftModesCoupled)
    d0 = split['modes'][split['soft'][0]]['direction']
    return dict(omega=np.concatenate([p[0] for p in parts]),
                X=np.hstack([p[1] for p in parts]), Y=np.hstack([p[2] for p in parts]),
                omega2=split['omega2'], omega_tda=w_tda,
                omega_tda_soft=float(w_tda[np.argmax(np.abs(u_tda.T @ d0))]),
                soft=soft_info, numerical_flags=gathered(flags))
