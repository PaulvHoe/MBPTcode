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

Model choices on purpose, not checked against alternatives: the straight path in d (no
relaxation of the other rotations), the mass M held at its Q = 0 value, D(Q) as the transition
density that couples back, and the flat 3D model (a ball of radius pi / sigma_1 rather than the
curved manifold of rotated determinants), and the
closure of a path whose minor pairs have not come back.

`soft_mode_triplets` is the driver; `GW.soft_mode_qp` puts its spectrum into the house's PSD
self-energies. Energies are in Hartree.
"""
import numpy as np
from pyscf import scf
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh, eigh_tridiagonal, expm

from src.Base.numerical_flags import collecting, flag, gathered
from src.Base.numerics.constants import (SOFT_MODE_BOUNDARY_SHIFT_MAX, SOFT_MODE_DEGENERACY_TOL,
                                         SOFT_MODE_EDGE_FRACTION, SOFT_MODE_EDGE_WEIGHT_MAX,
                                         SOFT_MODE_GRID_1D, SOFT_MODE_GRID_3D, SOFT_MODE_LEVELS,
                                         SOFT_MODE_OMEGA_CUT, SOFT_MODE_PATH_POINTS,
                                         SOFT_MODE_WEIGHT_CUT)
from src.Base.orbital_spaces import ov_energies
from src.Base.utils.linear_algebra.matrix_functions import eigh_sym, invsqrtm_sym, sqrtm_sym
from src.SingleReference.common.eri_blocks import MOEriBlocks
from src.SingleReference.LinearResponse.casida import CasidaInstability

#: The collective models `collective_levels` solves.
MODELS = ('1d', '3d')
#: The boundaries of the 1D model.
BOUNDARIES = ('periodic', 'walls')


class SoftModeEdgeWeight(RuntimeWarning):
    """The collective ground state reaches the edge of its domain, or its first excitation
    depends on how the path is closed there: the levels are kept."""


class SoftModesCoupled(RuntimeWarning):
    """More than one set of soft modes: each gets its own collective model along its own path, and
    the couplings between them are left out. The levels are kept."""


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


def collective_levels(mass, q, V, D, L, model='1d', boundary='periodic'):
    """(dE, <m|D|0>, edge weight) of the collective Hamiltonian P^2 / 2M + V(Q) of one soft mode,
    from the path samples (q, V, D) of `path_samples` on [0, L]: its excitations that couple to
    the ground state (Ha), their transition matrix elements of D, and the ground state's weight on
    the outer SOFT_MODE_EDGE_FRACTION of the domain.

    model '1d' (the collinear M_s = 0 component, boundary 'periodic' on [-L, L) or 'walls' at
    +-L) or '3d' (three isotropic triplet components, radial equation with nodes at r = 0 and
    r = L; boundary is not read).
    """
    if model not in MODELS:
        raise ValueError(f'model = {model!r}: one of {MODELS}')
    if boundary not in BOUNDARIES:
        raise ValueError(f'boundary = {boundary!r}: one of {BOUNDARIES}')
    Vs, Ds = _splines(q, V, D)
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


def soft_mode_triplets(mf, model='1d'):
    """The TDHF triplet spectrum of an RHF with its soft modes replaced by their collective levels:
    a dict with

    omega, X, Y        the spectrum (Ha) and its Casida vectors, regular modes first, ready
                       for the triplet sums of the PSD self-energies (`GW.soft_mode_qp`)
    omega2             every TDHF triplet Omega_n^2 (Ha^2), ascending
    omega_tda          the TDA triplet energies (Ha), ascending
    omega_tda_soft     the TDA energy of the root that overlaps the lowest soft direction most
    soft               per soft mode: index, omega2, mass, stiffness, half_period (pi / sigma_1),
                       sigma (singular values of d_ia), v_min, q_min, excitation (its first
                       collective level, Ha), excitation_walls (1D, the path cut by walls),
                       edge_weight
    numerical_flags    'soft_mode_edge_weight', 'soft_modes_coupled'

    A wrong reference is refused (`check_reference`); an A - B that is not positive definite is
    refused with `LinearResponse.casida.CasidaInstability`.
    """
    if model not in MODELS:
        raise ValueError(f'model = {model!r}: one of {MODELS}')
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
            q, V, D = path_samples(mf, d, L)
            dE, cm, edge = collective_levels(mode['mass'], q, V, D, L, model)
            info = dict(index=s, omega2=mode['omega2'], mass=mode['mass'],
                        stiffness=mode['stiffness'], half_period=L, sigma=sigma,
                        v_min=float(V.min()), q_min=float(q[np.argmin(V)]),
                        excitation=float(dE[0]), edge_weight=edge)
            moved = 0.0
            if model == '1d':
                dE_walls = collective_levels(mode['mass'], q, V, D, L, model, 'walls')[0]
                info['excitation_walls'] = float(dE_walls[0])
                moved = abs(dE_walls[0] - dE[0])
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
