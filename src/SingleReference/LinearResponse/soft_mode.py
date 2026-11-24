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

The fixed plane can also turn tangent to the path before the antipode, where Q itself turns back:
when d spreads over two pairs of comparable weight (sigma_2 / sigma_1 about 0.5 or more, the
polyenes), the relaxation turns the minor pair back while the leading one goes to 90 degrees, M(Q)
grows without bound, the constrained curvature along the minor pair's partner mode falls to zero
and the minimum leaves its branch for good: a fold of the curve of constrained stationary points
(grad E = lam e plus the holds) over Q, not a saddle, and not the holds (their multipliers are
1e-8 along butadiene's STO-3G paths, 1e-6 at most along hexatriene's). A path frozen that way,
with no negative curvature at its probes, is continued in its arc length x in the TV metric,
dx^2 = dkappa^T (A - B)^-1 dkappa / M (`_continued_samples`): the same curve, from its last
well-sampled fixed-plane point on, each further sample on the hyperplane normal to it, through the
turning points of Q to the antipode, where the loop closes and D(x) changes sign. It is quantized
in x, along which its inertia is M (up to the sampling), the Laplace-Beltrami quantization of the
path, which no ordering of P and M(Q) changes; the TDHF harmonic limit is the fixed planes', whose
samples near Q = 0 are kept. The curvature across the continued path is probed on the slices
(fixed plane cut by the normal hyperplane) and held as above. A soft mode of d's own symmetry
whose e_t has weight on d's leading pair forbids that pair's closed-shell antipode, where
e_t^T kappa is not zero, and the loop then does not close (flagged).

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

reference='excited' (`excited_reference`, `krein_split`; the straight path, 1D, 3D or GCM) admits
a closed shell that is a saddle of the Hartree-Fock energy, an excited (non-aufbau) reference such
as H2's sigma_u^2, where A - B is indefinite and (A - B)^1/2 does not exist. The modes are then
the eigenpairs of the non-symmetric (A - B)(A + B) d = Omega^2 d, d ~ (X + Y), with
M = d^T (A - B)^-1 d of either sign (its Krein signature: sign(X'X - Y'Y) = sign(M Omega)) and
mu = d^T (A + B) d = M Omega^2: M > 0 is the case above (a stable excitation for mu > 0, the
ground-state double well for mu < 0), M < 0 a stable negative transition (mu < 0) or the
excited-state instability (mu > 0, Burth, Kitsaras and Loos, arXiv:2610.36154). H with M < 0 is
unbounded below, so a soft mode of negative mass is quantized as

    -H = P^2 / 2|M| - V(Q),   levels e_m:   the reference is H's top level -e_0, and its
    transitions dE_m = -(e_m - e_0) < 0 go back through `collective_modes` with the signed M,

which for a harmonic V is again the removed TDHF eigenpair, its positive-norm member at
Omega < 0; in the GCM the reference is the even level of largest overlap with the reference
determinant above the even ground state (`gcm_levels`, level='top'; for one pair the top one), its
overlap recorded and flagged when another even level overlaps it more. A soft mode of positive
mass on such a block keeps the bottom level in 1D, and in the GCM, whose span holds configurations
below an excited reference, the even ground state unless another even state overlaps the
reference determinant more by over SOFT_MODE_GCM_OVERLAP_MARGIN (level='overlap'), its
transitions those above the kept state first. Two even states that the rule picks between and
that overlap the determinant within that margin of each other are flagged (a near tie). The
inversion of a mode with M < 0 and mu > 0 is validated on excited references (H2's sigma_u^2);
on a ground-state determinant that mode is a complex instability, which it does not describe
(flagged by `soft_mode_open_shell` on references aufbau in both spins; a closed shell's occupation
does not tell: sigma_u^2 is aufbau in its own Fock spectrum from 1.2 A on). A complex Omega^2 (two
modes of opposite signature collided into a quartet) is refused, and a soft mode close to such a
collision is flagged. The relaxed path is refused: from a saddle a constrained minimisation falls
toward the ground state.

collisions='gcm' (`krein_pairs`, `collision_levels`, `collision_modes`; reference='excited' only)
quantizes a Krein collision instead: the two modes of a quartet (a complex Omega^2 and its
conjugate), or a soft mode and its Krein neighbour, span a real plane invariant under
(A - B)(A + B) on which (A - B)^-1 is indefinite, one mass of each sign, so that a flat two-mode
P^T (A - B) P / 2 + V(Q) is bounded neither below nor above. The generator states are the
determinants Phi(kappa) of every rotation kappa in the plane instead (alpha by exp(K), beta by
exp(-K), or exp(K) in the singlet channel): Hill-Wheeler on their generic non-orthogonal kernels
(`_CollisionKernels`), in the even and odd blocks of kappa -> -kappa, whose levels are real by
construction. The reference continues as the even state of largest overlap with the reference
determinant, and every odd state m goes back as a pole at dE_m = E_odd,m - E_ref with
X + Y = sqrt2 sum_k <m|D_k|ref> e_k and X - Y = dE_m (A - B)^-1 (X + Y). For one occupied orbital
the plane spans {u^2, v_1^2, v_2^2, v_1 v_2} and {u v_1, u v_2}: the odd states are the plane's TDA
roots, measured from a reference correlated with the plane's double excitations, the
configuration a collision is resonant with (Omega_m + Omega_l = 0 in TDHF).

The singlet block (`soft_mode_singlets`, `singlet_tdhf_matrices`: A = De + 2 (ia|jb) - (ij|ab),
B = 2 (ia|jb) - (ib|ja), the same A - B as the triplet's) is split the same way along the RHF->RHF
path, both spins turned by exp(K) (`path_samples` and `_GCMKernels` with channel='singlet'),
whose V is even where a symmetry takes d to -d (checked: a direction whose |V(-Q) - V(Q)| exceeds
SOFT_MODE_SINGLET_PARITY_MAX of max |V| is refused, the low totally symmetric ones measuring 0.04
to 0.8, and a nearly even one below the bar is quantized as even): H2 sigma_u^2's instability
toward H+H- is in this block.

`soft_mode_open_shell` (opt-in) splits one TDHF block of a UHF reference the same sign-aware way
(`open_shell_reference`, `uhf_tdhf_matrices`): the spin-conserving block, whose paths turn alpha
and beta spin orbitals within their spin (UHF determinants), or the spin-flip block, whose paths
mix them (GHF determinants), kappa = Q d / sqrt2 over the block's spin-orbital pairs
(`open_shell_path`), quantized in 1D or by the GCM (`open_shell_gcm_levels`), Krein collisions by
the two-mode GCM (`open_shell_collision_levels`), all on the generic kernels between spin-orbital
determinants (`_OpenShellKernels`). The spin-flip block of a reference with n_alpha > n_beta holds
a global spin rotation g (`spin_rotation_direction`), a zero mode of A + B and A - B there, taken
out exactly (`_spin_rotation_split`) and kept as its pole at 0. `noci_levels` mixes given UHF
solutions by non-orthogonal CI: the symmetry restoration of a hole localized on either atom
(Burth, Kitsaras and Loos's core holes), which the soft-mode expansion about the symmetric
solution does not reach.

`soft_mode_triplets` is the driver; `GW.soft_mode_qp` puts its spectrum into the house's PSD
self-energies (aufbau references only). Energies are in Hartree.
"""
import numpy as np
from pyscf import ao2mo, scf
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.linalg import (cho_factor, cho_solve, cholesky, eig, eigh, eigh_tridiagonal, expm,
                          expm_frechet, schur, solve_triangular)
from scipy.optimize import brentq

from src.Base.numerical_flags import collecting, flag, gathered
from src.Base.numerics.constants import (SOFT_MODE_BOUNDARY_SHIFT_MAX,
                                         SOFT_MODE_COLLISION_DIRECTIONS,
                                         SOFT_MODE_COLLISION_OVERLAP_MIN, SOFT_MODE_DEGENERACY_TOL,
                                         SOFT_MODE_EDGE_FRACTION, SOFT_MODE_EDGE_WEIGHT_MAX,
                                         SOFT_MODE_EXCITED_GRADIENT_MAX,
                                         SOFT_MODE_GCM_CUT_SCAN, SOFT_MODE_GCM_OVERLAP_CUT,
                                         SOFT_MODE_GCM_OVERLAP_MARGIN,
                                         SOFT_MODE_GCM_PAIR_RATIO, SOFT_MODE_GCM_POINTS,
                                         SOFT_MODE_GRID_1D, SOFT_MODE_GRID_3D,
                                         SOFT_MODE_KREIN_GAP, SOFT_MODE_KREIN_IMAG_TOL,
                                         SOFT_MODE_KREIN_NORM_MIN, SOFT_MODE_KREIN_RESIDUAL_MAX,
                                         SOFT_MODE_LEVELS,
                                         SOFT_MODE_OMEGA_CUT, SOFT_MODE_OPEN_SHELL_PARITY_MAX,
                                         SOFT_MODE_PATH_POINTS,
                                         SOFT_MODE_RELAXED_ARC_GRADIENT,
                                         SOFT_MODE_RELAXED_ARC_SECANT,
                                         SOFT_MODE_RELAXED_ARC_SLOPE_STEP,
                                         SOFT_MODE_RELAXED_ARC_START, SOFT_MODE_RELAXED_ARMIJO,
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
                                         SOFT_MODE_SINGLET_PARITY_MAX, SOFT_MODE_WEIGHT_CUT)
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
#: The references the split admits: an aufbau closed shell with A - B positive definite (the
#: default), or any stationary closed shell, an excited (non-aufbau) one included, split
#: sign-aware (`excited_reference`, `krein_split`; straight path only).
REFERENCES = ('aufbau', 'excited')
#: The spin channels of the closed-shell TDHF problem: the triplet block (`soft_mode_triplets`,
#: RHF->UHF paths) and the singlet block (`soft_mode_singlets`, RHF->RHF paths).
CHANNELS = ('triplet', 'singlet')
#: The even level of the generator-coordinate model that continues the reference (`gcm_levels`):
#: the bottom one for a mode of positive mass, the top one for a mode of negative mass, and for a
#: mode of positive mass on a block whose A - B is indefinite (an excited reference) the one picked
#: by its overlap with the reference determinant ('overlap', `_reference_level`).
LEVELS = ('bottom', 'top', 'overlap')
#: What the sign-aware split does with a Krein collision (reference='excited'): 'refuse' a complex
#: Omega^2 and flag a soft mode near one (the default), or quantize the two colliding modes
#: together, 'gcm', by the two-mode generator-coordinate model (`krein_pairs`,
#: `collision_levels`).
COLLISIONS = ('refuse', 'gcm')
#: The TDHF blocks of a UHF reference that `soft_mode_open_shell` splits: the spin-conserving block
#: (alpha -> alpha and beta -> beta rotations, collinear UHF->UHF paths) and the spin-flip block
#: (alpha -> beta and beta -> alpha rotations, non-collinear UHF->GHF paths).
BLOCKS = ('spin_conserving', 'spin_flip')


class SoftModeEdgeWeight(RuntimeWarning):
    """The collective ground state reaches the edge of its domain, or its first excitation
    depends on how the path is closed there: the levels are kept."""


class SoftModesCoupled(RuntimeWarning):
    """More than one set of soft modes: each gets its own collective model along its own path, and
    the couplings between them are left out. The levels are kept."""


class SoftModePathShort(RuntimeWarning):
    """The relaxed collective path stopped short: a constrained minimisation ended above its
    gradient bar (its last iterate is kept), or the path left its branch before the antipode and
    the relaxation is frozen from there (the rotation continued along d; its continuation in arc
    length, tried only where the probes find no negative curvature, was not tried or did not reach
    the antipode), or a sample was taken from a fresh start or is a constrained saddle once the
    hold rounds are spent, or the loop does not close at its antipode. The levels are kept."""


class SoftModeGCMShort(RuntimeWarning):
    """The generator-coordinate model of a soft mode falls short: the two leading pairs of its
    direction are degenerate, so that the straight path turns them together (where the model is
    neither size-consistent nor spin-pure), or the canonical orthogonalization keeps fewer states
    than one pair's exact span (two even and one odd for a closed shell's pair, one of each for an
    open shell's spin-orbital pair). The levels are kept."""


class SoftModeExcitedShort(RuntimeWarning):
    """The sign-aware split of an excited reference falls short: a soft mode is not isolated from
    the modes of the other Krein signature (its Omega^2 within SOFT_MODE_KREIN_GAP of one of
    them, where a Krein collision mixes the two), or the generator-coordinate level kept as the
    reference's continuation (`_reference_level`: for a negative mass the even level of largest
    reference overlap above the even ground state, for a positive mass the bottom one, or on a
    block whose A - B is indefinite the one picked by overlap) is not the even state of largest
    overlap with the reference determinant, or the two even states its rule picks between overlap
    it within SOFT_MODE_GCM_OVERLAP_MARGIN of each other (a near tie); or, in
    `soft_mode_open_shell`, a soft mode of negative mass and positive stiffness (Omega^2 < 0) on a
    reference aufbau in both spins: a complex instability of a ground-state determinant, which the
    inversion of a negative mass is not validated on. The levels are kept."""


class SoftModeCollisionShort(RuntimeWarning):
    """The two-mode generator-coordinate model of a Krein collision (collisions='gcm') falls short:
    the even state kept as the reference's continuation, the one of largest overlap with the
    reference determinant, holds less than SOFT_MODE_COLLISION_OVERLAP_MIN of it (the reference
    shares its determinant with a doubly excited configuration of the plane), or the canonical
    orthogonalization keeps fewer states than the plane spans for one occupied orbital (four even,
    two odd). The levels are kept."""


class SoftModeNOCIShort(RuntimeWarning):
    """The non-orthogonal CI between given UHF solutions (`noci_levels`) keeps fewer states than
    solutions: the canonical orthogonalization drops overlap eigenvalues at or below
    SOFT_MODE_GCM_OVERLAP_CUT times the largest (two of the solutions are, to that cut, one
    determinant). The states kept are returned."""


def _check_mean_field(mf):
    """Refuse a mean field that is not a converged closed-shell RHF with exact integrals."""
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


def check_reference(mf):
    """Refuse a mean field the soft-mode split is not defined for: it needs a converged
    closed-shell RHF with exact integrals and an aufbau occupation."""
    _check_mean_field(mf)
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


def check_reference_option(reference, path):
    """Refuse a reference option that is not one of `REFERENCES`, and the relaxed path with an
    excited reference (NotImplementedError): a reference that is a saddle of the Hartree-Fock
    energy would need a constrained saddle search along the path, where a constrained minimisation
    falls toward the ground state."""
    if reference not in REFERENCES:
        raise ValueError(f'reference = {reference!r}: one of {REFERENCES}')
    if reference == 'excited' and path == 'relaxed':
        raise NotImplementedError("reference = 'excited' with path = 'relaxed': the relaxed path "
                                  'minimises the energy across the constraint plane, which from a '
                                  'saddle-point reference falls toward the ground state; a '
                                  'constrained saddle search is not implemented')


def check_collisions_option(collisions, reference):
    """Refuse a collisions option that is not one of `COLLISIONS`, and collisions='gcm' on an
    aufbau reference (reference='aufbau', whose A - B is positive definite: Omega^2 is real and
    every mode of one signature, so no two modes can collide)."""
    if collisions not in COLLISIONS:
        raise ValueError(f'collisions = {collisions!r}: one of {COLLISIONS}')
    if collisions == 'gcm' and reference != 'excited':
        raise ValueError(f"collisions = 'gcm' with reference = {reference!r}: a Krein collision "
                         "needs modes of both signatures, which only reference = 'excited' "
                         'admits (with an aufbau reference A - B is positive definite)')


def excited_reference(mf):
    """(ref, order) for reference='excited': ref a shallow copy of a converged closed-shell RHF
    whose occupation need not be aufbau (an excited reference, such as H2's sigma_u^2 from a
    maximum-overlap or a symmetry-constrained SCF), with its orbitals, energies and occupations
    reordered so that the doubly occupied ones come first (each block in mf's order), as every
    function of this module reads them; order holds mf's orbital indices in that order, so that
    the pair index i * nvirt + a of the split runs over ref's orbitals i < nocc <= a.

    Refused before any work: what `check_reference` refuses for the type, the functional, density
    fitting or convergence; an occupation other than 0 and 2, or one that leaves no
    occupied-virtual pair; and an occupation that is not stationary, the occupied-virtual block of
    the Fock matrix of mf's own density reaching SOFT_MODE_EXCITED_GRADIENT_MAX (Ha): an
    occupation changed after the SCF, which the split's expansion about a stationary point does
    not describe."""
    _check_mean_field(mf)
    occ = np.asarray(mf.mo_occ)
    if not np.all((occ == 0) | (occ == 2)):
        raise ValueError('mf.mo_occ: the soft-mode split needs a closed-shell occupation (0 or 2 '
                         'electrons in each orbital)')
    occupied, virtual = np.flatnonzero(occ == 2), np.flatnonzero(occ == 0)
    if not occupied.size or not virtual.size:
        raise ValueError('mf.mo_occ: no occupied-virtual pair to split')
    C = np.asarray(mf.mo_coeff)
    fock = mf.get_hcore() + mf.get_veff(mf.mol, mf.make_rdm1(C, occ))
    gradient = float(np.max(np.abs(C[:, occupied].T @ fock @ C[:, virtual])))
    if gradient >= SOFT_MODE_EXCITED_GRADIENT_MAX:
        raise ValueError(f'mf.mo_occ is not stationary: the occupied-virtual block of the Fock '
                         f'matrix of its density reaches {gradient:.1e} Ha (bar '
                         f'{SOFT_MODE_EXCITED_GRADIENT_MAX:.0e}); converge the SCF with this '
                         'occupation (maximum overlap, or symmetry-constrained)')
    order = np.r_[occupied, virtual]
    ref = mf.copy()
    ref.mo_coeff, ref.mo_energy = C[:, order], np.asarray(mf.mo_energy)[order]
    ref.mo_occ = occ[order]
    return ref, order


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


def singlet_tdhf_matrices(mf):
    """(A, B) of the closed-shell TDHF singlet problem, A = De + 2 (ia|jb) - (ij|ab) and
    B = 2 (ia|jb) - (ib|ja) (Hartree; pair index i * nvirt + a), from the MO blocks of
    `triplet_tdhf_matrices`: the triplet's A and B plus 2 (ia|jb) each, so that A - B is the same
    in both channels. The same matrices as `LinearResponse.linear_response.LinearResponseSolver`'s
    build_casida_matrices(nocc, lBSE=True) on the full integrals.
    """
    nocc = check_reference(mf)
    blk = MOEriBlocks.from_mol(mf.mol, mf.mo_coeff, nocc)
    de = ov_energies(mf.mo_energy, nocc)
    n = de.size
    coulomb = 2.0 * blk.ovov.reshape(n, n)
    A = np.diag(de) - blk.oovv.transpose(0, 2, 1, 3).reshape(n, n) + coulomb
    B = -blk.ovov.transpose(0, 3, 2, 1).reshape(n, n) + coulomb
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


def _real_modes(product, vec, omega2, amb_inv, scale):
    """(d, clusters): the modes of product = (A - B)(A + B) as real unit vectors (`krein_split`,
    `krein_pairs`), from its eigenvectors vec (columns) of the real eigenvalues omega2 (ascending).

    A lone eigenvalue's eigenvector is made real by its largest component. Each set of eigenvalues
    within SOFT_MODE_DEGENERACY_TOL of the next (clusters: its (start, stop) column range) gets
    an orthonormal basis of the real span of its eigenvectors' real and imaginary parts, as many
    vectors as eigenvalues (LAPACK can return a degenerate real eigenvalue as a conjugate pair
    whose imaginary parts are rounding, the two vectors then sharing one real part), turned so
    that the metric (A - B)^-1 is diagonal on it. Every mode d_n (|d_n| = 1, its largest component
    positive) must solve product d_n = omega2_n d_n to SOFT_MODE_KREIN_RESIDUAL_MAX times scale
    (the largest |Omega^2|), else refused with `LinearResponse.casida.CasidaInstability`: the
    eigenvectors of a set do not span it (a defective eigenvalue, an exceptional point)."""
    n = omega2.size
    d = np.real(vec / vec[np.argmax(np.abs(vec), axis=0), np.arange(n)])
    clusters, start = [], 0
    for stop in range(1, n + 1):
        if stop == n or omega2[stop] - omega2[stop - 1] >= SOFT_MODE_DEGENERACY_TOL:
            if stop - start > 1:
                parts = vec[:, start:stop]
                basis = np.linalg.svd(np.hstack([parts.real, parts.imag]),
                                      full_matrices=False)[0][:, :stop - start]
                d[:, start:stop] = basis @ eigh_sym(basis.T @ amb_inv @ basis)[1]
                clusters.append((start, stop))
            start = stop
    d /= np.linalg.norm(d, axis=0)
    d *= np.sign(d[np.argmax(np.abs(d), axis=0), np.arange(n)])
    if n:
        residual = np.linalg.norm(product @ d - d * omega2, axis=0)
        worst = int(np.argmax(residual))
        if residual[worst] > SOFT_MODE_KREIN_RESIDUAL_MAX * scale:
            raise CasidaInstability(f'(A - B)(A + B): its mode at Omega^2 = {omega2[worst]:+.6e} '
                                    f'Ha^2 misses its eigen-equation by {residual[worst]:.2e} Ha^2 '
                                    f'(bar {SOFT_MODE_KREIN_RESIDUAL_MAX:.0e} of the largest '
                                    f'|Omega^2|, {scale:.3e} Ha^2): the eigenvectors of its '
                                    'degenerate set do not span it, an exceptional point')
    return d, clusters


def krein_split(A, B):
    """The TDHF spectrum of a block whose A - B need not be positive definite (a reference that is
    a saddle of the Hartree-Fock energy, an excited one), split sign-aware: the dict of
    `soft_split`, with

    signature  sign(M_n) of every mode (int, +-1), the sign of its Krein signature
    modes      for each soft index also signature, with a mass M = d^T (A - B)^-1 d of either sign

    The modes are the eigenpairs of the non-symmetric (A - B)(A + B) d = Omega^2 d (d ~ (X + Y)),
    each made real and normalized |d| = 1 as in `soft_split` (the degenerate sets first given a
    real basis of their span, orthogonal in the indefinite metric (A - B)^-1, `_real_modes`), and
    mu = d^T (A + B) d = M Omega^2: a mode is a stable excitation (M > 0, mu > 0), the ground-state
    double well of `soft_split` (M > 0, mu < 0), a stable negative transition (M < 0, mu < 0; its
    positive-norm root has Omega < 0) or the excited-state instability (M < 0, mu > 0). The
    regular modes go back as their positive-norm member, Omega = sign(M) |Omega|,
    X + Y = d / (M Omega)^1/2 and X - Y = (M Omega)^1/2 (A - B)^-1 d / M (X'X - Y'Y = 1), so that a
    negative transition has Omega < 0; the soft set is `soft_split`'s, on Omega^2.

    A positive definite A - B is `soft_split`'s own case: its dict, to the bit, with every
    signature +1. Refused with `LinearResponse.casida.CasidaInstability`: an A - B with a zero
    eigenvalue (no inverse, no mass); a complex Omega^2 (an imaginary part above
    SOFT_MODE_KREIN_IMAG_TOL times the largest |Omega^2|: two modes of opposite signature have
    collided into a quartet, which no single collective coordinate describes); a mode of null
    norm (|M| below SOFT_MODE_KREIN_NORM_MIN times the largest |(A - B)^-1| eigenvalue: an
    exceptional point, where such a collision begins); and a mode that misses its eigen-equation
    by more than SOFT_MODE_KREIN_RESIDUAL_MAX of the largest |Omega^2| (`_real_modes`).
    """
    w_amb, v_amb = eigh_sym(A - B)
    if w_amb[0] > 0:
        split = soft_split(A, B)
        for mode in split['modes'].values():
            mode['signature'] = 1
        split['signature'] = np.ones(split['omega2'].size, dtype=int)
        return split
    if np.any(w_amb == 0.0):
        raise CasidaInstability('A - B has a zero eigenvalue: the soft mode has no '
                                'Thouless-Valatin mass')
    amb_inv = (v_amb / w_amb) @ v_amb.T
    product = (A - B) @ (A + B)
    lam, vec = eig(product)
    worst = float(np.max(np.abs(lam.imag)))
    scale = float(np.max(np.abs(lam)))
    if worst > SOFT_MODE_KREIN_IMAG_TOL * scale:
        raise CasidaInstability(f'(A - B)(A + B) has a complex eigenvalue (imaginary part '
                                f'{worst:.2e} Ha^2): modes of opposite Krein signature have '
                                'collided into a complex quartet, which the sign-aware split does '
                                'not describe')
    order = np.argsort(lam.real)
    omega2 = lam.real[order]
    n = omega2.size
    d = _real_modes(product, vec[:, order], omega2, amb_inv, scale)[0]
    y = amb_inv @ d
    mass = np.einsum('pn,pn->n', d, y)
    null = float(np.min(np.abs(mass)) * np.min(np.abs(w_amb)))
    if null < SOFT_MODE_KREIN_NORM_MIN:
        raise CasidaInstability(f'a TDHF mode has a null norm (|M| = {np.min(np.abs(mass)):.2e} '
                                'Ha^-1 for |d| = 1): an exceptional point of two modes of '
                                'opposite Krein signature')
    stiffness = np.einsum('pn,pn->n', d, (A + B) @ d)
    soft = {0} | set(np.flatnonzero(omega2 < SOFT_MODE_OMEGA_CUT ** 2).tolist())
    for s in list(soft):
        soft |= set(np.flatnonzero(np.abs(omega2 - omega2[s]) < SOFT_MODE_DEGENERACY_TOL).tolist())
    soft = sorted(soft)
    signature = np.where(mass > 0, 1, -1)
    modes = {s: dict(direction=d[:, s], conjugate=y[:, s] / mass[s], mass=float(mass[s]),
                     stiffness=float(stiffness[s]), omega2=float(omega2[s]),
                     signature=int(signature[s])) for s in soft}
    reg = np.setdiff1d(np.arange(n), soft)
    omega = signature[reg] * np.sqrt(omega2[reg])
    weight = np.sqrt(mass[reg] * omega)
    xpy = d[:, reg] / weight
    xmy = y[:, reg] * (weight / mass[reg])
    return dict(omega2=omega2, soft=soft, modes=modes, signature=signature,
                regular=(omega, (xpy + xmy) / 2, (xpy - xmy) / 2))


def krein_pairs(A, B):
    """`krein_split` with its Krein collisions taken out in pairs for the two-mode model
    (collisions='gcm', `collision_levels`): the dict of `krein_split`, with

    omega2     every Omega^2 (Ha^2), complex, ascending in its real part (a quartet's Omega^2 and
               its conjugate side by side)
    signature  sign(M_n) of every mode of real Omega^2 (+-1, the two modes of a 'neighbours'
               pair included), 0 for the two modes of a quartet (complex Omega^2, no definite
               signature)
    pairs      per colliding pair: index (its two positions in omega2), omega2 (theirs), kind
               ('quartet', a complex-conjugate Omega^2: two modes of opposite signature have
               collided; 'neighbours', a soft mode and the mode of the other signature nearest
               to it, each the other's nearest, within SOFT_MODE_KREIN_GAP of its Omega^2
               relative to the larger, about to collide), gap (that relative gap; for a quartet
               |Im Omega^2| / |Omega^2|), plane
               (n x 2, an orthonormal basis e_1, e_2 of the pair's real invariant plane of
               (A - B)(A + B), turned so that the metric (A - B)^-1 is diagonal on it), masses
               (e_m^T (A - B)^-1 e_m, the positive one first: one of each sign) and conjugates
               ((A - B)^-1 e_m, n x 2)

    and soft, modes and regular as in `krein_split` for the modes outside the pairs: the soft set
    is `krein_split`'s on the real parts of Omega^2 less the paired modes (so it can be empty).
    The plane is spanned by the first two Schur vectors of (A - B)(A + B) with the pair's
    eigenvalues ordered first, which stay well conditioned where the two eigenvectors coalesce
    (at the edges of a collision window). Neighbours pair only as mutual nearest neighbours: a soft
    mode pairs with its nearest mode of the other signature (over every mode of real Omega^2) when
    that mode's nearest is the soft mode in turn, so that the pairs do not depend on the order the
    modes are taken in. A soft mode in a degenerate set, or whose nearest mode of the other
    signature is in one, is left unpaired, also when it is another soft mode's nearest (a
    collision with a degenerate set, which two modes do not describe), and so is a soft mode whose
    nearest is another mode's mutual nearest neighbour (a near-collision of three modes, which two
    modes do not describe either); the drivers flag each as a Krein neighbour (its krein_gap counts
    the modes of a 'neighbours' pair, not a quartet's).

    Without a collision it is `krein_split`'s dict to the bit, with pairs = [] (always so for a
    positive definite A - B). Refused as there (CasidaInstability): an A - B with a zero
    eigenvalue, a mode of real Omega^2 that misses its eigen-equation (`_real_modes`), and a mode
    of null norm outside the pairs; refused here (NotImplementedError): a
    quartet whose Omega^2 lies within SOFT_MODE_DEGENERACY_TOL of another eigenvalue (a collision
    of degenerate sets, which two modes do not describe).
    """
    w_amb, v_amb = eigh_sym(A - B)
    if w_amb[0] > 0:
        split = krein_split(A, B)
        split['pairs'] = []
        return split
    if np.any(w_amb == 0.0):
        raise CasidaInstability('A - B has a zero eigenvalue: the soft mode has no '
                                'Thouless-Valatin mass')
    amb_inv = (v_amb / w_amb) @ v_amb.T
    product = (A - B) @ (A + B)
    lam, vec = eig(product)
    order = np.lexsort((lam.imag, lam.real))
    lam, vec = lam[order], vec[:, order]
    n = lam.size
    scale = float(np.max(np.abs(lam)))
    is_complex = np.abs(lam.imag) > SOFT_MODE_KREIN_IMAG_TOL * scale
    found = []
    lower = np.flatnonzero(is_complex & (lam.imag < 0))
    for k in np.flatnonzero(is_complex & (lam.imag > 0)):
        j = int(lower[np.argmin(np.abs(lam[lower] - np.conj(lam[k])))]) if lower.size else -1
        if j < 0 or abs(lam[j] - np.conj(lam[k])) > SOFT_MODE_KREIN_IMAG_TOL * scale:
            raise CasidaInstability(f'(A - B)(A + B): the complex eigenvalue {lam[k]:.6f} '
                                    'does not come with its conjugate')
        found.append(('quartet', sorted([j, int(k)]), float(abs(lam[k].imag) / abs(lam[k]))))
    for _, idx, _ in found:
        others = np.delete(lam, idx)
        if others.size and np.min(np.abs(others[:, None] - lam[idx][None, :])) < \
                SOFT_MODE_DEGENERACY_TOL:
            raise NotImplementedError(f'a Krein quartet (Omega^2 = {lam[idx[1]]:.6f}) within '
                                      f'{SOFT_MODE_DEGENERACY_TOL:.0e} Ha^2 of another '
                                      'eigenvalue: a collision of degenerate sets, which the '
                                      'two-mode model does not describe')
    if 2 * len(found) != int(np.count_nonzero(is_complex)):
        raise CasidaInstability('(A - B)(A + B): its complex eigenvalues do not pair into '
                                'conjugates')
    found.sort(key=lambda item: item[1][0])
    real = np.flatnonzero(~is_complex)
    column = {j: c for c, j in enumerate(real)}
    d, clusters = _real_modes(product, vec[:, real], lam[real].real, amb_inv, scale)
    degenerate = {int(j) for start, stop in clusters for j in real[start:stop]}
    y = amb_inv @ d
    mass = np.einsum('pn,pn->n', d, y)
    stiffness = np.einsum('pn,pn->n', d, (A + B) @ d)
    signature = np.zeros(n, dtype=int)
    signature[real] = np.where(mass > 0, 1, -1)
    soft = {0} | set(np.flatnonzero(lam.real < SOFT_MODE_OMEGA_CUT ** 2).tolist())
    for s in list(soft):
        soft |= set(np.flatnonzero(np.abs(lam - lam[s]) < SOFT_MODE_DEGENERACY_TOL).tolist())
    paired = {j for _, idx, _ in found for j in idx}

    def nearest(s):
        """(the real mode of the other signature nearest to s, the relative gap), or (None,
        None)."""
        near = [j for j in real.tolist() if signature[j] == -signature[s]]
        if not near:
            return None, None
        gaps = np.abs(lam[near].real - lam[s].real) / np.maximum(np.abs(lam[near].real),
                                                                 abs(lam[s].real))
        best = int(np.argmin(gaps))
        return near[best], float(gaps[best])
    for s in sorted(soft):
        if s in paired or s in degenerate or s not in column:
            continue
        j, gap = nearest(s)
        if j is None or gap >= SOFT_MODE_KREIN_GAP or j in degenerate or nearest(j)[0] != s:
            continue
        found.append(('neighbours', sorted([s, j]), gap))
        paired |= {s, j}
    if not found:
        split = krein_split(A, B)
        split['pairs'] = []
        return split
    pairs = []
    for kind, idx, gap in found:
        target = lam[idx]
        others = np.delete(lam, idx)
        sep = float(np.min(np.abs(others[:, None] - target[None, :]))) if others.size else np.inf
        _, Z, sdim = schur(product, output='real',
                           sort=lambda x, z: bool(np.min(np.abs(complex(x, z) - target))
                                                  < 0.5 * sep))
        if sdim != 2:
            raise CasidaInstability(f'(A - B)(A + B): the Schur form orders {sdim} eigenvalues '
                                    f'with the colliding pair {idx}, not two')
        masses, turn = eigh_sym(Z[:, :2].T @ amb_inv @ Z[:, :2])
        plane = Z[:, :2] @ turn[:, ::-1]
        plane *= np.sign(plane[np.argmax(np.abs(plane), axis=0), np.arange(2)])
        pairs.append(dict(index=idx, omega2=target, kind=kind, gap=gap, plane=plane,
                          masses=masses[::-1], conjugates=amb_inv @ plane))
    free = [column[j] for j in real.tolist() if j not in paired]
    if free and float(np.min(np.abs(mass[free])) * np.min(np.abs(w_amb))) < \
            SOFT_MODE_KREIN_NORM_MIN:
        raise CasidaInstability(f'a TDHF mode has a null norm (|M| = '
                                f'{np.min(np.abs(mass[free])):.2e} Ha^-1 for |d| = 1): an '
                                'exceptional point of two modes of opposite Krein signature')
    soft = sorted(soft - paired)
    modes = {s: dict(direction=d[:, column[s]], conjugate=y[:, column[s]] / mass[column[s]],
                     mass=float(mass[column[s]]), stiffness=float(stiffness[column[s]]),
                     omega2=float(lam[s].real), signature=int(signature[s])) for s in soft}
    reg = [j for j in real.tolist() if j not in paired and j not in soft]
    cols = [column[j] for j in reg]
    omega = signature[reg] * np.sqrt(lam[reg].real)
    weight = np.sqrt(mass[cols] * omega)
    xpy = d[:, cols] / weight
    xmy = y[:, cols] * (weight / mass[cols])
    return dict(omega2=lam, soft=soft, modes=modes, signature=signature,
                regular=(omega, (xpy + xmy) / 2, (xpy - xmy) / 2), pairs=pairs)


def path_period(direction, nocc):
    """(pi / sigma_1, sigma): the half-period of the RHF->UHF path along the soft direction, where
    the leading pair of d_ia (an nocc x nvirt matrix) has turned by 90 degrees, and the singular
    values sigma of d_ia (sigma_1 = 1 when d has rank one)."""
    sigma = np.linalg.svd(direction.reshape(nocc, -1), compute_uv=False)
    return np.pi / sigma[0], sigma


def path_samples(mf, direction, q_end, channel='triplet'):
    """(Q, V, D) on SOFT_MODE_PATH_POINTS points of [0, q_end]: V(Q) = E_HF - E_RHF (Ha) of the
    RHF determinant with its alpha orbitals rotated by exp(K) and its beta orbitals by exp(-K),
    K_ai = -K_ia = Q d_ia / 2, and D(Q) = 2 sum_ia d_ia P^alpha_ai(Q), the alpha density's
    occupied-virtual block along d. V is even and D odd in Q. channel='singlet' (`CHANNELS`) turns
    the beta orbitals by exp(K) too, the RHF->RHF path of a singlet direction, whose V is even only
    where a symmetry of the reference takes d to -d (`soft_mode_singlets` checks it)."""
    if channel not in CHANNELS:
        raise ValueError(f'channel = {channel!r}: one of {CHANNELS}')
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
        ca, cb = (C @ U)[:, :nocc], (C @ (U.T if channel == 'triplet' else U))[:, :nocc]
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

    @property
    def dimension(self):
        """The number of independent whitened variables: size (an `_ArcSlice` has one fewer)."""
        return self.size

    def project(self, w):
        """w itself: the plane's whitened variables have no null direction (`_ArcSlice`'s have)."""
        return w


class _ArcSlice:
    """The fixed plane of a `_ConstrainedMinima` at e^T kappa = Q cut by the hyperplane
    n^T (kappa - p) = 0 through a predictor p (n the path's normal in the TV metric), for
    `_continued_samples`: kappa = Q d + c(Q) a + Z Pi u(w), with n~ = Z^T n, a = Z n~ / |n~|^2 (on
    the plane, n^T a = 1), c(Q) = n^T (p - Q d), u(w) the plane's L^-T w and Pi the orthogonal
    projector off n~. The plane's whitened variables w keep their size; the one direction they gain
    without effect, L^T n~, is projected out of w and of the gradient (`dimension` = size - 1).
    `multiplier` is nu of grad E = lam e + nu n + the holds at a slice minimum, which leaves
    |nu| |L^-1 n~| (`scale`) of the fixed plane's whitened gradient: 0 on the path."""

    def __init__(self, minima, normal, point):
        self.minima, self.normal, self.point = minima, normal, point
        self.Z, self.size, self.dimension = minima.Z, minima.size, minima.size - 1
        self.nt = minima.Z.T @ normal
        self.nt2 = float(self.nt @ self.nt)
        self.a = minima.Z @ self.nt / self.nt2
        null = minima.chol.T @ self.nt
        self.null = null / np.linalg.norm(null)
        self.scale = float(np.linalg.norm(solve_triangular(minima.chol, self.nt, lower=True)))

    def project(self, w):
        """w (a vector, or the columns of a matrix) less its component along L^T n~."""
        return w - np.multiply.outer(self.null, self.null @ w)

    def u(self, w):
        """Pi L^-T w (columns of a matrix too): the rotation in the plane's basis Z."""
        u = self.minima.u(self.project(w))
        return u - np.multiply.outer(self.nt, self.nt @ u) / self.nt2

    def base(self, Q):
        """The point of the slice at Q with w = 0."""
        d = self.minima.direction
        return Q * d + (self.normal @ (self.point - Q * d)) * self.a

    def kappa(self, Q, w):
        return self.base(Q) + self.Z @ self.u(w)

    def whitened(self, Q, kappa):
        """w of a rotation kappa, projected on the slice at Q."""
        return self.project(self.minima.chol.T @ (self.Z.T @ (kappa - self.base(Q))))

    def value_and_gradient(self, Q, w):
        """(energy, its gradient in w) at Q and w."""
        E, g = self.minima.energy(self.kappa(Q, w))
        g = self.Z.T @ g
        g = g - self.nt * (self.nt @ g) / self.nt2
        return E, self.project(solve_triangular(self.minima.chol, g, lower=True))

    def minimum(self, Q, w, bar):
        """(w, V, converged) at Q from the start w."""
        return _capped_lbfgs(lambda x: self.value_and_gradient(Q, x), self.project(w),
                             lambda p: float(np.linalg.norm(self.u(p))), bar)

    def multiplier(self, kappa):
        """nu at a slice minimum kappa: Z^T grad E = nu n~ there."""
        return float(self.nt @ (self.Z.T @ self.minima.energy(kappa)[1])) / self.nt2


def _curvature_probe(minima, Q, w):
    """(theta, v): Ritz values (ascending) and vectors of the constrained Hessian at the relaxed
    sample (Q, w) in the whitened variables of `_ConstrainedMinima` (where the RHF's Hessian is 1)
    or of an `_ArcSlice` (its start projected), over a Krylov space of
    SOFT_MODE_RELAXED_PROBE_VECTORS vectors (Lanczos with full reorthogonalization from a start
    drawn with SOFT_MODE_RELAXED_PROBE_SEED), each Hessian-vector product a central difference of
    the whitened gradient over SOFT_MODE_RELAXED_PROBE_STEP. The lowest Ritz value lies above the
    lowest eigenvalue, so a negative one is a direction across the constraint plane along which the
    sample is a saddle."""
    n = min(SOFT_MODE_RELAXED_PROBE_VECTORS, minima.dimension)
    basis, image = np.zeros((minima.size, n)), np.zeros((minima.size, n))
    v = minima.project(np.random.default_rng(SOFT_MODE_RELAXED_PROBE_SEED)
                       .standard_normal(minima.size))
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


def _continued_samples(minima, path, conjugate, nocc, h, factor, mass):
    """The relaxed path continued in its arc length past a turning point of Q, from the samples of
    `_relaxed_samples` (path, frozen before its antipode) on the fixed planes of minima: None when
    fewer than three of them can be kept, else a dict with

    ok          whether the path reached its antipode
    q, V, kappa the samples (None unless ok): q their arc length x in the TV metric,
                x_k - x_(k-1) = |kappa_k - kappa_(k-1)| in the norm (v^T (A - B)^-1 v / M)^1/2, so
                that the inertia kappa'(x)^T (A - B)^-1 kappa'(x) is M to the order of the samples
    relaxed     their count (every sample a constrained stationary point)
    passed      the first sample past the antipode, where D(x) turns from positive to negative
    start       the last fixed-plane sample kept
    slice, probe   the `_ArcSlice` and (Q, w) of the last sample before the antipode, where
                `relaxed_path` probes the curvature across the path (ok only); not ok, the
                predictor of the step that left the branch, which it does not probe (slice None
                when the path went beyond reach instead)

    The fixed-plane samples are kept while every step stays within SOFT_MODE_RELAXED_ARC_START
    arc-length steps h and D stays positive. Each further sample lies on the hyperplane normal (in
    the TV metric) to the unit tangent t of the quadratic through the last three samples, through
    the predictor kappa + h t, at the point of the same curve of constrained stationary points
    (grad E = lam e plus the holds): the minimum of the `_ArcSlice` at e^T kappa = Q to
    SOFT_MODE_RELAXED_ARC_GRADIENT, Q found by a secant search on the hyperplane's multiplier nu
    (steps cut to h) until |nu| |L^-1 n~| is below SOFT_MODE_RELAXED_GRADIENT, within
    SOFT_MODE_RELAXED_ARC_SECANT minimisations. A sample that misses that bar, or whose step fails
    the branch test of `_relaxed_samples`, ends the continuation: before the antipode it is not ok,
    after it the path ends there. The path runs SOFT_MODE_RELAXED_PAST samples past passed and is
    not ok beyond SOFT_MODE_RELAXED_REACH half-periods. mass is the soft mode's M, factor the
    Cholesky factor of A - B, conjugate its e.
    """
    d = minima.direction
    nmo = minima.energy.nmo
    reach = SOFT_MODE_RELAXED_REACH * path_period(d, nocc)[0]

    def length(v):
        return float(np.sqrt(v @ cho_solve(factor, v) / mass))
    kept = path['kappa']
    start = 0
    while (start + 1 < path['relaxed']
           and length(kept[start + 1] - kept[start]) <= SOFT_MODE_RELAXED_ARC_START * h
           and _coupling(kept[start + 1], d, nocc, nmo) > 0):
        start += 1
    if start < 2:
        return None
    kap, V = list(path['kappa'][:start + 1]), list(path['V'][:start + 1])
    x = [0.0]
    for k in range(1, start + 1):
        x.append(x[-1] + length(kap[k] - kap[k - 1]))
    slope, passed, probe = None, None, None
    failed = dict(ok=False, q=None, V=None, kappa=None, relaxed=0, passed=None, start=start)
    while passed is None or len(kap) - 1 < passed + SOFT_MODE_RELAXED_PAST:
        if passed is None and x[-1] > reach:
            return dict(failed, slice=None, probe=None)
        (x0, x1, x2), (k0, k1, k2) = x[-3:], kap[-3:]
        t = (k0 * (x2 - x1) / ((x0 - x1) * (x0 - x2)) + k1 * (x2 - x0) / ((x1 - x0) * (x1 - x2))
             + k2 * (2 * x2 - x0 - x1) / ((x2 - x0) * (x2 - x1)))
        t /= length(t)
        predictor = k2 + h * t
        piece = _ArcSlice(minima, cho_solve(factor, t) / mass, predictor)
        Q = Q0 = float(conjugate @ predictor)
        w = w0 = piece.whitened(Q, predictor)
        accepted, last, kq = False, None, predictor
        for _ in range(SOFT_MODE_RELAXED_ARC_SECANT):
            w, E, ok = piece.minimum(Q, w, SOFT_MODE_RELAXED_ARC_GRADIENT)
            kq = piece.kappa(Q, w)
            nu = piece.multiplier(kq)
            if ok and abs(nu) * piece.scale < SOFT_MODE_RELAXED_GRADIENT:
                accepted = True
                break
            if last is not None and Q != last[0]:
                slope = (nu - last[1]) / (Q - last[0])
            last = Q, nu
            dq = (SOFT_MODE_RELAXED_ARC_SLOPE_STEP * h if slope is None or slope == 0
                  else -nu / slope)
            Q += float(np.clip(dq, -h, h))
            w = piece.whitened(Q, kq)
        bar = (SOFT_MODE_RELAXED_JUMP_RATIO * max(np.linalg.norm(k2 - k1),
                                                  np.linalg.norm(predictor - k2))
               + SOFT_MODE_RELAXED_JUMP_SLACK)
        if not accepted or np.linalg.norm(kq - k2) > bar:
            if passed is None:
                return dict(failed, slice=piece, probe=(Q0, w0))
            break
        kap.append(kq)
        V.append(E)
        x.append(x[-1] + length(kq - k2))
        if passed is None:
            if _coupling(kq, d, nocc, nmo) > 0:
                probe = piece, (Q, w)
            else:
                passed = len(kap) - 1
    if probe is None:
        probe = piece, (Q, w)
    return dict(ok=True, q=np.array(x), V=np.array(V), kappa=np.array(kap), relaxed=len(kap),
                passed=passed, start=start, slice=probe[0], probe=probe[1])


def relaxed_path(mf, A, B, mode, held):
    """The relaxed collective path of one soft mode (the module docstring's path='relaxed'): a dict
    with

    q, V, D            samples Q_k = k L / (SOFT_MODE_RELAXED_POINTS - 1), L = pi / sigma_1, from 0
                       to SOFT_MODE_RELAXED_PAST samples past the first one beyond the antipode (or
                       to where the path left its branch after it): V(Q) = E(kappa(Q)) - E_RHF (Ha)
                       and D(Q); on a path continued past a turning point of Q (continued_from not
                       None) the samples of `_continued_samples` and their arc length x, which
                       stands for Q in every entry below
    kappa              the triplet rotation at each sample (pair index i * nvirt + a)
    relaxed            the number of relaxed samples (0 .. relaxed - 1; any later ones are
                       continued along d)
    antipode           Q*, where the leading pair of kappa_ia has turned by 90 degrees (on a
                       continued path, where D changes sign)
    q_mass, mass       the path inertia M(Q) = kappa'^T (A - B)^-1 kappa' (Ha^-1) on [0, Q*], from
                       the cubic spline of kappa at samples with SOFT_MODE_RELAXED_TRUST relaxed
                       samples beyond them, M at Q* from that spline or (path ending sooner)
                       from fresh minima at Q* -+ SOFT_MODE_RELAXED_TANGENT_STEP h, and M on a
                       frozen part
    mass_antipode, coupling_antipode   M(Q*) (Ha^-1) and D(Q*)
    closure_overlap    |<alpha determinant at -Q*|alpha determinant at Q*>|: 1 for a closed loop
    tangent            where M(Q*) comes from: 'spline', 'minima', 'frozen' or 'arc' (the spline
                       of a continued path)
    held_modes         the TDHF triplet modes outside the soft set (indices into `soft_split`'s
                       omega2) held at zero besides held, for the negative curvatures they carried
    curvature          the lowest Ritz value of `_curvature_probe` over the probed samples (in units
                       of the RHF's curvature; None when no rotation is free): negative, a saddle
    restarted          the Q of the samples accepted from a fresh start at kappa = Q d
    continued_from     the Q of the last fixed-plane sample kept when the path is continued in its
                       arc length (`_continued_samples`), else None
    converged, cold_restarts, frozen_from, energy_calls

    A and B are `triplet_tdhf_matrices`, mode the soft mode's record from `soft_split`, held the
    conjugates e_t of the soft modes kept at zero along the path. The rotation is minimised over
    the full real triplet space. The curvature across the constraint plane is probed at every
    sample accepted from a fresh start and at the last relaxed sample up to the antipode; where it
    is below -SOFT_MODE_RELAXED_SADDLE_BAR, the modes that carry it (`_saddle_modes`) are held at
    zero too and the path is followed again, up to SOFT_MODE_RELAXED_HOLD_ROUNDS times, the last
    path's samples and probes kept (converged, cold_restarts, restarted and curvature are its own;
    energy_calls counts every round). A path frozen before its antipode with no negative curvature
    at those probes (Q turned back) is continued in its arc length (`_continued_samples`); when the
    continuation reaches the antipode, the curvature across it is probed on the `_ArcSlice` of its
    last sample before the antipode, a negative one held as above and the round followed again. A
    continuation that does not reach the antipode leaves the frozen path and its record as they
    were (its step that left the branch is not probed; energy_calls still counts its energies).
    The 1D model of `collective_levels` quantizes the path with mass_profile=(q_mass, mass) on
    [-Q*, Q*).
    """
    nocc = check_reference(mf)
    nmo = mf.mo_coeff.shape[1]
    d, M0 = mode['direction'], mode['mass']
    h = path_period(d, nocc)[0] / (SOFT_MODE_RELAXED_POINTS - 1)
    energy = _TripletEnergy(mf, nocc)
    factor = cho_factor(A - B)
    regular, extra = None, []
    for rounds in range(SOFT_MODE_RELAXED_HOLD_ROUNDS + 1):
        held_now = list(held) + [regular[3][:, j] for j in extra]
        minima = _ConstrainedMinima(energy, d, mode['conjugate'], held_now, A + B)
        path = _relaxed_samples(minima, nocc, h)
        curvature, negative, probed, arc = None, [], minima, None
        if minima.size:
            for k in sorted(set(path['restarted']) | {min(path['passed'], path['relaxed'] - 1)}):
                theta, vectors = _curvature_probe(minima, path['q'][k], path['w'][k])
                curvature = theta[0] if curvature is None else min(curvature, theta[0])
                negative.append(vectors[:, theta < -SOFT_MODE_RELAXED_SADDLE_BAR])
            if path['frozen_from'] is not None and not any(v.shape[1] for v in negative):
                arc = _continued_samples(minima, path, mode['conjugate'], nocc, h, factor, M0)
                if arc is not None and arc['ok']:
                    probed = arc['slice']
                    theta, vectors = _curvature_probe(probed, *arc['probe'])
                    curvature = min(curvature, theta[0])
                    negative.append(vectors[:, theta < -SOFT_MODE_RELAXED_SADDLE_BAR])
        if rounds == SOFT_MODE_RELAXED_HOLD_ROUNDS or not any(v.shape[1] for v in negative):
            break
        if regular is None:
            regular = _regular_modes(A, B)
        new = [j for j in _saddle_modes(probed, np.hstack(negative), regular) if j not in extra]
        if not new:
            break
        extra += new
    continued = arc is not None and arc['ok']
    if continued:
        q, kap, m, frozen, V = arc['q'], arc['kappa'], arc['relaxed'], None, arc['V']
        restarted = [k for k in path['restarted'] if k <= arc['start']]
    else:
        q, kap, m, frozen, V = (path['q'], path['kappa'], path['relaxed'], path['frozen_from'],
                                path['V'])
        restarted = path['restarted']
    spline = (CubicSpline(np.r_[-q[1:m][::-1], q[:m]], np.r_[-kap[1:m][::-1], kap[:m]], axis=0)
              if m > 1 else None)

    def kappa_at(Q):
        return kap[m - 1] + (Q - frozen) * d if frozen is not None and Q >= frozen else spline(Q)
    if continued:
        j = arc['passed']
        q_star = brentq(lambda x: _coupling(spline(x), d, nocc, nmo), q[j - 1], q[j])
    else:
        j = path['passed']
        q_star = brentq(lambda x: _largest_angle(kappa_at(x), nocc) - np.pi, q[j - 1], q[j])

    def inertia(t):
        return float(t @ cho_solve(factor, t))
    trusted = m - 1 - SOFT_MODE_RELAXED_TRUST
    table = [k for k in np.flatnonzero(q < q_star) if k == 0 or k <= trusted or k >= m]
    q_mass = [q[k] for k in table]
    mass = [inertia(spline(q[k], 1)) if k < m and m > 1 else M0 for k in table]
    converged = path['converged']
    if continued:
        m_star, tangent = inertia(spline(q_star, 1)), 'arc'
    elif frozen is not None and q_star > frozen:
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
    return dict(q=q, V=V, D=D, kappa=kap, relaxed=m, antipode=float(q_star),
                q_mass=np.array(q_mass + [q_star]), mass=np.array(mass + [m_star]),
                mass_antipode=m_star, coupling_antipode=_coupling(k_star, d, nocc, nmo),
                closure_overlap=float(overlap), tangent=tangent,
                held_modes=sorted(int(regular[0][j]) for j in extra),
                curvature=None if curvature is None else float(curvature),
                restarted=q[restarted], converged=bool(converged),
                cold_restarts=path['cold_restarts'], frozen_from=frozen,
                continued_from=float(path['q'][arc['start']]) if continued else None,
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

    channel='singlet' (`CHANNELS`): the RHF->RHF path of `path_samples` with channel='singlet',
    whose beta pairs turn as the alpha ones (beta_k = alpha_k): the same kernels with Mb_k = Ma_k,
    and <S^2> = 0 between its closed-shell determinants.
    """

    def __init__(self, mf, direction, nocc, channel='triplet'):
        self.flip = (np.array([[1.0, -1.0], [-1.0, 1.0]]) if channel == 'triplet'
                     else np.ones((2, 2)))
        self.triplet = channel == 'triplet'
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
        beta = alpha * self.flip
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
        S2 = (np.sum(np.sin(2.0 * bra) * np.sin(2.0 * ket) * rest, axis=1) if self.triplet
              else np.zeros(P))
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


def _reference_level(E0, F0, E1, to_reference, level):
    """(k, order): the even Hill-Wheeler state k that continues the reference (`LEVELS`; even
    energies E0 ascending, states F0) and the order of the odd states' transitions E1 - E0[k]
    (`gcm_levels`, `open_shell_gcm_levels`). 'bottom': the even ground state, the odd states
    ascending. 'top' (a mode of negative mass, whose reference continues as the top of its
    collective spectrum): of the even states above the ground state, the one of largest overlap
    with the reference determinant, (F0^T to_reference)^2 (with one pair, two even states, the top
    one; with more, the highest roots come from the smallest overlap eigenvalues kept and hold
    almost none of it); the odd states below it first, nearest first, then those above it,
    nearest first. The even ground state is left out: it holds the determinant on aufbau
    references with a negative-mass mode (0.66 to 0.95: O2's and square H4's triplet UHF, N2+'s
    symmetric hole, square H4's RHF), where the kept state holds little of it and is flagged, but
    keeping it there would need a test that tells a ground-state determinant from an excited one,
    which H2's sigma_u^2 defeats (aufbau in its own Fock spectrum from 1.2 A on, and its top state
    the closer to FCI where the even ground state holds more: -10.75 against -1.87 eV for FCI's
    -15.07 in the 6-31G triplet at 1.015 A, -0.67 against +5.47 eV for FCI's -0.07 in the
    cc-pVTZ singlet at 3.0 A). 'overlap' (a mode of positive mass on a block whose
    A - B is indefinite, an excited reference, whose span holds configurations below it): the
    even ground state unless the even state of largest overlap above it overlaps the reference
    determinant more by over SOFT_MODE_GCM_OVERLAP_MARGIN, then that one; the odd states above it
    first, nearest first, then those below it, nearest first (the mirror of 'top')."""
    if level == 'bottom':
        return 0, np.arange(E1.size)
    overlaps = (F0.T @ to_reference) ** 2
    k = 1 + int(np.argmax(overlaps[1:])) if overlaps.size > 1 else 0
    if level == 'overlap':
        k = k if overlaps[k] - overlaps[0] > SOFT_MODE_GCM_OVERLAP_MARGIN else 0
        dE = E1 - E0[k]
        return k, np.lexsort((np.abs(dE), dE < 0))
    dE = E1 - E0[k]
    return k, np.lexsort((np.abs(dE), dE > 0))


def _level_rivals(overlaps, level):
    """(largest, second): the overlaps with the reference determinant of the two even states that
    the level rule of `_reference_level` picks between by overlap, the states above the even ground
    state for 'top' and every even state for 'overlap'; None for 'bottom' or a single candidate."""
    if level == 'bottom':
        return None
    pool = np.sort(overlaps[1:] if level == 'top' else overlaps)[::-1]
    return (float(pool[0]), float(pool[1])) if pool.size > 1 else None


def _reference_flags(label, level, mass, overlaps, kept):
    """Flag the generator-coordinate level kept as the reference's continuation (`_reference_level`,
    state kept of the even states, overlaps their overlaps with the reference determinant):
    'soft_mode_reference_overlap' when another even state overlaps the determinant more, and
    'soft_mode_reference_tie' when the two states the level rule picks between overlap it within
    SOFT_MODE_GCM_OVERLAP_MARGIN of each other (`_level_rivals`), both `SoftModeExcitedShort`."""
    best = int(np.argmax(overlaps))
    if best != kept:
        flag('soft_mode_reference_overlap',
             f'{label}: the even state kept as the reference\'s continuation (state {kept} of '
             f'{overlaps.size}, ascending in energy, by the level rule {level!r}; its mass M = '
             f'{mass:+.3e} Ha^-1) overlaps the reference determinant by {overlaps[kept]:.3f}, '
             f'state {best} by {overlaps[best]:.3f}; the levels are kept', SoftModeExcitedShort)
    rivals = _level_rivals(overlaps, level)
    if rivals is not None and rivals[0] - rivals[1] <= SOFT_MODE_GCM_OVERLAP_MARGIN:
        flag('soft_mode_reference_tie',
             f'{label}: the two even states that the level rule {level!r} picks between overlap '
             f'the reference determinant by {rivals[0]:.4f} and {rivals[1]:.4f}, within '
             f'SOFT_MODE_GCM_OVERLAP_MARGIN ({SOFT_MODE_GCM_OVERLAP_MARGIN}) of each other, a near '
             'tie of the pick (the reference shares its determinant between them, or neither '
             'holds much of it): the kept level can switch between them with the geometry or the '
             'overlap cut; the levels are kept', SoftModeExcitedShort)


def gcm_levels(mf, direction, channel='triplet', level='bottom'):
    """The generator-coordinate (Hill-Wheeler) model of one soft mode on its straight path: a dict
    with

    excitations    dE_m = E_odd,m - E_even,0 (Ha) of every odd state kept, ascending
    couplings      c_m = |f_odd,m^T D f_even,0|, the matrix elements of D_alpha they couple with
    ranks          (even, odd): the states the canonical orthogonalization keeps
    spin_square    <S^2> of the even ground state and of the lowest odd state
    cut_scan       {cut: (dE_0, (even, odd) ranks)} at the cuts of SOFT_MODE_GCM_CUT_SCAN
    q, V           V(Q) (Ha) of the straight path on SOFT_MODE_PATH_POINTS points of [0, L], from
                   the kernels' diagonal
    reference_overlaps   |<Phi(0)|even m>|^2 of the even states kept, ascending in energy: their
                   overlap with the reference determinant
    reference      the index (into reference_overlaps) of the even state kept as the reference's
                   continuation (level)

    |Psi> = sum_j f_j |Phi(Q_j)> over the SOFT_MODE_GCM_POINTS midpoints Q_j of [-L, L]
    (L = pi / sigma_1), (H - E N) f = 0 with the kernels of `_GCMKernels`, solved in the even and
    the odd block of Q -> -Q (Phi(-Q) is Phi(Q) with its spins flipped, so the even block holds the
    even spins and the odd block the odd ones, M_s = 0) by canonical orthogonalization: the
    overlap's eigenvalues above SOFT_MODE_GCM_OVERLAP_CUT times its largest are kept, block by
    block, and f^T N f = 1.

    channel='singlet' (`CHANNELS`) solves the RHF->RHF path of `_GCMKernels`, whose blocks are
    those of the symmetry that takes d to -d (spin-pure, <S^2> = 0). level (`LEVELS`) is the even
    state that continues the reference: 'bottom', the even ground state (above), or 'top', for a
    mode of negative mass (`krein_split`), whose reference is the top of its collective spectrum:
    of the even states above the ground state, the one of largest overlap with the reference
    determinant (`_reference_level`; for one pair the top one); then
    dE_m = E_odd,m - E_even,reference, the odd states below the reference first, nearest first
    (descending), then any above it, nearest first, and so are the couplings, the <S^2> of the
    first odd state and the cut scan's dE_0 (each cut picking its reference state the same way).
    'overlap', for a mode of positive mass on a block whose A - B is indefinite (an excited
    reference): the even ground state unless another even state overlaps the reference
    determinant more by over SOFT_MODE_GCM_OVERLAP_MARGIN, then that one, and the odd states above
    it first, nearest first (ascending), then any below it, nearest first.
    """
    if channel not in CHANNELS:
        raise ValueError(f'channel = {channel!r}: one of {CHANNELS}')
    if level not in LEVELS:
        raise ValueError(f'level = {level!r}: one of {LEVELS}')
    nocc = check_reference(mf)
    L = path_period(direction, nocc)[0]
    kernels = _GCMKernels(mf, direction, nocc, channel)
    n = SOFT_MODE_GCM_POINTS
    grid = -L + (np.arange(n) + 0.5) * 2.0 * L / n
    N, H, D, S2 = kernels.matrices(grid)
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
    theta = 0.5 * np.outer(grid, kernels.sigma)
    to_reference = kernels.elements(np.zeros_like(theta), theta)[0]
    k, order = _reference_level(E0, F0, E1, to_reference, level)
    j = order[0]
    q = np.linspace(0.0, L, SOFT_MODE_PATH_POINTS)
    scan = {}
    for cut, e, o in zip(SOFT_MODE_GCM_CUT_SCAN, blocks[0][1:], blocks[1][1:]):
        kc, oc = _reference_level(e[0], e[1], o[0], to_reference, level)
        scan[cut] = (float(o[0][oc[0]] - e[0][kc]), (e[2], o[2]))
    return dict(excitations=(E1 - E0[k])[order], couplings=np.abs(F1.T @ D @ F0[:, k])[order],
                ranks=(r0, r1),
                spin_square=(float(F0[:, k] @ S2 @ F0[:, k]), float(F1[:, j] @ S2 @ F1[:, j])),
                cut_scan=scan, q=q, V=kernels.potential(q),
                reference_overlaps=(F0.T @ to_reference) ** 2, reference=k)


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


def _singlet_parity(mf, direction, nocc, L):
    """max |V(-Q) - V(Q)| / max |V(Q)| of the singlet path along direction on SOFT_MODE_PATH_POINTS
    points of [0, L], from the closed-form kernels' diagonal (`_GCMKernels`): 0 where a symmetry
    of the reference takes d to -d."""
    kernels = _GCMKernels(mf, direction, nocc, 'singlet')
    q = np.linspace(0.0, L, SOFT_MODE_PATH_POINTS)
    V = kernels.potential(q)
    scale = float(np.max(np.abs(V)))
    return float(np.max(np.abs(kernels.potential(-q) - V))) / scale if scale > 0 else 0.0


def _sign_aware_gcm(ref, mode, sigma, channel, reference, label, level):
    """(record entries, poles) of a soft mode in the generator-coordinate model along the channel's
    straight path, its reference level picked by the level rule (`_reference_level`: 'bottom' or,
    on a block whose A - B is indefinite, 'overlap' for a positive mass, 'top' for a negative
    one), with the flags of `_gcm_mode` and, for an excited reference, those of
    `_reference_flags` (`SoftModeExcitedShort`)."""
    sign = 1.0 if mode['mass'] > 0 else -1.0
    gcm = gcm_levels(ref, mode['direction'], channel, level)
    ranks, V = gcm['ranks'], sign * gcm['V']
    info = dict(v_min=float(V.min()), q_min=float(gcm['q'][np.argmin(V)]),
                excitation=float(gcm['excitations'][0]), coupling=float(gcm['couplings'][0]),
                ranks=ranks, spin_square=gcm['spin_square'], cut_scan=gcm['cut_scan'],
                reference_overlaps=gcm['reference_overlaps'], reference=gcm['reference'],
                level=level)
    if sigma.size > 1 and sigma[1] >= SOFT_MODE_GCM_PAIR_RATIO * sigma[0]:
        flag('soft_mode_gcm_degenerate_pairs',
             f'{label}: the two leading pairs of its direction are degenerate (sigma = '
             f'{sigma[0]:.6f}, {sigma[1]:.6f}), so that the straight path turns them together; '
             'there the generator-coordinate model is not size-consistent'
             + (f' nor spin-pure (<S^2> = {gcm["spin_square"][0]:.3f} in its reference state)'
                if channel == 'triplet' else ''), SoftModeGCMShort)
    if ranks[0] < 2 or ranks[1] < 1 or ranks[0] < ranks[1]:
        flag('soft_mode_gcm_few_states',
             f'{label}: the canonical orthogonalization keeps {ranks[0]} even and {ranks[1]} odd '
             'states, short of one pair\'s exact span (two even, one odd) or with fewer even '
             'states than odd ones', SoftModeGCMShort)
    if reference == 'excited':
        _reference_flags(label, level, mode['mass'], gcm['reference_overlaps'], gcm['reference'])
    return info, collective_modes(gcm['excitations'], gcm['couplings'], mode['mass'],
                                  mode['direction'], mode['conjugate'])


def _products_but(s):
    """(one, two) of the rows of s (E x p): one[e, k] = prod_{j != k} s[e, j] and
    two[e, k, l] = prod_{j != k, l} s[e, j] (k != l), from prefix and suffix products, free of
    divisions."""
    rows, p = s.shape

    def but_one(t):
        before = np.ones((rows, p + 1))
        before[:, 1:] = np.cumprod(t, axis=1)
        after = np.ones((rows, p + 1))
        after[:, :-1] = np.cumprod(t[:, ::-1], axis=1)[:, ::-1]
        return before[:, :-1] * after[:, 1:]
    two = np.empty((rows, p, p))
    for k in range(p):
        t = s.copy()
        t[:, k] = 1.0
        two[:, k] = but_one(t)
    return but_one(s), two


class _CollisionKernels:
    """The kernels of the two-mode generator-coordinate model (`collision_levels`) between the
    determinants of a collision's plane: Phi(kappa) for every rotation kappa = a e_1 + b e_2 in the
    plane of the orthonormal (e_1, e_2), its alpha orbitals turned by exp(K(kappa)) and its beta
    orbitals by exp(-K(kappa)) (channel 'triplet') or exp(K(kappa)) ('singlet'),
    K_ai = kappa_ia / 2, as along the straight paths of `path_samples`.

    Two determinants of the plane share no pair orbitals (the straight path's `_GCMKernels` close
    on them), so the kernels are the generic non-orthogonal ones, in each spin's corresponding
    orbitals x_k, y_k (the singular-value decomposition of the occupied overlap, x_k . y_l =
    s_k delta_kl) over the 2 n_occ pairs of both spins, free of divisions so that a zero s_k needs
    no special case:

        N           = c prod_k s_k
        H - E_0 N   = c [(E_nuc - E_0) prod_k s_k + sum_k h(P_k) prod_{j != k} s_j
                         + sum_{k<l} (J(P_k, P_l) - delta_{spin} K(P_k, P_l)) prod_{j != k, l} s_j]
        <D_m>       = c sum_{k alpha} D_m(P_k) prod_{j != k} s_j
        <S^2>       = n_occ N - c sum_{k alpha, l beta} (x_k . y_l)(x_l . y_k) prod_{j != k, l} s_j

    with P_k = x_k y_k^T, J and K as in `_GCMKernels`, c the product of the determinants of the
    rotations to corresponding orbitals, D_m = sum_ia (e_m)_ia (a+_a a_i + a+_i a_a)_alpha, S^2 on
    M_s = 0 determinants, and E_0 the reference determinant's energy in the same integrals. They
    are evaluated in the active orbitals: the occupied ones and the virtual span of e_1 and e_2
    read as n_occ x n_virt matrices, which hold every determinant of the plane.
    """

    def __init__(self, mf, plane, nocc, channel):
        self.triplet = channel == 'triplet'
        self.nocc = nocc
        C = mf.mo_coeff
        blocks = [np.reshape(plane[:, m], (nocc, C.shape[1] - nocc)) for m in range(2)]
        W = np.linalg.svd(np.vstack(blocks), full_matrices=False)[2].T
        self.generators = [b @ W for b in blocks]
        act = np.hstack([C[:, :nocc], C[:, nocc:] @ W])
        n = act.shape[1]
        self.size = n
        self.h = act.T @ mf.get_hcore() @ act
        eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, act), n)
        self.coulomb = eri.reshape(n * n, n * n)
        self.exchange = np.ascontiguousarray(eri.transpose(0, 3, 2, 1)).reshape(n * n, n * n)
        self.ops = []
        for g in self.generators:
            D = np.zeros((n, n))
            D[:nocc, nocc:] = g
            D[nocc:, :nocc] = g.T
            self.ops.append(D.reshape(n * n))
        o = eri[:nocc, :nocc, :nocc, :nocc]
        e0 = (mf.energy_nuc() + 2.0 * np.trace(self.h[:nocc, :nocc])
              + 2.0 * np.einsum('iijj->', o) - np.einsum('ijji->', o))
        self.shift = mf.energy_nuc() - e0

    def orbitals(self, coeffs):
        """(alpha, beta): the occupied orbitals (P x size x n_occ, in the active orbitals) of the
        determinants at kappa = a e_1 + b e_2 for the rows (a, b) of coeffs."""
        nocc, n = self.nocc, self.size
        alpha = np.empty((len(coeffs), n, nocc))
        beta = np.empty_like(alpha)
        for p, (a, b) in enumerate(coeffs):
            X = 0.5 * (a * self.generators[0] + b * self.generators[1])
            K = np.zeros((n, n))
            K[nocc:, :nocc] = X.T
            K[:nocc, nocc:] = -X
            U = expm(K)
            alpha[p] = U[:, :nocc]
            beta[p] = (U.T if self.triplet else U)[:, :nocc]
        return alpha, beta

    def elements(self, bra, ket):
        """(N, H - E_0 N, <D_1>, <D_2>, <S^2>) between the determinants bra = (alpha, beta) and
        ket (arrays of `orbitals`, element by element)."""
        nocc, n = self.nocc, self.size
        xs, ys, ss, sign = [], [], [], 1.0
        for X, Y in zip(bra, ket):
            U, s, Vt = np.linalg.svd(np.einsum('epi,epj->eij', X, Y))
            sign = sign * np.linalg.det(U) * np.linalg.det(Vt)
            xs.append(np.einsum('epi,eij->epj', X, U))
            ys.append(np.einsum('epi,eji->epj', Y, Vt))
            ss.append(s)
        x, y, s = np.concatenate(xs, axis=2), np.concatenate(ys, axis=2), np.concatenate(ss, axis=1)
        one, two = _products_but(s)
        total = np.prod(s, axis=1)
        rows, p = s.shape
        rho = np.einsum('epk,eqk->ekpq', x, y).reshape(rows, p, n * n)
        J = np.einsum('ekq,elq->ekl', rho @ self.coulomb, rho)
        K = np.einsum('ekq,elq->ekl', rho @ self.exchange, rho)
        spin = np.arange(p) >= nocc
        G = J - K * np.equal.outer(spin, spin)
        k, l = np.triu_indices(p, 1)
        H = sign * (self.shift * total + np.sum((rho @ self.h.reshape(n * n)) * one, axis=1)
                    + np.sum(G[:, k, l] * two[:, k, l], axis=1))
        D = [sign * np.sum((rho[:, :nocc] @ op) * one[:, :nocc], axis=1) for op in self.ops]
        cross = (np.einsum('epi,epj->eij', x[:, :, :nocc], y[:, :, nocc:])
                 * np.einsum('epj,epi->eij', x[:, :, nocc:], y[:, :, :nocc]))
        N = sign * total
        S2 = nocc * N - sign * np.sum(cross * two[:, :nocc, nocc:], axis=(1, 2))
        return N, H, D[0], D[1], S2

    def matrices(self, alpha, beta):
        """(N, H - E_0 N, D_1, D_2, S^2) between the determinants of `orbitals`, as P x P
        matrices."""
        P = alpha.shape[0]
        out = np.zeros((5, P, P))
        for i in range(P):
            j = np.arange(i, P)
            bra = (np.broadcast_to(alpha[i], alpha[j].shape),
                   np.broadcast_to(beta[i], beta[j].shape))
            values = np.array(self.elements(bra, (alpha[j], beta[j])))
            out[:, i, j] = values
            out[:, j, i] = values
        return out


def collision_levels(mf, plane, channel='triplet'):
    """The two-mode generator-coordinate (Hill-Wheeler) model of a Krein collision: the two modes
    of a pair of `krein_pairs` quantized together on the determinants of their plane, a dict with

    excitations        dE_m = E_odd,m - E_reference (Ha) of every odd state kept, ascending: de-
                       excitations (dE < 0) and excitations (dE > 0) of the reference
    couplings          c_m = (<odd m|D_1|reference>, <odd m|D_2|reference>), the matrix elements
                       of D_m = sum_ia (e_m)_ia (a+_a a_i + a+_i a_a)_alpha (each row's larger
                       component positive)
    even_levels        the even states' energies (Ha, relative to the reference determinant's)
    reference          the index of the even state kept as the reference's continuation: the one
                       of largest overlap with the reference determinant
    reference_overlaps |<Phi(0)|even m>|^2 of the even states, ascending in energy
    ranks              (even, odd): the states the canonical orthogonalization keeps
    spin_square        <S^2> of the reference state and of every odd state
    cut_scan           {cut: (the de-excitation and the excitation of largest |c_m|, or None,
                       (even, odd) ranks)} at the cuts of SOFT_MODE_GCM_CUT_SCAN
    half_periods       pi / sigma_1 along each direction of the grid

    |Psi> = sum_j f_j |Phi(kappa_j)> over the plane spanned by the orthonormal columns e_1, e_2 of
    plane (`krein_pairs`): kappa_j = r (cos phi_k e_1 + sin phi_k e_2) on
    SOFT_MODE_COLLISION_DIRECTIONS directions phi_k = k pi / n, each carrying the straight path's
    SOFT_MODE_GCM_POINTS midpoints r of [-L_k, L_k] (L_k = pi / sigma_1 along it, the 90-degree
    turn of its leading pair). (H - E N) f = 0 with the kernels of `_CollisionKernels`, solved in
    the even and odd blocks of kappa -> -kappa (the spin flip, channel 'triplet'; for 'singlet' the
    symmetry of the reference that takes the plane to minus itself, checked: a potential whose
    |V(-kappa) - V(kappa)| on the grid exceeds SOFT_MODE_SINGLET_PARITY_MAX of max |V| is refused
    with NotImplementedError) by canonical orthogonalization as in `gcm_levels`. The levels are
    real by construction (a Hermitian H on a positive semidefinite overlap). D_m is odd, so it
    couples the even reference to the odd states only; dE_m and c_m go back as ladder poles through
    `collision_modes`.

    For one occupied orbital (H2) the plane's determinants span exactly the even states {u^2, v_1^2,
    v_2^2, v_1 v_2 (singlet)} and the odd ones {u v_1, u v_2} of the channel's spin (u the occupied
    orbital, v_m = sum_a (e_m)_a phi_a), and the model is the CI in that space: the odd states are
    the TDA (CIS) of the plane, the reference u^2 correlated with the plane's double excitations.
    """
    if channel not in CHANNELS:
        raise ValueError(f'channel = {channel!r}: one of {CHANNELS}')
    nocc = check_reference(mf)
    plane = np.asarray(plane, dtype=float)
    kernels = _CollisionKernels(mf, plane, nocc, channel)
    n, m = SOFT_MODE_GCM_POINTS, SOFT_MODE_COLLISION_DIRECTIONS
    phi = np.pi * np.arange(m) / m
    half = np.array([path_period(np.cos(f) * plane[:, 0] + np.sin(f) * plane[:, 1], nocc)[0]
                     for f in phi])
    radial = (np.arange(n) + 0.5) * 2.0 / n - 1.0
    coeffs = np.concatenate([np.outer(L * radial, (np.cos(f), np.sin(f)))
                             for f, L in zip(phi, half)])
    alpha, beta = kernels.orbitals(coeffs)
    if channel == 'singlet':
        V = kernels.elements((alpha, beta), (alpha, beta))[1]
        mirror = np.concatenate([k * n + n - 1 - np.arange(n) for k in range(m)])
        scale = float(np.max(np.abs(V)))
        parity = float(np.max(np.abs(V[mirror] - V))) / scale if scale > 0 else 0.0
        if parity > SOFT_MODE_SINGLET_PARITY_MAX:
            raise NotImplementedError(
                f'the singlet plane of the collision is not even in kappa (|V(-kappa) - V(kappa)| '
                f'up to {parity:.1e} of max |V|, bar {SOFT_MODE_SINGLET_PARITY_MAX:.0e}): no '
                'symmetry of the reference takes it to minus itself, and the parity-blocked model '
                'solves the even part only')
    N, H, D1, D2, S2 = kernels.matrices(alpha, beta)
    start = (np.broadcast_to(np.eye(kernels.size)[:, :nocc], alpha.shape),) * 2
    to_reference = kernels.elements(start, (alpha, beta))[0]
    blocks = []
    for reflection in _reflection_blocks(n):
        basis = np.kron(np.eye(m), reflection)
        w, U = eigh_sym(basis.T @ N @ basis)
        Hb = basis.T @ H @ basis
        solved = []
        for cut in (SOFT_MODE_GCM_OVERLAP_CUT,) + tuple(SOFT_MODE_GCM_CUT_SCAN):
            keep = w > cut * w[-1]
            X = U[:, keep] / np.sqrt(w[keep])
            E, c = eigh_sym(X.T @ Hb @ X)
            solved.append((E, basis @ X @ c, int(np.count_nonzero(keep))))
        blocks.append(solved)

    def poles(even, odd):
        (E0, F0, r0), (E1, F1, r1) = even, odd
        overlaps = (F0.T @ to_reference) ** 2
        k = int(np.argmax(overlaps))
        c = np.stack([F1.T @ D1 @ F0[:, k], F1.T @ D2 @ F0[:, k]], axis=1)
        c *= np.sign(c[np.arange(c.shape[0]), np.argmax(np.abs(c), axis=1)])[:, None]
        return E1 - E0[k], c, overlaps, k, (r0, r1)

    def strongest(dE, c, sign):
        weight = np.where(sign * dE > 0, np.sum(c ** 2, axis=1), -1.0)
        return float(dE[np.argmax(weight)]) if np.max(weight) > 0 else None

    dE, c, overlaps, k, ranks = poles(blocks[0][0], blocks[1][0])
    scan = {}
    for cut, even, odd in zip(SOFT_MODE_GCM_CUT_SCAN, blocks[0][1:], blocks[1][1:]):
        e, cc, _, _, r = poles(even, odd)
        scan[cut] = (strongest(e, cc, -1.0), strongest(e, cc, 1.0), r)
    F0, F1 = blocks[0][0][1], blocks[1][0][1]
    return dict(excitations=dE, couplings=c, even_levels=blocks[0][0][0], reference=k,
                reference_overlaps=overlaps, ranks=ranks,
                spin_square=(float(F0[:, k] @ S2 @ F0[:, k]), np.einsum('pm,pq,qm->m', F1, S2, F1)),
                cut_scan=scan, half_periods=half)


def collision_modes(dE, couplings, plane, conjugates):
    """(Omega, X, Y) of a collision's two-mode levels as ladder poles: X + Y = sqrt2 (c_m1 e_1 +
    c_m2 e_2), the transition density of `collision_levels` on the plane, and X - Y = dE_m
    (A - B)^-1 (X + Y) = sqrt2 dE_m (c_m1 (A - B)^-1 e_1 + c_m2 (A - B)^-1 e_2) (conjugates, from
    `krein_pairs`), columns in the house's Casida layout, X'X - Y'Y = 2 dE_m c_m^T diag(masses)
    c_m: of the sign of dE_m M for a level coupled through one direction of mass M. With one
    direction it is `collective_modes`. Levels whose |c_m|^2 is below SOFT_MODE_WEIGHT_CUT are
    dropped."""
    keep = np.sum(couplings ** 2, axis=1) > SOFT_MODE_WEIGHT_CUT
    dE, c = dE[keep], couplings[keep]
    xpy = np.sqrt(2.0) * plane @ c.T
    xmy = np.sqrt(2.0) * (conjugates @ c.T) * dE
    return dE, (xpy + xmy) / 2, (xpy - xmy) / 2


def _collision_pair(ref, pair, channel, label, tda):
    """(record, poles) of a colliding pair of `krein_pairs` in the two-mode model
    (`collision_levels`), with its flags (`SoftModeCollisionShort`)."""
    gcm = collision_levels(ref, pair['plane'], channel)
    ranks, kept, overlaps = gcm['ranks'], gcm['reference'], gcm['reference_overlaps']
    dE, c = gcm['excitations'], gcm['couplings']
    weight = np.sum(c ** 2, axis=1)
    below, above = np.where(dE < 0, weight, -1.0), np.where(dE > 0, weight, -1.0)
    w_tda, u_tda = tda
    nearest = np.argsort(-np.linalg.norm(u_tda.T @ pair['plane'], axis=1))[:2]
    info = dict(index=list(pair['index']), omega2=pair['omega2'], kind=pair['kind'],
                gap=pair['gap'], masses=pair['masses'], half_periods=gcm['half_periods'],
                deexcitation=float(dE[np.argmax(below)]) if np.max(below) > 0 else None,
                excitation=float(dE[np.argmax(above)]) if np.max(above) > 0 else None,
                excitations=dE, couplings=c, reference_overlap=float(overlaps[kept]),
                reference_overlaps=overlaps, reference_shift=float(gcm['even_levels'][kept]),
                ranks=ranks, spin_square=gcm['spin_square'], cut_scan=gcm['cut_scan'],
                omega_tda=np.sort(w_tda[nearest]))
    if overlaps[kept] < SOFT_MODE_COLLISION_OVERLAP_MIN:
        second = float(np.sort(overlaps)[-2]) if overlaps.size > 1 else 0.0
        flag('soft_mode_collision_reference',
             f'{label}: the even state kept as the reference\'s continuation, the one of largest '
             f'overlap with the reference determinant, holds {overlaps[kept]:.3f} of it (another '
             f'even state {second:.3f}): the reference shares its determinant with a doubly '
             'excited configuration of the plane; the levels are kept', SoftModeCollisionShort)
    if ranks[0] < 4 or ranks[1] < 2:
        flag('soft_mode_collision_few_states',
             f'{label}: the canonical orthogonalization keeps {ranks[0]} even and {ranks[1]} odd '
             'states, short of the plane\'s span for one occupied orbital (four even, two odd)',
             SoftModeCollisionShort)
    return info, collision_modes(dE, c, pair['plane'], pair['conjugates'])


def _krein_gap(label, mode, signature, omega2, s):
    """The krein_gap of soft mode s (its record mode): the relative gap |Omega_s^2 - Omega_n^2| /
    max(|Omega_s^2|, |Omega_n^2|) to the nearest mode n of the other Krein signature, on the real
    parts of omega2 (the modes of a quartet, signature 0, left out; those of a 'neighbours' pair of
    `krein_pairs`, of definite signature, counted), or None; below SOFT_MODE_KREIN_GAP flagged
    'soft_mode_krein_neighbour' (`SoftModeExcitedShort`)."""
    other = (signature != signature[s]) & (signature != 0)
    if not np.any(other):
        return None
    o2 = np.real(omega2)
    gap = float(np.min(np.abs(o2[other] - o2[s]) / np.maximum(np.abs(o2[other]), abs(o2[s]))))
    if gap < SOFT_MODE_KREIN_GAP:
        flag('soft_mode_krein_neighbour',
             f'{label} (signature {int(np.sign(mode["mass"])):+d}): a mode of the other Krein '
             f'signature lies within {gap:.3f} of its Omega^2 = {mode["omega2"]:+.4e} Ha^2 '
             '(relative), where the two mix toward a Krein collision that the one-mode split '
             'leaves out; the levels are kept', SoftModeExcitedShort)
    return gap


def _sign_aware_spectrum(mf, channel, model, reference, collisions='refuse'):
    """The record of `soft_mode_singlets`, and of `soft_mode_triplets` with reference='excited':
    the channel's TDHF spectrum with its soft modes quantized sign-aware on the straight path, and
    with collisions='gcm' its colliding pairs in the two-mode model (the options already
    checked)."""
    if reference == 'excited':
        ref, order = excited_reference(mf)
    else:
        ref, order = mf, np.arange(np.asarray(mf.mo_occ).size)
    nocc = check_reference(ref)
    A, B = triplet_tdhf_matrices(ref) if channel == 'triplet' else singlet_tdhf_matrices(ref)
    if collisions == 'gcm':
        split = krein_pairs(A, B)
    else:
        split = krein_split(A, B) if reference == 'excited' else soft_split(A, B)
    signature = split.get('signature', np.ones(split['omega2'].size, dtype=int))
    indefinite = bool(np.any(signature != 1))
    w_tda, u_tda = eigh_sym(A)
    flags, soft_info, parts, pair_info = [], [], [split['regular']], []
    with collecting(flags):
        for s in split['soft']:
            mode = split['modes'][s]
            d, M = mode['direction'], mode['mass']
            sign = 1.0 if M > 0 else -1.0
            label = f'soft {channel} mode {s}'
            L, sigma = path_period(d, nocc)
            if channel == 'singlet':
                parity = _singlet_parity(ref, d, nocc, L)
                if parity > SOFT_MODE_SINGLET_PARITY_MAX:
                    raise NotImplementedError(
                        f'{label}: its RHF->RHF path is not even in Q (|V(-Q) - V(Q)| up to '
                        f'{parity:.1e} of max |V|, bar {SOFT_MODE_SINGLET_PARITY_MAX:.0e}): no '
                        'symmetry of the reference takes d to -d (a totally symmetric direction), '
                        'and the parity-blocked collective models solve the even part of V only')
            info = dict(index=s, omega2=mode['omega2'], mass=M, stiffness=mode['stiffness'],
                        half_period=L, sigma=sigma, path='straight', signature=int(sign),
                        level='bottom' if sign > 0 else 'top')
            if model == 'gcm':
                level = 'top' if sign < 0 else ('overlap' if indefinite else 'bottom')
                entries, poles = _sign_aware_gcm(ref, mode, sigma, channel, reference, label,
                                                 level)
                info.update(entries)
            else:
                q, V, D = path_samples(ref, d, L, channel)
                dE, cm, edge = collective_levels(abs(M), q, sign * V, D, L, model)
                info.update(v_min=float((sign * V).min()), q_min=float(q[np.argmin(sign * V)]),
                            excitation=float(sign * dE[0]), edge_weight=edge)
                moved = 0.0
                if model == '1d':
                    walls = collective_levels(abs(M), q, sign * V, D, L, model, 'walls')[0]
                    info['excitation_walls'] = float(sign * walls[0])
                    moved = abs(walls[0] - dE[0])
                if edge > SOFT_MODE_EDGE_WEIGHT_MAX or moved > SOFT_MODE_BOUNDARY_SHIFT_MAX:
                    flag('soft_mode_edge_weight',
                         f'{label}: the collective reference state has {edge:.1e} of its weight '
                         f'near the ends of its path, and its first transition moves by '
                         f'{moved:.1e} Ha when the periodic path is cut by walls; its levels '
                         'depend on how the path closes', SoftModeEdgeWeight)
                poles = collective_modes(sign * dE, cm, M, d, mode['conjugate'])
            if reference == 'excited':
                info['krein_gap'] = _krein_gap(label, mode, signature, split['omega2'], s)
            soft_info.append(info)
            parts.append(poles)
        for pair in split.get('pairs', ()):
            label = f'{channel} collision of modes {pair["index"]} ({pair["kind"]})'
            info, poles = _collision_pair(ref, pair, channel, label, (w_tda, u_tda))
            pair_info.append(info)
            parts.append(poles)
        groups = _degenerate_groups(split['soft'], split['omega2'])
        if len(groups) > 1 and not pair_info:
            flag('soft_modes_coupled',
                 f'{len(groups)} sets of soft {channel} modes ({groups}): each has its own '
                 'collective model, and the couplings between them are left out',
                 SoftModesCoupled)
        elif len(groups) + len(pair_info) > 1:
            flag('soft_modes_coupled',
                 f'{len(groups)} sets of soft {channel} modes ({groups}) and {len(pair_info)} '
                 f'colliding pairs ({[p["index"] for p in pair_info]}): each has its own '
                 'collective model, and the couplings between them are left out',
                 SoftModesCoupled)
    X, Y = np.hstack([p[1] for p in parts]), np.hstack([p[2] for p in parts])
    d0 = split['modes'][split['soft'][0]]['direction'] if split['soft'] else None
    out = dict(omega=np.concatenate([p[0] for p in parts]), X=X, Y=Y,
               signature=np.where(np.sum(X * X - Y * Y, axis=0) < 0, -1, 1),
               omega2=split['omega2'], omega_tda=w_tda,
               omega_tda_soft=(None if d0 is None
                               else float(w_tda[np.argmax(np.abs(u_tda.T @ d0))])),
               soft=soft_info, orbitals=order, numerical_flags=gathered(flags))
    if collisions == 'gcm':
        out['pairs'] = pair_info
    return out


def soft_mode_triplets(mf, model='1d', path='straight', reference='aufbau', collisions='refuse'):
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
                       cold_restarts, restarted, held_modes, curvature, frozen_from,
                       energy_calls and continued_from (`relaxed_path`; on a continued path
                       antipode, q_min and restarted are in its arc length); with model='gcm'
                       coupling (c_1), ranks,
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

    reference (`REFERENCES`): 'aufbau', the above; or 'excited', any stationary closed-shell RHF,
    its occupation aufbau or not (`excited_reference`; the relaxed path refused with
    NotImplementedError, `check_reference_option`), split sign-aware (`krein_split`): a soft mode
    of mass M > 0 is quantized as above, one of M < 0 (a negative transition, or the instability
    of an excited reference) by -H = P^2 / 2|M| - V(Q), whose lowest level, the top of H's, is the
    reference's (in the GCM the even level of largest reference overlap above the even ground
    state, `gcm_levels`), so that its transitions dE_m < 0 are de-excitations. In the GCM a mode of
    M > 0 on a block whose A - B is indefinite keeps the even ground state unless another even
    state overlaps the reference determinant more by over SOFT_MODE_GCM_OVERLAP_MARGIN
    (level='overlap', its transitions those above the kept state first). The inversion of a mode
    with M < 0 and mu > 0 is validated on excited references only; on a ground-state determinant
    (an aufbau reference whose A - B is indefinite, a complex instability) it is not, and the
    occupation does not tell the two apart here (H2's sigma_u^2 is aufbau in its own Fock
    spectrum from 1.2 A on): reference='excited' declares the reference excited. The record is the
    one of `soft_mode_singlets` (on the triplet block): the above with every mass, excitation and
    pole signed, plus signature (the sign of X'X - Y'Y of each column), orbitals (mf's orbital
    indices in the order of the pair index), and per soft mode signature (sign of M), level
    ('bottom' or 'top'; in the GCM its rule, 'bottom', 'top' or 'overlap'), krein_gap and, with
    model='gcm', reference_overlaps and reference (the index of the even state kept); its flags
    add 'soft_mode_krein_neighbour', 'soft_mode_reference_overlap' and 'soft_mode_reference_tie'
    (`SoftModeExcitedShort`). A complex Omega^2 (a Krein quartet) is refused with
    `LinearResponse.casida.CasidaInstability`. With an aufbau reference whose A - B is positive
    definite it gives the default's levels and poles.

    collisions (`COLLISIONS`, reference='excited' only): 'refuse' (the default) as above, or
    'gcm', where each Krein quartet, and each soft mode flagged as a Krein neighbour together with
    that neighbour when each is the other's nearest (else the mode is left unpaired, and
    flagged: the neighbour is in a degenerate set or another mode's partner), is quantized as one
    pair by the two-mode generator-coordinate model on the
    determinants of the pair's plane (`krein_pairs`, `collision_levels`, `collision_modes`; the
    other soft modes in `model`): its levels are real, its reference state the even one of
    largest overlap with the reference determinant, and every odd state goes back as a pole. The
    record then holds omega2 complex, soft without the paired modes (omega_tda_soft None when it
    is empty; a soft mode's krein_gap leaves a quartet's modes out and counts a 'neighbours'
    pair's), and pairs, per pair: index,
    omega2, kind ('quartet' or 'neighbours'), gap, masses (of the plane's basis, one of each
    sign), half_periods, deexcitation and excitation (the poles of largest coupling below and
    above the reference), excitations and couplings (every pole), reference_overlap and
    reference_overlaps, reference_shift (the reference state's energy above the reference
    determinant's), ranks, spin_square, cut_scan and omega_tda (the TDA roots nearest the plane);
    its flags add 'soft_mode_collision_reference' and 'soft_mode_collision_few_states'
    (`SoftModeCollisionShort`), and 'soft_modes_coupled' counts the colliding pairs beside the
    soft sets (raised for more than one of either). Without a collision it gives the record of
    collisions='refuse', with pairs = [].
    """
    if model not in MODELS:
        raise ValueError(f'model = {model!r}: one of {MODELS}')
    check_path(path, model)
    check_reference_option(reference, path)
    check_collisions_option(collisions, reference)
    if reference == 'excited':
        return _sign_aware_spectrum(mf, 'triplet', model, reference, collisions)
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
                            frozen_from=rel['frozen_from'], energy_calls=rel['energy_calls'],
                            continued_from=rel['continued_from'])
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
                         'SOFT_MODE_RELAXED_REACH half-periods; its continuation in arc length, '
                         'tried only when the probes find no negative curvature and three '
                         'fixed-plane samples can be kept, was not tried or did not reach the '
                         'antipode', SoftModePathShort)
                curvature = rel['curvature']
                saddle = curvature is not None and curvature < -SOFT_MODE_RELAXED_SADDLE_BAR
                x, along = (('Q', '') if rel['continued_from'] is None
                            else ('x', ', x the arc length of the continued path'))
                if rel['restarted'].size or saddle:
                    flag('soft_mode_path_branch',
                         f'soft triplet mode {s}: its relaxed path is not a constrained minimum '
                         f'throughout: {rel["restarted"].size} samples ({x} / L = '
                         f'{np.round(rel["restarted"] / L, 3).tolist()}{along}) were taken from a '
                         'fresh start at kappa = Q d after the warm start left the branch, and the '
                         f'lowest curvature probed across the constraint plane is {curvature:+.3e} '
                         '(in units of the RHF curvature) with the TDHF triplet modes '
                         f'{rel["held_modes"]} held besides the soft ones; the samples are kept',
                         SoftModePathShort)
                if (rel['closure_overlap'] < SOFT_MODE_RELAXED_CLOSURE_MIN
                        or abs(rel['coupling_antipode']) > SOFT_MODE_RELAXED_COUPLING_MAX):
                    flag('soft_mode_path_closure',
                         f'soft triplet mode {s}: its relaxed loop does not close at the antipode '
                         f'{x}* = {rel["antipode"] / L:.4f} L{along}: the alpha determinants at '
                         f'-+{x}* overlap by {rel["closure_overlap"]:.4f} and D({x}*) = '
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


def soft_mode_singlets(mf, model='1d', path='straight', reference='aufbau', collisions='refuse'):
    """The TDHF singlet spectrum of an RHF with its soft modes replaced by their collective levels,
    split sign-aware: the singlet block A = De + 2 (ia|jb) - (ij|ab), B = 2 (ia|jb) - (ib|ja)
    (`singlet_tdhf_matrices`), whose soft modes turn both spins alike along d, an RHF->RHF path
    (`path_samples` with channel='singlet'; a spatial symmetry breaking such as H2 sigma_u^2's
    toward H+H-), quantized in the 1D model or the GCM: a dict with

    omega, X, Y        the spectrum (Ha) and its Casida vectors, regular modes first: each regular
                       mode's positive-norm member (X'X - Y'Y = 1; a negative transition of an
                       excited reference has omega < 0, and at |omega| it is the negative-norm
                       member (Y, X): the indefinite metric's sign), then the collective levels'
                       poles (`collective_modes`, X'X - Y'Y = 2 M dE_m c_m^2, of the sign of
                       M dE_m)
    signature          per column, the sign of X'X - Y'Y: +1, or -1 for a collective pole against
                       the sign of its mode's mass (a GCM level on the other side of the reference)
    omega2, omega_tda, omega_tda_soft   as in `soft_mode_triplets`, for the singlet block
    orbitals           mf's orbital indices in the order of the pair index (occupied first)
    soft               per soft mode: index, omega2, mass (M, of either sign), stiffness (mu),
                       signature (sign of M), level ('bottom' for M > 0, 'top' for M < 0: the
                       collective level that continues the reference; in the GCM the rule of
                       `gcm_levels` that picks it, 'overlap' for M > 0 on a block whose A - B is
                       indefinite), half_period, sigma, path,
                       v_min and q_min (of the potential quantized, sign(M) V), excitation (the
                       first transition from the reference's level, Ha, negative for M < 0),
                       excitation_walls and edge_weight (1D), with reference='excited' krein_gap
                       (the relative gap in Omega^2 to the nearest mode of the other signature,
                       a quartet's modes of collisions='gcm' left out, or None), and with
                       model='gcm' coupling, ranks, spin_square (0), cut_scan,
                       reference_overlaps and reference (the index of the even state kept,
                       `gcm_levels`) in place of excitation_walls and edge_weight
    numerical_flags    'soft_mode_edge_weight', 'soft_modes_coupled' (more than one soft set, or
                       with collisions='gcm' more than one soft set or colliding pair), with
                       model='gcm' 'soft_mode_gcm_degenerate_pairs' and
                       'soft_mode_gcm_few_states', and with reference='excited'
                       'soft_mode_krein_neighbour', 'soft_mode_reference_overlap' and
                       'soft_mode_reference_tie' (`_reference_flags`)

    model is '1d' or 'gcm' ('3d', the three triplet components, is refused: a singlet has one);
    path is 'straight' only (the relaxed path is refused with NotImplementedError: it is minimised
    over triplet rotations, and an excited reference would need a saddle search). reference
    (`REFERENCES`): 'aufbau', a converged aufbau closed shell with A - B positive definite (else
    `LinearResponse.casida.CasidaInstability`), its modes all of positive mass; or 'excited', any
    stationary closed shell (`excited_reference`), split by `krein_split`: a mode of M < 0 is
    quantized as -H = P^2 / 2|M| - V(Q), whose lowest level is the top of H's and continues the
    reference (in the GCM the top even level by overlap, `gcm_levels`), so that its transitions
    are de-excitations, and in the GCM a mode of M > 0 on a block whose A - B is indefinite keeps
    the even state of `gcm_levels`' level='overlap' (as in `soft_mode_triplets`). A soft
    direction whose path is not even in Q (|V(-Q) - V(Q)| above SOFT_MODE_SINGLET_PARITY_MAX of
    max |V|: no symmetry takes d to -d) is refused with NotImplementedError, the parity-blocked
    models being built on an even V. Refusals of the options come before any work. collisions
    (`COLLISIONS`): as in `soft_mode_triplets`, on the singlet block's plane (its potential checked
    even under kappa -> -kappa, as for a soft direction).
    """
    if model not in MODELS:
        raise ValueError(f'model = {model!r}: one of {MODELS}')
    if model == '3d':
        raise ValueError("model = '3d': the 3D model quantizes the three triplet components; a "
                         "singlet soft mode has one ('1d' or 'gcm')")
    check_path(path, model)
    if path == 'relaxed':
        raise NotImplementedError("path = 'relaxed' for a singlet soft mode: the relaxed path is "
                                  'minimised over the triplet rotations; the RHF->RHF one is not '
                                  'implemented')
    check_reference_option(reference, path)
    check_collisions_option(collisions, reference)
    return _sign_aware_spectrum(mf, 'singlet', model, reference, collisions)


def _open_shell_occupation(mf):
    """(occ, counts): the occupations of a UHF (2 x nmo) and its (n_alpha, n_beta), after the
    refusals of `open_shell_reference` that need no integrals: the type, a functional, density
    fitting, convergence, an occupation other than 0 and 1 per spin orbital, n_beta > n_alpha and
    no occupied-virtual pair in either spin."""
    if not isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError(f'{type(mf).__name__}: the open-shell soft-mode split takes an '
                                  'unrestricted Hartree-Fock reference (scf.UHF); a closed shell '
                                  'goes to soft_mode_triplets and soft_mode_singlets')
    if getattr(mf, 'xc', None) is not None:
        raise ValueError(f'mf.xc = {mf.xc!r}: the open-shell soft-mode split needs a Hartree-Fock '
                         'reference, whose energy defines its paths')
    if getattr(mf, 'with_df', None) is not None:
        raise ValueError('mf.with_df is set: the open-shell split builds its matrices and path '
                         'energies from exact integrals, which a density-fitted reference does '
                         'not make stationary')
    if not getattr(mf, 'converged', False):
        raise ValueError('mf.converged = False: the open-shell paths start at a converged UHF')
    occ = np.asarray(mf.mo_occ, dtype=float)
    if occ.ndim != 2 or not np.all((occ == 0) | (occ == 1)):
        raise ValueError('mf.mo_occ: the open-shell split needs 0 or 1 electron in each spin '
                         'orbital of a UHF')
    counts = [int(np.count_nonzero(o)) for o in occ]
    if counts[0] < counts[1]:
        raise ValueError(f'mf has {counts[1]} beta and {counts[0]} alpha electrons: the open-shell '
                         'split takes n_alpha >= n_beta (the spin mirror is the same reference)')
    nmo = occ.shape[1]
    if all(n in (0, nmo) for n in counts):
        raise ValueError('mf.mo_occ: no occupied-virtual pair to split in either spin')
    return occ, tuple(counts)


def open_shell_reference(mf):
    """(ref, orders) for the open-shell split (`soft_mode_open_shell`): ref a shallow copy of a
    converged UHF whose occupation need not be aufbau in either spin (a maximum-overlap core hole),
    with each spin's orbitals, energies and occupations reordered so that its occupied orbitals come
    first (each block in mf's order), as `uhf_tdhf_matrices` reads them; orders = (alpha order,
    beta order), mf's orbital indices in that order.

    Refused before any work: a mean field that is not an unrestricted Hartree-Fock one (RHF, ROHF
    and GHF, NotImplementedError: a closed shell goes to `soft_mode_triplets` and
    `soft_mode_singlets`, or through `scf.addons.convert_to_uhf`); a functional, density fitting or
    an unconverged SCF (ValueError, as `check_reference`); an occupation other than 0 and 1 per
    spin orbital, more beta than alpha electrons (its spin mirror is the same reference), or no
    occupied-virtual pair in either spin; and an occupation that is not stationary, the
    occupied-virtual block of either spin's Fock matrix of mf's own densities reaching
    SOFT_MODE_EXCITED_GRADIENT_MAX (Ha), as `excited_reference` refuses it."""
    occ = _open_shell_occupation(mf)[0]
    C = np.asarray(mf.mo_coeff)
    fock = mf.get_hcore() + mf.get_veff(mf.mol, mf.make_rdm1(C, occ))
    gradient = 0.0
    for s in (0, 1):
        o, v = np.flatnonzero(occ[s] == 1), np.flatnonzero(occ[s] == 0)
        if o.size and v.size:
            gradient = max(gradient,
                           float(np.max(np.abs(C[s][:, o].T @ fock[s] @ C[s][:, v]))))
    if gradient >= SOFT_MODE_EXCITED_GRADIENT_MAX:
        raise ValueError(f'mf.mo_occ is not stationary: an occupied-virtual block of the Fock '
                         f'matrices of its densities reaches {gradient:.1e} Ha (bar '
                         f'{SOFT_MODE_EXCITED_GRADIENT_MAX:.0e}); converge the UHF with this '
                         'occupation (maximum overlap, or symmetry-constrained)')
    orders = tuple(np.r_[np.flatnonzero(o == 1), np.flatnonzero(o == 0)] for o in occ)
    ref = mf.copy()
    ref.mo_coeff = np.array([C[s][:, orders[s]] for s in (0, 1)])
    ref.mo_energy = np.array([np.asarray(mf.mo_energy[s])[orders[s]] for s in (0, 1)])
    ref.mo_occ = np.array([occ[s][orders[s]] for s in (0, 1)])
    return ref, orders


def _open_shell_counts(ref):
    """(n_alpha, n_beta) of a UHF whose occupied orbitals come first in each spin; anything else
    refused (ValueError)."""
    if not isinstance(ref, scf.uhf.UHF):
        raise NotImplementedError(f'{type(ref).__name__}: a UHF reference is needed '
                                  '(`open_shell_reference`)')
    occ = np.asarray(ref.mo_occ, dtype=float)
    counts = []
    for o in occ:
        n = int(np.count_nonzero(o))
        if not (np.all(o[:n] == 1) and np.all(o[n:] == 0)):
            raise ValueError('ref.mo_occ: the occupied orbitals of each spin must come first; '
                             'reorder with open_shell_reference')
        counts.append(n)
    return tuple(counts)


def _aufbau_in_both_spins(ref):
    """Whether, in each spin of a UHF whose occupied orbitals come first (`open_shell_reference`),
    the occupied orbitals lie below the virtual ones in its own orbital energies (a spin with every
    orbital occupied or empty counts): the occupation of a ground-state determinant, not a
    maximum-overlap hole."""
    n = _open_shell_counts(ref)
    eps = np.asarray(ref.mo_energy)
    return all(n[s] in (0, eps[s].size)
               or float(np.max(eps[s][:n[s]])) < float(np.min(eps[s][n[s]:])) for s in (0, 1))


def _mo_block(mol, c1, c2, c3, c4):
    """(c1 c2 | c3 c4) over the columns of four orbital blocks (exact integrals), as an
    n1 x n2 x n3 x n4 array; zeros when a block is empty."""
    shape = (c1.shape[1], c2.shape[1], c3.shape[1], c4.shape[1])
    if 0 in shape:
        return np.zeros(shape)
    return ao2mo.general(mol, (c1, c2, c3, c4), compact=False).reshape(shape)


def uhf_tdhf_matrices(ref, block):
    """(A, B) of a UHF's TDHF block (Ha), from MO integral blocks, for a reference whose occupied
    orbitals come first in each spin (`open_shell_reference`), in the pair layout of `BLOCKS`:

    spin_conserving  pairs (i alpha -> a alpha), index i * nvirt_alpha + a, then
                     (i beta -> a beta), index n_alpha nvirt_alpha + i * nvirt_beta + a:
                     A = De + (ia|jb) - delta_st (ij|ab), B = (ia|jb) - delta_st (ib|ja)
                     (s, t the spins of the two pairs), the matrices of pyscf's TDHF on a UHF
    spin_flip        pairs (i alpha -> a beta), index i * nvirt_beta + a, then (i beta -> a alpha),
                     index n_alpha nvirt_beta + i * nvirt_alpha + a: A = De - (ij|ab) within each
                     sector (De = e_a(beta) - e_i(alpha) in the first, e_a(alpha) - e_i(beta) in
                     the second) and zero between them, B = -(ib|ja) between them (i, a of one
                     sector, j, b of the other) and zero within, so that D (A + B) D = A - B with
                     D = +1 on the first sector and -1 on the second

    These are the blocks of the orbital Hessian: turning the spin orbitals by exp(K), K_ai = -K_ia
    = kappa_P over the block's pairs P = (i, a), changes the energy by kappa^T (A + B) kappa to
    second order (a UHF determinant for the spin-conserving block, a GHF one for the spin-flip
    block). On a closed shell the spin-flip block holds the TDHF triplet spectrum twice (its
    M_s = +-1 components) and the spin-conserving one the singlet and the triplet's M_s = 0. The
    same matrices as `LinearResponse.linear_response.LinearResponseSolver`'s unrestricted
    build_casida_matrices(nocc, lBSE=True) and build_spin_flip_casida_matrices(nocc) on the full
    integrals."""
    if block not in BLOCKS:
        raise ValueError(f'block = {block!r}: one of {BLOCKS}')
    n_alpha, n_beta = _open_shell_counts(ref)
    mol = ref.mol
    C, eps = np.asarray(ref.mo_coeff), np.asarray(ref.mo_energy)
    occ = (C[0][:, :n_alpha], C[1][:, :n_beta])
    vir = (C[0][:, n_alpha:], C[1][:, n_beta:])
    e_occ = (eps[0][:n_alpha], eps[1][:n_beta])
    e_vir = (eps[0][n_alpha:], eps[1][n_beta:])
    if block == 'spin_conserving':
        sizes = [occ[s].shape[1] * vir[s].shape[1] for s in (0, 1)]
        de = np.concatenate([(e_vir[s][None, :] - e_occ[s][:, None]).ravel() for s in (0, 1)])
        A, B = np.diag(de), np.zeros((sum(sizes), sum(sizes)))
        start = (0, sizes[0])
        for s in (0, 1):
            for t in range(s, 2):
                ovov = _mo_block(mol, occ[s], vir[s], occ[t], vir[t])
                coulomb = ovov.reshape(sizes[s], sizes[t])
                rows = slice(start[s], start[s] + sizes[s])
                cols = slice(start[t], start[t] + sizes[t])
                A[rows, cols] += coulomb
                B[rows, cols] += coulomb
                if s == t:
                    oovv = _mo_block(mol, occ[s], occ[s], vir[s], vir[s])
                    A[rows, cols] -= oovv.transpose(0, 2, 1, 3).reshape(sizes[s], sizes[s])
                    B[rows, cols] -= ovov.transpose(0, 3, 2, 1).reshape(sizes[s], sizes[s])
                else:
                    A[cols, rows] += coulomb.T
                    B[cols, rows] += coulomb.T
        return A, B
    n1, n2 = n_alpha * vir[1].shape[1], n_beta * vir[0].shape[1]
    de = np.r_[(e_vir[1][None, :] - e_occ[0][:, None]).ravel(),
               (e_vir[0][None, :] - e_occ[1][:, None]).ravel()]
    A, B = np.diag(de), np.zeros((n1 + n2, n1 + n2))
    A[:n1, :n1] -= _mo_block(mol, occ[0], occ[0], vir[1], vir[1]).transpose(0, 2, 1, 3).reshape(
        n1, n1)
    A[n1:, n1:] -= _mo_block(mol, occ[1], occ[1], vir[0], vir[0]).transpose(0, 2, 1, 3).reshape(
        n2, n2)
    coupling = -_mo_block(mol, occ[0], vir[0], occ[1], vir[1]).transpose(0, 3, 2, 1).reshape(n1, n2)
    B[:n1, n1:] = coupling
    B[n1:, :n1] = coupling.T
    return A, B


def spin_rotation_direction(ref):
    """g, the spin-flip block's direction of a global spin rotation (`uhf_tdhf_matrices`' layout):
    the occupied-virtual part of the real rotation S_- - S_+ of the reference,
    g = (<a beta|i alpha>, -<a alpha|i beta>) (spatial overlaps in the AO metric). The GHF energy
    is unchanged by a spin rotation, so that (A + B) g = 0 and (A - B) D g = 0 (D = +1 on the first
    sector, -1 on the second) to the reference's convergence, and g^T D g = n_alpha - n_beta. It
    vanishes for a closed shell."""
    n_alpha, n_beta = _open_shell_counts(ref)
    S = ref.mol.intor_symmetric('int1e_ovlp')
    C = np.asarray(ref.mo_coeff)
    lower = C[0][:, :n_alpha].T @ S @ C[1][:, n_beta:]
    upper = C[1][:, :n_beta].T @ S @ C[0][:, n_alpha:]
    return np.r_[lower.ravel(), -upper.ravel()]


def _open_shell_pairs(ref, block, vector):
    """{(s, t): the n_occ(s) x n_virt(t) matrix} of a block vector: its occupied spin s ->
    virtual spin t blocks (0 alpha, 1 beta), (0, 0) and (1, 1) for 'spin_conserving', (0, 1) and
    (1, 0) for 'spin_flip'."""
    n = _open_shell_counts(ref)
    nmo = np.asarray(ref.mo_occ).shape[1]
    pairs = ((0, 0), (1, 1)) if block == 'spin_conserving' else ((0, 1), (1, 0))
    out, start = {}, 0
    for s, t in pairs:
        size = n[s] * (nmo - n[t])
        out[s, t] = np.reshape(vector[start:start + size], (n[s], nmo - n[t]))
        start += size
    return out


def _open_shell_half_period(ref, block, vector):
    """(pi / (sqrt2 sigma_1), sigma): sigma the singular values (descending) of the block vector
    read as its occupied x virtual spin-orbital matrix, the union of its blocks'
    (`_open_shell_pairs`); on kappa = Q d / sqrt2 its leading pair has turned by 90 degrees at
    Q = pi / (sqrt2 sigma_1) (pi / sigma_1 for a closed shell's triplet d, whose spin-orbital
    singular values are sigma_k / sqrt2)."""
    blocks = [b for b in _open_shell_pairs(ref, block, vector).values() if b.size]
    sigma = np.sort(np.concatenate([np.linalg.svd(b, compute_uv=False) for b in blocks]))[::-1]
    return np.pi / (np.sqrt(2.0) * sigma[0]), sigma


class _OpenShellKernels:
    """The kernels between Slater determinants of spin orbitals, UHF or GHF ones (each spin orbital
    with alpha and beta components), given by their occupied spin orbitals in the orthonormal
    active basis (W_alpha x alpha) + (W_beta x beta): column k of an (m_alpha + m_beta) x N array
    holds the alpha components of spin orbital k on W_alpha, then its beta components on W_beta.
    In each pair's corresponding orbitals x_k, y_k (the singular-value decomposition of the
    occupied overlap over all N spin orbitals, so that a GHF pair needs no special case), free of
    divisions as `_CollisionKernels`:

        N          = c prod_k s_k
        H - E N    = c [(E_nuc - E) prod_k s_k + sum_k h(P_k) prod_{j != k} s_j
                        + sum_{k<l} (J(P_k, P_l) - K(P_k, P_l)) prod_{j != k, l} s_j]
        <O>        = c sum_k x_k^T O y_k prod_{j != k} s_j
        <S^2>      = 3/4 N_el N + 2 c sum_{k<l} [s_kk . s_ll - s_kl . s_lk] prod_{j != k, l} s_j

    with P_k = x_k y_k^T in its four spin blocks P_k^st, J(P_k, P_l) over their spin traces,
    K(P_k, P_l) = sum_st K(P_k^st, P_l^ts) (J and K as in `_GCMKernels`), s_kl = <x_k|s|y_l> the
    one-electron spin vector (s_z, and s_+, s_- through the spatial overlap W_alpha^T S W_beta), c
    the product of the determinants of the rotations to corresponding orbitals, E the energy
    subtracted (energy) and O any one-body operator on the active spin orbitals; hcore is the
    one-electron Hamiltonian in the AO basis. Exact for determinants inside the active spans
    (checked against exact Fock-space vectors)."""

    def __init__(self, mol, bases, hcore, energy):
        W = [np.asarray(w) for w in bases]
        self.m = (W[0].shape[1], W[1].shape[1])
        self.h = [w.T @ hcore @ w for w in W]
        eri = {}
        for s in (0, 1):
            for t in range(s, 2):
                eri[s, t] = _mo_block(mol, W[s], W[s], W[t], W[t])
        eri[1, 0] = eri[0, 1].transpose(2, 3, 0, 1)
        self.coulomb = {k: v.reshape(v.shape[0] * v.shape[1], v.shape[2] * v.shape[3])
                        for k, v in eri.items()}
        self.exchange = {k: np.ascontiguousarray(v.transpose(0, 3, 2, 1)).reshape(
            v.shape[0] * v.shape[3], v.shape[2] * v.shape[1]) for k, v in eri.items()}
        self.cross = W[0].T @ mol.intor_symmetric('int1e_ovlp') @ W[1]
        self.shift = mol.energy_nuc() - energy

    def elements(self, bra, ket, operators=()):
        """(N, H - E N, [<O> for each operator], <S^2>) between the determinants bra and ket
        (arrays P x (m_alpha + m_beta) x N, element by element)."""
        ma = self.m[0]
        U, s, Vt = np.linalg.svd(np.einsum('epi,epj->eij', bra, ket))
        sign = np.linalg.det(U) * np.linalg.det(Vt)
        x = np.einsum('epi,eij->epj', bra, U)
        y = np.einsum('epi,eji->epj', ket, Vt)
        one, two = _products_but(s)
        total = np.prod(s, axis=1)
        rows, _, n = x.shape
        xs, ys = (x[:, :ma], x[:, ma:]), (y[:, :ma], y[:, ma:])
        rho = {(a, b): np.einsum('epk,eqk->ekpq', xs[a], ys[b]).reshape(
            rows, n, xs[a].shape[1] * ys[b].shape[1]) for a in (0, 1) for b in (0, 1)}
        G = np.zeros((rows, n, n))
        for a in (0, 1):
            for b in (0, 1):
                G += np.einsum('ekq,elq->ekl', rho[a, a] @ self.coulomb[a, b], rho[b, b])
                G -= np.einsum('ekq,elq->ekl', rho[a, b] @ self.exchange[a, b], rho[b, a])
        k, l = np.triu_indices(n, 1)
        h1 = rho[0, 0] @ self.h[0].reshape(-1) + rho[1, 1] @ self.h[1].reshape(-1)
        H = sign * (self.shift * total + np.sum(h1 * one, axis=1)
                    + np.sum(G[:, k, l] * two[:, k, l], axis=1))
        values = [sign * np.sum(np.einsum('epk,pq,eqk->ek', x, op, y) * one, axis=1)
                  for op in operators]
        z = 0.5 * (np.einsum('epk,epl->ekl', xs[0], ys[0])
                   - np.einsum('epk,epl->ekl', xs[1], ys[1]))
        up = np.einsum('epk,pq,eql->ekl', xs[0], self.cross, ys[1])
        down = np.einsum('epk,qp,eql->ekl', xs[1], self.cross, ys[0])
        zd, ud, dd = (np.einsum('ekk->ek', t) for t in (z, up, down))
        direct = zd[:, :, None] * zd[:, None, :] + 0.5 * (ud[:, :, None] * dd[:, None, :]
                                                          + dd[:, :, None] * ud[:, None, :])
        swapped = z * z.transpose(0, 2, 1) + 0.5 * (up * down.transpose(0, 2, 1)
                                                     + down * up.transpose(0, 2, 1))
        S2 = sign * (0.75 * n * total + 2.0 * np.sum((direct - swapped)[:, k, l] * two[:, k, l],
                                                      axis=1))
        return sign * total, H, values, S2

    def matrices(self, orbitals, operators=()):
        """(N, H - E N, S^2, then <O> for each operator) between the determinants of orbitals
        (P x (m_alpha + m_beta) x N), as P x P matrices."""
        P = orbitals.shape[0]
        out = np.zeros((3 + len(operators), P, P))
        for i in range(P):
            j = np.arange(i, P)
            N, H, values, S2 = self.elements(np.broadcast_to(orbitals[i], orbitals[j].shape),
                                             orbitals[j], operators)
            out[:, i, j] = np.array([N, H, S2] + values)
            out[:, j, i] = out[:, i, j]
        return out


class _OpenShellSpace:
    """The determinants exp(K(kappa)) Phi_0 of an open-shell reference for kappa in the span of
    some block vectors (`uhf_tdhf_matrices`' layout; K_ai = -K_ia = kappa_P): each spin's active
    basis holds its occupied orbitals and the virtual span the vectors reach in it (the
    singular-value decomposition of their stacked occupied x virtual blocks, as in
    `_CollisionKernels`), which holds every such determinant (`bases`, AO coefficients); the
    reference determinant is the unit columns of the occupied orbitals, and `kernels`
    (`_OpenShellKernels`) are relative to its energy."""

    def __init__(self, ref, block, vectors):
        self.ref, self.block = ref, block
        self.n = _open_shell_counts(ref)
        C = np.asarray(ref.mo_coeff)
        reach = ([], [])
        for v in vectors:
            for (s, t), b in _open_shell_pairs(ref, block, v).items():
                reach[t].append(b)
        self.span, bases = [], []
        for s in (0, 1):
            n_virt = C.shape[2] - self.n[s]
            stacked = [b for b in reach[s] if b.size]
            V = (np.linalg.svd(np.vstack(stacked), full_matrices=False)[2].T if stacked
                 else np.zeros((n_virt, 0)))
            self.span.append(V)
            bases.append(np.hstack([C[s][:, :self.n[s]], C[s][:, self.n[s]:] @ V]))
        self.bases = bases
        self.kernels = _OpenShellKernels(ref.mol, bases, ref.get_hcore(), ref.e_tot)
        ma, mb = self.kernels.m
        self.start = ((0, self.n[0]), (ma, ma + self.n[1]))
        occupied = np.r_[np.arange(self.n[0]), ma + np.arange(self.n[1])]
        self.reference = np.eye(ma + mb)[:, occupied]

    def lower(self, vector):
        """L, the virtual <- occupied part of K(vector) on the active spin orbitals."""
        ma, mb = self.kernels.m
        L = np.zeros((ma + mb, ma + mb))
        for (s, t), b in _open_shell_pairs(self.ref, self.block, vector).items():
            part = (b @ self.span[t]).T
            row, col = self.start[t][1], self.start[s][0]
            L[row:row + part.shape[0], col:col + part.shape[1]] += part
        return L

    def generator(self, vector):
        """K(vector) on the active spin orbitals (antisymmetric, K_ai = vector_P)."""
        L = self.lower(vector)
        return L - L.T

    def operator(self, vector):
        """D = 2^-1/2 sum_P vector_P (a+_a a_i + a+_i a_a) on the active spin orbitals, whose
        expectation along kappa = Q d / sqrt2 is Q |d|^2 to first order."""
        L = self.lower(vector)
        return (L + L.T) / np.sqrt(2.0)

    def orbitals(self, generators, coefficients):
        """The occupied spin orbitals (P x (m_alpha + m_beta) x N) of exp(sum_k c_k K_k) Phi_0 for
        the rows c of coefficients."""
        out = np.empty((len(coefficients),) + self.reference.shape)
        for p, c in enumerate(coefficients):
            out[p] = expm(sum(ck * K for ck, K in zip(c, generators))) @ self.reference
        return out


def open_shell_path(ref, block, direction, q_end):
    """(Q, V, D) on SOFT_MODE_PATH_POINTS points of [0, q_end] along the straight open-shell path
    of a block vector d (|d| = 1, `uhf_tdhf_matrices`' layout): V(Q) = E[exp(K(Q d / sqrt2)) Phi_0]
    - E_ref (Ha), the energy of the UHF determinant ('spin_conserving') or GHF determinant
    ('spin_flip') whose spin orbitals are turned by exp(K), and D(Q) = <D_d>, D_d =
    2^-1/2 sum_P d_P (a+_a a_i + a+_i a_a), which is Q to first order; V''(0) = d^T (A + B) d. On a
    closed shell's triplet direction (d, -d) / sqrt2 in the spin-conserving block it is
    `path_samples`' path. From the kernels' diagonal (`_OpenShellKernels`)."""
    if block not in BLOCKS:
        raise ValueError(f'block = {block!r}: one of {BLOCKS}')
    space = _OpenShellSpace(ref, block, [direction])
    q = np.linspace(0.0, q_end, SOFT_MODE_PATH_POINTS)
    orbitals = space.orbitals([space.generator(direction) / np.sqrt(2.0)], q[:, None])
    _, V, (D,), _ = space.kernels.elements(orbitals, orbitals, [space.operator(direction)])
    return q, V, D


def _open_shell_parity(ref, block, direction, L):
    """max |V(-Q) - V(Q)| / max |V(Q)| of the open-shell path along direction on
    SOFT_MODE_PATH_POINTS points of [0, L]: 0 where a symmetry of the reference takes d to -d
    (always so in the spin-flip block, by the spin rotation of pi about z)."""
    V = open_shell_path(ref, block, direction, L)[1]
    scale = float(np.max(np.abs(V)))
    mirrored = open_shell_path(ref, block, -direction, L)[1]
    return float(np.max(np.abs(mirrored - V))) / scale if scale > 0 else 0.0


def _open_shell_hill_wheeler(N, H, bases):
    """Per parity block (the columns of each basis): [(E, F, rank)] at SOFT_MODE_GCM_OVERLAP_CUT and
    at each cut of SOFT_MODE_GCM_CUT_SCAN, by canonical orthogonalization as in `gcm_levels`
    (F^T N F = 1)."""
    out = []
    for basis in bases:
        w, U = eigh_sym(basis.T @ N @ basis)
        Hb = basis.T @ H @ basis
        solved = []
        for cut in (SOFT_MODE_GCM_OVERLAP_CUT,) + tuple(SOFT_MODE_GCM_CUT_SCAN):
            keep = w > cut * w[-1]
            X = U[:, keep] / np.sqrt(w[keep])
            E, c = eigh_sym(X.T @ Hb @ X)
            solved.append((E, basis @ X @ c, int(np.count_nonzero(keep))))
        out.append(solved)
    return out


def open_shell_gcm_levels(ref, block, direction, level='bottom'):
    """The generator-coordinate (Hill-Wheeler) model of one soft mode of an open-shell block on its
    straight path (`open_shell_path`): the dict of `gcm_levels` (excitations, couplings, ranks,
    spin_square, cut_scan, q, V, reference_overlaps, reference), on the SOFT_MODE_GCM_POINTS
    midpoints of [-L, L], L = pi / (sqrt2 sigma_1) (`_open_shell_half_period`), with the kernels of
    `_OpenShellKernels` and the coupling D_d of `open_shell_path`, solved in the even and odd
    blocks of Q -> -Q (for 'spin_flip' the spin rotation of pi about z, exact; for
    'spin_conserving' a symmetry of the reference that takes d to -d, which the driver checks).
    level (`LEVELS`): the even state that continues the reference, 'bottom' for a mode of positive
    mass, 'top' (picked by overlap, `_reference_level`) for a negative one, and 'overlap' for a
    positive one on a block whose A - B is indefinite, as in `gcm_levels`. On a closed shell's
    triplet direction in the spin-conserving block it is `gcm_levels`' model."""
    if block not in BLOCKS:
        raise ValueError(f'block = {block!r}: one of {BLOCKS}')
    if level not in LEVELS:
        raise ValueError(f'level = {level!r}: one of {LEVELS}')
    L = _open_shell_half_period(ref, block, direction)[0]
    space = _OpenShellSpace(ref, block, [direction])
    generator = space.generator(direction) / np.sqrt(2.0)
    n = SOFT_MODE_GCM_POINTS
    grid = -L + (np.arange(n) + 0.5) * 2.0 * L / n
    orbitals = space.orbitals([generator], grid[:, None])
    N, H, S2, D = space.kernels.matrices(orbitals, [space.operator(direction)])
    to_reference = space.kernels.elements(np.broadcast_to(space.reference, orbitals.shape),
                                          orbitals)[0]
    blocks = _open_shell_hill_wheeler(N, H, _reflection_blocks(n))
    (E0, F0, r0), (E1, F1, r1) = blocks[0][0], blocks[1][0]
    k, order = _reference_level(E0, F0, E1, to_reference, level)
    j = order[0]
    q = np.linspace(0.0, L, SOFT_MODE_PATH_POINTS)
    path = space.orbitals([generator], q[:, None])
    scan = {}
    for cut, e, o in zip(SOFT_MODE_GCM_CUT_SCAN, blocks[0][1:], blocks[1][1:]):
        kc, oc = _reference_level(e[0], e[1], o[0], to_reference, level)
        scan[cut] = (float(o[0][oc[0]] - e[0][kc]), (e[2], o[2]))
    return dict(excitations=(E1 - E0[k])[order], couplings=np.abs(F1.T @ D @ F0[:, k])[order],
                ranks=(r0, r1),
                spin_square=(float(F0[:, k] @ S2 @ F0[:, k]), float(F1[:, j] @ S2 @ F1[:, j])),
                cut_scan=scan, q=q, V=space.kernels.elements(path, path)[1],
                reference_overlaps=(F0.T @ to_reference) ** 2, reference=k)


def open_shell_collision_levels(ref, block, plane):
    """The two-mode generator-coordinate model of a Krein collision of an open-shell block: the
    dict of `collision_levels` on the determinants exp(K(kappa)) Phi_0 of the pair's plane
    (kappa = r (cos phi_k e_1 + sin phi_k e_2) / sqrt2 on SOFT_MODE_COLLISION_DIRECTIONS directions
    with SOFT_MODE_GCM_POINTS midpoints of [-L_k, L_k] each, L_k = pi / (sqrt2 sigma_1) along it),
    with the kernels of `_OpenShellKernels` and the couplings D_m of `open_shell_path` along e_1 and
    e_2 (columns of plane, orthonormal), solved in the even and odd blocks of kappa -> -kappa: for
    'spin_flip' the spin rotation of pi about z (exact: the even block holds the reference's M_s
    parity, the odd one the other), for 'spin_conserving' a symmetry of the reference that takes
    the plane to minus itself, checked: |V(-kappa) - V(kappa)| on the grid above
    SOFT_MODE_OPEN_SHELL_PARITY_MAX of max |V| is refused with NotImplementedError."""
    if block not in BLOCKS:
        raise ValueError(f'block = {block!r}: one of {BLOCKS}')
    plane = np.asarray(plane, dtype=float)
    e1, e2 = plane[:, 0], plane[:, 1]
    space = _OpenShellSpace(ref, block, [e1, e2])
    generators = [space.generator(e) / np.sqrt(2.0) for e in (e1, e2)]
    n, m = SOFT_MODE_GCM_POINTS, SOFT_MODE_COLLISION_DIRECTIONS
    phi = np.pi * np.arange(m) / m
    half = np.array([_open_shell_half_period(ref, block, np.cos(f) * e1 + np.sin(f) * e2)[0]
                     for f in phi])
    radial = (np.arange(n) + 0.5) * 2.0 / n - 1.0
    coeffs = np.concatenate([np.outer(L * radial, (np.cos(f), np.sin(f)))
                             for f, L in zip(phi, half)])
    orbitals = space.orbitals(generators, coeffs)
    if block == 'spin_conserving':
        V = space.kernels.elements(orbitals, orbitals)[1]
        mirror = np.concatenate([k * n + n - 1 - np.arange(n) for k in range(m)])
        scale = float(np.max(np.abs(V)))
        parity = float(np.max(np.abs(V[mirror] - V))) / scale if scale > 0 else 0.0
        if parity > SOFT_MODE_OPEN_SHELL_PARITY_MAX:
            raise NotImplementedError(
                f'the spin-conserving plane of the collision is not even in kappa (|V(-kappa) - '
                f'V(kappa)| up to {parity:.1e} of max |V|, bar '
                f'{SOFT_MODE_OPEN_SHELL_PARITY_MAX:.0e}): no symmetry of the reference takes it '
                'to minus itself, and the parity-blocked model solves the even part only')
    N, H, S2, D1, D2 = space.kernels.matrices(orbitals, [space.operator(e1), space.operator(e2)])
    to_reference = space.kernels.elements(np.broadcast_to(space.reference, orbitals.shape),
                                          orbitals)[0]
    blocks = _open_shell_hill_wheeler(N, H, [np.kron(np.eye(m), r) for r in _reflection_blocks(n)])

    def poles(even, odd):
        (E0, F0, r0), (E1, F1, r1) = even, odd
        overlaps = (F0.T @ to_reference) ** 2
        k = int(np.argmax(overlaps))
        c = np.stack([F1.T @ D1 @ F0[:, k], F1.T @ D2 @ F0[:, k]], axis=1)
        c *= np.sign(c[np.arange(c.shape[0]), np.argmax(np.abs(c), axis=1)])[:, None]
        return E1 - E0[k], c, overlaps, k, (r0, r1)

    def strongest(dE, c, sign):
        weight = np.where(sign * dE > 0, np.sum(c ** 2, axis=1), -1.0)
        return float(dE[np.argmax(weight)]) if np.max(weight) > 0 else None

    dE, c, overlaps, k, ranks = poles(blocks[0][0], blocks[1][0])
    scan = {}
    for cut, even, odd in zip(SOFT_MODE_GCM_CUT_SCAN, blocks[0][1:], blocks[1][1:]):
        e, cc, _, _, r = poles(even, odd)
        scan[cut] = (strongest(e, cc, -1.0), strongest(e, cc, 1.0), r)
    F0, F1 = blocks[0][0][1], blocks[1][0][1]
    return dict(excitations=dE, couplings=c, even_levels=blocks[0][0][0], reference=k,
                reference_overlaps=overlaps, ranks=ranks,
                spin_square=(float(F0[:, k] @ S2 @ F0[:, k]), np.einsum('pm,pq,qm->m', F1, S2, F1)),
                cut_scan=scan, half_periods=half)


def _same_molecule(mol, other):
    """Whether two pyscf molecules are one for `noci_levels`: the same atoms (nuclear charges and
    coordinates), electrons and basis, compared shell by shell (atom, angular momentum, exponents
    and contraction coefficients, exactly) with the AO labels and the Cartesian or spherical form.
    AO labels alone do not tell two bases apart (6-31G and 3-21G on He2+ have the same four)."""
    if not (mol.natm == other.natm and mol.nbas == other.nbas and mol.cart == other.cart
            and mol.nelec == other.nelec
            and np.array_equal(mol.atom_charges(), other.atom_charges())
            and np.array_equal(mol.atom_coords(), other.atom_coords())
            and mol.ao_labels() == other.ao_labels()):
        return False
    return all(mol.bas_atom(i) == other.bas_atom(i) and mol.bas_angular(i) == other.bas_angular(i)
               and np.array_equal(mol.bas_exp(i), other.bas_exp(i))
               and np.array_equal(mol.bas_ctr_coeff(i), other.bas_ctr_coeff(i))
               for i in range(mol.nbas))


def noci_levels(solutions):
    """The non-orthogonal CI (Hill-Wheeler) between given UHF solutions of one molecule: the
    symmetry restoration of broken-symmetry solutions, such as a core hole localized on either atom
    of a homonuclear diatomic (a two-well resonance, whose g and u states it gives), or the mixing
    of two solutions of one occupation. A dict with

    energies            the states' energies (Ha, absolute), ascending
    states              their coefficients on the solutions (columns, f^T N f = 1)
    overlap             N between the solutions
    hamiltonian         H between the solutions (Ha, absolute)
    solution_energies   each solution's own energy, H_kk / N_kk
    spin_square         <S^2> of each state
    rank                the states the canonical orthogonalization keeps (overlap eigenvalues above
                        SOFT_MODE_GCM_OVERLAP_CUT times the largest)
    numerical_flags     'soft_mode_noci_rank' when rank is below the number of solutions (two of
                        them are, to that cut, one determinant; SoftModeNOCIShort)

    Each solution is checked as `open_shell_reference` checks a reference (a converged UHF, 0/1
    occupations, n_alpha >= n_beta, stationary); all must share the molecule (`_same_molecule`:
    atoms, geometry, electrons and basis, checked before any work) and (n_alpha, n_beta), else
    ValueError; fewer than two are refused. The kernels are those of
    `_OpenShellKernels` on each spin's span of every solution's occupied orbitals (the first
    solution's occupied orbitals and the singular-value span of the others' parts in its virtual
    space, exact)."""
    solutions = list(solutions)
    if len(solutions) < 2:
        raise ValueError(f'{len(solutions)} solutions: the non-orthogonal CI needs two or more')
    first = solutions[0].mol
    if not all(_same_molecule(first, mf.mol) for mf in solutions[1:]):
        raise ValueError('the solutions do not share one molecule (atoms, geometry, electrons and '
                         'basis, shell by shell)')
    refs = [open_shell_reference(mf)[0] for mf in solutions]
    if any(_open_shell_counts(mf) != _open_shell_counts(refs[0]) for mf in refs[1:]):
        raise ValueError('the solutions do not share their occupation counts (n_alpha, n_beta)')
    n = _open_shell_counts(refs[0])
    S = first.intor_symmetric('int1e_ovlp')
    C0 = np.asarray(refs[0].mo_coeff)
    bases = []
    for s in (0, 1):
        occ, vir = C0[s][:, :n[s]], C0[s][:, n[s]:]
        outside = np.hstack([vir.T @ S @ np.asarray(mf.mo_coeff)[s][:, :n[s]] for mf in refs[1:]])
        U = (np.linalg.svd(outside, full_matrices=False)[0] if outside.size
             else np.zeros((vir.shape[1], 0)))
        bases.append(np.hstack([occ, vir @ U]))
    kernels = _OpenShellKernels(first, bases, refs[0].get_hcore(), refs[0].e_tot)
    ma, mb = kernels.m
    dets = np.zeros((len(refs), ma + mb, n[0] + n[1]))
    for p, mf in enumerate(refs):
        C = np.asarray(mf.mo_coeff)
        dets[p, :ma, :n[0]] = bases[0].T @ S @ C[0][:, :n[0]]
        dets[p, ma:, n[0]:] = bases[1].T @ S @ C[1][:, :n[1]]
    N, H, S2 = kernels.matrices(dets)
    E, F, rank = _open_shell_hill_wheeler(N, H, [np.eye(len(refs))])[0][0]
    flags = []
    with collecting(flags):
        if rank < len(refs):
            flag('soft_mode_noci_rank',
                 f'the non-orthogonal CI of {len(refs)} solutions keeps {rank} states: the overlap '
                 f'has eigenvalues at or below {SOFT_MODE_GCM_OVERLAP_CUT:.0e} of its largest, so '
                 'that two of the solutions are, to that cut, one determinant; the states kept are '
                 'returned', SoftModeNOCIShort)
    e_ref = refs[0].e_tot
    return dict(energies=E + e_ref, states=F, overlap=N, hamiltonian=H + e_ref * N,
                solution_energies=np.diag(H) / np.diag(N) + e_ref,
                spin_square=np.einsum('pm,pq,qm->m', F, S2, F), rank=rank,
                numerical_flags=gathered(flags))


def _spin_rotation_split(A, B, g, sectors):
    """(A_r, B_r, coordinates, momenta, pole): the spin-flip block with its spin-rotation mode
    taken out. With D = +1 on the first `sectors` pairs and -1 on the rest, (A + B) g = 0 and
    (A - B) D g = 0, and the other modes live where kappa is orthogonal to D g and pi to g (the
    symplectic complement of the pair). For Q an orthonormal basis of g's complement and
    P = 1 - g (D g)^T / (g^T D g), the RPA equations of (A_r, B_r) = (Q^T A Q, Q^T B Q) are the
    block's exactly, with X + Y = P Q (X_r + Y_r) and X - Y = Q (X_r - Y_r) (coordinates P Q,
    momenta Q), and the spin-rotation mode is the pole at Omega = 0 with X + Y = g and X - Y = D g
    over (g^T D g)^1/2 (X'X - Y'Y = 1): the M_s - 1 component of the reference's multiplet."""
    n = g.size
    D = np.r_[np.ones(sectors), -np.ones(n - sectors)]
    norm = float(g @ (D * g))
    w, v = eigh_sym(np.eye(n) - np.outer(g, g) / (g @ g))
    Q = v[:, w > 0.5]
    coordinates = (np.eye(n) - np.outer(g, D * g) / norm) @ Q
    X = 0.5 * (g + D * g) / np.sqrt(norm)
    Y = 0.5 * (g - D * g) / np.sqrt(norm)
    return Q.T @ A @ Q, Q.T @ B @ Q, coordinates, Q, (X, Y, norm)


def soft_mode_open_shell(mf, block='spin_conserving', model='1d', collisions='refuse'):
    """The TDHF spectrum of one block of a UHF reference (`uhf_tdhf_matrices`) with its soft modes
    replaced by their collective levels, split sign-aware (`krein_split`; `krein_pairs` with
    collisions='gcm'), the open-shell counterpart of `soft_mode_singlets`: a dict with

    omega, X, Y        the spectrum (Ha) and its vectors in the block's layout: the regular modes'
                       positive-norm members (X'X - Y'Y = 1, omega < 0 for a negative transition),
                       then for the spin-flip block of a reference with n_alpha > n_beta the
                       spin-rotation pole at omega = 0 (the reference multiplet's M_s - 1
                       component, column `spin_rotation['column']`), then the soft modes' and the
                       colliding pairs' collective poles (`collective_modes`, `collision_modes`)
    signature          per column, the sign of X'X - Y'Y
    block              the block
    orbitals           (alpha, beta): mf's orbital indices in the pair layout's order
    omega2             the block's Omega^2 (Ha^2), ascending, the spin-rotation mode left out
                       (complex with collisions='gcm')
    omega_tda          the block's TDA energies, eigenvalues of A (Ha), the spin-rotation
                       direction included
    omega_tda_soft     the TDA energy of the root that overlaps the lowest soft direction most
                       (None without a soft mode)
    soft               per soft mode, as in `soft_mode_singlets`: index (in omega2), omega2, mass,
                       stiffness, signature, level (in the GCM 'bottom', 'top' or 'overlap', the
                       last for M > 0 on a block whose A - B is indefinite), half_period
                       (pi / (sqrt2 sigma_1)), sigma
                       (the singular values of d's spin blocks together,
                       `_open_shell_half_period`), path ('straight'), v_min, q_min, excitation,
                       krein_gap, and excitation_walls
                       and edge_weight (1D) or coupling, ranks, spin_square, cut_scan,
                       reference_overlaps and reference (GCM)
    pairs              with collisions='gcm', per colliding pair as in `soft_mode_triplets`
    spin_rotation      for the spin-flip block of a reference with n_alpha > n_beta: column,
                       residual |(A + B) g| / |g| (Ha) and norm g^T D g (= n_alpha - n_beta) of
                       `spin_rotation_direction`; else None
    numerical_flags    those of `soft_mode_singlets` with reference='excited' and
                       collisions='gcm' ('soft_mode_edge_weight', 'soft_modes_coupled',
                       'soft_mode_gcm_degenerate_pairs', here when two pairs of one spin block of
                       d are as strong as its leading pair, 'soft_mode_gcm_few_states', here
                       below one spin-orbital pair's exact span, one even and one odd state (a
                       single-spin rank-one direction spans exactly that), or with fewer even
                       states than odd ones, 'soft_mode_krein_neighbour',
                       'soft_mode_reference_overlap', 'soft_mode_reference_tie',
                       'soft_mode_collision_reference', 'soft_mode_collision_few_states'), and
                       'soft_mode_complex_instability' (`SoftModeExcitedShort`): a soft mode with
                       M < 0 and mu > 0 on a reference aufbau in both spins

    The reference is any stationary UHF (`open_shell_reference`: an aufbau one, a broken-symmetry
    one, or a maximum-overlap core hole), every mode signed by its Krein signature: a soft mode of
    M > 0 is quantized by H = P^2 / 2M + V(Q), one of M < 0 by -H, the reference its bottom or top
    level (in the GCM picked as in `soft_mode_triplets` with reference='excited'). The inversion
    of a mode with M < 0 and mu > 0 is validated on excited references only (the excited-state
    instability of H2's sigma_u^2): on a reference aufbau in both spins (each spin's occupied
    orbitals below its virtual ones, `_aufbau_in_both_spins`; O2/STO-3G's default triplet UHF)
    the mode is a complex instability of a ground-state determinant, and the levels kept are
    flagged ('soft_mode_complex_instability'; the occupation is only a sign of a ground-state
    determinant: an excited closed shell aufbau in its own Fock spectrum is flagged too, H2's
    sigma_u^2 as a UHF from 1.34 A on, where the inversion is the validated one). Each mode is
    quantized
    along kappa = Q d / sqrt2 (`open_shell_path`; d the mode's direction, in the spin-flip
    block taken orthogonal to the spin rotation g, which leaves V''(0) = mu) in the 1D model or the
    GCM (`open_shell_gcm_levels`), and a Krein collision with collisions='gcm' by the two-mode GCM
    on its plane (`open_shell_collision_levels`). In the spin-conserving block the determinants
    stay UHF ones (collinear paths); in the spin-flip block they are GHF ones (non-collinear
    paths), and the global spin rotation, a zero mode of both A + B and A - B there, is taken out
    exactly (`_spin_rotation_split`) and kept as its pole.

    block is 'spin_conserving' or 'spin_flip' (`BLOCKS`); model is '1d' or 'gcm' ('3d', the three
    triplet components of a closed shell, is refused); collisions as in `soft_mode_triplets`.
    Refused before any work: an unknown option (ValueError), what `open_shell_reference` refuses
    without integrals (`_open_shell_occupation`), and the spin-flip block of a reference with
    n_alpha = n_beta (NotImplementedError: there each spin-flip mode comes with its M_s partner,
    degenerate and of the other Krein signature, and a broken-symmetry M_s = 0 UHF has a
    spin-rotation mode of null norm); then, after one Fock build, an occupation that is not
    stationary (`open_shell_reference`); refused after the split: a
    soft direction or a collision plane of the spin-conserving block whose path is not even in Q
    (|V(-Q) - V(Q)| above SOFT_MODE_OPEN_SHELL_PARITY_MAX of max |V|, NotImplementedError), a
    complex Omega^2 with collisions='refuse' and an A - B with a zero eigenvalue
    (`LinearResponse.casida.CasidaInstability`), a degenerate quartet (NotImplementedError)."""
    if block not in BLOCKS:
        raise ValueError(f'block = {block!r}: one of {BLOCKS}')
    if model not in MODELS:
        raise ValueError(f'model = {model!r}: one of {MODELS}')
    if model == '3d':
        raise ValueError("model = '3d': the 3D model quantizes the three triplet components of a "
                         "closed shell; an open-shell soft mode has one ('1d' or 'gcm')")
    if collisions not in COLLISIONS:
        raise ValueError(f'collisions = {collisions!r}: one of {COLLISIONS}')
    counts = _open_shell_occupation(mf)[1]
    if block == 'spin_flip' and counts[0] == counts[1]:
        raise NotImplementedError(
            "block = 'spin_flip' with n_alpha = n_beta: there each spin-flip mode comes with its "
            'M_s partner, degenerate and of the other Krein signature (on a closed shell the '
            "triplet's M_s = +-1 components; use soft_mode_triplets), and a broken-symmetry "
            'M_s = 0 UHF has a spin-rotation mode of null norm')
    ref, orders = open_shell_reference(mf)
    n_alpha, n_beta = _open_shell_counts(ref)
    A, B = uhf_tdhf_matrices(ref, block)
    rotation = None
    if block == 'spin_flip':
        g = spin_rotation_direction(ref)
        sectors = n_alpha * (np.asarray(ref.mo_occ).shape[1] - n_beta)
        Ar, Br, coordinates, momenta, (gx, gy, norm) = _spin_rotation_split(A, B, g, sectors)
        rotation = dict(residual=float(np.linalg.norm((A + B) @ g) / np.linalg.norm(g)),
                        norm=norm)
    else:
        Ar, Br, coordinates, momenta = A, B, None, None

    def full(vectors, basis):
        return vectors if basis is None else basis @ vectors

    split = krein_pairs(Ar, Br) if collisions == 'gcm' else krein_split(Ar, Br)
    signature = split['signature']
    indefinite, aufbau = bool(np.any(signature != 1)), _aufbau_in_both_spins(ref)
    w_tda, u_tda = eigh_sym(A)
    omega, xr, yr = split['regular']
    parts = [(omega, 0.5 * (full(xr + yr, coordinates) + full(xr - yr, momenta)),
              0.5 * (full(xr + yr, coordinates) - full(xr - yr, momenta)))]
    if rotation is not None:
        rotation['column'] = omega.size
        parts.append((np.zeros(1), gx[:, None], gy[:, None]))
    flags, soft_info, pair_info = [], [], []
    label0 = f'soft {block} mode'
    with collecting(flags):
        for s in split['soft']:
            mode = split['modes'][s]
            M = mode['mass']
            sign = 1.0 if M > 0 else -1.0
            d = full(mode['direction'], momenta)
            label = f'{label0} {s}'
            L, sigma = _open_shell_half_period(ref, block, d)
            if block == 'spin_conserving':
                parity = _open_shell_parity(ref, block, d, L)
                if parity > SOFT_MODE_OPEN_SHELL_PARITY_MAX:
                    raise NotImplementedError(
                        f'{label}: its path is not even in Q (|V(-Q) - V(Q)| up to {parity:.1e} of '
                        f'max |V|, bar {SOFT_MODE_OPEN_SHELL_PARITY_MAX:.0e}): no symmetry of the '
                        'reference takes d to -d, and the parity-blocked collective models solve '
                        'the even part of V only')
            info = dict(index=s, omega2=mode['omega2'], mass=M, stiffness=mode['stiffness'],
                        half_period=L, sigma=sigma, path='straight', signature=int(sign),
                        level='bottom' if sign > 0 else 'top')
            if aufbau and M < 0 and mode['stiffness'] > 0:
                lowest = float(np.linalg.eigvalsh(Ar + Br)[0])
                flag('soft_mode_complex_instability',
                     f'{label}: M = {M:+.3e} Ha^-1 and mu = {mode["stiffness"]:+.3e} Ha (Omega^2 '
                     '< 0) on a reference aufbau in both spins: a complex instability of a '
                     'ground-state determinant (A - B indefinite along d), which the inversion of '
                     'a negative mass, validated on excited references, does not describe'
                     + (f'; A + B is indefinite too (lowest eigenvalue {lowest:+.3e} Ha), a real '
                        'instability of the block, whose solution lies lower' if lowest < 0
                        else '') + '; the levels are kept', SoftModeExcitedShort)
            if model == 'gcm':
                level = 'top' if sign < 0 else ('overlap' if indefinite else 'bottom')
                gcm = open_shell_gcm_levels(ref, block, d, level)
                ranks, V = gcm['ranks'], sign * gcm['V']
                info.update(v_min=float(V.min()), q_min=float(gcm['q'][np.argmin(V)]),
                            excitation=float(gcm['excitations'][0]),
                            coupling=float(gcm['couplings'][0]), ranks=ranks,
                            spin_square=gcm['spin_square'], cut_scan=gcm['cut_scan'],
                            reference_overlaps=gcm['reference_overlaps'],
                            reference=gcm['reference'], level=level)
                for (s_from, s_to), b in _open_shell_pairs(ref, block, d).items():
                    lead = np.linalg.svd(b, compute_uv=False) if b.size else np.zeros(0)
                    if lead.size > 1 and lead[1] >= SOFT_MODE_GCM_PAIR_RATIO * sigma[0]:
                        flag('soft_mode_gcm_degenerate_pairs',
                             f'{label}: two pairs of its spin {s_from} -> {s_to} block are as '
                             f'strong as its leading one (sigma = {lead[0]:.6f}, {lead[1]:.6f}; '
                             f'{sigma[0]:.6f} the largest), so that the straight path turns them '
                             'together; there the generator-coordinate '
                             'model is not size-consistent (<S^2> = '
                             f'{gcm["spin_square"][0]:.3f} in its reference state)',
                             SoftModeGCMShort)
                if ranks[0] < 1 or ranks[1] < 1 or ranks[0] < ranks[1]:
                    flag('soft_mode_gcm_few_states',
                         f'{label}: the canonical orthogonalization keeps {ranks[0]} even and '
                         f'{ranks[1]} odd states, short of one spin-orbital pair\'s exact span '
                         '(one even, one odd) or with fewer even states than odd ones',
                         SoftModeGCMShort)
                _reference_flags(label, level, M, gcm['reference_overlaps'], gcm['reference'])
                dE, cm = gcm['excitations'], gcm['couplings']
            else:
                q, V, D = open_shell_path(ref, block, d, L)
                levels, cm, edge = collective_levels(abs(M), q, sign * V, D, L, '1d')
                walls = collective_levels(abs(M), q, sign * V, D, L, '1d', 'walls')[0]
                moved = abs(walls[0] - levels[0])
                info.update(v_min=float((sign * V).min()), q_min=float(q[np.argmin(sign * V)]),
                            excitation=float(sign * levels[0]),
                            excitation_walls=float(sign * walls[0]), edge_weight=edge)
                if edge > SOFT_MODE_EDGE_WEIGHT_MAX or moved > SOFT_MODE_BOUNDARY_SHIFT_MAX:
                    flag('soft_mode_edge_weight',
                         f'{label}: the collective reference state has {edge:.1e} of its weight '
                         f'near the ends of its path, and its first transition moves by '
                         f'{moved:.1e} Ha when the periodic path is cut by walls; its levels '
                         'depend on how the path closes', SoftModeEdgeWeight)
                dE = sign * levels
            info['krein_gap'] = _krein_gap(label, mode, signature, split['omega2'], s)
            soft_info.append(info)
            parts.append(collective_modes(dE, cm, M, full(mode['direction'], coordinates),
                                          full(mode['conjugate'], momenta)))
        for pair in split.get('pairs', ()):
            label = f'{block} collision of modes {pair["index"]} ({pair["kind"]})'
            plane = full(pair['plane'], momenta)
            gcm = open_shell_collision_levels(ref, block, plane)
            ranks, kept, overlaps = gcm['ranks'], gcm['reference'], gcm['reference_overlaps']
            dE, c = gcm['excitations'], gcm['couplings']
            weight = np.sum(c ** 2, axis=1)
            below, above = np.where(dE < 0, weight, -1.0), np.where(dE > 0, weight, -1.0)
            nearest = np.argsort(-np.linalg.norm(u_tda.T @ plane, axis=1))[:2]
            pair_info.append(dict(
                index=list(pair['index']), omega2=pair['omega2'], kind=pair['kind'],
                gap=pair['gap'], masses=pair['masses'], half_periods=gcm['half_periods'],
                deexcitation=float(dE[np.argmax(below)]) if np.max(below) > 0 else None,
                excitation=float(dE[np.argmax(above)]) if np.max(above) > 0 else None,
                excitations=dE, couplings=c, reference_overlap=float(overlaps[kept]),
                reference_overlaps=overlaps, reference_shift=float(gcm['even_levels'][kept]),
                ranks=ranks, spin_square=gcm['spin_square'], cut_scan=gcm['cut_scan'],
                omega_tda=np.sort(w_tda[nearest])))
            if overlaps[kept] < SOFT_MODE_COLLISION_OVERLAP_MIN:
                second = float(np.sort(overlaps)[-2]) if overlaps.size > 1 else 0.0
                flag('soft_mode_collision_reference',
                     f'{label}: the even state kept as the reference\'s continuation, the one of '
                     f'largest overlap with the reference determinant, holds {overlaps[kept]:.3f} '
                     f'of it (another even state {second:.3f}): the reference shares its '
                     'determinant with a doubly excited configuration of the plane; the levels are '
                     'kept', SoftModeCollisionShort)
            if ranks[0] < 4 or ranks[1] < 2:
                flag('soft_mode_collision_few_states',
                     f'{label}: the canonical orthogonalization keeps {ranks[0]} even and '
                     f'{ranks[1]} odd states, short of the plane\'s span for one occupied orbital '
                     '(four even, two odd)', SoftModeCollisionShort)
            parts.append(collision_modes(dE, c, full(pair['plane'], coordinates),
                                         full(pair['conjugates'], momenta)))
        groups = _degenerate_groups(split['soft'], np.real(split['omega2']))
        if len(groups) + len(pair_info) > 1:
            flag('soft_modes_coupled',
                 f'{len(groups)} sets of soft {block} modes ({groups}) and {len(pair_info)} '
                 f'colliding pairs ({[p["index"] for p in pair_info]}): each has its own '
                 'collective model, and the couplings between them are left out',
                 SoftModesCoupled)
    X, Y = np.hstack([p[1] for p in parts]), np.hstack([p[2] for p in parts])
    d0 = (full(split['modes'][split['soft'][0]]['direction'], momenta) if split['soft']
          else None)
    out = dict(omega=np.concatenate([p[0] for p in parts]), X=X, Y=Y,
               signature=np.where(np.sum(X * X - Y * Y, axis=0) < 0, -1, 1), block=block,
               orbitals=orders, omega2=split['omega2'], omega_tda=w_tda,
               omega_tda_soft=(None if d0 is None
                               else float(w_tda[np.argmax(np.abs(u_tda.T @ d0))])),
               soft=soft_info, spin_rotation=rotation, numerical_flags=gathered(flags))
    if collisions == 'gcm':
        out['pairs'] = pair_info
    return out
