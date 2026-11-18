# Soft-mode triplet instabilities

At the RHF→UHF (triplet) instability of a closed shell, the lowest eigenvalue Ω_s² of the TDHF
triplet problem, A = Δε − (ij|ab) and B = −(ib|ja), goes through zero. Any self-energy that sums
the triplet ladder over its poles then diverges as 1/Ω_s. In this code those are the triplet sums of
PSD2, PSD4, PSD7 and PSD9 on TDHF screening (`GW.self_energy.TRIPLET_VERTEX_MODES`):
- they lose their quasiparticle before the instability;
- past it, the dense Casida solve refuses the reference (`CasidaInstability`).

The instability is not limited to stretched bonds. In cc-pVDZ at their MP2 minima it occurs for:
- planar ethylene (just past it, Ω_T = 0.94i eV);
- butadiene, hexatriene, octatetraene and benzene;
- O3, where TDA is unstable too.

N2 and H2CO are stable but soft (Ω_T = 1.9 and 1.3 eV).

The consistent Hedin TDHF scheme (`Sigma_TDHF@L_TDHF`, GWΓ∞ on TDHF, see
[Second-order BSE kernels](second-order-bse.md)) carries no triplet pole at all: its exchange-ladder
poles cancel against W, and it runs smoothly through the instability. The soft-mode treatment is for
the self-energies that do carry the triplet ladder.

The idea comes from the Sachdev–Ye–Kitaev model. There, the four-point ladder diverges at its
reparametrization modes, and Maldacena and Stanford (arXiv:1604.07818) take those modes out of the
ladder and give them an action of their own. Here each soft triplet mode gets:
- a collective coordinate Q along the RHF→UHF orbital rotation;
- its Thouless–Valatin mass;
- the Hartree–Fock energy along the rotation as its potential.

Its quantum levels then go back into the ladder as poles.

## Running it

```python
from pyscf import gto, scf
from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.LinearResponse.soft_mode import soft_mode_triplets
from src.SingleReference.GW.soft_mode_qp import soft_mode_qp

mol = gto.M(atom='H 0 0 0; H 0 0 1.5', basis='cc-pvdz')
mf = scf.RHF(mol).run()

rec = soft_mode_triplets(mf, model='1d')        # the triplet spectrum, soft mode quantized
print(rec['soft'][0]['excitation'] * HARTREE_TO_EV)   # its first level: 3.53 eV (FCI 2.70)

qp = soft_mode_qp(mf, selfenergy='PSD2')        # PSD2@TDHF on that spectrum, HOMO
print(-qp['eps'][0] * HARTREE_TO_EV, qp['Z'][0])      # IP 14.67 eV (FCI 13.16), Z 0.82

rec = soft_mode_triplets(mf, path='relaxed')    # the same along the relaxed path
print(rec['soft'][0]['excitation'] * HARTREE_TO_EV)   # 3.36 eV

rec = soft_mode_triplets(mf, model='gcm')       # the generator-coordinate model (opt-in)
print(rec['soft'][0]['excitation'] * HARTREE_TO_EV)   # 2.88 eV
```

## What is computed

Steps, for each soft mode s:

| step | quantity |
|---|---|
| split | T = (A − B)^½ (A + B) (A − B)^½ = Σ_n Ω_n² t_n t_nᵀ, Ω_n² of either sign |
| direction | d ∝ (A − B)^½ t_s, mass M = dᵀ(A − B)⁻¹d, stiffness μ = dᵀ(A + B)d = M Ω_s², conjugate e = (A − B)⁻¹d / M |
| path | V(Q) = E_HF[α orbitals rotated by exp(K), β by exp(−K)] − E_RHF, with K_ai = Q d_ia / 2, so that V''(0) = μ |
| coupling | D(Q) = 2 Σ_ia d_ia P^α_ai(Q), which is Q to first order and sin Q for one occupied orbital |
| levels | H = P²/2M + V(Q): 1D (the collinear M_s = 0 component) or 3D (the three triplet components, radial) |
| poles | each level m with ⟨m\|D\|0⟩ ≠ 0 becomes a ladder pole at ΔE_m, with X + Y = √2 ⟨m\|D\|0⟩ d and X − Y = √2 M ΔE_m ⟨m\|D\|0⟩ e |

For a harmonic V the levels are exactly the removed TDHF eigenpair, in the house's Casida layout.
Past the instability V is a double well, and its levels stay finite.

Read as the n_occ × n_virt matrix d_ia, d turns pair k of its singular-value decomposition by
σ_k Q/2. At Q = ±π/σ_1 the leading pair has turned by ±90°, which is one and the same determinant,
so the 1D model is periodic on [−π/σ_1, π/σ_1):
- exact when d has rank one (H2);
- exact when the leading pairs are degenerate (N2's two π pairs);
- approximate otherwise.

The same model with walls at ±π/σ_1 measures how much the levels owe to that closure.

The soft set is:
- the lowest mode;
- every mode below `SOFT_MODE_OMEGA_CUT` (0.05 Ha, negative Ω² included);
- the rest of their degenerate sets.

Each soft set gets its own model along its own path; the couplings between them are left out.

## Options

| function | option | values |
|---|---|---|
| `soft_mode_triplets(mf, model, path)` | model | `'1d'` (default), `'3d'`, or `'gcm'` (straight path only, see [The generator-coordinate model](#the-generator-coordinate-model)) |
| | path | `'straight'` (default) or `'relaxed'` (1D only, see [The relaxed path](#the-relaxed-path)) |
| `soft_mode_qp(mf, selfenergy, states, model, path)` | selfenergy | `'PSD2'` (default), `'PSD4'`, `'PSD7'`, `'PSD9'` |
| | states | orbital indices; the HOMO by default |
| | model | `'1d'`, `'3d'`, `'gcm'`, or `None` for the plain TDHF triplets (the house route's spectrum) |
| | path | as for `soft_mode_triplets` |

The reference must be a converged closed-shell RHF with exact integrals; anything else is refused
before any work, as are an unknown path and the relaxed path with any model but `'1d'` (with
`'gcm'` as `NotImplementedError`: the GCM over the relaxed path is future work). An A − B
that is not positive definite (90°-twisted ethylene) leaves the mass undefined and is refused with
`CasidaInstability`.

## Results and flags

`soft_mode_triplets` returns:
- the replaced spectrum (`omega`, `X`, `Y`);
- every TDHF Ω² and the TDA energies;
- the TDA root of the soft mode;
- per soft mode, a record with its mass, stiffness, half-period, singular values, well depth,
  first collective excitation (also with walls) and edge weight.

`soft_mode_qp` returns the quasiparticle energies and pole strengths, from the one root finder.

Both return `numerical_flags`:
- `'soft_mode_edge_weight'`: the levels depend on how the path closes;
- `'soft_modes_coupled'`: more than one soft set;
- with `path='relaxed'`, `'soft_mode_path_not_converged'` (a constrained minimisation stopped
  short), `'soft_mode_path_frozen'` (the relaxation was frozen before the antipode),
  `'soft_mode_path_branch'` (after the hold rounds a sample still came from a fresh start, or a
  negative curvature across the constraint plane remains) and `'soft_mode_path_closure'` (the loop
  does not close at its antipode);
- with `model='gcm'`, `'soft_mode_gcm_degenerate_pairs'` (the two leading pairs of the soft
  direction are degenerate) and `'soft_mode_gcm_few_states'` (the canonical orthogonalization kept
  fewer states than one pair's span, two even and one odd, or fewer even states than odd ones);
- the root finder's own flags.

With `path='relaxed'` each soft mode's record also holds its antipode Q*, M(Q*)/M, D(Q*), the
overlap of the α determinants at ±Q*, whether every constrained minimisation converged, the fresh
starts tried and the samples taken from one, the TDHF modes held for a saddle, the lowest
curvature probed, where the relaxation was frozen (or `None`) and the number of UHF energy
evaluations; its `v_min` is then the relaxed well depth. All the flags are raised by
`soft_mode_triplets` (and so by `soft_mode_qp`) from the records of `relaxed_path` and
`gcm_levels`. With `model='gcm'` it holds, in place of the walls'
excitation and the edge weight, the first level's coupling c₁, the ranks of the two parity blocks,
⟨S²⟩ of the even ground state and of the lowest odd state, and the excitation and ranks at the
cuts of `SOFT_MODE_GCM_CUT_SCAN`.

See [Warnings, flags and refusals](../warnings.md).

## The relaxed path

`path='relaxed'` keeps the split, the coupling D and the pole construction, and changes the path
and the mass. At each Q the UHF energy is minimised over the full real triplet rotation κ, α
orbitals rotated by exp(K(κ)) and β by exp(−K(κ)) with K_ai = κ_ia/2, in the Thouless–Valatin form:

```
κ(Q) = argmin { E(κ) : eᵀκ = Q,  e_tᵀκ = 0 for the other soft modes t }
V(Q) = E(κ(Q)) − E_RHF,      M(Q) = κ'(Q)ᵀ (A − B)⁻¹ κ'(Q)   (the RHF A − B)
H    = −½ ∂_Q M(Q)⁻¹ ∂_Q + V(Q),   periodic on [−Q*, Q*)
```

Q* is the antipode of the relaxed loop, where the leading pair of κ_ia has turned by 90°
(its largest singular value is π; L = π/σ₁ on the straight path). The poles keep the Q = 0 mass
and e, so that X − Y = ΔE_m (A − B)⁻¹ (X + Y), the TDHF relation with Ω replaced by ΔE_m. That this
stays right for a coordinate-dependent mass is argued (the velocity identity
⟨m|½{M(Q)⁻¹, P}|0⟩ = i ΔE_m ⟨m|Q|0⟩ holds for any M(Q)), not checked.

**The harmonic limit stays TDHF's.** Since (A + B) d = Ω_s² M e, the minimum of the quadratic
energy ½κᵀ(A + B)κ on eᵀκ = Q is κ = Q d, so V''(0) = μ, M(0) = M and the harmonic level is Ω_s.
The constraint dᵀκ = Q would not keep it: its stiffness 1/dᵀ(A + B)⁻¹d lies below μ.

**How the path is followed** (`relaxed_path`):
- samples Q_k = kL/80, from 0 to four samples past Q* (three past the first sample beyond it);
- at each, L-BFGS in variables whitened by Zᵀ(A + B)Z (Z spans the rotations the constraints
  leave free), warm-started by the quadratic extrapolation of the last three minima, every step
  cut to |Δκ| ≤ 0.1. The plane eᵀκ = Q also holds wrap-around points, where pair angles reach 180°
  and V = 0;
- a minimum whose step jumps (more than twice the last or the extrapolated step, plus 0.05) has
  left the branch, and is tried again from κ = Q d. If that jumps too, the relaxation is frozen
  (κ continued along d, flagged `'soft_mode_path_frozen'`) before the antipode, or the path ends
  after it;
- Q* is the root of σ₁(κ(Q)) = π on the cubic spline of κ, the inertia the spline's tangent, and
  M(Q*) comes from fresh minima at Q* ∓ L/640 when the path ends less than two samples past Q*;
- every other soft mode is held at zero, as on the straight path each has its own model;
- the samples are constrained stationary points, and a path can turn into a constrained saddle
  along a stiffer mode of another symmetry (below). So the lowest curvature across the plane is
  probed at every sample taken from a fresh start and at the last relaxed sample up to the
  antipode: Lanczos (20 vectors) on the Hessian whitened by the RHF's, from central differences of
  the analytic gradient. The TDHF triplet modes that carry a negative curvature (a fifth of it or
  more, with their degenerate partners) are held at zero too, and the path is followed again, up
  to three times. A sample still taken from a fresh start, or a negative curvature still found,
  is flagged `'soft_mode_path_branch'`; a loop whose α determinants at ±Q* overlap by less than
  0.999, or whose D(Q*) exceeds 10⁻³, `'soft_mode_path_closure'`.

The prototype and its audit fitted a parabola through the 90° cusp of the largest principal angle
to place Q*, froze the relaxation at any jump, and took the inertia at a truncated antipode from a
one-sided spline; the three are replaced as above.

**Measured** (eV; errors against FCI for H2/cc-pVDZ, recorded EOM-CCSD for N2, H2CO, H2O and
ethylene, and spin-flip EOM-CCSD from the ADA run for O3 and ethylene twisted by 45–70°):

| system | reference | straight | relaxed | Q*/L | M(Q*)/M |
|---|---|---|---|---|---|
| H2 0.75 Å | 10.543 | −0.044 | −0.110 | 1.006 | 1.000 |
| H2 1.0 Å | 7.164 | +0.394 | +0.267 | 1.009 | 1.006 |
| H2 1.2104 Å | 4.907 | +0.672 | +0.516 | 1.011 | 1.011 |
| H2 1.5 Å | 2.704 | +0.826 | +0.658 | 1.012 | 1.012 |
| H2 2.0 Å | 0.792 | +0.574 | +0.425 | 1.013 | 1.008 |
| H2 3.0 Å | 0.042 | +0.012 | −0.009 | 1.019 | 1.003 |
| N2 | 7.160 | −0.482 | −0.589 | 1.000 | 1.022 |
| H2CO | 5.910 | +0.691 | +0.538 | 0.999 | 1.043 |
| H2O | 7.438 | +1.000 | +0.961 | 0.998 | 1.018 |
| ethylene | 4.417 | +0.518 | +0.369 | 0.997 | 1.050 |
| O3 | 1.513 | +0.781 | +0.615 | 1.006 | 1.013 |
| ethylene 45° | 2.757 | +0.406 | +0.058 | 1.008 | 1.436 |
| ethylene 60° | 1.693 | +0.371 | −0.007 | 1.029 | 1.131 |
| ethylene 70° | 0.974 | +0.302 | −0.079 | 1.047 | 1.061 |

| group (MSE / MAE / max) | straight | relaxed |
|---|---|---|
| H2 against FCI (6) | +0.406 / 0.420 / 0.826 | +0.291 / 0.331 / 0.658 |
| small molecules against EOM-CCSD (4) | +0.432 / 0.673 / 1.000 | +0.320 / 0.614 / 0.961 |
| diradicaloids against spin-flip EOM-CCSD (4) | +0.465 / 0.465 / 0.781 | +0.147 / 0.190 / 0.615 |
| all 14 | +0.430 / 0.505 / 1.000 | +0.258 / 0.372 / 0.961 |
| ethylene twisted by 10–40° (4, EOM-CCSD) | +0.520 / 0.520 / 0.523 | +0.227 / 0.227 / 0.288 |
| ethylene's C=C at 1.22, 1.30, 1.36, 1.44 Å (4, EOM-CCSD) | +0.501 / 0.501 / 0.569 | +0.350 / 0.350 / 0.422 |

These reproduce the tweak study's audited values to 2·10⁻⁵ eV on every small system, N2 included
(the audit's continuous symmetric branch, 6.57186 eV; see below). The relaxed loop closes on
itself: the overlap of the α determinants at ±Q* is 1 to 4·10⁻¹³ and |D(Q*)| below 4·10⁻⁸ for
every system, where the straight path's overlap at ±L is 0.875 to 0.974 for the molecules. The
relaxed path lowers every level, so it helps where the straight path is too high (the twisted
ethylenes most) and makes N2 and H2 at 0.75 Å, already too low, lower still. A path costs 200 to
470 UHF energies and gradients with its probes (1190 for H2CO and 2440 for N2, with their hold
rounds), 0.2 to 2.5 s per mode on four threads.

Two findings while implementing it:
- **The relaxed samples are constrained stationary points, not always minima.** H2CO's path turns
  into a constrained saddle from 0.9 L along its second triplet (3.37 eV; the lowest eigenvalue of
  ZᵀHZ is −0.006 Ha there, −0.071 Ha at L), N2's from 0.69 L along its ³Δu pair (5.01 eV; −0.11 Ha
  at 0.70 L). Along those directions a warm start falls onto the wrap-around points (V = 0), while
  the fresh start from κ = Q d, which keeps the symmetry of d, stays on the saddle: before the
  probe H2CO took five samples from fresh starts (where they fell depends on the thread count),
  and N2 was frozen at 0.76 L (6.57225 eV, its loop closing to 0.972 only). The probe finds the
  negative curvatures and holds their modes (H2CO's mode 1, then 2 and 4 past the antipode; N2's
  ³Δu pair, then its modes 3 to 5), after which both paths are minima through Q* with no fresh
  start: H2CO's level is unchanged (6.44781 eV) and N2's is the symmetric branch's (6.57186 eV),
  at 1 to 4 threads. No other system measured (the 14, the 29 other ethylene rows of the cluster
  test, twisted by 5 to 65° and stretched from 1.22 to 1.46 Å, and BF) holds anything.
- **Holding only the unstable soft modes is not enough**: O3's third soft mode (Ω² = 8·10⁻⁵ Ha²,
  stable) turns its lowest mode's path into a saddle just before the antipode (between 0.95 and
  1.0 L), and a minimisation falls into that mode's own well. Held, the path is a minimum through
  its antipode, with the same V along the branch.

The 3D model stays on the straight path.

## The generator-coordinate model

`model='gcm'` (`gcm_levels`, opt-in, straight path only) keeps the split, the path and the pole
construction, and replaces the flat quantization P²/2M + V(Q) by the Hill–Wheeler equation on the
path's own determinants:

```
|Ψ⟩ = Σ_j f_j |Φ(Q_j)⟩,      Σ_j' [H(Q_j, Q_j') − E N(Q_j, Q_j')] f_j' = 0
                             (in the even and the odd block of Q → −Q)
d_ia = Σ_k σ_k U_ik V_ak:    pair k (u_k, v_k) turns by θ_k = σ_k Q/2, t_k = cos(θ_k − θ'_k)
N(Q, Q')               = Π_k t_k²
H(Q, Q') − E_RHF N     = Σ_k [f(M_k) t_k + S_k + J(M^α_k, M^β_k)] P_k
                         + Σ_{k<l} [J(M_k, M_l) − Σ_s K(M^s_k, M^s_l)] t_k t_l P_kl
⟨Φ(Q)|D_α|Φ(Q')⟩       = Σ_k σ_k sin(θ_k + θ'_k) t_k P_k        (its diagonal is D(Q))
⟨Φ(Q)|S²|Φ(Q')⟩        = Σ_k sin 2θ_k sin 2θ'_k P_k
ΔE_m = E_odd,m − E_even,0,     c_m = |f_odd,mᵀ D f_even,0|
```

Two determinants of the path are already in corresponding orbitals, so every kernel closes in the
2r pair orbitals {u_k, v_k} (r = min(n_occ, n_virt)), around the RHF and its Fock matrix f, with no
division (P_k = Π_{j≠k} t_j², P_kl = Π_{j≠k,l} t_j², M^s_k the 2 × 2 transition density of pair k
less its RHF part, S_k its self term; the formulas are in `_GCMKernels`). The generator points are
the 32 midpoints of [−L, L] (`SOFT_MODE_GCM_POINTS`), and each parity block is solved by canonical
orthogonalization, keeping the overlap eigenvalues above 10⁻⁸ of its largest
(`SOFT_MODE_GCM_OVERLAP_CUT`). Φ(−Q) is Φ(Q) with its spins flipped, so the even block holds the
even spins and the odd block the odd ones. Every odd level ΔE_m becomes a ladder pole through
`collective_modes`, with X + Y = √2 c_m d and X − Y = √2 M ΔE_m c_m e.

**One pair, exactly.** For H2 the path spans {|gḡ⟩, |vv̄⟩} and T0 on any grid of three points or
more, so the model is the CI in the two orbitals {g, v_d}: the FCI in STO-3G (to 3·10⁻¹⁴ eV), and in
cc-pVDZ a two-orbital CI without dynamic correlation (−0.227 eV at 0.75 Å, +0.210 eV at 2.0 Å
against FCI). Its coupling c is close to FCI's, where the 1D model loses 0.09–0.24 to the spread
of its ground state over D(Q) = sin Q:

| R (Å) | 0.75 | 1.0 | 1.2104 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|
| c, FCI | 0.751 | 0.788 | 0.823 | 0.875 | 0.947 | 0.984 |
| c, GCM | 0.773 | 0.811 | 0.844 | 0.887 | 0.944 | 0.987 |
| c, 1D model | 0.579 | 0.594 | 0.608 | 0.633 | 0.705 | 0.892 |

**The harmonic limit is not TDHF's.** The overlap is Gaussian at small amplitude, N ≈ exp(−s²/4)
with s = Q − Q', and the energy kernel's curvature across the diagonal is h₂(0) = dᵀ(A − B)d/4, so
the model's inertia is the Peierls–Yoccoz mass M_PY = 1/dᵀ(A − B)d ≤ M (real generator states
carry no momentum). A harmonic mode then comes out at √(μ dᵀ(A − B)d) = Ω_s (M/M_PY)^½: 1 for
H2/STO-3G, where d is an eigenvector of A − B, but 1.024 (H2/cc-pVDZ at 1.0 Å), 1.023 (H2O), 1.027
(N2) and 1.037 (H2CO) times Ω_s. The poles still close X − Y with the Thouless–Valatin mass M and
e, so that X − Y = ΔE_m (A − B)⁻¹(X + Y) holds as for the other models, while the levels carry
M_PY: the model breaks the identity "the soft mode at its harmonic level is the removed TDHF
eigenpair", which the 1D model and the relaxed path keep. A consistent X − Y would need the model's
own matrix element of the time-odd rotation along e; it is not computed.

**Measured** (eV, errors against the same references as the relaxed path's table; ranks (even,
odd), ⟨S²⟩ of the even ground state, c₁ the GCM's coupling):

| system | reference | straight 1D | relaxed 1D | GCM | ranks | ⟨S²⟩ | c₁ |
|---|---|---|---|---|---|---|---|
| H2 0.75 Å | 10.543 | −0.044 | −0.110 | −0.227 | (2, 1) | 0 | 0.773 |
| H2 1.0 Å | 7.164 | +0.394 | +0.267 | −0.098 | (2, 1) | 0 | 0.811 |
| H2 1.2104 Å | 4.907 | +0.672 | +0.516 | +0.037 | (2, 1) | 0 | 0.844 |
| H2 1.5 Å | 2.704 | +0.826 | +0.658 | +0.178 | (2, 1) | 0 | 0.887 |
| H2 2.0 Å | 0.792 | +0.574 | +0.425 | +0.210 | (2, 1) | 0 | 0.944 |
| H2 3.0 Å | 0.042 | +0.012 | −0.009 | +0.002 | (2, 1) | 0 | 0.987 |
| N2 | 7.160 | −0.482 | −0.589 | −1.557 | (5, 5) | 0.184 | 0.940 |
| H2CO | 5.910 | +0.691 | +0.538 | −0.232 | (4, 4) | 0.011 | 0.842 |
| H2O | 7.438 | +1.000 | +0.961 | +0.831 | (4, 3) | 0.005 | 0.725 |
| ethylene | 4.417 | +0.518 | +0.369 | −0.144 | (4, 4) | 0.017 | 0.852 |
| O3 | 1.513 | +0.781 | +0.615 | −0.060 | (4, 4) | 0.020 | 0.924 |
| ethylene 45° | 2.757 | +0.406 | +0.058 | −0.032 | (4, 4) | 0.022 | 0.880 |
| ethylene 60° | 1.693 | +0.371 | −0.007 | +0.137 | (4, 4) | 0.026 | 0.905 |
| ethylene 70° | 0.974 | +0.302 | −0.079 | +0.291 | (4, 4) | 0.029 | 0.927 |

| group (MSE / MAE / max) | straight 1D | relaxed 1D | GCM |
|---|---|---|---|
| H2 against FCI (6) | +0.406 / 0.420 / 0.826 | +0.291 / 0.331 / 0.658 | +0.017 / 0.125 / 0.227 |
| small molecules against EOM-CCSD (4) | +0.432 / 0.673 / 1.000 | +0.320 / 0.614 / 0.961 | −0.276 / 0.691 / 1.557 |
| diradicaloids against spin-flip EOM-CCSD (4) | +0.465 / 0.465 / 0.781 | +0.147 / 0.190 / 0.615 | +0.084 / 0.130 / 0.291 |
| all 14 | +0.430 / 0.505 / 1.000 | +0.258 / 0.372 / 0.961 | −0.047 / 0.288 / 1.557 |
| ethylene twisted by 10–40° (4, EOM-CCSD) | +0.520 / 0.520 / 0.523 | +0.227 / 0.227 / 0.288 | −0.059 / 0.077 / 0.132 |
| ethylene's C=C at 1.22, 1.30, 1.36, 1.44 Å (4, EOM-CCSD) | +0.501 / 0.501 / 0.569 | +0.350 / 0.350 / 0.422 | −0.143 / 0.143 / 0.155 |

These reproduce the tweak study's audited prototype to 7·10⁻¹³ eV, its ranks, and the ⟨S²⟩ of the
audit's independent kernels. The model removes the 1D model's systematic overestimate, which is a
quantization error (H2/STO-3G: the flat model on the same V(Q) gives 7.171 eV at 1.2104 Å, the GCM
and FCI 6.064), and does best on H2 and the molecules dominated by one pair (σ₂/σ₁ ≈ 0.1 or less:
H2CO, ethylene and its twists, O3). A system costs 0.01–0.4 s on four threads: one integral
transformation over the 2r pair orbitals and 528 closed-form kernel pairs, no UHF energies. Where
it falls short:
- **Degenerate leading pairs (N2, −1.557 eV).** The straight path turns N2's two π pairs together;
  the model is then neither size-consistent nor spin-pure (⟨S²⟩ = 0.184 in the even ground state,
  0.005–0.029 for the others), and it is flagged `'soft_mode_gcm_degenerate_pairs'` from
  σ₂ ≥ 0.9 σ₁ on (`SOFT_MODE_GCM_PAIR_RATIO`). Two H2 at 1.5 Å, 30 Å apart, along
  cos b d₁ + sin b d₂ (d_k on molecule k), against the single molecule's 2.882 eV:

  | σ₂/σ₁ | 0 | 0.5 | 0.7 | 0.9 | 0.97 | 1 |
  |---|---|---|---|---|---|---|
  | GCM (eV) | 2.882 | 2.734 | 2.138 | 1.872 | 1.847 | 1.845 |
  | ⟨S²⟩, even | 0 | 0.07 | 0.38 | 0.49 | 0.50 | 0.50 |

  Benzene's degenerate e₁g/e₂u pairs should put its lowest soft mode in the same place.
- **H2O stays at +0.831 eV** (8.269 eV), near its own harmonic value (8.279 eV), above TDHF's
  (8.089 eV).
- **The overlap cut sets the multi-pair levels.** The overlap eigenvalues of each block fall off
  geometrically with no gap (H2CO's even block: 1, 0.5, 2·10⁻³, 10⁻⁶, 3·10⁻¹⁰), so a cut from 10⁻⁶ to
  10⁻¹⁰ moves single systems by up to 0.07 eV (H2CO 5.650–5.721 eV), while the mean absolute error of
  the 14 stays 0.287–0.295 eV; the grid does not (24 to 64 points against 32: 5·10⁻⁴ eV at most,
  N2). At 10⁻⁸ most molecules keep as many even states as odd ones, which leaves the even ground
  state the shorter space: a cut keeping one more even state, (5, 4), raises ethylene and its
  twists by 0.05 eV, H2CO by 0.04 and O3 by 0.03 (10⁻⁹ or 10⁻¹⁰), and two H2 30 Å apart along a
  direction mostly on one of them (σ₂/σ₁ = 0.3) give 2.683 eV with (4, 4) states at 10⁻⁸ but the
  single molecule's 2.882 with (5, 4) at 10⁻⁹. The record's `cut_scan` holds the excitation and
  ranks at 10⁻⁶, 10⁻⁷, 10⁻⁹ and 10⁻¹⁰.
- **PSD2 moves away from FCI.** The larger coupling gives the soft pole the weight of a TDHF
  pole, 2 M ΔE c² = 0.97 for H2 at 1.0 Å (1D model 0.55, harmonic TDHF 1), and the PSD2 HOMO of
  H2/cc-pVDZ comes out at 16.71, 16.10, 16.13, 16.79 and 18.34 eV at 0.75, 1.0, 1.19, 1.5 and 2.0 Å
  (Z 0.83 to 0.67), against 16.36, 15.30, 14.85, 14.67, 15.54 with the 1D model and 16.20, 14.70,
  13.92, 13.16, 12.85 from FCI.

Not implemented: the GCM over the relaxed path (refused with `NotImplementedError`), independent
angles for degenerate pairs (the pair-resolved form, 3^m states), spin projection, and a
conjugate-momentum generator along e that would restore the TDHF limit.

## Checked in `tests/test_soft_mode_triplet.py`

- The triplet matrices from MO blocks equal the house builder to the cross-path bar (H2, H2O).
- The split is exact (μ/M = Ω_s², dᵀe = 1). The soft mode at its harmonic level is the house
  Casida's root to 1e-10 (H2, and N2 with its multi-pair soft direction).
- The path and solvers check out:
  - V''(0) = μ (1e-7);
  - D(Q) = sin Q for one occupied orbital;
  - a harmonic V gives Ω and 1/(2MΩ) to 1e-5 in all three solvers.
- PSD2 with the soft mode at its harmonic level is PSD2@TDHF to the cross-path bar. With
  `model=None` the driver reproduces `calc_qp_energy`'s PSD2@TDHF to 2.6e-8 eV.
- H2/cc-pVDZ against FCI:
  - The collective 1D triplet is within 1.0 eV from 0.75 to 2.0 Å (worst +0.83 eV at 1.5 Å), and
    closer than TDA from 1.0 Å on.
  - PSD2@TDHF loses its quasiparticle at 1.19 Å (Z = 0.36, IP 26 eV) and is refused at 1.212 Å.
    With the soft mode split off it keeps Z = 0.84, moves by 0.03 eV across R_CF, and lies closer
    to FCI.
- At the MP2 minima, against recorded EOM-CCSD:
  - For N2, H2CO, ethylene and O3, the collective 1D triplet misses by −0.48, +0.69, +0.52 and
    +1.15 eV, closer than TDA's root of the same mode (−1.60, −1.20, −0.93 eV, negative for O3).
  - H2O, a stiff triplet, misses by +1.00 eV, where TDHF misses by +0.65.
- The refusals and both flags.
- The relaxed path:
  - Its harmonic limit is TDHF's: on the quadratic model of the energy it is the straight path
    exactly (H2, N2); on the real one V''(0) = μ to 1.2·10⁻⁷ and M(0) = M to 3·10⁻¹³ (H2 and
    ethylene twisted by 60°, an unstable mode), which holds the RHF and the split rather than the
    relaxation (at Q = 4·10⁻³ the straight point is already the constrained minimum); the
    variable-mass solver gives a harmonic oscillator's Ω to 5·10⁻⁷ and, with a constant mass, the
    constant-mass levels to 1.4·10⁻⁶ eV.
  - One pair (H2/STO-3G) has nothing to relax: V, M and Q* are the straight path's, and its level
    too to 1.2·10⁻⁶ eV.
  - Ethylene twisted by 60°: V_relaxed ≤ V_straight along the path and 24 mHa below it at L/2,
    eᵀκ = Q, and its 1D triplet within 0.05 eV of spin-flip EOM-CCSD (−0.007 eV, where the straight
    path is +0.371). O3's lowest mode keeps its two other soft modes at zero to 7·10⁻¹⁵.
  - The constrained saddles: without hold rounds H2CO takes samples from fresh starts and its
    probe finds a negative curvature (flagged `'soft_mode_path_branch'`); with them H2CO holds its
    mode 1 at the same level (to 2·10⁻⁸ eV) and N2 its ³Δu pair, after which N2 is unfrozen through
    Q* at the symmetric branch's level (6.57186 eV, to 2·10⁻⁶ eV).
  - The 14 small systems' relaxed levels against their record (to 10⁻⁴ eV; 1, 2 and 4 threads
    agree to 3·10⁻⁹), every path closed, unfrozen, converged and free of fresh starts, and the
    constrained Hessian at the sample nearest Q* positive definite for H2CO and O3 (+0.082 and
    +0.042 Ha).
  - `soft_mode_qp` passes the path through; the refusals and the flags (not converged, frozen,
    closure).
- The generator-coordinate model (`model='gcm'`):
  - Its closed-form kernels equal a generic non-orthogonal evaluation of N, H, D_α and S² at random
    (Q, Q') to 1.6·10⁻¹³ Ha (H2O, N2, and H2O/STO-3G, whose occupied orbitals outnumber its
    virtuals); their diagonal is the straight path's V and D.
  - One pair: H2/STO-3G gives the FCI triplet to 3·10⁻¹⁴ eV; H2/cc-pVDZ gives the two-orbital CI in
    {g, v_d}, gap and coupling, to 5·10⁻¹⁴, with spin-pure states, on any grid; against FCI it is
    within 0.25 eV from 0.75 to 2.0 Å and closer than the 1D model from 1.0 Å on.
  - 24 and 48 generator points give the level of 32 to 5.4·10⁻⁴ eV (N2) at the same ranks.
  - The kernels' small-amplitude limit carries the Peierls–Yoccoz mass (V''(0) = μ, overlap width
    ½, h₂(0) = dᵀ(A − B)d/4 to 2.5·10⁻⁷), so a harmonic level is (M dᵀ(A − B)d)^½ = 1.023 to
    1.027 times Ω_s for H2/cc-pVDZ, H2O and N2, and exactly Ω_s for H2/STO-3G.
  - N2, H2CO, H2O, ethylene and O3 reproduce the audited prototype's levels and ranks; N2 is flagged
    with ⟨S²⟩ = 0.184; H2CO and ethylene lie within 0.25 eV of EOM-CCSD.
  - `soft_mode_qp` passes the model through with the GCM's level and coupling in its pole; the
    relaxed path with `'gcm'` is refused with `NotImplementedError`; both flags.

## Checked in `tests/test_soft_mode_triplet_slow.py`

- Ethylene twisted from 0 to 70°: the collective 1D triplet tracks EOM-CCSD with an offset of
  +0.52 to +0.55 eV. TDA misses by −0.93 to −1.15 eV and is negative at 70°.
- Ethylene's C=C stretched from 1.22 to 1.46 Å, through the instability at 1.3345 Å:
  - PSD2@TDHF has no quasiparticle from 1.30 Å on (IP 47 eV, Z = 0.08 at 1.3325 Å) and is refused
    beyond.
  - With the soft mode split off, the PSD2 HOMO stays between 11.03 and 11.22 eV with Z ≈ 0.71.
- Butadiene and benzene: the collective 1D triplet is +0.35 and −0.06 eV from EOM-CCSD, against
  −0.70 and −0.63 eV for TDA.

## Checked on a cluster: `tests/test_soft_mode_triplet_full.py`

- Hexatriene, octatetraene and the stretch past 1.30 Å, against live EOM-CCSD seeded with the soft
  direction.
- O3 and ethylene twisted by 45–70°, against spin-flip EOM-CCSD.
- The PSD2 roots of the four polyenes.

## Checked on a cluster: `tests/test_soft_mode_variants_full.py`

The three variants side by side (straight path in 1D, the relaxed path, the GCM), against the
references recorded in `tests/soft_mode_references.py`: EOM-CCSD recorded on a workstation
(`EOM_CCSD`: butadiene, benzene, planar ethylene, its twists other than 45, 60 and 70° and its
stretch up to 1.30 Å), the EOM-CCSD and spin-flip EOM-CCSD of the ADA run of the heavy file above
(job 1238268, to the meV: hexatriene, octatetraene, the stretch past 1.30 Å, and O3 and the twists
of 45, 60 and 70°), and for BF the frozen-core CIPSI IPs of Marie and Loos (JCTC 20, 4751 (2024):
11.054 eV in aug-cc-pVDZ, 11.149 eV in aug-cc-pVTZ; the "FCI" 11.15 eV their 2025 paper and the
[parquet page](parquet.md) quote is this CIPSI value rounded).
- Butadiene, benzene, hexatriene, octatetraene, O3, ethylene twisted from 0 to 70° and its C=C
  stretched from 1.22 to 1.46 Å (38 rows; spin-flip for O3 and the twists of 45, 60 and 70°):
  every variant's level is finite; every relaxed path is a closed constrained minimum through its
  antipode (unfrozen, converged, no sample from a fresh start, no negative curvature at its
  probes), as on the 34 rows a workstation can run; and the relaxed path's mean absolute error
  lies below the straight path's, as on the small systems (0.372 against 0.505 eV on the 14).
- BF in aug-cc-pVDZ and aug-cc-pVTZ (QUEST geometry): the relaxed path is a minimum at the sample
  nearest Q* (dense constrained Hessian) and lies more than 1 mHa below the straight one at L/2,
  and it keeps the TDHF harmonic limit (V''(0) = μ to 10⁻⁵ at Q = 10⁻², M(0) = M to 10⁻⁸), a check
  of the RHF and the split whose scatter over RHF initial guesses is 2·10⁻⁶.
- Printed for the record, not held: the GCM's levels with their ranks, ⟨S²⟩, couplings and cut
  scans; the relaxed paths' antipodes, inertia, closure, held modes, curvature and cost; errors by
  group; BF's PSD2/4/7/9 HOMO with the plain TDHF triplets and with each variant, with their Z,
  and G0W0@HF and Σ_TDHF@L_TDHF as controls.

## Measured besides

- **The Σ-side-only vertex.** A vertex in Σ alone (Γ from δΣ_x/δG to all orders, with W from RPA)
  carries the triplet ladder and diverges at the instability: H2/cc-pVDZ's IP goes 16.3, 14.1,
  11.5 and 6.3 eV at 0.75 to 1.19 Å, then has no root. Its quasiparticle has Z > 1 (1.04 at
  0.75 Å, 2.5 at 1.19 Å), so it is not in the code. The soft-mode split keeps it finite.
- **EOM-CCSD from the RHF.** It falls short for O3's soft triplet (³B₂, −0.36 eV) and for ethylene
  twisted by 70° (−0.18 eV) against FCI in STO-3G, while spin-flip EOM-CCSD stays within 0.07 eV.
- **PSD2 itself.** It is far from EOM-IP-CCSD for these molecules (H2O −3 eV, N2 +5 eV). The split
  makes it finite and smooth, not accurate. O3's soft-mode PSD2 HOMO has Z = 0.22.

## Model choices

These were made on purpose. Three of them make the default and are measured above against an
alternative:
- the straight path along d, with no relaxation of the other rotations (it reaches 80% of the UHF
  well depth for twisted ethylene, 62% for O3), against `path='relaxed'`, which relaxes the triplet
  rotations (89–93% of it, O3 64%); the singlet rotations, which hold the rest of the well, stay
  unrelaxed in both;
- the mass held at its Q = 0 value, against the relaxed path's inertia M(Q);
- the flat quantization P²/2M of a coordinate whose phase space is bounded, against
  `model='gcm'`, which quantizes on the path's own determinants.

The others are not checked against alternatives:
- D(Q) as the transition density that couples back;
- the flat 3D ball, which runs 0.6 to 1.1 eV above the 1D model for the soft modes, in place of
  the curved manifold of rotated determinants;
- independent soft sets (O3's second mode stops being unstable once the first condenses);
- the closure of a straight path whose minor pairs have not come back.

There is no large parameter here as there is in SYK, so the collective model is not a controlled
expansion. Its levels sit 0.5 eV above EOM-CCSD fairly uniformly along ethylene's twist.
