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

An opt-in sign-aware version, `reference='excited'`, admits excited (non-aufbau) closed-shell
references, where the mass can be negative, and `soft_mode_singlets` splits the singlet block the
same way (H2's doubly excited σ_u² and its symmetry breaking toward H⁺H⁻): see
[Excited references and the singlet block](#excited-references-and-the-singlet-block). With it,
`collisions='gcm'` quantizes the two modes of a Krein collision (a complex quartet) together
instead of refusing them: see [Krein collisions: the two-mode GCM](#krein-collisions-the-two-mode-gcm).
`soft_mode_open_shell` splits a UHF's spin-conserving or spin-flip block the same way, and
`noci_levels` mixes broken-symmetry UHF solutions: see
[Open shells: UHF blocks, spin flip and core holes](#open-shells-uhf-blocks-spin-flip-and-core-holes).

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

An excited reference, H2's doubly excited σ_u² in 6-31G past its symmetry-breaking instability
(see [Excited references and the singlet block](#excited-references-and-the-singlet-block)):

```python
from src.SingleReference.LinearResponse.soft_mode import soft_mode_singlets

mol = gto.M(atom='H 0 0 0; H 0 0 1.5', basis='6-31g', symmetry=True)
mf = scf.RHF(mol)
mf.irrep_nelec = {'A1u': 2}                       # both electrons in σ_u (or a maximum-overlap SCF)
mf.run()
rec = soft_mode_singlets(mf, model='gcm', reference='excited')
print(rec['soft'][0]['excitation'] * HARTREE_TO_EV)   # −3.08 eV, a de-excitation (FCI −3.20)
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
| `soft_mode_triplets(mf, model, path, reference, collisions)` | model | `'1d'` (default), `'3d'`, or `'gcm'` (straight path only, see [The generator-coordinate model](#the-generator-coordinate-model)) |
| | path | `'straight'` (default) or `'relaxed'` (1D only, see [The relaxed path](#the-relaxed-path)) |
| | reference | `'aufbau'` (default) or `'excited'`: any stationary closed shell, split sign-aware (straight path only, see [Excited references and the singlet block](#excited-references-and-the-singlet-block)) |
| | collisions | `'refuse'` (default) or `'gcm'`: with `reference='excited'`, each Krein quartet and each soft mode near one quantized with its partner by the two-mode GCM (see [Krein collisions: the two-mode GCM](#krein-collisions-the-two-mode-gcm)) |
| `soft_mode_singlets(mf, model, path, reference, collisions)` | model | `'1d'` (default) or `'gcm'` (the singlet block, RHF→RHF paths) |
| | path | `'straight'` only |
| | reference | `'aufbau'` (default) or `'excited'`, as for `soft_mode_triplets` |
| | collisions | `'refuse'` (default) or `'gcm'`, as for `soft_mode_triplets` |
| `soft_mode_open_shell(mf, block, model, collisions)` | block | `'spin_conserving'` (default) or `'spin_flip'` (a UHF's TDHF blocks, see [Open shells](#open-shells-uhf-blocks-spin-flip-and-core-holes)) |
| | model | `'1d'` (default) or `'gcm'` |
| | collisions | `'refuse'` (default) or `'gcm'`, as for `soft_mode_triplets` (any UHF, no `reference` option: the split is always sign-aware) |
| `noci_levels(solutions)` | solutions | two or more converged UHF solutions of one molecule (non-orthogonal CI) |
| `soft_mode_qp(mf, selfenergy, states, model, path)` | selfenergy | `'PSD2'` (default), `'PSD4'`, `'PSD7'`, `'PSD9'` |
| | states | orbital indices; the HOMO by default |
| | model | `'1d'`, `'3d'`, `'gcm'`, or `None` for the plain TDHF triplets (the house route's spectrum) |
| | path | as for `soft_mode_triplets` |

The reference must be a converged closed-shell RHF with exact integrals; anything else is refused
before any work, as are an unknown path and the relaxed path with any model but `'1d'` (with
`'gcm'` as `NotImplementedError`: the GCM over the relaxed path is future work), and
`collisions='gcm'` with an aufbau reference (`ValueError`: no two modes can collide there). An A − B
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
- `'soft_modes_coupled'`: more than one soft set (with `collisions='gcm'`, more than one soft set
  or colliding pair);
- with `path='relaxed'`, `'soft_mode_path_not_converged'` (a constrained minimisation stopped
  short), `'soft_mode_path_frozen'` (the relaxation was frozen before the antipode),
  `'soft_mode_path_branch'` (after the hold rounds a sample still came from a fresh start, or a
  negative curvature across the constraint plane remains) and `'soft_mode_path_closure'` (the loop
  does not close at its antipode);
- with `model='gcm'`, `'soft_mode_gcm_degenerate_pairs'` (the two leading pairs of the soft
  direction are degenerate) and `'soft_mode_gcm_few_states'` (the canonical orthogonalization kept
  fewer states than one pair's span, two even and one odd, or fewer even states than odd ones);
- with `reference='excited'`, `'soft_mode_krein_neighbour'` (a soft mode within
  `SOFT_MODE_KREIN_GAP` of a mode of the other Krein signature) and, with `model='gcm'`,
  `'soft_mode_reference_overlap'` (the even level kept as the reference's continuation is not the
  one of largest overlap with the reference determinant) and `'soft_mode_reference_tie'` (the two
  even states its level rule picks between overlap the determinant within
  `SOFT_MODE_GCM_OVERLAP_MARGIN` of each other);
- with `collisions='gcm'`, `'soft_mode_collision_reference'` (the even state kept for a colliding
  pair holds less than `SOFT_MODE_COLLISION_OVERLAP_MIN` of the reference determinant) and
  `'soft_mode_collision_few_states'` (fewer states kept than a pair's span for one occupied
  orbital);
- the root finder's own flags.

With `path='relaxed'` each soft mode's record also holds its antipode Q*, M(Q*)/M, D(Q*), the
overlap of the α determinants at ±Q*, whether every constrained minimisation converged, the fresh
starts tried and the samples taken from one, the TDHF modes held for a saddle, the lowest
curvature probed, where the relaxation was frozen (or `None`), where the path was continued in
its arc length past a turning point of Q (`continued_from`, or `None`; Q* and `q_min` are then in
that arc length) and the number of UHF energy evaluations; its `v_min` is then the relaxed well
depth. All the flags are raised by
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
  after it; a path frozen before its antipode at a turning point of Q is continued in its arc
  length instead (see the turning points below), and stays frozen only if that fails too;
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

**Turning points of Q: the path continued in its arc length.** On the cluster (job 1242171) the
relaxed paths of hexatriene and octatetraene in cc-pVDZ, two soft modes each, froze before their
antipodes (from 0.80 and 0.78 L the lowest ones) and did not close (overlap 0.891 and 0.483). Their
STO-3G versions (38 and 50 basis functions) and butadiene/STO-3G (26, two soft modes too) freeze the
same way. The cause is not the hold of the other soft mode, whose multiplier along these paths is
10⁻⁸ to 10⁻⁶ (it has another symmetry), but the coordinate. Their soft directions spread over two
pairs of comparable weight (σ₂/σ₁ = 0.58 and 0.93 for butadiene's two modes, 0.62 and 0.97 for
hexatriene's), and the relaxation turns the minor pair back while the leading one goes to 90°.
Along the curve of constrained stationary points (∇E = λe plus the holds), Q = eᵀκ then rises to a
maximum before the antipode and falls back: the fixed plane turns tangent to the curve, M(Q) grows
without bound (11 M at the last sample before butadiene's turning point), the curvature across the
plane along the minor pair's partner mode (98 % of the softest direction) falls to zero, and the
minimum jumps past the antipode onto points that are saddles on their planes, and on to a
wrap-around point; a fresh start from κ = Q d jumps the same way.

Such a path, frozen with no negative curvature at its probes, is continued in its arc length
(`_continued_samples`): the fixed-plane samples are kept while their steps stay within 1.5
arc-length steps h (dx² = dκᵀ(A − B)⁻¹dκ/M), and each further sample is the point of the same curve
on the hyperplane normal to it (in that metric) through the predictor κ + h t: the minimum on the
slice of that hyperplane at eᵀκ = Q, with Q found by a secant search on the hyperplane's
multiplier (2 to 5 minimisations a sample). The path runs to the antipode x*, where the loop closes
and D(x) changes sign, and four samples past it; it is quantized in x, along which its inertia is M
(to 5·10⁻⁴ and 2.4·10⁻³ for butadiene's two modes), and its curvature across itself is probed on
the slices and held as on the fixed planes, when the continuation reaches the antipode. Past the
turning point the continued samples are minima across the path, on their slices, but saddles of
their fixed planes: the curve's other branch, where κ(Q) = argmin {E : eᵀκ = Q} no longer holds
(the dense ZᵀHZ on the fixed plane, butadiene/STO-3G: +0.40 Ha at x = L/2, −0.0005 at mode 0's
turning point, −0.0085 at the sample nearest x*, and −0.41 Ha at mode 1's x*, where the slices
stay positive, +0.05 to +0.40 and +0.009). So the probe of a continued path is the slice's, and the
fixed plane's dense Hessian (the fast test's `lowest_constrained_curvature`) does not apply to it. A
continuation that leaves its branch before the antipode leaves the frozen path and its record as
they were, flagged: the step that left the branch is not probed, so it adds no curvature and no
hold, and only `energy_calls` counts its energies. Octatetraene/STO-3G's frozen modes 1 and 2 give
the records of the code before the continuation but for `energy_calls` and `continued_from`
(levels 3.001530 and 4.003626 eV, curvatures +0.109 and +0.168, no holds); with that step probed,
mode 1's curvature had dropped to +0.019 and mode 2 had held TDHF modes 5 and 16 at five times the
energies, its frozen samples moved by 10⁻⁷ Ha.

| system (basis functions) | mode | frozen at (overlap at the antipode) | level frozen | continued from | x*/L | level continued | straight |
|---|---|---|---|---|---|---|---|
| butadiene/STO-3G (26) | 0 | 0.8125 L (0.839) | 3.495 | 0.7625 L | 0.9997 | 3.571 | 3.679 |
| | 1 | 0.7375 L (0.649) | 5.068 | 0.7250 L | 1.1768 | 5.086 | 5.115 |
| hexatriene/STO-3G (38) | 0 | 0.775 L (0.456) | 2.499 | 0.7375 L | 1.0509 | 2.547 | 2.675 |
| | 1 (TDHF modes 2, 36 held) | 0.787 L (0.754) | 4.149 | 0.7750 L | 1.2873 | 4.153 | 4.266 |
| hexatriene, 6-31G on C, STO-3G on H (62) | 0 | 0.787 L (0.658) | 2.486 | 0.7375 L | 1.0178 | 2.541 | 2.703 |
| | 1 (TDHF modes 2, 24 held) | 0.775 L (0.796) | 3.957 | 0.7625 L | 1.2687 | 3.959 | 4.078 |

(eV.) Every continued loop closes (overlap within 10⁻¹³ of 1 on the STO-3G paths and within the six
digits printed on the 62-function one, |D(x*)| below 10⁻¹⁴) with a positive curvature across it and
no `'soft_mode_path_*'` flag; the levels move by 4·10⁻⁵ eV at most when the
continuation takes over at 1.25 or 2 steps instead of 1.5, and by 8·10⁻⁵ eV when h is halved. A
continued path costs about twice the energies of the frozen one (butadiene: 538 and 871 against 298
and 328). The paths that close on their fixed planes are not touched: their records are bitwise
those of the code before the continuation (the 14 small systems, BF in cc-pVDZ and aug-cc-pVDZ,
ethylene twisted by 10 and 20° and stretched to 1.3365 and 1.46 Å, butadiene/3-21G, and the
straight and GCM variants).

The arc length removes the ordering of P and M(Q) from the 1D model, and that ordering is not
negligible where M(Q) is steep, which is near a turning point. Quantizing the samples of the paths
that close in Q in their own arc length instead moves the level by −10 to +4 meV on the 14 small
systems, but by +18 to +40 meV for ethylene twisted by 10 to 30° (M(Q*)/M = 2.7 to 4.9) and
+159 meV for butadiene/3-21G (M(Q*)/M = 12.7): a path that closes just short of a turning point
keeps the ordering −½ ∂_Q M(Q)⁻¹ ∂_Q in the flat measure dQ (BenDaniel–Duke's, `_levels_1d_mass`),
one just past it is quantized in x, and the two differ by that much. The ordering in Q is not
Pauli's: the Laplace–Beltrami operator of the metric M(Q) dQ², −½ M^−½ ∂_Q M^−½ ∂_Q on √M dQ, is
the arc-length quantization written in Q, and in the flat measure it is the code's operator plus
V_q = M''/(8M²) − 7M'²/(32M³), large where M(Q) is steep. Solved in Q on the same grid and the
same M(Q), it gives the arc-length level to 0.04 meV (butadiene/3-21G 3.407975 against 3.407947
eV, the code's 3.248755 eV; ethylene twisted by 10 and 20°, N2, H2O alike), so V_q accounts for
the whole 159 meV.

Not fixed, octatetraene/STO-3G (three soft modes): its lowest mode is continued but closes to 0.979
only (flagged `'soft_mode_path_closure'`; the level moves by 0.2 meV when walls replace the
closure). Its third soft mode has the same symmetry (B_u), and that mode's hold forbids the
closed-shell antipode of the leading pair: e_tᵀκ = −1.02 there, where it is zero for a soft mode of
another symmetry (butadiene, hexatriene, O3). A hold on the density, e_tᵀ(2P^α_ov) = 0, equal to
e_tᵀκ to third order and zero at that antipode, would allow it; it is not implemented. Its other two
modes still freeze (from 0.80 and 0.74 L): their continued paths turn into saddles across themselves
before the antipode; the step that left the branch is not probed, so their records keep the
fixed-plane probes (+0.109 and +0.168, no holds). In cc-pVDZ the two soft modes of hexatriene and
octatetraene are expected to have different symmetries, as with 6-31G on carbon, where both close;
not run here.

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

## Excited references and the singlet block

`reference='excited'` (opt-in, straight path, in `soft_mode_triplets` and `soft_mode_singlets`)
admits a closed shell that is a saddle of the Hartree–Fock energy: an excited, non-aufbau reference
such as H2's doubly excited σ_u², from a maximum-overlap or a symmetry-constrained SCF. There A − B
is indefinite, (A − B)^½ does not exist and the default refuses the reference (`CasidaInstability`;
a non-aufbau occupation already at `check_reference`). The analysis of Burth, Kitsaras and Loos
(arXiv:2610.36154) in the soft-mode notes proposed the sign-aware version implemented here.

| step | quantity |
|---|---|
| reference | `excited_reference`: any 0/2 occupation, reordered so that the occupied orbitals come first (`orbitals` in the record maps back); refused if its own Fock matrix has an occupied–virtual element of `SOFT_MODE_EXCITED_GRADIENT_MAX` (10⁻⁴ Ha) or more |
| split | `krein_split`: the non-symmetric (A − B)(A + B) d = Ω² d, d ∝ (X + Y), real and normalized; each degenerate set given a real basis of its eigenvectors' span (`_real_modes`: LAPACK can return a degenerate real Ω² as a conjugate pair with imaginary parts of rounding, whose two vectors share their real part) and orthogonalized in the indefinite metric (A − B)⁻¹; every mode checked against its eigen-equation |
| signs | M = dᵀ(A − B)⁻¹d of either sign (its Krein signature), μ = dᵀ(A + B)d = M Ω² |
| classes | M > 0: a stable excitation (μ > 0) or the ground-state double well (μ < 0); M < 0: a stable negative transition (μ < 0) or the excited-state instability (μ > 0) |
| regular modes | the positive-norm member, Ω = sign(M) \|Ω\|, X + Y = d/(MΩ)^½, X − Y = (MΩ)^½ (A − B)⁻¹d/M: a negative transition has Ω < 0 |
| levels, M > 0 | as before: H = P²/2M + V(Q), the reference the bottom level; in the GCM, on a block whose A − B is indefinite (whose span holds configurations below an excited reference), the even ground state unless another even state overlaps the reference determinant more by over `SOFT_MODE_GCM_OVERLAP_MARGIN` (0.1; `gcm_levels(level='overlap')`), its transitions those above the kept state first, nearest first, then those below |
| levels, M < 0 | H is unbounded below, so −H = P²/2\|M\| − V(Q) is solved; the reference is its lowest level, H's top one, and its transitions ΔE_m = −(e_m − e_0) < 0 are de-excitations; in the GCM the reference is the even level of largest overlap with the reference determinant above the even ground state (`gcm_levels(level='top')`; for one pair the top one) |
| poles | `collective_modes` with the signed M and ΔE_m: X − Y = √2 M ΔE_m c_m e, positive norm for a transition of the mode's own sign; `signature` holds the sign of X'X − Y'Y of every column |

For a harmonic V the levels are again the removed TDHF eigenpair, now its positive-norm member at
Ω < 0 for M < 0 (checked to 2·10⁻¹⁶ against a dense solve of the RPA matrix). Refused:
- a complex Ω² (an imaginary part above `SOFT_MODE_KREIN_IMAG_TOL` = 10⁻⁸ of the largest |Ω²|):
  two modes of opposite signature have collided into a quartet, which no single collective
  coordinate describes (H2/6-31G σ_u²: the singlet at 0.7625–0.77 Å, the triplet at
  0.945–1.01 Å); `collisions='gcm'` quantizes the pair instead
  ([Krein collisions: the two-mode GCM](#krein-collisions-the-two-mode-gcm));
- a mode of null norm (|M| below `SOFT_MODE_KREIN_NORM_MIN` = 10⁻⁶ of the largest eigenvalue of
  |(A − B)⁻¹|), the exceptional point where such a collision begins;
- a mode that misses (A − B)(A + B) d = Ω² d by more than `SOFT_MODE_KREIN_RESIDUAL_MAX` = 10⁻⁶
  of the largest |Ω²| (|d| = 1): a degenerate set whose eigenvectors do not span it. With the sets
  made real together every mode measured solves it to 2.3·10⁻¹⁴ (N2/6-31G's doubly excited closed
  shell 3σ_g² → 3σ_u², the localized N2⁺ 1s hole, cyclobutadiene/STO-3G, doubled H2 blocks);
  made real one vector at a time, as before, the second vector of a pair returned with imaginary
  parts of rounding was noise (2.2·10⁻³ to 0.75): N2's soft π pair got the masses −12.02 and
  −1.20 Ha⁻¹ and the levels −1.35 and −16.47 eV instead of one each, and 3 to 6 regular columns
  missed the RPA equations by 0.15 to 8.2 Ha (N2, N2⁺, cyclobutadiene at D4h);
- the relaxed path (`NotImplementedError`): from a saddle, a constrained minimisation falls toward
  the ground state, and a constrained saddle search is not implemented.

A soft mode within `SOFT_MODE_KREIN_GAP` (0.1, relative) of a mode of the other signature is
flagged `'soft_mode_krein_neighbour'`: the one-mode split leaves out the mixing that leads to a
collision (H2/6-31G σ_u²: 0.03–0.08 within 0.015 Å of the windows, 0.97 and more at the rows
below). On an aufbau reference with A − B positive definite, `reference='excited'` is the default:
`krein_split` is `soft_split` to the bit and the levels and poles agree to 3·10⁻¹⁶.

**The singlet block** (`soft_mode_singlets`, `singlet_tdhf_matrices`): A = Δε + 2(ia|jb) − (ij|ab),
B = 2(ia|jb) − (ib|ja), the same A − B as the triplet's, so the same mass. Its path turns both spins
by exp(K), an RHF→RHF path (`path_samples` and `_GCMKernels` with `channel='singlet'`; the GCM
kernels are the triplet ones with M^β_k = M^α_k, and ⟨S²⟩ = 0), whose V is even only where a
symmetry of the reference takes d to −d: a direction with |V(−Q) − V(Q)| above
`SOFT_MODE_SINGLET_PARITY_MAX` (10⁻³ of max |V| on [0, L]) is refused, and a nearly even one below
the bar is quantized as even. The soft modes measure 10⁻¹⁶ to 10⁻¹³ and the directions no symmetry
keeps even at most 10⁻¹¹; the totally symmetric directions form a continuum that the bar lies
inside, not a gap: the lowest ones measure 0.04 to 0.8 (H2O and H2CO in cc-pVDZ), but over the
whole spectrum H2O's 33 A1 directions run from 7.9·10⁻⁴ to 0.61 and H2CO's 87 (at exact C2v) from
1.4·10⁻⁴ to 0.45, a few core-level ones below the bar. The 3D model (three triplet components)
and the relaxed path are refused for singlets. H2 σ_u²'s instability toward H⁺H⁻ is in this block.
Both references are admitted; on an aufbau one the singlet's harmonic level is the house singlet
Casida root (to 2·10⁻¹⁴).

**Spin channels covered**: the closed-shell singlet and triplet blocks (the triplet's M_s = 0
component in 1D, the three components in 3D), for aufbau and excited references; open-shell
(UHF spin-conserving and spin-flip) blocks in `soft_mode_open_shell` (see
[Open shells](#open-shells-uhf-blocks-spin-flip-and-core-holes)). Not covered: Krein quartets by
default (refused; the two-mode GCM of `collisions='gcm'` below), the singlet channel of the
screening W in `soft_mode_qp` (which takes aufbau references only, its triplet list from the
default split).

**Measured against FCI** (eV; H2/6-31G, the paper's basis, σ_u² from a symmetry-constrained SCF;
FCI's transition is from the singlet of largest σ_u² weight to the state of the channel it couples
to most through D):

| R (Å) | M | FCI | TDHF | TDA | 1D | GCM |
|---|---|---|---|---|---|---|
| 1.2 (singlet) | −3.01 | −6.035 | −3.743 | −5.121 | −6.319 | −5.704 |
| 1.3 | −3.42 | −4.903 | −1.792 | −3.985 | −5.369 | −4.666 |
| 1.34 | −3.59 | −4.508 | 0.240i | −3.576 | −5.026 | −4.301 |
| 1.4 | −3.86 | −3.969 | 2.023i | −3.005 | −4.546 | −3.800 |
| 1.5 | −4.37 | −3.199 | 2.999i | −2.150 | −3.830 | −3.077 |
| 2.0 | −8.00 | −1.002 | 3.989i | +0.943 | −1.404 | −0.985 |
| 3.0 | −29.96 | −0.069 | 2.812i | +4.187 | −0.039 | −0.200 |
| 0.2 (triplet) | +18.79 | +0.249 | 1.792i | −0.406 | +0.792 | +0.268 |
| 0.3 | +9.82 | +1.375 | 1.293i | +1.058 | +1.979 | +1.417 |
| 0.34 | +8.08 | +1.943 | +0.650 | +1.719 | +2.525 | +1.995 |

| group (MSE / MAE / max) | 1D | GCM |
|---|---|---|
| σ_u² singlet, 1.2–3.0 Å (7) | −0.407 / 0.415 / 0.631 | +0.136 / 0.173 / 0.331 |
| σ_u² triplet, 0.2–0.34 Å (3) | +0.576 / 0.576 / 0.604 | +0.038 / 0.038 / 0.052 |

- **The singlet symmetry breaking** (σ_u → σ_g, σ_g′; Ω² changes sign between 1.33 and 1.34 Å):
  M < 0 throughout, μ < 0 before the crossing (a negative transition) and > 0 after it (the
  excited-state instability). TDHF turns imaginary from 1.34 Å on and TDA changes sign from 2.0 Å
  on, while both sign-aware levels stay real, negative and smooth through the crossing and rise
  toward 0 at dissociation with FCI, the ionic pair's degeneracy. The 1D model is too negative by
  0.3–0.6 eV (the mirror image of its overestimate for the ground-state triplet); the GCM, for one
  pair the CI in {σ_u, v_d} from its top even state (to 2·10⁻¹⁴), is within 0.33 eV.
- **The short-bond triplet** (σ_u → σ_g′, below 0.335 Å): M > 0, the ground-state double-well
  class on an excited reference, so nothing is inverted: the reference is the bottom level, the
  transitions positive. The GCM is within 0.05 eV of FCI, the 1D model 0.54–0.60 eV high. Its
  block's A − B is indefinite only because of the stable negative mode σ_u → σ_g (Ω = −30 to
  −34 eV).
- **The reference level.** At 3.0 Å the GCM's two even states share the σ_u² determinant
  (0.502 / 0.498). The top one, the ionic state, as FCI's σ_u²-like state is, is kept (−0.200 eV
  against FCI's −0.069), and the run is flagged `'soft_mode_reference_overlap'`, as are 3.0 Å in
  cc-pVDZ and cc-pVTZ; the level of largest overlap would be the covalent one, at +8.7 eV.
- **The reference level with more than one pair.** One pair gives two even states, and the top one
  continues the reference. With more, the highest roots come from the smallest overlap eigenvalues
  kept and hold almost none of the reference determinant, so `gcm_levels(level='top')` keeps the
  even state of largest overlap above the even ground state (the reference continues as the top of
  the mode's collective spectrum), as the analysis prescribed ("chosen by overlap"). Measured, not
  against FCI: N2/STO-3G's π_u⁴ → π_g⁴ singlet, whose top root held 4.5·10⁻⁴ of the determinant and
  gave +1.447 eV (8.889 to 3.028 eV over the cut scan), now keeps the state of 0.817 at −2.153 eV;
  LiH/6-31G 1s² 1π_x² +0.870 → −0.509 eV; Be/cc-pVDZ 1s² 2p_z² +0.921 (136 eV at the cut 10⁻⁶) →
  −0.920 eV; rectangular H4/6-31G HOMO² → LUMO² at 1.5 × 1.0 Å +4.637 → −5.232 eV: de-excitations,
  as the negative mass makes them, the same over the cut scan to 0.05 eV but for N2 (−2.30 to −4.46
  eV; its π pairs are degenerate, flagged). Every one-pair row keeps its level (H2/6-31G σ_u²
  singlet and triplet, Be/6-31G, square H4/6-31G). The even ground state is never kept for a
  negative mass, though it can hold most of the determinant: on aufbau references with a
  negative-mass mode (0.66 to 0.95: O2's and square H4's triplet UHF, N2⁺'s symmetric hole, square
  H4's RHF) the kept state holds little of it, flagged `'soft_mode_reference_overlap'` (square
  H4/6-31G's triplet UHF in its spin-flip block: −4.39 and −1.17 eV at 1.2 and 1.0 Å, the even
  ground state +1.88 and +0.46, FCI's singlet −0.20 and +0.05 eV from the triplet). Keeping it there
  would need a test that tells a ground-state determinant from an excited one, which H2's σ_u²
  defeats: it is aufbau in its own Fock spectrum from 1.2 Å on, and its top state is the closer to
  FCI where the even ground state holds more (−10.75 against −1.87 eV for FCI's −15.07 in the 6-31G
  triplet at 1.015 Å, 0.821 / 0.179; −0.67 against +5.47 eV for FCI's −0.07 in the cc-pVTZ singlet
  at 3.0 Å, 0.619 / 0.381).
- **The reference level of a positive mass on an indefinite block.** There too the span holds
  configurations below an excited reference, and the even ground state can be a near-null root of
  the cut or a lower configuration: `gcm_levels(level='overlap')` keeps the even ground state unless
  another even state overlaps the reference determinant more by over `SOFT_MODE_GCM_OVERLAP_MARGIN`
  (0.1), and orders the odd states above the kept state first, nearest first (the mirror of
  `'top'`). Square H4/6-31G b2u² b3u² (triplet), whose even ground state holds 0.000 of the
  determinant and the next 0.996, gave −6.24 eV from the bottom (3.35, 11.76, −6.24 and −6.24 eV
  over the cuts) and gives +3.356 eV (3.349 to 3.356 over the cuts; its strongest pole +6.38, FCI's
  strongest D-coupled transition +4.95); rectangular H4 at 1.5 × 1.0 Å keeps states of 0.96, 0.82
  and 0.61 over bottoms of 0.003, 0.115 and 0.233; linear H4/6-31G 1σ_g² 2σ_g² at 1.5 Å (singlet),
  whose even ground state holds the determinant (0.95) with an odd state 0.04 eV below it, gives the
  nearest odd state above, +5.55 eV (FCI's strongest D-coupled transition +3.98, nothing near 0).
  The poles are the same under any order. Where the span has no odd state above the kept state but
  the spurious high roots, the first transition is such a root: LiH/6-311G 1s² 1π_x² at 3.0 Å, its
  singlet's second soft mode, keeps the state of 0.856, 2.3 eV above the even ground state, and
  reports +66.0 eV (c = 0.001), its strong pole the de-excitation −0.67 eV (c = 0.92). The H2 σ_u²
  triplet rows keep their bottom state (0.87 to 0.96) and levels.
- **A near tie.** Where the two even states a rule picks between share the determinant, the kept
  level jumps with the geometry or the cut: rectangular H4/6-31G 1.5 × W Å b1u² b3g² (singlet, M < 0)
  keeps state 2 at W = 0.72 Å (0.4361 against 0.4356, −4.98 eV) and state 3 at 0.73 Å (−0.41 eV), as
  FCI's two singlets share it (0.406 / 0.404). Two candidates within `SOFT_MODE_GCM_OVERLAP_MARGIN`
  of each other are flagged `'soft_mode_reference_tie'` (also N2/STO-3G's π_u⁴ → π_g⁴ singlet at
  1.3 Å, 0.511 / 0.432, and square H4's triplet UHF at 1.2 Å in its spin-flip block, 0.057 / 0.034).
- **Larger bases** (measured, not held; FCI by a dense two-electron solve): the instability moves
  out (to between 1.2 and 1.5 Å in cc-pVDZ, 1.5 and 2.0 Å in cc-pVTZ), and at 1.2, 1.5, 2.0 and
  3.0 Å the 1D model gives −6.44, −3.93, −1.50, −0.09 (cc-pVDZ, FCI −6.57, −3.46, −1.03, −0.07) and
  −6.65, −4.01, −1.60, −0.21 (cc-pVTZ, FCI −7.08, −3.42, −1.05, −0.07), the GCM −6.04, −3.22,
  −1.04, −0.38 and −5.77, −3.25, −1.15, −0.67 eV.

Where it does badly (measured, not held):
- **Stiff modes.** The lowest mode is always split. The triplet block of σ_u² at 1.2–2.0 Å has a
  stiff negative transition as its lowest mode (to the b ³Σ_u⁺ state; TDHF −12.3 to −6.7 eV, FCI
  −13.1 to −9.8): the inverted 1D model gives −8.5 to −4.1 eV, 4.6–5.8 eV short, the GCM −12.7 to
  −11.7 (within 0.6 eV of FCI to 1.5 Å, 1.85 eV too negative at 2.0 Å). The aufbau singlet block
  of H2/cc-pVDZ (the B ¹Σ_u⁺ state at 0.75–2.0 Å, FCI 13.83 to 8.74 eV) is the same story: the 1D
  model is 1.6–4.5 eV low, TDHF 0.0–3.1 eV low, the GCM within 0.23 eV. On a stiff mode the
  anharmonicity of V over the whole path moves the 1D levels by eV from the harmonic one, as for
  the ground-state triplet's stiff modes; the GCM is the route for those (or a rule that splits
  soft modes only).
- **Near a Krein collision** the soft mode is stiff too (TDHF Ω = 12.6 and 14.8 eV at 0.76 and
  0.94 Å), and its 1D levels also raise `'soft_mode_edge_weight'`.

Not validated: anything beyond H2 against a reference method (the multi-pair rule above is checked
for its sign and its cut stability only). The inversion of a mode with M < 0 and μ > 0 is validated
on excited references only (σ_u²'s excited-state instability): on a ground-state determinant that
mode is a complex instability, which it does not describe (O2's UHF in the open shells below).
`reference='excited'` also runs on an aufbau reference whose A − B is indefinite (a complex
instability): ethylene twisted by 90° (cc-pVDZ), which the default refuses, gets a soft triplet of
M = −142 Ha⁻¹ and a level at −1.81 eV (the singlet −2.21 eV), flagged `'soft_mode_edge_weight'`, with
no reference to hold them against. The closed-shell drivers take `reference='excited'` as the
caller's statement that the reference is excited: its occupation does not tell, σ_u² being aufbau in
its own Fock spectrum from 1.2 Å on.

Not implemented: the relaxed path for excited references (a constrained saddle search) and for
singlets, and the quasiparticle driver on excited references. Krein quartets have the two-mode GCM
below, open-shell references the open-shell split further down.

## Krein collisions: the two-mode GCM

`collisions='gcm'` (opt-in, with `reference='excited'`, in `soft_mode_triplets` and
`soft_mode_singlets`) quantizes a Krein collision instead of refusing it. This is the mechanism (ii)
of Burth, Kitsaras and Loos: on an excited reference a positive excitation Ω_m meets the
positive-frequency partner of a negative transition Ω_l, and at Ω_m + Ω_l = 0 both turn complex, a
quartet ±Ω, ±Ω*, possible only where A + B and A − B are both indefinite. H2/6-31G σ_u² has two
windows, the singlet at 0.7625–0.77 Å and the triplet at 0.945–1.01 Å (the paper's "about 0.75"
and "0.9–1.0 Å"). The resonance means that the reference is degenerate, in TDHF, with the double
excitation that carries both modes; the analysis in the soft-mode notes proposed quantizing the
two modes on the determinants they generate, since a flat P²/2M model of two modes is ill-posed.

| step | quantity |
|---|---|
| pairs | `krein_pairs`: every quartet (a complex-conjugate Ω²), and every soft mode flagged as a Krein neighbour (within `SOFT_MODE_KREIN_GAP` of a mode of the other signature) with that neighbour when each is the other's nearest mode of the other signature (mutual nearest neighbours, over every mode of real Ω², so the pairs do not depend on the order); a soft mode whose nearest is in a degenerate set, or is another mode's mutual nearest neighbour (a near-collision of three modes), stays unpaired and flagged; a degenerate quartet is refused (`NotImplementedError`) |
| plane | the pair's real invariant plane of (A − B)(A + B), from its Schur vectors (well conditioned where the two eigenvectors coalesce), turned so that (A − B)⁻¹ is diagonal on it: e₁ of positive mass, e₂ of negative mass |
| why not P²/2M | (A − B)⁻¹ is indefinite on the plane, so Pᵀ(A − B)P/2 + V(Q) is bounded neither below nor above |
| generator states | Φ(κ) for every rotation κ in the plane (α by exp(K), β by exp(−K); both by exp(K) for singlets), on `SOFT_MODE_COLLISION_DIRECTIONS` (8) directions with the straight path's `SOFT_MODE_GCM_POINTS` (32) points each |
| kernels | `_CollisionKernels`: the generic non-orthogonal ones in corresponding orbitals, free of divisions (orthogonal determinants need no special case), over the occupied orbitals and the plane's virtual span |
| levels | `collision_levels`: Hill–Wheeler in the even and odd blocks of κ → −κ (for singlets the symmetry that takes the plane to minus itself, checked as for a soft direction), real by construction |
| reference | the even state of largest overlap with the reference determinant |
| poles | `collision_modes`, every odd state m: ΔE_m = E_odd,m − E_ref, X + Y = √2 Σ_k ⟨m\|D_k\|ref⟩ e_k, X − Y = ΔE_m (A − B)⁻¹(X + Y), so X'X − Y'Y = 2 ΔE_m c_mᵀ diag(M) c_m |

For one occupied orbital the plane's determinants span {u², v₁², v₂², v₁v₂} and {uv₁, uv₂} of the
channel's spin (u the occupied orbital, v_k = Σ_a (e_k)_a φ_a): the model is the CI in {u, v₁, v₂},
its odd states are the TDA roots of the plane (e₁, e₂ restricted), and the reference is u²
correlated with the plane's double excitations, which moves it by +0.1 to +0.4 eV on the 6-31G rows
below (+1.0 to +1.2 eV in cc-pVDZ's window, −0.4 to −0.7 eV in cc-pVTZ's at 0.7–0.9 Å). The paired modes leave the soft set and the
regular modes, so the record's `soft` can be empty (`omega_tda_soft` is then `None`), `omega2` is
complex, a soft mode's `krein_gap` leaves a quartet's modes out and counts a 'neighbours' pair's (so
that a soft mode left unpaired next to a pair is flagged), and `'soft_modes_coupled'` counts the
pairs beside the soft sets. The record adds `pairs`, per pair: `index`, `omega2`,
`kind` (`'quartet'` or `'neighbours'`), `gap`, `masses`, `half_periods`, `deexcitation` and
`excitation` (the poles of largest coupling below and above the reference), every pole's
`excitations` and `couplings`, `reference_overlap(s)`, `reference_shift`, `ranks`, `spin_square`,
`cut_scan` and `omega_tda`. Without a collision the record is `collisions='refuse'`'s to the bit,
with `pairs = []`.

```python
mol = gto.M(atom='H 0 0 0; H 0 0 0.98', basis='6-31g', symmetry=True)
mf = scf.RHF(mol)
mf.irrep_nelec = {'A1u': 2}
mf.run()
rec = soft_mode_triplets(mf, model='gcm', reference='excited', collisions='gcm')
pair = rec['pairs'][0]                            # a quartet: TDHF 15.11 ± 0.58i eV
print(pair['deexcitation'] * HARTREE_TO_EV, pair['excitation'] * HARTREE_TO_EV)
# −15.614 +15.057 eV (FCI −15.528 +15.329)
```

**Measured against FCI** (eV; H2/6-31G σ_u²; FCI from the singlet of largest σ_u² weight to the
states of the channel's spin, weighted by |⟨m\|D₁\|ref⟩|² + |⟨m\|D₂\|ref⟩|²: its strongest state
below and above, and for the singlet's excitation, which FCI splits between two states near 8.5 and
18 eV, their weighted centroid in brackets; "one-mode" is `collisions='refuse'` with the GCM, its
soft mode's level; * flagged):

| R (Å) | pair | TDHF | TDA | one-mode | GCM down | GCM up | FCI down | FCI up | overlap |
|---|---|---|---|---|---|---|---|---|---|
| 0.94 (triplet) | neighbours | −15.24, +14.75 | −15.903, +14.506 | +7.99 | −16.170 | +14.239 | −16.083 | +14.485 | 0.961 |
| 0.945 | quartet | 15.01 ± 0.22i | −15.829, +14.613 | refused | −16.098 | +14.343 | −16.012 | +14.592 | 0.960 |
| 0.95 | quartet | 15.03 ± 0.37i | −15.755, +14.718 | refused | −16.027 | +14.446 | −15.941 | +14.699 | 0.960 |
| 0.98 | quartet | 15.11 ± 0.58i | −15.326, +15.344 | refused | −15.614 | +15.057 | −15.528 | +15.329 | 0.958 |
| 1.0 | quartet | 15.17 ± 0.41i | −15.050, +15.752 | refused | −15.349 | +15.453 | −15.265 | +15.738 | 0.957 |
| 1.01 | quartet | 15.20 ± 0.05i | −14.916, +15.952 | refused | −15.220 | +15.648 | −15.137 | +15.940 | 0.956 |
| 1.015 | neighbours | −14.89, +15.53 | −14.849, +16.051 | −10.75 | −15.156 | +15.744 | −15.073 | +16.039 | 0.956 |
| 0.755 (singlet) | neighbours | +12.50, −12.99 | −13.347, +12.500 | +10.03 | −13.589 | +12.258 | −14.521 | +8.454 (12.351) | 0.925 |
| 0.76 | neighbours | +12.64, −12.83 | −13.215, +12.612 | +5.88 | −13.499 | +12.328 | −14.388 | +8.488 (12.453) | 0.802 |
| 0.7625 | quartet | 12.73 ± 0.07i | −13.149, +12.668 | refused | −13.470 | +12.348 | −14.322 | +8.504 (12.503) | 0.681* |
| 0.765 | quartet | 12.72 ± 0.11i | −13.084, +12.724 | refused | −13.456 | +12.352 | −14.256 | +8.521 (12.554) | 0.529* |
| 0.77 | quartet | 12.71 ± 0.05i | −12.955, +12.836 | refused | −13.064 | +12.727 | −14.125 | +8.555 (12.653) | 0.710* |
| 0.7725 | neighbours | −12.58, +12.82 | −12.890, +12.893 | −11.19 | −13.018 | +12.765 | −14.060 | +8.572 (12.701) | 0.783 |
| 0.775 | neighbours | −12.50, +12.89 | −12.826, +12.949 | −12.40 | −12.966 | +12.809 | −13.995 | +8.589 (12.746) | 0.830 |

- **Real, and smooth through the triplet window.** Where TDHF is complex and the default refuses,
  the model's poles are real, of positive norm, and their second differences across both edges of
  the triplet window are 6·10⁻⁴ eV, where TDHF's two real roots meet at a square-root branch point
  and the two modes exchange their order (the positive one is the lower below the window, the
  negative one above it).
- **Triplet**: within 0.09 eV of FCI below (−0.083 to −0.087) and 0.30 eV above (−0.246 to
  −0.295). **Singlet**: 0.80–1.06 eV above FCI below; above, FCI splits the strength between two
  states and the model's one pole lies within 0.20 eV of their centroid (−0.20 to +0.07), 3.8–4.2 eV
  above the stronger one.
- **Against TDA**, real as well (the paper's recommendation): the model is closer to FCI for the
  singlet (0.80–1.06 against 1.17 eV below, 0.06–0.20 against 0.15–0.20 above) and the triplet's
  de-excitation (0.08 against 0.18–0.22), and farther for the triplet's excitation (0.25–0.30
  against 0.01–0.02): for one occupied orbital its odd levels are TDA's, measured from the
  correlated reference. Against the one-mode route, which refuses inside the windows and is
  1.6–6.5 eV from FCI's strongest state at the neighbour rows, it is the better near-collision
  treatment throughout.
- **The reference by overlap.** In the singlet window the reference determinant is shared by two
  even states of the model, an avoided crossing with the plane's double excitation that FCI does
  not show (its reference state keeps 0.975 of σ_u²); the kept state switches between 0.765 and
  0.766 Å (0.529 / 0.520) and the de-excitation jumps by 0.34 eV there. Below
  `SOFT_MODE_COLLISION_OVERLAP_MIN` (0.75) this is flagged `'soft_mode_collision_reference'`.

Where it does badly (measured, not held):
- **Larger bases** (H2 σ_u², FCI by a dense two-electron solve). cc-pVDZ's triplet window
  (1.25–1.4 Å): −0.19 to −0.32 below and −1.32 to −1.45 eV above, where TDA is within 0.04–0.22 and
  0.26–0.33 eV (the reference raised by 1.0–1.2 eV). cc-pVTZ's triplet at 0.7–0.9 Å: −0.39 to −0.41
  below (TDA 0.9–1.2 eV off), −1.0 to +0.4 above. cc-pVTZ's triplet at 1.7–1.8 Å and singlet at
  0.5 Å: the reference shared 0.50 / 0.51 (flagged), and the levels 3.6–8.8 eV off.
- **More occupied orbitals.** Linear H4/6-31G on 1σ_g² 2σ_g² (two occupied; quartets in the singlet
  at 0.70–0.90 Å and the triplet at 1.10–2.0 Å): the reference determinant is shared on every row
  (0.47–0.60, flagged), as it is in FCI, whose reference state holds only 0.25–0.56 of it; the
  levels are within 0.2 eV below and 0.65 eV above FCI at 0.80 and 0.85 Å, 5–6 eV off on one side at
  0.70, 0.75 and 0.90 Å (the pick by overlap a near tie between states 5.5 eV apart, flipping with
  the overlap cut), and 1.2–4.2 eV off at 1.1–1.5 Å in the triplet (9.2 eV above at 1.8 Å). Linear
  H6/6-31G (three occupied) runs, 0.6 s a pair, with the reference shared on 9 of its 15 pair rows
  (0.47–0.95); its FCI (six electrons) is not run here.
- **Reach.** Pairs form only within `SOFT_MODE_KREIN_GAP` (0.1) of a collision. Just beyond it the
  one-mode split is still poor for these stiff modes: H2/6-31G's triplet soft mode misses FCI by
  1.2–5.0 eV at gaps of 0.12–0.25 (0.93–0.92 and 1.02–1.05 Å) and by 0.1–1.4 eV at 0.29–0.39, where
  the two-mode model, forced to pair them, stays within 0.1 eV below and 0.42 eV above out to a gap
  of 0.58 (1.2 Å).

Not done: the relaxed path and the quasiparticle driver for collisions, as for excited references;
degenerate collisions (refused) and near-collisions of three or more modes (the mutual nearest
neighbours pair, and a soft mode next to their pair is left to the one-mode split, flagged
`'soft_mode_krein_neighbour'`: Ω² of 1.00, 1.06 and 1.07·10⁻³ Ha² with signatures +, −, + pair the
last two, the first flagged at a gap of 0.057); a reference choice other than the largest overlap
where it is near a tie. Open-shell collisions are in the next section.

## Open shells: UHF blocks, spin flip and core holes

The paper of Burth, Kitsaras and Loos (arXiv:2610.36154) has two more cases, both on open-shell UHF
references: full spin-flip RPAx on high-spin triplets (cyclobutadiene, complex at both geometries)
and Δ-RPAx on core holes (BF⁺ F1s with two solutions, the delocalized N2⁺ 1s hole). The analysis in
the soft-mode notes found that the harmonic soft-mode expansion does not apply to them as such: the
spin-flip targets are the broken-symmetry states, A − B is indefinite there, and a delocalized core
hole is a high-index saddle best reached from the localized solutions by GCM/NOCI. What can be built
and checked on a workstation is implemented, opt-in, and validated on three- and four-electron
models against FCI; the paper's own systems are rows of a cluster test; the rest is a design below.

`soft_mode_open_shell(mf, block, model, collisions)` splits one TDHF block of a UHF reference
sign-aware, as `soft_mode_singlets` splits a closed shell's, and `noci_levels(solutions)` mixes
given UHF solutions by non-orthogonal CI.

| step | quantity |
|---|---|
| reference | `open_shell_reference`: any stationary UHF (aufbau, broken-symmetry, a maximum-overlap core hole; the `SOFT_MODE_EXCITED_GRADIENT_MAX` test per spin), each spin's orbitals reordered occupied-first (`orbitals` in the record maps back); RHF, ROHF and GHF refused, n_α ≥ n_β |
| blocks | `uhf_tdhf_matrices`: spin-conserving (α→α pairs, then β→β) A = Δε + (ia\|jb) − δ_στ(ij\|ab), B = (ia\|jb) − δ_στ(ib\|ja), pyscf's UHF TDHF; spin-flip (α→β pairs, then β→α) A = Δε − (ij\|ab) within each sector, B = −(ib\|ja) between them, so that D(A + B)D = A − B (D = ±1 on the sectors). Both are `LinearResponseSolver`'s unrestricted builders; on a closed shell the spin-flip block is the triplet twice (M_s = ±1) and the spin-conserving one the singlet and the triplet's M_s = 0 |
| split | `krein_split` (or `krein_pairs` with `collisions='gcm'`) on the block: always sign-aware, since A − B is indefinite for a core hole (its core-filling transitions, Ω ≈ −IP) and in the spin-flip block whenever A + B is |
| spin rotation | the spin-flip block of a reference with n_α > n_β holds a global spin rotation, g = (⟨aβ\|iα⟩, −⟨aα\|iβ⟩) (`spin_rotation_direction`): (A + B)g = 0, (A − B)Dg = 0, gᵀDg = n_α − n_β. It is taken out exactly (`_spin_rotation_split`): with Q an orthonormal basis of g's complement and P = 1 − g(Dg)ᵀ/(gᵀDg), the RPA of (QᵀAQ, QᵀBQ) is the block's, X + Y = PQ(X_r + Y_r), X − Y = Q(X_r − Y_r); its own pole sits at Ω = 0 (the M_s − 1 component of the reference's multiplet, X'X − Y'Y = 1) |
| path | κ = Q d/√2 over the block's spin-orbital pairs, the spin orbitals turned by exp(K), K_ai = κ_ia: a UHF determinant in the spin-conserving block (collinear), a GHF one in the spin-flip block (non-collinear); V''(0) = μ, D(Q) = 2^−½ Σ d_P (a†_a a_i + a†_i a_a) = Q to first order, L = π/(√2 σ₁) (`open_shell_path`); in the spin-flip block d is taken orthogonal to g, which leaves μ |
| kernels | `_OpenShellKernels`: the generic non-orthogonal kernels between spin-orbital determinants, corresponding orbitals over all N spin orbitals (a GHF pair needs no spin blocks), free of divisions, with ⟨S²⟩ from the one-electron spin vectors; each spin's active basis its occupied orbitals and the virtual span the generators reach |
| models | 1D and GCM (`open_shell_gcm_levels`) for a soft mode, its level the bottom (M > 0; in the GCM on a block whose A − B is indefinite by overlap, `level='overlap'`) or top (M < 0, by overlap) one, as in `gcm_levels`; the two-mode GCM (`open_shell_collision_levels`) for a collision. The parity Q → −Q is exact in the spin-flip block (the spin rotation by π about z: the even block keeps the reference's M_s parity, the odd one the other) and checked in the spin-conserving block (`SOFT_MODE_OPEN_SHELL_PARITY_MAX`, 10⁻³ of max \|V\|; 5·10⁻¹⁶ to 4·10⁻¹³ for directions a symmetry takes to minus themselves, 10⁻² to 0.27 for most totally symmetric ones) |
| NOCI | `noci_levels`: Hill–Wheeler between the given solutions on the same kernels (each spin's span of their occupied orbitals, exact), canonical orthogonalization at `SOFT_MODE_GCM_OVERLAP_CUT`; fewer states than solutions is flagged `'soft_mode_noci_rank'`; solutions of different molecules are refused before any work, the atoms, electrons and basis compared shell by shell (AO labels alone do not tell He2⁺'s 6-31G from its 3-21G, whose mixed CI gave an overlap of −1.02) |

The record is `soft_mode_singlets`' with `block`, `spin_rotation` (its column, the residual
\|(A + B)g\|/\|g\| and gᵀDg) and, with `collisions='gcm'`, `pairs`; its flags are the closed shell's
(`'soft_mode_gcm_degenerate_pairs'` when two pairs of one spin block are as strong as the leading
pair, `'soft_mode_gcm_few_states'` below one spin-orbital pair's exact span, one even and one odd
state: He2⁺/STO-3G turns a single β spin orbital, its GCM keeps exactly those two and equals FCI's
gap to 2·10⁻¹⁴ eV, where the closed shell's two-even, one-odd test flagged every row;
`'soft_mode_reference_tie'` as for the closed shells), and `'soft_mode_complex_instability'`
(below). Refused besides the reference's checks: the spin-flip block with n_α = n_β, from the
occupations alone before any Fock build (`NotImplementedError`: there each spin-flip mode comes with
its M_s partner, degenerate and of the other signature, and a broken-symmetry M_s = 0 UHF has a
spin-rotation mode of null norm), the 3D model, a spin-conserving direction or plane whose path is
not even.

**A complex instability of a ground-state determinant.** The inversion of a soft mode with M < 0
and μ > 0 is the excited-state instability of an excited reference (σ_u²); on a ground-state
determinant that mode is a complex instability instead, which the inversion does not describe.
O2/STO-3G's default triplet UHF at 1.21 Å, aufbau in both spins and a saddle in real rotations as
well (A + B has a degenerate pair at −0.0327 Ha, pyscf's internal stability check fails), has one
in its spin-conserving block: A − B's lowest eigenvalue −0.0278 Ha, M = −35.97 Ha⁻¹, μ = +0.330 Ha.
Inverted, it gives −2.45 eV in 1D and −14.69 eV in the GCM (whose kept state holds 0.063 of the
determinant against the even ground state's 0.937, flagged `'soft_mode_reference_overlap'`; from
that ground state the transition is +5.37 eV), where FCI (1200 determinants) keeps 0.934 of the UHF
determinant in its ground state, has no state below it (first excitation +6.16 eV) and its
strongest D-coupled transition at +6.27 eV. Following the real instability reaches a UHF 1.5 mHa
lower whose A − B is positive definite, with no such mode; O2's default UHF in 6-31G and cc-pVDZ
has none either. `soft_mode_open_shell` flags such a mode on a reference aufbau in both spins (each
spin's occupied orbitals below its virtual ones) with `'soft_mode_complex_instability'`
(`SoftModeExcitedShort`), naming A + B's lowest eigenvalue when it is negative too, and keeps the
levels. The occupation is only a sign of a ground-state determinant: H2's σ_u² given as a UHF,
aufbau in its own Fock spectrum from 1.2 Å on, is flagged too from 1.34 Å on, where the inversion
is the validated one.

```python
from src.SingleReference.LinearResponse.soft_mode import soft_mode_open_shell

mol = gto.M(atom='He 0 0 0; He 0 0 2.0', basis='cc-pvdz', charge=1, spin=1, symmetry=True)
mf = scf.UHF(mol).run()                          # the symmetric sigma_g^2 sigma_u hole
rec = soft_mode_open_shell(mf, 'spin_conserving', 'gcm')
print(rec['soft'][0]['excitation'] * HARTREE_TO_EV)   # 1.361 eV, 2Sg+ - 2Su+ (FCI 1.207; TDHF 1.26i)

mol = gto.M(atom='H 0 0 0; H 1.5 0 0; H 1.5 1.5 0; H 0 1.5 0', basis='6-31g', spin=2)
mf = scf.UHF(mol).run()                          # the high-spin triplet of square H4
rec = soft_mode_open_shell(mf, 'spin_flip', 'gcm', collisions='gcm')
pair = rec['pairs'][0]                           # TDHF 2.19 +- 3.87i eV, refused by default
print(pair['deexcitation'] * HARTREE_TO_EV)      # -0.092 eV to the singlet (FCI -0.432)
```

**He2⁺, the hole-localization model** (cc-pVDZ, three electrons; the σ_g²σ_u UHF; FCI's
²Σ_g⁺ − ²Σ_u⁺ gap; NOCI of the two UHF solutions with the hole on either atom, from guesses
(σ_g ± σ_u)/√2; eV; * flagged `'soft_mode_edge_weight'`):

| R (Å) | Ω² (Ha²) | M | TDHF | 1D | GCM | NOCI | FCI |
|---|---|---|---|---|---|---|---|
| 1.08 | +0.11291 | 2.50 | 9.143 | 11.290* | 10.269 | – | 9.997 |
| 1.25 | +0.04449 | 3.66 | 5.740 | 7.683* | 7.001 | – | 6.735 |
| 1.5 | +0.00862 | 6.37 | 2.526 | 4.396* | 4.029 | – | 3.795 |
| 1.75 | −0.00056 | 11.1 | 0.645i | 2.528 | 2.338 | 2.273 | 2.146 |
| 2.0 | −0.00214 | 19.5 | 1.258i | 1.448 | 1.361 | 1.305 | 1.207 |
| 2.5 | −0.00117 | 63.3 | 0.930i | 0.430 | 0.473 | 0.401 | 0.364 |
| 3.0 | −0.00038 | 223 | 0.529i | 0.076 | 0.194 | 0.114 | 0.103 |
| 4.0 | −0.00002 | 4785 | 0.117i | 0.000 | 0.089 | 0.005 | 0.005 |

- The soft mode (the β σ_g → σ_u rotation that moves the hole onto one atom) has M > 0 throughout
  and μ < 0 past 1.75 Å: the ground-state double well of the closed shell's triplet, here in the
  spin-conserving block of an open shell. TDHF turns imaginary there; the GCM stays within 0.27 eV
  of FCI on every row, and the 1D model within 0.38 eV from 1.75 Å on. As for H2's triplet, the
  constant mass makes the 1D tunnel splitting vanish deep in the broken phase (0.000 eV at 4 Å,
  FCI 0.005), and on the stiff mode before the instability the 1D model is 0.6 to 1.3 eV high
  (flagged), the GCM 0.23 to 0.27 eV.
- The localized solutions exist from 1.75 Å on (0.004 to 0.64 eV below the symmetric one). Their
  NOCI, the route the analysis proposed for delocalized core holes, is within 0.13 eV of FCI and
  closer than the GCM on every row, to the meV at 4 Å: the relaxed localized orbitals are what the
  straight path from the symmetric solution lacks. At 1.5 Å both guesses fall back to the symmetric
  solution, and the CI of the two is flagged `'soft_mode_noci_rank'`.

**Square and rectangular H4, the four-electron model of cyclobutadiene's spin flip** (6-31G, the
high-spin triplet UHF; FCI's lowest singlet and quintet relative to its triplet; SF-TDA the lowest
root of the α→β sector of A; eV; * flagged `'soft_mode_collision_reference'`):

| sides (Å) | SF-TDHF √Ω² | GCM down | GCM up | FCI singlet | FCI quintet | SF-TDA | overlap |
|---|---|---|---|---|---|---|---|
| 1.5 × 1.5 | 2.189 ± 3.869i | −0.092 | +5.627 | −0.432 | +5.646 | −0.476 | 0.870 |
| 1.4 × 1.6 | 2.428 ± 3.850i | −0.446 | +5.721 | −0.986 | +5.748 | −1.004 | 0.867 |
| 1.3 × 1.7 | 3.073 ± 3.815i | −1.350 | +6.014 | −2.071 | +6.056 | −2.189 | 0.860 |
| 2.0 × 2.0 | 0.344 ± 4.186i | −0.127 | +1.493 | −0.390 | +1.565 | −1.166 | 0.647* |

- As in cyclobutadiene, full spin-flip TDHF is complex on every row: the negative transition to
  the singlet (M_s 1 → 0) collides with the positive one to the quintet (M_s 1 → 2) through B,
  a quartet whose real part is 2 to 4 eV off. The two-mode GCM on the non-collinear determinants
  of their plane gives real poles: the quintet 0.02 to 0.07 eV below FCI, the singlet 0.26 to 0.72 eV
  above it (the singlet–triplet gap too small). Its singlet is spin-contaminated (⟨S²⟩ 0.15 to
  0.45): the parity blocks separate M_s parities, not spins, and the reference state is correlated
  by 1.7 eV (3.3 eV at 2.0 Å) by the plane's double spin flips, more than the singlet.
- SF-TDA, the paper's recommendation, is closer to FCI on the three rows near 1.5 Å (0.02 to
  0.12 eV), the two-mode model on the stretched square (0.26 against 0.78 eV). The model is the
  route where TDHF is complex, not a replacement for SF-TDA.

**Measured besides** (not held):
- Cyclobutadiene in STO-3G at the paper's geometries: SF-TDHF 3.751 ± 4.538i (D2h) and
  3.245 ± 4.411i eV (D4h); the two-mode GCM puts the triplet 1.584 and 0.554 eV above the singlet,
  against 2.217 and 0.777 eV for a π CASSCF(4,4) (state-specific orbitals) and 2.42 and 0.88 eV for
  SF-TDA; 140 to 150 s a pair on four threads (an active space of 24 + 24 spin orbitals, 256
  generator states).
- N2⁺ 1s in cc-pVDZ (maximum overlap from the neutral's orbitals): Δ-UHF 412.19 eV for the hole on
  one atom, 420.71 eV for the symmetric hole. The localized hole's blocks are real (soft modes: a
  spin-conserving π pair, 8.72 eV in 1D and 8.27 eV in the GCM, and a spin-flip one, 3.05 and
  4.39 eV); the symmetric hole's are complex in both blocks (√Ω² 409.88 ± 1.44i and
  408.58 ± 5.30i eV, core-filling transitions meeting excitations at the core IP), its quartets in
  degenerate π pairs, which the two-mode model refuses. The NOCI of the hole on either atom gives
  412.17 and 412.21 eV (overlap 1.6·10⁻³).
- BF⁺ F1s in aug-cc-pVDZ: 697.26 eV, its spin-conserving block real in this basis; its CVS Hessian
  (A + B over the pairs that do not fill the hole) has a negative π pair (−0.045 Ha), and turning
  the orbitals by 0.3 along it and reconverging reaches a second solution 0.128 eV lower (the
  paper: 0.12 eV), real in the spin-conserving block and complex in the spin-flip one, as in the
  paper; their NOCI lies 0.065 eV lower still (overlap −0.90).

### Design: what is not built

What the paper's remaining cases need beyond the above, with the cluster runs that would test it:

1. **Spin-projected GCM for spin flip.** The non-collinear generator states mix M_s, and the
   parity blocks separate only M_s parity (R_z(π)), which leaves H4's singlet 0.15 to 0.45 in ⟨S²⟩
   and its gap 0.26 to 0.72 eV small. Projecting, |Ψ_SM⟩ = P^S_{MK} Σ_j f_j |Φ(κ_j)⟩ with
   P^S_{MK} = (2S + 1)/(8π²) ∫ dΩ D^S_{MK}(Ω)* R(Ω), needs the kernels ⟨Φ(κ)|H R(Ω)|Φ(κ′)⟩ of
   spin-rotated, complex GHF determinants (those here are real) on a quadrature in Ω (for M_s
   alone, R_z(φ) on a few φ). Test: H4/6-31G against FCI (local), cyclobutadiene/cc-pVTZ against
   EOM-SF-CCSD(fT) (cluster).
2. **Degenerate collisions.** The symmetric N2⁺ hole's quartets come in π pairs (also cyclobutadiene's
   at D4h, if they are degenerate there): a four-mode model on the invariant subspace of the pair of
   quartets (four Schur vectors, the metric (A − B)⁻¹ diagonalized on it, two masses of each sign),
   its generator states on a 4D grid (or on the two planes with their cross kernels), at about
   (directions)² times the cost of a pair. Refused today (`NotImplementedError` in `krein_pairs`).
3. **CVS-constrained paths for core holes.** The core-filling directions (valence → hole, Ω ≈ −IP)
   are negative transitions that make A − B indefinite and drive the quartets near the IP; with
   them removed (A_CVS, B_CVS over the pairs that do not fill the hole) the hole's block can be
   stable, and V(Q) and the relaxed path should be evaluated on the CVS manifold (the hole kept
   empty), where a constrained minimization does not collapse to the ground-state cation. Needs: a
   CVS block in `uhf_tdhf_matrices`, the relaxed path of `relaxed_path` for UHF with the hole held,
   and the CVS-FCI references of Marie, Burth and Loos (2026). Test: BF⁺, N2⁺, CO⁺, H2O⁺ and CO2⁺ in
   aug-cc-pCVTZ against Δ-CVS-FCI (cluster).
4. **Relaxed paths for open shells**, in both blocks (the non-collinear one with the spin rotation
   held), and the GCM on relaxed paths: the straight path's GCM misses He2⁺'s tunnel splitting at
   4 Å by 0.08 eV where the NOCI of the relaxed localized solutions is exact to the meV.
5. **The quasiparticle driver on open shells.** `GW.soft_mode_qp` takes closed shells only; the
   unrestricted PSD self-energies of the house (their spin-flip triplet sums) would take the split
   spectrum of the spin-flip block, with the spin-rotation pole at 0 dropped or treated as the
   multiplet's own state.
6. **Correlation energies** (the paper's plasmon formula on the split spectrum): an anharmonic
   counterpart of the subtracted A terms and a double-counting rule for the GCM reference's own
   correlation (1.7 eV for H4's triplet), not attempted.

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
  - A turning point of Q (butadiene/STO-3G, two soft modes): with the continuation switched off
    both paths freeze (0.81 and 0.74 L) and stay open, at the levels of the code before it, and a
    continuation that fails at its first step leaves their records as they were without it, in
    every field (energy calls, curvature, holds); the other soft mode's hold carries no force
    (10⁻⁸) and Q rises to a maximum before the antipode and falls back; continued in their arc
    length both paths close, minima across the path on their slices, with M(x) = M to 2.4·10⁻³ and
    at their recorded levels (3.570698 and 5.085984 eV, to 10⁻⁴ eV).
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

## Checked in `tests/test_soft_mode_excited.py`

- The singlet matrices from MO blocks equal the house builder (H2, H2O, and H2/6-31G σ_u² on its
  reordered orbitals), with the triplet's A − B.
- The aufbau limit: `krein_split` is `soft_split` to the bit (H2O, both blocks),
  `reference='excited'` is the default to 3·10⁻¹⁶ (H2/cc-pVDZ at 1.0 and 1.5 Å, all three models),
  and the singlet soft mode at its harmonic level is the house singlet Casida root.
- The split of H2/6-31G σ_u²: μ/M = Ω² and dᵀe = 1; the regular modes solve the RPA equations with
  X'X − Y'Y = 1 and Ω = sign(M)|Ω|; the harmonic soft pole is the positive-norm RPA eigenpair at
  Ω < 0 (M < 0); the signs of M and μ along R, the singlet instability between 1.33 and 1.34 Å.
- The singlet path: V''(0) = μ (M < 0, and H2O), D(Q) = sin Q, V even; the singlet GCM kernels
  against a generic non-orthogonal evaluation (H2O/STO-3G, H2O/cc-pVDZ, σ_u²) and their diagonal
  against `path_samples`.
- σ_u²'s singlet symmetry breaking and short-bond triplet against FCI, the table above: real
  levels of the right sign where TDHF is imaginary, the 1D model within 0.7 eV, the GCM within 0.4
  (singlet) and 0.1 eV (triplet) and equal to the two-orbital CI from its top (singlet) or bottom
  (triplet) even state.
- A maximum-overlap reference (a plain RHF, σ_u above σ_g at 0.3 Å so that mo_occ is [0, 2, 0, 0])
  gives the symmetry-constrained levels to 3·10⁻⁶ eV.
- The aufbau singlet block of H2/cc-pVDZ against FCI's B state: the GCM within 0.3 eV.
- The refusals (unknown reference, the relaxed path, the 3D singlet and the singlet's relaxed path
  on an aufbau H2 the driver would otherwise split, open-shell and non-stationary occupations, a
  complex Ω², a totally symmetric singlet soft mode), each by its class and message, and both new
  flags, every other row of the FCI tables in both models raising none.
- A degenerate soft π pair with A − B indefinite (N2/6-31G 3σ_g² → 3σ_u², singlet): every mode an
  eigenvector (5.6·10⁻¹⁵ relative), the RPA equations held, one mass and one level for the pair.
- The reference of a negative mass in a multi-pair GCM (LiH/6-31G 1s² 1π_x²): the even state of
  largest overlap above the ground state, not the top root; a de-excitation, the same at every cut,
  no overlap flag.
- The reference of a positive mass on an indefinite block (H4/6-31G excited closed shells): square
  H4's b2u² b3u² triplet keeps the even state of largest overlap (0.996) over its near-null bottom,
  a positive first transition, the same over the cuts; linear H4's 1σ_g² 2σ_g² singlet keeps its
  even ground state and reports the nearest odd state above it first; rectangular H4 at
  1.5 × 0.72 Å (b1u² b3g², singlet) raises `'soft_mode_reference_tie'`.

## Checked in `tests/test_soft_mode_collisions.py`

- Without a collision `collisions='gcm'` changes nothing: `krein_pairs` is `krein_split` to the bit
  (H2O, both blocks; H2/6-31G σ_u² at 1.5 Å singlet and 0.3 Å triplet), and so are the records.
- The plane of a quartet and of neighbours (σ_u² triplet at 0.98 and 0.94 Å, singlet at 0.765 and
  0.76 Å): invariant, carrying the pair's Ω², orthonormal, its metric diagonal with one mass of
  each sign (to 9·10⁻¹⁶); the default refuses the quartets and flags the neighbours.
- The kernels: against the exact two-electron vectors of one occupied orbital, four orthogonal
  pairs included (to 2·10⁻¹⁵ Ha), and against the generic non-orthogonal rules with two and five
  occupied orbitals (H4, H2O/STO-3G, H2O/cc-pVDZ; to 3·10⁻¹³ Ha), their diagonal against
  `path_samples`.
- One occupied orbital: the model is the CI in {u, v₁, v₂} (to 10⁻¹⁴ Ha), its odd levels the TDA
  roots of the plane (to the reference's convergence, 1.5·10⁻¹⁰ Ha), spin-pure, on any grid.
- The table above against FCI: real poles of positive norm on every row, the triplet within 0.1 eV
  below and 0.35 eV above, the singlet within 1.1 eV below and 0.25 eV of FCI's centroid above;
  smooth across the triplet window's edges (second differences 6·10⁻⁴ eV); the singlet's reference
  switch at 0.765–0.766 Å flagged, no flag on the triplet rows.
- Linear H4 (two occupied): finite levels, more states than a pair's span, the shared reference
  flagged as FCI shares it (printed against FCI over all 784 states, not held); the soft mode's
  `krein_gap` leaves the quartet out, and `'soft_modes_coupled'` counts it.
- The pairing rule next to a degenerate set (a synthetic block): a soft mode whose nearest mode
  of the other signature is a degenerate pair stays unpaired, and so does the mode nearest to it.
- Three near-colliding modes (synthetic blocks, signatures +, −, + and +, +, −): the mutual nearest
  neighbours pair, and the third stays soft, flagged as a Krein neighbour of the pair; with four
  (−, +, −, +) the first two pair and the other two, the third one's nearest in that pair, stay
  unpaired and flagged.
- The poles' identities, `collision_modes` with one direction = `collective_modes`; the refusals
  (an unknown option, `'gcm'` on an aufbau reference or the relaxed path, a degenerate quartet, a
  totally symmetric singlet plane, the quartet under the default) and the few-states flag.

## Checked in `tests/test_soft_mode_open_shell.py`

- The blocks: the spin-conserving one gives pyscf's UHF TDHF roots (H2O triplet, 2·10⁻¹³ Ha) and
  both are `LinearResponseSolver`'s unrestricted builders (4·10⁻¹⁵); on a closed shell the
  spin-flip block is the triplet twice and the spin-conserving one the singlet and the triplet
  (10⁻¹² Ha²); D(A + B)D = A − B.
- A maximum-overlap O1s hole of H2O⁺ reordered occupied-first; an occupation set without
  reconverging refused.
- The kernels against the exact Fock-space vectors of random GHF determinants (M_s mixed; H4/STO-3G
  triplet, He2⁺/6-31G): N, H, a one-body operator and S² to 10⁻¹⁵ Ha; the per-spin active spaces
  of a spin-conserving and a spin-flip plane against the complete basis (10⁻¹⁴).
- The paths: V''(0) = μ and D'(0) = 1 (10⁻⁷, central differences) in both blocks; a closed shell's
  triplet direction gives `path_samples`' path in both (10⁻¹⁵).
- The closed-shell limit: H2/cc-pVDZ at 1.0 and 1.5 Å as a UHF gives `soft_mode_triplets`' levels,
  1D to 3·10⁻¹⁰ eV and GCM to 10⁻¹⁴ eV.
- The spin rotation: (A + B)g and (A − B)Dg at the SCF's convergence (4·10⁻¹⁰ to 10⁻⁸ relative),
  gᵀDg = n_α − n_β; the reduced spectrum is the block's without its zero (3.5·10⁻¹⁰ Ha²); the
  regular columns and the spin-rotation pole solve the RPA equations with X'X − Y'Y = 1; a spin-flip
  soft mode's harmonic pole through the reduced split is the dense RPA eigenpair (2·10⁻¹⁵, He2⁺ at
  1.08 Å).
- He2⁺ and H4 against FCI, the tables above, with their flags (none on H4's other rows);
  He2⁺/STO-3G's single-spin GCM equal to FCI with ranks (1, 1) and no few-states flag; the
  refusals, by class and, where a later refusal would hide behind another, by message, the
  spin-flip block with n_α = n_β before any Fock build and `noci_levels` over two bases with the
  same AO labels among them before any work.
- O2/STO-3G's default triplet UHF: its spin-conserving soft mode with M < 0 and μ > 0 flagged
  `'soft_mode_complex_instability'` in 1D and in the GCM, its modes of positive mass not.

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
  every variant's level is finite; every relaxed path is closed and a minimum across the path
  through its antipode (unfrozen, converged, no sample from a fresh start, no negative curvature at
  its probes, which are on the slices for a path continued in its arc length), as on the 34 rows a
  workstation can run; and the relaxed path's mean absolute error
  lies below the straight path's, as on the small systems (0.372 against 0.505 eV on the 14). In
  job 1242171 every row held this but hexatriene and octatetraene, whose paths froze at a turning
  point of Q; they are now continued in their arc length (above), not yet run on the cluster.
- BF in aug-cc-pVDZ and aug-cc-pVTZ (QUEST geometry): the relaxed path is a minimum at the sample
  nearest Q* (dense constrained Hessian) and lies more than 1 mHa below the straight one at L/2,
  and it keeps the TDHF harmonic limit (V''(0) = μ to 10⁻⁵ at Q = 10⁻², M(0) = M to 10⁻⁸), a check
  of the RHF and the split whose scatter over RHF initial guesses is 2·10⁻⁶.
- Printed for the record, not held: the GCM's levels with their ranks, ⟨S²⟩, couplings and cut
  scans; the relaxed paths' antipodes, inertia, closure, held modes, curvature and cost; errors by
  group; BF's PSD2/4/7/9 HOMO with the plain TDHF triplets and with each variant, with their Z,
  and G0W0@HF and Σ_TDHF@L_TDHF as controls.

## Checked on a cluster: `tests/test_soft_mode_open_shell_full.py`

The paper's own open shells, at its geometries and in its bases (not run on a workstation):
- Cyclobutadiene in cc-pVTZ (D2h, D4h), the high-spin triplet UHF (pyscf's solution and, if
  another, the one its internal instabilities lead to: at D2h in STO-3G the first is unstable and
  the second 5.3 mHa lower; one of them must be the paper's): the paper's SF-TDA-RPAx energies
  among the α→β TDA roots, its complex SF-RPAx (one quartet at √Ω² = 3.556 ± 2.552i and
  3.157 ± 2.084i eV: the triplet's row of Tables S1 and S2 is −Re Ω_X, since its M_s = 0
  component is the spin-rotation pole at 0) and its other rows less Re Ω_X among the real roots,
  each to 0.01 eV; the default's refusal; the two-mode GCM's real poles with the singlet below the
  triplet, printed against EOM-SF-CCSD(fT) (1.516 and 0.163 eV).
- N2⁺ 1s in aug-cc-pCVTZ: Δ-UHF of the localized and symmetric holes (409.42, 419.25 eV, to
  0.05 eV), the localized blocks real and the symmetric spin-flip block complex, the NOCI of the
  hole on either atom (two states within 0.2 eV, below the localized Δ-UHF) printed against
  Δ-CVS-FCI (409.88 eV).
- BF⁺ F1s in aug-cc-pCVTZ: the first solution (694.32 eV) with a complex spin-conserving block, the
  second (694.20 eV) along its CVS Hessian, real in that block and unstable in the spin-flip one;
  their NOCI printed.

pyscf does not ship aug-cc-pCVTZ (it loads it from basis-set-exchange only where that is
installed), so the file assembles it as cc-pCVTZ plus the diffuse shells of aug-cc-pVTZ (118 basis
functions for N2 and BF, checked by building the molecules); each system runs in its own guard, so
that a refusal or an error is that system's failure and the others still run.

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
