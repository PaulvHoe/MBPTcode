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
| `soft_mode_triplets(mf, model)` | model | `'1d'` (default) or `'3d'` |
| `soft_mode_qp(mf, selfenergy, states, model)` | selfenergy | `'PSD2'` (default), `'PSD4'`, `'PSD7'`, `'PSD9'` |
| | states | orbital indices; the HOMO by default |
| | model | `'1d'`, `'3d'`, or `None` for the plain TDHF triplets (the house route's spectrum) |

The reference must be a converged closed-shell RHF with exact integrals; anything else is refused
before any work. An A − B that is not positive definite (90°-twisted ethylene) leaves the mass
undefined and is refused with `CasidaInstability`.

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
- the root finder's own flags.

See [Warnings, flags and refusals](../warnings.md).

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

These were made on purpose and not checked against alternatives:
- the straight path along d, with no relaxation of the other rotations (it reaches 80% of the UHF
  well depth for twisted ethylene, 62% for O3);
- the mass held at its Q = 0 value;
- D(Q) as the transition density that couples back;
- the flat 3D ball, which runs 0.6 to 1.1 eV above the 1D model for the soft modes, in place of
  the curved manifold of rotated determinants;
- independent soft sets (O3's second mode stops being unstable once the first condenses);
- the closure of a path whose minor pairs have not come back.

There is no large parameter here as there is in SYK, so the collective model is not a controlled
expansion. Its levels sit 0.5 eV above EOM-CCSD fairly uniformly along ethylene's twist.
