# The FCI vertex split

How much of the error of G0W0@HF for an ionization energy comes from the screened interaction W,
and how much from the self-energy Σ = iGWΓ itself? On a small frozen-core active space both
questions have exact answers. `fci_vertex_split` computes them:
- the exact ionization energy from FCI;
- G0W@HF with four screenings that share G0 and differ only in W_c:

| rung | W_c from the density response of |
|---|---|
| `rpa` | direct RPA (Hartree kernel, singlet): G0W0@HF |
| `tdhf` | TDHF (Hartree + exchange kernel, singlet) |
| `bse` | static BSE@HF (the TDHF kernel, its exchange-type terms screened by W_RPA(ω = 0)) |
| `fci` | the active space's FCI ground state: the exact W_c on this Hamiltonian |

The split of the G0W0@HF error is then:
- the W side, IP[G0W_FCI] − IP[G0W0];
- the Σ side, IP_exact − IP[G0W_FCI].

The two add up to minus the G0W0@HF error. The TDHF and static-BSE rungs measure how close those
kernels come to the exact W.

It also computes the density side on the HF reference: the oracle Hartree (and Hartree + exchange)
change of the exact density, and the screened tadpole T_c. In a large-N_f counting (v → (2/N_f)v,
N_f = 2 the molecule), the quasiparticle energy through O(1/N_f) is Σ_x + Σ_c(ε) + T on the
Hartree reference, where T is the static Hartree response to the linearized-GW density of
G(Σ_x + Σ_c)G, screened by the static RPA. On HF, G0W0 holds every term of that order except
T_c, the part sourced by Σ_c, so G0W0@HF + T_c is exact through O(1/N_f). This counting is the
study's own; it was checked against exact N_f-flavour diagonalizations with one occupied orbital
per flavour.

The module ports the study's pilot script `exactw_alpha.py` and reproduces its numbers (the
pilot's provenance is in `tests/fci_vertex_split_references.py`); the pilot's starting-point dial
(Hartree + α Fock) is not ported.

## Running it

```python
import json
from pyscf import gto, scf
from src.SingleReference.GW.fci_vertex_split import fci_vertex_split

mol = gto.M(atom='F 0 0 0; H 0 0 0.917', basis='6-31g')
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

rec = fci_vertex_split(mf, ncore=1)        # the HOMO (1π) and the HOMO-1 level (3σ)
homo = rec['orbitals'][0]
print(homo['exact']['ip_ev'], homo['exact']['z'])         # 15.3192 eV, Z 0.9353
print({r: v['ip_ev'] for r, v in homo['rungs'].items()})
# rpa 15.3897, tdhf 14.6408, bse 14.8193, fci 14.6998
print(homo['splits_ev']['w_side'], homo['splits_ev']['sigma_side'])   # -0.6899, +0.6195
print(homo['density']['tc_ev'], homo['density']['ip_g0w0_tc_ev'])     # -0.2804, 15.6701
print(rec['numerical_flags'])                              # []
with open('hf_631g.json', 'w') as f:
    json.dump(rec, f)
```

## What is computed

**The active space.** Every MO above the ncore lowest is active. h1 and the full four-index
(pq|rs) come from pyscf's CASCI (`get_h1eff`, `get_h2eff`). e = mo_energy[ncore:], and the active
Fock matrix h1 + 2J − K must be diag(e) (flagged otherwise). m is the number of doubly occupied
active orbitals.

**The reference state |Ψ0⟩** is the state pyscf's direct_spin1 reaches from its default guess,
the determinant of lowest diagonal energy. pyscf takes that determinant as it is: an open-shell
one is not spin-adapted.
- For every panel system the guess is the closed-shell HF determinant, by 0.05 Ha or more
  (MgO/6-31G the least). The Davidson keeps that guess's spatial symmetry and its alpha-beta
  symmetry. So |Ψ0⟩ is the lowest singlet of the HF determinant's symmetry, not necessarily the
  ground state of the active space. MgO/STO-3G with ncore = 6 has its ³Π 0.234 eV below |Ψ0⟩.
- For 400 determinants or fewer and a non-degenerate lowest level, pyscf diagonalizes exactly,
  and |Ψ0⟩ is then the lowest M_s = 0 state. With a degenerate lowest level it runs the Davidson
  from the guess.
- A triplet the solver does reach is flagged. O2 as an RHF singlet in STO-3G with ncore = 4 is
  diagonalized exactly, and its lowest M_s = 0 state is the triplet. C2/STO-3G with ncore = 4 has
  an open-shell guess and a degenerate lowest triplet pair, and its Davidson ends on a triplet.
  Both have ⟨S²⟩ = 2.
- The record's `ground_state` fields describe |Ψ0⟩. `'fci_split_ground_not_singlet'` reads
  |Ψ0⟩'s own ⟨S²⟩; a lower state of another spin or symmetry is not looked for.
- The Davidson stops once its residual is below pyscf's default, the square root of
  `FCI_CONV_TOL` (3·10⁻⁷ Ha; 5·10⁻⁸ to 2·10⁻⁷ Ha measured). So |Ψ0⟩ carries a vector error of
  about 10⁻⁷ relative: 2.6·10⁻⁸ for H2O/STO-3G and 2·10⁻⁷ for MgO/STO-3G, against the dense
  state. Every field built on |Ψ0⟩ carries about as much: up to 5·10⁻⁷ eV in the oracle shifts
  and the G0W_FCI root, and 4·10⁻⁸ in Z and the occupations. With 400 determinants or fewer and
  a non-degenerate lowest level, |Ψ0⟩ is exact to rounding.

**Σ^c in pole form**, the same for every rung:

```
Σ^c_pp(ω) = Σ_{q<m}  Σ_k (W^k_pq)² / (ω − e_q + Ω_k)
          + Σ_{q≥m}  Σ_k (W^k_pq)² / (ω − e_q − Ω_k),
W^k_pq    = Σ_rs (pq|rs) ⟨k|E_rs|0⟩,   E_rs = Σ_σ a†_rσ a_sσ
```

- **Casida rungs**: W^k_pq = √2 Σ_ia (pq|ia)(X + Y)_ia,k, with the house normalization
  XᵀX − YᵀY = 1. The matrices are the house's: `build_casida_matrices` with `lBSE=False` (RPA),
  `lBSE=True` (TDHF), and `lBSE=True` with `W_aux` from `static_screening_aux` (static BSE@HF).
  Each is solved by `CasidaSolver`.
- **FCI rung**: for each active q, the start vector O_pq|Ψ0⟩ = Σ_rs (pq|rs) E_rs|Ψ0⟩ (pyscf
  `contract_1e`), projected off |Ψ0⟩. Lanczos of depth D on the projected Hamiltonian Q H Q gives
  Ritz values θ_k, with Ω_k = θ_k − E0, and weights |O_pqΨ0|² U_0k². This is a Gauss quadrature
  of the resolvent with D nodes, exact once D reaches the Krylov dimension. It runs without
  reorthogonalization: a ghost copy of a converged pole shares that pole's weight and leaves the
  quadrature as it is.
- **Each Lanczos vector is kept off |Ψ0⟩.** Q H Q holds |Ψ0⟩ at eigenvalue 0. The core energy is
  left out of H, and where the active-space spectrum lies below 0 (N2/STO-3G, and Ne, HF and H2O
  in 6-31G), that eigenvalue lies above it. The three-term recurrence alone then lets rounding
  seed that direction and amplifies it at every step, until a Ritz value converges at 0 and takes
  Gauss nodes at steps that rounding decides. In cc-pVDZ the spectrum reaches above 0 (He, H2,
  LiH, Ne), so 0 lies inside it. Each new Lanczos vector is projected off |Ψ0⟩ as it is made, as
  the pilot does (`spectral_measure` with `deflate`), which removes the direction in either case.
  Without that, N2/STO-3G's 2σg had 20 Ritz values at 0 in its eight runs at D = 40, and its D/2
  root lay 2.2 meV from the 20-node quadrature.
- **The lowest excitation** (`min_omega_ev`). For the Casida rungs it is the lowest excitation of
  the screening. For the FCI rung it is the lowest Ritz value whose weight exceeds
  `FCI_SPLIT_LOWEST_POLE_WEIGHT` (10⁻¹²) of Σ_pp's total weight, the lowest excitation p's
  self-energy sees. Without reorthogonalization, rounding seeds states the start vectors do not
  couple to (triplets, other symmetries). They converge with relative weights of 6·10⁻²¹ or less:
  Ne/6-31G's HOMO is the largest measured, with 7·10⁻²⁴ at D = 40 and 5.5·10⁻²¹ at D = 60 and 80.
  H2/cc-pVDZ's lowest Ritz value overall is its triplet at 10.68 eV; the record keeps the singlet
  at 13.91 eV.
- **No pole.** A start vector whose norm is below `ACTIVE_SPACE_VERTEX_TOL` carries no pole. If
  every one does, Σ^c_pp of the FCI rung is zero: its root is e_p with Z = 1, every start counts
  in `skipped_starts`, and `min_omega_ev` is `None`. A physical ground state does not reach this
  case.

**The quasiparticle equation** ω = e_p + Σ^c_pp(ω) is solved for every rung by the one root
finder (`Solvers.qp_equation.solve_qp_equation_poles`, the root of largest weight). On HF,
Σ_x − v_xc = 0, so the exchange is already in e_p. IP = −ω, and Z is the root's exact pole
strength. A root with Z at or below `QP_WINDOW_Z_MIN` (0.5) is no quasiparticle. It is kept
(with its splits), its `quasiparticle` is False, and it is flagged. Beside it the record keeps
the linearized solution ω = e_p + Z(e_p) Σ^c_pp(e_p), with Z(e_p) = 1/(1 − Σ^c_pp′(e_p)): one
Newton step from e_p. A small `z_linearized` means that e_p lies next to a pole of Σ^c, and the
linearized IP then stays at −e_p and carries no information. For N2/STO-3G's 2σg, G0W_FCI has
`z_linearized` 1.0·10⁻⁴ and a linearized IP of 39.2831 eV, against −e_p = 39.2564 eV.

**The exact hole.** Lanczos with full reorthogonalization (the house `lanczos_extend`) on
a_p,α|Ψ0⟩ in the (m − 1, m) sector gives the poles IP_s = E_s(N − 1) − E0 and their weights
|⟨s|a_p,α|0⟩|², which sum to D_pp/2. The main pole is the one of largest weight, and its weight is
the exact Z. Without reorthogonalization the main pole splits into ghost copies past about 100
steps, which would halve its weight. Each Ritz value lies within its residual bound
β_D |U_D−1,k| of a cation energy. The main pole's bound is checked against
`FCI_SPLIT_HOLE_RESIDUAL_BAR` (10⁻⁵ Ha). The record also keeps the second-largest weight, its
pole and its bound. How to read a bound:
- below the bar, the pole is certified as a cation state to within its bound;
- above the bar, it is not certified. It may be a quadrature node standing for unresolved
  satellites (HF/6-31G's HOMO: a second pole near 72.4 eV with a bound of 0.20 Ha). It may also be
  a converged state whose bound overstates its error, which is about bound²/gap: N2/STO-3G's 2σg
  satellite has a bound of 1.9·10⁻⁵ Ha and lies 3·10⁻¹⁰ Ha from the dense cation state. So do not
  drop a second pole on its bound alone;
- the bounds depend on rounding and on Ψ0's own error, by a factor of about 2 at the 10⁻⁶ Ha
  level. N2's main-pole bound, 2.4·10⁻⁶ Ha, is 4.3·10⁻⁶ Ha in an equally valid run, a margin of
  2 to 4 below the bar.

**The half-depth check.** The FCI rung is also read at depth D/2, from the leading block of the
same tridiagonal matrix (bitwise what a separate D/2 run gives). A change of its root above
`FCI_SPLIT_HALF_DEPTH_BAR` (10⁻⁵ Ha) is flagged. The change measures the error of the D/2 value,
and the depth-D error is orders of magnitude smaller, so the check is conservative: N2/STO-3G's
2σg changes by 1.1·10⁻³ eV from D = 20 to 40, but its D = 40 root lies 1.5·10⁻¹² eV from
D = 120. The record's `half_depth_change_ev` is IP(D) − IP(D/2), for the FCI rung and for the
hole, which keeps its own D/2 change as a diagnostic only. Before each Lanczos vector was kept
off |Ψ0⟩, the D/2 values depended on rounding (N2's change was +1.1·10⁻³ eV, against −1.1·10⁻³
eV now).

**The density side.** Each term is a static shift s of the G0W0 quasiparticle energy, ω → ω + s,
so its IP is IP[G0W0] − s:

| key | shift |
|---|---|
| `dh_oracle_ev` | ⟨p\|J[D_FCI − D_HF]\|p⟩, the exact density's Hartree change |
| `dhx_oracle_ev` | ⟨p\|(J − K/2)[D_FCI − D_HF]\|p⟩, with its exchange |
| `tc_ev` | T_c = ⟨p\|W_RPA(0)[δD_lin]\|p⟩, the screened tadpole |
| `tc_bare_ev` | ⟨p\|J[δD_lin]\|p⟩, its unscreened part |
| `dhx_lin_scr_ev` | ⟨p\|(J − K/2)[δD_lin]\|p⟩ plus its TDHF (A + B) static response, the pilot's variant |

- δD_lin is the spin-summed linearized-GW density of Σ_c alone on the RPA rung's poles. Its oo and
  vv blocks are the usual ones, and its ov block is [Σ^virt_ia(e_i) + Σ^occ_ia(e_a)]/(e_i − e_a).
  It equals the house's unrelaxed GW density, and its trace is zero.
- W_RPA(0) = v − 4v(A + B)⁻¹v (`static_screened_coulomb_chemist`). So T_c = v δn⁽¹⁾ with
  δn⁽¹⁾ = (1 − P0(0)v)⁻¹ δn_bare, the correlation part of the O(1/N_f) tadpole. G0W0@HF + T_c,
  i.e. IP[G0W0] − T_c (`ip_g0w0_tc_ev`), is exact through O(1/N_f).

**Default orbitals.** The defaults are the HOMO and the next lower occupied level outside the
HOMO's degenerate set: HOMO−1 by level, not by index. A level is a block of neighbours closer than
`QP_DEGENERACY_TOL` (10⁻⁴ Ha), the house's one separation (`degenerate_blocks`), and it is
represented by its highest orbital. For HF/6-31G this gives the 1π HOMO and the 3σ, not the other
π.

## Options

| argument | meaning | default |
|---|---|---|
| `mf` | a converged closed-shell RHF on exact integrals, in the gas phase | (required) |
| `ncore` | the number of frozen lowest MOs, 0 ≤ ncore < nocc | (required) |
| `orbitals` | occupied active orbitals in full-MO numbering, ncore ≤ p < nocc | the HOMO and the HOMO−1 level |
| `depth` | the FCI rung's Lanczos depth D, an even integer ≥ 2 | `FCI_SPLIT_SCREENING_DEPTH` = 40 |

`depth` is a numerical option on purpose, an exception to the house rule that keeps numerical
choices out of signatures (as BSE(2)'s `dynamical_roots`). The cluster panel runs the split at
more than one depth: the default, and twice the pilot's 16, so that the D/2 values compare with
the pilot at its own depth.

The hole runs `FCI_SPLIT_HOLE_DEPTH` = 60 steps. The FCI ground state uses pyscf's direct_spin1
Davidson to `FCI_CONV_TOL` (10⁻¹³ Ha) within `FCI_SPLIT_MAX_CYCLE` (300) cycles. It holds as much
memory as this job may give it (`allocation_max_memory_mb`: its share of a SLURM allocation, else
`FCI_SPLIT_MAX_MEMORY_MB` = 8000 MB). The record's `max_memory_mb` keeps that allowance.

Refused before any work (ValueError, the message starting with the argument):
- a UHF, ROHF, Kohn-Sham, density-fitted, solvated or unconverged mean field;
- one carrying an environment (`mf.with_screening` other than `NoEnvironment`);
- an open shell, a non-aufbau occupation, or a basis without virtual orbitals;
- ncore, orbitals or depth outside the ranges above;
- an ncore that splits a degenerate occupied level (`check_core`: a block of
  `degenerate_blocks` at `QP_DEGENERACY_TOL` with members on both sides of the cut), since
  freezing part of a level breaks the molecule's symmetry. Ne/6-31G with ncore = 3 is refused
  with "ncore = 3: it splits the degenerate occupied level [2, 3, 4] ...";
- an active space whose estimated peak memory exceeds that allowance (the message starts with
  `ncore`).

The peak is the larger of two terms, each 8 bytes per number:
- the Davidson's `FCI_SPLIT_DAVIDSON_VECTORS` (32) vectors in the N sector;
- the hole's basis, `FCI_SPLIT_HOLE_COPIES` (3) × 60 vectors in the N − 1 sector.

The house Lanczos stacks its stored vectors at every step, so three copies are alive at once.
LiF or BeO in 6-31G with ncore = 1 (38M determinants) would need about 21 GB and is refused on a
16 GB job.

## Results and flags

The record is JSON-serializable. Energies are in eV, except where a key ends in `_ha`. NaN (an
unstable rung, a missing root) is written by `json.dumps` as the token `NaN`, which Python's json
reads back; a value that does not apply (no second pole) is `None`.

| key | content |
|---|---|
| `method` | `'fci_vertex_split'` |
| `parameters` | `ncore`, `orbitals`, `screening_depth`, `hole_depth`, `max_memory_mb` |
| `sizes` | `nmo`, `nocc`, `n_act`, `nelec_act`, `n_ov`, `dets_n`, `dets_n_minus_1`, `memory_estimate_mb` |
| `reference` | `e_hf_ha` (the RHF energy) and `fock_miss_ha`, the active Fock matrix's largest miss of diag(e) |
| `ground_state` | \|Ψ0⟩'s `e_fci_ha` (with the core energy), `e_corr_ha`, `converged`, `spin_square` |
| `orbitals` | one record per orbital (below) |
| `timings_s` | wall seconds: `active_space`, `ground_state`, `casida_rungs`, `density_side`, `hole`, `fci_rung`, `total` |
| `numerical_flags` | a list of `{'flag', 'message'}`: the flags below and the root finder's `'no_root'`, `'z_outside'` |

Per orbital:
- `orbital` (full-MO), `active_orbital`, `level` (its degenerate set), `eps_hf_ev`, and
  `occupation_fci` (the spin-summed FCI occupation D_pp);
- `exact`: `ip_ev`, `z`, `residual_ha` (the main pole's Ritz residual bound), `second_ip_ev`,
  `second_z`, `second_residual_ha`, `window_weight` (the weight of the other poles within
  `FCI_SPLIT_SECOND_POLE_WINDOW`), `weight_sum` (= D_pp/2), `ip_half_depth_ev`,
  `half_depth_change_ev` (IP(D) − IP(D/2)), `depth`, `steps`, `breakdown`;
- `rungs`: for each of `rpa`, `tdhf`, `bse` and `fci`, the keys `ip_ev`, `z`, `quasiparticle`,
  `ip_linearized_ev`, `z_linearized`, `min_omega_ev` and `stable`. The `fci` rung also has
  `ip_half_depth_ev`, `half_depth_change_ev` (IP(D) − IP(D/2)), `depth`, `steps` and
  `skipped_starts` (start vectors below `ACTIVE_SPACE_VERTEX_TOL`); its `min_omega_ev` is `None`
  when it has no pole;
- `splits_ev`: `w_side`, `sigma_side`, `tdhf_minus_fci`, `bse_minus_fci`, `g0w0_minus_exact`;
- `density`: the shifts of the table above, `trace_dd_lin`, and the IPs `ip_g0w0_dh_oracle_ev`,
  `ip_g0w0_dhx_oracle_ev`, `ip_g0w0_tc_ev` and `ip_g0w0_dhx_lin_scr_ev`.

The flags, each warned as `FCISplitShort`:
- `'fci_split_fock_not_diagonal'`: the active Fock matrix misses diag(e) by more than
  `FCI_SPLIT_FOCK_BAR` (10⁻⁶ Ha);
- `'fci_split_ground_not_converged'` and `'fci_split_ground_not_singlet'` (|Ψ0⟩'s own ⟨S²⟩
  above `FCI_SPLIT_SPIN_SQUARE_BAR`; a lower triplet is not looked for);
- `'fci_split_depth_unconverged'`: the FCI rung's half-depth check, or the hole's main pole with a
  residual bound above `FCI_SPLIT_HOLE_RESIDUAL_BAR`;
- `'fci_split_main_pole_ambiguous'`: the main pole's weight is `QP_WINDOW_Z_MIN` (0.5) or less, or
  the other poles within 0.1 Ha (`FCI_SPLIT_SECOND_POLE_WINDOW`) together carry half its weight
  or more (`FCI_SPLIT_SECOND_POLE_RATIO`), so a quadrature node standing for several satellites
  counts with their total weight;
- `'fci_split_rung_no_quasiparticle'`: a rung's root has Z at or below `QP_WINDOW_Z_MIN`;
- `'fci_split_rung_unstable'`: a Casida rung is unstable (`CasidaInstability`). It is not repaired:
  its IP and Z are NaN, and so are the splits that use it. The same flag marks a TDHF A + B that
  is not positive definite, which leaves `dhx_lin_scr_ev` NaN.

See [Warnings, flags and refusals](../warnings.md).

## Checked in `tests/test_fci_vertex_split.py`

- **The FCI rung is the exact W_c.** At full Lanczos depth its Σ^c_pp(ω) equals the dense Lehmann
  sum over every FCI state (He/cc-pVDZ, H2/6-31G**) at five frequencies, to 4·10⁻¹⁶ relative. Its
  root equals `tests/fci_screened_gw.py`'s at η = 0 to 10⁻¹⁴ eV, and its linearized solution the
  dense poles' to 3·10⁻¹⁶ relative. These spaces have 400 determinants or fewer and a
  non-degenerate lowest level, where pyscf diagonalizes exactly and |Ψ0⟩ is exact to rounding;
  beyond them the comparison holds only to |Ψ0⟩'s own error of about 10⁻⁷ relative (see the
  reference state above).
- **The lowest FCI excitation is a coupled state.** At full depth it is the dense lowest coupled
  state to 2·10⁻¹³ eV. At the default depth it is the coupled singlet although a lower triplet
  is seeded into the Lanczos runs: 13.9137 eV for H2/cc-pVDZ (triplet 10.6757) and 1.5689 eV for
  LiH/cc-pVDZ at 3.0 Å (triplet 0.7921).
- **The RPA rung is the house G0W0@HF.** On H2O/6-31G (full space) it equals QPqb's root and Z to
  6·10⁻¹⁶ relative, and `calc_qp_energy` to 2.4·10⁻⁸ eV (the graphical scan's stop). Through the
  driver on H2O/STO-3G its roots and linearized solutions equal QPqb's poles' to rounding.
- **The TDHF rung.** Its energies are pyscf's TDHF singlets to 4.8·10⁻¹⁴ Ha. Its residues obey
  Σ_k Ω_k (W^k_pq)² = 2v(A − B)v and Σ_k (W^k_pq)²/Ω_k = 2v(A + B)⁻¹v to 4·10⁻¹⁶ (RPA too). Its
  roots equal `calc_qp_energy`'s GW@TDHF to 9·10⁻⁹ eV.
- **The static-BSE rung.** Its energies equal `solve_bse` (dense, full integrals, qp=None) to
  9.8·10⁻¹⁴ Ha and an independent einsum BSE from the four-index W_RPA(0) to 7.6·10⁻¹⁴ Ha. Its
  roots equal `calc_qp_energy`'s GW@BSE to 7.8·10⁻⁸ eV.
- **The exact hole is the dense cation.** The (m − 1, m) sector is diagonalized densely: H2O/STO-3G
  (735 determinants; HOMO, HOMO−1, HOMO−2) and N2/STO-3G's inner-valence 2σg (ncore = 2, 3920
  determinants). The main pole is the state of largest |⟨s|a_p,α|Ψ0⟩|², its IP to 1.1·10⁻¹⁰ eV
  and its weight to 7.4·10⁻¹¹. The weights sum to D_pp/2 to 10⁻¹⁵.
- **N2's 2σg.** The main pole is 39.6451 eV (Z 0.5629). The second, 42.1253 eV (0.2631), lies
  within its residual bound of the dense second state. Its G0W_FCI root (41.9550 eV, Z 0.4244)
  is no quasiparticle and is flagged. The half-depth check flags it as well (4.2·10⁻⁵ Ha from
  D = 20 to 40), although the depth-40 root lies 1.5·10⁻¹² eV from the depth-60 one. The D/2
  root is the 20-node Gauss quadrature, a Lanczos with full reorthogonalization kept off |Ψ0⟩
  written out in the test, to 10⁻⁹ eV (1.1·10⁻¹³ measured). A recurrence that lets rounding seed
  |Ψ0⟩ misses it by 2.2·10⁻³ eV.
- **The pilot's numbers** for HF/6-31G and LiH/cc-pVDZ (ncore = 1), recorded at full precision in
  Hartree:
  - IP_exact, G0W0, G0W_TDHF, G0W_FCI and its Z;
  - the lowest FCI-rung excitation and both sides of the split;
  - dHx_lin_scr and the oracle IPs.

  Each class has its own bar:
  - the quantities of the HF reference alone to 10⁻⁹ eV (1.8·10⁻¹⁴ eV measured);
  - those of the FCI ground state to 2·10⁻⁶ eV (5.2·10⁻⁷ measured). Both Davidsons stop on
    pyscf's residual tolerance, the square root of the energy tolerance: 3.2·10⁻⁶ Ha in the
    pilot (conv_tol 10⁻¹¹) and 3.2·10⁻⁷ Ha here (`FCI_CONV_TOL` = 10⁻¹³). For HF and H2O in
    6-31G the pilot's settings leave vector errors of 2.0·10⁻⁷ and 6.8·10⁻⁷, the module's
    2.6·10⁻⁸ and 6.9·10⁻⁸. So most of the miss is the pilot's |Ψ0⟩, and part of it is the
    module's own;
  - the oracle IPs to 10⁻⁵ eV (2.4·10⁻⁶ measured: the pilot's reference is its own SCF in the
    active space);
  - Z to 2·10⁻⁷.
- **The density side against independent code.** δD_lin is the house's unrelaxed GW density
  (`GWDensityMatrixSolver`, RPA) to rounding, and T_c is the pilot's `screened_hartree_shift` to
  rounding. T_c is −0.2804 eV for HF (bare −0.5333) and +0.0164 eV for LiH (bare +0.0474).
- **Refusals and flags.**
  - Every refusal is raised before the active space is built, an ncore that splits HF/6-31G's
    π pair included, and a gas phase marked by `NoEnvironment` is accepted.
  - With `ACTIVE_SPACE_VERTEX_TOL` raised to 10¹⁰, the FCI rung has no pole: its root is e_p with
    Z = 1, every start counts as skipped, `min_omega_ev` is `None`, and nothing is flagged.
  - Each flag fires when its event occurs: depth 4, a hole depth of 4 and a zero residual bar,
    one Davidson cycle, the ambiguity bars, `QP_WINDOW_Z_MIN` at 0.99, and a Fock and a spin bar
    set below the measured misses.
  - Three real instabilities each give a NaN rung and the flag, with the other rungs finite:
    MgO/STO-3G (TDHF A − B indefinite, as for MgO/6-31G), N2/STO-3G at 1.6 Å (TDHF, BSE and the
    TDHF A + B) and H2/6-31G at 2.5 Å (static BSE).
- **Nothing existing changes.** The mean field's arrays are bitwise their stored copies after a
  run, the defaults are as described above, and the record survives a JSON round trip.

25 s on two threads.

## Checked in `tests/test_fci_vertex_split_slow.py`

The pilot's H2O/6-31G, Ne/cc-pVDZ and NH3/6-31G (ncore = 1, HOMO) are checked to the same bars on
the same quantities. The worst misses are NH3's: 5.2·10⁻⁷ eV for G0W_FCI and 2.4·10⁻⁶ eV for the
oracle IP[G0W0] − dH.

Wall times on two threads: H2O 16 s, Ne 43 s, NH3 108 s; the file takes about 170 s.

## Measured besides

HOMO, HF reference, ncore = 1, eV:

| system | dets N | IP_exact (Z) | G0W0 | G0W_TDHF | G0W_BSE | G0W_FCI (Z) | W side | Σ side | T_c |
|---|---|---|---|---|---|---|---|---|---|
| LiH/cc-pVDZ | 324 | 7.8521 (0.890) | 7.9661 | 7.3125 | 7.4756 | 7.1508 (0.754) | −0.8153 | +0.7012 | +0.0164 |
| HF/6-31G | 44,100 | 15.3192 (0.935) | 15.3897 | 14.6408 | 14.8193 | 14.6998 (0.932) | −0.6899 | +0.6195 | −0.2804 |
| H2O/6-31G | 245,025 | 11.8872 (0.923) | 12.0580 | 11.3618 | 11.5341 | 11.4101 (0.921) | −0.6480 | +0.4771 | −0.1966 |
| Ne/cc-pVDZ | 511,225 | 20.8950 (0.952) | 20.8704 | 20.1439 | 20.3165 | 20.2064 (0.947) | −0.6640 | +0.6886 | −0.2528 |
| NH3/6-31G | 1,002,001 | 9.8347 (0.920) | 10.0834 | 9.5016 | 9.6449 | 9.5376 (0.917) | −0.5457 | +0.2970 | −0.1017 |

- The 3σ (HOMO−1) of HF/6-31G: IP_exact 19.2558 eV (Z 0.946), G0W0 19.1839, G0W_FCI 18.7133.
- MgO/STO-3G (ncore = 6), whose TDHF rung is unstable: IP_exact 4.147 eV (Z 0.773), G0W0 3.736,
  G0W_BSE 3.429, G0W_FCI 3.393.
- The HF HOMO need not be the first ionization. In MgO/6-31G the exact π (HOMO−1 level) lies at
  7.842 eV (Z 0.519), below the σ HOMO at 8.692 eV (Z 0.619), as it does in G0W0 and G0W_FCI.
  Compare results orbital by orbital, not by Koopmans order.
- The FCI rung at D = 15, 30 and 40 agrees to below 10⁻⁶ eV on the systems of the table. D = 10
  against 5 moves it by 8·10⁻⁵ eV (Ne) to 1.8·10⁻² eV (LiH), and D = 40 against 20 by
  3.8·10⁻⁹ eV or less (H2O).
- At the panel's largest spaces, measured before each Lanczos vector was kept off |Ψ0⟩, the error
  of the root at D = 30 / 40 was 4.7·10⁻⁷ / 4·10⁻⁹ eV (MgO/6-31G σ), 2.3·10⁻⁷ / 5·10⁻¹⁰ (BeO π)
  and 2.3·10⁻⁸ / 10⁻¹⁰ (LiF π). At D = 30 the half-depth check then flagged BeO and MgO, and at
  the default 40 neither. N2/STO-3G's 2σg, which then had errors of 2.7·10⁻⁷ / 6·10⁻¹⁰ eV, now
  has 4.4·10⁻⁹ / 1.5·10⁻¹² eV. These spaces exceed the local size limit, so the panel run records
  their half-depth changes anew.
- Time on one pyscf thread (the macOS pyscf wheel has no OpenMP), at the default depths. HOMO and
  HOMO−1: HF/6-31G 3.7 s. HOMO only: H2O 16 s, Ne 43 s, NH3 108 s.
- The FCI rung takes about 90 % of each: n_act Lanczos runs of D steps per orbital, each step one
  pyscf `contract_2e` (with precomputed link tables, 2.2× faster than without at 1M
  determinants). The hole takes 60 steps in the smaller (N − 1) sector.
- Memory at the panel's largest space (3.3M determinants in N, 1.02M in N − 1):
  - the hole's basis peaks at 1.47 GB, three copies of 60 vectors;
  - the FCI Davidson holds 27 to 31 N-sector vectors, 0.7 to 0.8 GB;
  - the FCI rung holds about seven.

  On macOS the resident size overstates this several-fold, since freed copies stay mapped. Read
  the peak footprint from `/usr/bin/time -l` instead.

## What it leaves out

- **Exact G.** GW with the exact G and the exact W (block Lanczos), which would isolate the pure
  vertex in Hedin's sense, is not computed. Every rung here is G0W on the HF G0.
- **Electron affinities.** The (m + 1, m) sector and the LUMO are not computed. In these small
  bases most anions are unbound anyway.
- **Non-HF references.** There is no evGW, qsGW or Hartree + α Fock reference. The rungs read
  Σ_x − v_xc = 0, so a Kohn-Sham mean field is refused.
- **HOMO−1 identification.** HOMO−1 is identified by the HF level (Koopmans order), not from the
  exact spectral functions. The record holds each orbital's two largest exact poles, and an
  ambiguous main pole is flagged.
- **Other roots of a rung.** Each rung records the root of largest weight and its Z, as asked of
  it. A rung without a quasiparticle is flagged, but its other roots are not recorded; the one
  root finder could give them (`QPRoot.found`, `PoleSum.solve` on each interval).
- **Branch tracking.** Each rung picks its root of largest weight on its own, so for an
  inner-valence hole the rungs can land on different branches. Only Z ≤ 0.5 is flagged, and a jump
  to a root with Z > 0.5 passes into the splits unmarked. In N2/STO-3G's 2σg (−e_p = 39.2564 eV,
  exact 39.6451 eV, Z 0.563), G0W0 gives 37.1791 eV (Z 0.839), G0W_TDHF 42.1432 (0.576),
  G0W_BSE 41.7166 (0.651) and G0W_FCI 41.9550 (0.424): TDHF and BSE sit 5.0 and 4.5 eV above
  G0W0 with no flag, so `w_side` (+4.776 eV) and `sigma_side` (−2.310 eV) compare roots on
  different branches. Read the splits of roots that lie far apart with care. Following the G0W0
  root through the rungs (`find_qp_root` with `follow`) is not done. No panel orbital run locally
  jumped branch (Ar/cc-pVDZ 3s: 31.14, 29.44, 29.80 and 29.62 eV on the four rungs, exact 30.16).
- **Lower states of another spin or symmetry.** From the HF determinant, pyscf's default guess
  for every panel system, |Ψ0⟩ is the lowest singlet of that determinant's symmetry. A lower
  triplet (MgO/STO-3G's ³Π, 0.234 eV below |Ψ0⟩ with ncore = 6) or a lower state of another
  symmetry is not looked for: pyscf's Davidson keeps the symmetry of its guess. A check would
  need a start in every symmetry. In the M_s = 1 sector from pyscf's default guess, MgO/STO-3G
  lands on a ³Σ 1.90 eV above |Ψ0⟩ and misses the ³Π. A random start finds the ³Π there, but
  took 10 times as many contractions on H2O/6-31G and did not converge in 300 cycles on
  Ne/cc-pVDZ. A three-root M_s = 0 Davidson put MgO/6-31G's lowest triplets, a degenerate pair,
  0.34 eV above |Ψ0⟩.
- **Direct_spin0.** pyscf's direct_spin0 would roughly halve the FCI rung's cost on these singlet
  vectors, but it is not a drop-in replacement. Its contraction assumes an exactly
  alpha-beta-symmetric vector, which c0 and the Lanczos iterates are only to rounding. Without
  symmetrization, a review run put a spurious Ritz value 3.7 eV below E0 into BeO/6-31G. Adopting
  it would mean symmetrizing c0 and every vector before and after each contraction. The module
  keeps direct_spin1 throughout.
