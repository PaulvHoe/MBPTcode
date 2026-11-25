# Second-order BSE kernels: BSE(2)

The second-order Bethe-Salpeter kernels of van Hoegaerden and Förster: the bare
first-order BSE plus a second-order kernel A(2), B(2), and a dynamical
correction dBSE(2). One kernel serves three methods; they differ in the
quasiparticle energies on the diagonal and in what sits in the kernel's six
exchange-type W slots.

| method | energies `eps` | `kernel` (W slots) |
|---|---|---|
| BSE(2)@HF | `'Sigma_TDHF@L_TDHF'`: Σ_TDHF@L_TDHF (HF sequence, cycle 1) | `'v'` |
| BSE(1)@GF2 | `'G0F2'` | `'v'` |
| BSE(2)@GW | `'Sigma_BSE@L_BSE'`: Σ_BSE@L_BSE (GW sequence, cycle 2) | `'2sosex'`: 2W0 − v, the paper's Eq. 28 |
| BSE(2)_Wv@GW | as BSE(2)@GW | `'v'` |

W0 is the static RPA screened interaction at the same energies; under the
Tamm-Dancoff approximation it is the TDA one, as in the paper's runs. The
orbitals are those of the RHF reference.

```python
from src.SingleReference.BSE.second_order_bse import second_order_bse

res = second_order_bse(mf, eps='Sigma_BSE@L_BSE', kernel='2sosex', spin='singlet')
res['omega']                  # static roots (Ha)
res['dynamical']['omega']     # dBSE(2) roots of the lowest 30
res['dynamical']['zeta']      # their renormalization factors
```

## The quasiparticle sequences

Naming a sequence in `eps` runs the paper's quasiparticle cycles first, as
its AMS runs did them (`src/SingleReference/BSE/second_order_qp.py`), and puts
the last cycle's energies on the diagonal:

| `eps=` | cycles, each at the energies the one before left | feeds |
|---|---|---|
| `'Sigma_TDHF@L_TDHF'` | TDHF at the HF energies, GWGammaInf with the bare vertex | BSE(2)@HF |
| `'Sigma_BSE@L_BSE'` | G0W0@HF; then W0 and BSE(1) (W0 in the direct term) at those energies, GWGammaInf with W0 in the vertex | BSE(2)@GW, BSE(2)_Wv@GW |
| `'G0F2'` | Σ(2) at the HF energies, unbroadened | BSE(1)@GF2 |
| `'G0W0'` | G0W0@HF | |

```python
res = second_order_bse(mf, eps='Sigma_BSE@L_BSE', kernel='2sosex', tda=True)
res['qp']['cycles']          # per cycle: 'name', 'eps', 'Z', 'mu'
res['numerical_flags']       # the named flags of the cycles and of dBSE(2)
```

`qp_sequence(mf, name, tda=False, eta=None)` returns the same record on its
own. Every cycle keeps the RHF orbitals, puts G and the screening at the
energies of every orbital the cycle before left, and solves
w = ε_HF,p + Re Σc,pp(w) for every orbital, the equation anchored on the HF
energies in every cycle. Σc carries the broadening η = 10⁻³ Ha (G0F2 none, as
in AMS). The cycles screen with singlet responses; `tda=True` puts every
response of the sequence in the TDA, the RPA behind W0 included, as the
paper's TDA runs did.

The root is AMS's (moGWmodule `selfEnergy`): one scan of 100 points over
ε_HF,p ± h per orbital, with h = 0.5, 1 or 2 Ha set by |ε_HF,p − μ| and μ
halfway between the HOMO and the LUMO of the energies the poles sit at; every
sign change between neighbouring points bisected as AMS bisects it, to a
10⁻⁴ Ha bracket; and among those bracket midpoints the one of largest
Z = 1/(1 − dΣ/dω) > 0 kept, a Z > 1 one included. The scan and the bisection
are those of MBPTcode's root engine (`SampledEquation`), the choice is AMS's.
An orbital without a root of positive weight keeps ε_HF,p. Named flags: an
orbital that keeps ε_HF,p raises `'qp_sequence_no_root'`, a kept Z ≤ 0.5 (no
quasiparticle) `'qp_sequence_low_weight'`, a kept Z > 1
`'qp_sequence_weight_above_one'`. Two things differ from AMS on purpose:

- the root kept is converged inside the bracket AMS's bisection left it in,
  where AMS takes that bracket's midpoint, on a lattice of 2.15 meV about
  ε_HF,p: at most 1.07 meV per orbital on the same self-energy;
- in the unbroadened G0F2 a sign change across a pole of Σ(2) is dropped as a
  pole, where AMS keeps it as a root of vanishing weight; no kept root moves.

Checked in `tests/test_second_order_qp.py`:

- the rule against a line-for-line transcription of AMS's `selfEnergy` loop
  on model self-energies: the same bracket midpoints and weights, and the same
  root kept, also where two weights tie to 10⁻⁵ and the converged weights
  would choose the other root;
- G0W0@HF and Σ_TDHF@L_TDHF (full and TDA) and Σ_BSE@L_BSE (full) equal
  MBPTcode's `calc_qp_energy` recipe for every quasiparticle of H2O/6-31G, to
  10⁻⁷ eV. The TDA Σ_BSE@L_BSE does not, on purpose: `calc_qp_energy` keeps
  the full-RPA W0 under the TDA (48 meV on the HOMO). G0F2 equals
  `g0f2_energies` except where a pole of Σ(2) shares the grid cell of that
  routine's root and hides it from AMS's scan (two orbitals of H2O/6-31G);
- H2O/aug-cc-pVTZ at the paper's geometry against the root lists of the
  paper's AMS runs: HOMO−2…LUMO keep AMS's root in every cycle of the HF, GW
  and GF2 sequences, full and TDA, within about 3 meV in the shift
  ε_QP − ε_HF (test bar 5 meV) and 0.001 in Z. The roots AMS printed for
  orbitals 3–12, satellites included, are as many as here. The exception is
  the chained TDA Σ_BSE@L_BSE: within about 10 meV (bar 12 meV), and one
  printed satellite of its HOMO−1 and of its HOMO not found. Fed AMS's own
  first-cycle shifts, that cycle's HOMO−2…HOMO land within 3 meV, so the rest
  comes from the higher levels of its first cycle.

Measured besides (`BSE2_port/mbptcode_port_2026-11/QP_CYCLES_2026-11-10.md`):
the de-excitation (Y) part of the vertex, the one piece of the AMS vertex
outside the pasted Fortran, is AMS's: without it the full cycles' HOMO−2…HOMO
miss AMS by 86–202 meV. The higher virtuals differ: their HF energies already
differ between the two codes by up to 0.36 eV (BAND's fit and grid, and its
treatment of the near dependencies of the diffuse functions).

## Options

Options: `tda=True` drops B and screens in the TDA; `qp_tda` sets the TDA of
the sequence that `eps` names apart from it (`None`, the default, follows
`tda`, as the paper's runs did), so that `tda=True, qp_tda=False` runs the
quasiparticle cycles with full responses and keeps the TDA in the final
BSE(2) only, its W0 included (a `W_aux` passed in replaces that W0);
`channels=('ph',)` drops
the pp and hh ladders (the paper's "pp off"); `first_order='screened'` puts W0
in the direct term of A(1) and B(1), the double-counting "exch" variant of the
SI; `rings=False` drops the kernel's two ring terms, +2(ab|kc)(kc|ij) over
both ph resolvents (`RING_TERMS`, the fourth ph term of the paper's second-cycle
kernel), and with them their relabellings in B(2) and their part of the dBSE(2)
step. `first_order='screened', rings=False` keeps W0 in the first order and
removes the double counting the paper's appendix finds: the second-order term
of -(ij|W0|ab) is 4 Σ_kc (ab|kc)(kc|ij) / (ε_c - ε_k), the same integrals as
that ring term. A root whose renormalization factor is not positive raises the named flag
`'second_order_bse_zeta_nonpositive'`.

The kernel itself, `SecondOrderKernel` in
`src/SingleReference/LinearResponse/second_order_kernel.py`, takes any orbital
energies and returns A(2)(ω), dA(2)/dω and B(2), or the quadratic forms
xᵀA(2)(ω)y that dBSE(2) needs. It is evaluated in tiles from the three-index
factors, so no (vv|vv) tensor is formed. The paper's BSE(1)@GF2 is
`second_order_bse(mf, eps='G0F2', kernel='v')`. A BSE@GF2 calculation of
another kind (Monino and Loos, QuAcK) can take its G0F2 energies from
`g0f2_energies`, in `src/SingleReference/ADC/second_order.py`, which keeps
for each orbital the root of largest weight over the whole axis. That is
AMS's root wherever AMS's scan brackets it:

```python
from src.SingleReference.ADC.second_order import g0f2_energies

eps_gf2, Z = g0f2_energies(mf.mo_energy, nocc, B_aa=coeff_df)
res = second_order_bse(mf, eps=eps_gf2, kernel='v')
```

The kernel reproduces the paper's Fortran (the verbatim routines compiled in
the port's harness) on H2O/6-31G to the six printed decimals, and the v-only
kernel reproduces QuAcK's BSE@GF2 routines with the effective static
reference on H2O/cc-pVDZ (`tests/test_second_order_bse.py`).

It is not Max Graml's `static_second_order_kernel_df`: that is the
Monino–Loos Θ^GW with W on both lines and Zhang static denominators. That
routine carries only the six ring terms of each block, and none of the
exchange-type W-slot terms.
