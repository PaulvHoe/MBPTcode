# Hedin's equations, cycle by cycle: four vertex schemes

`src/SingleReference/HedinCycles` iterates Hedin's equations for a few cycles with vertex
corrections, in four schemes and from two starts.

**Notation.** A *line* is a one-particle Green's function G. Σ is a self-energy, and Σ_H[G], Σ_x[G]
and Σ_HF[G] = Σ_H[G] + Σ_x[G] are the Hartree, exchange and Hartree–Fock self-energies of the line G.
Σ_xc is the exchange-correlation part of a self-energy, everything but its Hartree term. v is the
bare Coulomb interaction, P a polarizability, W a screened interaction and Γ a vertex. L is the
two-particle correlation function and L_0 = −iGG its non-interacting part. A BSE is a Bethe–Salpeter
equation, and the RPA (random-phase approximation) and TDHF (time-dependent Hartree–Fock) responses
are the BSE's solutions with the Hartree kernel and with the Hartree and exchange kernels.

Every cycle n builds the self-energy Σ^(n) from the line G^(n−1) of the previous cycle, and every
Dyson equation uses the same start line G^(0):

    G^(n) = G^(0) + G^(0) [Σ^(n) − Σ^(0)] G^(n)

Here Σ^(0) is the self-energy of the start line. The schemes come from two choices. The vertex needs
the derivative of Σ^(n−1) with respect to G^(n−1), and it is read either **along the map** that the
cycles compute (through G^(0), which every Σ^(n−1) depends on explicitly) or **at fixed form** (the
previous cycle's formula, differentiated with respect to its own line). The self-energy is closed
either on Hedin's **Γ/W route** or on the **I/L route** of the talk and the group note, which closes
Σ with L and its kernel Ξ:

| scheme | name in the code | derivative | route | solved in each cycle |
|---|---|---|---|---|
| SG-I/L | `'sg_il'` | along the map | I/L | nothing beyond the start's BSE |
| SG-Γ/W | `'sg_gw'` | along the map | Γ/W | a Dyson equation for W and a two-point inversion |
| cycle-based Γ/W | `'cb_gw'` | at fixed form | Γ/W | a BSE for the vertex, and from cycle 3 on a BSE inside every kernel application |
| cycle-based I/L | `'cb_il'` | at fixed form | I/L | a BSE for L, and from cycle 3 on a BSE inside every kernel application |

SG stands for Schindlmayr and Godby (PRL 80, 1702 (1998)). Their construction, extended as described
under "Along the map" below, solves the BSE of the schemes along the map explicitly. There are two
starts:

| start | line G^(0) | Σ^(0) | the start's BSE gives | cycle 1, the same in every scheme |
|---|---|---|---|---|
| `'H'` | h, self-consistent Hartree | Σ_H[h] | L^(1) the RPA, Γ^(1) = 1 | G0W0 on h: GW with the RPA screening of h, evaluated once on h |
| `'HF'` | f, self-consistent Hartree–Fock | Σ_HF[f] | L^(1) the TDHF response, Γ^(1) the exchange ladder | GW with the TDHF vertex and TDHF screening on f |

Status: the cycle loop and its zero-dimensional (0-D) backend, which gives the diagram counts and the
0-D values of every scheme. The backend for molecules and model Hamiltonians on the Matsubara axis is
the next stage and is not in the code yet.

## Running it

In zero dimensions (section "The zero-dimensional backend" below) the start line is G^(0) = 1, the
coupling is y = v (G^(0))², a closed fermion loop has the weight ℓ (the loop weight), and a cycle is
measured by κ^(n) = G^(0) (Σ^(n) − Σ^(0)), its self-energy relative to the start line. At ℓ = 1 the
coefficient of y^k of κ^(n) is the number of the cycle's self-energy diagrams at order v^k that the
start line does not already hold. The HF line f holds the first-order (Fock) diagram, so the counts
from `'HF'` begin with 0 at v^1:

```python
from src.SingleReference.HedinCycles import zero_d

zero_d.counts('sg_gw', 'H', cycles=3, order=5)
# [[1, 1, 1, 1, 1], [1, 4, 14, 49, 177], [1, 4, 26, 169, 1032]]
zero_d.counts('cb_il', 'HF', cycles=2, order=5, loop_weight=1)
# [[0, 2, 4, 8, 16], [0, 2, 14, 72, 408]]
```

The values of an exact 0-D model at a finite coupling, here the Rossi–Werner spin-1/2 fermions
(ℓ = −2) at the coupling y of their Hartree line:

```python
from decimal import Decimal, localcontext
with localcontext() as ctx:
    ctx.prec = 50
    kappas = zero_d.kappa_at('sg_il', 'H', Decimal('0.1339745962155613532362768'), cycles=6, loop_weight=-2)
# G^(n) = G^(0) / (1 - kappas[n - 1])
```

Any backend with the operations of `algebra.Backend` runs the same loop:

```python
from src.SingleReference.HedinCycles import run_cycles
history = run_cycles(backend, 'cb_gw', 'HF', cycles=3)
history.sigmas[n], history.lines[n]        # Σ^(n) and G^(n), n = 0 .. 3
```

## What is computed

### The two routes

Γ/W route:

    Σ^(n) = Σ_H[G^(n−1)] + i G^(n−1) W^(n) Γ^(n)
    P^(n) = −i G^(n−1) G^(n−1) Γ^(n),    W^(n) = v + v P^(n) W^(n)
    Γ^(n) = 1 + (δΣ_xc^(n−1) / δG^(n−1)) G^(n−1) G^(n−1) Γ^(n)

I/L route, with Ξ^(n) its kernel and L_0^(n) = −i G^(n−1) G^(n−1):

    Σ^(n) = Σ_HF[G^(n−1)] + i v G^(n−1) Ξ^(n) L^(n)
    L^(n) = L_0^(n) + L_0^(n) Ξ^(n) L^(n)
    Ξ^(n) = i δΣ^(n−1) / δG^(n−1)

### At fixed form

The cycle-based schemes treat the formula of cycle k as a functional of a generic line G, Hedin's
level functional S_k[G], the exchange-correlation part of the formula. On the Γ/W route

    S_k[G] = i G W_k Γ_k,   Γ_k = 1 + (δS_(k−1)/δG) G G Γ_k,   P_k = −i G G Γ_k,   W_k = v + v P_k W_k

and on the I/L route S_k[G] = Σ_x[G] + i v G Ξ_k L_k, with Ξ_k = i δ(Σ_H + S_(k−1))/δG and
L_k = L_0 + L_0 Ξ_k L_k, L_0 = −iGG at the line G. The start sets S_0 = 0 for `'H'` and S_0[G] = Σ_x[G]
for `'HF'`. Cycle n is Σ^(n) = Σ_H[G^(n−1)] + S_n[G^(n−1)], and its kernel δS_(n−1)/δG is taken at
G = G^(n−1). The level functional builds everything from its own argument, so on the Γ/W route that
kernel holds W[G^(n−1)], never the screened interaction of the start line, and on the I/L route its
L_(n−1) is likewise built at G^(n−1).

### Along the map

Every Σ^(n−1) is an explicit functional of G^(0). Its derivative along the map is the derivative with
respect to G^(0), composed with J^(n−1) = δG^(0)/δG^(n−1). Two identities then give the BSE's solution
explicitly. On the I/L route J^(n−1) L^(n) = L^(1), and on the Γ/W route

    J^(n−1) G^(n−1) Γ^(n) G^(n−1) = G^(0) Γ^(1) G^(0) [1 + Λ_H^(n−1)]^(−1)

with Λ_H^(n−1) defined below. The code differentiates along the map with
`along_map(X, legs) = (δX/δG^(0))·legs`: the derivative of X with respect to the start line,
contracted with the pair of lines at one end of `legs` (its other end, where v attaches, stays open).
The legs are L^(1) for SG-I/L and the start's vertex with a start line on each side,
G^(0) Γ^(1) G^(0), for SG-Γ/W:

    SG-I/L:   Σ^(n) = Σ_HF[G^(n−1)] + i v G^(n−1) Ξ̃^(n) L^(1),    Ξ̃^(n) L^(1) = i (δΣ^(n−1)/δG^(0)) L^(1)
    SG-Γ/W:   Γ^(n) = 1 + Λ^(n−1) [1 + Λ_H^(n−1)]^(−1)
              Λ^(n−1)   = (δΣ_xc^(n−1)/δG^(0)) G^(0) Γ^(1) G^(0),       Σ_xc^(n−1) = Σ^(n−1) − Σ_H[G^(n−2)]
              Λ_H^(n−1) = (δ(Σ_H[G^(n−2)] − Σ_H[G^(0)])/δG^(0)) G^(0) Γ^(1) G^(0)

Schindlmayr and Godby's own result is the Γ/W identity with the Hartree term frozen at that of G^(0),
which is Λ_H = 0 and gives their Eq. (7), Γ^(n) = 1 + (δΣ_xc^(n−1)/δG^(0)) G^(0) G^(0) Γ^(1). The group
note extends it to the Hartree update (the factor [1 + Λ_H]^(−1)) and to the I/L route.

The Hartree term inside Σ^(n−1) was built from G^(n−2), not from G^(n−1). Λ_H carries this one-cycle
lag, which is part of SG-Γ/W. Taken with respect to the Hartree term of G^(n−1) itself instead
(`run_cycles(..., lag=False)`), the cycle is SG-I/L exactly. At cycle 2 the lag term vanishes, both SG
schemes share Γ^(2), and they differ only in the screened interaction: SG-I/L closes with W^(1), the
screened interaction of cycle 1, and SG-Γ/W with W^(2), so that their difference is
i G^(1) (W^(1) − W^(2)) Γ^(2).

### The cycle loop and its backends

`schemes.run_cycles` is written against the operations listed in `algebra.py` (the Hartree and
exchange self-energies, the Dyson equation, the polarizability, the screening, the two closures, the
two BSEs, the SG vertex, `along_map` and `kernel(S, G) = δS[G]/δG`) and against sums and differences
of the objects they return. Each operation carries its own factors of i and signs, so the loop holds
none, and every derivative is taken with respect to a line. `algebra.py` gives each operation's
index form and two requirements a backend must meet: `along_map` differentiates through everything,
earlier outputs of `along_map` included, and `kernel` at a perturbed line returns the kernel there.

### The zero-dimensional backend

In zero dimensions every space-time label collapses to one point (Molinari, PRB 71, 113102 (2005)),
the factors i and −1 are absorbed so that every Feynman diagram has weight +1, and a closed fermion
loop has the weight ℓ. With G^(0) = 1 and v = y, every object of a cycle is (G^(0))^a q(y): a power a
of the start line times a series q in y. Hedin's equations do not change under G^(0) → λG^(0),
v → v/λ² for any number λ, and so the derivative along the map is (a + 2y d/dy) q. For a self-energy
(a = −1) this is θ = 2y d/dy − 1, which multiplies y^k by 2k − 1, the number of lines of a diagram of
order k. The derivative at fixed form evaluates the level functional at G + ε, with ε a second
variable, and reads off the part linear in ε. Adding two objects that carry different powers of
G^(0) is refused, a dimensional check on the loop.

At ℓ = 1 the coefficient of y^k of κ^(n) counts diagrams (section "Running it"), and at a general
integer ℓ it is a polynomial in ℓ whose coefficient of ℓ^j counts the diagrams with j closed loops.
At a finite coupling y_0 the backend expands in Taylor jets about y_0 instead, and each derivative
along the map costs one order of the jet. In two exactly solvable 0-D models, the complex boson
(ℓ = 1) and the Rossi–Werner spin-1/2 fermions (ℓ = −2), this gives κ^(n) and G^(n) = G^(0)/(1 − κ^(n))
for comparison with the exact G.

What the counts show (checked in the test below):
- cycle n is exact through v^n from `'H'` and through v^(n+1) from `'HF'`, relative to the start line,
- beyond that order, coefficient by coefficient, SG-I/L ≤ SG-Γ/W ≤ cycle-based ≤ exact, which the test
  checks at ℓ = 1 for cycles 1 to 5 through v^8 (the group note has it through cycle 10 and v^10, with
  no proof for all orders),
- the two cycle-based schemes have the same 0-D map, so the same counts and 0-D values.

| from `'H'`, ℓ = 1 | v^1 | v^2 | v^3 | v^4 | v^5 |
|---|---|---|---|---|---|
| exact | 1 | 4 | 26 | 238 | 2732 |
| cycle 1, every scheme | 1 | 1 | 1 | 1 | 1 |
| SG-I/L, cycle 2 | 1 | 4 | 11 | 27 | 61 |
| SG-Γ/W, cycle 2 | 1 | 4 | 14 | 49 | 177 |
| cycle-based, cycle 2 | 1 | 4 | 17 | 86 | 469 |

A count above the exact one always means double counting. Two such mistakes are known, and the test
builds both from the same operations. Schindlmayr and Godby's Eq. (7) used literally after the Hartree
update (the SG-Γ/W vertex without its factor [1 + Λ_H]^(−1)) gives 241 diagrams at v^4 in cycle 4 from
`'H'`, where 238 exist. The vertex taken along L^(1) instead of the start's vertex legs,
Γ^(n) = 1 + (δΣ_xc^(n−1)/δG^(0)) L^(1), with W also built from P, gives 27 at v^3 from cycle 3 on, where
26 exist. In 0-D the cycle-2 difference of the SG schemes reads G^(1) (W^(1) − W^(2)) Γ^(2), its i
absorbed with the other factors.

## Options

`run_cycles(alg, scheme, start, cycles, lag=True)`:
- `scheme`: `'sg_il'`, `'sg_gw'`, `'cb_gw'` or `'cb_il'`,
- `start`: `'H'` or `'HF'`,
- `cycles`: a positive integer,
- `lag=False`: SG-Γ/W with its vertex taken with respect to the Hartree term of G^(n−1) itself, which
  is SG-I/L (refused for the other schemes).

`zero_d.ZeroDBackend(loop_weight=1, coupling=0, order=8, eps_order=1)`:
- `loop_weight`: ℓ,
- `coupling`: the point y_0 of the expansion, 0 for exact power series,
- `order`: the truncation in y, which at a finite coupling must be at least the number of
  derivatives along the map (cycles − 1 for the SG schemes),
- `eps_order`: the order of the fixed-form variable ε, at least the number of cycles of a
  cycle-based scheme and at least 1.

The helpers:
- `zero_d.counts(scheme, start, cycles, order=8, loop_weight=1)`: the counts at v^1 .. v^order,
- `zero_d.kappa_series(scheme, start, cycles, order=8, loop_weight=1, lag=True)`: the coefficients of
  κ^(n) at y^0 .. y^order,
- `zero_d.kappa_at(scheme, start, coupling, cycles, loop_weight=1, lag=True)`: κ^(n) at a coupling.

`counts` and `kappa_series` take `order` from the caller and choose `eps_order` from the cycles.
`kappa_at` chooses both. The counts need an integer ℓ. At a finite coupling a `decimal.Decimal` works
to the precision of its context, and a `fractions.Fraction` is exact but slow beyond four cycles. The
two number types do not mix.

## Results and refusals

`run_cycles` returns a `CycleHistory` with `lines[n]` = G^(n) and `sigmas[n]` = Σ^(n) for
n = 0 .. cycles. For SG-Γ/W it also holds `vertices[n]` = Γ^(n) and `screened[n]` = W^(n) from cycle 2
on, and `legs` = G^(0) Γ^(1) G^(0). For SG-I/L it holds `l1` = L^(1). It refuses, before any work, an
unknown scheme or start, a cycle count that is not a positive integer, `lag=False` outside SG-Γ/W, and
a backend that lacks an operation. The helpers refuse the same, and a non-integer ℓ or an order below
1 for the counts, before any cycle. The 0-D backend refuses a sum of objects that carry different
powers of G^(0) and a jet too short for the derivatives asked of it. No numerical flags are raised
yet, since the 0-D backend is exact or works to a chosen precision.

## Checked in `tests/test_hedin_cycles_zero_d.py`

Against `tests/data/hedin_cycles_reference_values.json` (the theory repository's reference values,
from its own 0-D module at 40 digits) and the independent closed-form maps of
`tests/hedin_zero_d_oracle.py`:
1. every count at ℓ = 1, v^1 to v^8, cycles 1 to 5, both starts, all four schemes, and the exact series,
2. the counts as polynomials in ℓ, v^1 to v^6, cycles 1 to 3, from runs at ℓ = 0 to 8,
3. κ^(n) of the loop and of the oracle at two couplings of each exact model, cycles 1 to 6, to 1e-12
   relative (both agree to 1e-24 at 50 digits), the relative error of G to its six printed digits, the
   reference's own columns and exact models (the boson's closed form through scipy's `erfcx`), and
   the loop against the oracle exactly in fractions at the coupling −1/10,
4. the two double-counting traps (241 and 27, read from the reference file),
5. the 0-D part of the test plan's Tier 1: cycle-based Γ/W = cycle-based I/L, SG-Γ/W without its lag
   = SG-I/L, the SG identities against the BSE with the kernel along the map (for SG-I/L both L^(n)
   and the Σ^(n) closed on it, for SG-Γ/W the vertex Γ^(n)), cycle 1 the same in every scheme with its
   closed form at ℓ = −2, 1 and 3, and the cycle-2 difference of the SG schemes,
6. the exactness orders and the order of the counts,
7. the refusals.

The whole file runs in 0.6 s as a script and in about 2 s under pytest.

## What it leaves out

- The molecular and model-Hamiltonian backend on the Matsubara axis, and with it everything that
  needs orbitals and frequencies: quasiparticle energies, excitation energies, finite differences of
  the kernels, the Hubbard dimer and the comparison with existing GW and BSE(2) routes.
- Counts say how many diagrams a cycle holds, not which ones, and nothing about accuracy, signs,
  cancellations or convergence at finite coupling.
- The Hartree–Fock start with its exchange self-energy held fixed (G0W0@HF as cycle 1) is not a start
  here.
