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

Status: the cycle loop, its zero-dimensional (0-D) backend, which gives the diagram counts and the 0-D
values of every scheme, and the finite-basis backend on the Matsubara axis, which runs cycles 1 and 2 of
every scheme in spin orbitals (the Hubbard dimer and H2/STO-3G are tested). Cycle 3 and beyond, spin
adaptation and larger molecules are the next stages.

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

Any backend with the operations of `algebra.Backend` runs the same loop. On the Matsubara axis, for the
Hubbard dimer (hopping t, on-site interaction U) or a molecule's restricted mean field:

```python
from pyscf import gto, scf
from src.Base.numerics.numerics import Numerics
from src.SingleReference.HedinCycles import run_cycles
from src.SingleReference.HedinCycles.matsubara.backend import MatsubaraBackend
from src.SingleReference.HedinCycles.matsubara.systems import from_pyscf, hubbard_dimer

dimer = MatsubaraBackend(hubbard_dimer(t=1.0, U=0.5), 'HF')        # beta from the gap, default grids
h2 = from_pyscf(scf.RHF(gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')).run())
backend = MatsubaraBackend(h2, 'H', Numerics(hedin={'beta': 20.0, 'm_vertex': 8}), w_order='4,1')
history = run_cycles(backend, 'cb_gw', 'H', cycles=2)
history.sigmas[n].at_grid()                # Σ^(n)(iν) on the fermionic grid, n = 0 .. 2
history.lines[n].at(range(4))              # G^(n)(iν_k), k = 0 .. 3
history.numerical_flags                    # the named flags of the run
```

## What is computed

### The two routes

Γ/W route, with W met at the vertex's density end first (section "The W ordering" below):

    Σ^(n) = Σ_H[G^(n−1)] + i G^(n−1) W^(n) Γ^(n),    Σ_xc(1,2) = i G(1,3) W(4,1⁺) Γ(3,2;4)
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

### The Matsubara backend

`matsubara/` holds systems in spin orbitals (`systems.py`: the Hubbard dimer from its integrals, a
restricted pyscf mean field, any one-electron Hamiltonian h with Coulomb factors B), the start lines
(`starts.py`: aufbau SCF with DIIS, Hartree or Hartree–Fock), the grids (`grids.py`) and the operations
and their tangents (`backend.py`, `tangents.py`). Everything lives in the start line's orbitals, at the
inverse temperature β, with the chemical potential μ in the middle of the start's gap. ν_n = (2n + 1)π/β
is a fermionic and Ω_m = 2mπ/β a bosonic Matsubara frequency. The Coulomb interaction is factorized,
v = Σ_P B^P B^P over N_aux factors (an exact eigendecomposition of the (ij|kl) matrix), so a polarizability
or a screened interaction is an N_aux × N_aux matrix at each Ω_m, and W = Σ_PQ B^P M_PQ B^Q with
M = (1 − P)^(−1).

**Lines and sums.** A line is G(iν) = [iν + μ − h_G − Σ_dyn(iν)]^(−1). h_G is the static part of its
inverse (for G^(n), the start's levels plus the static part of Σ^(n) − Σ^(0)), and Σ_dyn its
frequency-dependent self-energy, kept on the one-particle grid. Its *model line* is the same with
Σ_dyn = 0. Every sum over all
fermionic frequencies (a density, a pair sum G X G, a triangle of three lines) is the model line's sum,
done exactly through Fermi functions and their divided differences, plus the rest summed where it lives:
on the one-particle grid, where a line differs from its model, and on the vertex box, where a direction
depends on ν.

**Three-point objects.** A vertex, the response of a self-energy and a kernel applied to L carry a pair
end (outgoing ν + Ω, incoming ν) and a density end contracted with a factor B^Q, the source of frequency
Ω. Each is stored as the bare vertex (vertices only), a part independent of ν on every Ω of the boson
grid, and a part on the vertex box (2 n_vertex fermionic frequencies by 2 m_vertex + 1 transfers), zero
outside it.

**Tails.** The self-energy of a three-point object, −(1/β) Σ_Ω C(Ω) G(ν + Ω) X(ν; Ω) with C the coupling
(v or W − v), runs over the boson grid for the static part and over the transfer box for the rest. Beyond
the cutoff m_c (`m_boson` for the static part, `m_vertex` for the rest) the coupling and X side by side
decay as Z_1(ν)/(iΩ). Z_1 is taken from the two ends of the cutoff, where the next order cancels, and G is
replaced by its model line. For a level x of the model line and z = iν − x, the sum
(1/β) Σ_{|m|>m_c} 1/((iΩ_m + z) iΩ_m) is a difference of digamma functions. The frequency-dependent part
of a vertex decays as 1/Ω, so without the tail the transfer box converged as its inverse (on the
half-filled dimer at β = 40/t and U = 0.1t, SG-I/L's cycle-2 error was 3.6e-5, 2.3e-5 and 1.5e-5 for
m_vertex = 8, 16 and 32, against 4.9e-6 converged). With it every cutoff converges as its cube. The tail is linear in the summand at the two
ends, so the Γ/W and I/L forms of one self-energy, whose summands agree transfer by transfer, agree with
their tails as well.

**Tangents.** A derivative is a tangent: the response of every object of a cycle to a perturbation
δG(ν; Ω) = G(ν + Ω) Y(ν; Ω) G(ν) of a leaf line, one transfer Ω at a time, for a set of directions Y
(amputated, three-point objects themselves). A Dyson line has Y = Y_0 + δ(Σ − Σ^(0)), with Y_0 the start
line's direction. Every operation has its tangent rule, and the rules carry the tails of their own boson
sums (the line term with a sum of two model lines over 1/(iΩ), minus the divided difference of the
one-line weight). At Ω = 0 a tangent is then the derivative of the discrete self-energy, tails included,
and agrees with finite differences to 1e-9. At Ω ≠ 0 it is the same construction, whose shifted sums
lose the frequencies near the grid's edge: its triangles agree with brute-force sums, and its other
rules are held by the identities between the routes. The vertex and boson tangents are built one
transfer at a time, and the internal sums run in blocks of frequencies that fit
`Numerics.layout.tile_gb`. Along the map the leaf is the start line and Y_0 its legs' vertex. At fixed form the leaf is a copy of the line, and `Kernel.apply(Y)` is the
tangent of the level functional there.

**The BSEs.** A static kernel (Hartree and exchange, the start's) makes the BSE a pair-space solve at
every Ω. A kernel at fixed form depends on frequency: Γ^Q = B^Q + x^Q with x = K G (B + x) G is solved by
GMRES for every source and transfer at once, each iteration one kernel application. On the dimer it
takes 8 applications at U = 0.1t, 16 to 21 from (HF) at U = 1.9t to 3t, and 11 from (H) at 6.5t to 8t.

**The W ordering.** A truncated cycle's P^(n) = G G Γ^(n) carries its vertex at one end only, so P^(n)
and W^(n) are not symmetric in general. Hedin's derivation gives W(4,1), with the vertex end first,
before the symmetry W(1,4) = W(4,1) of the exact theory is used. That order keeps the Γ/W route an exact
rewriting of the I/L route when the vertex is a response of the line, so SG-Γ/W without its lag equals
SG-I/L and the two cycle-based schemes coincide. The printed form W(1,4) holds the mirror-image diagrams,
with the same counts and the same 0-D values. The two agree wherever P is symmetric: in 0-D, at cycle 1,
on the particle-hole symmetric half-filled Hubbard dimer, and at cycle 2 of the cycle-based schemes from
`'H'` (whose kernel is a second derivative of the GW functional) up to the vertex box. A molecule has no
such symmetry: H2/STO-3G, though half filled, has an asymmetric P^(2). `w_order` chooses.

**Instabilities.** Where an eigenvalue of K χ(Ω = 0) crosses 1, with χ the pair propagator, the BSE
matrix becomes singular (the RHF triplet instability of the dimer crosses between U = 2.0t and 2.0002t at
β = 10/t). The density sources of the vertex and of L drive only the charge channel, so the cycles run
through such a crossing and beyond it, finite, where they describe an unstable reference. Their
treatment by the soft-mode split is a later stage. Near a crossing in a channel the sources do drive,
only the GMRES of a kernel at fixed form is checked: it slows and is flagged when it stops short. The
static pair-space solves (the start's BSE, the SG cycles, the static vertex tangents) have no check, so
there they return the inverse of a nearly singular matrix, unflagged.

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

`matsubara.backend.MatsubaraBackend(system, start, numerics=None, w_order='4,1')`:
- `system`: a `systems.System` (`hubbard_dimer(t, U, eps, nelec)`, `from_pyscf(mf)`,
  `spin_orbital(h, factors, nelec)`),
- `start`: `'H'` or `'HF'`,
- `numerics`: a `Numerics`, whose `hedin` section gives `beta` (1/Hartree, None for `HEDIN_BETA_GAP`
  = 56 over the start's gap, so that the distance to T = 0, of order exp(−β·gap/2), is 7e-13),
  `n_fermion` and `m_boson` (the one-particle and boson grids, default 128 each), `n_vertex` and
  `m_vertex` (the vertex box, default 16 each, inside those grids), `bse_tol` and `bse_max_iter` (the
  GMRES of a kernel at fixed form, 1e-10 relative in at most 300 inner iterations, in restart cycles of
  `HEDIN_BSE_RESTART` = 60) and `scf_tol` and `scf_max_iter` (the start's SCF). The grid defaults are
  provisional, measured on the dimer and H2 only. Keep `m_vertex` well inside `m_boson` (on H2 the
  routes' identities lose a factor 10 when it is a quarter of it) and `n_vertex` ≥ `m_vertex` (mirroring a
  vertex shears its box by up to `m_vertex`, and the asymmetry this leaves in P^(2) falls as `n_vertex`
  grows). The defaults keep `m_vertex` = `m_boson`/8 and `n_vertex` = `m_vertex`. A `Grids` is read as a
  `hedin` section with those grids and every other setting at its default. The tangent engine's working
  set follows `numerics.layout.tile_gb` (0.25 GB when it is None).
- `w_order`: `'4,1'` (the default) reads Σ_xc with W(4,1), `'1,4'` with the printed W(1,4).

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
powers of G^(0) and a jet too short for the derivatives asked of it. The Matsubara backend refuses an
unknown `w_order`, a `hedin` section (or `Grids`) with a non-positive β or count or a vertex box outside
its grids, a `run_cycles` start other than the one it was built on, and a start without a gap (the Hartree start of a one-electron system, whose level is degenerate and
half filled), before any cycle. A start whose SCF does not converge raises.

`CycleHistory.numerical_flags` holds the named flags raised during the run. The Matsubara backend raises
`'hedin_bse_not_converged'` (`HedinBSEWarning`) when the GMRES of a kernel at fixed form stops above
`bse_tol`, and keeps its last iterate. The 0-D backend raises none.

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

## Checked in the Matsubara tests

`tests/test_hedin_cycles_matsubara.py` (fast, 52 s): cycle 1 against the house pole sums of
`BSE.second_order_qp.cycle_self_energy` (G0W0 on the Hartree levels, GW with the TDHF vertex and screening
on the HF levels) for the dimer and H2/STO-3G, on the diagonal of Σ (on these systems the off-diagonal
elements vanish by symmetry), converging as the cube of the grids, cycle 1 the same in all four schemes
(to 2e-15), the refusals, the tangents along the map against finite differences (to 1e-9) and gauge
covariance (to 1e-16), the SG cycle-2 difference (in either reading of W) and the lag identity from both
starts (with W(4,1), and the printed W(1,4) breaking it wherever P^(2) is not symmetric), the kernels at
fixed form against finite differences (to 2e-9), cycle-based Γ/W equal to cycle-based I/L at cycle 2,
`w_order` changing neither the cycle-1 tangents nor the kernels at fixed form, the `hedin` section and
the flag, and the triangles at Ω ≠ 0 on the cycle-1 line against brute-force sums (to 1e-9, limited by
the brute force's window).

`tests/test_hedin_cycles_h2.py` (fast, 42 s): the same Tier-1 checks on H2/STO-3G, the lag identity from
both starts included, whose Coulomb factors are not diagonal in one basis, so that the tails of the boson
sums do not cancel as on the dimer.

`tests/test_hedin_cycles_explicit_bse.py` (fast, 21 s): the SG shortcuts against the BSEs with the kernel
along the map, K^(2) = (δΣ^(1)/δG^(0)) J^(−1), with J the map of a start-line direction to the cycle-1
line, solved on the dimer by GMRES with a GMRES for J^(−1) inside every kernel application. On the I/L
route (7e-14) this reduces exactly at cycle 2 to the start's BSE in the form the tangents apply it, so it
checks the start's L^(1) and the solver. The Γ/W vertex of SG-Γ/W without its lag (1.3e-13) does not
reduce: J, the Hartree response Λ_H and the factor (1 + Λ_H)^(−1) all enter. On H2/STO-3G the reduced I/L
form holds at the L^(1) of the pair-space solve (5e-16).

`tests/test_hedin_cycles_dimer.py` (fast, 53 s) and `tests/test_hedin_cycles_dimer_full.py` (slow, about 4
minutes, on ADA): exactness against exact diagonalization on the half-filled dimer (the dimer report is
AutomaticEquationGenerator's notes/rw_branch_test_data/dimer/convergence/REPORT.md), whose exact
self-energy on its Hartree line is purely second order, (U²/4)/(iν − 3t) in the bonding orbital. The
error of [G^(n)]^(−1) in the bonding orbital at the four lowest frequencies (β = 20/t, U = 0.1 and 0.05):

| scheme | from `'H'` | from `'HF'` |
|---|---|---|
| cycle 1 | 0.069 U², slope 1.86 | 0.033 U³, slope 2.97 |
| SG-I/L, cycle 2 | 0.049 U⁴, slope 3.90 | 0.0110 U⁴, slope 3.89 |
| SG-Γ/W, cycle 2 | 0.054 U³, slope 2.81 | 0.0301 U⁴, slope 3.90 |
| cycle-based Γ/W, cycle 2 | 0.026 U³, slope 2.79 | 0.021 U⁴, slope 3.92 |

The slopes carry each scheme's next order (SG-I/L from `'H'` has a U⁵ term of about −0.06 U⁵). SG-Γ/W
from `'H'`, both SG schemes from `'HF'` and the dimer report's own cycle-based kernel K[G^(1), W[h]] (W
from the start line, built in the test from the algebra) agree with the report's large box to 2 %. SG-I/L
from `'H'` gives 0.049 U⁴ where the report has 0.034 U⁴. Here it is converged in every cutoff and at
β = 20/t and 40/t, while the report's value halved between its two boxes and sums the vertex without a
tail. Off half filling (site energies 0 and 0.4t) the SG schemes are exact through each cycle's order
(slopes 1.89, 3.50 and 2.87 from `'H'`, 2.98, 3.91 and 3.94 from `'HF'`). SG-I/L's extra order is special
to half filling: here it keeps a small U³ term. The dimer file also runs cycle 2 past the instabilities:
every scheme just past the RHF crossing (U = 2.0002t, with the largest eigenvalue of K χ(0) asserted below
1 at 2.0t and above it there) and at 3t from `'HF'`, every scheme at 8t and both SG schemes at 16t from
`'H'`. The slow file also asserts that no GMRES stopped short.

## What it leaves out

- Cycle 3 and beyond on the Matsubara axis: the tangent of a self-energy built on a frequency-dependent
  vertex, and the kernels of a kernel, are not written yet (`NotImplementedError`).
- Spin adaptation (spin orbitals only), molecules beyond H2/STO-3G, quasiparticle and excitation
  energies by continuation, and the comparison with the BSE(2) routes.
- The treatment of instabilities: the cycles run through them, but an unstable channel is not replaced.
- Counts say how many diagrams a cycle holds, not which ones, and nothing about accuracy, signs,
  cancellations or convergence at finite coupling.
- The Hartree–Fock start with its exchange self-energy held fixed (G0W0@HF as cycle 1) is not a start
  here.
