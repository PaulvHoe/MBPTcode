# Parquet approximation

The parquet approximation for molecules of Marie and Loos,
[J. Chem. Phys. 163, 194115 (2025)](https://doi.org/10.1063/5.0301331), in the
static kernel form of its Sec. IV A, ported from their program Quack (the
tree the paper was computed with, git e3528a40) into
`src/SingleReference/Parquet/`. It gives quasiparticle energies, the principal
ionization potentials of the paper, from a self-energy that holds the
second-order diagrams and both two-body channels at once, electron-hole (eh)
and particle-particle (pp), coupled: the Bethe-Salpeter equation (BSE) of each
channel has a kernel built from the reducible vertices of the channels, and
the vertices are iterated to self-consistency before the self-energy is built
from the eh and pp spectra.

The port follows Quack step by step, its oddities included, with Quack's
order of operations wherever that decides a number (each denominator, the
damping, the convergence tests); sums and matrix products run in other
orders, so its numbers agree with Quack's to rounding. One step is the
house's by default: the quasiparticle equation is solved by the house's root
finder ([Options](#options)). The package docstring (`Parquet/__init__.py`)
maps every Quack routine to the function that ports it and lists what is not
ported, what of Quack's behaviour is kept on purpose and where the port
departs from it. The port was checked against Quack's own routines, called
from Python, on small molecules, and against the paper's Table I for five of
its eight molecules and Table II for Ne at s_2b = 1 and 10
([Reproducing the paper](#reproducing-the-paper)).

## Running it

```python
from pyscf import gto, scf

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.Parquet.restricted import run_parquet

mol = gto.M(atom='O 0 0 0; H 0.9591 0 0; H -0.2373 0.9293 0', basis='cc-pvdz',
            verbose=0)
mf = scf.RHF(mol).run(conv_tol=1e-10)

# osPA, TDA, s_2b = 1, unregularized self-energy: Table II's settings
res = run_parquet(mf, 'os', eta_1b=1e8, eta_2b=1.0, max_diis_2b=5)
homo = res.eqp[:res.nO].argmax()
print(-res.eqp[homo] * HARTREE_TO_EV, res.z[homo])     # 11.774 eV, Z 0.926
print(res.converged, res.n_it_2b)                       # True [35]
print({f['flag'] for f in res.numerical_flags})         # set()
```

Energies are in Hartree; the principal IP is minus the highest occupied
quasiparticle energy. This run takes about five seconds on one thread, most
of it the house's root finder on the quasiparticle equation of every orbital
(about 100 000 evaluations of the self-energy). It raises no flag: an energy
kept with Z outside (0, 1], beside the dense poles of an unregularized
self-energy, would be kept and flagged `'z_outside'`, not refused
([Results and flags](#results-and-flags)).

## What is computed

**Two nested loops** (the paper's Fig. 8). The *two-body loop* holds the
one-body energies and the integrals fixed and iterates the reducible vertices
Phi^eh and Phi^pp (closed shell: singlet and triplet of each, four tensors of
norb^4 numbers). One iteration builds the static kernels of the eh and pp
BSEs from the previous vertices (zero at the first iteration of the run, and
in qsPA at the first two-body iteration of every one-body iteration), solves
the BSEs, contracts their eigenvectors with the irreducible vertices into the
effective integrals M^eh, M^ee and M^hh, and builds new vertices from those
and the new eigenvalues. The change max |Phi_new - Phi_old| is taken before
the vertices are damped (and, optionally, extrapolated by DIIS); the loop
ends when it is at most `conv_2b`. The *one-body step* then builds the
self-energy from the last spectra alone: for FLEX, osPA and evPA its diagonal
Sigma_pp(w) and the quasiparticle equation w = eHF_p + Sigma_pp(w); for qsPA a
static Hermitian matrix added to the Fock matrix, which is diagonalized.

| `scheme` | paper | what runs | Quack |
|---|---|---|---|
| `'flex'` | FLEX | one two-body iteration with zero kernels (the BSEs are the eh-RPA and pp-RPA problems, or their TDA), one one-body step | R_evParquet / G_evParquet, max_it_1b = max_it_2b = 1 |
| `'os'` | osPA | the two-body loop to convergence on the Hartree-Fock energies, one one-body step | the same, max_it_1b = 1 |
| `'ev'` | evPA | one-body self-consistency on the diagonal quasiparticle energies: each one-body iteration converges a two-body loop on the previous energies | R_evParquet / G_evParquet |
| `'qs'` | qsPA | a static Hermitian self-energy added to the Fock matrix, diagonalized: orbitals and energies both iterate | R_qsParquet / G_qsParquet |

In evPA the vertices, the previous eigenvalues and the two-body DIIS history
carry over from one one-body iteration to the next, as in Quack, so each
two-body loop starts from the last one's vertices; qsPA resets all three at
every one-body iteration (the orbitals changed).

**Static kernels.** Every reducible vertex inside a BSE kernel is taken at
zero bosonic frequency (the static kernel approximation, the paper's
Eqs. (28) and (47)): each two-body iteration builds the vertices at omega = 0
(Eqs. (39) and (45)), as Quack's R_*_Phi and G_*_Phi are called. The
self-energy is not static: it takes each channel's eigenvalues Omega and
effective integrals M, the whole frequency dependence of Eqs. (37) and (43).
A vertex averaged over a frequency window can take the place of the point
omega = 0, at all of its insertions or in each of its roles on its own
([Frequency-averaged vertices](#frequency-averaged-vertices)).

**Regularization.** The denominators d (Hartree) of the vertices and of the
self-energy are regularized as Quack regularizes them, with the
energy-dependent regularizer of the paper's Eqs. (52) and (53),

    R(d) = (1 - exp(-2 s d^2)) / d,        s in Ha⁻²,

so s = 0 makes R vanish and s -> infinity is the bare 1/d (s = 1e8 is
unregularized in practice: the paper's eta = 0). There are two flow
parameters: s_2b (`eta_2b`) on the vertex denominators omega -+ Omega,
Eq. (52), where s_2b = 0 makes every vertex vanish and osPA becomes FLEX;
and s_1b (`eta_1b`) on every denominator of the self-energy, the
frequency-independent ones included, Eq. (53). Each self-energy term is a
product of regularized denominators; where both bare denominators of a term
hold w, Quack's closed-shell and spin-orbital branches regularize it in two
ways (partial fractions first, or as it stands: `restricted_self_energy`),
and so do the two drivers, which therefore agree as s_1b grows and differ at
finite s_1b ([The two drivers](#the-two-drivers)). Z_p = 1/(1 - dSigma_pp/dw)
is Quack's: each term is differentiated with its regularizer held fixed,
which is not the derivative of the regularized self-energy. At an exactly
zero denominator, where Quack's formula gives 0/0, the port returns 0 for
R(d), its limit, and 0 for the slope R(d)/d in Z, the value of Quack's
formula a few ulps off zero, where 1 - exp(-2 s d^2) rounds to 0 (the
slope's limit would be 2s): a zero mode's pole adds nothing to Sigma or to
its derivative, whether rounding puts the mode at exactly 0 or a few ulps
off. H/STO-3G's spin flip, e_beta - e_alpha - J, zero in exact arithmetic,
comes out 2.2e-16 (two ulps of J) or exactly 0 as the mean-field energies
and J round, and gives the same numbers either way in the eh TDA. Without
the TDA the eh eigenvectors carry Omega^-1/2
(X + Y = Omega^-1/2 Z^T (A - B)^1/2), which an exactly zero root makes 0
times inf: the run is NaN, as in Quack.

The static self-energy of qsPA comes in two forms. The SRG (similarity
renormalization group) form, the paper's (`reg_pa=True`), regularizes each
term's bosonic denominator as R(d) and its frequency denominator, at the
pair of energies (eps_p, eps_q), with the two-orbital kernel

    K(D_p, D_q) = (D_p + D_q) / (D_p^2 + D_q^2) (1 - exp(-s_1b (D_p^2 + D_q^2))),

K(D, D) = R(D), the form of the SRG-regularized qsGW (Marie and Loos,
[J. Chem. Theory Comput. 19, 3943 (2023)](https://doi.org/10.1021/acs.jctc.3c00281));
the matrix is symmetrized. This is Quack's R_Parquet_self_energy_SRG_sp. The
spin-orbital G_qsParquet calls G_Parquet_self_energy_SRG_dp instead, which
agrees with it on the diagonal as s_1b grows but not off it, at any s_1b. The
eta form (`reg_pa=False`, closed shell only, R_Parquet_self_energy_eta) first
recombines the two terms of a group that share a bosonic denominator, which
cancels it, then replaces every denominator x by x / (x^2 + eta^2) with a
broadening eta (`broadening_1b`, Hartree), the frequency denominators at
omega = eps_p; its matrix is not symmetric before the symmetrization.

**Tamm-Dancoff approximation**, per channel: `tda_eh` and `tda_pp` (Quack's
TDAeh and TDApp; both True by default, as in the paper's Tables II and III).

- eh: with the TDA the excitations are the eigenpairs of A; without it, of
  (A - B)^1/2 (A + B) (A - B)^1/2 (Quack's phRLR, phGLR), which needs A - B
  positive definite.
- pp: with the TDA the double attachments (ee) are the eigenvalues of C and
  the double removals (hh) those of -D; without it the general matrix
  [[C, B], [-B^T, -D]] is diagonalized (dgeev), split at zero energy into the
  two branches and each branch normalized in the pp metric (Quack's ppRLR,
  ppGLR and sort_ppRPA).

The kernels follow the same switches. The correlation energies `ec_eh` and
`ec_pp` are Quack's EcRPA: for eh half the sum of the eigenvalues less tr A,
for pp half of the sum of the ee eigenvalues less that of the hh ones less
tr C and tr D. Under the TDA both vanish in exact arithmetic and are rounding
noise; they are correlation energies only for the full BSEs. The paper's
Table I has both (FLEX RPA without, FLEX TDA with).

**Soft and unstable eh modes.** Without the eh TDA a closed shell can meet
triplet instabilities of the eh problem as the vertices grow (the paper's
Sec. VII B). A closed shell breaks no continuous symmetry, so it has no zero
mode of its own, but at the edge of an instability its lowest eh excitation
goes to zero, X + Y grows as Omega^-1/2 and the self-energy as 1/Omega:
H2/6-31G 1e-6 Angstrom before its RHF triplet instability has Omega =
3.6e-4 Ha and Sigma_c of the HOMO -5346 eV at s_1b = 50, and every
quasiparticle root the finder keeps has Z below 0.012 (the HOMO's lies at
+22.9 eV). Both drivers flag such a root below 1e-3 Ha
(`'parquet_eh_zero_mode'`) and keep the numbers; past the instability the
excitation is imaginary (`'parquet_eh_negative_excitations'`), and a negative
Tamm-Dancoff one is flagged too (`'parquet_eh_tda_negative_excitations'`).

A collinear open shell always has a zero mode, the rotation of its spin
axis: OH UHF/6-31G FLEX without the TDA comes out NaN, flagged
`'parquet_eh_a_minus_b_not_positive'`. Run open shells with `tda_eh=True`
(the default), and at a moderate s_2b. In the TDA such modes are small
rather than zero, and nothing flags them: OH UHF/6-31G has eh TDA roots at
7.5e-4 Ha (the rotation of its pi hole about the bond) and 3.0e-3 and
7.4e-3 Ha (spin flips). A vertex weighs such a root by R(Omega), about
2 s_2b Omega while 2 s_2b Omega^2 is small and 1/Omega once it nears 1
(1338 Ha⁻¹ for OH's lowest), so these roots matter only at weak two-body
regularization: OH's FLEX vertex (largest element) is 2.2 at s_2b = 50 and
40, 347 and 516 at 1e4, 1e6 and 1e8, where water's stays at 1.85, and OH's
osPA with two-body DIIS 5 and damping 0.75 converges at s_2b = 1 and
diverges at 1e6 and 1e8. In evPA and qsPA an open shell's eh TDA problem
has negative spin-flip roots from the second one-body iteration on, before
any kernel: the quasiparticle energies narrow the gap of a spin flip below
its exchange term (OH UHF/6-31G after one one-body iteration: -0.18 Ha in
evPA, -0.23 Ha in qsPA), so each later two-body iteration is flagged
`'parquet_eh_tda_negative_excitations'`. At s_2b = 1 both still converge
(evPA with two-body DIIS 5 and damping 0.75 in 12 one-body iterations, qsPA
in 10); osPA, on the Hartree-Fock energies, raises no such flag.

## The two drivers

| | `restricted.run_parquet(mf, scheme)` | `spin_orbital.run_parquet_so(mf, scheme)` |
|---|---|---|
| reference | a real closed-shell RHF | a real RHF, UHF or GHF |
| Quack | R_evParquet, R_qsParquet | G_evParquet (called directly: Quack's own GQuAcK program runs G_evParquet_v2 instead), G_qsParquet |
| arrays | spin-adapted, singlet and triplet, norb^4 | spin orbitals, (2 norb)^4 = 16 norb^4 |
| diagonal self-energy, terms whose two denominators hold w | split into partial fractions, R(Delta) R(w - x) | as they stand, R(w - x1) R(w - x2) |
| qsPA static self-energy | SRG_sp, or the eta form | SRG_dp |
| default `damping` | 0.75 for every scheme | 0 for flex, os, ev (Quack's, see below); 0.75 for qs |
| result | `ParquetResult` | `SpinOrbitalParquetResult` |

Both live in `SingleReference.Parquet`. Take a closed shell to
`run_parquet`: the paper's numbers are this driver's, its arrays are a
sixteenth of the size, and closed-shell qsPA as the paper runs it exists only
here, since R_qsParquet's SRG_sp and G_qsParquet's SRG_dp are other matrices.
Take an open shell to `run_parquet_so`, with the eh TDA. `run_parquet_so`
reproduces G_evParquet called directly, not the output of Quack's GQuAcK
program, which runs G_evParquet_v2 for evParquet (it mixes and extrapolates
the BSE matrices instead of the vertices).

On one closed shell the two drivers give other FLEX, osPA and evPA energies at
finite s_1b and another qsPA. On water/6-31G at the geometry of the tests
(FLEX with the full BSEs, whose spectra agree to 4e-13 Ha): at s_1b = 50 the
linearized HOMO lies 2.4 meV apart, the 2a1 level 23 meV, the farthest level
0.18 eV; at s_1b = 1e8 they agree to a few 1e-11 eV (1.3e-11 eV; 3.7e-11 eV
at the QUEST geometry of [Reproducing the paper](#reproducing-the-paper)).
qsPA at the paper's Table III settings puts their HOMOs 17 to 21 meV apart
(two runs with different DIIS lengths, each converged to 1e-4). For an RHF,
`run_parquet_so` solves the alpha and beta copies of each orbital as one pair
(`ev_parquet_so`'s `spin_pairs`), so they stay degenerate; a UHF or GHF of a
closed shell gets no pairing, and its evPA at large s_1b can then depend on
roundoff. The two drivers' converged two-body loops stop at other counts on
the same closed shell (water/6-31G osPA, s_2b = 1, damping 0.75 in both,
`conv_2b=1e-8`: 151 and 143 iterations), because each tests the change of its
own vertices.

An open shell, run as written here:

```python
from pyscf import gto, scf

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.Parquet.spin_orbital import run_parquet_so

mol = gto.M(atom='O 0 0 0; H 0 0 0.9697', basis='6-31g', spin=1, verbose=0)
mf = scf.UHF(mol).run(conv_tol=1e-10)

res = run_parquet_so(mf, 'os', eta_2b=1.0, max_diis_2b=5)    # TDA, s_1b = 50
top = res.eqp[:res.nO].argmax()
print(-res.eqp[top] * HARTREE_TO_EV)                   # 11.961 eV
print(res.spin[top], res.orbital[top])                 # 1 3
print(res.converged, res.n_it_2b, res.options['damping'])   # True [19] 0.0
```

The principal IP comes from the beta spin orbital of the mean field's orbital
3 (`res.spin` is 0 for alpha and 1 for beta; -1 for every spin orbital of a
GHF). One virtual spin orbital is flagged: its root has Z = 1.006
(`'z_outside'`).

**The defaults are Quack's, not production settings.** The comparisons with
Quack test the implementation, and the defaults are Quack's so that those
comparisons hold: the paper's Table III s_1b = s_2b = 50 and tau = 1e-4, no
DIIS, and Quack's damping, `damping=0.75` (Quack's mixing weight alpha =
0.25) in `run_parquet` and in `run_parquet_so`'s qsPA, none in
`run_parquet_so`'s flex, os and ev (G_evParquet's alpha = 1). They were not
chosen to give numbers. At s_2b = 50 the
undamped two-body loop without DIIS diverges: `run_parquet_so`'s osPA of
water, ammonia (RHF) and OH (UHF) in 6-31G runs its 200 two-body iterations
to a change of 1e122 to 1e128, warns `'parquet_two_body_not_converged'` (with
`'parquet_pp_ec_inconsistent'` and `'parquet_eh_tda_negative_excitations'` on
the way) and returns energies that mean nothing (principal IPs of 27.2 eV
for water and 34.0 eV for OH with Z of 1e-117, and spin orbitals without a
root, flagged `'no_root'`, after two to three minutes of root search on
that self-energy). What converges, on the same three (osPA, TDA,
s_1b = 50, one BLAS thread; the two-body iterations to tau = 1e-4, or the
change left after 200):

| `run_parquet_so(mf, 'os', ...)`, 6-31G | water | ammonia | OH |
|---|---|---|---|
| s_2b = 50, the defaults (damping 0, no DIIS) | diverges (1e122) | diverges (4e124) | diverges (2e128) |
| s_2b = 50, `damping=0.75` (the restricted driver's) | not converged (2.8) | not converged (0.017) | diverges (5e16) |
| s_2b = 50, `max_diis_2b=5` | 154 | 60 | diverges (2e13) |
| s_2b = 50, `damping=0.75, max_diis_2b=5` | 46 | 43 | not converged (1e-3) |
| `eta_2b=1.0`, the defaults | 170 | 85 | not converged (0.08) |
| `eta_2b=1.0, damping=0.75` | 47 | 46 | 56 |
| `eta_2b=1.0, max_diis_2b=5` (the example above) | 19 | 16 | 19 |

So give an open shell two-body DIIS or damping, at a moderate s_2b: OH at
s_2b = 50 converges with none of these. The damped loop of `run_parquet` at
its defaults does not converge at s_2b = 50 either: `run_parquet(mf)` (osPA)
on water and ammonia in 6-31G at the geometries of [Reproducing the
paper](#reproducing-the-paper) ends its 200 iterations at a change of 5.6
and 0.017 Ha, flagged, with a principal IP of -0.77 eV for
water (in cc-pVDZ: a change of 26 Ha, an IP of -32 eV); with `max_diis_2b=5`
they converge in 53 and 54 iterations ([Table III](#reproducing-the-paper)).

Below the front doors, `ev_parquet(eps_hf, eri, nO, scheme)` and
`qs_parquet(eps_hf, c_hf, s, x, hc, eri_ao, nO)` (and `ev_parquet_so`,
`qs_parquet_so`) take the arrays Quack's routines take: orbital energies,
`<pq|rs>` in physicist order, the AO overlap, orthogonalizer, core Hamiltonian
and AO integrals.

## Options

Quack's names are those of its `input/options` line for the parquet
methods; its defaults are those of the shipped `input/options.default`. The
drivers take these as keyword arguments, Quack's option set: they are not
read from the run's `Numerics` object, as the house's evGW and qsGW loops read
theirs (evGW's damping, the same weight, is `Numerics(evgw={'damping': d})`).
A wrong option is refused with a ValueError (frozen orbitals:
NotImplementedError) before any integral is built.

| option | meaning | unit | default | Quack (default) |
|---|---|---|---|---|
| `scheme` | `'flex'`, `'os'`, `'ev'` or `'qs'` | | `'os'` | evParquet or qsParquet in the methods file (F, F); FLEX and osPA are evParquet with caps of 1 |
| `tda_eh`, `tda_pp` | Tamm-Dancoff approximation of the eh and the pp BSEs | | True, True | TDAeh, TDApp (T, T) |
| `eta_1b` | s_1b, the self-energy's flow parameter, Eq. (53); for qsPA with `reg_pa=True` the s of the static self-energy; not read under `reg_pa=False` | Ha⁻² | 50 (Table III) | reg_1b, passed as eta_1b (1000.0) |
| `eta_2b` | s_2b, the vertices' flow parameter, Eq. (52) | Ha⁻² | 50 (Table III) | reg_2b, passed as eta_2b (1000.0) |
| `max_it_1b` | one-body iteration cap; reaching it counts as failed | count | flex and os 1 (the only value they take); ev and qs 50 | max_it_1b (1) |
| `conv_1b` | tau_1b: evPA max \|eQP - eOld\|, qsPA max \|F P S - S P F\|; in (0, 1), since Quack's loops start at an error of 1 | Ha | 1e-4 (the paper's) | conv_1b (0.01) |
| `max_it_2b` | two-body iteration cap; reaching it counts as failed | count | flex 1 (the only value); otherwise 200 | max_it_2b (1) |
| `conv_2b` | tau_2b: max \|Phi_new - Phi_old\| over the vertices, before damping; in (0, 1) | Ha | 1e-4 (the paper's) | conv_2b (0.01) |
| `max_diis_1b` | one-body DIIS length: evPA on eQP (error eQP - eOld), qsPA on F (error F P S - S P F); 1 is none | count | 1 | max_diis_1b (1) |
| `max_diis_2b` | two-body DIIS length, over all the vertices as one vector, error Phi_damped - Phi_old; 1 is none | count | 1 | max_diis_2b (1) |
| `damping` | the weight d of the previous vertices, Phi <- (1 - d) Phi_new + d Phi_old, as evGW's damping means it; in [0, 1) | | `run_parquet` 0.75; `run_parquet_so` None: 0 for flex, os, ev (Quack's G_evParquet, which diverges at s_2b = 50 without DIIS: [The two drivers](#the-two-drivers)), 0.75 for qs | not an input: Quack's mixing weight alpha = 1 - d is hard-coded, 0.25 in R_evParquet, R_qsParquet and G_qsParquet, 1 in G_evParquet |
| `qp_solver` | how FLEX, osPA and evPA solve w = eHF_p + Sigma_pp(w): `'find_qp_root'` (the house's one root finder), `'linearized'` (eQP = eHF + Z SigC), or a function `qp_solver(sigma, eps_hf, eqp_lin, z_lin, orbitals) -> (eqp, z, converged)`, one contract for both drivers (`Parquet.options.checked_qp_solver`); qsPA refuses any value | | None: `'find_qp_root'` | lin_parquet (F: Quack's own search, not ported; G_evParquet runs only the linearized branch) |
| `reg_pa` | `run_parquet`, qsPA: True the SRG static self-energy, False the eta form (flex, os and ev refuse False) | | True | reg_PA (T; R_evParquet, G_evParquet and G_qsParquet take it and ignore it) |
| `broadening_1b` | `run_parquet`, qsPA with `reg_pa=False` only, where it is required: the broadening eta, finite and > 0 | Ha | None | Quack reads reg_1b for it, the input that is the flow parameter elsewhere |
| `vertex_builder` | the reducible-vertex hook ([The vertex hook](#the-vertex-hook)), per vertex or, with `per_insertion = True`, per vertex and insertion ([One tensor per insertion](#one-tensor-per-insertion)); `averaged_vertices.averaged_vertex_builder` makes frequency-averaged vertices, `insertion_vertex_builder` and `transverse_vertex_builder` per-insertion ones ([Frequency-averaged vertices](#frequency-averaged-vertices)) | | None: the static vertex | R_*_Phi, G_*_Phi called at omega = 0d0 |
| `frozen` | frozen orbitals are not ported: only None or 0 | | None | nC = nR = 0 |

The damping is evaluated as Quack evaluates it, alpha Phi_new +
(1 - alpha) Phi_old with alpha = 1 - d, so the default 0.75 is Quack's 0.25
to the bit, and d = 0 (G_evParquet's alpha 1) still computes 1 Phi_new +
0 Phi_old, so a NaN in the old vertices propagates. The loop damps first and
extrapolates the damped vertices, as Quack does and as the paper describes
("damping prior to DIIS"). Two-body DIIS solves its systems below rcond = eps
without a warning, as Quack does (its error vectors become nearly linearly
dependent), so a run with two-body DIIS can change with the number of BLAS
threads, by up to about its convergence threshold: water/6-31G qsPA at
Table III settings with `max_diis_2b=5` gives a HOMO of 12.038131, 12.038442
and 12.038171 eV at 1, 2 and 4 threads, each converged. `'diis_rcond'` in
the history shows the conditioning.

`qp_solver='find_qp_root'` hands each orbital's equation, with Quack's slope,
to the house's root finder (`src/Solvers/qp_equation.py`, [GW](gw.md)): the
root of largest valid weight in its scan windows about eHF_p at the first
one-body iteration, and from the second on the root continuing the previous
iteration's eQP_p where that root's Z lay in (0, 1], so that evPA stays on each
orbital's branch; an orbital whose previous Z lay outside is searched again as
at the first iteration. A followed root that the finder reports the largest
weight has left is warned about and flagged (`'parquet_qp_branch_flipped'`):
its branch has gone, or a root of larger weight lies in its scan window,
which the finder searches whole for this self-energy, whose residues have
both signs (ammonia/6-31G evPA at s_1b = 50 and s_2b = 1: the N 1s
followed at Z 0.58 beside a root of Z 0.70, flagged from the third
iteration on). The finder costs 150 to 14 000 self-energy evaluations per
orbital without a root to follow (water/aug-cc-pVDZ, FLEX TDA: 7.7 s per
one-body step at s_1b = 50 and 33 s at 1e8) and about as many when it
follows: 150 to 3 200 per orbital on water and ammonia 6-31G (evPA at
s_1b = 50 and s_2b = 1), an evPA iteration after the first costing about
what the first does (13 600 to 16 400 evaluations against 13 300 and
18 100). A qp_solver function is called
once per one-body iteration; what it returns is checked, kept and flagged
(`'parquet_qp_not_converged'`, `'parquet_qp_not_finite'`, `'z_outside'`).

Fixed settings are the `PARQUET_*` constants of
`src/Base/numerics/constants.py`, each with its reason there: the options'
defaults; the error of 1 at which Quack's loops start
(`PARQUET_QUACK_INITIAL_ERROR`), which bounds `conv_1b` and `conv_2b`;
Quack's thresholds (the pp spectrum split at zero energy, qsPA's
orthogonalization threshold, the gates of its Loewdin and EcRPA warnings);
the port's own gate of an eh zero mode, 1e-3 Ha (`PARQUET_EH_ZERO_MODE_TOL`,
also the band about zero in which a spin-orbital Tamm-Dancoff root is not
flagged negative); tolerances of the spin-orbital reference checks, LAPACK
driver choices and block sizes.

## Results and flags

Both drivers return a dataclass, `restricted.ParquetResult` or
`spin_orbital.SpinOrbitalParquetResult`, in Hartree:

- `scheme`: the scheme run; `eqp`: the quasiparticle energies of the last
  one-body iteration, occupied first (qsPA: the eigenvalues of the last
  effective Fock matrix, ascending); `nO` occupied (closed shell: doubly
  occupied) orbitals, `eps_hf` the reference's energies;
- `sigc`: Sigma_pp at w = eOld_p of the last iteration (qsPA: the symmetrized
  static matrix in the final orbitals, whose diagonal Quack prints); `z`: the
  root finder's Z at its root (`'linearized'`: the diagonal Z at eOld, the
  one in `eqp_lin`; a qp_solver function: the z it returned; qsPA: ones);
  `eqp_lin`: eHF + Z SigC (None for qsPA); `qp_converged`: whether the root
  finder found a root, or a function's verdict, per orbital (None for
  `'linearized'` and for qsPA); `qp_followed`, `qp_flipped`: the root
  finder's branch report of the last iteration, per orbital (None but for
  `'find_qp_root'`);
- `spectra` and `vertices`: the channels (Omega, M) of the last two-body
  iteration, which the last self-energy was built from, and the reducible
  vertices after it, damped and extrapolated; with a per-insertion builder
  `insertions` holds the tensor of every vertex in every role and
  `vertices` the ones the eh kernels read ([One tensor per
  insertion](#one-tensor-per-insertion)), otherwise `insertions` is None;
- `ec_eh`, `ec_pp`: the correlation energies of the last BSEs as a
  spin-orbital problem has them (closed shell: singlet + 3 triplet, with the
  two parts in `ec_eh_spin`, `ec_pp_spin`); rounding noise under the TDA;
- `converged`, `converged_1b`, `converged_2b`, `n_it_1b`, `n_it_2b`: Quack's
  verdicts, with None where a scheme has no such loop (FLEX and osPA have no
  one-body verdict, FLEX no two-body one), and False also where a change came
  out NaN;
- `history`: one record per one-body iteration, with every two-body
  iteration's vertex changes, Quack's printed eigenvalue changes and the
  reciprocal condition number of its DIIS system (`'diis_rcond'`), the
  energies and Z, for FLEX, osPA and evPA the quasiparticle solver's
  evaluations of the self-energy (`'qp_n_iter'`) and branch report, and the
  one-body change;
- `times` (wall seconds per stage), `options` (the options that apply to the
  scheme, defaults resolved: `frozen` is not recorded, `qp_solver` only for
  FLEX, osPA and evPA, a function by its module and name, `reg_pa` and
  `broadening_1b` only for the closed-shell qsPA, `eta_1b` as None under
  `reg_pa=False`; `run_parquet_so` adds the `spin_pairs` it handed
  `ev_parquet_so`; the vertex builder by name), `numerical_flags`;
- qsPA adds `mo_coeff`, `mo_coeff_two_body` (the orbitals the last spectra
  and vertices are expressed in) and `sigc_ao`; the spin-orbital result adds
  `reference` (`'RHF'`, `'UHF'` or `'GHF'`) and the `spin` (0 alpha, 1 beta;
  -1 for every spin orbital of a GHF) and `orbital` of each spin orbital
  (qsPA: the spin of each final orbital, -1 where a degenerate alpha/beta
  pair came out mixed, and no orbital label).

A numerical shortfall is warned about and named in `result.numerical_flags`;
the run goes on and returns its last iterate. The warning classes are the
package's `ParquetNotConverged`, `ParquetNotFinite` and `ParquetBSEUnstable`
(`Parquet.events`), and the house's `QPWeightOutside` and `QPNoRoot`
(`'z_outside'`, `'no_root'`) and the RuntimeWarning of a flipped branch; an
unconverged reference is `'scf_not_converged'`. [Warnings, flags and
refusals](../warnings.md) has a row for each. The ones met most often:
`'parquet_two_body_not_converged'` (the cap reached: without two-body DIIS
the damped loop can oscillate from s_2b of about 10, water/6-31G at
s_2b = 10 stalling at a change of 0.26 after 400 iterations, and at the
default s_2b = 50 neither driver's default loop converges, [The two
drivers](#the-two-drivers)), `'z_outside'` (an energy beside a pole of the
regularized self-energy, mostly a high virtual), an open shell's
`'parquet_eh_tda_negative_excitations'` in evPA and qsPA ([What is
computed](#what-is-computed)), and, with a qp_solver function,
`'parquet_qp_not_converged'` (its verdict False). Check a flagged orbital
before using its energy.

A wrong request is refused before any work. With a ValueError: an option out
of its range or of the wrong type, a reference without an occupied or a
virtual orbital, and in `run_parquet_so` an open-shell ROHF and PySCF's
one-electron HF1e; for qsPA, a complex core Hamiltonian, a GHF whose core
Hamiltonian depends on the spin and a reference whose orbital space Quack's
canonical orthogonalization would shrink (all before the two-electron
integrals). With a NotImplementedError: frozen orbitals and a reference the
port does not take (Kohn-Sham, density fitting, an attached environment,
complex orbitals; in `run_parquet` a UHF, a GHF, an open-shell ROHF and
HF1e). A closed-shell ROHF is taken by both drivers, as the RHF it is.

## Reproducing the paper

The paper's runs are all in aug-cc-pVTZ at the QUEST CC3/aug-cc-pVTZ
geometries (Quack's `mol/CC3_AVTZ`, in Angstrom), from RHF; the principal IP
is minus the highest occupied quasiparticle energy (for N2 the 3sigma_g).

| molecule | geometry |
|---|---|
| Ne | `Ne 0 0 0` |
| HF | `H 0 0 0; F 0 0 0.9196` |
| H2O | `O 0 0 0; H 0.9591 0 0; H -0.2373 0.9293 0` |
| NH3 | `N 0 0 0; H 0.3816 0.9375 0; H 0.3816 -0.4687 0.8119; H 0.3816 -0.4687 -0.8119` |
| CH4 | `C 0 0 0; H 1.0879 0 0; H -0.3626 1.0257 0; H -0.3626 -0.5128 -0.8883; H -0.3626 -0.5128 0.8883` |
| BF | `B 0 0 0; F 0 0 1.2685` |
| CO | `C 0 0 0; O 0 0 1.1335` |
| N2 | `N 0 0 0; N 0 0 1.1007` |

The Table I, II and III entries below were computed by the port with the
calls shown and, as the paper's were, Quack's own quasiparticle search (qsPA
solves no quasiparticle equation), which the tests carry for their
comparisons with Quack (`qp_solver=quack_newton` from
`tests/parquet_quack_newton.py`). The default root finder gives the same
principal IPs wherever both ran: to 1e-12 eV, and Z to 2e-7, in FLEX TDA and
RPA at s_1b = 1e8 on Ne, water and ammonia in cc-pVDZ, Table II's osPA on
Ne/cc-pVDZ and Table III's osPA on water/cc-pVDZ; these aug-cc-pVTZ runs with
it are left for the cluster. The port's later changes moved the
quasiparticle energies of its regression cases by less than 1e-13 Ha (other
fields of the cases with two-body DIIS and of qsPA moved more: qsPA's
degenerate orbitals rotated), so the entries stand for the code as it is to
their printed digits. These runs are not part of the test suite.

**Table I** (FLEX, eta = 0: an unregularized self-energy):

```python
mol = gto.M(atom=geometry, basis='aug-cc-pvtz', verbose=0)
mf = scf.RHF(mol).run(conv_tol=1e-10)
# tda=True for FLEX TDA, False for FLEX RPA
res = run_parquet(mf, 'flex', tda_eh=tda, tda_pp=tda, eta_1b=1e8)
ip = -res.eqp[:res.nO].max() * HARTREE_TO_EV
```

| molecule | norb | FLEX TDA, eV (Z) | paper | FLEX RPA, eV (Z) | paper |
|---|---|---|---|---|---|
| Ne | 46 | 20.414 (0.858) | 20.41 (0.86) | 20.036 (0.834) | 20.04 (0.83) |
| HF | 69 | 14.845 (0.791) | 14.85 (0.79) | 14.404 (0.751) | 14.40 (0.75) |
| H2O | 92 | 11.538 (0.756) | 11.54 (0.76) | 11.247 (0.701) | 11.25 (0.70) |
| NH3 | 115 | 10.216 (0.753) | 10.22 (0.75) | 10.228 (0.685) | 10.23 (0.69) |
| N2 | 92 | 14.926 (0.722) | 14.93 (0.72) | 14.710 (0.636) | 14.71 (0.64) |

Every entry agrees with the paper to its printed digits. CH4, BF and CO were
not run (CH4, norb 138, runs on the cluster only). In FLEX s_2b plays no part:
its one two-body iteration has zero kernels, and the vertices it builds are
not read.

**Table II** (osPA, TDA, s_1b -> infinity, tau_2b = 1e-4, as a function of
s_2b):

```python
# s_2b = 1 and 10, each with max_diis_2b = 1 (no DIIS) and 5
res = run_parquet(mf, 'os', eta_1b=1e8, eta_2b=s_2b, conv_2b=1e-4, max_it_2b=400,
                  max_diis_2b=max_diis_2b)
```

| Ne/aug-cc-pVTZ | IP, eV (Z) | two-body iterations, no DIIS / DIIS 5 | paper |
|---|---|---|---|
| s_2b = 1 | 21.172 (0.933) | 58 / 30 | 21.17 |
| s_2b = 10 | 21.245 (0.935) | 62 / 26 | 21.24 |

The two DIIS settings agree to 3e-4 eV at s_2b = 1 and to 5e-5 eV at
s_2b = 10. At s_2b = 10 the port lies 0.005 eV above the paper's 21.24 (it
would print 21.25). The other molecules and s_2b values have not been run.

**Table III** (TDA, s_1b = s_2b = 50, except evPA's s_1b = 25 for CO and 10
for N2; tau_1b = tau_2b = 1e-4): `run_parquet`'s defaults, but for the DIIS
length (`run_parquet_so`'s damping differs as well). The paper accelerates
the two-body loop by DIIS after damping and does not say how long its
history was. Quack's default, and the drivers', is 1, no DIIS, and at
s_2b = 50 the damped loop then need not converge ([The two
drivers](#the-two-drivers)). So give a length:

```python
# evPA's eta_1b is 25 for CO and 10 for N2
res = run_parquet(mf, 'os', max_diis_2b=5)                    # osPA
res = run_parquet(mf, 'ev', max_diis_1b=5, max_diis_2b=5)     # evPA
res = run_parquet(mf, 'qs', max_diis_1b=5, max_diis_2b=5)     # qsPA
```

Five is a choice, not the paper's. With it, run on the cluster:

| molecule | norb | osPA, eV (Z) | paper | evPA, eV (Z) | paper | qsPA, eV | paper | FCI |
|---|---|---|---|---|---|---|---|---|
| Ne | 46 | 21.2750 (0.936) | 21.28 | 21.0064 (0.917) | 21.01 | 21.2796 | 21.28 | 21.46 |
| HF | 69 | 16.0565 (0.920) | 16.06 | 15.7868 (0.891) | 15.79 | 15.9807 | 15.98 | 16.15 |
| H2O | 92 | 12.6830 (0.919) | 12.68 | 12.5068 (0.893) | 12.51 | 12.5331 | 12.53 | 12.68 |
| NH3 | 115 | 10.8867 (0.928) | 10.89 | 10.7701 (0.910) | 10.77 | 10.6528 | 10.65 | 10.90 |
| CH4 | 138 | 14.1443 (0.936) | 14.14 | 14.0122 (0.921) | 14.01 | 13.8272 | 13.83 | 14.38 |
| BF | 92 | 10.7093 (0.950) | 10.71 | 10.6617 (0.938) | 10.66 | 10.5516 | 10.56 | 11.15 |
| CO | 92 | 14.2307 (0.927) | 14.23 | 14.1411 (0.909) | 14.14 | 13.7601 | 13.76 | 13.93 |
| N2 | 92 | 15.9646 (0.914) | 15.96 | 16.1062 (0.918) | 16.11 | — | 15.44 | 15.49 |

osPA and evPA agree with the paper to its printed digits (Ne's osPA, 21.2750,
at the rounding boundary), and so does qsPA but for BF, 8 meV below it; the
fourth decimal can change with the number of BLAS threads, as any run with
two-body DIIS can ([Options](#options)). Against the paper's FCI references
the mean absolute error is 0.218 eV for osPA and 0.350 eV for evPA, as the
paper's 0.22 and 0.35, and 0.295 eV for qsPA over the seven molecules it ran
on (the paper's values give 0.294 over those seven and 0.26 over all eight).
N2's qsPA did not finish within the 36 h of its cluster job (BF's took 26 h
and CH4's 15 h). Every one-body loop converged, but some two-body loops ended
at the cap of 200 iterations, flagged `'parquet_two_body_not_converged'`:
3 of BF's 18 in evPA, 29 of its 30 in qsPA and 4 of CO's 16 in qsPA. Without
two-body DIIS Ne's and HF's osPA still converge (in 62 and 172 two-body
iterations, against 26 and 47), H2O's does not in 200, and HF's evPA and
qsPA end in NaN; without one-body DIIS evPA (Ne, HF) and qsPA (Ne) do not
converge in 50 one-body iterations. At small size, on water
in cc-pVDZ: osPA converges in 73 two-body iterations; evPA in 11 one-body
iterations with two-body DIIS alone and in 7 with `max_diis_1b=5` as well;
qsPA with both in 5 one-body iterations (11 s on two threads).

## Cost and memory

A two-body iteration costs O(N^6), the pp vertices, the pp effective
integrals and the pp diagonalizations first; the closed-shell two-body loop
holds about seven norb^4 arrays, and two-body DIIS of length m adds 8 m. The
measured times and memory, the comparison with Quack's Fortran and the
projection to the paper's largest case are in the docstring of `restricted`
(COST AND MEMORY); the spin-orbital driver's, whose arrays are 16 times
larger, in that of `spin_orbital`.

## The vertex hook

Each two-body iteration calls `vertex_builder(name, spectra, eta_2b)` once per
reducible vertex, `'eh_singlet'`, `'eh_triplet'`, `'pp_singlet'`,
`'pp_triplet'` in the closed-shell driver and `'eh'`, `'pp'` in the
spin-orbital one, with that iteration's new spectra
(`spectra.RestrictedSpectra` or `SpinOrbitalSpectra`: the eigenvalues and
effective integrals of each channel) and s_2b. The default,
`restricted.static_reducible_vertex`
(`spin_orbital.static_reducible_vertex_so`), is Quack's vertex at omega = 0.
The contract:

- return a real array of the vertex's shape, (norb,)\*4 or (nso,)\*4; it is
  taken as float64 and only read (the damping writes into the loop's own
  arrays), so a builder may return a cached array. Another shape, a complex
  array or anything that is not real numbers (a dict, a ragged list) is
  refused with a ValueError that names `vertex_builder` and the vertex, and
  so is a `vertex_builder` that is not callable;
- NaN or inf in it is kept and flagged `'parquet_vertex_not_finite'`; since
  the change passes over NaN (gfortran's maxval), the loop can report
  convergence with it;
- what it returns is compared, damped and extrapolated as the default's is:
  the change, damping, DIIS and the convergence test stay the driver's. FLEX
  calls it too;
- the options record names it (`result.options['vertex_builder']`: a
  function's name, `'partial(<name>)'`, or the module and qualified name of a
  callable object's type).

A builder whose attribute `per_insertion` is True is called instead once per
vertex and per role, where the vertex is inserted (the eh vertex's two terms
in the pp kernel being one role), as
`vertex_builder(name, role, spectra, eta_2b, eps=eps, nO=nO)`, and the loop
carries one tensor per role ([One tensor per
insertion](#one-tensor-per-insertion)); each tensor is held to the contract
above on its own, and a refusal or a flag names its vertex and role. The
attribute is read before any work (and again by the spin-orbital two-body
loop at each call, which refuses a state made for the other protocol): True
or False (numpy's bool too), anything else is refused with a ValueError; a
`functools.partial` without an attribute of its own takes its function's
(`options.builder_protocol`).

Its limits. A builder of the first protocol gives one tensor per source
channel, inserted at several places: in the closed-shell driver the eh
vertices enter the eh kernels (crossed) and the pp kernels (direct and
swapped), the pp vertices the eh kernels, and the effective integrals
contract the same tensors as their channel's kernel; in the spin-orbital
driver the eh vertex enters both kernels and the pp vertex the eh kernel. So
such a builder cannot weight a vertex differently where it enters different
kernels, and it sees neither nO nor the one-body energies, which change at
every one-body iteration. What it can express is a frequency average
sum_k c_k Phi(omega_k) whose weights depend on neither and are the same at
every insertion: `restricted_vertices.eh_vertex_from_weights` and
`pp_vertex_from_weights` (and those of `spin_orbital_two_body`) contract the
effective integrals with any pole weights, at the cost of one static vertex.
For example, the vertices averaged over three bosonic frequencies, run after
the example of [Running it](#running-it):

```python
import numpy as np

from src.SingleReference.Parquet.restricted_vertices import (
    eh_vertex_from_weights, pp_vertex_from_weights, regularized_inverse)

OMEGAS = (-0.1, 0.0, 0.1)          # Hartree, equal weights


def mean_r(shifts, eta_2b):
    """R(omega + shift), averaged over OMEGAS."""
    return np.mean([regularized_inverse(w + shifts, eta_2b) for w in OMEGAS], axis=0)


def averaged_vertex(name, spectra, eta_2b):
    """The vertex averaged over OMEGAS: one tensor, used at every insertion."""
    if name.startswith('eh'):
        om, rho = getattr(spectra, name)
        return eh_vertex_from_weights(rho, mean_r(-om, eta_2b), -mean_r(om, eta_2b))
    om_ee, rho_ee = getattr(spectra, 'ee_' + name[3:])
    om_hh, rho_hh = getattr(spectra, 'hh_' + name[3:])
    return pp_vertex_from_weights(rho_ee, mean_r(-om_ee, eta_2b),
                                  rho_hh, -mean_r(-om_hh, eta_2b))


res = run_parquet(mf, 'os', eta_1b=1e8, eta_2b=1.0, max_diis_2b=5,
                  vertex_builder=averaged_vertex)
```

On that water/cc-pVDZ RHF it converges in 35 two-body iterations, as the
static vertex does, and moves the principal IP from 11.7736 to 11.7707 eV;
`OMEGAS = (0.0,)` gives the default run's numbers. It shows the contract,
not a validated method. `averaged_vertices` builds the averages over whole
windows, with their weights in closed form, for both drivers
([Frequency-averaged vertices](#frequency-averaged-vertices)).

### One tensor per insertion

The reducible vertices enter the kernels of the next iteration (Eq. (47)),
and the effective integrals of each kernel, in three roles:

| role | vertex | where | channel | its frequency in the exact equations |
|---|---|---|---|---|
| `'phbar_eh'` | eh | crossed into the eh kernel, Phi^eh_pqsr | transverse, ph-bar | nu' - nu |
| `'eh_pp'` | eh | into the pp kernel, direct Phi^eh_pqrs and swapped Phi^eh_pqsr | longitudinal ph and transverse ph-bar | nu' - nu and -(omega + nu + nu') |
| `'pp_eh'` | pp | direct into the eh kernel, Phi^pp_pqrs | pp | omega + nu + nu' |

The static kernel approximation takes them all at 0, as one tensor per
vertex. The eh vertex enters the pp kernel twice, and its two terms there
are one role because they are exchange partners: exchanging the two
particles of a pp pair turns the direct (longitudinal) term into the swapped
(transverse) one, and their transfers nu' - nu and -(omega + nu + nu') into
each other. The exact vertex follows that exchange, since its frequency
moves with it; a static tensor per role, the same for every matrix element,
cannot, and other tensors in the two terms would leave the pp kernel
D_pqrs +- S_pqsr (D the direct and S the swapped tensor) without its
(anti)symmetry under the exchange. Its value would then depend on the basis
of pairs it is written in: the closed-shell kernel (Quack's spin adaptation,
pairs p <= q) and the spin-orbital one (pairs P < Q) part on the singlet
elements between the pairs (p,p) and (r,r), by [(Phi^s + Phi^t)^D -
(Phi^s + Phi^t)^S]_pprr / 4 (in the B, C and D blocks alike), and both part
from the antisymmetrized kernel everywhere else. In a two-tensor variant, not
kept, on water/6-31G, the imaginary exponential at 0.3 Ha on one of the two
terms alone moved the pp eigenvalues of one kernel, built by each driver
from the same iteration, apart by up to 0.12 Ha, and osPA's HOMO (TDA,
s_1b = s_2b = 50, two-body DIIS 5, damping 0.75 in both drivers, the
quasiparticle equation linearized) by +35.5 meV on the direct term or
-38.5 meV on the swapped one, against -6.4 meV for both alike (closed shell;
+27.8, -31.3 and -6.5 meV over spin orbitals): most of a one-sided window's
effect was the broken symmetry. With the swapped term and the transverse
insertion into the eh kernel at the exponential of the iteration's gap
(s = 50, two-body DIIS 5, one-body DIIS 3), the spin-orbital qsPA did not
converge in 15 one-body iterations, where the closed-shell one converged in
5; with one tensor for both terms, as kept, both converge (in 6 and 4). One role for both terms keeps the pp kernel
(anti)symmetric whatever a builder returns. Telling the two apart would take
a transfer frequency that follows the matrix element, on shell (eps_r - eps_p
in the direct term, eps_s - eps_p in the swapped one), which no tensor per
role expresses.

A builder with `per_insertion = True` gives each role its own tensor. The
drivers call it once per vertex and role at every two-body iteration, in the
order of `spectra.RESTRICTED_INSERTIONS` (the eh singlet vertex in its two
roles, the eh triplet one in its two, the pp singlet and the pp triplet one:
six calls) or `spectra.SPIN_ORBITAL_INSERTIONS` (eh in its two, pp: three),
as `vertex_builder(name, role, spectra, eta_2b, eps=eps, nO=nO)`: `eps` the
one-body energies of the iteration's BSE diagonals (the reference's in osPA
and at the first evPA iteration, eOld after it, the current eQP in qsPA),
read-only, and `nO` the occupied (closed shell: doubly occupied) orbitals.
What the loop does with the tensors:

- it carries them in place of the vertices (`spectra.RestrictedInsertions`,
  `SpinOrbitalInsertions`): the eh kernels and their effective integrals
  read the eh vertices in their `'phbar_eh'` role and the pp ones in
  `'pp_eh'`, the pp kernels and their effective integrals the eh vertices
  in `'eh_pp'`; the kernel and integral functions are the default's, handed
  other tensors;
- the change max |Phi_new - Phi_old|, taken before the damping, the damping
  and the convergence test run over all of them, and the history's `'err'`
  lists the change of each in the order above;
- two-body DIIS extrapolates them as one vector, each tensor weighing one
  over the number of its vertex's tensors in the Gram matrix
  (`spectra.insertion_weights`: an eh vertex's two a half each), so that
  each vertex counts once, as in Quack's vector of four (with weight 1 each
  the eh vertices would count twice, and the extrapolation would take
  another path: water/6-31G osPA at s_2b = 1 to 1e-10, 67 two-body
  iterations against the 76 of the default and of the weighted run);
- evPA carries them from one one-body iteration to the next and qsPA resets
  them, as the vertices;
- `result.insertions` holds them after the last two-body iteration and
  `result.vertices` the ones the eh kernels read (each eh vertex in its
  `'phbar_eh'` role, each pp vertex in `'pp_eh'`);
  `result.options['vertex_builder']` names the builder.

Memory: two vertices more, 2 N4 (5.8 GB at norb 138), and two-body DIIS
vectors of 6 N4, 12 max_diis_2b N4 of history; traced on water/cc-pVDZ (the
memory test's osPA), 9.71 N4 against the default's 7.71, and 69.71 N4
against 47.71 with max_diis_2b 5. Over spin orbitals one vertex more, and
6 max_diis_2b nso^4 of history: 8.09 and 38.09 nso^4 against 7.09 and 27.09
on water/6-31G. A builder that builds each role's vertex anew builds every
eh vertex twice per iteration.

The default is not touched: a builder without the attribute, or with
False, the default among them, runs the loop as before, its tensors, memory
and DIIS, to the bit (so the comparisons with Quack hold). A per-insertion
builder that returns the static vertex in every role gives the default run
to the bit without two-body DIIS, and with it to rounding at tight
thresholds (water/6-31G, s_2b = 1, two-body DIIS 5 to 1e-10: closed-shell
osPA and qsPA within 5e-14 Ha with the same iteration counts, evPA, whose
history carries over, to about 1e-11 Ha; the spin-orbital eqp to the bit),
which `tests/test_parquet_insertions.py` holds to 1e-12 Ha (the closed-shell
evPA to 1e-9 Ha; over spin orbitals in osPA). At the default conv_2b a
rounding difference can change an iteration count, and the two runs then
agree to the threshold rather than to rounding (s = 50, conv_2b 1e-4: the
closed-shell evPA by 9.3e-6 Ha and qsPA by 4.2e-5 Ha, with other counts in
their later two-body loops), or not at all where a two-body loop does not
converge. As with any builder other than the default's, the loop flags
`'parquet_vertex_not_finite'` where a per-insertion tensor holds a NaN, so a
run whose vertices go NaN carries more flags than the default's.

## Frequency-averaged vertices

`Parquet/averaged_vertices.py` builds the reducible vertices of the kernels as
averages over a frequency window, through the vertex hook and at the cost of
the static vertex. The default run is not touched: these are builders handed
in.

**Where the static approximation drops frequency.** A reducible vertex enters
the kernels at four places (Eq. (47)): the eh vertex crossed into the eh
kernel and direct and swapped into the pp kernel, the pp vertex direct into
the eh kernel. In the frequency-dependent equations the argument of each
insertion is a combination of the receiving channel's frequencies (an energy
transfer for the eh insertions, a pair energy for the pp one), integrated
over; the static kernel approximation (Eq. (28)) takes each vertex at its own
bosonic frequency omega = 0, so that the pole n of a channel weighs
R(0 -+ Omega_n), with R(x) = (1 - exp(-2 s x^2)) / x of Eq. (52). The
self-energy keeps its frequency dependence and does not read the vertices
([What is computed](#what-is-computed)).

**The SRG weight is already an average.** For an even weight w(y) of unit
area, with the time window g(t) = int w(y) cos(y t) dy, the average of the pole
1/x over the bosonic frequency is

    real axis       R_w(x) = P.V. int w(y) / (x - y) dy = int_0^inf sin(x t) g(t) dt
    imaginary axis  R_w(x) = x int w(nu) / (nu^2 + x^2) dnu = sign(x) L(|x|),
                    L(x) = int_0^inf g(t) exp(-x t) dt

On the real axis the principal value is kept and the absorptive part,
-i pi w(x) per pole, which a real static kernel cannot hold, is dropped; on the
imaginary axis (omega = i nu, a Matsubara average) nothing is dropped, since
the average of a time-ordered pole with an even weight is real. Eq. (52)'s R is
the real-axis average with the positive weight
w_s(y) = 2 F(sqrt(2 s) y) / (pi^1.5 y), F Dawson's function, of unit area, time
window erfc(t / sqrt(8 s)) and half width at half maximum 0.7517 / sqrt(s) Ha
(2.9 eV at s_2b = 50, 20 eV at 1). The static vertex of the paper is therefore
the principal-value vertex averaged over a Dawson window about omega = 0, and
the s_2b scan of its Table II is a scan of that window's width, FLEX
(s_2b = 0) the infinitely wide one. A window here is another weight, not an
average where there was none.

**Yirga and Campbell's central cluster.** Yirga and Campbell
([arXiv:2010.02163](https://arxiv.org/abs/2010.02163), App. C) average the
vertex of their functional renormalization group over patches of imaginary
time; their cluster function at tau_m = 0, f_0(omega) = sinc(omega dtau)
(Eq. (C4)), is a box of imaginary time, of half width T here. Read on the
Matsubara axis, where their sums run, it gives the `'box'` window,
R = (1 - exp(-T |x|)) / x: below the bare pole, monotone, and exponentially
close to 1/x beyond 1/T, the exponential cousin of the SRG's
(1 - exp(-2 s x^2)) / x. Read on the real axis, the same sinc weight gives the
`'sinc'` window, which is no regularizer (below).

**The windows.** Each has one width, its half-weight point x_half (Ha), where
x R(x) = 1/2 (rule H, on either axis), which fixes its parameter:

| axis | `window` | g(t) | frequency weight | R(x) for x > 0 | parameter |
|---|---|---|---|---|---|
| `'imaginary'` | `'box'` | 1 for t <= T | sin(T y) / (pi y) | (1 - exp(-x T)) / x | T = ln 2 / x_half |
| `'imaginary'` | `'exponential'` | exp(-gamma t) | Lorentzian, half width gamma | 1 / (x + gamma) | gamma = x_half |
| `'imaginary'` | `'gaussian'` | exp(-sigma^2 t^2 / 2) | Gaussian, standard deviation sigma | sqrt(pi / 2) / sigma erfcx(x / (sqrt(2) sigma)) | sigma = x_half / (sqrt(2) 0.43275) |
| `'imaginary'` | `'triangle'` | (1 - t / T)_+ | Fejer, (1 - cos T y) / (pi T y^2) | (x T - 1 + exp(-x T)) / (T x^2) | T = 1.59362 / x_half |
| `'real'` | `'gaussian'` | exp(-sigma^2 t^2 / 2) | Gaussian, standard deviation sigma | (sqrt(2) / sigma) F(x / (sqrt(2) sigma)) | sigma = x_half / (sqrt(2) 0.55236) |
| `'real'` | `'lorentzian'` | exp(-eta t) | Lorentzian, half width eta | x / (x^2 + eta^2), the eta form | eta = x_half |
| `'real'` | `'fejer'` | (1 - t / T)_+ | Fejer | (1 - sin(x T) / (x T)) / x | T = 1.89549 / x_half |
| `'real'` | `'sinc'` | 1 for t <= T | sin(T y) / (pi y) | (1 - cos(x T)) / x | T = pi / (3 x_half) |
| `'real'` | `'srg'` | erfc(t / sqrt(8 s)) | w_s | (1 - exp(-2 s x^2)) / x | s = ln 2 / (2 x_half^2) |

The roots without a closed form are the registry's `PARQUET_RULE_H_*`, each
the double nearest the root. By default (x_half None) every window takes the
SRG's half-weight point at the run's s_2b, sqrt(ln 2 / (2 s_2b)) (0.0833 Ha at
50), so that it weighs a pole there as the static vertex does; `'srg'` then
takes s = s_2b itself and is the static vertex to the bit, a check of the
plumbing. s_2b = 0 is an infinitely wide window: every weight 0, as the SRG's.
On the imaginary axis every window gives a weight below the bare pole that
falls monotonically with |x| (past the zero-mode taper, below); at one
half-weight point the exponential
approaches 1/x the slowest (1/x - gamma/x^2: it shifts every pole by gamma),
the box the fastest. On the real axis the Gaussian overshoots the bare pole
(x R up to 1.285, at x = 2.12 sigma), the Lorentzian stays below it and the
Fejer window reaches it in decaying oscillations (x R up to 1.217).

**Why the real-axis sinc is no regularizer.** (1 - cos x T) / x =
2 sin^2(x T / 2) / x vanishes at x T = 2 pi k and swings between 0 and twice
the bare pole forever: x R = 1 - cos(x T) never tends to 1, so a pole far out
weighs anything from nothing to twice its bare weight as Omega T falls modulo
2 pi, and the absorptive part dropped with it is as large as what is kept.
It is offered only so that it can be compared.

**Zero modes.** An imaginary-axis weight jumps at x = 0, to +-L(0) on either
side (the average of an eh pole pair is 0 at Omega = 0 exactly and -2 L(0)
just above), so rounding of an exact zero mode would change the run. Every
imaginary-axis weight is multiplied by the taper 1 - exp(-(x/x0)^2),
x0 = `PARQUET_AVERAGED_TAPER_X0` = 1e-7 Ha, evaluated as Quack evaluates its
regularizer (1 - exp, not expm1): exactly 0 below 7.45e-16 Ha and exactly 1
above 6.12e-7 Ha. H/STO-3G's spin flip, 2.2e-16 or exactly 0 by machine,
then gives the same run to the bit, as it does with the static vertex
([Regularization](#what-is-computed)). The real-axis weights are continuous
and 0 at x = 0 and take no taper: a mode a few ulps off weighs R'(0) times
its offset. Without the eh TDA a mode near zero carries eigenvectors of order
Omega^-1/2, and an imaginary-axis weight, which tends to L(0) and not to 0,
gives it a vertex of order L(0) / Omega, where the SRG's 2 s Omega keeps it
finite: the taper removes what rounding makes of an exact zero, not an open
shell's spin rotation at the SCF residual (1e-7 to 1e-4 Ha, flagged
`'parquet_eh_zero_mode'`), nor a closed shell's soft triplet near its
instability. Run the imaginary windows under the eh TDA wherever an eh mode
can come near zero.

**Where it acts.** At the insertions only: the kernels of the next two-body
iteration and their effective integrals read the averaged tensors, and the
change, the damping, DIIS and the convergence test stay the driver's. The
self-energy does not read the vertices; it changes only through the spectra
(Omega, M) that the averaged kernels give. FLEX builds vertices that nothing
reads and is unaffected.

**Running it.** After the example of [Running it](#running-it):

```python
from src.SingleReference.Parquet.averaged_vertices import averaged_vertex_builder
from src.SingleReference.Parquet.spin_orbital import run_parquet_so

# the box of imaginary time at the static vertex's half-weight point
# (0.0833 Ha at the default s_2b = 50)
res = run_parquet(mf, 'os', max_diis_2b=5, vertex_builder=averaged_vertex_builder('box'))
print(res.options['vertex_builder'])
# averaged_vertex_builder('box', 'imaginary', x_half_eh=None, x_half_pp=None,
#                         pp_center=0.0)

# the exponential at the HOMO-LUMO gap, the pp vertices taken about
# 2 mu = eps_HOMO + eps_LUMO instead of 0
nO = mf.mol.nelectron // 2
gap = mf.mo_energy[nO] - mf.mo_energy[nO - 1]
two_mu = mf.mo_energy[nO] + mf.mo_energy[nO - 1]
res = run_parquet(mf, 'os', max_diis_2b=5, vertex_builder=averaged_vertex_builder(
    'exponential', x_half=gap, pp_center=two_mu))

# spin orbitals: the same builder serves run_parquet_so (whose defaults, Quack's,
# diverge at s_2b = 50; with damping and DIIS water/cc-pVDZ takes about 1.5 min)
res = run_parquet_so(mf, 'os', damping=0.75, max_diis_2b=5,
                     vertex_builder=averaged_vertex_builder('gaussian', 'real'))
```

One builder serves both drivers, which call it with different vertex names;
`driver='restricted'` or `'spin_orbital'` restricts it to one. `x_half_eh` and
`x_half_pp` give the eh and the pp source vertices their own widths. The builder's `__name__` names every parameter, so that
`result.options['vertex_builder']` identifies the run; `widths(eta_2b)` gives
each source vertex's half-weight point and window parameter at an s_2b, and
`averaged_inverse(x, window, axis, x_half=...)` is the weight itself. A wrong
argument is refused with a ValueError that names it.

**Its limits.** `averaged_vertex_builder` gives one window per source vertex,
the same at every insertion and for every matrix element, and sees no
one-body energies: a width or a pp centre tied to the gap or to 2 mu is given
as a number, taken from the reference, and does not follow evPA or qsPA.
These are the limits of the hook's first protocol ([The vertex
hook](#the-vertex-hook)); the per-insertion builders below give each
role a window of its own and see the one-body energies of every
iteration. The width that the frequency integration of an insertion calls
for depends on the element: in a scalar model of one insertion between two
pair propagators it is the imaginary-axis exponential window with gamma the
mean of the two pairs' energies (exact when the two energies are equal, off
by a relative (delta / (Omega + gamma))^2 otherwise, delta half their
difference), of the order of the gap and other for every element, which no
builder here expresses (one window per role at most). These builders are not
a validated method: on six of the paper's aug-cc-pVTZ molecules (osPA, Table
III's settings with two-body DIIS 5: Ne, HF, H2O, BF, CO, N2) none of the 24
windows tried on whole source vertices (the box, exponential, Gaussian and
triangle on the Matsubara axis, the Gaussian, Lorentzian, Fejer and SRG
weights on the real one; the box and the exponential at several widths, the
others at the SRG's half-weight point, the SRG weight with its pp centre at
2 mu) brought the mean absolute error against the reference ionization
potentials below the static vertex's 250 meV (the real-axis Gaussian tied
it, with its BF run unconverged). Widening the window of the eh vertex
lowers every principal IP: with both vertices at the imaginary exponential,
the eh vertex's from the SRG's half-weight point to 0.5 Ha, by 0.08 eV (BF)
to 0.39 eV (N2), which brings CO and N2, above the reference with the static
vertex, toward it and takes Ne, HF, H2O and BF away from it.
`tests/test_parquet_averaged_vertex.py` holds the weights against quadrature,
the builders against a pole sum of their own, `'srg'` against the static
vertex to the bit, and the runs of both drivers.

**Per insertion.** A window on the eh source vertex damps its insertion into
the pp kernel together with its transverse one into the eh kernel. The
transverse particle-hole channel (ph-bar) should enter at another frequency
than the other channels, or its contribution comes out too large (Anna
Kauch, private communication). `insertion_vertex_builder` gives each role
([One tensor per insertion](#one-tensor-per-insertion)) a window of its own,
and `transverse_vertex_builder(window, axis, x_half)` is Kauch's
prescription: the transverse insertion into the eh kernel at the window, the
other roles static. In the pp kernel the transverse term is the exchange
partner of the longitudinal one and shares its tensor, so the prescription
leaves it static with it.

```python
from src.SingleReference.Parquet.averaged_vertices import (insertion_vertex_builder,
                                                           transverse_vertex_builder)

# the transverse insertion at the imaginary exponential of half width the
# iteration's HOMO-LUMO gap, the other roles static
res = run_parquet(mf, 'ev', max_diis_2b=5,
                  vertex_builder=transverse_vertex_builder('exponential', x_half='gap'))
print(res.options['vertex_builder'])
# insertion_vertex_builder(phbar_eh=('exponential', 'imaginary', 'gap'), eh_pp=None,
#     pp_eh=None, pp_center=0.0)

# the transverse insertion at the Matsubara point i gamma, gamma = 0.25 Ha
# (below), and the eh vertex in the pp kernel at the box of the SRG's width
res = run_parquet(mf, 'os', max_diis_2b=5, vertex_builder=insertion_vertex_builder(
    phbar_eh=('lorentzian', 'real', 0.25), eh_pp='box'))
res.insertions.eh_singlet_phbar_eh      # the tensors the loop carried, per role
```

Each role, `phbar_eh`, `eh_pp` and `pp_eh`, takes None, the static vertex
(the default's, to the bit), or a window as `averaged_inverse` takes it,
`(window, axis, x_half)`, axis `'imaginary'` and x_half None (the SRG's
half-weight point at the run's s_2b) where left out, a window name alone
too. x_half `'gap'` is the HOMO-LUMO gap of the one-body energies the driver
hands the builder, min eps[nO:] - max eps[:nO] (`iteration_gap`), taken
anew at every two-body iteration, so it follows evPA and qsPA (a gap that is
not positive is refused). `pp_center` is the point a windowed `pp_eh` is
taken about (with `pp_eh` None it is refused). The builder's `__name__`
names every role's window, `widths(eta_2b, eps, nO)` gives each role's
half-weight point and parameter, and a wrong argument is refused with a
ValueError that names it. The roles windowed alike are
`averaged_vertex_builder` with that window. On water/6-31G at the tests'
geometry (TDA, s_1b = s_2b = 50, two-body DIIS 5, the quasiparticle equation
linearized) the imaginary exponential at the gap moves the principal IP of
osPA from 12.068 eV (static) to 11.585 eV on the transverse role alone,
12.087 eV on `'eh_pp'` alone, 12.456 eV on `'pp_eh'` alone, 11.555 eV on
both eh roles and 11.891 eV on all three: the transverse insertion makes
nearly all of the eh vertex's lowering, and the pp vertex raises the IP. Not
validated either (below).

**Why a Matsubara frequency.** "Another frequency" is an imaginary one. Taken
at the point omega = i gamma (a point, not an average), an eh pole pair
1/(omega - Omega) - 1/(omega + Omega) = 2 Omega / (omega^2 - Omega^2) weighs
-2 Omega / (Omega^2 + gamma^2), smaller in magnitude than its -2 / Omega at
omega = 0: the real-axis
`'lorentzian'` weight x / (x^2 + eta^2) at eta = gamma, so a role taken at
i gamma is the window `('lorentzian', 'real', gamma)` (the vertex's imaginary
part there, gamma times the difference of its resonant and antiresonant
products, a real static kernel cannot hold). A real frequency omega_t
would instead bring the poles closer, 2 Omega / (omega_t^2 - Omega^2), and
enlarge the very insertion it was to soften. The imaginary-axis windows
average over the Matsubara axis rather than take one point of it; the
`'exponential'` shifts every pole by gamma, 1/(x + gamma). At one gamma the
point keeps Omega^2 / (Omega^2 + gamma^2) of a bare pole and the exponential
Omega / (Omega + gamma), both half at Omega = gamma: the point keeps more of
every pole above gamma and less of every pole below it.

**Against the reference IPs.** On the six molecules and with the settings of
the 24 windows above (in both with Quack's own quasiparticle search, as
Tables I and II), no window on the transverse role alone did better than the
static vertex's mean absolute error, 249.6 meV, by more than the 0.2 meV by
which two runs of the per-insertion builder with every role static differ
(249.4 and 249.6 meV). The imaginary exponential at 0.1, 0.2 and 0.5 Ha and
at the RHF HOMO-LUMO gap gave 268 to 370 meV, the box at 0.2 Ha and at the
gap 252 and 324 meV, and the point i gamma, `('lorentzian', 'real', gamma)`,
at 0.1, 0.2, 0.5 and 1 Ha and at the gap 249 to 382 meV, its 249.4 meV at
0.1 Ha a tie with the static vertex; `'eh_pp'` alone at the exponential of
0.2 Ha gave 255 meV. The transverse insertion lowers the principal IPs of Ne,
HF, H2O, CO and N2, the more the wider the window, and moves BF's by less
than 10 meV: it takes CO and N2, above the reference with the static vertex,
toward it (N2 to within 6 meV of it at the gap, with the box or the point
i gamma) and Ne, HF and H2O away from it. The errors' range narrows, from
-441 to +475 meV with the static vertex to -435 to +164 meV at the point
i gamma at 0.5 Ha, only as their mean falls, from +10 to -185 meV. At the
same gamma the point lowered the IPs less than the exponential, by 0.10 to
0.21 of its shift at 0.1 Ha, 0.19 to 0.40 at 0.2 Ha, 0.46 to 0.78 at 0.5 Ha
and 0.77 to 0.90 at the gap (BF aside), and at a similar mean error it did
about as well as the box and the exponential: 315 meV at the gap (mean
-269 meV) against 324 meV for the box at the gap and 327 meV for the
exponential at 0.5 Ha (means -279 and -276 meV). In the runs at the point
i gamma, and in an all-static run alongside them, the principal orbital's
quasiparticle equation converged every time: the `'parquet_qp_not_converged'`
they carry comes from virtual orbitals and from the inner-valence 3sigma of
CO and 2sigma_g of N2 (the earlier runs did not record which orbitals).
`tests/test_parquet_insertions.py` holds the per-insertion loop and builders.
