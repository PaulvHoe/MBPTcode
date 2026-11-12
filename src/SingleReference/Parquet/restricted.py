"""The closed-shell parquet drivers: FLEX, osPA, evPA and qsPA, as Quack runs them.

The parquet approximation of Marie and Loos (J. Chem. Phys. 163, 194115
(2025)) in its static kernel form (Sec. IV A) is solved by two nested loops
(Fig. 8 of the paper). The TWO-BODY loop holds the one-body energies eps and
the integrals fixed and iterates the four closed-shell reducible vertices
Phi^eh_s, Phi^eh_t, Phi^pp_s, Phi^pp_t (the real i*Phi of the paper, stored
Phi[p, q, r, s]): from the vertices of the previous iteration it builds the
static kernels (Eq. (47)) and solves the four BSEs (eh singlet and triplet,
Eqs. (34)-(35); pp singlet and triplet, Eqs. (40)-(41)), contracts their
eigenvectors with the irreducible vertices into the effective integrals
M^eh, M^ee, M^hh (Eqs. (38), (44)), and from the new eigenvalues and those
integrals builds new vertices at zero frequency (Eqs. (39), (45)), regularized
with the two-body flow parameter s_2b (Eq. (52)). The ONE-BODY step then
builds the self-energy (Eqs. (21), (48), (50), (51), regularized with s_1b,
Eq. (53)) from the converged spectra and updates the one-body energies.

Four schemes, as the paper names them:

    FLEX   one two-body iteration (the vertices are zero, the BSEs are RPA
           problems) and one one-body step (Table I);
    osPA   the two-body loop to convergence, then one one-body step: Quack's
           R_evParquet with max_it_1b = 1 (Table II);
    evPA   R_evParquet with one-body self-consistency on the diagonal
           quasiparticle energies (Table III);
    qsPA   Quack's R_qsParquet: a static Hermitian self-energy is added to the
           Fock matrix, which is diagonalized, so orbitals and energies are
           both updated (Table III).

THE TWO-BODY LOOP, exactly as Quack's R_evParquet and R_qsParquet run it:

* the BSEs are solved in the order eh singlet, eh triplet, pp singlet, pp
  triplet, on the energies of the current one-body iteration (eOld in
  R_evParquet: eps_HF at the first iteration, the previous quasiparticle
  energies after; eQP in R_qsParquet);
* at the very first two-body iteration of the whole run the kernels Gamma are
  zero (R_qsParquet's later one-body iterations restart from zero vertices,
  so their first kernels vanish too);
* the effective integrals contract the eigenvectors with the irreducible
  vertices built from the OLD reducible vertices (the ones the BSE kernels
  were built from), and the new vertices are built from those integrals and
  the NEW eigenvalues, at omega = 0;
* the change err = max |Phi_new - Phi_old| over the four vertices is taken
  before damping; the vertices are then damped,
      Phi <- (1 - d) Phi_new + d Phi_old,
  d the `damping` as evGW means it, the weight of the previous iterate
  (0.75 by default: Quack's mixing weight alpha = 1 - d is 0.25 in
  R_evParquet and R_qsParquet), evaluated as Quack evaluates it,
  alpha Phi_new + (1 - alpha) Phi_old; for max_diis_2b > 1 the damped
  vertices are then extrapolated by DIIS over the four vertices together
  (one vector of 4 norb^4 numbers, error Phi_damped - Phi_old): the loop
  damps first and extrapolates the damped iterate, as Quack does;
* the loop runs while err > conv_2b and fewer than max_it_2b iterations ran;
  as in Quack it counts as failed when it ran max_it_2b iterations, even if
  the last one met conv_2b.

R_evParquet keeps the vertices, the previous eigenvalues and the two-body
DIIS history from one one-body iteration to the next, so an evPA two-body
loop starts from the vertices the previous one converged to, and its first
DIIS extrapolation mixes in vectors of the previous one-body iteration.
R_qsParquet resets all three at every one-body iteration (the orbitals
changed).

THE ONE-BODY STEP. FLEX, osPA and evPA (R_evParquet): the diagonal
self-energy SigC_p and Z_p are evaluated at w = eOld_p with every other
denominator on eOld, the linearized energies are eQPlin = eHF + Z SigC
(anchored on eHF at every iteration), and the quasiparticle equation
w = eHF_p + Sigma_pp(w) is solved as `qp_solver` says: by the house's one
root finder on that equation ('find_qp_root', the default, not Quack's:
`restricted_self_energy.DiagonalSelfEnergy.solve_qp_scan`), which keeps the
root of largest weight about eHF_p at the first one-body iteration and, from
the second on, the root continuing the previous iteration's eQP_p, so that
evPA stays on each orbital's branch (an orbital whose previous root had Z
outside (0, 1] is searched again as at the first: `options.roots_to_follow`;
a branch the finder reports the largest weight has left is flagged
'parquet_qp_branch_flipped'); taken linearized ('linearized', eQP = eQPlin);
or by a function handed in (`options.checked_qp_solver`; Quack's own search,
R_Parquet_QP_graph, is not ported: the package docstring);
an energy kept with Z outside (0, 1] is flagged 'z_outside' on every route;
for max_diis_1b > 1 the new energies are DIIS extrapolated with error
eQP - eOld; the one-body change is max |eOld - eQP|.
qsPA (R_qsParquet): the static self-energy matrix (R_Parquet_self_energy_SRG_sp
with flow parameter eta_1b for reg_pa, else R_Parquet_self_energy_eta with
broadening broadening_1b) is symmetrized, taken to the AO basis
(S C Sigma (S C)^T), added to the Fock matrix F = Hc + J[P] + K[P]/2 +
Sigma_AO (Quack's K carries the minus sign), the one-body error is
max |F P S - S P F| (taken before DIIS), F is DIIS extrapolated for
max_diis_1b > 1, diagonalized in the orthogonal basis X (Quack's canonical
orthogonalization) by dsyev (Quack's diagonalize_matrix,
PARQUET_QS_EIGH_DRIVER) to new energies and orbitals, and the density
rebuilt from the lowest nO orbitals; the MO integrals are transformed anew at
the start of every one-body iteration. qsPA solves no quasiparticle equation,
so it takes no `qp_solver`.

THE PAPER'S CLOSED-SHELL NUMBERS are this driver's. Quack's closed-shell and
spin-orbital self-energies part under the one-body regularizer. Both write
each term as a product of regularized denominators, mostly R(Delta) R(w - x),
a frequency-independent one times a frequency-dependent one; where both bare
denominators hold w (the spin-orbital eh 1, eh 4, pp 1 and pp 4 terms),
R_Parquet_self_energy_diag first splits the term into partial fractions, two
terms R(Delta) R(w - x), and G_Parquet_self_energy_diag regularizes it as it
stands, R(w - x1) R(w - x2) (as does the unused R_Parquet_self_energy_diag_dp).
And R_qsParquet's static R_Parquet_self_energy_SRG_sp and G_qsParquet's
G_Parquet_self_energy_SRG_dp are other matrices off the diagonal at any s_1b
(`restricted_static_self_energy`). So on one closed shell Quack's
spin-orbital branch, and the spin-orbital driver that ports it, give other
FLEX, osPA and evPA energies at finite s_1b and another qsPA at any: with
FLEX RPA at s_1b = 50 on the same spectra (to 1e-12), G_evParquet's HOMO lies
2.4 meV from this driver's on water/6-31G and 1.6 meV on ammonia/6-31G, its
2a1 level 23 and 114 meV, its farthest level 0.18 and 0.53 eV, while at
s_1b = 1e8 the two agree to 2e-11 eV. The two-body spectra and EcRPA of one
set of energies do not depend on the self-energy and agree at any s_1b.

WHAT IS QUACK'S AND KEPT, oddities included: the order of operations above;
the failed test n_it == max_it; gfortran's maxval and max, which skip NaN
unless every element is NaN, so a NaN vertex error stops the two-body loop
as a success (here it also raises 'parquet_two_body_not_finite'); FLEX and osPA, whose
single one-body iteration Quack reports as a failed one-body loop (here no
one-body verdict is given for them: `converged_1b` is None); EcGM = 0. What is
not Quack's: the paper-tree R_qsParquet calls the four R_*_Phi routines
without their omega argument (an argument-count mismatch, undefined
behaviour); the static vertices at omega = 0 are built here, as R_evParquet
builds them. R_qsParquet's failed one-body loop ends the program ('stop');
here it warns and raises 'parquet_one_body_not_converged', and the result is
returned. The
plotting blocks of R_evParquet (plot_phi, plot_self) are not ported.
One occupied or one virtual orbital (nO = 1: H2, He; nV = 1): the triplet pp
space has no occupied or no virtual pair, and Quack's
R_pp_triplet_screened_integral hands dgemm a zero leading dimension, which
ends the run with Accelerate or the reference BLAS (OpenBLAS and MKL print
the argument error and skip the empty product; `restricted_vertices`). Both
are accepted here. Quack's spin-orbital G_evParquet runs these closed shells,
and the port reproduces it (H2/6-31G, He/6-31G, water/6-31G cut to six
orbitals; FLEX and three osPA two-body iterations, TDA and full BSEs,
damping 0 as G_evParquet's alpha = 1, linearized): the spectra and EcRPA to
4e-13 Ha, and at s_1b = 1e8, where the two self-energies coincide, eQP to
4e-14 Ha and Z to 6e-13.

CONVENTIONS. Arrays are indexed as Quack's Fortran arrays, 0-based:
eri[p, q, r, s] = <pq|rs> (MO, physicist, not antisymmetrized), eps occupied
first (nO doubly occupied, no frozen core or virtuals: Quack's nC = nR = 0);
eri_ao[mu, nu, la, si] = <mu nu|la si> (Quack's ERI_AO). Energies in
Hartree. The spectra and vertices are those of the `spectra` module.

COST AND MEMORY. A two-body iteration is O(N^6): the pp vertices
(2 norb^4 (nVV + nOO) flops per spin), the pp effective integrals and the pp
diagonalizations (O(nVV^3)) dominate. Measured on an Apple M4 shared with
other work (wall times move with the load), osPA, TDA, s_2b = 1,
s_1b = 1e8, two-body DIIS 5, on water at the paper's geometry: in 6-31G
(norb 13) 29 two-body iterations in 0.1 s and the quasiparticle step by the
house's root finder (the default) in 0.5 s, on two threads; in cc-pVDZ
(norb 24) 35 iterations in 0.9 s on two threads and 1.1 s on one, the root
finder 3.7 s (104 000 evaluations of the self-energy), about 5 s in all; in
aug-cc-pVDZ (norb 41) 31 iterations in 9.5 s, 0.31 s per iteration, on four
threads, the root finder 21 s (211 000 evaluations). qsPA at the paper's
Table III settings with one-body and two-body DIIS 5 on water/cc-pVDZ:
5 one-body iterations, 11 to 13 s on two threads. Measured with Quack's
quasiparticle search in the place of the root finder, whose cost was not
measured at these sizes, on four threads: Ne/aug-cc-pVTZ (norb 46) osPA to
tau = 1e-4 (Table II), 47 to 74 s for 26 to 62 two-body iterations; FLEX in
aug-cc-pVTZ, TDA / RPA, Ne 5 / 4 s, HF (norb 69) 10 / 21 s, H2O (92)
47 / 112 s, N2 (92) 44 / 109 s, NH3 (115) 177 / 370 s; and on N2/aug-cc-pVTZ
(norb 92, nO 7, nVV 3655/3570) one TDA two-body iteration in 40-45 s (pp BSE
blocks, kernels and diagonalizations 14 s, pp vertices 13 s, pp effective
integrals 7.5 s, eh vertices 5 s, eh integrals and BSE 1 s, change and
mixing 1.3 s) and the diagonal self-energy in 4 s; peak RSS 4.9 GB. Against
Quack's Fortran on the same cases, one thread each, the port given Quack's
search as Quack runs it: water/6-31G osPA (125 two-body iterations) 0.4 s
against 1.6 s; water/cc-pVDZ osPA (133) 8.6 s against 88 s; ammonia/6-31G
evPA (20 one-body iterations) 10.7 s against 53 s; water/cc-pVDZ qsPA
(3 one-body iterations) 26.6 s against 256 s.
At the paper's largest problem (CH4/aug-cc-pVTZ: norb 138, nO 5, nS 665,
nVV 8911/8778, nOO 15/10), projected from the flop counts of the components
(not run): about 21 TFLOP per two-body iteration, mostly matrix products (the
pp vertices 2 x 6.4 TFLOP, the pp effective integrals 2 x 3.0 TFLOP, the eh
vertices and integrals about 2 TFLOP), and the pp TDA diagonalizations
O(nVV^3), about 12 times N2's: some 6 to 7 minutes per two-body iteration on
the same four threads and 2 to 3 minutes on a dedicated 16- to 32-core node,
the self-energy about 30 s; with 50 to 200 two-body iterations one osPA takes
about 5 to 23 hours on the four threads, about 2 to 10 hours on such a node.
qsPA's static SRG self-energy adds about 100 s per one-body iteration.
Memory, in norb^4 arrays of doubles (N4: 22.6 MB at norb 41, 0.57 GB at
norb 92, 2.90 GB at norb 138): the floor of the two-body loop is 7 N4,
eri, the four vertices, the vertex being built and the six channels'
effective integrals (which hold norb^4 numbers together). Each pass adds its
own temporaries in blocks of at most 1/PARQUET_MIN_BLOCKS_PER_VERTEX of a
vertex: the vertex change and the damping, the eh and pp vertex products,
and the pp effective integrals, whose irreducible vertex is gathered onto
the pairs a block of orbitals at a time; the eh vertex product also holds
its stacked unweighted integrals (2 norb^2 nS). The previous iteration's
integrals are released before the BSEs of the next are solved. Measured with
tracemalloc on water/aug-cc-pVDZ (norb 41; osPA, TDA, s_2b = 1): without
DIIS the two-body loop peaks at 7.37 N4, in the eh vertex product (its
stacked integrals are 0.21 N4 there), and the run at 7.40 N4, in the
diagonal self-energy's build beside the vertices and spectra (its residues
alone are 0.44 N4 at this size, 0.14 N4 at norb 138). Two-body DIIS keeps
max_diis_2b error and iterate vectors of 4 norb^4 numbers each, as Quack does
(23 GB per history entry at norb 138), in slots allocated once that the loop
writes the damped vertices and their errors into; the extrapolation is
accumulated into the four vertices in place through one scratch block
(`diis`), so DIIS adds 8 max_diis_2b N4 and nothing more: the peak is the
one without DIIS plus 8 max_diis_2b N4 (47.40 N4 at norb 41 and
max_diis_2b 5). At norb 138 the two-body loop peaks at about 7.16 N4
(21 GB) and a run at 7.2 to 7.5 N4 (21 to 22 GB) without two-body DIIS and
47.2 to 47.5 N4 (137 to 138 GB) with max_diis_2b 5, projected from the
array sizes, not run. Quack's R_evParquet holds 17 + 8 max_diis_2b N4 by its own memory
count, which leaves out the integrals it is handed: 18 + 8 max_diis_2b N4
counted as here, 26 N4 without DIIS, since it allocates the DIIS arrays
anyway, and 58 N4 (168 GB at norb 138) with max_diis_2b 5. `run_parquet`
reorders pyscf's integrals from chemist to physicist order in place, a slab
at a time, so it adds no second norb^4 (qsPA: nBas^4) array to them (pyscf's
own ao2mo peaks at 3.4 N4 at norb 41, before the parquet run starts). qsPA
adds the AO integrals (1 N4 when no function is dropped, 2.9 GB at
norb 138) and, while they are transformed, two intermediates of nBas^3 norb
numbers.

FLAGS. The numerical events are warned about and raised as the named flags of
`events` (ParquetNotConverged, ParquetNotFinite, ParquetBSEUnstable; the
house's 'scf_not_converged', 'z_outside' and, from the root finder of
qp_solver='find_qp_root', 'no_root'), which the result lists in
`numerical_flags`. A wrong option is refused before any work (`options`).
"""
import dataclasses
import time

import numpy as np
import scipy.linalg as la
from pyscf.scf import hf as scf_hf

from src.Base.environment_seam import NoEnvironment, environment_of
from src.Base.numerical_flags import collecting, flag, gathered
from src.Base.numerics.constants import (PARQUET_CHANGE_BLOCK_BYTES, PARQUET_CONVERGENCE,
                                         PARQUET_MAX_DIIS_1B, PARQUET_MAX_DIIS_2B,
                                         PARQUET_MAX_IT_1B, PARQUET_MAX_IT_2B,
                                         PARQUET_MIN_BLOCKS_PER_VERTEX,
                                         PARQUET_ORTHOGONALIZATION_THRESHOLD,
                                         PARQUET_QS_EIGH_DRIVER, PARQUET_QUACK_INITIAL_ERROR,
                                         PARQUET_REGULARIZATION, PARQUET_VERTEX_DAMPING)
from src.Base.pyscf.pyscf_interface import (get_two_electron_integrals_chemist,
                                            require_closed_shell_or_unrestricted)
from src.Base.utils.timing import stage
from src.Solvers.qp_equation import QPWeightOutside
from src.SingleReference.Parquet.diis import QuackDIIS
from src.SingleReference.Parquet.events import (FLAG_ONE_BODY_NOT_CONVERGED,
                                                FLAG_ONE_BODY_NOT_FINITE, FLAG_QS_FOCK_NOT_FINITE,
                                                FLAG_SCF_NOT_CONVERGED,
                                                FLAG_TWO_BODY_NOT_CONVERGED,
                                                FLAG_TWO_BODY_NOT_FINITE, FLAG_VERTEX_NOT_FINITE,
                                                FLAG_Z_OUTSIDE, ParquetNotConverged,
                                                ParquetNotFinite)
from src.SingleReference.Parquet.options import (DEFAULT_QP_SOLVER, builder_name,
                                                 checked_occupied, checked_qp_solver,
                                                 checked_real, checked_switch, loop_options,
                                                 qp_solver_label, refuse_frozen,
                                                 roots_to_follow, solve_with)
from src.SingleReference.Parquet.pairs import TRIPLET_MULTIPLICITY
from src.SingleReference.Parquet.restricted_bse import (SPINS, eh_bse_blocks, eh_kernel_blocks,
                                                        pp_bse_blocks, pp_kernel_blocks,
                                                        solve_eh_bse, solve_pp_bse)
from src.SingleReference.Parquet.restricted_self_energy import (DiagonalSelfEnergy,
                                                                linearized_qp_energies)
from src.SingleReference.Parquet.restricted_static_self_energy import (static_self_energy_eta,
                                                                       static_self_energy_srg)
from src.SingleReference.Parquet.restricted_vertices import (VERTEX_NAMES, eh_effective_integrals,
                                                             eh_reducible_vertex,
                                                             pp_effective_integrals,
                                                             pp_reducible_vertex)
from src.SingleReference.Parquet.spectra import Channel, RestrictedSpectra, RestrictedVertices

#: The schemes of the paper: FLEX (one two-body iteration, one one-body
#: step), osPA, evPA (Quack's R_evParquet) and qsPA (R_qsParquet).
SCHEMES = ('flex', 'os', 'ev', 'qs')

#: The six channels of a `spectra.RestrictedSpectra`, in its order.
CHANNEL_NAMES = ('eh_singlet', 'eh_triplet', 'ee_singlet', 'ee_triplet', 'hh_singlet',
                 'hh_triplet')


@dataclasses.dataclass
class ParquetResult:
    """What a parquet run returns (Hartree throughout).

    Attributes
    ----------
    scheme : {'flex', 'os', 'ev', 'qs'}
    eqp : ndarray, shape (norb,)
        Quasiparticle energies of the last one-body iteration (Quack's eQP
        after its one-body DIIS; qsPA: the eigenvalues of the last effective
        Fock matrix, ascending).
    sigc : ndarray, shape (norb,) or (norb, norb)
        FLEX, osPA and evPA: SigC_p at w = eOld_p of the last one-body
        iteration (Quack's printed Sig_c). qsPA: the symmetrized static
        self-energy of the last iteration in the final orbitals (its diagonal
        is Quack's printed Sig_c).
    z : ndarray, shape (norb,)
        Renormalization factors (find_qp_root: 1/(1 - dSigma/dw) at the
        root; linearized: the diagonal Z; a qp_solver function: the z it
        returned; qsPA: ones).
    eps_hf : ndarray, shape (norb,)
    nO : int
        Doubly occupied orbitals (the first nO of eqp; qsPA: of the
        ascending eigenvalues).
    eqp_lin : ndarray or None
        FLEX, osPA and evPA: eHF + Z SigC of the last iteration (Z the
        diagonal one); None for qsPA.
    qp_converged : ndarray of bool or None
        FLEX, osPA and evPA: for 'find_qp_root' whether the orbital's
        equation had a root, for a qp_solver function its verdict per
        orbital (None when linearized, and for qsPA).
    qp_followed, qp_flipped : ndarray of bool or None
        'find_qp_root' (FLEX, osPA and evPA): the house finder's branch
        report of the last one-body iteration, per orbital: the root
        continues the one followed from the previous iteration's eQP
        (qp_followed), and the finder reports that the largest weight has
        left it (qp_flipped: its branch has gone and the root of largest
        weight is kept again, or a root of larger weight lies in the scan
        window of the root followed, every bracket of which the finder
        refines: `options.flag_branch`), which is warned and flagged
        'parquet_qp_branch_flipped'. All False where nothing is followed
        (FLEX, osPA, the first evPA iteration). None for 'linearized', a
        qp_solver function and qsPA.
    spectra : spectra.RestrictedSpectra
        The six channels (Omega, rho) of the last two-body iteration, which
        the last self-energy was built from.
    vertices : spectra.RestrictedVertices
        The reducible vertices after the last two-body iteration (damped and
        DIIS extrapolated: Quack's old_*_Phi at the end of the run).
    ec_eh, ec_pp : float
        The eh and pp correlation energies (EcRPA) of the last two-body
        iteration's BSEs as a spin-orbital problem has them, singlet +
        3 x triplet (Quack's printed Ec(1) + 3 Ec(2); `ec_eh_spin`). Under
        TDA (tda_eh, tda_pp; the default) Quack's EcRPA is zero in exact
        arithmetic, half the sum of the eigenvalues less the trace of the
        same matrix, so the value is rounding noise (below 2e-13 Ha on water
        and ammonia 6-31G, against -0.37 Ha for water's full-BSE FLEX); it
        is a correlation energy for the full BSEs.
    ec_eh_spin, ec_pp_spin : tuple of float
        Quack's EcRPA of the (singlet, triplet) eh and pp BSEs of the last
        two-body iteration, Ec(1) and Ec(2).
    converged : bool
        Every loop that was to converge did (FLEX and osPA have no one-body
        loop to converge, FLEX no two-body loop).
    converged_1b : bool or None
        Quack's verdict on the one-body loop (n_it_1b < max_it_1b), False
        also for a NaN change, which ends Quack's loop as a success; None
        for FLEX and osPA, whose single one-body step is by design.
    converged_2b : list of (bool or None)
        Quack's verdict on each one-body iteration's two-body loop (None for
        FLEX); False also where its change came out NaN.
    n_it_1b : int
    n_it_2b : list of int
    history : list of dict
        One entry per one-body iteration: 'two_body' (one dict per two-body
        iteration: 'err' the four vertex changes, 'err_2b' their max,
        'err_eig' Quack's printed eigenvalue changes (eh singlet, eh triplet,
        pp singlet, pp triplet), 'diis_rcond', 'seconds'), 'converged_2b',
        'n_it_2b', 'om' (the six channels' eigenvalues after the two-body
        loop, which Quack prints on success), 'ec_eh', 'ec_pp' (as the
        fields), 'ec_eh_spin', 'ec_pp_spin', 'eqp', 'sigc' (diagonal), 'z',
        'err_1b', 'diis_rcond', 'seconds', and for FLEX, osPA and evPA only
        (qsPA records have none of them) 'eqp_lin', 'qp_solver' and
        'qp_n_iter' (the solver of the quasiparticle equation, a name or a
        function's label (`options.qp_solver_label`), and, per orbital, its
        evaluations of the self-energy: those of the scan, or of the search
        about the root followed, and of the refinements for 'find_qp_root',
        those a qp_solver function asked for, None for 'linearized'),
        'qp_followed' and 'qp_flipped' (the fields' report of that
        iteration; None but for 'find_qp_root'). 'diis_rcond' is dsysvx's
        reciprocal condition number of that iteration's DIIS system, two-body
        or one-body (NaN for a non-finite history; None without that DIIS):
        Quack solves below rcond = eps without a warning, and so does the
        port, so this is where an ill-conditioned extrapolation shows.
    times : dict
        Wall seconds accumulated per stage ('eh_bse', 'pp_bse',
        'eh_integrals', 'pp_integrals', 'eh_vertex', 'pp_vertex', 'mixing',
        'self_energy', 'qp_equation', 'fock', 'integral_transform', 'total',
        and from `run_parquet` 'integrals', its integral set-up before the
        loops).
    numerical_flags : list of dict
        Every named flag raised during the run.
    options : dict
        The options the run used, defaults resolved ('damping' the weight of
        the previous vertices; 'qp_solver' for FLEX, osPA and evPA; 'reg_pa'
        and 'broadening_1b' for qsPA, whose 'eta_1b' is None under
        reg_pa=False, where it is not read; 'vertex_builder' by name, the
        type's qualified name for a callable object).
    mo_coeff : ndarray or None
        qsPA: the orbitals of the last diagonalization.
    mo_coeff_two_body : ndarray or None
        qsPA: the orbitals the last spectra, vertices and self-energy are
        expressed in (those at the start of the last one-body iteration).
    sigc_ao : ndarray or None
        qsPA: the symmetrized static self-energy of the last iteration in the
        AO basis.
    """
    scheme: str
    eqp: np.ndarray
    sigc: np.ndarray
    z: np.ndarray
    eps_hf: np.ndarray
    nO: int
    eqp_lin: object
    qp_converged: object
    spectra: RestrictedSpectra
    vertices: RestrictedVertices
    ec_eh: float
    ec_pp: float
    ec_eh_spin: tuple
    ec_pp_spin: tuple
    converged: bool
    converged_1b: object
    converged_2b: list
    n_it_1b: int
    n_it_2b: list
    history: list
    times: dict
    numerical_flags: list
    options: dict
    mo_coeff: object = None
    mo_coeff_two_body: object = None
    sigc_ao: object = None
    qp_followed: object = None
    qp_flipped: object = None


# ---------------------------------------------------------------------------
# the two-body iteration


def two_body_spectra(eps, eri, nO, vertices, *, first, tda_eh, tda_pp, timer=None):
    """The four BSEs of one two-body iteration and their effective integrals.

    Steps 3 to 5 of the paper's Fig. 8, as R_evParquet and R_qsParquet run
    them: for the eh singlet, eh triplet, pp singlet and pp triplet channel
    in turn, the bare BSE blocks on `eps` (phRLR_A/B, ppRLR_B/C/D) plus the
    static kernel of `vertices` (R_eh_*_Gamma_A/B, R_pp_*_Gamma_B/C/D;
    Eqs. (35), (41), (47)), zero when `first`, the BSE solved (phRLR, ppRLR;
    Eqs. (34), (40)), and its eigenvectors contracted with the irreducible
    vertex of the same `vertices` (R_*_screened_integral; Eqs. (38), (44)).
    Each channel's eigenvectors are dropped once its integrals are built.

    Parameters
    ----------
    eps : ndarray, shape (norb,)
        One-body energies of the BSE diagonals (eOld, or eQP in qsPA).
    eri : ndarray, shape (norb,)*4
        <pq|rs>.
    nO : int
        Doubly occupied orbitals.
    vertices : spectra.RestrictedVertices
        The reducible vertices of the previous two-body iteration.
    first : bool
        Quack's first two-body iteration of the run: the kernels are zero.
    tda_eh, tda_pp : bool
        Tamm-Dancoff approximation of the eh and the pp BSEs.
    timer : dict, optional
        Accumulates wall seconds under 'eh_bse', 'eh_integrals', 'pp_bse',
        'pp_integrals'.

    Returns
    -------
    spectra : spectra.RestrictedSpectra
        The six channels (Omega, rho[p, q, n]).
    ec_eh, ec_pp : tuple of float
        Quack's EcRPA of the (singlet, triplet) eh and pp BSEs.
    """
    channels, ec_eh, ec_pp = {}, [], []
    for spin in SPINS:
        with stage(timer, 'eh_bse'):
            A, B = eh_bse_blocks(eps, eri, nO, spin, tda_eh)
            if not first:
                gam_a, gam_b = eh_kernel_blocks(vertices, nO, spin, tda_eh)
                A += gam_a
                if B is not None:
                    B += gam_b
                del gam_a, gam_b
            om, xpy, xmy, ec = solve_eh_bse(A, B, tda_eh)
            del A, B
        with stage(timer, 'eh_integrals'):
            rho = eh_effective_integrals(eri, vertices, xpy, xmy, nO, spin)
        del xpy, xmy
        channels['eh_' + spin] = Channel(om, rho)
        ec_eh.append(float(ec))
    for spin in SPINS:
        with stage(timer, 'pp_bse'):
            B, C, D = pp_bse_blocks(eps, eri, nO, spin, tda_pp)
            if not first:
                gam_b, gam_c, gam_d = pp_kernel_blocks(vertices, nO, spin, tda_pp)
                C += gam_c
                D += gam_d
                if B is not None:
                    B += gam_b
                del gam_b, gam_c, gam_d
            om_ee, x1, y1, om_hh, x2, y2, ec = solve_pp_bse(B, C, D, tda_pp)
            del B, C, D
        with stage(timer, 'pp_integrals'):
            rho_ee, rho_hh = pp_effective_integrals(eri, vertices, x1, y1, x2, y2, nO, spin)
        del x1, y1, x2, y2
        channels['ee_' + spin] = Channel(om_ee, rho_ee)
        channels['hh_' + spin] = Channel(om_hh, rho_hh)
        ec_pp.append(float(ec))
    return RestrictedSpectra(**channels), tuple(ec_eh), tuple(ec_pp)


def static_reducible_vertex(name, spectra, eta_2b):
    """The reducible vertex `name` at zero frequency from the BSE spectra.

    Quack's R_eh_singlet_Phi, R_eh_triplet_Phi, R_pp_singlet_Phi and
    R_pp_triplet_Phi called with omega = 0d0, as R_evParquet calls them:
    Eqs. (39) and (45), regularized as Eq. (52). This is the default
    `vertex_builder` of the drivers. Another builder of the same signature
    returns a real (norb,)*4 array, taken as float64, which the loop only
    reads (it may be cached: the damping writes into the loop's own arrays);
    any other shape is refused (ValueError), and so are a complex array,
    whose imaginary part a cast would drop (the closed-shell reducible vertex
    is real), and anything numpy cannot read as real numbers; NaN or inf in
    it is kept and flagged ('parquet_vertex_not_finite'). It can express a
    frequency average
    (`restricted_vertices.eh_vertex_from_weights` and `pp_vertex_from_weights`
    take the averaged weights) only where the average does not depend on the
    orbitals or the quasiparticle energies and is the same at every place
    the vertex is inserted: one tensor per source channel feeds both the eh
    and the pp kernels, and the builder sees neither eps nor nO. Weighting
    each insertion on its own needs a redesign (one tensor per insertion
    place, eps and nO handed to the builder), which docs/methods/parquet.md
    describes; it is not implemented.

    Parameters
    ----------
    name : {'eh_singlet', 'eh_triplet', 'pp_singlet', 'pp_triplet'}
    spectra : spectra.RestrictedSpectra
        The new eigenvalues and the effective integrals of this iteration.
    eta_2b : float
        The two-body flow parameter s_2b (Hartree^-2).

    Returns
    -------
    phi : ndarray, shape (norb,)*4
    """
    if name in ('eh_singlet', 'eh_triplet'):
        om, rho = getattr(spectra, name)
        return eh_reducible_vertex(om, rho, eta_2b, 0.0)
    if name in ('pp_singlet', 'pp_triplet'):
        spin = name[3:]
        om_ee, rho_ee = getattr(spectra, 'ee_' + spin)
        om_hh, rho_hh = getattr(spectra, 'hh_' + spin)
        return pp_reducible_vertex(om_ee, rho_ee, om_hh, rho_hh, eta_2b, 0.0)
    raise ValueError(f'unknown reducible vertex {name!r}; one of {VERTEX_NAMES}')


class _TwoBodyState:
    """What Quack's two-body loop carries between its iterations: the four
    reducible vertices (old_*_Phi), the previous eigenvalues (old_*_Om, only
    printed) and the two-body DIIS history (None when max_diis_2b is 1, which
    Quack skips; its slots are allocated once, here)."""

    def __init__(self, norb, max_diis_2b):
        self.norb = norb
        self.max_diis_2b = max_diis_2b
        self.diis = (QuackDIIS(max_diis_2b, [(norb,) * 4] * len(VERTEX_NAMES),
                               label='two-body') if max_diis_2b > 1 else None)
        self.reset()
        self.started = False

    def reset(self):
        """Zero vertices and eigenvalues and an empty DIIS history (Quack's
        initialization, and R_qsParquet's at every one-body iteration)."""
        self.vertices = None
        self.vertices = [np.zeros((self.norb,) * 4) for _ in VERTEX_NAMES]
        self.om = None
        if self.diis is not None:
            self.diis.reset()

    def as_vertices(self):
        return RestrictedVertices(*self.vertices)


def _quack_maxval(a):
    """gfortran's maxval(a): NaN elements are skipped unless every element is
    NaN; -huge for an empty array."""
    a = np.asarray(a, dtype=float).ravel()
    if a.size == 0:
        return -np.finfo(float).max
    return float(np.fmax.reduce(a))


def _quack_max(values):
    """gfortran's max(a, b, ...) at run time (arm64 fmaxnm): NaN arguments are
    skipped unless every one is NaN."""
    return float(np.fmax.reduce(np.asarray(values, dtype=float)))


def _block_step(a):
    """Values of the first index of the (norb,)*4 array `a` per block of a
    pass over it: at most PARQUET_CHANGE_BLOCK_BYTES and
    1/PARQUET_MIN_BLOCKS_PER_VERTEX of the array."""
    block_bytes = min(PARQUET_CHANGE_BLOCK_BYTES, a.nbytes // PARQUET_MIN_BLOCKS_PER_VERTEX)
    return max(1, block_bytes // max(1, a[0].nbytes))


def _max_abs_change(old, new):
    """maxval(abs(old - new)) of two (norb,)*4 arrays, block by block over the
    first index (`_block_step`), with gfortran's NaN handling."""
    step = _block_step(old)
    result = np.nan
    for p0 in range(0, old.shape[0], step):
        d = np.subtract(old[p0:p0 + step], new[p0:p0 + step])
        np.abs(d, out=d)
        if d.size:
            result = np.fmax(result, np.fmax.reduce(d.ravel()))
        del d                   # before the next block is made: one block at a time
    return float(result)


def _built_vertex(phi, name, norb):
    """What a vertex builder returned, as a C-contiguous float64 array (no copy
    when it is one already) that the loop only reads; ValueError unless it is
    a real (norb,)*4 array (a complex one is refused, not cast, whatever its
    imaginary part; so is anything numpy cannot read as real numbers, a dict,
    a ragged list or another object)."""
    try:
        is_complex = np.iscomplexobj(phi)       # numpy reads a ragged list here
        if not is_complex:
            phi = np.ascontiguousarray(phi, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError(f'vertex_builder returned a {type(phi).__name__} for {name!r}, not '
                         f'an array of real numbers ({err}); a closed-shell reducible vertex '
                         f'is a real {(norb,) * 4} array') from None
    if is_complex:
        raise ValueError(f'vertex_builder returned a complex array for {name!r}; the '
                         f'closed-shell reducible vertex is real (return its real part if '
                         f'that is the vertex meant)')
    if phi.shape != (norb,) * 4:
        raise ValueError(f'vertex_builder returned an array of shape {phi.shape} for '
                         f'{name!r}; a closed-shell reducible vertex is {(norb,) * 4}')
    return phi


def _nonfinite_count(a):
    """The number of NaN and inf elements of the (norb,)*4 array `a`, block by
    block over the first index (`_block_step`): no temporary beyond one
    block's bools."""
    step = _block_step(a)
    return sum(int(a[p0:p0 + step].size - np.count_nonzero(np.isfinite(a[p0:p0 + step])))
               for p0 in range(0, a.shape[0], step))


def _damp_in_place(old, new, alpha):
    """old <- alpha new + (1 - alpha) old as Quack evaluates it,
    fl(fl(alpha new) + fl((1 - alpha) old)), block by block over the first
    index (`_block_step`, one scratch block); `new` is only read."""
    beta = 1.0 - alpha
    step = _block_step(old)
    scratch = np.empty((min(step, old.shape[0]),) + old.shape[1:])
    for p0 in range(0, old.shape[0], step):
        part = old[p0:p0 + step]
        t = scratch[:part.shape[0]]
        np.multiply(new[p0:p0 + step], alpha, out=t)
        part *= beta
        part += t


def _damp_into(damped, err, new, old, alpha):
    """damped <- alpha new + (1 - alpha) old as Quack evaluates it, and
    err <- damped - old: the next DIIS pair, written into its slot without
    another temporary; `new` and `old` are only read."""
    np.multiply(new, alpha, out=damped)
    np.multiply(old, 1.0 - alpha, out=err)
    damped += err
    np.subtract(damped, old, out=err)


def _eigenvalue_changes(previous, spectra):
    """Quack's printed eigenvalue changes max |old_Om - Om| of the eh singlet,
    eh triplet, pp singlet (max of ee and hh) and pp triplet channels; the
    previous eigenvalues are zero at the start (Quack's old_*_Om = 0)."""
    def change(name):
        om = getattr(spectra, name).om
        old = np.zeros_like(om) if previous is None else previous[name]
        return _quack_maxval(np.abs(old - om))
    return (change('eh_singlet'), change('eh_triplet'),
            _quack_max((change('ee_singlet'), change('hh_singlet'))),
            _quack_max((change('ee_triplet'), change('hh_triplet'))))


def _update_vertices(state, spectra, eta_2b, alpha, vertex_builder, timer):
    """The new vertices of one two-body iteration, damped and extrapolated
    into `state.vertices`; returns the four changes max |Phi_old - Phi_new|
    taken before damping (Quack's err_eh_sing, ..., err_pp_trip).

    Each vertex is built (`vertex_builder`; what it returns is only read,
    refused unless it is a real (norb,)*4 array, and, from a builder other
    than the default, flagged 'parquet_vertex_not_finite' where it holds NaN
    or inf), compared with the old one and damped, one at a time, as Quack
    evaluates Phi = alpha Phi + (1 - alpha) old_Phi (alpha = 1 - damping).
    Without DIIS the damped vertex is written over the old one in place, a
    block at a time. With DIIS each damped vertex and its
    error (damped - old) are written straight into the next slot of the
    two-body history, whose four blocks are Quack's vector in its order (eh
    singlet, eh triplet, pp singlet, pp triplet), and the extrapolation is
    accumulated into the state's four vertices in place. The slot is taken
    once the first vertex has passed its check (a later vertex refused costs
    a full history its oldest pair; the run ends there anyway)."""
    errors = []
    slot = None
    for k, name in enumerate(VERTEX_NAMES):
        timer_key = 'eh_vertex' if name.startswith('eh') else 'pp_vertex'
        with stage(timer, timer_key):
            new = _built_vertex(vertex_builder(name, spectra, eta_2b), name, state.norb)
            if vertex_builder is not static_reducible_vertex:
                bad = _nonfinite_count(new)
                if bad:
                    flag(FLAG_VERTEX_NOT_FINITE,
                         f'vertex_builder {builder_name(vertex_builder)} returned {bad} NaN or '
                         f'inf element(s) of {new.size} in the {name!r} vertex; kept as returned '
                         f'(the change max |Phi_new - Phi_old| passes over NaN, as gfortran\'s '
                         f'maxval, so the loop can report convergence with them)',
                         ParquetNotFinite, stacklevel=4)
        with stage(timer, 'mixing'):
            old = state.vertices[k]
            errors.append(_max_abs_change(old, new))
            if state.diis is None:
                _damp_in_place(old, new, alpha)
            else:
                if slot is None:
                    slot = state.diis.next_slot()
                slot_errors, slot_iterates = slot
                _damp_into(slot_iterates[k], slot_errors[k], new, old, alpha)
            del new, old
    if state.diis is not None:
        with stage(timer, 'mixing'):
            state.diis.extrapolate(out=state.vertices)
    return tuple(errors)


def _two_body_loop(state, eps, eri, nO, opts, timer, check):
    """Quack's two-body do-while loop on fixed one-body energies `eps`.

    Returns (spectra, ec_eh, ec_pp, record): the spectra and EcRPA of the
    last iteration and a dict with the iteration records, the count and
    Quack's verdict (None when `check` is False, i.e. FLEX)."""
    n_it, err_2b = 0, PARQUET_QUACK_INITIAL_ERROR
    iterations = []
    spectra = ec_eh = ec_pp = None
    while err_2b > opts['conv_2b'] and n_it < opts['max_it_2b']:
        n_it += 1
        t0 = time.perf_counter()
        spectra = None      # the previous iteration's integrals (1 N4), read: let them go
        spectra, ec_eh, ec_pp = two_body_spectra(
            eps, eri, nO, state.as_vertices(), first=not state.started,
            tda_eh=opts['tda_eh'], tda_pp=opts['tda_pp'], timer=timer)
        state.started = True
        err_eig = _eigenvalue_changes(state.om, spectra)
        state.om = {name: getattr(spectra, name).om for name in CHANNEL_NAMES}
        errors = _update_vertices(state, spectra, opts['eta_2b'], opts['alpha'],
                                  opts['vertex_builder'], timer)
        err_2b = _quack_max(errors)
        iterations.append({'err': errors, 'err_2b': err_2b, 'err_eig': err_eig,
                           'diis_rcond': None if state.diis is None else state.diis.rcond,
                           'seconds': time.perf_counter() - t0})
    converged = None
    if check:
        converged = n_it != opts['max_it_2b']
        if np.isnan(err_2b):
            converged = False
            flag(FLAG_TWO_BODY_NOT_FINITE,
                 f'parquet two-body loop: every reducible vertex change is NaN at '
                 f'iteration {n_it}; Quack ends the loop there and reports success, '
                 f'the spectra and vertices returned are NaN',
                 ParquetNotFinite, stacklevel=3)
        elif not converged:
            flag(FLAG_TWO_BODY_NOT_CONVERGED,
                 f'parquet two-body loop not converged after {n_it} iterations: max '
                 f'|Phi_new - Phi_old| = {err_2b:.3e} against conv_2b = '
                 f'{opts["conv_2b"]:.1e} (Quack counts n_it == max_it_2b as failed); '
                 f'the run goes on with the last vertices',
                 ParquetNotConverged, stacklevel=3)
    record = {'two_body': iterations, 'n_it_2b': n_it, 'converged_2b': converged,
              'om': dict(state.om)}
    return spectra, ec_eh, ec_pp, record


# ---------------------------------------------------------------------------
# osPA, evPA and FLEX: R_evParquet


def ev_parquet(eps_hf, eri, nO, scheme='os', *, tda_eh=True, tda_pp=True,
               eta_1b=PARQUET_REGULARIZATION, eta_2b=PARQUET_REGULARIZATION, max_it_1b=None,
               conv_1b=PARQUET_CONVERGENCE, max_it_2b=None, conv_2b=PARQUET_CONVERGENCE,
               max_diis_1b=PARQUET_MAX_DIIS_1B, max_diis_2b=PARQUET_MAX_DIIS_2B,
               qp_solver=DEFAULT_QP_SOLVER, damping=PARQUET_VERTEX_DAMPING,
               vertex_builder=None):
    """FLEX, osPA or evPA on given orbital energies and MO integrals: Quack's
    R_evParquet.

    Runs the one-body loop of R_evParquet (module docstring): at every
    one-body iteration the two-body loop to convergence on the energies eOld
    (eps_hf at the first), then the diagonal self-energy and Z at w = eOld
    (R_Parquet_self_energy_diag), eQPlin = eps_hf + Z SigC, and eQP from the
    quasiparticle equation as `qp_solver` says (the house's root finder by
    default), then the one-body DIIS (max_diis_1b > 1, error eQP - eOld),
    the change max |eOld - eQP|, and eOld = eQP.

    Parameters
    ----------
    eps_hf : ndarray, shape (norb,)
        HF orbital energies, occupied first.
    eri : ndarray, shape (norb,)*4
        <pq|rs> in the HF orbitals (Quack's ERI_MO).
    nO : int
        Doubly occupied orbitals, a whole number between 1 and norb - 1 (a
        bool or a string is refused).
    scheme : {'flex', 'os', 'ev'}
        'flex': max_it_1b = max_it_2b = 1; 'os': max_it_1b = 1; 'ev': one-body
        self-consistency.
    tda_eh, tda_pp : bool
        Tamm-Dancoff approximation of the eh and pp BSEs (Quack's TDAeh,
        TDApp): True or False (numpy's bool too); anything else is refused.
    eta_1b : float
        One-body SRG flow parameter s_1b (Hartree^-2) of the self-energy,
        Eq. (53); large is weakly regularized (1e8 is effectively none).
    eta_2b : float
        Two-body flow parameter s_2b of the reducible vertices, Eq. (52).
    max_it_1b : int, optional
        One-body iteration cap; default 1 for 'flex' and 'os' (the only value
        they take) and PARQUET_MAX_IT_1B for 'ev'.
    conv_1b : float
        One-body threshold on max |eOld - eQP| (Hartree), below
        PARQUET_QUACK_INITIAL_ERROR.
    max_it_2b : int, optional
        Two-body iteration cap; default 1 for 'flex' (the only value it
        takes) and PARQUET_MAX_IT_2B otherwise.
    conv_2b : float
        Two-body threshold on max |Phi_new - Phi_old|, below
        PARQUET_QUACK_INITIAL_ERROR.
    max_diis_1b, max_diis_2b : int
        DIIS lengths (1: none, as Quack's default).
    qp_solver : {'find_qp_root', 'linearized'} or callable
        How eQP comes from the quasiparticle equation w = eHF_p + Sigma_pp(w)
        (`options.QP_SOLVERS`, or a function; None is taken only by
        `run_parquet`): 'find_qp_root', the default
        (`options.DEFAULT_QP_SOLVER`), the house's one root finder on that
        equation (`DiagonalSelfEnergy.solve_qp_scan`; not Quack's): at the
        first one-body iteration the root of largest valid weight in its
        scan windows about eHF_p, and from the second on the root
        continuing the orbital's previous eQP (eOld, after any one-body
        DIIS), whatever its weight, so that
        evPA does not jump between roots of similar weight (ammonia/6-31G's
        N 1s at s_1b = 50 and s_2b = 1 cycled between three roots when
        re-decided by weight; followed, it converges); an orbital whose
        previous root has Z outside (0, 1] is not followed but searched as
        at the first iteration (`options.roots_to_follow`), and a branch the
        finder reports the largest weight has left is warned and flagged
        ('parquet_qp_branch_flipped'). It takes 150 to
        14 000 evaluations of the self-energy per orbital at the first
        iteration (water/aug-cc-pVDZ, norb 41: 7.7 s per one-body step at
        s_1b = 50 and 33 s at 1e8) and, following, about as many, since it
        searches the whole scan window of the root followed for the branch
        report (150 to 3 200 per orbital at s_1b = 50 and s_2b = 1 on water
        and ammonia 6-31G). 'linearized', eQP = eQPlin. A
        function, called at every one-body iteration as
        qp_solver(sigma, eps_hf, eqp_lin, z_lin, orbitals) -> (eqp, z,
        converged), the same contract in both drivers
        (`options.checked_qp_solver`): sigma(p, w) gives (Sigma_pp(w),
        dSigma_pp/dw) of orbital p at w, the other denominators on eOld,
        each call counted in 'qp_n_iter'; eps_hf, eqp_lin and z_lin are
        the bare levels, eQPlin and the diagonal Z (norb,); orbitals are
        the orbitals to solve, here all; it returns three (norb,) arrays,
        eqp and z real and converged bool, which are checked, kept as
        returned and recorded with the function's label
        (`options.qp_solver_label`); a finite energy not converged is
        flagged 'parquet_qp_not_converged', one not finite
        'parquet_qp_not_finite'. Quack's own search, R_Parquet_QP_graph, is
        not ported (the package docstring). Every route flags an
        energy kept with Z outside (0, 1] ('z_outside'): the finder's root,
        the linearized energy, a function's energy.
    damping : float or None
        The weight of the previous vertices, as in evGW,
        Phi <- (1 - damping) Phi_new + damping Phi_old, a real number in
        [0, 1) (1 would freeze them; a bool, a string or an array is
        refused); None is the default PARQUET_VERTEX_DAMPING = 0.75,
        Quack's alpha = 1 - damping = 0.25 of R_evParquet. The loop damps
        first and, for max_diis_2b > 1, extrapolates the damped vertices by
        DIIS, with error Phi_damped - Phi_old, as Quack does.
    vertex_builder : callable, optional
        (name, spectra, eta_2b) -> Phi of one channel, a real (norb,)*4
        array, taken as float64, that the loop only reads (any other shape or
        type, a complex array included, is refused; NaN or inf in it is
        flagged 'parquet_vertex_not_finite'); default
        `static_reducible_vertex` (omega = 0, Quack's), whose docstring says
        what another builder can express.

    Returns
    -------
    ParquetResult
    """
    if scheme not in ('flex', 'os', 'ev'):
        raise ValueError(f'ev_parquet runs FLEX, osPA and evPA (scheme flex, os, ev), '
                         f'not {scheme!r}; qsPA is qs_parquet')
    eps_hf = np.array(eps_hf, dtype=float)          # the result's own, never the caller's
    norb, nO = _check_integrals(eps_hf, eri, nO)
    opts = _loop_options(scheme, tda_eh, tda_pp, eta_1b, eta_2b, max_it_1b, conv_1b,
                         max_it_2b, conv_2b, max_diis_1b, max_diis_2b, damping, vertex_builder)
    opts['qp_solver'] = checked_qp_solver(qp_solver)
    flags, timer = [], {}
    t_start = time.perf_counter()
    with collecting(flags):
        result = _ev_loops(eps_hf, np.asarray(eri, dtype=float), nO, norb, scheme, opts,
                           timer)
    timer['total'] = time.perf_counter() - t_start
    result.numerical_flags = gathered(flags)
    result.times = timer
    return result


def _ev_loops(eps_hf, eri, nO, norb, scheme, opts, timer):
    state = _TwoBodyState(norb, opts['max_diis_2b'])
    diis_1b = (QuackDIIS(opts['max_diis_1b'], [(norb,)], label='evPA one-body')
               if opts['max_diis_1b'] > 1 else None)
    e_old = eps_hf.copy()
    n_it_1b, err_1b = 0, PARQUET_QUACK_INITIAL_ERROR
    history = []
    while err_1b > opts['conv_1b'] and n_it_1b < opts['max_it_1b']:
        n_it_1b += 1
        t0 = time.perf_counter()
        spectra = None          # the previous iteration's integrals, before new ones exist
        spectra, ec_eh, ec_pp, record = _two_body_loop(state, e_old, eri, nO, opts, timer,
                                                       check=scheme != 'flex')
        with stage(timer, 'self_energy'):
            se = DiagonalSelfEnergy(opts['eta_1b'], e_old, eri, nO, spectra)
            sigc, z = se.diagonal()
        eqp_lin = linearized_qp_energies(eps_hf, sigc, z)
        with stage(timer, 'qp_equation'):
            # find_qp_root follows the previous iteration's roots (eOld) from the second on,
            # each where its Z lay in (0, 1]
            follow = None
            if _is_named(opts['qp_solver'], 'find_qp_root') and n_it_1b > 1:
                follow = roots_to_follow(e_old, history[-1]['z'])
            eqp, z, qp_converged, qp_n_iter, followed, flipped = _solve_qp(
                se, opts['qp_solver'], eps_hf, eqp_lin, z, follow)
        del se
        if diis_1b is not None:
            eqp = diis_1b.push(eqp - e_old, eqp)
        err_1b = _quack_maxval(np.abs(e_old - eqp))
        e_old = np.array(eqp, dtype=float)
        record.update(_energy_record(ec_eh, ec_pp), eqp=e_old.copy(), sigc=sigc.copy(),
                      z=np.array(z, dtype=float), eqp_lin=eqp_lin,
                      qp_solver=qp_solver_label(opts['qp_solver']), qp_n_iter=qp_n_iter,
                      qp_followed=followed, qp_flipped=flipped, err_1b=err_1b,
                      diis_rcond=None if diis_1b is None else diis_1b.rcond,
                      seconds=time.perf_counter() - t0)
        history.append(record)
    converged_1b = None
    if scheme == 'ev':
        converged_1b = _one_body_verdict('evPA', n_it_1b, err_1b, opts, 'max |eQP - eOld|',
                                         'the energies returned are the last iterate')
    return _result(scheme, history, e_old, sigc, np.array(z, dtype=float), eps_hf, nO, eqp_lin,
                   qp_converged, spectra, state, ec_eh, ec_pp, converged_1b, n_it_1b, opts)


def _is_named(solver, name):
    """Whether `solver` (a qp_solver) is the route named `name`, never
    asking a function to compare itself with a string."""
    return isinstance(solver, str) and solver == name


def _solve_qp(se, solver, eps_hf, eqp_lin, z, follow=None):
    """(eqp, z, qp_converged, qp_n_iter, qp_followed, qp_flipped) of one FLEX,
    osPA or evPA one-body step: the quasiparticle equation of the diagonal
    self-energy `se` solved by `solver` (a name of `options.QP_SOLVERS` or a
    qp_solver function, `options.checked_qp_solver`), 'find_qp_root'
    following the roots `follow` (`options.roots_to_follow` of the previous
    iteration's eQP) when given; qp_n_iter counts the self-energy
    evaluations per orbital (None when linearized); qp_followed and
    qp_flipped are the finder's branch report per orbital (None for
    'linearized' and a function, which follow nothing). Every route flags an
    energy kept with Z outside (0, 1] ('z_outside'): the linearized one
    here, a function's in `options.solve_with` (with its verdict), the root
    finder's in its own solver, where a flipped branch is flagged too
    ('parquet_qp_branch_flipped')."""
    if _is_named(solver, 'linearized'):
        _flag_linearized_outside(eqp_lin, z)
        return eqp_lin.copy(), z, None, None, None, None
    if _is_named(solver, 'find_qp_root'):
        out = {}
        eqp, z, found = se.solve_qp_scan(eps_hf, follow=follow, out=out)
        return eqp, z, found, out['n_eval'], out['followed'], out['flipped']
    eqp, z, converged, n_eval = solve_with(solver, se.at, eps_hf, eqp_lin, z,
                                           np.arange(len(eps_hf)), 'orbital')
    return eqp, z, converged, n_eval, None, None


def _flag_linearized_outside(eqp, z):
    """Warn (src.Solvers.qp_equation.QPWeightOutside) and raise the named flag
    'z_outside' for each orbital whose linearized energy, the solution
    qp_solver='linearized' keeps, is finite and has its diagonal Z outside
    (0, 1]; the energies stay as computed."""
    eqp, z = np.asarray(eqp, dtype=float), np.asarray(z, dtype=float)
    with np.errstate(invalid='ignore'):
        outside = np.isfinite(eqp) & ~((z > 0.0) & (z <= 1.0))
    for p in np.flatnonzero(outside):
        flag(FLAG_Z_OUTSIDE,
             f'the linearized parquet quasiparticle energy of orbital {p} (0-based), '
             f'{float(eqp[p])!r} Ha, has the diagonal Z = {float(z[p])!r}, outside (0, 1], which '
             f'no root of a positive-weight self-energy has: kept, as Quack keeps it',
             QPWeightOutside, stacklevel=4)


def _energy_record(ec_eh, ec_pp):
    """The correlation energies of one two-body loop's last BSEs: the
    spin-orbital channel energies (singlet + 3 triplet) and Quack's
    (singlet, triplet) parts."""
    return dict(ec_eh=_channel_energy(ec_eh), ec_pp=_channel_energy(ec_pp),
                ec_eh_spin=tuple(ec_eh), ec_pp_spin=tuple(ec_pp))


def _channel_energy(parts):
    """Ec(1) + 3 Ec(2), the spin-orbital channel energy of the (singlet,
    triplet) EcRPA parts, in Quack's order of operations."""
    singlet, triplet = parts
    return float(singlet + TRIPLET_MULTIPLICITY * triplet)


# ---------------------------------------------------------------------------
# qsPA: R_qsParquet


def qs_parquet(eps_hf, c_hf, s, x, hc, eri_ao, nO, *, p_hf=None, tda_eh=True, tda_pp=True,
               eta_1b=PARQUET_REGULARIZATION, eta_2b=PARQUET_REGULARIZATION,
               max_it_1b=PARQUET_MAX_IT_1B, conv_1b=PARQUET_CONVERGENCE,
               max_it_2b=PARQUET_MAX_IT_2B, conv_2b=PARQUET_CONVERGENCE,
               max_diis_1b=PARQUET_MAX_DIIS_1B, max_diis_2b=PARQUET_MAX_DIIS_2B, reg_pa=True,
               broadening_1b=None, damping=PARQUET_VERTEX_DAMPING, vertex_builder=None):
    """qsPA from AO integrals and an RHF start: Quack's R_qsParquet.

    Every one-body iteration (module docstring): J and K of the current
    density (Hartree_matrix_AO_basis, exchange_matrix_AO_basis), the MO
    integrals in the current orbitals (AOtoMO_ERI_RHF), the two-body loop
    from zero vertices on the current energies eQP, the static self-energy
    on eQP (R_Parquet_self_energy_SRG_sp for reg_pa, else
    R_Parquet_self_energy_eta), symmetrized and taken to the AO basis
    (MOtoAO), F = Hc + J + K/2 + Sigma_AO, the commutator error
    max |F P S - S P F|, the one-body DIIS on F (max_diis_1b > 1), and the
    diagonalization of X^T F X (dsyev, Quack's diagonalize_matrix) into new
    energies and orbitals, with P = 2 C_occ C_occ^T. No quasiparticle
    equation is solved, so there is no `qp_solver`.

    The reducible vertices are built at omega = 0 (`static_reducible_vertex`),
    as R_evParquet builds them; the paper tree's R_qsParquet omits the omega
    argument of its four R_*_Phi calls (undefined behaviour, not ported).

    Parameters
    ----------
    eps_hf : ndarray, shape (norb,)
        RHF orbital energies (the first eQP, and the printed e_HF).
    c_hf : ndarray, shape (nbas, norb)
        RHF orbitals.
    s : ndarray, shape (nbas, nbas)
        AO overlap.
    x : ndarray, shape (nbas, norb)
        Orthogonalizer, X^T S X = 1 (Quack's X, `orthogonalization_matrix`;
        any X spanning the same space gives the same energies to rounding).
    hc : ndarray, shape (nbas, nbas)
        Core Hamiltonian (Quack's Hc = T + V): the mean field's own
        (run_parquet takes mf.get_hcore(), with an ECP, X2C or a field, so
        that the first Fock matrix is the reference's).
    eri_ao : ndarray, shape (nbas,)*4
        <mu nu|la si>, AO, physicist (Quack's ERI_AO).
    nO : int
        Doubly occupied orbitals.
    p_hf : ndarray, shape (nbas, nbas), optional
        RHF density; default 2 C_occ C_occ^T.
    reg_pa : bool
        True: the SRG static self-energy with flow parameter eta_1b
        (Hartree^-2, R_Parquet_self_energy_SRG_sp, the paper's qsPA); False:
        the eta-broadened one (R_Parquet_self_energy_eta) with the broadening
        broadening_1b (Quack reads its reg_1b for both, a flow parameter in
        one and a broadening in the other). True or False (numpy's bool too);
        anything else is refused.
    broadening_1b : float, optional
        reg_pa=False only, where it is required: the broadening eta of the
        eta-form static self-energy, Hartree, a finite real number > 0 (small
        is weakly broadened; 0 would be unregularized). Refused with
        reg_pa=True.
    eta_1b : float or None
        As for `ev_parquet` under reg_pa=True. Under reg_pa=False it is not
        read: None is accepted there, a number given is still checked, and
        the result's options record None.
    tda_eh, tda_pp, eta_2b, max_it_1b, conv_1b, max_it_2b, conv_2b,
    max_diis_1b, max_diis_2b, damping, vertex_builder
        As for `ev_parquet`; conv_1b bounds the commutator error, and the
        default damping 0.75 is Quack's alpha = 1 - damping = 0.25 of
        R_qsParquet.

    Returns
    -------
    ParquetResult
    """
    eps_hf = np.array(eps_hf, dtype=float)          # the result's own, never the caller's
    c_hf = np.asarray(c_hf, dtype=float)
    s, x, hc = (np.asarray(m, dtype=float) for m in (s, x, hc))
    eri_ao = np.ascontiguousarray(eri_ao, dtype=float)
    nbas, norb = c_hf.shape
    if eps_hf.shape != (norb,):
        raise ValueError(f'eps_hf has shape {eps_hf.shape}, c_hf {c_hf.shape}')
    for name, m, shape in (('s', s, (nbas, nbas)), ('x', x, (nbas, norb)),
                           ('hc', hc, (nbas, nbas)), ('eri_ao', eri_ao, (nbas,) * 4)):
        if m.shape != shape:
            raise ValueError(f'{name} has shape {m.shape}, expected {shape}')
    nO = checked_occupied(nO, norb, 'doubly occupied orbitals', 'norb')
    reg_pa = checked_switch('reg_pa', reg_pa)
    opts = _loop_options('qs', tda_eh, tda_pp, eta_1b, eta_2b, max_it_1b, conv_1b,
                         max_it_2b, conv_2b, max_diis_1b, max_diis_2b, damping, vertex_builder,
                         eta_1b_read=reg_pa)
    opts['reg_pa'] = reg_pa
    opts['broadening_1b'] = _checked_broadening(reg_pa, broadening_1b)
    if p_hf is None:
        p_hf = 2.0 * c_hf[:, :nO] @ c_hf[:, :nO].T
    p_hf = np.asarray(p_hf, dtype=float)
    if p_hf.shape != (nbas, nbas):
        raise ValueError(f'p_hf has shape {p_hf.shape}, expected {(nbas, nbas)}')
    flags, timer = [], {}
    t_start = time.perf_counter()
    with collecting(flags):
        result = _qs_loops(eps_hf, c_hf, p_hf, s, x, hc, eri_ao, nO, opts, timer)
    timer['total'] = time.perf_counter() - t_start
    result.numerical_flags = gathered(flags)
    result.times = timer
    return result


def _qs_loops(eps_hf, c_hf, p_hf, s, x, hc, eri_ao, nO, opts, timer):
    norb = c_hf.shape[1]
    state = _TwoBodyState(norb, opts['max_diis_2b'])
    diis_1b = (QuackDIIS(opts['max_diis_1b'], [s.shape], label='qsPA one-body')
               if opts['max_diis_1b'] > 1 else None)
    eqp, c, p = eps_hf.copy(), c_hf.copy(), p_hf.copy()
    n_it_1b, err_1b = 0, PARQUET_QUACK_INITIAL_ERROR
    history = []
    eri = None
    while err_1b > opts['conv_1b'] and n_it_1b < opts['max_it_1b']:
        n_it_1b += 1
        t0 = time.perf_counter()
        with stage(timer, 'fock'):
            j, k = hartree_exchange(p, eri_ao)
        with stage(timer, 'integral_transform'):
            eri = spectra = None    # the previous orbitals' integrals and spectra
            eri = ao_to_mo_eri(c, eri_ao)
        state.reset()
        c_two_body = c
        spectra, ec_eh, ec_pp, record = _two_body_loop(state, eqp, eri, nO, opts, timer,
                                                       check=True)
        with stage(timer, 'self_energy'):
            if opts['reg_pa']:
                sigc, z = static_self_energy_srg(opts['eta_1b'], eqp, eri, nO, spectra)
            else:
                sigc, z = static_self_energy_eta(opts['broadening_1b'], eqp, eri, nO, spectra)
        with stage(timer, 'fock'):
            sigc = 0.5 * (sigc + sigc.T)
            sc = s @ c
            sigc_ao = sc @ (sigc @ sc.T)
            fock = hc + j + 0.5 * k + sigc_ao
            err_f = fock @ (p @ s) - (s @ p) @ fock
            err_1b = _quack_maxval(np.abs(err_f))
            if diis_1b is not None:
                fock = diis_1b.push(err_f, fock)
            eqp, cp = _diagonalize_fock(x.T @ (fock @ x), n_it_1b)
            c = x @ cp
            sigc = c.T @ (sigc_ao @ c)
            p = 2.0 * c[:, :nO] @ c[:, :nO].T
        record.update(_energy_record(ec_eh, ec_pp), eqp=eqp.copy(), sigc=np.diag(sigc).copy(),
                      z=np.array(z, dtype=float), err_1b=err_1b,
                      diis_rcond=None if diis_1b is None else diis_1b.rcond,
                      seconds=time.perf_counter() - t0)
        history.append(record)
    converged_1b = _one_body_verdict('qsPA', n_it_1b, err_1b, opts,
                                     'max |F P S - S P F|',
                                     'Quack stops the program here, the run returns its last '
                                     'iterate')
    result = _result('qs', history, eqp, sigc, np.array(z, dtype=float), eps_hf, nO, None, None,
                     spectra, state, ec_eh, ec_pp, converged_1b, n_it_1b, opts)
    result.mo_coeff = c
    result.mo_coeff_two_body = c_two_body
    result.sigc_ao = sigc_ao
    return result


def _diagonalize_fock(fp, n_it_1b):
    """Eigenpairs of X^T F X from its upper triangle by dsyev 'U'
    (PARQUET_QS_EIGH_DRIVER), as Quack's diagonalize_matrix. A non-finite
    matrix or a failed LAPACK call gives NaN eigenpairs with the flag
    'parquet_qs_fock_not_finite' (Quack prints 'Problem in
    diagonalize_matrix' and goes on with what dsyev left)."""
    n = fp.shape[0]
    if np.isfinite(np.triu(fp)).all():
        try:
            return la.eigh(fp, lower=False, driver=PARQUET_QS_EIGH_DRIVER, check_finite=False)
        except la.LinAlgError as err:
            reason = f'its diagonalization failed ({err})'
    else:
        reason = 'it holds NaN or inf'
    flag(FLAG_QS_FOCK_NOT_FINITE,
         f'qsPA one-body iteration {n_it_1b}: the effective Fock matrix cannot be '
         f'diagonalized, {reason}; the energies and orbitals are NaN from here on',
         ParquetNotFinite, stacklevel=3)
    return np.full(n, np.nan), np.full((n, n), np.nan)


def _one_body_verdict(label, n_it_1b, err_1b, opts, measure, outcome):
    """Quack's verdict on a one-body loop (failed when n_it_1b == max_it_1b),
    False also for a NaN change, which ends Quack's loop as a success."""
    converged = n_it_1b != opts['max_it_1b']
    if np.isnan(err_1b):
        flag(FLAG_ONE_BODY_NOT_FINITE,
             f'{label} one-body loop: the change {measure} is NaN at iteration {n_it_1b}; '
             f'Quack ends the loop there and reports success', ParquetNotFinite,
             stacklevel=4)
        return False
    if not converged:
        flag(FLAG_ONE_BODY_NOT_CONVERGED,
             f'{label} one-body loop not converged after {n_it_1b} iterations: {measure} = '
             f'{err_1b:.3e} against conv_1b = {opts["conv_1b"]:.1e} (Quack counts n_it == '
             f'max_it_1b as failed); {outcome}', ParquetNotConverged, stacklevel=4)
    return converged


def orthogonalization_matrix(s):
    """Quack's orthogonalization_matrix: the canonical orthogonalization
    X = U s^-1/2 over the overlap eigenvalues s above
    PARQUET_ORTHOGONALIZATION_THRESHOLD, in ascending order of s (dsyev 'U',
    PARQUET_QS_EIGH_DRIVER, as Quack's diagonalize_matrix), so that
    X^T S X = 1.

    Parameters
    ----------
    s : ndarray, shape (nbas, nbas)
        AO overlap.

    Returns
    -------
    x : ndarray, shape (nbas, norb)
    """
    u, vecs = la.eigh(np.asarray(s, dtype=float), lower=False, driver=PARQUET_QS_EIGH_DRIVER)
    keep = u > PARQUET_ORTHOGONALIZATION_THRESHOLD
    return vecs[:, keep] / np.sqrt(u[keep])


def hartree_exchange(p, eri_ao):
    """(J, K) of the density p: Quack's Hartree_matrix_AO_basis and
    exchange_matrix_AO_basis,

        J[mu, nu] = sum_{la, si} P[la, si] <mu la|nu si>,
        K[mu, nu] = - sum_{la, si} P[la, si] <mu la|si nu>,

    K with Quack's minus sign (its Fock matrix is Hc + J + K/2).

    Parameters
    ----------
    p : ndarray, shape (nbas, nbas)
        Density matrix (2 C_occ C_occ^T for a closed shell).
    eri_ao : ndarray, shape (nbas,)*4
        <mu nu|la si>, AO, physicist.

    Returns
    -------
    j, k : ndarray, shape (nbas, nbas)
    """
    j = np.einsum('mlns,ls->mn', eri_ao, p)
    k = -np.einsum('mlsn,ls->mn', eri_ao, p)
    return j, k


def ao_to_mo_eri(c, eri_ao):
    """<pq|rs> in the orbitals c: Quack's AOtoMO_ERI_RHF, the semi-direct
    O(N^5) transform, one index at a time.

    Each of the four GEMMs contracts the leading AO index with c and appends
    the MO index last, so after four the order is [p, q, r, s]: the same
    sequence as Quack's four dgemm. Memory: two nbas^3 norb intermediates
    while a GEMM runs (2.9 GB each at nbas = norb = 138), no copy of eri_ao.

    Parameters
    ----------
    c : ndarray, shape (nbas, norb)
    eri_ao : ndarray, shape (nbas,)*4
        Any index convention: physicist in, physicist out.

    Returns
    -------
    eri : ndarray, shape (norb,)*4
    """
    c = np.asarray(c, dtype=float)
    nbas, norb = c.shape
    t = np.ascontiguousarray(eri_ao, dtype=float)
    if t.shape != (nbas,) * 4:
        raise ValueError(f'eri_ao has shape {t.shape}, expected {(nbas,) * 4}')
    for _ in range(4):
        # [mu, rest] -> [rest, p]: the leading AO index contracted, the MO one last
        t = t.reshape(nbas, -1).T @ c
    return t.reshape((norb,) * 4)


# ---------------------------------------------------------------------------
# the front door


def run_parquet(mf, scheme='os', *, frozen=None, tda_eh=True, tda_pp=True,
                eta_1b=PARQUET_REGULARIZATION, eta_2b=PARQUET_REGULARIZATION, max_it_1b=None,
                conv_1b=PARQUET_CONVERGENCE, max_it_2b=None, conv_2b=PARQUET_CONVERGENCE,
                max_diis_1b=PARQUET_MAX_DIIS_1B, max_diis_2b=PARQUET_MAX_DIIS_2B,
                qp_solver=None, reg_pa=True, broadening_1b=None,
                damping=PARQUET_VERTEX_DAMPING, vertex_builder=None):
    """The closed-shell parquet approximation on a PySCF RHF reference.

    scheme 'flex', 'os' and 'ev' run Quack's R_evParquet (`ev_parquet`) on
    the RHF energies and MO integrals; 'qs' runs R_qsParquet (`qs_parquet`)
    on the AO integrals, starting from the RHF orbitals, with Quack's
    canonical orthogonalization (`orthogonalization_matrix`). Every option,
    and the reference's occupations (at least one occupied and one virtual
    orbital), are checked before the integrals are built.

        # Table II (osPA, TDA, s_2b = 1, unregularized self-energy)
        res = run_parquet(mf, 'os', eta_1b=1e8, eta_2b=1.0, max_diis_2b=5)
        ip = -res.eqp[:res.nO].max()     # principal IP, Ha (Ne/aug-cc-pVTZ: 21.17 eV)
        res.converged, res.n_it_2b, res.numerical_flags

    THE DEFAULTS ARE QUACK'S, not production settings: the paper's Table III
    s_1b = s_2b = 50 and tau = 1e-4, and Quack's algorithm, damping 0.75
    (Quack's alpha 0.25) and no DIIS, so that the comparisons with Quack
    hold. At s_2b = 50 the damped two-body loop without DIIS need not
    converge: run_parquet(mf), osPA at the defaults, ends its 200 two-body
    iterations at a change of 5.6 Ha on water/6-31G (the paper's geometry)
    and 0.017 Ha on ammonia/6-31G, warns 'parquet_two_body_not_converged'
    and returns energies that mean nothing (water: a principal IP of
    -0.77 eV; in cc-pVDZ a change of 26 Ha and an IP of -32 eV); evPA runs
    the same loop first, and qsPA at every one-body iteration. Give the loop
    two-body DIIS (max_diis_2b=5: water and ammonia 6-31G converge in 53 and
    54 iterations), as docs/methods/parquet.md does for the paper's Table III.

    Parameters
    ----------
    mf : pyscf RHF
        A converged closed-shell RHF with four-index integrals. UHF, ROHF
        with open shells, Kohn-Sham, density fitting, an attached
        environment and complex orbitals are refused, and for qs a complex
        core Hamiltonian (before any two-electron integral; the port is
        real); an unconverged one is flagged ('scf_not_converged').
    scheme : {'flex', 'os', 'ev', 'qs'}
    frozen : None
        Frozen orbitals are not ported (Quack's nC = nR = 0): anything but
        None or 0 is refused.
    tda_eh, tda_pp, eta_1b, eta_2b, max_it_1b, conv_1b, max_it_2b, conv_2b,
    max_diis_1b, max_diis_2b, damping, vertex_builder
        As for `ev_parquet` (eta_1b under qsPA with reg_pa=False as for
        `qs_parquet`: not read); the defaults are the paper's Table III
        settings (TDA, s_1b = s_2b = 50, tau = 1e-4) and Quack's algorithm
        (damping 0.75, Quack's alpha 0.25, for every scheme; no DIIS).
    qp_solver : {'find_qp_root', 'linearized'} or callable, optional
        flex, os and ev: as for `ev_parquet` (a function under the same
        contract), None being 'find_qp_root', the house's root finder (the
        default; only this front door takes None). qs refuses any value: it
        diagonalizes F + Sigma and solves no quasiparticle equation.
    reg_pa : bool
        qsPA: the SRG (True, the paper's) or the eta-broadened static
        self-energy (False, with `broadening_1b`). R_evParquet has no eta
        form (it takes reg_PA and ignores it), so flex, os and ev refuse
        reg_pa=False. True or False (numpy's bool too); anything else is
        refused.
    broadening_1b : float, optional
        qsPA with reg_pa=False only, where it is required: as for
        `qs_parquet`.

    Returns
    -------
    ParquetResult
    """
    if scheme not in SCHEMES:
        raise ValueError(f'scheme={scheme!r}: one of {SCHEMES}')
    refuse_frozen(frozen, 'run_parquet')
    _check_reference(mf)
    reg_pa = checked_switch('reg_pa', reg_pa)
    if scheme == 'qs':
        if qp_solver is not None:
            raise ValueError(f'qp_solver={qp_solver!r}: qsPA diagonalizes F + Sigma and '
                             f'solves no quasiparticle equation, so scheme qs takes no '
                             f'qp_solver (leave it None)')
        _checked_broadening(reg_pa, broadening_1b)
    else:
        if not reg_pa:
            raise ValueError(f'reg_pa=False selects the eta-broadened static self-energy of '
                             f'qsPA; scheme {scheme!r} runs R_evParquet, which has no eta '
                             f'form (its dynamic self-energy is SRG-regularized by eta_1b)')
        if broadening_1b is not None:
            raise ValueError(f'broadening_1b={broadening_1b!r} is read only by qsPA with '
                             f'reg_pa=False, not by scheme {scheme!r}')
        qp_solver = DEFAULT_QP_SOLVER if qp_solver is None else checked_qp_solver(qp_solver)
    # the loop options are checked again by ev_parquet and qs_parquet, after the integrals
    _loop_options(scheme, tda_eh, tda_pp, eta_1b, eta_2b, max_it_1b, conv_1b, max_it_2b,
                  conv_2b, max_diis_1b, max_diis_2b, damping, vertex_builder,
                  eta_1b_read=reg_pa)
    mol = mf.mol
    nO = mol.nelectron // 2
    eps_hf = np.asarray(mf.mo_energy, dtype=float)
    if not 0 < nO < eps_hf.shape[0]:
        raise ValueError(f'run_parquet: the RHF has {nO} doubly occupied orbitals of '
                         f'{eps_hf.shape[0]}; the eh and pp problems of the parquet '
                         f'approximation need at least one occupied and one virtual orbital '
                         f'(a larger basis gives the virtual ones)')
    hc = None
    if scheme == 'qs':
        # read once, and refused before any two-electron integral where it is complex
        hc = np.asarray(mf.get_hcore())
        if np.iscomplexobj(hc):
            raise ValueError('run_parquet: the core Hamiltonian of the RHF is complex '
                             '(spin-orbit terms); the closed-shell parquet port is real')
    common = dict(tda_eh=tda_eh, tda_pp=tda_pp, eta_1b=eta_1b, eta_2b=eta_2b,
                  conv_1b=conv_1b, conv_2b=conv_2b, max_diis_1b=max_diis_1b,
                  max_diis_2b=max_diis_2b, damping=damping, vertex_builder=vertex_builder)
    flags = []
    t0 = time.perf_counter()
    with collecting(flags):
        if not getattr(mf, 'converged', True):
            flag(FLAG_SCF_NOT_CONVERGED, 'run_parquet: the RHF reference is not converged; '
                 'the parquet run starts from its last iterate', ParquetNotConverged)
        if scheme == 'qs':
            c_hf = np.asarray(mf.mo_coeff, dtype=float)
            s = np.asarray(mf.get_ovlp(), dtype=float)
            x = orthogonalization_matrix(s)
            if x.shape[1] != c_hf.shape[1]:
                raise ValueError(
                    f'Quack\'s canonical orthogonalization keeps {x.shape[1]} functions '
                    f'and the RHF has {c_hf.shape[1]} orbitals; the qsPA port needs '
                    f'the same space')
            hc = np.asarray(hc, dtype=float)
            # the closed-shell density as the RHF builds it, 2 C_occ C_occ^T, also for a
            # closed-shell ROHF, whose own make_rdm1 hands it out per spin
            p_hf = np.asarray(scf_hf.make_rdm1(mf.mo_coeff, mf.mo_occ), dtype=float)
            # (mu la|nu si) -> <mu nu|la si> in place: no second nbas^4 array
            eri_ao = _physicist_in_place(mol.intor('int2e', aosym='s1'))
            seconds = time.perf_counter() - t0
            result = qs_parquet(eps_hf, c_hf, s, x, hc, eri_ao, nO, p_hf=p_hf,
                                max_it_1b=PARQUET_MAX_IT_1B if max_it_1b is None else max_it_1b,
                                max_it_2b=PARQUET_MAX_IT_2B if max_it_2b is None else max_it_2b,
                                reg_pa=reg_pa, broadening_1b=broadening_1b, **common)
        else:
            norb = eps_hf.shape[0]
            # (pr|qs) -> <pq|rs> in place: no second norb^4 array
            eri = _physicist_in_place(
                np.asarray(get_two_electron_integrals_chemist(mol, mf)).reshape((norb,) * 4))
            seconds = time.perf_counter() - t0
            result = ev_parquet(eps_hf, eri, nO, scheme, max_it_1b=max_it_1b,
                                max_it_2b=max_it_2b, qp_solver=qp_solver, **common)
    result.times['integrals'] = seconds
    result.numerical_flags = gathered(flags)
    return result


def _physicist_in_place(t):
    """The chemist integrals t[p, r, q, s] = (pr|qs) turned in place into the
    physicist t[p, q, r, s] = <pq|rs>, one slab t[p] at a time: an (n, n, n)
    temporary instead of a second n^4 array, the same numbers as
    np.ascontiguousarray(t.transpose(0, 2, 1, 3)). t is a C-contiguous
    float64 array the caller owns (copied first if it is not one)."""
    t = np.require(t, dtype=float, requirements=['C', 'W'])
    for p in range(t.shape[0]):
        t[p] = t[p].transpose(1, 0, 2).copy()
    return t


# ---------------------------------------------------------------------------
# checks and the result


def _check_reference(mf):
    """Refuse a reference the closed-shell parquet port cannot take."""
    who = 'run_parquet'
    if getattr(mf, 'mo_coeff', None) is None or getattr(mf, 'mo_energy', None) is None:
        raise ValueError(f'{who}: the mean field has no orbitals; run it first')
    if np.asarray(mf.mo_energy).ndim != 1:
        raise NotImplementedError(f'{who}: an unrestricted reference; the closed-shell '
                                  'parquet port takes an RHF')
    names = {cls.__name__ for cls in type(mf).__mro__}
    if names & {'UHF', 'GHF', 'KohnShamDFT', 'RKS', 'UKS', 'GKS'} or hasattr(mf, 'xc'):
        raise NotImplementedError(f'{who}: {type(mf).__name__} is not an RHF; the parquet '
                                  'drivers start from Hartree-Fock orbitals and energies')
    if getattr(mf, 'with_df', None) is not None:
        raise NotImplementedError(f'{who}: a density-fitted RHF; the parquet port uses '
                                  'four-index integrals, as Quack does')
    if np.asarray(mf.mo_coeff).shape[0] != mf.mol.nao_nr():
        raise NotImplementedError(f'{who}: the orbitals are not spatial orbitals of the '
                                  'AO basis (a generalized reference?)')
    if np.iscomplexobj(mf.mo_energy) or np.iscomplexobj(mf.mo_coeff):
        raise NotImplementedError(f'{who}: complex orbitals; the closed-shell parquet port '
                                  'is real')
    if not isinstance(environment_of(mf), NoEnvironment):
        raise NotImplementedError(f'{who}: an environment is attached to the mean field; '
                                  "Quack's parquet approximation is gas phase")
    refuse_frozen(getattr(mf, 'frozen', None), who, 'mf.frozen')
    require_closed_shell_or_unrestricted(mf, who)


def _check_integrals(eps, eri, nO):
    """(norb, nO) of (eps, eri, nO), or ValueError."""
    if eps.ndim != 1:
        raise ValueError(f'eps must be one-dimensional, got shape {eps.shape}')
    norb = eps.shape[0]
    if np.shape(eri) != (norb,) * 4:
        raise ValueError(f'eri has shape {np.shape(eri)}, expected {(norb,) * 4}')
    return norb, checked_occupied(nO, norb, 'doubly occupied orbitals', 'norb')


def _loop_options(scheme, tda_eh, tda_pp, eta_1b, eta_2b, max_it_1b, conv_1b, max_it_2b,
                  conv_2b, max_diis_1b, max_diis_2b, damping, vertex_builder,
                  eta_1b_read=True):
    """The options of a run, checked (`options.loop_options`), with this
    driver's defaults: damping PARQUET_VERTEX_DAMPING for every scheme, the
    builder `static_reducible_vertex`; eta_1b_read False (qsPA with reg_pa
    False): eta_1b is not read, may be None, and is recorded as None."""
    return loop_options(scheme, tda_eh, tda_pp, eta_1b, eta_2b, max_it_1b, conv_1b, max_it_2b,
                        conv_2b, max_diis_1b, max_diis_2b, damping, vertex_builder,
                        default_damping=PARQUET_VERTEX_DAMPING,
                        default_builder=static_reducible_vertex, eta_1b_read=eta_1b_read)


def _checked_broadening(reg_pa, broadening_1b):
    """The broadening of qsPA's eta-form static self-energy (reg_pa False),
    Hartree, a finite real number > 0 (a bool, a string or an array is
    refused, not converted); None for the SRG form (reg_pa True), which
    refuses one. ValueError otherwise."""
    if reg_pa:
        if broadening_1b is not None:
            raise ValueError(f'broadening_1b={broadening_1b!r} is the broadening of the '
                             f'eta-form static self-energy (reg_pa=False); the SRG form '
                             f'(reg_pa=True) takes its flow parameter in eta_1b')
        return None
    what = ('reg_pa=False takes the broadening of its eta-form static self-energy in '
            'broadening_1b (Hartree, a finite real number > 0); eta_1b, an SRG flow '
            'parameter in Hartree^-2, is not read there')
    value = checked_real('broadening_1b', broadening_1b, what)
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f'broadening_1b = {broadening_1b!r}: {what}')
    return value


def _result(scheme, history, eqp, sigc, z, eps_hf, nO, eqp_lin, qp_converged, spectra,
            state, ec_eh, ec_pp, converged_1b, n_it_1b, opts):
    converged_2b = [h['converged_2b'] for h in history]
    converged = converged_1b is not False and all(c is not False for c in converged_2b)
    # 'alpha' is the loop's own 1 - damping, not an option
    options = {k: v for k, v in opts.items() if k not in ('vertex_builder', 'alpha')}
    options['vertex_builder'] = builder_name(opts['vertex_builder'])
    if 'qp_solver' in options:
        options['qp_solver'] = qp_solver_label(options['qp_solver'])
    energies = _energy_record(ec_eh, ec_pp)
    return ParquetResult(
        scheme=scheme, eqp=np.array(eqp, dtype=float), sigc=sigc, z=z, eps_hf=eps_hf,
        nO=int(nO), eqp_lin=eqp_lin, qp_converged=qp_converged, spectra=spectra,
        vertices=state.as_vertices(), converged=converged, converged_1b=converged_1b,
        converged_2b=converged_2b, n_it_1b=n_it_1b, n_it_2b=[h['n_it_2b'] for h in history],
        history=history, times={}, numerical_flags=[], options=options,
        qp_followed=history[-1].get('qp_followed'), qp_flipped=history[-1].get('qp_flipped'),
        **energies)
