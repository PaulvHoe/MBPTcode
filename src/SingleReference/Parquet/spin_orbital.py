"""The spin-orbital parquet drivers, Quack's G branch: FLEX, osPA and evPA as
Quack's G_evParquet, qsPA as its G_qsParquet (Marie and Loos, J. Chem. Phys.
163, 194115 (2025), Sec. V, static kernel approximation of Sec. IV A),
ported from the paper-era Quack (git e3528a40).

    res = run_parquet_so(mf, 'os', eta_1b=1e8, eta_2b=1.0, max_diis_2b=5)
    res.eqp, res.converged, res.n_it_2b, res.numerical_flags

The closed-shell drivers (`restricted`) run the same algorithm spin-adapted;
this module takes any real Hartree-Fock reference (RHF, UHF, GHF) in spin
orbitals and returns a result of the same shape (`SpinOrbitalParquetResult`:
the fields of restricted.ParquetResult over spin orbitals, without its
singlet and triplet parts, with the spin-orbital layout). The options mean
what they mean there: `damping` is the weight of the previous vertices,
`qp_solver` how the quasiparticle equation is solved, `vertex_builder` the
reducible-vertex hook.

TWO NESTED LOOPS (the paper's steps 1 to 10, its Fig. 8). The TWO-BODY loop
holds the one-body energies eps and the integrals <pq|rs> fixed and iterates
the reducible vertices i Phi^eh and i Phi^pp (real, Phi[p, q, r, s]). One
iteration, as G_evParquet and G_qsParquet run it:

* the eh and pp BSE blocks on eps (phGLR_A/B, ppGLR_B/C/D) plus the static
  kernels of the previous vertices (G_eh_Gamma_A/B, G_pp_Gamma_B/C/D;
  Eqs. (35), (41), (47)), zero at the very first two-body iteration of the
  run, solved (phGLR, ppGLR; Eqs. (34), (40));
* the effective integrals M^eh, M^ee, M^hh from the eigenvectors and the
  irreducible vertices of the same previous vertices (G_eh_screened_integral,
  G_pp_screened_integral; Eqs. (38), (44));
* new vertices at omega = 0 from those integrals and the new eigenvalues
  (G_eh_Phi, G_pp_Phi; Eqs. (39), (45), regularized with s_2b, Eq. (52)),
  or from the `vertex_builder` handed in;
* the change err = max |Phi_new - Phi_old| of each vertex before damping,
  then
      Phi <- (1 - d) Phi_new + d Phi_old,
  d the `damping` as evGW means it, the weight of the previous iterate (0 by
  default for FLEX, osPA and evPA: Quack's mixing weight alpha = 1 - d is 1
  in G_evParquet, no damping; 0.75 for qsPA: alpha 0.25 in G_qsParquet),
  evaluated as Quack evaluates it, alpha Phi_new + (1 - alpha) Phi_old; for
  max_diis_2b > 1 the damped vertices are then extrapolated by DIIS over both
  together (one vector of 2 nso^4 numbers, eh then pp, error
  Phi_damped - Phi_old): the loop damps first and extrapolates the damped
  iterate, as Quack does;
* while err_2b = max(err_eh, err_pp) > conv_2b and fewer than max_it_2b
  iterations ran.

The ONE-BODY step builds the self-energy from the last iteration's spectra.

osPA and evPA (G_evParquet): Sigma_pp and Z_p at w = eOld_p, every other
    denominator on eOld (G_Parquet_self_energy_diag; Eqs. (21), (48), (50),
    (51), regularized with s_1b, Eq. (53)); eQPlin = eHF + Z SigC (anchored on
    eHF at every iteration); eQP from the quasiparticle equation (33),
    w = eHF_p + Sigma_pp(w), as `qp_solver` says: 'find_qp_root' (the
    default), the house's one root finder on that equation (not Quack's,
    `solve_qp_scan_so`), which from the second evPA iteration on follows the
    root of the previous one where that root's Z lay in (0, 1] and otherwise
    searches as at the first (`options.roots_to_follow`), flagging a branch
    it reports the largest weight has left ('parquet_qp_branch_flipped');
    'linearized', eQP = eQPlin, G_evParquet's own and only branch and the
    Fortran reference of this driver; or a function handed in, under the
    closed-shell driver's contract (`options.checked_qp_solver`; G_evParquet
    has no search of its own, and Quack's closed-shell one,
    R_Parquet_QP_graph, is not ported); an energy kept
    with Z outside (0, 1] is flagged 'z_outside' on every route;
    DIIS on eQP (error eQP - eOld) for max_diis_1b > 1;
    err_1b = max |eOld - eQP|; eOld <- eQP. The vertices, the previous
    eigenvalues and the two-body DIIS history carry over from one one-body
    iteration to the next. FLEX is one two-body iteration and one one-body
    step, osPA one one-body step, evPA the loop to err_1b <= conv_1b.

    A spin-symmetric reference (an RHF, interleaved alpha, beta) has its
    alpha and beta copy of each spatial orbital equal only to roundoff in
    the computed self-energy (1e-13 Ha), and evPA at a large s_1b is chaotic
    between dense poles: a root search can take the two copies of a
    near-pole orbital to different roots, and even their linearized eQP,
    fed back through eOld into every denominator, part by a factor of 2 to
    10 or more per iteration (water/6-31G at s_1b = 1e8, s_2b = 1, damping
    0.75: the linearized evPA splits an orbital by 4.7 meV after 18
    iterations and by 39 meV after 19; the house's root finder, following
    each root, kept the copies within 1.4e-13 eV over four iterations). With
    `spin_pairs` (run_parquet_so passes them for an RHF) the
    first of each pair is solved, whatever the `qp_solver`, and its eQP and
    Z are copied to the second, so the pair stays degenerate to the bit (the
    records' sigc and eqp_lin are those computed, degenerate to roundoff).
    UHF and GHF references of closed shells get no pairing, so their evPA
    at large s_1b can depend on roundoff: closed shells belong in the
    restricted driver (`restricted.run_parquet`).

qsPA (G_qsParquet): in the spin-AO basis (the alpha AO block, then the beta
    one), the Fock matrix of the density P = C_occ C_occ^T (Coulomb and
    exchange per spin block, `ghf_fock`, on the mean field's own core
    Hamiltonian), the integrals in the current orbitals
    (`spin_orbital_eri`, AOtoMO_ERI_GHF over the four spin blocks), the
    two-body state reset (vertices, eigenvalues, DIIS) and the two-body loop
    run on the current eQP; the static self-energy matrix
    (G_Parquet_self_energy_SRG_dp) symmetrized, taken to the AO basis,
    S C Sigma (S C)^T, and added to F; err_1b = max |F P S - S P F| (before
    DIIS); DIIS on F for max_diis_1b > 1; X^T F X diagonalized (dsyev, the
    upper triangle, PARQUET_QS_EIGH_DRIVER) to eQP and C = X C', P rebuilt
    from the lowest nO. qsPA solves no quasiparticle equation, so it takes no
    `qp_solver`. G_qsParquet reads its reg_PA and ignores it (it has no
    eta-broadened static self-energy), so this driver has no counterpart of
    the restricted reg_pa and broadening_1b.

CONVENTIONS. Energies in Hartree. Spin orbitals occupied first (nO of them;
Quack's nC = nR = 0). eri[p, q, r, s] = <pq|rs> over spin orbitals,
physicist, not antisymmetrized; AO integrals eri_ao[mu, nu, la, si] =
<mu nu|la si> (Quack's ERI_AO); spin-AO coefficients (2 nao, nso), alpha AO
rows first (Quack's GHF layout). `spin_orbital_reference` lays out a mean
field: an RHF interleaved, (0 alpha, 0 beta, 1 alpha, 1 beta, ...), Quack's
GHF order of a closed shell and that of the restricted-to-spin-orbital
fixtures; a UHF the same interleaving of (alpha k, beta k), then, like a GHF,
the occupied spin orbitals first and the virtual ones after, each block
ascending in energy by a stable sort (an RHF keeps its interleaving).

WHAT IS QUACK'S AND KEPT, oddities included:
- G_evParquet does not damp (alpha = 1): Phi = 1 Phi_new + 0 Phi_old is
  still evaluated, so a NaN in the old vertices propagates.
- Quack's failure tests n_it_2b == max_it_2b and n_it_1b == max_it_1b: a
  loop meeting its threshold at its last allowed iteration counts as failed.
  Its maxima are gfortran's at run time: MAXVAL skips NaN elements unless all
  are NaN, MAX of two (arm64 fmaxnm) skips a NaN argument unless both are;
  so a NaN eh vertex beside a finite pp one does not end the loop, and an
  all-NaN change ends it, which Quack reports as a success (here it is
  converged False with the flag 'parquet_two_body_not_finite').
- The printed eigenvalue changes max |Om - Om_old| are taken against zero
  eigenvalues at the first iteration (Quack's old_Om = 0) and are not tested.
- The two-body DIIS history survives from one evPA one-body iteration to the
  next: the next loop's first extrapolations hold the converged errors of the
  last one (condition 1e16 to 1e18), keep its vertices for about max_diis_2b
  iterations, and are decided by roundoff (LAPACK-dependent).
- qsPA returns Z = 1 and Quack's printed Sigma_c, the diagonal of
  C^T Sigma_AO C in the orbitals after the diagonalization. Within a
  degenerate pair (every alpha, beta pair of a closed shell) the eigensolver
  rotates the orbitals at will, so the vertex elements in that basis, and
  their max-norm changes, depend on the LAPACK; energies and spectra do not.
  The final orbitals come in ascending order of eQP, not in the reference's
  layout: run_parquet_so labels their spin anew (`result.spin`) and has no
  orbital label for them (`result.orbital` None); eps_hf stays the
  reference's energies in the reference layout.
- The diagonal self-energy is Quack's G-branch product form
  (spin_orbital_self_energy): equal to the restricted R_*_dp variants, not
  to R_evParquet's partial fractions, at finite s_1b; they agree as s_1b
  goes to infinity. The static self-energy of qsPA is G_qsParquet's
  G_Parquet_self_energy_SRG_dp, which differs from R_qsParquet's
  R_Parquet_self_energy_SRG_sp off the diagonal even as s_1b grows (9.7e-2
  Ha on water/6-31G at s_1b = 1e8, where the diagonals agree to 5e-12; the
  qsPA water HOMO at the paper's settings 21 meV apart), so closed-shell
  qsPA as the paper runs it belongs in the restricted driver
  (`restricted.run_parquet`).
- One occupied spin orbital (any one-electron system: no 2h pair) or one
  virtual (no 2p pair) leaves that pp branch empty; G_evParquet runs on
  (maxval of the empty eigenvalue change is -huge, EMPTY_MAXVAL), and so
  does this module.
- FLEX and osPA have one one-body step, which Quack reports as a failed
  one-body loop; here converged_1b is None for them (and converged_2b None
  for FLEX).
- Quack's main program (GQuAcK) runs G_evParquet_v2 for evParquet, which
  mixes and extrapolates the BSE matrices A, C, D (alpha = 0.5) instead of
  the vertices; this module ports G_evParquet.
Not Quack's: an exactly zero denominator of a regularized weight, where
Quack's (1 - exp(-2 s d^2)) / d is 0/0, gives the weight its limit 0 and
the slope in Z 0, the formula's value a few ulps off zero
(`spin_orbital_two_body.regularized_inverse`, `spin_orbital_self_energy`),
so H/STO-3G's spin flip, e_beta - e_alpha - J, which rounding of the
mean-field energies and J puts at exactly 0.0 or at 2.2e-16, gives the same
numbers either way in the eh TDA (the full eh problem's
X + Y = Om^-1/2 Z^T (A - B)^1/2 is 0 times inf at an exactly zero
A - B = A + B, NaN as in Quack, and is kept).

COST AND MEMORY at nso = 138 spin orbitals (nO = 10, nS = 1280, nVV = 8128,
nOO = 45; one nso^4 array is 2.9 GB), one BLAS thread, from the component
modules: one two-body iteration about 12 min with both TDAs (6 min of it the
pp TDA solve; about 11 min more without the pp TDA, dgeev), holding the
integrals, the two old vertices, one new vertex at a time and the iteration's
pieces (rho_ee 1.2 GB, the pp blocks), about 14 GB without DIIS; this
module's own passes over the vertices (the changes and the damping, in
blocks of at most PARQUET_SO_DRIVER_BLOCK_BYTES and
1/PARQUET_MIN_BLOCKS_PER_VERTEX of a vertex, into the state's own arrays:
the builder's vertex is only read) about 6 s per iteration. Two-body DIIS
stores max_diis_2b error and iterate vectors of 2 nso^4 numbers (11.6 GB per
entry), as Quack does, in slots allocated once that the loop writes the
damped vertices and their errors into, and accumulates its extrapolation into
the two vertices in place through one scratch block (`diis`): nothing beyond
the history and that block, so the peak is the one without DIIS plus
4 max_diis_2b nso^4 numbers (water/6-31G, nso 26: 7.09 nso^4 without DIIS,
27.09 with max_diis_2b 5, both set by the eh vertex product, whose blocks
are whole vertices at that size). The diagonal self-energy 45 s and 0.6 GB;
the quasiparticle equation 0.1 s per orbital for its self-energy's residues
(again when a qp_solver function changes orbital) plus 0.34 s per
evaluation (half the orbitals with `spin_pairs`): find_qp_root 150 to
10 000 evaluations per orbital on water and ammonia 6-31G without
following, and about as many when it follows, the scan window of the root
followed being searched whole for the branch report (`solve_qp_scan_so`); the static
self-energy matrix 53 s and 1.7 GB; per qsPA one-body
iteration the integral transformation 12 s (1.9e11 flops, the 2.9 GB output
and a 0.7 GB intermediate) and the Fock matrix 0.3 s (nao = 69). Spin
orbitals double the restricted norb: at the paper's aug-cc-pVTZ sizes
(nso = 184 for N2 and CO, 276 for CH4) one vertex is 9 GB and 46 GB, so
those runs belong to the closed-shell drivers. Measured on water/aug-cc-pVDZ
(nso 82, one thread): 19 s per TDA two-body iteration and 3.1 s for the
diagonal self-energy.

FLAGS. The numerical events are warned about and raised as the named flags of
`events` (ParquetNotConverged, ParquetNotFinite, ParquetBSEUnstable; the
house's 'scf_not_converged', 'z_outside' and, from the root finder of
qp_solver='find_qp_root', 'no_root'), under the names the closed-shell drivers
use for the same events; the result lists them in `numerical_flags`. A wrong
option is refused before any work, as in the closed-shell drivers (`options`).
"""
import dataclasses
import time

import numpy as np
import scipy.linalg as la

from src.Base.environment_seam import NoEnvironment, environment_of
from src.Base.numerical_flags import collecting, flag, flags_into_out, gathered
from src.Base.numerics.constants import (PARQUET_CONVERGENCE, PARQUET_MAX_DIIS_1B,
                                         PARQUET_MAX_DIIS_2B, PARQUET_MAX_IT_1B,
                                         PARQUET_MAX_IT_2B, PARQUET_MIN_BLOCKS_PER_VERTEX,
                                         PARQUET_ORTHOGONALIZATION_THRESHOLD,
                                         PARQUET_QS_EIGH_DRIVER, PARQUET_QUACK_INITIAL_ERROR,
                                         PARQUET_REGULARIZATION, PARQUET_SO_DRIVER_BLOCK_BYTES,
                                         PARQUET_SO_EV_VERTEX_DAMPING, PARQUET_SO_HCORE_SPIN_TOL,
                                         PARQUET_SO_SPIN_WEIGHT_TOL, PARQUET_VERTEX_DAMPING)
from src.Base.utils.timing import stage
from src.Solvers.qp_equation import QPNoRoot, QPWeightOutside, solve_qp_equation_scan
from src.SingleReference.Parquet.diis import QuackDIIS
from src.SingleReference.Parquet.events import (FLAG_NO_ROOT, FLAG_ONE_BODY_NOT_CONVERGED,
                                                FLAG_ONE_BODY_NOT_FINITE, FLAG_QS_FOCK_NOT_FINITE,
                                                FLAG_SCF_NOT_CONVERGED,
                                                FLAG_TWO_BODY_NOT_CONVERGED,
                                                FLAG_TWO_BODY_NOT_FINITE, FLAG_VERTEX_NOT_FINITE,
                                                FLAG_Z_OUTSIDE, ParquetNotConverged,
                                                ParquetNotFinite)
from src.SingleReference.Parquet.options import (DEFAULT_QP_SOLVER, builder_name,
                                                 checked_occupied, checked_qp_solver,
                                                 flag_branch, loop_options, qp_solver_label,
                                                 refuse_frozen, roots_to_follow, solve_with)
from src.SingleReference.Parquet.pairs import pp_pair_count
from src.SingleReference.Parquet.spectra import Channel, SpinOrbitalSpectra, SpinOrbitalVertices
from src.SingleReference.Parquet.spin_orbital_self_energy import (
    DiagonalSelfEnergySO, diagonal_self_energy_so, static_self_energy_srg_so)
from src.SingleReference.Parquet.spin_orbital_two_body import (
    eh_bse_blocks_so, eh_effective_integrals_so, eh_kernel_blocks_so, eh_reducible_vertex_so,
    pp_bse_blocks_so, pp_effective_integrals_so, pp_kernel_blocks_so, pp_reducible_vertex_so,
    solve_eh_bse_so, solve_pp_bse_so)

#: The schemes of the paper: FLEX (one two-body iteration, one one-body
#: step), osPA, evPA (Quack's G_evParquet) and qsPA (G_qsParquet).
SCHEMES = ('flex', 'os', 'ev', 'qs')

#: The two spin-orbital reducible vertices, in the order of
#: `spectra.SpinOrbitalVertices` and of Quack's DIIS vector.
VERTEX_NAMES = ('eh', 'pp')

#: gfortran's MAXVAL of an empty array (-huge(0d0)): the eigenvalue change of
#: an empty pp branch (one occupied spin orbital: no 2h pair; one virtual: no
#: 2p pair), which Quack's max over the two branches then passes over.
EMPTY_MAXVAL = -np.finfo(float).max


@dataclasses.dataclass
class SpinOrbitalParquetResult:
    """What a spin-orbital parquet run returns (Hartree throughout); the
    fields of restricted.ParquetResult over spin orbitals, without its
    singlet and triplet parts (ec_eh_spin, ec_pp_spin), and the spin-orbital
    layout.

    Attributes
    ----------
    scheme : {'flex', 'os', 'ev', 'qs'}
    eqp : ndarray, shape (nso,)
        Quasiparticle energies of the last one-body iteration (Quack's eQP
        after its one-body DIIS; qsPA: the eigenvalues of the last effective
        Fock matrix, ascending).
    sigc : ndarray, shape (nso,) or (nso, nso)
        FLEX, osPA and evPA: SigC_p at w = eOld_p of the last one-body
        iteration (Quack's printed Sig_c). qsPA: the symmetrized static
        self-energy of the last iteration in the final orbitals (its
        diagonal is Quack's printed Sig_c).
    z : ndarray, shape (nso,)
        Renormalization factors (find_qp_root: 1/(1 - dSigma/dw) at the
        root; linearized: the diagonal Z; a qp_solver function: the z it
        returned; with `spin_pairs` the second of a pair takes the first's;
        qsPA: ones).
    eps_hf : ndarray, shape (nso,)
        The reference's spin-orbital energies, in the reference's layout
        (also for qsPA, whose eqp come in the final diagonalization's order).
    nO : int
        Occupied spin orbitals (the first nO of eqp; qsPA: of the ascending
        eigenvalues).
    eqp_lin : ndarray or None
        FLEX, osPA and evPA: eHF + Z SigC of the last iteration (Z the
        diagonal one, as computed: not paired); None for qsPA.
    qp_converged : ndarray of bool or None
        FLEX, osPA and evPA: for 'find_qp_root' whether the spin orbital's
        equation had a root, for a qp_solver function its verdict (None
        when linearized or qsPA).
    qp_followed, qp_flipped : ndarray of bool or None
        'find_qp_root': the house finder's branch report of the last
        one-body iteration per spin orbital, as in restricted.ParquetResult
        (with `spin_pairs` the second of a pair takes the first's; a flip is
        flagged 'parquet_qp_branch_flipped' for both). None for
        'linearized', a qp_solver function and qsPA.
    spectra : spectra.SpinOrbitalSpectra
        The eh, ee and hh channels (Omega, rho) of the last two-body
        iteration, which the last self-energy was built from.
    vertices : spectra.SpinOrbitalVertices
        The reducible vertices after the last two-body iteration (damped and
        extrapolated: Quack's old_*_Phi at the end of the run).
    ec_eh, ec_pp : float
        The eh and pp correlation energies (Quack's Ec_eh and Ec_pp) of the
        last two-body iteration's BSEs, the channel energies of the
        spin-orbital problem (the restricted driver's singlet + 3 triplet).
        Under the TDA (the default) they are half the sum of the eigenvalues
        less the trace, zero in exact arithmetic: rounding noise (0 to 6e-14
        Ha on water/6-31G and OH/STO-3G); a correlation energy only for the
        full BSEs.
    converged : bool
        Every loop that was to converge did (FLEX and osPA have no one-body
        loop to converge, FLEX no two-body loop).
    converged_1b : bool or None
        Quack's verdict on the one-body loop (n_it_1b < max_it_1b), False
        also for a NaN change; None for FLEX and osPA.
    converged_2b : list of (bool or None)
        Quack's verdict on each one-body iteration's two-body loop (None for
        FLEX); False also where its change came out NaN.
    n_it_1b : int
    n_it_2b : list of int
    history : list of dict
        One entry per one-body iteration: 'two_body' (one dict per two-body
        iteration: 'err' the two vertex changes (eh, pp), 'err_2b' their
        max, 'err_eig' Quack's printed eigenvalue changes (eh, pp),
        'diis_rcond', 'seconds'), 'converged_2b', 'n_it_2b', 'om' (the eh,
        ee and hh eigenvalues after the two-body loop, which Quack prints on
        success), 'ec_eh', 'ec_pp' (as the fields), 'eqp', 'sigc'
        (diagonal), 'z', 'eqp_lin', 'qp_solver' and 'qp_n_iter' (FLEX, osPA
        and evPA: the solver of the quasiparticle equation, a name or a
        function's label (`options.qp_solver_label`), and, per spin orbital,
        its evaluations of the self-energy, those of the scan and its
        refinements for 'find_qp_root', those a qp_solver function asked
        for, None for 'linearized'; 0 for a spin orbital that took its
        partner's root, `spin_pairs`), 'qp_followed' and 'qp_flipped' (the fields'
        report of that iteration; None but for 'find_qp_root'), 'err_1b',
        'diis_rcond', 'seconds'. 'diis_rcond' is
        dsysvx's reciprocal condition number of that iteration's DIIS
        system, two-body or one-body (NaN for a non-finite history; None
        without that DIIS), as in restricted.ParquetResult. qsPA records have
        no eqp_lin or qp_* keys.
    times : dict
        Wall seconds per stage ('eh_bse', 'pp_bse', 'eh_integrals',
        'pp_integrals', 'eh_vertex', 'pp_vertex', 'mixing', 'self_energy',
        'qp_equation', 'fock', 'integral_transform', 'integrals', 'total').
    numerical_flags : list of dict
    options : dict
        The options the run used, defaults resolved ('damping' the weight of
        the previous vertices; 'qp_solver' and 'spin_pairs' for FLEX, osPA
        and evPA; 'vertex_builder' by name).
    mo_coeff : ndarray or None
        qsPA: the spin-AO orbitals (2 nao, nso) of the last diagonalization.
    mo_coeff_two_body : ndarray or None
        qsPA: the orbitals the last spectra, vertices and self-energy are
        expressed in (those at the start of the last one-body iteration).
    sigc_ao : ndarray or None
        qsPA: the symmetrized static self-energy of the last iteration in
        the spin-AO basis.
    reference : str or None
        run_parquet_so: 'RHF', 'UHF' or 'GHF'.
    spin, orbital : ndarray or None
        run_parquet_so. FLEX, osPA and evPA: the spin (0 alpha, 1 beta, -1 a
        GHF spinor) and the mean field's own orbital index of each spin
        orbital, in the reference layout that eqp keeps. qsPA: the spin of
        each final orbital (mo_coeff, eqp's order), 0 or 1 where its alpha
        or beta weight c^T S c is 1 to PARQUET_SO_SPIN_WEIGHT_TOL and -1
        where it is mixed (a closed shell's degenerate alpha/beta pairs,
        which the eigensolver rotates at will); orbital None.
    """
    scheme: str
    eqp: np.ndarray
    sigc: np.ndarray
    z: np.ndarray
    eps_hf: np.ndarray
    nO: int
    eqp_lin: object
    qp_converged: object
    spectra: SpinOrbitalSpectra
    vertices: SpinOrbitalVertices
    ec_eh: float
    ec_pp: float
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
    reference: object = None
    spin: object = None
    orbital: object = None
    qp_followed: object = None
    qp_flipped: object = None


class SpinOrbitalReference:
    """A mean field laid out over spin orbitals (`spin_orbital_reference`):
    eps (nso,), occupied first; mo_coeff (2 nao, nso) spin-AO coefficients,
    alpha AO rows first; nO occupied spin orbitals; spin (nso,) 0 alpha,
    1 beta, -1 a GHF spinor; orbital (nso,) the mean field's own index of
    each (per spin for RHF and UHF); ovlp (nao, nao) the AO overlap; hcore
    (nao, nao) the mean field's own core Hamiltonian, mf.get_hcore() (a
    GHF's alpha block), or None where qsPA cannot take it and refuses it: a
    complex core Hamiltonian, one of another shape (a UHF's per-spin
    (2, nao, nao), a GHF's other than (2 nao, 2 nao)), or a GHF's that
    depends on the spin; eri_ao (nao,)*4 the AO integrals <mu nu|la si>;
    kind 'RHF', 'UHF' or 'GHF'."""

    def __init__(self, eps, mo_coeff, nO, spin, orbital, ovlp, hcore, eri_ao, kind):
        self.eps, self.mo_coeff, self.nO = eps, mo_coeff, nO
        self.spin, self.orbital, self.kind = spin, orbital, kind
        self.ovlp, self.hcore, self.eri_ao = ovlp, hcore, eri_ao


# ---------------------------------------------------------------------------
# the two-body iteration


def two_body_spectra_so(eps, eri, nO, vertices, *, first, tda_eh, tda_pp, timer=None):
    """Both BSEs of one two-body iteration and their effective integrals.

    Steps 3 to 5 of the paper's Fig. 8 as G_evParquet and G_qsParquet run
    them: the bare eh blocks on `eps` (phGLR_A/B) plus the static kernel of
    `vertices` (G_eh_Gamma_A/B; Eqs. (35), (47a)), zero when `first`, solved
    (phGLR, Eq. (34)); the same for the pp BSE (ppGLR_B/C/D, G_pp_Gamma_B/C/D,
    ppGLR; Eqs. (40), (41), (47b)); the eigenvectors contracted with the
    irreducible vertices of the same `vertices` (G_eh_screened_integral,
    G_pp_screened_integral; Eqs. (38), (44)).

    Parameters
    ----------
    eps : ndarray, shape (nso,)
        One-body energies of the BSE diagonals (eOld, or eQP in qsPA).
    eri : ndarray, shape (nso,)*4
        <pq|rs> over spin orbitals.
    nO : int
        Occupied spin orbitals.
    vertices : spectra.SpinOrbitalVertices
        The reducible vertices of the previous two-body iteration.
    first : bool
        Quack's first two-body iteration of the run: the kernels are zero.
    tda_eh, tda_pp : bool
        Tamm-Dancoff approximation of the eh and the pp BSE.
    timer : dict, optional
        Accumulates wall seconds under 'eh_bse', 'eh_integrals', 'pp_bse',
        'pp_integrals'.

    Returns
    -------
    spectra : spectra.SpinOrbitalSpectra
        The eh, ee and hh channels (Omega, rho[p, q, n]).
    ec_eh, ec_pp : float
        Quack's Ec_eh and Ec_pp of the two BSEs.
    """
    with stage(timer, 'eh_bse'):
        A, B = eh_bse_blocks_so(eps, eri, nO, tda_eh)
        if not first:
            gam_a, gam_b = eh_kernel_blocks_so(vertices, nO, tda_eh)
            A += gam_a
            if B is not None:
                B += gam_b
            del gam_a, gam_b
        eh = solve_eh_bse_so(A, B, tda_eh)
        del A, B
    with stage(timer, 'eh_integrals'):
        rho_eh = eh_effective_integrals_so(eri, vertices, eh.XpY, eh.XmY, nO)
    with stage(timer, 'pp_bse'):
        B, C, D = pp_bse_blocks_so(eps, eri, nO, tda_pp)
        if not first:
            gam_b, gam_c, gam_d = pp_kernel_blocks_so(vertices, nO, tda_pp)
            if B is not None:
                B += gam_b
            C += gam_c
            D += gam_d
            del gam_b, gam_c, gam_d
        pp = solve_pp_bse_so(B, C, D, tda_pp)
        del B, C, D
    with stage(timer, 'pp_integrals'):
        rho_ee, rho_hh = pp_effective_integrals_so(eri, vertices, pp.X1, pp.Y1, pp.X2, pp.Y2, nO)
    spectra = SpinOrbitalSpectra(Channel(eh.om, rho_eh), Channel(pp.om_ee, rho_ee),
                                 Channel(pp.om_hh, rho_hh))
    return spectra, float(eh.ec), float(pp.ec)


def static_reducible_vertex_so(name, spectra, eta_2b):
    """The reducible vertex `name` ('eh' or 'pp') at zero frequency from the
    BSE spectra: Quack's G_eh_Phi and G_pp_Phi (Eqs. (39), (45), regularized
    as Eq. (52)). The default `vertex_builder` of the drivers. Another
    builder of the same signature returns a real (nso,)*4 array, which the
    loop only reads (it may be cached: the damping writes into the loop's own
    arrays) and refuses in any other shape or type (ValueError); NaN or inf
    in it is kept and flagged ('parquet_vertex_not_finite'). It can express
    a frequency average (spin_orbital_two_body.eh_vertex_from_weights and
    pp_vertex_from_weights take averaged pole weights) only where the average
    does not depend on the orbitals or the quasiparticle energies and is the
    same at every place the vertex is inserted: one tensor per source channel
    is inserted at several places, the eh tensor in both the eh and the pp
    kernels (and their effective integrals), the pp tensor in the eh kernel
    and its effective integrals, and the builder sees neither eps nor nO.
    Weighting each insertion on its own needs a redesign (one tensor per
    insertion place, eps and nO handed to the builder), which
    docs/methods/parquet.md describes; it is not implemented.

    Parameters
    ----------
    name : {'eh', 'pp'}
    spectra : spectra.SpinOrbitalSpectra
        The new eigenvalues and the effective integrals of this iteration.
    eta_2b : float
        The two-body flow parameter s_2b (Hartree^-2).

    Returns
    -------
    phi : ndarray, shape (nso,)*4
    """
    if name == 'eh':
        return eh_reducible_vertex_so(spectra.eh.om, spectra.eh.rho, eta_2b, 0.0)
    if name == 'pp':
        return pp_reducible_vertex_so(spectra.ee.om, spectra.ee.rho, spectra.hh.om,
                                      spectra.hh.rho, eta_2b, 0.0)
    raise ValueError(f'unknown spin-orbital reducible vertex {name!r}; one of {VERTEX_NAMES}')


class TwoBodyState:
    """What Quack's two-body loop carries between its iterations (and, in
    G_evParquet, between one-body iterations): the reducible vertices
    (old_eh_Phi, old_pp_Phi), the previous eigenvalues (old_*_Om, zero at the
    start, only printed) and the two-body DIIS history (None when
    max_diis_2b is 1, which Quack skips; its slots are allocated once, here).
    The two vertex arrays are the state's own: every two-body iteration
    overwrites them in place (the damping without DIIS, the extrapolation
    with it), and only `reset` replaces them."""

    def __init__(self, nso, nO, max_diis_2b):
        self.nso, self.nO = int(nso), int(nO)
        self.max_diis_2b = int(max_diis_2b)
        self.diis = (QuackDIIS(self.max_diis_2b, [(self.nso,) * 4] * len(VERTEX_NAMES),
                               label='two-body') if self.max_diis_2b > 1 else None)
        self.reset()
        self.started = False

    def reset(self):
        """Zero vertices and eigenvalues and an empty DIIS history (Quack's
        initialization, and G_qsParquet's at every one-body iteration)."""
        nV = self.nso - self.nO
        self.phi = None
        self.phi = [np.zeros((self.nso,) * 4) for _ in VERTEX_NAMES]
        self.om = {'eh': np.zeros(self.nO * nV),
                   'ee': np.zeros(pp_pair_count(nV, 'spin-orbital')),
                   'hh': np.zeros(pp_pair_count(self.nO, 'spin-orbital'))}
        if self.diis is not None:
            self.diis.reset()

    def as_vertices(self):
        return SpinOrbitalVertices(*self.phi)


def _rows(a):
    """a as a C-contiguous (len(a), size / len(a)) view, the block rows of the
    passes (explicit sizes: an empty array, the eigenvalues of an empty pp
    branch, is (0, 0))."""
    a = np.ascontiguousarray(a)
    rows = a.shape[0] if a.ndim else 1
    return a.reshape(rows, a.size // rows if rows else 0)


def _row_blocks(rows):
    """Slices of the rows of an (n, m) array, each block at most
    PARQUET_SO_DRIVER_BLOCK_BYTES and 1/PARQUET_MIN_BLOCKS_PER_VERTEX of the
    array (one row at least), so a pass over a vertex never holds a temporary
    of its size."""
    n, m = rows.shape
    block_bytes = min(PARQUET_SO_DRIVER_BLOCK_BYTES, rows.nbytes // PARQUET_MIN_BLOCKS_PER_VERTEX)
    step = max(1, block_bytes // (8 * max(m, 1)))
    return [slice(r0, min(n, r0 + step)) for r0 in range(0, n, step)]


def maxval_abs_diff(a, b):
    """gfortran's maxval(abs(a - b)): the largest |a - b| over the elements
    that are not NaN, NaN when every one is NaN, -huge(0d0) (EMPTY_MAXVAL)
    for an empty array; in blocks, no temporary of the size of a."""
    a, b = _rows(np.asarray(a, dtype=float)), _rows(np.asarray(b, dtype=float))
    if a.shape != b.shape:
        raise ValueError(f'arrays of {a.size} and {b.size} elements: shapes differ')
    if a.size == 0:
        return EMPTY_MAXVAL
    best = np.nan
    with np.errstate(invalid='ignore', over='ignore'):
        for blk in _row_blocks(a):
            best = np.fmax(best, np.fmax.reduce(np.abs(a[blk] - b[blk]), axis=None))
    return float(best)


def quack_max(*values):
    """gfortran's max(a, b, ...) of the errors as the paper build runs it
    (arm64 fmaxnm): NaN arguments are skipped unless every one is NaN."""
    return float(np.fmax.reduce(np.asarray(values, dtype=float)))


def _built_vertex(phi, name, nso):
    """What a vertex builder returned, as a C-contiguous float64 array (no copy
    when it is one already) that the loop only reads; ValueError unless it is
    a real (nso,)*4 array (a complex one is refused, not cast, whatever its
    imaginary part; so is anything numpy cannot read as real numbers, a ragged
    list among them)."""
    try:
        is_complex = np.iscomplexobj(phi)       # numpy reads a ragged list here
        if not is_complex:
            phi = np.ascontiguousarray(phi, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError(f'vertex_builder returned a {type(phi).__name__} for {name!r}, not '
                         f'an array of real numbers ({err}); a spin-orbital reducible vertex '
                         f'is a real {(nso,) * 4} array') from None
    if is_complex:
        raise ValueError(f'vertex_builder returned a complex array for {name!r}; the '
                         f'spin-orbital reducible vertex is real (return its real part if '
                         f'that is the vertex meant)')
    if phi.shape != (nso,) * 4:
        raise ValueError(f'vertex_builder returned an array of shape {phi.shape} for '
                         f'{name!r}; a spin-orbital reducible vertex is {(nso,) * 4}')
    return phi


def _nonfinite_count(a):
    """The number of NaN and inf elements of `a`, in blocks (`_row_blocks`):
    no temporary beyond one block's bools."""
    rows = _rows(a)
    if rows.size == 0:
        return 0
    return sum(int(rows[blk].size - np.count_nonzero(np.isfinite(rows[blk])))
               for blk in _row_blocks(rows))


def _damp_in_place(old, new, alpha):
    """old <- alpha new + (1 - alpha) old as Quack evaluates it,
    fl(fl(alpha new) + fl((1 - alpha) old)), 1 - alpha formed once as Quack's
    (1d0 - alpha), block by block (`_row_blocks`, one scratch block); `new`
    is only read."""
    beta = 1.0 - alpha
    o2, n2 = old.reshape(old.shape[0], old.size // old.shape[0]), _rows(new)
    blocks = _row_blocks(o2)
    scratch = np.empty((blocks[0].stop - blocks[0].start, o2.shape[1]))
    with np.errstate(invalid='ignore', over='ignore'):
        for blk in blocks:
            t = scratch[:blk.stop - blk.start]
            np.multiply(n2[blk], alpha, out=t)
            o2[blk] *= beta
            o2[blk] += t


def _damp_into(damped, err, new, old, alpha):
    """damped <- alpha new + (1 - alpha) old as Quack evaluates it, and
    err <- damped - old: the next DIIS pair, written into its slot without
    another temporary (err holds (1 - alpha) old on the way); `new` and `old`
    are only read."""
    with np.errstate(invalid='ignore', over='ignore'):
        np.multiply(new, alpha, out=damped)
        np.multiply(old, 1.0 - alpha, out=err)
        damped += err
        np.subtract(damped, old, out=err)


def _update_vertices(state, spectra, eta_2b, alpha, vertex_builder, timer):
    """The new vertices of one two-body iteration, damped and extrapolated
    into state.phi; returns the changes max |Phi_new - Phi_old| (eh, pp)
    taken before damping (Quack's err_eh, err_pp).

    Each vertex is built (`vertex_builder`; what it returns is only read,
    refused unless it is a real (nso,)*4 array, and, from a builder other
    than the default, flagged 'parquet_vertex_not_finite' where it holds NaN
    or inf), compared with the old one and damped, one at a time, as Quack
    evaluates Phi = alpha Phi + (1 - alpha) old_Phi (alpha = 1 - damping),
    so only one new nso^4 array exists at once.
    Without DIIS the damped vertex is written over the old one in place, a
    block at a time. With DIIS each damped vertex and its error
    (damped - old) are written straight into the next slot of the two-body
    history, whose two blocks are Quack's vector in its order (eh then pp),
    and the extrapolation is accumulated into the state's two vertices in
    place. The slot is taken once the first vertex has passed its check, so a
    builder refused there leaves the state and its history as they were; one
    refused at the pp vertex leaves the iteration half done (without DIIS
    the eh vertex is already damped into state.phi; with DIIS state.phi is
    untouched, but a full history has lost its oldest pair, whose slot was
    being rewritten; the run ends there anyway)."""
    errors = []
    slot = None
    for k, name in enumerate(VERTEX_NAMES):
        with stage(timer, name + '_vertex'):
            new = _built_vertex(vertex_builder(name, spectra, eta_2b), name, state.nso)
            if vertex_builder is not static_reducible_vertex_so:
                bad = _nonfinite_count(new)
                if bad:
                    flag(FLAG_VERTEX_NOT_FINITE,
                         f'vertex_builder {builder_name(vertex_builder)} returned {bad} NaN or '
                         f'inf element(s) of {new.size} in the {name!r} vertex; kept as returned '
                         f'(the change max |Phi_new - Phi_old| passes over NaN, as gfortran\'s '
                         f'maxval, so the loop can report convergence with them)',
                         ParquetNotFinite, stacklevel=4)
        with stage(timer, 'mixing'):
            old = state.phi[k]
            errors.append(maxval_abs_diff(new, old))
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
            state.diis.extrapolate(out=state.phi)
    return tuple(errors)


def two_body_loop_so(state, eps, eri, nO, opts, timer=None, check=True):
    """Quack's two-body do-while loop on fixed one-body energies `eps` (Sec. V,
    steps 3 to 8; module docstring), updating `state` (TwoBodyState) in place.

    Parameters
    ----------
    state : TwoBodyState
    eps : ndarray, shape (nso,)
        One-body energies of the BSE diagonals.
    eri : ndarray, shape (nso,)*4
        <pq|rs>.
    nO : int
    opts : dict
        'tda_eh', 'tda_pp', 'eta_2b', 'max_it_2b', 'conv_2b', 'alpha' (Quack's
        mixing weight, 1 - damping, `_loop_options`) and 'vertex_builder'
        (`static_reducible_vertex_so` for Quack's).
    timer : dict, optional
    check : bool
        Give Quack's verdict (False for FLEX: None).

    Returns
    -------
    spectra, ec_eh, ec_pp, record
        The spectra and BSE correlation energies of the last iteration, and a
        dict: 'two_body' (the iteration records), 'n_it_2b', 'converged_2b',
        'om' (the eigenvalues after the loop).
    """
    n_it, err_2b = 0, PARQUET_QUACK_INITIAL_ERROR
    iterations = []
    spectra = ec_eh = ec_pp = None
    while err_2b > opts['conv_2b'] and n_it < opts['max_it_2b']:
        n_it += 1
        t0 = time.perf_counter()
        spectra, ec_eh, ec_pp = two_body_spectra_so(
            eps, eri, nO, state.as_vertices(), first=not state.started,
            tda_eh=opts['tda_eh'], tda_pp=opts['tda_pp'], timer=timer)
        state.started = True
        om = {'eh': spectra.eh.om, 'ee': spectra.ee.om, 'hh': spectra.hh.om}
        err_eig = (maxval_abs_diff(state.om['eh'], om['eh']),
                   quack_max(maxval_abs_diff(state.om['ee'], om['ee']),
                             maxval_abs_diff(state.om['hh'], om['hh'])))
        state.om = om
        errors = _update_vertices(state, spectra, opts['eta_2b'], opts['alpha'],
                                  opts['vertex_builder'], timer)
        err_2b = quack_max(*errors)
        iterations.append({'err': errors, 'err_2b': err_2b, 'err_eig': err_eig,
                           'diis_rcond': None if state.diis is None else state.diis.rcond,
                           'seconds': time.perf_counter() - t0})
    converged = None
    if check:
        converged = n_it != opts['max_it_2b']
        if np.isnan(err_2b):
            converged = False
            flag(FLAG_TWO_BODY_NOT_FINITE,
                 f'spin-orbital parquet two-body loop: both reducible vertex changes are NaN '
                 f'at iteration {n_it}; Quack ends the loop there and reports success, the '
                 f'spectra and vertices returned are NaN', ParquetNotFinite,
                 stacklevel=3)
        elif not converged:
            flag(FLAG_TWO_BODY_NOT_CONVERGED,
                 f'spin-orbital parquet two-body loop not converged after {n_it} iterations: '
                 f'max |Phi_new - Phi_old| = {err_2b:.3e} against conv_2b = '
                 f'{opts["conv_2b"]:.1e} (Quack counts n_it == max_it_2b as failed); the run '
                 f'goes on with the last vertices', ParquetNotConverged, stacklevel=3)
    record = {'two_body': iterations, 'n_it_2b': n_it, 'converged_2b': converged,
              'om': dict(state.om)}
    return spectra, ec_eh, ec_pp, record


# ---------------------------------------------------------------------------
# the quasiparticle equation


def _pair_sources(spin_pairs, nso):
    """source[p]: the spin orbital whose quasiparticle root p takes; p itself
    unless p is the second of one of the (alpha, beta) `spin_pairs`."""
    source = np.arange(nso)
    if spin_pairs is not None:
        pairs = np.asarray(spin_pairs, dtype=int).reshape(-1, 2)
        source[pairs[:, 1]] = pairs[:, 0]
    return source


@flags_into_out
def solve_qp_scan_so(eta, eps_hf, eps, eri, nO, spectra, follow=None, spin_pairs=None,
                     out=None):
    """(eqp, z, found): the quasiparticle equation of every spin orbital
    solved by the house's one root finder (qp_solver='find_qp_root'; not
    Quack's, which has no search on the spin-orbital self-energy).

    The equation is f(w) = w - eps_hf[p] - Sigma_pp(w), Sigma_pp
    G_Parquet_self_energy_diag with eps[p] -> w in its w-dependent
    denominators (spin_orbital_self_energy.DiagonalSelfEnergySO.at) and
    every other denominator on eps (eOld), handed to
    src.Solvers.qp_equation.solve_qp_equation_scan about the bare level
    a = eps_hf[p] with that finder's own tolerances. Without `follow` it
    takes the scan windows about a narrowest first and keeps the root of
    largest valid weight (0 < Z <= 1) of the first window where that weight
    exceeds QP_WINDOW_Z_MIN; failing that, the widest window's largest
    valid weight, or, with no valid weight there, the root nearest a. With
    `follow`, the roots of the previous evPA iteration, it keeps the root
    continuing the one followed, the root of valid weight nearest follow[p]
    on the scan of the narrowest window about a that holds it, however small
    its weight: the largest-weight rule can jump between roots of similar
    weight from one iteration to the next and keep evPA from converging (the
    restricted driver's ammonia/6-31G N 1s at s_1b = 50 and s_2b = 1, roots
    of weight 0.70 and 0.58, cycled with period 3; this driver's product-form
    self-energy did not jump there, with or without following). A spin
    orbital whose follow[p] is not finite (NaN) is solved as without
    `follow`. The drivers hand NaN for one whose previous root has Z outside
    (0, 1] (`options.roots_to_follow`): the finder follows valid weights
    only, so it would follow such a root to the valid root nearest it where
    the windows about it hold one, and otherwise fall back, after the scan
    about it, to the search without `follow` (ammonia's virtuals of Z = 1.04
    at s_1b = 50, whose windows hold no valid weight). Each bracket is
    refined by a Newton safeguarded by it on the slope f'(w) =
    1 - dSigma_pp/dw (the regularizers held constant, as Quack's Z), and
    z = 1/f' at the root.
    positive_weight is off: the parquet self-energy has residues of both
    signs, so a bracket where f falls may hold a root. The pole-sum route
    (`PoleSum`) is never taken: its largest weight over the whole axis picks
    roots far from the quasiparticle here.

    An orbital without a root in the widest window is NaN, with the
    finder's QPNoRoot warning and named flag 'no_root', and found[p] False;
    a root kept with z outside (0, 1] carries the finder's QPWeightOutside
    warning and named flag 'z_outside'; a root followed that the finder
    reports 'flipped' is warned as the house warns for a followed root
    (RuntimeWarning, qp_equation.branch_warning) and flagged
    'parquet_qp_branch_flipped' (`options.flag_branch`: the branch has gone,
    or a root of larger weight lies in the scan window of the root followed,
    every bracket of which the finder refines). With
    `spin_pairs` the first of each pair is solved and its root, z and
    verdict copied to the second, and the same warnings and flags are
    raised for the second, so that the flags name both.

    Parameters
    ----------
    eta : float
        One-body regularization s_1b (eta_1b).
    eps_hf : ndarray, shape (nso,)
        The anchor of the equation (Quack's eHF).
    eps : ndarray, shape (nso,)
        Energies of the self-energy's denominators (eOld).
    eri : ndarray, shape (nso,)*4
    nO : int
    spectra : spectra.SpinOrbitalSpectra
    follow : ndarray, shape (nso,), optional
        The roots to follow, one per spin orbital, NaN (not finite) for one
        not followed, whose largest weight is kept; None follows none.
    spin_pairs : ndarray of int, shape (npair, 2), optional
        (alpha, beta) partners, the second taking the first's root (checked
        by `ev_parquet_so`); None solves every spin orbital.
    out : dict, optional
        Receives 'n_eval' (evaluations of Sigma_pp per spin orbital, 0 for one
        that took its partner's root), and one bool per spin orbital in
        'no_root', 'z_outside', 'followed' (the branch followed continued)
        and 'flipped' (the branch followed was not continued and a root of
        largest weight was kept again, or the finder found a root of more
        weight than the one kept: QPRoot.flipped; an equation left without a
        root is 'no_root' instead), the finder's `out` record, and
        'numerical_flags', the flags raised.

    Returns
    -------
    eqp, z : ndarray, shape (nso,)
    found : ndarray of bool
        Whether the spin orbital's equation had a root.

    Cost: 150 to 10 000 evaluations of Sigma_pp per spin orbital solved
    without `follow` on water and ammonia 6-31G, one evaluation serving f
    and f' at the same w: 6 to 36 s per one-body step at nso = 26 to 30.
    Following a root of valid weight scans the cells about it and then the
    rest of its scan window, every bracket refined for the branch report, as
    the search without `follow` refines that window: water's evPA at
    s_1b = 50 and s_2b = 1 takes 11 900 to 14 300 evaluations per iteration
    after the first (11 100), as many as solving every spin orbital without
    `follow`. A root of Z outside (0, 1] that the finder cannot follow would
    cost the scan about it and then the whole search, twice the evaluations
    of that spin orbital, which the drivers save by not following it:
    ammonia's evPA at s_1b = 50 and s_2b = 1, whose e pair of virtuals keeps
    Z = 1.04, takes 22 000 to
    22 600 evaluations per iteration from the third on, against 35 600 to
    36 600 following it (the same roots, bitwise). Where the Z came back
    into (0, 1] at the next iteration, following costs what the search
    does: water's spin orbital 22 (Z 1.106 at the first iteration, 0.927 at
    the second) takes 1 937 evaluations at the second either way (the same
    root).
    """
    eps_hf = np.asarray(eps_hf, dtype=float)
    nso = len(np.asarray(eps))
    if eps_hf.shape != (nso,):
        raise ValueError(f'eps_hf has shape {eps_hf.shape}, expected ({nso},)')
    if follow is not None:
        follow = np.asarray(follow, dtype=float)
        if follow.shape != (nso,):
            raise ValueError(f'follow has shape {follow.shape}, expected ({nso},)')
    source = _pair_sources(spin_pairs, nso)
    eqp, z = np.full(nso, np.nan), np.full(nso, np.nan)
    n_eval = np.zeros(nso, dtype=int)
    no_root, z_out, followed, flipped = (np.zeros(nso, dtype=bool) for _ in range(4))
    records = {}
    for p in np.flatnonzero(source == np.arange(nso)):
        se = DiagonalSelfEnergySO(eta, eps, eri, nO, spectra, orbitals=[p])
        residual, slope, count = _qp_equation_so(se, int(p), float(eps_hf[p]))
        record = {}
        label = f'the spin-orbital parquet quasiparticle equation of spin orbital {p} (0-based)'
        w_follow = (None if follow is None or not np.isfinite(follow[p]) else float(follow[p]))
        eqp[p], z[p] = solve_qp_equation_scan(
            residual, slope, float(eps_hf[p]), follow=w_follow, label=label, out=record,
            positive_weight=False)
        n_eval[p] = count[0]
        no_root[p], z_out[p] = record['no_root'], record['z_outside']
        followed[p] = bool(record['followed'])
        if w_follow is not None:
            flipped[p] = flag_branch(label, eqp[p], z[p], record, w_follow)
        records[int(p)] = (record, w_follow)
    copied = np.flatnonzero(source != np.arange(nso))
    for arr in (eqp, z, no_root, z_out, followed, flipped):
        arr[copied] = arr[source[copied]]
    for p in copied:
        # the finder flagged the first of the pair; the second carries the same root
        record, w_follow = records[int(source[p])]
        if w_follow is not None:
            flag_branch(f'the spin-orbital parquet quasiparticle equation of spin orbital {p} '
                        f'(0-based), taken from its spin partner {int(source[p])}', eqp[p],
                        z[p], record, w_follow)
        if no_root[p]:
            flag(FLAG_NO_ROOT,
                 f'the spin-orbital parquet quasiparticle equation of spin orbital {p} '
                 f'(0-based) takes the root of its spin partner {int(source[p])}, which has no '
                 f'root in the widest scan window: NaN', QPNoRoot)
        elif z_out[p]:
            flag(FLAG_Z_OUTSIDE,
                 f'the quasiparticle root of the spin-orbital parquet quasiparticle equation of '
                 f'spin orbital {p} (0-based), taken from its spin partner {int(source[p])}, '
                 f'{float(eqp[p])!r} Ha, has Z = {float(z[p])!r}, outside (0, 1], which no root '
                 f'of a positive-weight self-energy has: kept', QPWeightOutside)
    if out is not None:
        out.update(n_eval=n_eval, no_root=no_root, z_outside=z_out, followed=followed,
                   flipped=flipped)
    return eqp, z, ~no_root


def _qp_equation_so(se, p, a):
    """(f, f', count) of spin orbital p's equation f(w) = w - a - Sigma_pp(w)
    for the root finder: f and f' at one w share one evaluation of `se.at`,
    and count[0] counts the evaluations."""
    memo = {'w': None}
    count = [0]

    def sigma(w):
        w = float(w)
        if memo['w'] != w:
            memo['sig'], memo['dsig'] = se.at(p, w)
            memo['w'] = w
            count[0] += 1
        return memo['sig'], memo['dsig']

    def f(w):
        return w - a - sigma(w)[0]

    def fprime(w):
        return 1.0 - sigma(w)[1]

    return f, fprime, count


# ---------------------------------------------------------------------------
# FLEX, osPA and evPA: G_evParquet


def ev_parquet_so(eps_hf, eri, nO, scheme='os', *, tda_eh=True, tda_pp=True,
                  eta_1b=PARQUET_REGULARIZATION, eta_2b=PARQUET_REGULARIZATION,
                  max_it_1b=None, conv_1b=PARQUET_CONVERGENCE, max_it_2b=None,
                  conv_2b=PARQUET_CONVERGENCE, max_diis_1b=PARQUET_MAX_DIIS_1B,
                  max_diis_2b=PARQUET_MAX_DIIS_2B, qp_solver=DEFAULT_QP_SOLVER,
                  damping=PARQUET_SO_EV_VERTEX_DAMPING, vertex_builder=None, spin_pairs=None):
    """FLEX, osPA or evPA over spin orbitals: Quack's G_evParquet.

    Runs G_evParquet's one-body loop (module docstring): at every one-body
    iteration the two-body loop to convergence on eOld (eps_hf at the first),
    then the diagonal self-energy and Z at w = eOld (G_Parquet_self_energy_diag,
    `diagonal_self_energy_so`), eQPlin = eps_hf + Z SigC, and eQP from the
    quasiparticle equation as `qp_solver` says, then the one-body DIIS
    (max_diis_1b > 1, error eQP - eOld), the change max |eOld - eQP|, and
    eOld = eQP.

    Parameters
    ----------
    eps_hf : ndarray, shape (nso,)
        Hartree-Fock spin-orbital energies, occupied first.
    eri : ndarray, shape (nso,)*4
        <pq|rs> in those spin orbitals (Quack's ERI_MO).
    nO : int
        Occupied spin orbitals (1 and nso - 1 included: G_evParquet runs the
        empty pp branch).
    scheme : {'flex', 'os', 'ev'}
        'flex': max_it_1b = max_it_2b = 1; 'os': max_it_1b = 1; 'ev': one-body
        self-consistency.
    tda_eh, tda_pp : bool
        Tamm-Dancoff approximation of the eh and pp BSEs (Quack's TDAeh,
        TDApp).
    eta_1b : float
        One-body SRG flow parameter s_1b (Hartree^-2) of the self-energy,
        Eq. (53); large is weakly regularized (1e8 is effectively none).
    eta_2b : float
        Two-body flow parameter s_2b of the reducible vertices, Eq. (52).
    max_it_1b : int, optional
        One-body iteration cap; default 1 for 'flex' and 'os' (the only value
        they take) and PARQUET_MAX_IT_1B for 'ev'.
    conv_1b : float
        One-body threshold on max |eOld - eQP| (Hartree), in (0, 1).
    max_it_2b : int, optional
        Two-body iteration cap; default 1 for 'flex' (the only value it
        takes) and PARQUET_MAX_IT_2B otherwise.
    conv_2b : float
        Two-body threshold on max |Phi_new - Phi_old|, in (0, 1).
    max_diis_1b, max_diis_2b : int
        DIIS lengths (1: none, Quack's default).
    qp_solver : {'find_qp_root', 'linearized'} or callable
        How eQP comes from the quasiparticle equation w = eHF_p + Sigma_pp(w)
        (`options.QP_SOLVERS`, or a function): 'find_qp_root', the default
        (`options.DEFAULT_QP_SOLVER`), the house's one root finder on that
        equation (`solve_qp_scan_so`; not Quack's, 150 to 10 000 evaluations
        of the self-energy per spin orbital without following), which keeps
        the root of largest valid weight about eHF_p at the first one-body
        iteration and from the second on follows the previous iteration's
        root (the spin orbital's previous eQP, the root of valid weight
        nearest it), so that evPA does not jump between roots of similar
        weight; a spin orbital whose previous root has Z outside (0, 1] is
        not followed but searched as at the first iteration
        (`options.roots_to_follow`), and a branch the finder reports the
        largest weight has left is warned and flagged
        ('parquet_qp_branch_flipped'); 'linearized', eQP = eQPlin,
        G_evParquet's own and only branch, the Fortran reference of this
        driver. A function, called at every one-body iteration as
        qp_solver(sigma, eps_hf, eqp_lin, z_lin, orbitals) -> (eqp, z,
        converged), the contract of the closed-shell driver
        (`options.checked_qp_solver`): sigma(p, w) gives (Sigma_pp(w),
        dSigma_pp/dw) of spin orbital p at w, the other denominators on
        eOld, each call counted in 'qp_n_iter' (it builds the residues of
        one spin orbital at a time); eps_hf, eqp_lin and z_lin are the
        bare levels, eQPlin and the diagonal Z (nso,); orbitals are the
        spin orbitals to solve, with `spin_pairs` the first of each pair,
        whose eqp, z and verdict the second takes; it returns three (nso,)
        arrays, eqp and z real and converged bool, which are checked, kept
        as returned and recorded with the function's label
        (`options.qp_solver_label`); a finite energy not converged is
        flagged 'parquet_qp_not_converged', one not finite
        'parquet_qp_not_finite'. G_evParquet has no search on its
        self-energy (it stops unless the equation is linearized). Every
        route flags an energy kept with Z outside (0, 1] ('z_outside'):
        the finder's root, the linearized energy, a function's energy. None
        is refused here: only run_parquet_so takes it, as 'find_qp_root'.
    damping : float or None
        The weight of the previous vertices, as in evGW,
        Phi <- (1 - damping) Phi_new + damping Phi_old, in [0, 1) (1 would
        freeze them); None is the default PARQUET_SO_EV_VERTEX_DAMPING = 0,
        Quack's alpha = 1 - damping = 1 of G_evParquet (no damping). The loop
        damps first and, for max_diis_2b > 1, extrapolates the damped
        vertices by DIIS, with error Phi_damped - Phi_old, as Quack does.
        The default is Quack's so that the comparisons with Quack hold, not
        a production setting: undamped and without DIIS, the two-body loop
        at s_2b = 50 diverges (osPA of water, ammonia and OH in 6-31G: a
        change of 1e122 to 1e128 after 200 iterations, flagged
        'parquet_two_body_not_converged', the energies returned
        meaningless); give max_diis_2b=5, and damping=0.75 with it, or a
        smaller eta_2b (run_parquet_so has the measured cases).
    vertex_builder : callable, optional
        (name, spectra, eta_2b) -> Phi of the 'eh' or 'pp' vertex, a (nso,)*4
        float64 array the loop only reads (any other shape or type is
        refused; NaN or inf in it is flagged 'parquet_vertex_not_finite');
        default `static_reducible_vertex_so` (omega = 0, Quack's), whose
        docstring says what another builder can express.
    spin_pairs : array of int, shape (npair, 2), optional
        The (alpha, beta) spin orbitals of each spatial orbital of a
        spin-symmetric reference (run_parquet_so passes them for an RHF):
        the quasiparticle equation is solved once per pair, for the first,
        and its eQP and Z copied to the second, with every `qp_solver`
        ('linearized': the first one's eQPlin and Z), so the pair stays
        degenerate through evPA (module docstring). Each index at most once,
        both of a pair occupied or both virtual, eps_hf equal within a pair;
        the caller vouches that eri is spin-symmetric. None (the default)
        solves every spin orbital.

    Returns
    -------
    SpinOrbitalParquetResult
    """
    if scheme not in ('flex', 'os', 'ev'):
        raise ValueError(f'ev_parquet_so runs FLEX, osPA and evPA (scheme flex, os, ev), not '
                         f'{scheme!r}; qsPA is qs_parquet_so')
    eps_hf = np.array(eps_hf, dtype=float)
    nso = _check_integrals(eps_hf, eri, nO)
    opts = _loop_options(scheme, tda_eh, tda_pp, eta_1b, eta_2b, max_it_1b, conv_1b,
                         max_it_2b, conv_2b, max_diis_1b, max_diis_2b, damping, vertex_builder)
    opts['qp_solver'] = checked_qp_solver(qp_solver)
    opts['spin_pairs'] = _checked_spin_pairs(spin_pairs, eps_hf, int(nO))
    flags, timer = [], {}
    t_start = time.perf_counter()
    with collecting(flags):
        result = _ev_loops(eps_hf, np.asarray(eri, dtype=float), int(nO), nso, scheme,
                           opts, timer)
    timer['total'] = time.perf_counter() - t_start
    result.numerical_flags = gathered(flags)
    result.times = timer
    return result


def _ev_loops(eps_hf, eri, nO, nso, scheme, opts, timer):
    state = TwoBodyState(nso, nO, opts['max_diis_2b'])
    diis_1b = (QuackDIIS(opts['max_diis_1b'], [(nso,)], label='evPA one-body')
               if opts['max_diis_1b'] > 1 else None)
    e_old = eps_hf.copy()
    n_it_1b, err_1b = 0, PARQUET_QUACK_INITIAL_ERROR
    history = []
    while err_1b > opts['conv_1b'] and n_it_1b < opts['max_it_1b']:
        n_it_1b += 1
        t0 = time.perf_counter()
        spectra = None          # the previous iteration's integrals, before new ones exist
        spectra, ec_eh, ec_pp, record = two_body_loop_so(state, e_old, eri, nO, opts, timer,
                                                         check=scheme != 'flex')
        with stage(timer, 'self_energy'):
            sigc, z = diagonal_self_energy_so(opts['eta_1b'], e_old, eri, nO, spectra)
        eqp_lin = eps_hf + z * sigc
        with stage(timer, 'qp_equation'):
            # find_qp_root follows the previous iteration's roots (eOld) from the second on,
            # each where its Z lay in (0, 1]
            follow = None
            if _is_named(opts['qp_solver'], 'find_qp_root') and n_it_1b > 1:
                follow = roots_to_follow(e_old, history[-1]['z'])
            eqp, z, qp_converged, qp_n_iter, followed, flipped = _solve_qp_so(
                opts['qp_solver'], opts['eta_1b'], eps_hf, eqp_lin, z, e_old, eri, nO, spectra,
                opts['spin_pairs'], follow)
        if diis_1b is not None:
            eqp = diis_1b.push(eqp - e_old, eqp)
        err_1b = maxval_abs_diff(e_old, eqp)
        e_old = np.array(eqp, dtype=float)
        record.update(ec_eh=ec_eh, ec_pp=ec_pp, eqp=e_old.copy(), sigc=sigc.copy(),
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


def _solve_qp_so(solver, eta, eps_hf, eqp_lin, z, e_old, eri, nO, spectra, spin_pairs, follow):
    """(eqp, z, qp_converged, qp_n_iter, qp_followed, qp_flipped) of one
    FLEX/osPA/evPA one-body step: the quasiparticle equation of the diagonal
    self-energy on e_old solved by `solver` (a name of `options.QP_SOLVERS`
    or a qp_solver function, `options.checked_qp_solver`), 'find_qp_root'
    following the roots `follow` (`options.roots_to_follow` of the previous
    iteration's eQP) when given; qp_n_iter counts the self-energy
    evaluations per spin orbital (None when linearized); qp_followed and
    qp_flipped are the finder's branch report per spin orbital (None for
    'linearized' and a function, which follow nothing). With `spin_pairs`
    the second of each pair takes the first's eqp and z, whatever the
    solver; a function sees one spin orbital's self-energy at a time, built
    when it first asks for that spin orbital (`DiagonalSelfEnergySO` of that
    orbital alone). Every route flags an energy kept with Z outside (0, 1]
    ('z_outside'): the linearized one here, a function's in
    `options.solve_with` (with its verdict), the root finder's in its own
    solver, where a flipped branch is flagged too
    ('parquet_qp_branch_flipped')."""
    if _is_named(solver, 'linearized'):
        if spin_pairs is None:
            eqp, z = eqp_lin.copy(), z
        else:
            source = _pair_sources(spin_pairs, len(eqp_lin))
            eqp, z = eqp_lin[source], np.asarray(z, dtype=float)[source]
        _flag_linearized_outside(eqp, z)
        return eqp, z, None, None, None, None
    if _is_named(solver, 'find_qp_root'):
        out = {}
        eqp, z, found = solve_qp_scan_so(eta, eps_hf, e_old, eri, nO, spectra, follow=follow,
                                         spin_pairs=spin_pairs, out=out)
        return eqp, z, found, out['n_eval'], out['followed'], out['flipped']
    built = {'p': None, 'se': None}

    def sigma(p, w):
        p = int(p)
        if built['p'] != p:
            built['p'], built['se'] = p, None           # one orbital's residues at a time
            built['se'] = DiagonalSelfEnergySO(eta, e_old, eri, nO, spectra, orbitals=[p])
        return built['se'].at(p, w)

    eqp, z, converged, n_eval = solve_with(solver, sigma, eps_hf, eqp_lin, z,
                                           _pair_sources(spin_pairs, len(eps_hf)),
                                           'spin orbital')
    return eqp, z, converged, n_eval, None, None


def _flag_linearized_outside(eqp, z):
    """Warn (src.Solvers.qp_equation.QPWeightOutside) and raise the named flag
    'z_outside' for each spin orbital whose linearized energy, the solution
    qp_solver='linearized' keeps (with spin pairs the first one's, for both),
    is finite and has its diagonal Z outside (0, 1]; the energies stay as
    computed."""
    eqp, z = np.asarray(eqp, dtype=float), np.asarray(z, dtype=float)
    with np.errstate(invalid='ignore'):
        outside = np.isfinite(eqp) & ~((z > 0.0) & (z <= 1.0))
    for p in np.flatnonzero(outside):
        flag(FLAG_Z_OUTSIDE,
             f'the linearized spin-orbital parquet quasiparticle energy of spin orbital {p} '
             f'(0-based), {float(eqp[p])!r} Ha, has the diagonal Z = {float(z[p])!r}, outside '
             f'(0, 1], which no root of a positive-weight self-energy has: kept, as Quack '
             f'keeps it', QPWeightOutside, stacklevel=4)


# ---------------------------------------------------------------------------
# qsPA: G_qsParquet


def qs_parquet_so(eps_hf, c_hf, s, x, hc, eri_ao, nO, *, tda_eh=True, tda_pp=True,
                  eta_1b=PARQUET_REGULARIZATION, eta_2b=PARQUET_REGULARIZATION,
                  max_it_1b=PARQUET_MAX_IT_1B, conv_1b=PARQUET_CONVERGENCE,
                  max_it_2b=PARQUET_MAX_IT_2B, conv_2b=PARQUET_CONVERGENCE,
                  max_diis_1b=PARQUET_MAX_DIIS_1B, max_diis_2b=PARQUET_MAX_DIIS_2B,
                  damping=PARQUET_VERTEX_DAMPING, vertex_builder=None):
    """qsPA over spin orbitals from AO integrals and a Hartree-Fock start:
    Quack's G_qsParquet.

    Every one-body iteration (module docstring): the spin-AO Fock matrix of
    the current density (`ghf_fock`), the spin-orbital integrals in the
    current orbitals (`spin_orbital_eri`), the two-body state reset and its
    loop run on the current eQP, the static self-energy matrix
    (G_Parquet_self_energy_SRG_dp, `static_self_energy_srg_so`; not
    R_qsParquet's SRG_sp, so a closed shell does not reproduce the
    restricted qsPA) symmetrized and taken to the AO basis,
    S C Sigma (S C)^T, added to F; the change max |F P S - S P F| before
    DIIS; DIIS on F (error F P S - S P F) for max_diis_1b > 1; X^T F X
    diagonalized (dsyev, PARQUET_QS_EIGH_DRIVER) to the new eQP and C = X C'.
    No quasiparticle equation is solved, so there is no `qp_solver`.

    Parameters
    ----------
    eps_hf : ndarray, shape (nso,)
        Hartree-Fock spin-orbital energies (Quack's eHF), the first eQP.
    c_hf : ndarray, shape (2 nao, nso)
        Their spin-AO coefficients, alpha AO rows first, occupied columns
        first (Quack's cHF).
    s : ndarray, shape (nao, nao)
        AO overlap (Quack's Ov).
    x : ndarray, shape (nao, nmo), or None
        Orthogonalizer of the AO basis, x^T s x = 1, 2 nmo = nso (Quack's Or);
        None: Quack's canonical orthogonalization (`orthogonalization_matrix`).
        The energies and the density do not depend on the choice (to
        rounding: 2e-14 Ha and 5e-15 on water/6-31G against Loewdin's
        S^-1/2); the orbitals do, by column signs and by rotations within
        degenerate levels (a closed shell's alpha/beta pairs), and so do the
        spin labels run_parquet_so derives from them (`result.spin`).
    hc : ndarray, shape (nao, nao)
        Core Hamiltonian (Quack's Hc), on both spin blocks: the mean field's
        own (run_parquet_so takes mf.get_hcore(), so that the first Fock
        matrix is the reference's).
    eri_ao : ndarray, shape (nao,)*4
        <mu nu|la si>, AO, physicist (Quack's ERI_AO).
    nO : int
        Occupied spin orbitals.
    tda_eh, tda_pp, eta_1b, eta_2b, max_it_1b, conv_1b, max_it_2b, conv_2b,
    max_diis_1b, max_diis_2b, vertex_builder
        As for ev_parquet_so (eta_1b is the s of the SRG static self-energy;
        conv_1b bounds the commutator error).
    damping : float or None
        The weight of the previous vertices, as for ev_parquet_so; None is
        the default PARQUET_VERTEX_DAMPING = 0.75, Quack's alpha =
        1 - damping = 0.25 of G_qsParquet.

    Returns
    -------
    SpinOrbitalParquetResult
        With mo_coeff, mo_coeff_two_body and sigc_ao.
    """
    eps_hf = np.array(eps_hf, dtype=float)
    c_hf = np.array(c_hf, dtype=float)
    s = np.asarray(s, dtype=float)
    hc = np.asarray(hc, dtype=float)
    eri_ao = np.asarray(eri_ao, dtype=float)
    nao = s.shape[0]
    nso = eps_hf.shape[0]
    if (eps_hf.ndim != 1 or c_hf.shape != (2 * nao, nso) or s.shape != (nao, nao)
            or hc.shape != (nao, nao) or eri_ao.shape != (nao,) * 4):
        raise ValueError(f'eps_hf {eps_hf.shape}, c_hf {c_hf.shape}, s {s.shape}, hc '
                         f'{hc.shape}, eri_ao {eri_ao.shape}: expected (nso,), (2 nao, nso), '
                         f'(nao, nao), (nao, nao), (nao,)*4')
    nO = checked_occupied(nO, nso, 'occupied spin orbitals', 'nso')
    x = orthogonalization_matrix(s) if x is None else np.asarray(x, dtype=float)
    if x.shape[0] != nao or 2 * x.shape[1] != nso:
        raise ValueError(f'the orthogonalizer has shape {x.shape}: its {x.shape[1]} columns '
                         f'span {2 * x.shape[1]} spin orbitals, the reference has {nso}')
    opts = _loop_options('qs', tda_eh, tda_pp, eta_1b, eta_2b, max_it_1b, conv_1b, max_it_2b,
                         conv_2b, max_diis_1b, max_diis_2b, damping, vertex_builder)
    flags, timer = [], {}
    t_start = time.perf_counter()
    with collecting(flags):
        result = _qs_loops(eps_hf, c_hf, _spin_blocks(s), _spin_blocks(x), hc, eri_ao, int(nO),
                           opts, timer)
    timer['total'] = time.perf_counter() - t_start
    result.numerical_flags = gathered(flags)
    result.times = timer
    return result


def _qs_loops(eps_hf, c_hf, s2, x2, hc, eri_ao, nO, opts, timer):
    nso = c_hf.shape[1]
    state = TwoBodyState(nso, nO, opts['max_diis_2b'])
    diis_1b = (QuackDIIS(opts['max_diis_1b'], [s2.shape], label='qsPA one-body')
               if opts['max_diis_1b'] > 1 else None)
    eqp, c = eps_hf.copy(), c_hf.copy()
    p = c[:, :nO] @ c[:, :nO].T
    n_it_1b, err_1b = 0, PARQUET_QUACK_INITIAL_ERROR
    history = []
    eri = None
    while err_1b > opts['conv_1b'] and n_it_1b < opts['max_it_1b']:
        n_it_1b += 1
        t0 = time.perf_counter()
        with stage(timer, 'fock'):
            fock = ghf_fock(hc, eri_ao, p)
        with stage(timer, 'integral_transform'):
            eri = spectra = None    # the previous orbitals' integrals and spectra
            eri = spin_orbital_eri(eri_ao, c)
        state.reset()
        c_two_body = c
        spectra, ec_eh, ec_pp, record = two_body_loop_so(state, eqp, eri, nO, opts, timer)
        with stage(timer, 'self_energy'):
            sigc, z = static_self_energy_srg_so(opts['eta_1b'], eqp, eri, nO, spectra)
        with stage(timer, 'fock'):
            sigc = 0.5 * (sigc + sigc.T)
            sc = s2 @ c
            sigc_ao = sc @ (sigc @ sc.T)
            fock = fock + sigc_ao
            err_f = fock @ (p @ s2) - (s2 @ p) @ fock
            err_1b = maxval_abs_diff(err_f, np.zeros_like(err_f))
            if diis_1b is not None:
                fock = diis_1b.push(err_f, fock)
            eqp, cp = _diagonalize_fock(x2.T @ (fock @ x2), n_it_1b)
            c = x2 @ cp
            sigc = c.T @ (sigc_ao @ c)
            p = c[:, :nO] @ c[:, :nO].T
        record.update(ec_eh=ec_eh, ec_pp=ec_pp, eqp=eqp.copy(), sigc=np.diag(sigc).copy(),
                      z=np.array(z, dtype=float), err_1b=err_1b,
                      diis_rcond=None if diis_1b is None else diis_1b.rcond,
                      seconds=time.perf_counter() - t0)
        history.append(record)
    converged_1b = _one_body_verdict('qsPA', n_it_1b, err_1b, opts, 'max |F P S - S P F|',
                                     'the run returns its last iterate')
    result = _result('qs', history, eqp, sigc, np.array(z, dtype=float), eps_hf, nO, None,
                     None, spectra, state, ec_eh, ec_pp, converged_1b, n_it_1b, opts)
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


def _spin_blocks(a):
    """[[a, 0], [0, a]]: an AO matrix on both spin blocks (Quack's super
    overlap and orthogonalizer)."""
    a = np.asarray(a, dtype=float)
    n, m = a.shape
    out = np.zeros((2 * n, 2 * m))
    out[:n, :m] = a
    out[n:, m:] = a
    return out


def orthogonalization_matrix(s):
    """Quack's orthogonalization_matrix: the canonical orthogonalization
    X = U s^-1/2 over the overlap eigenvalues above
    PARQUET_ORTHOGONALIZATION_THRESHOLD, in ascending order (dsyev 'U',
    PARQUET_QS_EIGH_DRIVER, as Quack's diagonalize_matrix), so X^T S X = 1.

    Parameters
    ----------
    s : ndarray, shape (nao, nao)
        AO overlap.

    Returns
    -------
    x : ndarray, shape (nao, nmo)
    """
    u, vecs = la.eigh(np.asarray(s, dtype=float), lower=False, driver=PARQUET_QS_EIGH_DRIVER)
    keep = u > PARQUET_ORTHOGONALIZATION_THRESHOLD
    return vecs[:, keep] / np.sqrt(u[keep])


def ghf_fock(hc, eri_ao, p):
    """The spin-AO Fock matrix G_qsParquet builds at each one-body iteration,

        F = [[Hc + J[Paa] + J[Pbb] + K[Paa],  K[Pba]                        ],
             [K[Pab],                         Hc + J[Pbb] + J[Paa] + K[Pbb]]],

    J[P]_mu,nu = sum P_la,si <mu la|nu si> and K[P]_mu,nu = - sum P_la,si
    <mu la|si nu> (Quack's Hartree_matrix_AO_basis and
    exchange_matrix_AO_basis, K with its minus sign), Pab the (alpha rows,
    beta columns) block of the spin-AO density P = C_occ C_occ^T.

    Parameters
    ----------
    hc : ndarray, shape (nao, nao)
    eri_ao : ndarray, shape (nao,)*4
        <mu nu|la si>.
    p : ndarray, shape (2 nao, 2 nao)

    Returns
    -------
    ndarray, shape (2 nao, 2 nao)

    Cost: six contractions of nao^4 multiply-adds.
    """
    hc = np.asarray(hc, dtype=float)
    eri_ao = np.asarray(eri_ao, dtype=float)
    p = np.asarray(p, dtype=float)
    nao = hc.shape[0]
    if p.shape != (2 * nao, 2 * nao) or eri_ao.shape != (nao,) * 4:
        raise ValueError(f'hc {hc.shape}, eri_ao {eri_ao.shape}, p {p.shape}: expected '
                         f'(nao, nao), (nao,)*4 and (2 nao, 2 nao)')

    def coulomb(d):
        return np.tensordot(eri_ao, d, axes=([1, 3], [0, 1]))

    def exchange(d):
        return -np.tensordot(eri_ao, d, axes=([1, 2], [0, 1]))

    a, b = slice(0, nao), slice(nao, 2 * nao)
    j_aa, j_bb = coulomb(p[a, a]), coulomb(p[b, b])
    fock = np.empty((2 * nao, 2 * nao))
    fock[a, a] = hc + j_aa + j_bb + exchange(p[a, a])
    fock[a, b] = exchange(p[b, a])
    fock[b, a] = exchange(p[a, b])
    fock[b, b] = hc + j_bb + j_aa + exchange(p[b, b])
    return fock


def spin_orbital_eri(eri_ao, c):
    """<PQ|RS> in the spin orbitals of the spin-AO coefficients c: Quack's
    AOtoMO_ERI_GHF summed over the four spin blocks (Ca, Ca), (Ca, Cb),
    (Cb, Ca), (Cb, Cb),

        <pq|rs> = sum_{s, t} sum <mu nu|la si> C^s_mu,p C^t_nu,q C^s_la,r C^t_si,s,

    s the spin of electron 1 (p, r), t of electron 2 (q, s).

    Parameters
    ----------
    eri_ao : ndarray, shape (nao,)*4
        <mu nu|la si>.
    c : ndarray, shape (2 nao, nso)
        Spin-AO coefficients, alpha AO rows first.

    Returns
    -------
    ndarray, shape (nso,)*4

    Cost: 4 nso nao^4 + 4 nso^2 nao^3 + 4 nso^3 nao^2 + 4 nso^4 nao flops
    (1.9e11, 12 s on one core at nao = 69); memory the nso^4 output, an
    nso^2 nao^2 intermediate (0.7 GB there) and blocks of
    PARQUET_SO_DRIVER_BLOCK_BYTES.
    """
    eri_ao = np.asarray(eri_ao, dtype=float)
    c = np.asarray(c, dtype=float)
    nao = eri_ao.shape[0]
    if eri_ao.shape != (nao,) * 4 or c.ndim != 2 or c.shape[0] != 2 * nao:
        raise ValueError(f'eri_ao {eri_ao.shape} and c {c.shape}: expected (nao,)*4 and '
                         f'(2 nao, nso)')
    nso = c.shape[1]
    spins = (c[:nao], c[nao:])
    half = np.zeros((nso, nao, nao, nso))           # [p, nu, si, r]: electron 1 done
    for cs in spins:
        t = np.tensordot(cs, eri_ao, axes=(0, 0))   # [p, nu, la, si]
        half += np.tensordot(t, cs, axes=(2, 0))    # [p, nu, si, r]
        del t
    out = np.zeros((nso,) * 4)
    step = max(1, PARQUET_SO_DRIVER_BLOCK_BYTES // (8 * nso ** 3))
    for p0 in range(0, nso, step):
        blk = slice(p0, min(nso, p0 + step))
        for ct in spins:
            u = np.tensordot(half[blk], ct, axes=(1, 0))   # [p, si, r, q]
            u = np.tensordot(u, ct, axes=(1, 0))           # [p, r, q, s]
            out[blk] += u.transpose(0, 2, 1, 3)
    return out


# ---------------------------------------------------------------------------
# the front door


def spin_orbital_reference(mf):
    """The spin orbitals of a real RHF, UHF or GHF mean field, laid out for
    the drivers (module docstring): RHF interleaved (0 alpha, 0 beta, 1 alpha,
    ...), UHF the same interleaving of (alpha k, beta k), GHF its own order,
    then the occupied spin orbitals first and each block ascending in energy
    (a stable sort, which keeps an RHF's interleaving).

    The AO pieces are those Quack's G_qsParquet takes: the overlap, the core
    Hamiltonian and the four-index <mu nu|la si>. The core Hamiltonian is the
    mean field's own, mf.get_hcore() (T + V, and whatever else its Fock matrix
    holds: a scalar ECP, X2C, an external field), so that qsPA starts from
    the reference's Fock matrix; a GHF's is (2 nao, 2 nao), of which the alpha
    block is taken where the beta block equals it and the spin-off-diagonal
    blocks vanish (to PARQUET_SO_HCORE_SPIN_TOL), and None is kept otherwise
    (spin-orbit or Zeeman terms, which qsPA refuses).

    Parameters
    ----------
    mf : PySCF RHF, UHF or GHF

    Returns
    -------
    SpinOrbitalReference
    """
    _check_reference(mf)
    mol = mf.mol
    nao = mol.nao_nr()
    e = np.asarray(mf.mo_energy, dtype=float)
    coeff = np.asarray(mf.mo_coeff, dtype=float)
    kind = _reference_kind(mf)
    occ = _occupations(mf, kind)
    if kind == 'GHF':
        energy, occupied, cols = e, occ > 0, coeff
        spin = np.full(len(e), -1)
        orbital = np.arange(len(e))
    else:
        if kind == 'RHF':
            e2, occ2, c2 = np.stack((e, e)), np.stack((occ > 0, occ > 0)), np.stack((coeff, coeff))
        else:
            e2, occ2, c2 = e, occ > 0, coeff
        nmo = e2.shape[1]
        energy = e2.T.ravel()                        # (k alpha, k beta) interleaved
        occupied = occ2.T.ravel()
        spin = np.tile([0, 1], nmo)
        orbital = np.repeat(np.arange(nmo), 2)
        cols = np.zeros((2 * nao, 2 * nmo))
        cols[:nao, 0::2] = c2[0]
        cols[nao:, 1::2] = c2[1]
    order = np.lexsort((energy, ~occupied))          # occupied first, then by energy; stable
    hcore, _ = _core_hamiltonian(mf, kind, nao)
    chem = mol.intor('int2e', aosym='s1').reshape((nao,) * 4)
    return SpinOrbitalReference(
        eps=np.ascontiguousarray(energy[order]), mo_coeff=np.ascontiguousarray(cols[:, order]),
        nO=int(np.count_nonzero(occupied)), spin=spin[order], orbital=orbital[order],
        ovlp=mol.intor_symmetric('int1e_ovlp'), hcore=hcore,
        eri_ao=np.ascontiguousarray(chem.transpose(0, 2, 1, 3)), kind=kind)


def _reference_kind(mf):
    """'UHF', 'GHF' or 'RHF': the spin-orbital layout of a checked mean field."""
    if np.asarray(mf.mo_energy).ndim == 2:
        return 'UHF'
    return 'GHF' if np.asarray(mf.mo_coeff).shape[0] == 2 * mf.mol.nao_nr() else 'RHF'


def _core_hamiltonian(mf, kind, nao):
    """(hcore, problem): the mean field's own core Hamiltonian mf.get_hcore()
    as one (nao, nao) block for both spins, and None; for a GHF its alpha
    block, where the beta block equals it and the spin-off-diagonal blocks
    vanish to PARQUET_SO_HCORE_SPIN_TOL, else (None, why not)."""
    h = np.asarray(mf.get_hcore())
    if np.iscomplexobj(h):
        return None, (f'the core Hamiltonian of the {kind} is complex (spin-orbit terms); the '
                      f'spin-orbital parquet port is real')
    h = np.asarray(h, dtype=float)
    if kind != 'GHF':
        if h.shape != (nao, nao):
            return None, (f'the core Hamiltonian of the {kind} has shape {h.shape}, not the '
                          f'({nao}, {nao}) of one block for both spins')
        return np.ascontiguousarray(h), None
    if h.shape != (2 * nao, 2 * nao):
        return None, (f'the core Hamiltonian of the GHF has shape {h.shape}, not '
                      f'({2 * nao}, {2 * nao})')
    a, b = slice(0, nao), slice(nao, 2 * nao)
    spread = max(float(np.max(np.abs(h[b, b] - h[a, a]))), float(np.max(np.abs(h[a, b]))),
                 float(np.max(np.abs(h[b, a]))))
    if not spread <= PARQUET_SO_HCORE_SPIN_TOL:
        return None, (f'the core Hamiltonian of the GHF depends on the spin (its beta block '
                      f'differs from its alpha block, or its spin-off-diagonal blocks are '
                      f'nonzero, by up to {spread:.3e} Ha against PARQUET_SO_HCORE_SPIN_TOL = '
                      f'{PARQUET_SO_HCORE_SPIN_TOL:g}: spin-orbit or Zeeman terms), while '
                      f"Quack's G_qsParquet puts one core Hamiltonian on both spin blocks")
    return np.ascontiguousarray(h[a, a]), None


def run_parquet_so(mf, scheme='os', *, frozen=None, tda_eh=True, tda_pp=True,
                   eta_1b=PARQUET_REGULARIZATION, eta_2b=PARQUET_REGULARIZATION,
                   max_it_1b=None, conv_1b=PARQUET_CONVERGENCE, max_it_2b=None,
                   conv_2b=PARQUET_CONVERGENCE, max_diis_1b=PARQUET_MAX_DIIS_1B,
                   max_diis_2b=PARQUET_MAX_DIIS_2B, qp_solver=None, damping=None,
                   vertex_builder=None):
    """The spin-orbital parquet approximation on a PySCF RHF, UHF or GHF.

    scheme 'flex', 'os' and 'ev' run Quack's G_evParquet (`ev_parquet_so`) on
    the reference's spin orbitals and their <PQ|RS>; 'qs' runs G_qsParquet
    (`qs_parquet_so`) on the AO integrals from the reference's orbitals, with
    Quack's canonical orthogonalization and the mean field's own core
    Hamiltonian. The layout is spin_orbital_reference's (an RHF interleaved
    alpha, beta). Every option, and the reference's occupations (at least one
    occupied and one virtual spin orbital), are checked before any integral
    is built; the qsPA checks of the reference itself (a spin-free core
    Hamiltonian, a canonical orthogonalization that keeps every orbital)
    need the one-electron integrals and come before the two-electron ones.

        res = run_parquet_so(mf, 'os', eta_1b=1e8, eta_2b=1.0, max_diis_2b=5)
        ip = -res.eqp[:res.nO].max()

    Closed shells belong in the restricted driver (`restricted.run_parquet`):
    its qsPA is R_qsParquet's, which this driver's G_qsParquet does not
    reproduce (another static self-energy, `spin_orbital_self_energy`), and
    only an RHF gets its alpha and beta copies solved as one pair here (a UHF
    or GHF of a closed shell does not, so its evPA at large s_1b can depend
    on roundoff). Open shells belong in the TDA of the eh problem (the
    default), at a moderate s_2b: without the TDA, the rotation of a
    collinear open shell's spin axis is an exact zero mode of the eh
    problem, which roundoff makes NaN ('parquet_eh_a_minus_b_not_positive')
    or a near-zero root that dominates every vertex ('parquet_eh_zero_mode';
    spin_orbital_two_body). In the TDA such modes are small rather than
    zero, and nothing flags them (OH UHF/6-31G: 7.5e-4 Ha, the rotation of
    its pi hole about the bond, and spin flips at 3.0e-3 and 7.4e-3 Ha); a
    vertex weighs them by R(Om), about 1/Om once 2 s_2b Om^2 nears 1, so at
    weak two-body regularization they take over (OH's osPA with
    max_diis_2b=5 and damping=0.75 converges at s_2b = 1 and diverges at
    1e6 and 1e8). In evPA and qsPA an open shell's TDA problem has negative
    spin-flip roots from the second one-body iteration on, before any
    kernel, the quasiparticle energies having narrowed a spin flip's gap
    below its exchange term (OH after one iteration: -0.18 Ha in evPA,
    -0.23 Ha in qsPA): every later two-body iteration is flagged
    'parquet_eh_tda_negative_excitations', and at s_2b = 1 both still
    converge.

    THE DEFAULTS ARE QUACK'S, not production settings. The comparisons with
    Quack test the implementation, and the defaults are Quack's so that
    those comparisons hold: G_evParquet's alpha = 1 (damping 0 for 'flex',
    'os' and 'ev'), no DIIS, and the paper's Table III s_1b = s_2b = 50 and
    tau = 1e-4; they were not chosen to give numbers. (The quasiparticle
    equation is the exception: the house's root finder solves it by
    default, qp_solver='find_qp_root'; G_evParquet's own branch, which the
    comparisons pass, is 'linearized'.) At s_2b = 50 the
    undamped two-body loop without DIIS diverges: osPA on water, ammonia
    (RHF) and OH (UHF) in 6-31G runs its 200 iterations to a change of 1e122
    to 1e128 and warns 'parquet_two_body_not_converged' (with
    'parquet_pp_ec_inconsistent' and 'parquet_eh_tda_negative_excitations'
    on the way), and the result it returns holds meaningless energies
    (principal IPs of 27.2 eV for water and 34.0 eV for OH with Z of 1e-117,
    and spin orbitals without a root, NaN, flagged 'no_root'; the root
    finder spends two to three minutes on that self-energy there). Two-body
    DIIS converges the closed shells (max_diis_2b=5: water in 154 two-body
    iterations, ammonia in 60; with damping=0.75 too, the restricted
    driver's damping, in 46 and 43), damping 0.75 alone does not (water
    stalls at a change of 2.8); OH converges at s_2b = 50 with none of these
    in 200 iterations, and at s_2b = 1 with either (max_diis_2b=5: 19
    iterations; damping=0.75: 56; neither: not in 200).

    Parameters
    ----------
    mf : pyscf RHF, UHF or GHF
        Real Hartree-Fock orbitals with integer occupations and four-index
        integrals (a closed-shell ROHF is taken as the RHF it is).
        Kohn-Sham, an open-shell ROHF, density fitting, an attached
        environment, frozen orbitals, a reference without an occupied or a
        virtual spin orbital, and PySCF's one-electron HF1e (what scf.HF,
        RHF, ROHF, UHF and GHF return for one electron, without the
        two-electron terms: use pyscf.scf.uhf.UHF(mol) itself) are refused;
        an unconverged reference is flagged ('scf_not_converged'). For an
        RHF the alpha and beta copy of each orbital are solved as one pair
        (`ev_parquet_so`'s `spin_pairs`).
    scheme : {'flex', 'os', 'ev', 'qs'}
    frozen : None
        Frozen orbitals are not ported (Quack's nC = nR = 0).
    tda_eh, tda_pp, eta_1b, eta_2b, max_it_1b, conv_1b, max_it_2b, conv_2b,
    max_diis_1b, max_diis_2b, vertex_builder
        As for ev_parquet_so; the defaults are the paper's Table III
        settings (TDA, s_1b = s_2b = 50, tau = 1e-4) and Quack's (no DIIS).
    qp_solver : {'find_qp_root', 'linearized'} or callable, optional
        flex, os and ev: as for ev_parquet_so (a function under the same
        contract), None being 'find_qp_root', the house's root finder (the
        default; G_evParquet's own branch is 'linearized'). qs refuses any
        value: it diagonalizes F + Sigma and solves no quasiparticle
        equation.
    damping : float or None, optional
        The weight of the previous vertices in [0, 1) (ev_parquet_so). None:
        PARQUET_SO_EV_VERTEX_DAMPING (0, Quack's alpha 1 of G_evParquet) for
        'flex', 'os', 'ev' and PARQUET_VERTEX_DAMPING (0.75, alpha 0.25 of
        G_qsParquet) for 'qs'.

    Returns
    -------
    SpinOrbitalParquetResult
        With reference, spin and orbital set (for qsPA the spin of the final
        orbitals and orbital None).
    """
    if scheme not in SCHEMES:
        raise ValueError(f'scheme={scheme!r}: one of {SCHEMES}')
    refuse_frozen(frozen, 'run_parquet_so')
    _check_reference(mf)
    nao = mf.mol.nao_nr()
    if scheme == 'qs':
        if qp_solver is not None:
            raise ValueError(f'qp_solver={qp_solver!r}: qsPA diagonalizes F + Sigma and '
                             f'solves no quasiparticle equation, so scheme qs takes no '
                             f'qp_solver (leave it None)')
    else:
        qp_solver = DEFAULT_QP_SOLVER if qp_solver is None else checked_qp_solver(qp_solver)
    # every option before any integral; ev_parquet_so and qs_parquet_so check them again
    _loop_options(scheme, tda_eh, tda_pp, eta_1b, eta_2b, max_it_1b, conv_1b, max_it_2b,
                  conv_2b, max_diis_1b, max_diis_2b, damping, vertex_builder)
    kind = _reference_kind(mf)
    _check_occupations(mf, kind)
    if scheme == 'qs':
        # the reference's own qsPA checks need the one-electron integrals only
        _, problem = _core_hamiltonian(mf, kind, nao)
        if problem is not None:
            raise ValueError(f'run_parquet_so: {problem}; qsPA needs a spin-free core '
                             f'Hamiltonian')
        x = orthogonalization_matrix(mf.mol.intor_symmetric('int1e_ovlp'))
        nso = np.asarray(mf.mo_energy).size * (2 if kind == 'RHF' else 1)
        if 2 * x.shape[1] != nso:
            raise ValueError(f"Quack's canonical orthogonalization keeps {x.shape[1]} functions "
                             f"and the reference has {nso // 2} orbitals per spin; the qsPA "
                             f"port needs the same space")
    common = dict(tda_eh=tda_eh, tda_pp=tda_pp, eta_1b=eta_1b, eta_2b=eta_2b, conv_1b=conv_1b,
                  conv_2b=conv_2b, max_diis_1b=max_diis_1b, max_diis_2b=max_diis_2b,
                  damping=damping, vertex_builder=vertex_builder)
    flags = []
    t0 = time.perf_counter()
    with collecting(flags):
        if not getattr(mf, 'converged', True):
            flag(FLAG_SCF_NOT_CONVERGED, 'run_parquet_so: the reference is not converged; the '
                 'parquet run starts from its last iterate', ParquetNotConverged)
        ref = spin_orbital_reference(mf)
        if scheme == 'qs':
            seconds = time.perf_counter() - t0
            result = qs_parquet_so(ref.eps, ref.mo_coeff, ref.ovlp, x, ref.hcore, ref.eri_ao,
                                   ref.nO, max_it_1b=max_it_1b, max_it_2b=max_it_2b, **common)
        else:
            eri = spin_orbital_eri(ref.eri_ao, ref.mo_coeff)
            seconds = time.perf_counter() - t0
            result = ev_parquet_so(ref.eps, eri, ref.nO, scheme, max_it_1b=max_it_1b,
                                   max_it_2b=max_it_2b, qp_solver=qp_solver,
                                   spin_pairs=_rhf_spin_pairs(ref), **common)
            del eri
    result.times['integrals'] = seconds
    result.numerical_flags = gathered(flags)
    result.reference = ref.kind
    if scheme == 'qs':
        result.spin, result.orbital = _spin_labels(result.mo_coeff, ref.ovlp), None
    else:
        result.spin, result.orbital = ref.spin, ref.orbital
    return result


def _rhf_spin_pairs(ref):
    """The (alpha, beta) spin orbitals of each spatial orbital of an RHF
    reference (`spin_orbital_reference`), an int array (nmo, 2); None for a
    UHF or GHF, whose closed shells get no pairing."""
    if ref.kind != 'RHF':
        return None
    alpha = np.flatnonzero(ref.spin == 0)
    beta = np.flatnonzero(ref.spin == 1)
    alpha = alpha[np.argsort(ref.orbital[alpha], kind='stable')]
    beta = beta[np.argsort(ref.orbital[beta], kind='stable')]
    return np.stack((alpha, beta), axis=1)


def _spin_labels(c, s):
    """The spin of each spin-AO orbital (columns of c, alpha AO rows first):
    0 or 1 where its alpha weight c_a^T s c_a or its beta weight c_b^T s c_b
    is 1 to PARQUET_SO_SPIN_WEIGHT_TOL, -1 otherwise (mixed, or NaN)."""
    nao = s.shape[0]
    ca, cb = c[:nao], c[nao:]
    w_alpha = np.einsum('mk,mn,nk->k', ca, s, ca)
    w_beta = np.einsum('mk,mn,nk->k', cb, s, cb)
    spin = np.full(c.shape[1], -1)
    spin[np.abs(w_alpha - 1.0) <= PARQUET_SO_SPIN_WEIGHT_TOL] = 0
    spin[np.abs(w_beta - 1.0) <= PARQUET_SO_SPIN_WEIGHT_TOL] = 1
    return spin


# ---------------------------------------------------------------------------
# checks and the result


def _check_reference(mf):
    """Refuse a reference the spin-orbital parquet port cannot take."""
    who = 'run_parquet_so'
    if getattr(mf, 'mo_coeff', None) is None or getattr(mf, 'mo_energy', None) is None:
        raise ValueError(f'{who}: the mean field has no orbitals; run it first')
    names = {cls.__name__ for cls in type(mf).__mro__}
    if names & {'KohnShamDFT', 'RKS', 'UKS', 'GKS'} or hasattr(mf, 'xc'):
        raise NotImplementedError(f'{who}: {type(mf).__name__} is Kohn-Sham; the parquet '
                                  'drivers start from Hartree-Fock orbitals and energies')
    if 'HF1e' in names:
        raise ValueError(
            f'{who}: {type(mf).__module__}.{type(mf).__name__} is PySCF\'s one-electron mean '
            f'field (pyscf.scf.HF, RHF, ROHF, UHF and GHF return it for one electron), whose '
            f'orbitals and energies are those of the core Hamiltonian alone: it drops the '
            f'Coulomb and exchange terms of the UHF Fock matrix, so its beta and virtual '
            f'levels are not the UHF ones (an H atom\'s beta 1s comes out degenerate with the '
            f'occupied alpha 1s, a 0/0 in Quack\'s denominators). Build the reference with '
            f'the class itself, mf = pyscf.scf.uhf.UHF(mol) (or pyscf.scf.ghf.GHF(mol)), then '
            f'mf.kernel(): it keeps those terms and converges to the proper UHF')
    if getattr(mf, 'with_df', None) is not None:
        raise NotImplementedError(f'{who}: a density-fitted reference; the parquet port uses '
                                  'four-index integrals, as Quack does')
    if not isinstance(environment_of(mf), NoEnvironment):
        raise NotImplementedError(f'{who}: an environment is attached to the mean field; '
                                  "Quack's parquet approximation is gas phase")
    refuse_frozen(getattr(mf, 'frozen', None), who, 'mf.frozen')
    nao = mf.mol.nao_nr()
    e = np.asarray(mf.mo_energy)
    c = np.asarray(mf.mo_coeff)
    if np.iscomplexobj(e) or np.iscomplexobj(c):
        raise NotImplementedError(f'{who}: complex orbitals; the spin-orbital port is real')
    shapes_ok = ((e.ndim == 2 and c.ndim == 3 and c.shape[:2] == (2, nao))
                 or (e.ndim == 1 and c.ndim == 2 and c.shape[0] in (nao, 2 * nao)))
    if not shapes_ok:
        raise ValueError(f'{who}: mo_energy {e.shape} and mo_coeff {c.shape} with {nao} AOs '
                         f'are not those of an RHF, UHF or GHF')


def _occupations(mf, kind):
    """mf.mo_occ as an array, refused (ValueError) unless the occupations are
    those of a spin-orbital Hartree-Fock reference: 0 or 2 for an RHF, 0 or 1
    for a UHF or GHF."""
    occ = np.asarray(mf.mo_occ)
    allowed = (0.0, 2.0) if kind == 'RHF' else (0.0, 1.0)
    if not np.isin(occ, allowed).all():
        raise ValueError(f'{kind} occupations {sorted(set(occ.ravel().tolist()))}: only '
                         f'{allowed} (an ROHF or fractional occupation is not a spin-orbital '
                         f'Hartree-Fock reference)')
    return occ


def _check_occupations(mf, kind):
    """Refuse, from the occupations alone (before any integral is built), a
    reference whose occupations are not a spin-orbital Hartree-Fock
    reference's (`_occupations`) or that has no occupied or no virtual spin
    orbital."""
    occ = _occupations(mf, kind)
    per_orbital = 2 if kind == 'RHF' else 1
    nso = per_orbital * occ.size
    nO = per_orbital * int(np.count_nonzero(occ > 0))
    if nO == 0 or nO == nso:
        empty = 'occupied' if nO == 0 else 'virtual'
        raise ValueError(f'run_parquet_so: the {kind} reference has no {empty} spin orbital '
                         f'({nO} occupied of {nso}); the eh and pp problems of the parquet '
                         f'approximation need at least one occupied and one virtual spin '
                         f'orbital (a larger basis gives the virtual ones)')


def _check_integrals(eps, eri, nO):
    """nso of (eps, eri, nO), or ValueError."""
    if eps.ndim != 1:
        raise ValueError(f'eps must be one-dimensional, got shape {eps.shape}')
    nso = eps.shape[0]
    if np.shape(eri) != (nso,) * 4:
        raise ValueError(f'eri has shape {np.shape(eri)}, expected {(nso,) * 4}')
    checked_occupied(nO, nso, 'occupied spin orbitals', 'nso')
    return nso


def _loop_options(scheme, tda_eh, tda_pp, eta_1b, eta_2b, max_it_1b, conv_1b, max_it_2b,
                  conv_2b, max_diis_1b, max_diis_2b, damping, vertex_builder):
    """The options of a run, checked (`options.loop_options`), with this
    driver's defaults: damping PARQUET_VERTEX_DAMPING for qsPA (G_qsParquet's
    alpha 0.25) and PARQUET_SO_EV_VERTEX_DAMPING otherwise (G_evParquet's
    alpha 1), the builder `static_reducible_vertex_so`."""
    return loop_options(scheme, tda_eh, tda_pp, eta_1b, eta_2b, max_it_1b, conv_1b, max_it_2b,
                        conv_2b, max_diis_1b, max_diis_2b, damping, vertex_builder,
                        default_damping=(PARQUET_VERTEX_DAMPING if scheme == 'qs'
                                         else PARQUET_SO_EV_VERTEX_DAMPING),
                        default_builder=static_reducible_vertex_so)


def _checked_spin_pairs(spin_pairs, eps_hf, nO):
    """The (alpha, beta) spin-orbital pairs as an int array (npair, 2), or None
    (none given, or an empty set); ValueError unless every index lies in
    [0, nso) at most once, both of a pair are occupied or both virtual, and
    eps_hf is equal within each pair."""
    if spin_pairs is None:
        return None
    pairs = np.asarray(spin_pairs)
    if pairs.size == 0:
        return None
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError(f'spin_pairs: an integer array of (alpha, beta) pairs, shape '
                         f'(npair, 2), not {pairs.dtype} of shape {pairs.shape}')
    nso = len(eps_hf)
    flat = pairs.ravel()
    if flat.min() < 0 or flat.max() >= nso or len(np.unique(flat)) != flat.size:
        raise ValueError(f'spin_pairs: every spin orbital in [0, {nso}) at most once')
    a, b = pairs[:, 0], pairs[:, 1]
    if np.any((a < nO) != (b < nO)):
        raise ValueError('spin_pairs: a pair joins an occupied and a virtual spin orbital')
    if not np.array_equal(eps_hf[a], eps_hf[b]):
        raise ValueError('spin_pairs: the two spin orbitals of a pair of a spin-symmetric '
                         'reference have equal eps_hf, and these do not')
    return pairs.astype(int)


def _one_body_verdict(label, n_it_1b, err_1b, opts, measure, outcome):
    """Quack's verdict on a one-body loop (failed when n_it_1b == max_it_1b),
    False also for a NaN change, which ends Quack's loop as a success."""
    converged = n_it_1b != opts['max_it_1b']
    if np.isnan(err_1b):
        flag(FLAG_ONE_BODY_NOT_FINITE,
             f'spin-orbital {label} one-body loop: the change {measure} is NaN at iteration '
             f'{n_it_1b}; Quack ends the loop there and reports success',
             ParquetNotFinite, stacklevel=4)
        return False
    if not converged:
        flag(FLAG_ONE_BODY_NOT_CONVERGED,
             f'spin-orbital {label} one-body loop not converged after {n_it_1b} iterations: '
             f'{measure} = {err_1b:.3e} against conv_1b = {opts["conv_1b"]:.1e} (Quack counts '
             f'n_it == max_it_1b as failed); {outcome}', ParquetNotConverged,
             stacklevel=4)
    return converged


def _result(scheme, history, eqp, sigc, z, eps_hf, nO, eqp_lin, qp_converged, spectra, state,
            ec_eh, ec_pp, converged_1b, n_it_1b, opts):
    converged_2b = [h['converged_2b'] for h in history]
    converged = converged_1b is not False and all(c is not False for c in converged_2b)
    # 'alpha' is the loop's own 1 - damping, not an option
    options = {k: v for k, v in opts.items() if k not in ('vertex_builder', 'alpha')}
    options['vertex_builder'] = builder_name(opts['vertex_builder'])
    if 'spin_pairs' in options:
        pairs = options['spin_pairs']
        options['spin_pairs'] = None if pairs is None else pairs.tolist()
    if 'qp_solver' in options:
        options['qp_solver'] = qp_solver_label(options['qp_solver'])
    return SpinOrbitalParquetResult(
        scheme=scheme, eqp=eqp, sigc=sigc, z=z, eps_hf=eps_hf, nO=nO, eqp_lin=eqp_lin,
        qp_converged=qp_converged, spectra=spectra, vertices=state.as_vertices(),
        ec_eh=ec_eh, ec_pp=ec_pp, converged=bool(converged), converged_1b=converged_1b,
        converged_2b=converged_2b, n_it_1b=n_it_1b, n_it_2b=[h['n_it_2b'] for h in history],
        history=history, times={}, numerical_flags=[], options=options,
        qp_followed=history[-1].get('qp_followed'), qp_flipped=history[-1].get('qp_flipped'))
