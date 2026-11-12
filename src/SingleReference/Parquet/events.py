"""The warning classes and named flags of the parquet package.

Every numerical event of the parquet drivers and their pieces (`restricted`,
`spin_orbital` and the modules under them) is warned about and raised as a
named flag (`src.Base.numerical_flags.flag`), warned with one of the three
warning classes below but for three events of the quasiparticle equation,
which warn as the house warns them on every quasiparticle route: 'z_outside'
(QPWeightOutside), 'no_root' (QPNoRoot) and 'parquet_qp_branch_flipped' (a
RuntimeWarning, as for a root followed from a seed). The result of a driver
lists the flags raised inside it in
`numerical_flags`. Each flag name is held by one FLAG_* constant here, and
the same event carries the same name in the closed-shell and the spin-orbital
branch. docs/warnings.md has a row for each.

    ParquetNotConverged   a loop or a search stopped short: an iteration cap
                          reached (Quack's 'convergence failed'), a
                          qp_solver function's verdict False, or a
                          reference that was not converged;
    ParquetNotFinite      a quantity came out NaN or infinite, or the
                          arithmetic met an exact zero where Quack's has no
                          defined value (a DIIS pivot; an exactly zero
                          denominator of a regularized weight is no event:
                          the weight and its slope are 0);
    ParquetBSEUnstable    an eh or pp Bethe-Salpeter problem is unstable or
                          its diagonalization failed (Quack's printed BSE
                          warnings, and two events it lets pass without a
                          word: a negative Tamm-Dancoff excitation and an eh
                          zero mode).

Three events carry the house's names rather than parquet ones: an unconverged
mean-field reference is 'scf_not_converged' (as the excitation ADC and the
dense GW-BSE surfaces name it; warned as ParquetNotConverged); a quasiparticle
energy kept with Z outside (0, 1] is 'z_outside'
(`src.Solvers.qp_equation.QPWeightOutside`), and a quasiparticle equation
without a root 'no_root' (`src.Solvers.qp_equation.QPNoRoot`), as on every
quasiparticle route. The
house's root finder raises both for the roots it finds; the drivers raise
them for what they keep beside it: the energies a qp_solver function
returns, the linearized solution, and the second spin orbital of a spin
pair. A root followed from the previous evPA iteration that the house
finder says the largest weight has left warns as the house warns for a
followed root, a RuntimeWarning (`src.Solvers.qp_equation.branch_warning`),
under the parquet name 'parquet_qp_branch_flipped'.
"""


class ParquetNotConverged(RuntimeWarning):
    """A parquet loop or search stopped short: the two-body or one-body loop
    ran its iteration cap (Quack counts n_it == max_it as failed), a
    qp_solver function did not converge on a quasiparticle equation, or the
    mean-field reference was not converged. The run goes on with the last iterate; the named flag
    says which event."""


class ParquetNotFinite(RuntimeWarning):
    """A quantity of the parquet run came out NaN or infinite (a loop's
    change, a DIIS history, the qsPA Fock matrix, a BSE matrix, a
    self-energy, a quasiparticle energy a qp_solver function returned), or
    the arithmetic met an exact zero
    where Quack's result is undefined (a pivot of the DIIS system). The
    values are returned as computed; the named flag says which event. An
    exactly zero denominator of a regularized weight, Quack's 0/0, is no
    event: the weight is 0, its limit, and so is a pole's slope, the value of
    Quack's formula a few ulps off zero."""


class ParquetBSEUnstable(RuntimeWarning):
    """An eh or pp Bethe-Salpeter problem of the parquet two-body loop is
    unstable or its diagonalization failed, as Quack's printed BSE warnings
    say, or it has an event Quack lets pass without a word (a negative
    Tamm-Dancoff excitation; an eh excitation near zero, the spin rotation
    of a collinear open shell or an instability). The spectrum returned is
    what Quack's arithmetic gives (NaN where Quack's is NaN); the named flag
    says which event."""


#: ParquetNotConverged: the two-body loop ran max_it_2b iterations.
FLAG_TWO_BODY_NOT_CONVERGED = 'parquet_two_body_not_converged'
#: ParquetNotConverged: the evPA or qsPA one-body loop ran max_it_1b iterations.
FLAG_ONE_BODY_NOT_CONVERGED = 'parquet_one_body_not_converged'
#: ParquetNotConverged: a qp_solver function returned a finite quasiparticle
#: energy with its verdict False (`options.flag_qp_solution`).
FLAG_QP_NOT_CONVERGED = 'parquet_qp_not_converged'
#: ParquetNotConverged: the mean-field reference is not converged (the
#: house's name for this event).
FLAG_SCF_NOT_CONVERGED = 'scf_not_converged'

#: ParquetNotFinite: every reducible-vertex change of a two-body iteration is NaN.
FLAG_TWO_BODY_NOT_FINITE = 'parquet_two_body_not_finite'
#: ParquetNotFinite: the one-body change of an evPA or qsPA iteration is NaN.
FLAG_ONE_BODY_NOT_FINITE = 'parquet_one_body_not_finite'
#: ParquetNotFinite: the error history of a DIIS holds NaN or inf (only the
#: errors are tested), where Quack's weights are undefined; the iterate is NaN.
FLAG_DIIS_NOT_FINITE = 'parquet_diis_not_finite'
#: ParquetNotFinite: the DIIS system is exactly singular, where Quack's
#: weights are undefined; the weights are its least-squares solution.
FLAG_DIIS_SINGULAR = 'parquet_diis_singular'
#: ParquetNotFinite: the qsPA effective Fock matrix cannot be diagonalized.
FLAG_QS_FOCK_NOT_FINITE = 'parquet_qs_fock_not_finite'
#: ParquetNotFinite: NaN or inf entered a BSE diagonalization.
FLAG_BSE_NOT_FINITE = 'parquet_bse_not_finite'
#: ParquetNotFinite: the diagonal self-energy or its derivative is not finite:
#: an input that is not finite, an overflow, a pole hit at eta = inf (the
#: bare 1/d), or, in the closed-shell diagonal, dSigma/dw exactly 1 (Z =
#: 1/0). A regularized term whose denominator is exactly zero is 0, its
#: slope too, and no event.
FLAG_SELF_ENERGY_NOT_FINITE = 'parquet_self_energy_not_finite'
#: ParquetNotFinite: the static (qsPA) self-energy matrix is not finite: an
#: input that is not finite, an overflow, or an unregularized form at a zero
#: denominator (the SRG form at eta = inf, which the drivers refuse; the eta
#: form at a broadening whose square underflows to 0, below about 1.6e-162
#: Ha, which they take, or at 0, which they refuse).
FLAG_STATIC_SIGMA_NOT_FINITE = 'parquet_static_sigma_not_finite'
#: ParquetNotFinite: a qp_solver function returned a quasiparticle energy that
#: is not finite (`options.flag_qp_solution`).
FLAG_QP_NOT_FINITE = 'parquet_qp_not_finite'
#: ParquetNotFinite: a reducible-vertex weight is not finite: an energy (or
#: omega) that is not finite (both branches test the weights); the vertex is
#: NaN there. A denominator omega -+ Omega exactly zero is no event: its
#: weight is the limit 0 (at eta = inf, the bare 1/d, which the closed-shell
#: weight accepts but no driver, it stays a pole and is flagged).
FLAG_VERTEX_POLE = 'parquet_vertex_pole'
#: ParquetNotFinite: a vertex_builder other than the default returned a
#: reducible vertex holding NaN or inf (counted before the damping); the loop
#: keeps it as returned, and its change, gfortran's maxval, passes over NaN,
#: so the run can report convergence with it. The default builder's
#: non-finite vertices carry 'parquet_vertex_pole' or a BSE flag instead.
FLAG_VERTEX_NOT_FINITE = 'parquet_vertex_not_finite'

#: ParquetBSEUnstable: A - B of the full eh BSE is not positive definite.
FLAG_EH_A_MINUS_B_NOT_POSITIVE = 'parquet_eh_a_minus_b_not_positive'
#: ParquetBSEUnstable: the full eh BSE has negative squared excitation energies.
FLAG_EH_NEGATIVE_EXCITATIONS = 'parquet_eh_negative_excitations'
#: ParquetBSEUnstable: the Tamm-Dancoff eh BSE has negative excitation energies
#: (over spin orbitals, below -PARQUET_EH_ZERO_MODE_TOL: a one-electron system's
#: spin flip is an exact zero that roundoff puts on either side).
FLAG_EH_TDA_NEGATIVE_EXCITATIONS = 'parquet_eh_tda_negative_excitations'
#: ParquetBSEUnstable: the full pp BSE has other than nVV roots above zero.
FLAG_PP_BRANCH_COUNT = 'parquet_pp_branch_count'
#: ParquetBSEUnstable: the full pp BSE has complex roots.
FLAG_PP_COMPLEX_ROOTS = 'parquet_pp_complex_roots'
#: ParquetBSEUnstable: a pp branch's metric overlap is near singular.
FLAG_PP_LOWDIN_SMALL_EIGENVALUE = 'parquet_pp_lowdin_small_eigenvalue'
#: ParquetBSEUnstable: the pp correlation energies of the two branches disagree.
FLAG_PP_EC_INCONSISTENT = 'parquet_pp_ec_inconsistent'
#: ParquetBSEUnstable: dgeev of the full pp BSE did not converge.
FLAG_PP_DGEEV_FAILED = 'parquet_pp_dgeev_failed'
#: ParquetBSEUnstable: a symmetric BSE diagonalization failed in LAPACK.
FLAG_BSE_DIAGONALIZATION_FAILED = 'parquet_bse_diagonalization_failed'
#: ParquetBSEUnstable: the full eh BSE has an excitation energy below
#: PARQUET_EH_ZERO_MODE_TOL, a zero mode (a collinear open shell's spin
#: rotation, or a closed shell at the edge of an instability) whose Om^-1/2
#: amplitudes take over the vertices and the self-energy; kept, as Quack
#: keeps it.
FLAG_EH_ZERO_MODE = 'parquet_eh_zero_mode'

#: src.Solvers.qp_equation.QPWeightOutside: a quasiparticle energy kept with Z
#: outside (0, 1]: the house finder's root, an energy a qp_solver function
#: returned, or the linearized solution (the house's name for this event on
#: every quasiparticle route).
FLAG_Z_OUTSIDE = 'z_outside'
#: src.Solvers.qp_equation.QPNoRoot: a quasiparticle equation without a root in
#: the house finder's widest scan window, its root NaN (the house's name; the
#: spin-orbital driver raises it for the second spin orbital of a pair, which
#: takes the first one's missing root, as the finder raises it for the first).
FLAG_NO_ROOT = 'no_root'
#: RuntimeWarning, as the house warns for a root followed from a seed
#: (src.Solvers.qp_equation.branch_warning): with qp_solver='find_qp_root',
#: an evPA root followed from the previous one-body iteration that the house
#: finder reports the largest weight has left (`options.flag_branch`): its
#: branch has gone (the root of largest weight is kept again), or it names a
#: root of larger weight beside it (the followed root is kept). One per
#: orbital and one-body iteration; the history's 'qp_flipped' marks the same.
FLAG_QP_BRANCH_FLIPPED = 'parquet_qp_branch_flipped'
