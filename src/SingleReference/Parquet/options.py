"""The options both parquet drivers take, checked one way.

The closed-shell drivers (`restricted`: run_parquet, ev_parquet, qs_parquet)
and the spin-orbital ones (`spin_orbital`: run_parquet_so, ev_parquet_so,
qs_parquet_so) take the same loop options, and refuse a wrong one the same
way: a ValueError that names it, raised before any integral is built
(docs/warnings.md). A number is one real number, Python's or numpy's, and
never a bool, a string, an array, a complex number or None; a count is a
whole number, never truncated; a switch is True or False (numpy's bool too),
never read by its truth value. What each option means is in the drivers'
docstrings.

    QP_SOLVERS          the names `qp_solver` takes; DEFAULT_QP_SOLVER,
                        the drivers' default ('find_qp_root')
    loop_options        every loop option of a run, checked, with the
                        scheme's iteration caps and Quack's mixing weight
    checked_real, checked_whole, checked_count, checked_switch,
    checked_occupied, checked_damping, checked_qp_solver
                        one option each; checked_qp_solver states the
                        contract of a qp_solver function
    refuse_frozen       frozen orbitals, the front doors' keyword and the
                        mean field's own (NotImplementedError)
    builder_name        a vertex builder as the options record it
    qp_solver_label     a qp_solver as the options and the history record it
    solve_with          one one-body step solved by a qp_solver function:
                        its call, the check of what it returns
                        (checked_qp_solution), the spin partners' copies
                        and the flags of what it returns (flag_qp_solution)
    roots_to_follow     the roots qp_solver='find_qp_root' follows at an
                        evPA iteration
    flag_branch         the warning and flag of a followed root the house
                        finder reports the largest weight has left
"""
import functools
import inspect
import math
import numbers

import numpy as np

from src.Base.numerical_flags import flag
from src.Base.numerics.constants import (PARQUET_MAX_IT_1B, PARQUET_MAX_IT_2B,
                                         PARQUET_QUACK_INITIAL_ERROR)
from src.Solvers.qp_equation import (QPRoot, QPWeightOutside, branch_warning,
                                     quasiparticle_weight)
from src.SingleReference.Parquet.events import (FLAG_QP_BRANCH_FLIPPED, FLAG_QP_NOT_CONVERGED,
                                                FLAG_QP_NOT_FINITE, FLAG_Z_OUTSIDE,
                                                ParquetNotConverged, ParquetNotFinite)

#: The names `qp_solver` takes for the quasiparticle equation of FLEX, osPA
#: and evPA: 'find_qp_root', the house's one root finder on that equation
#: (restricted_self_energy.DiagonalSelfEnergy.solve_qp_scan,
#: spin_orbital.solve_qp_scan_so), and 'linearized', eQP = eQPlin (Quack's
#: lin_parquet, G_evParquet's only branch). In their place `qp_solver` takes
#: a function (`checked_qp_solver`).
QP_SOLVERS = ('find_qp_root', 'linearized')
#: The drivers' default `qp_solver`: the house's one root finder.
DEFAULT_QP_SOLVER = 'find_qp_root'
#: What a qp_solver function is called with (`checked_qp_solver`).
QP_SOLVER_ARGUMENTS = ('sigma', 'eps_hf', 'eqp_lin', 'z_lin', 'orbitals')


def checked_real(name, value, what='a real number'):
    """value as a float when it is one real number (Python's or numpy's; not
    a bool, a string, an array, a complex number or None, and not an
    integer too large for a float), else ValueError naming `what`."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f'{name} = {value!r}: {what}')
    try:
        return float(value)
    except OverflowError:
        raise ValueError(f'{name} = {value!r}: {what}') from None


def checked_whole(name, value, what):
    """value as an int when it is a whole number: an integer (Python's or
    numpy's) or a float without a fraction, finite as a float (not a bool, a
    string, an array, a Fraction, NaN or inf, and never truncated), else
    ValueError naming `what`."""
    whole = (not isinstance(value, (bool, np.bool_))
             and isinstance(value, (numbers.Integral, float, np.floating)))
    if whole:
        try:
            number = float(value)
        except OverflowError:
            whole = False
        else:
            whole = math.isfinite(number) and number.is_integer()
    if not whole:
        raise ValueError(f'{name} = {value!r}: {what}')
    return int(value)


def checked_count(name, value, what):
    """value as an int >= 1 (`checked_whole`), else ValueError naming
    `what`."""
    count = checked_whole(name, value, what)
    if count < 1:
        raise ValueError(f'{name} = {value!r}: {what}')
    return count


def checked_switch(name, value):
    """A True/False option (Python's or numpy's bool) as a bool, or
    ValueError: a string, None or a number is refused, not read by its truth
    value."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} = {value!r}: True or False')
    return bool(value)


def checked_occupied(nO, n, occupied, size):
    """nO as an int, a whole number between 1 and n - 1 (`checked_whole`),
    or ValueError; the message calls the nO orbitals `occupied` and their
    number n `size` (the drivers' 'doubly occupied orbitals' of norb,
    'occupied spin orbitals' of nso)."""
    what = f'the {occupied}, an integer between 1 and {size} - 1 = {n - 1}'
    count = checked_whole('nO', nO, what)
    if not 0 < count < n:
        raise ValueError(f'nO = {nO!r}: {what}')
    return count


def checked_damping(damping, default):
    """The weight of the previous vertices, a real number in [0, 1), as a
    float (None: `default`, the driver's), or ValueError (a bool, a string
    or an array is refused, not converted)."""
    value = checked_real('damping', default if damping is None else damping,
                         'the weight of the previous vertices, a real number in [0, 1)')
    if not 0.0 <= value < 1.0:
        raise ValueError(f'damping = {damping!r}: the weight of the previous vertices, in '
                         f'[0, 1) (Quack\'s mixing weight alpha is 1 - damping; 1 would '
                         f'freeze the vertices)')
    return value


def checked_qp_solver(qp_solver):
    """qp_solver, one of the names QP_SOLVERS or a qp_solver function, or
    ValueError (None too: only the front doors take None, as
    DEFAULT_QP_SOLVER).

    A qp_solver function solves the quasiparticle equations of one FLEX,
    osPA or evPA one-body iteration; both drivers call it the same way,

        eqp, z, converged = qp_solver(sigma, eps_hf, eqp_lin, z_lin, orbitals)

    sigma(p, w)
        (Sigma_pp(w), dSigma_pp/dw), two numpy float64: the diagonal
        correlation self-energy of orbital p (0-based; a spin orbital in
        the spin-orbital driver) at the frequency w (Hartree, a real
        number), every other denominator on the iteration's energies (eOld),
        the derivative holding the regularizers fixed (Quack's Z is
        1/(1 - dSigma_pp/dw)). Each call is one evaluation, counted per
        orbital in the history's 'qp_n_iter'. The spin-orbital driver builds
        the residues of one orbital at a time, again when the orbital asked
        for changes: solve one orbital before the next.
    eps_hf : ndarray, shape (n,)
        The bare levels: the equation of orbital p is
        w = eps_hf[p] + Sigma_pp(w).
    eqp_lin, z_lin : ndarray, shape (n,)
        The linearized energies eHF + Z SigC and the diagonal Z, both at eOld
        (Quack's start and fallback).
    orbitals : ndarray of int
        The orbitals to solve, ascending: all n, or, with the spin-orbital
        driver's spin_pairs, the first of each pair.

    It returns three arrays of shape (n,), read at `orbitals`: eqp and z,
    real, and converged, bool. The arrays handed in are the function's own
    copies. The driver checks what it returns (`checked_qp_solution`:
    anything else is refused with a ValueError at the first one-body
    iteration), keeps it as it is, gives the second of each spin pair the
    first's eqp, z and verdict, records the function by its label
    (`qp_solver_label`) in the options and the history, and flags what it
    keeps (`flag_qp_solution`); it follows no root (qp_followed and
    qp_flipped are None). A function those five arguments cannot be bound
    to, by position, is refused here, before any work; so is a class."""
    if isinstance(qp_solver, str):
        if qp_solver in QP_SOLVERS:
            return qp_solver
    elif callable(qp_solver) and not isinstance(qp_solver, type):
        try:
            signature = inspect.signature(qp_solver)
        except (TypeError, ValueError):
            return qp_solver                # no signature to check: called as given
        try:
            signature.bind(*QP_SOLVER_ARGUMENTS)
        except TypeError as err:
            raise ValueError(f'qp_solver {qp_solver_label(qp_solver)}: a qp_solver function '
                             f'is called as qp_solver({", ".join(QP_SOLVER_ARGUMENTS)}), and '
                             f'this one cannot take that ({err})') from None
        return qp_solver
    raise ValueError(f'qp_solver={qp_solver!r}: one of {QP_SOLVERS}, or a function '
                     f'({", ".join(QP_SOLVER_ARGUMENTS)}) -> (eqp, z, converged)')


def qp_solver_label(qp_solver):
    """qp_solver as the options and the history record it: a name of
    QP_SOLVERS as it is; a function by its module and qualified name
    ('tests.parquet_quack_newton.quack_newton'), a functools.partial as
    'partial(<that>)', any other callable by the module and qualified name
    of its type. The label is the same from run to run (never a repr, whose
    address differs) and, holding a dot or a bracket, never one of the
    names."""
    if isinstance(qp_solver, str):
        return qp_solver
    if isinstance(qp_solver, functools.partial):
        return f'partial({qp_solver_label(qp_solver.func)})'
    module = getattr(qp_solver, '__module__', None)
    name = getattr(qp_solver, '__qualname__', None)
    if isinstance(module, str) and isinstance(name, str):
        return f'{module}.{name}'
    kind = type(qp_solver)
    return f'{kind.__module__}.{kind.__qualname__}'


def checked_qp_solution(qp_solver, solution, n):
    """(eqp, z, converged), what the qp_solver function returned for n
    orbitals, as new float64, float64 and bool arrays of shape (n,), or
    ValueError naming the function: anything but three arrays of that shape,
    eqp and z real (float or integer) and converged bool, is refused, not
    converted (a complex or object array, another shape, a converged of 0
    and 1)."""
    label = qp_solver_label(qp_solver)
    try:
        eqp, z, converged = solution
    except (TypeError, ValueError):
        raise ValueError(f'qp_solver {label} returned {type(solution).__name__}, not the '
                         f'three arrays (eqp, z, converged)') from None
    checked = []
    for name, value, kinds, dtype in (('eqp', eqp, 'fiu', float), ('z', z, 'fiu', float),
                                      ('converged', converged, 'b', bool)):
        arr = np.asarray(value)
        if arr.shape != (n,) or arr.dtype.kind not in kinds:
            raise ValueError(f'qp_solver {label} returned {name} of shape {arr.shape} and '
                             f'dtype {arr.dtype}; it must be {"bool" if dtype is bool else "real"}'
                             f' of shape ({n},)')
        checked.append(np.array(arr, dtype=dtype))
    return tuple(checked)


def flag_qp_solution(label, eqp, z, converged, what):
    """Warn and flag what the qp_solver function `label` returned, over every
    orbital (`what`, 'orbital' or 'spin orbital'; a spin pair's second as
    the first's): one 'parquet_qp_not_converged' (ParquetNotConverged) for
    the orbitals whose energy is finite and whose verdict is False, one
    'parquet_qp_not_finite' (ParquetNotFinite) for those whose energy is not
    finite, and a 'z_outside' (src.Solvers.qp_equation.QPWeightOutside) for
    each finite energy with Z outside (0, 1], as on every quasiparticle
    route. The energies stay as returned."""
    finite = np.isfinite(eqp)
    late = np.flatnonzero(finite & ~converged)
    if late.size:
        flag(FLAG_QP_NOT_CONVERGED,
             f'qp_solver {label} did not converge on the parquet quasiparticle equation of '
             f'{what}s {late.tolist()} (0-based); their energies are kept as it returned them',
             ParquetNotConverged, stacklevel=5)
    lost = np.flatnonzero(~finite)
    if lost.size:
        flag(FLAG_QP_NOT_FINITE,
             f'qp_solver {label} returned a parquet quasiparticle energy that is not finite '
             f'for {what}s {lost.tolist()} (0-based); kept as returned',
             ParquetNotFinite, stacklevel=5)
    with np.errstate(invalid='ignore'):
        outside = finite & ~((z > 0.0) & (z <= 1.0))
    for p in np.flatnonzero(outside):
        flag(FLAG_Z_OUTSIDE,
             f'the parquet quasiparticle energy qp_solver {label} kept for {what} {p} '
             f'(0-based), {float(eqp[p])!r} Ha, has Z = {float(z[p])!r}, outside (0, 1], which '
             f'no root of a positive-weight self-energy has: kept', QPWeightOutside,
             stacklevel=5)


def solve_with(qp_solver, sigma, eps_hf, eqp_lin, z_lin, source, what):
    """(eqp, z, converged, n_eval) of one one-body step solved by the
    qp_solver function (`checked_qp_solver`): `sigma(p, w)` the driver's
    evaluation of Sigma_pp and its derivative at w, handed to the function
    as two numpy float64 and counted per orbital (n_eval); source[p] the
    orbital whose root p takes (p itself, or the first of its spin pair),
    the function solving the orbitals with source[p] == p. What it returns
    is checked (`checked_qp_solution`), copied to the spin partners and
    flagged (`flag_qp_solution`, `what` naming the orbitals)."""
    n = len(eps_hf)
    n_eval = np.zeros(n, dtype=int)

    def counted(p, w):
        sig, dsig = sigma(p, w)
        n_eval[p] += 1
        return np.float64(sig), np.float64(dsig)

    source = np.asarray(source)
    solved = np.flatnonzero(source == np.arange(n))
    solution = qp_solver(counted, np.array(eps_hf, dtype=float), np.array(eqp_lin, dtype=float),
                         np.array(z_lin, dtype=float), solved.copy())
    eqp, z, converged = checked_qp_solution(qp_solver, solution, n)
    copied = np.flatnonzero(source != np.arange(n))
    for arr in (eqp, z, converged):
        arr[copied] = arr[source[copied]]
    flag_qp_solution(qp_solver_label(qp_solver), eqp, z, converged, what)
    return eqp, z, converged, n_eval


def refuse_frozen(frozen, who, what='frozen'):
    """Refuse frozen orbitals, which the port does not have (Quack's nC = nR =
    0): only None, or a whole number 0 (Python's or numpy's integer, or a
    float 0.0), means none. Anything else is refused with
    NotImplementedError naming `who` and `what` (the front door's keyword,
    or the mean field's own mf.frozen): a count above 0, a list or an array
    of orbitals (one orbital too, np.arange(1), whose == 0 numpy reads as
    true; a 0-d array as well), a bool, a string."""
    if frozen is None:
        return
    zero = (not isinstance(frozen, (bool, np.bool_)) and isinstance(frozen, numbers.Real)
            and frozen == 0)
    if not zero:
        raise NotImplementedError(f'{who}: {what} = {frozen!r}: frozen orbitals are not part '
                                  f"of the parquet port (Quack's nC = nR = 0); only None or 0")


def builder_name(builder):
    """The vertex builder as the options record it: a function's __name__, a
    partial's 'partial(<name>)', else the qualified name of the callable's
    type (never a repr, whose address differs between identical runs)."""
    name = getattr(builder, '__name__', None)
    if isinstance(name, str):
        return name
    func = getattr(builder, 'func', None)
    if func is not None and callable(func):
        return f'partial({builder_name(func)})'
    kind = type(builder)
    return f'{kind.__module__}.{kind.__qualname__}'


def loop_options(scheme, tda_eh, tda_pp, eta_1b, eta_2b, max_it_1b, conv_1b, max_it_2b, conv_2b,
                 max_diis_1b, max_diis_2b, damping, vertex_builder, *, default_damping,
                 default_builder, eta_1b_read=True):
    """The loop options of a parquet run, checked in this order (the first
    wrong one is the one refused), with the scheme's iteration caps.

    Parameters
    ----------
    scheme : {'flex', 'os', 'ev', 'qs'}
        'flex' takes max_it_1b = max_it_2b = 1 and 'os' max_it_1b = 1 (None
        is that value; another is refused); otherwise None is
        PARQUET_MAX_IT_1B and PARQUET_MAX_IT_2B.
    tda_eh, tda_pp : bool
    eta_1b, eta_2b : float
        Finite and >= 0.
    max_it_1b, max_it_2b : int or None
        Whole numbers >= 1.
    conv_1b, conv_2b : float
        In (0, PARQUET_QUACK_INITIAL_ERROR): Quack's loops start at that
        error.
    max_diis_1b, max_diis_2b : int
        Whole numbers >= 1 (1 is no DIIS).
    damping : float or None
        In [0, 1); None is `default_damping`.
    vertex_builder : callable or None
        None is `default_builder`.
    default_damping : float
        The driver's (and scheme's) default weight of the previous vertices.
    default_builder : callable
        The driver's default vertex builder.
    eta_1b_read : bool
        False where eta_1b is not read (the closed-shell qsPA with
        reg_pa=False): None is accepted, a value given is still checked,
        and None is recorded.

    Returns
    -------
    dict
        scheme, tda_eh, tda_pp, eta_1b, eta_2b, max_it_1b, conv_1b,
        max_it_2b, conv_2b, max_diis_1b, max_diis_2b, damping (floats and
        ints, as checked), alpha = 1 - damping (Quack's mixing weight, the
        loop's own) and vertex_builder (the callable).
    """
    switches = {name: checked_switch(name, value)
                for name, value in (('tda_eh', tda_eh), ('tda_pp', tda_pp))}
    fixed = {'flex': (1, 1), 'os': (1, None)}.get(scheme, (None, None))
    defaults = {'max_it_1b': PARQUET_MAX_IT_1B, 'max_it_2b': PARQUET_MAX_IT_2B}
    caps = {}
    for name, value, forced in (('max_it_1b', max_it_1b, fixed[0]),
                                ('max_it_2b', max_it_2b, fixed[1])):
        if value is None:
            value = defaults[name] if forced is None else forced
        caps[name] = checked_count(name, value, 'a positive integer')
        if forced is not None and caps[name] != forced:
            raise ValueError(f'scheme {scheme!r} takes {name} = {forced}, not {value!r}')
    thresholds = {}
    for name, value in (('conv_1b', conv_1b), ('conv_2b', conv_2b)):
        what = (f'Quack starts its loops at an error of {PARQUET_QUACK_INITIAL_ERROR}, so a '
                f'threshold must lie in (0, 1)')
        thresholds[name] = checked_real(name, value, what)
        if not 0.0 < thresholds[name] < PARQUET_QUACK_INITIAL_ERROR:
            raise ValueError(f'{name} = {value!r}: {what}')
    regularizations = {}
    for name, value, read in (('eta_1b', eta_1b, eta_1b_read), ('eta_2b', eta_2b, True)):
        if value is None and not read:
            regularizations[name] = None
            continue
        what = 'a finite regularization >= 0'
        number = checked_real(name, value, what)
        if not (math.isfinite(number) and number >= 0.0):
            raise ValueError(f'{name} = {value!r}: {what}')
        regularizations[name] = number if read else None
    lengths = {name: checked_count(name, value, 'a DIIS length >= 1 (1 is none)')
               for name, value in (('max_diis_1b', max_diis_1b), ('max_diis_2b', max_diis_2b))}
    damping = checked_damping(damping, default_damping)
    if vertex_builder is not None and not callable(vertex_builder):
        raise ValueError(f'vertex_builder = {vertex_builder!r}: a callable (name, spectra, '
                         f'eta_2b) -> Phi, or None for {default_builder.__name__}')
    return dict(scheme=scheme, **switches, **regularizations,
                max_it_1b=caps['max_it_1b'], conv_1b=thresholds['conv_1b'],
                max_it_2b=caps['max_it_2b'], conv_2b=thresholds['conv_2b'], **lengths,
                damping=damping, alpha=1.0 - damping,
                vertex_builder=default_builder if vertex_builder is None else vertex_builder)


def roots_to_follow(e_old, z_old):
    """What qp_solver='find_qp_root' follows at an evPA one-body iteration
    after the first: each orbital's previous eQP (eOld, after any one-body
    DIIS) where the previous root's Z lies in (0, 1], NaN (not followed,
    solved as at the first iteration) where it lies outside or is NaN.

    The house finder follows roots of valid weight only. A root kept with Z
    outside (0, 1] (beside a pole of the regularized self-energy) is
    followed to the valid root nearest it where one lies in the windows about
    it, and otherwise the follow falls back to the search without one, which
    gives the same root at twice the evaluations (the spin-orbital driver's
    ammonia/6-31G virtuals of Z 1.04 at s_1b = 50, at every iteration); such
    an orbital is solved without following instead. Where the Z of the next
    iteration's root came back into (0, 1], following it costs what the
    search does, the finder searching the whole scan window of a root
    followed for its branch report (the restricted ammonia/6-31G orbitals 12
    and 13 at the second iteration: 2 302 evaluations each either way); the
    root is the same in both cases measured.

    Parameters
    ----------
    e_old : ndarray, shape (n,)
        The previous one-body iteration's energies (the next denominators).
    z_old : ndarray, shape (n,)
        The previous iteration's Z of its roots.

    Returns
    -------
    follow : ndarray, shape (n,)
        e_old, NaN where not followed.
    """
    e_old = np.asarray(e_old, dtype=float)
    z_old = np.asarray(z_old, dtype=float)
    if e_old.shape != z_old.shape:
        raise ValueError(f'e_old has shape {e_old.shape}, z_old {z_old.shape}')
    with np.errstate(invalid='ignore'):
        valid = (z_old > 0.0) & (z_old <= 1.0)
    return np.where(valid, e_old, np.nan)


def flag_branch(label, root, z, record, follow):
    """Warn and flag a root followed from the previous evPA iteration that the
    house finder reports the largest weight has left, as the house warns for a
    root followed from a seed: a RuntimeWarning with
    src.Solvers.qp_equation.branch_warning's text, and the named flag
    'parquet_qp_branch_flipped'. Returns whether it did.

    `record` is the finder's `out` of solve_qp_equation_scan for this root
    (`root`, `z`), followed from `follow`, and its 'flipped' is passed on as
    it is: the branch has gone (followed False: the root of largest weight
    is kept again), or a root of larger weight lies in the scan window of
    the root followed (kept on its branch, 'largest' naming the other). With
    positive_weight off, as the parquet self-energy's residues of both signs
    need, the finder refines every bracket of that window, whatever the Z of
    the root followed (evPA, s_2b 1: water/6-31G at the tests' geometry and
    s_1b = 1e8, third iteration: orbital 10 kept at Z 0.27 beside a root of
    Z 0.96 0.02 Ha away; ammonia/6-31G at s_1b = 50, third and fourth
    iterations: the N 1s kept at Z 0.58 and 0.56 beside roots of Z 0.70
    and 0.77; all flagged). An
    equation left without any root (NaN, the finder's 'no_root') is no
    flip: its branch is not followed anywhere."""
    if not record.get('flipped') or not np.isfinite(root):
        return False
    got = QPRoot(float(root), float(z), quasiparticle_weight(z), (), record.get('labels'),
                 record.get('largest'), record.get('followed'))
    flag(FLAG_QP_BRANCH_FLIPPED,
         f'{branch_warning(label, got)} (evPA with qp_solver=\'find_qp_root\': the root '
         f'followed is the previous one-body iteration\'s, {float(follow):.10f} Ha)',
         RuntimeWarning, stacklevel=3)
    return True
