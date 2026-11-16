"""Frequency-averaged reducible vertices for the static kernels of the parquet
loop: builders for the drivers' `vertex_builder` hook.

Where the static kernel approximation drops frequency dependence
----------------------------------------------------------------
Every reducible vertex inside a BSE kernel is its value at its own bosonic
frequency omega = 0 (Marie and Loos, J. Chem. Phys. 163, 194115 (2025),
Eqs. (28) and (47)): the eh vertex inserted crossed into the eh kernel and
direct and swapped into the pp kernel, the pp vertex direct into the eh
kernel, four insertions, each at omega = 0 (the eh transfer frequency, the
pp total frequency). In the frequency-dependent equations each insertion's
argument is a combination of the receiving channel's frequencies, which the
kernel integrates over. The static vertex (Eqs. (39), (45)) gives the pole n
of a channel the weight R_s(0 -+ Omega_n), with the SRG regularizer of
Eq. (52), R_s(x) = (1 - exp(-2 s x^2)) / x and s = s_2b (`eta_2b`).

Averaged weights
----------------
For an even, normalized weight w(y) (int w = 1) with the time window
g(t) = int w(y) cos(y t) dy (g(0) = 1), the bosonic-frequency average of
the pole 1/x is

    real axis (the principal value kept; the absorptive part -i pi w(x),
    which a real static kernel cannot hold, dropped):
        R_w(x) = P.V. int w(y) / (x - y) dy = int_0^inf sin(x t) g(t) dt
    imaginary axis (omega = i nu, a Matsubara average: the average of a
    time-ordered pole is real, nothing is dropped):
        R_w(x) = x int w(nu) / (nu^2 + x^2) dnu = sign(x) L(|x|),
        L(x) = int_0^inf g(t) exp(-x t) dt   (the Laplace transform of g)

Each R_w is odd, and a vertex averaged so is the static vertex with R_w in
place of R_s, at the cost of one static vertex (the weights are one number
per pole). A builder serves either driver, as the vertex names it is called
with tell them apart, and mirrors each driver's default exactly:

    closed shell (`restricted.static_reducible_vertex`):
        eh: restricted_vertices.eh_vertex_from_weights(rho, R_w(0 - Omega),
                                                       -R_w(0 + Omega))
        pp: restricted_vertices.pp_vertex_from_weights(
                rho_ee, R_w(c - Omega^ee), rho_hh, -R_w(c - Omega^hh))
    spin orbitals (`spin_orbital.static_reducible_vertex_so`, whose
    contractions subtract the antiresonant and hh terms themselves):
        eh: spin_orbital_two_body.eh_vertex_from_weights(rho, R_w(0 - Omega),
                                                         R_w(0 + Omega))
        pp: spin_orbital_two_body.pp_vertex_from_weights(
                rho_ee, R_w(c - Omega^ee), rho_hh, R_w(c - Omega^hh))

with c the pp centre (`pp_center`, Hartree): 0 as the paper and Quack take
the pp vertex (at the zero of the orbital energies, Quack's eF = 0), or any
other point, e.g. 2 mu = eps_HOMO + eps_LUMO. The eh vertices are always
taken about 0, the symmetric point of their pole pairs.

R_s is itself a real-axis average, with the positive weight
w_s(y) = 2 F(sqrt(2 s) y) / (pi^(3/2) y) (F Dawson's function; time window
erfc(t / sqrt(8 s)); half width at half maximum 0.7517 / sqrt(s) Ha, 0.106 Ha
at s = 50): the static vertex is already a window average, and a window here
is another weight, not an average where there was none. Yirga and Campbell's
central cluster function (arXiv:2010.02163, App. C, Eq. (C4), at tau_m = 0)
is sinc(omega dtau), a box of imaginary time: on the Matsubara axis it gives
(1 - exp(-T |x|)) / x, the 'box' window.

The windows
-----------
    axis        window        g(t)                 L(x) (imaginary) or R(x) (real), x > 0
    imaginary   'box'         1 for t <= T         -expm1(-x T) / x
                'exponential' exp(-gamma t)        1 / (x + gamma)
                'gaussian'    exp(-sigma^2 t^2/2)  sqrt(pi/2) / sigma erfcx(x / (sqrt(2) sigma))
                'triangle'    (1 - t/T)_+          (u + expm1(-u)) / (x u), u = x T
    real        'gaussian'    exp(-sigma^2 t^2/2)  (sqrt(2) / sigma) F(x / (sqrt(2) sigma))
                'lorentzian'  exp(-eta t)          x / (x^2 + eta^2)
                'fejer'       (1 - t/T)_+          (1 - sin(x T) / (x T)) / x
                'sinc'        1 for t <= T         (1 - cos(x T)) / x = 2 sin^2(x T/2) / x
                'srg'         erfc(t / sqrt(8 s))  (1 - exp(-2 s x^2)) / x

The frequency weights are: a box of time, sin(T y) / (pi y) (sinc); the
exponential, the Lorentzian (gamma / pi) / (y^2 + gamma^2); the Gaussian, the
Gaussian of standard deviation sigma; the triangle, Fejer's
(1 - cos T y) / (pi T y^2); so the imaginary 'exponential' is the Lorentzian
(the eta form x / (x^2 + eta^2) of the real axis) read on the Matsubara
axis, where it shifts every pole by gamma. On the imaginary axis every
window (g >= 0) gives an R_w <= 1 / |x| that falls with |x| past the
zero-mode taper (no overshoot, no zero); on the real axis the Gaussian
overshoots the bare pole (x R up to 1.285) and the Fejer weight approaches
it in decaying oscillations. The real
'sinc' is no regularizer: it vanishes at x T = 2 pi k and swings between 0
and twice the bare pole forever (x R = 1 - cos(x T) does not tend to 1); it
is offered only to be compared.

The width
---------
One number per window, the half-weight point x_half (Hartree) where
x R_w(x) = 1/2 (rule H, either axis). It fixes the window's parameter:
box T = ln 2 / x_half; exponential gamma = x_half; imaginary Gaussian
sigma = x_half / (sqrt(2) v) and triangle T = u / x_half, with v and u the
roots PARQUET_RULE_H_IMAGINARY_GAUSSIAN and PARQUET_RULE_H_IMAGINARY_TRIANGLE;
real Gaussian sigma = x_half / (sqrt(2) u) and Fejer T = u / x_half
(PARQUET_RULE_H_REAL_GAUSSIAN, PARQUET_RULE_H_REAL_FEJER); Lorentzian
eta = x_half; sinc T = pi / (3 x_half); srg s = ln 2 / (2 x_half^2). The
default, x_half None, is the SRG's own at the run's eta_2b,
x_half = sqrt(ln 2 / (2 eta_2b)) (0.0833 Ha at eta_2b = 50), so every window
compares at the half-weight point of the static vertex it replaces; 'srg'
then takes s = eta_2b itself and is the default vertex to the bit (a check of
the plumbing). eta_2b = 0 is x_half = inf: an infinitely wide window, every
weight 0, the vertices FLEX's, as R_s at s = 0. The eh and the pp source
vertices may take their own widths (`x_half_eh`, `x_half_pp`).

Zero modes
----------
An imaginary-axis R_w jumps by 2 L(0) at x = 0: the average of an eh pole
pair is 0 at Omega = 0 exactly and -2 L(0) just above, so rounding of an
exact zero mode (H/STO-3G's spin flip comes out at 2.2e-16 or exactly 0.0
by machine, as the mean-field energies round) would change the run. Every
imaginary-axis weight is therefore multiplied by the taper
1 - exp(-(x/x0)^2), x0 = PARQUET_AVERAGED_TAPER_X0 (1e-7 Ha), evaluated as
Quack evaluates its regularizer: exactly 0 below 7.45e-16 Ha and exactly 1
above 6.12e-7 Ha, so a zero mode adds nothing whichever way rounding puts
it, as the SRG weight, exactly 0 there, does. The real-axis weights are
continuous and 0 at x = 0 and take no taper: a zero mode a few ulps off
weighs R'(0) times its rounding there. Without the eh TDA a mode near zero
carries eigenvectors of order Omega^-1/2, so an imaginary-axis weight, which
does not vanish as Omega -> 0+, gives it a vertex of order L(0) / Omega,
where the SRG's 2 s Omega keeps it finite: the taper removes only what
rounding makes of an exact zero, not an open shell's spin rotation at the
SCF residual (1e-7 to 1e-4 Ha, flagged 'parquet_eh_zero_mode'), nor a
closed shell's soft triplet near its instability: run the imaginary windows
under the eh TDA wherever an eh mode can come near zero.

What changes, what does not
---------------------------
Only the insertions: the kernels of the next two-body iteration and their
effective integrals read the averaged tensors (the change, the damping, DIIS
and the convergence test are the driver's, as for any builder). The
self-energy never reads the vertices; it changes only through the spectra
(Omega, M) the averaged kernels give. FLEX builds vertices nothing reads and
is unaffected.

`averaged_vertex_builder` gives one window per source vertex, the same at
every insertion and for every matrix element, and sees no one-body energies
(a width tied to the gap is given as a number and does not follow evPA or
qsPA). A scalar model of one insertion between two pair propagators,
integrated over frequency, gives an element-dependent imaginary-axis width,
of the order of the receiving pairs' energies. These builders are not a
validated method: in osPA on six of the paper's aug-cc-pVTZ molecules (Ne,
HF, H2O, BF, CO and N2) none of the 24 windows of whole source vertices
tried brought the mean absolute error against the reference IPs below the
static vertex's (250 meV; one tied it), and widening the window of the eh
vertex lowers every principal IP (the imaginary exponential, from the SRG's
half-weight point to 0.5 Ha with the pp vertex's kept there: by 0.08 to
0.39 eV), toward the reference for CO and N2 and away from it for the other
four.

Per insertion
-------------
Each insertion has its own argument in the frequency-dependent equations:
the eh vertex crossed into the eh kernel (the transverse particle-hole
channel ph-bar, role 'phbar_eh') at nu' - nu, the pp vertex into the eh
kernel ('pp_eh') at omega + nu + nu', and the eh vertex into the pp kernel
twice, direct (the longitudinal channel ph) at nu' - nu and swapped (ph-bar)
at -(omega + nu + nu'), one role ('eh_pp', below); the static approximation
takes them all at 0. The transverse channel should enter at another
frequency than the others, or its contribution comes out too large (Anna
Kauch, private communication); widening a whole eh source vertex damps its
insertion into the pp kernel with it. `insertion_vertex_builder` is a
per-insertion builder (the drivers call it once per vertex and role, with
the iteration's one-body energies, `InsertionVertexBuilder`) that gives each
role its own window, None being the static vertex of the default to the
bit, and `transverse_vertex_builder` windows the transverse insertion into
the eh kernel alone. A width may follow the iteration: x_half = 'gap' is the
HOMO-LUMO gap of the one-body energies the driver hands the builder
(`iteration_gap`), anew at every two-body iteration of evPA and qsPA.

"Another frequency" is a Matsubara one. Taken at the point omega = i gamma
(a point, not an average), an eh pole pair 1/(omega - Omega) -
1/(omega + Omega) = 2 Omega / (omega^2 - Omega^2) weighs
-2 Omega / (Omega^2 + gamma^2): the real-axis 'lorentzian' window
x / (x^2 + eta^2) at eta = gamma, so the role taken at i gamma is the window
('lorentzian', 'real', gamma) (the vertex's imaginary part there, gamma times
the difference of its resonant and antiresonant products, a real static
kernel cannot hold); its weight stays below the bare pole's 1 / Omega. A
real frequency omega_t would instead bring the poles closer,
2 Omega / (omega_t^2 - Omega^2), and enlarge the insertion it was to soften.

The two eh insertions into the pp kernel take one tensor because they are
exchange partners: exchanging the two particles of a pair turns the direct
term into the swapped one, and their transfers nu' - nu and
-(omega + nu + nu') into each other. A window of a role is the same for
every matrix element and cannot follow that exchange: one window on the
direct term and another on the swapped one would leave the pp kernel
D_pqrs -+ S_pqsr without its (anti)symmetry under the exchange, dependent on
the basis of pairs it is written in (the closed-shell and the spin-orbital
driver would differ on one closed shell), and most of their effect would be
that asymmetry (docs/methods/parquet.md, One tensor per insertion). Within
the pp kernel the transverse channel is so not apart from the longitudinal
one, and Kauch's prescription is the transverse insertion into the eh
kernel.

A NaN eigenvalue gives a NaN weight and so a NaN vertex there, which the
driver flags ('parquet_vertex_not_finite') as it does for any builder. An
eigenvalue of inf weighs 0, the limit of every window, but 'srg' evaluates the
default's own expression at every denominator, so it gives what the default
gives, NaN there at eta_2b = 0 included.

Cost: one static vertex (4 norb^4 nS flops for an eh vertex, 2 norb^4
(nVV + nOO) for a pp vertex; at norb 138, nO 5: 0.97 and 6.5 TFLOP) and one
evaluation of the window per pole; memory as the static vertex. A
per-insertion builder builds each eh vertex once per role, twice (at
norb 138 about 2 TFLOP more per two-body iteration, against some 21), and
the loop carries its six tensors instead of four (2 norb^4 numbers more;
over spin orbitals three instead of two, 1 nso^4 more), with DIIS vectors
half as long again (docs/methods/parquet.md, One tensor per insertion).
"""
import math

import numpy as np
from scipy import special

from src.Base.numerics.constants import (PARQUET_AVERAGED_FEJER_TERMS,
                                         PARQUET_AVERAGED_MIN_X_HALF,
                                         PARQUET_AVERAGED_SERIES_BELOW,
                                         PARQUET_AVERAGED_TAPER_X0,
                                         PARQUET_AVERAGED_TRIANGLE_TERMS,
                                         PARQUET_RULE_H_IMAGINARY_GAUSSIAN,
                                         PARQUET_RULE_H_IMAGINARY_TRIANGLE,
                                         PARQUET_RULE_H_REAL_FEJER, PARQUET_RULE_H_REAL_GAUSSIAN)
from src.SingleReference.Parquet import restricted_vertices, spin_orbital_two_body
from src.SingleReference.Parquet.options import checked_real
from src.SingleReference.Parquet.restricted import static_reducible_vertex
from src.SingleReference.Parquet.restricted_vertices import VERTEX_NAMES
from src.SingleReference.Parquet.spectra import EH_ROLES, PP_ROLES
from src.SingleReference.Parquet.spin_orbital import VERTEX_NAMES as SO_VERTEX_NAMES
from src.SingleReference.Parquet.spin_orbital import static_reducible_vertex_so

#: The roles of a per-insertion builder's windows, in the order
#: `insertion_vertex_builder` takes them (`spectra.EH_ROLES`, `PP_ROLES`).
ROLES = ('phbar_eh', 'eh_pp', 'pp_eh')
#: The x_half of a per-insertion window that follows the iteration: the
#: HOMO-LUMO gap of its one-body energies (`iteration_gap`).
GAP = 'gap'

#: The windows of each axis, in the order the module docstring lists them.
WINDOWS = {
    'imaginary': ('box', 'exponential', 'gaussian', 'triangle'),
    'real': ('gaussian', 'lorentzian', 'fejer', 'sinc', 'srg'),
}
#: The drivers, and the vertex names each calls a builder with (disjoint).
DRIVERS = {'restricted': VERTEX_NAMES, 'spin_orbital': SO_VERTEX_NAMES}
#: Each window's own parameter, by (axis, window): its name and unit.
PARAMETERS = {
    ('imaginary', 'box'): 'T (Ha^-1)',
    ('imaginary', 'exponential'): 'gamma (Ha)',
    ('imaginary', 'gaussian'): 'sigma (Ha)',
    ('imaginary', 'triangle'): 'T (Ha^-1)',
    ('real', 'gaussian'): 'sigma (Ha)',
    ('real', 'lorentzian'): 'eta (Ha)',
    ('real', 'fejer'): 'T (Ha^-1)',
    ('real', 'sinc'): 'T (Ha^-1)',
    ('real', 'srg'): 's (Ha^-2)',
}

LN2 = math.log(2.0)
SQRT2 = math.sqrt(2.0)
#: (u + expm1(-u)) / u^2 = sum_n (-1)^n u^n / (n + 2)!, below
#: PARQUET_AVERAGED_SERIES_BELOW (the registry gives the reason and the length)
TRIANGLE_SERIES = tuple((-1.0) ** n / math.factorial(n + 2)
                        for n in range(PARQUET_AVERAGED_TRIANGLE_TERMS))
#: (u - sin u) / u^3 = sum_n (-1)^n (u^2)^n / (2 n + 3)!
FEJER_SERIES = tuple((-1.0) ** n / math.factorial(2 * n + 3)
                     for n in range(PARQUET_AVERAGED_FEJER_TERMS))


def _polynomial(coefficients, z):
    """sum_n coefficients[n] z^n by Horner's rule."""
    out = np.zeros_like(z)
    for c in coefficients[::-1]:
        out = out * z + c
    return out


# The windows: L(p) (imaginary axis) or R(p) (real axis) for p > 0 finite, of the
# window's own parameter.
def _box(p, T):
    return -np.expm1(-T * p) / p


def _exponential(p, gamma):
    return 1.0 / (p + gamma)


def _gaussian_imaginary(p, sigma):
    return math.sqrt(math.pi / 2.0) / sigma * special.erfcx(p / (SQRT2 * sigma))


def _triangle(p, T):
    u = T * p
    out = np.empty_like(u)
    small = u < PARQUET_AVERAGED_SERIES_BELOW
    out[small] = T * _polynomial(TRIANGLE_SERIES, u[small])
    big = ~small
    out[big] = (1.0 + np.expm1(-u[big]) / u[big]) / p[big]
    return out


def _gaussian_real(p, sigma):
    return SQRT2 / sigma * special.dawsn(p / (SQRT2 * sigma))


def _lorentzian(p, eta):
    return 1.0 / (p + eta * eta / p)


def _fejer(p, T):
    u = T * p
    out = np.empty_like(u)
    small = u < PARQUET_AVERAGED_SERIES_BELOW
    us = u[small]
    out[small] = T * us * _polynomial(FEJER_SERIES, us * us)
    big = ~small
    ub = u[big]
    with np.errstate(invalid='ignore'):
        ratio = np.where(np.isinf(ub), 0.0, np.sin(ub) / ub)    # sin(u)/u -> 0
    out[big] = (1.0 - ratio) / p[big]
    return out


def _sinc(p, T):
    half = 0.5 * T * p
    with np.errstate(invalid='ignore'):
        # an argument that overflows takes the mean of 2 sin^2, the bare pole
        return np.where(np.isinf(half), 1.0, 2.0 * np.sin(half) ** 2) / p


def _srg(x, s):
    # the default's own expression (restricted_vertices and
    # spin_orbital_two_body.regularized_inverse) on the signed x, with its
    # limit 0 at x = 0: the same bits at every x, inf and NaN included
    r = (1.0 - np.exp(-2.0 * s * x * x)) / x
    return np.where(x == 0.0, 0.0, r)


#: (axis, window) -> (L or R on p > 0, the parameter at a finite x_half).
_WINDOWS = {
    ('imaginary', 'box'): (_box, lambda x: LN2 / x),
    ('imaginary', 'exponential'): (_exponential, lambda x: x),
    ('imaginary', 'gaussian'): (_gaussian_imaginary,
                                lambda x: x / (SQRT2 * PARQUET_RULE_H_IMAGINARY_GAUSSIAN)),
    ('imaginary', 'triangle'): (_triangle, lambda x: PARQUET_RULE_H_IMAGINARY_TRIANGLE / x),
    ('real', 'gaussian'): (_gaussian_real, lambda x: x / (SQRT2 * PARQUET_RULE_H_REAL_GAUSSIAN)),
    ('real', 'lorentzian'): (_lorentzian, lambda x: x),
    ('real', 'fejer'): (_fejer, lambda x: PARQUET_RULE_H_REAL_FEJER / x),
    ('real', 'sinc'): (_sinc, lambda x: math.pi / (3.0 * x)),
    ('real', 'srg'): (_srg, lambda x: (LN2 / 2.0) / x / x),
}


def _odd(magnitude, x, parameter):
    """sign(x) magnitude(|x|, parameter): 0 at x = 0 and at |x| = inf (the
    limit of every window, 1/x -> 0), NaN at NaN."""
    p = np.abs(x)
    out = np.zeros_like(p)
    inside = (p > 0.0) & (p < np.inf)
    out[inside] = magnitude(p[inside], parameter)
    out[np.isnan(p)] = np.nan
    return np.sign(x) * out


def zero_mode_taper(x):
    """1 - exp(-(x/x0)^2), x0 = PARQUET_AVERAGED_TAPER_X0, elementwise: the
    factor of every imaginary-axis weight, exactly 0 for |x| below
    2^-27 x0 = 7.45e-16 Ha and exactly 1 above 6.12 x0 (module docstring)."""
    x = np.asarray(x, dtype=float)
    # past |x| ~ 1e147 the square overflows: exp(-inf) = 0 and the taper is 1,
    # so the overflow is no event (as in the default's regularizer)
    with np.errstate(over='ignore'):
        return 1.0 - np.exp(-(x / PARQUET_AVERAGED_TAPER_X0) ** 2)


def srg_half_weight_point(eta_2b):
    """x_half = sqrt(ln 2 / (2 eta_2b)) (Hartree), where the SRG weight
    (1 - exp(-2 eta_2b x^2)) / x is half the bare pole; inf at eta_2b = 0.

    Parameters
    ----------
    eta_2b : float
        The two-body flow parameter s_2b (Hartree^-2), finite and >= 0.
    """
    eta_2b = checked_real('eta_2b', eta_2b, 'a finite flow parameter >= 0 (Ha^-2)')
    if not (math.isfinite(eta_2b) and eta_2b >= 0.0):
        raise ValueError(f'eta_2b = {eta_2b!r}: a finite flow parameter >= 0 (Ha^-2)')
    # the square roots taken apart, so that no eta_2b overflows (2 eta_2b would
    # past 9e307, and x_half would come out 0)
    return math.inf if eta_2b == 0.0 else math.sqrt(LN2 / 2.0) / math.sqrt(eta_2b)


def window_parameter(window, axis='imaginary', *, x_half=None, eta_2b=None):
    """The parameter of `window` on `axis` (`PARAMETERS`) at the half-weight
    point x_half by rule H, or, with x_half None, at the SRG's half-weight
    point at eta_2b (`srg_half_weight_point`; 'srg' then takes s = eta_2b
    itself). x_half = inf gives the infinitely wide window (T = 0, gamma,
    sigma, eta = inf, s = 0), whose weights are all 0.

    Parameters
    ----------
    window : str
        One of WINDOWS[axis].
    axis : {'imaginary', 'real'}
    x_half : float or None
        Hartree, > 0 (inf allowed).
    eta_2b : float or None
        Hartree^-2, finite and >= 0; read only where x_half is None.

    Returns
    -------
    float
    """
    axis = _checked_axis(axis)
    window = _checked_window(window, axis)
    x_half = _checked_half_weight('x_half', x_half)
    if x_half is None:
        if eta_2b is None:
            raise ValueError('x_half and eta_2b are both None: give the half-weight point, or '
                             'the eta_2b whose SRG half-weight point the window takes')
        x_half = srg_half_weight_point(eta_2b)
        if window == 'srg':
            return float(eta_2b)
    return _WINDOWS[axis, window][1](x_half)


def averaged_inverse(x, window, axis='imaginary', *, x_half=None, eta_2b=None):
    """The window-averaged odd inverse R_w(x) of the module docstring,
    elementwise: the weight of a pole whose denominator is x (Hartree).

    The window's width is its half-weight point x_half (rule H:
    x_half R_w(x_half) = 1/2), or, with x_half None, the SRG's at eta_2b
    (`window_parameter`). Imaginary-axis weights carry the zero-mode taper
    (`zero_mode_taper`). R_w(-x) = -R_w(x) to the bit, up to the sign of a
    zero; R_w(0) = 0; R_w(+-inf) = 0, the limit ('srg': as the default
    gives, NaN at s = 0); NaN gives NaN. Overflow inside a window (a
    denominator or a parameter past about 1e150) takes the window's limit
    and is no event, as in the default's regularizer.

    Parameters
    ----------
    x : array_like
        Denominators, Hartree.
    window, axis, x_half, eta_2b
        As `window_parameter`.

    Returns
    -------
    ndarray, shaped as x.
    """
    parameter = window_parameter(window, axis, x_half=x_half, eta_2b=eta_2b)
    x = np.asarray(x, dtype=float)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        if window == 'srg':
            return _srg(x, parameter)
        r = _odd(_WINDOWS[axis, window][0], x, parameter)
    if axis == 'imaginary':
        r = r * zero_mode_taper(x)
    return r


def averaged_vertex_builder(window, axis='imaginary', *, x_half=None, x_half_eh=None,
                            x_half_pp=None, pp_center=0.0, driver=None):
    """A `vertex_builder` that averages the reducible vertices over a
    frequency window (`AveragedVertexBuilder`).

    Parameters
    ----------
    window : str
        One of WINDOWS[axis]: imaginary 'box', 'exponential', 'gaussian',
        'triangle'; real 'gaussian', 'lorentzian', 'fejer', 'sinc', 'srg'.
    axis : {'imaginary', 'real'}
        The axis of the average (module docstring).
    x_half : float or None
        Half-weight point of both source vertices, Hartree (> 0, inf
        allowed); None is the SRG's at the run's eta_2b.
    x_half_eh, x_half_pp : float or None
        The eh and the pp source vertices' own, where given (None: x_half).
    pp_center : float
        The point c the pp weights are taken about, R_w(c - Omega^ee) and
        R_w(c - Omega^hh), Hartree; 0 is the default vertex's.
    driver : {None, 'restricted', 'spin_orbital'}
        None: either driver (the vertex names it is called with tell them
        apart); or run_parquet (ev_parquet, qs_parquet) or run_parquet_so
        (ev_parquet_so, qs_parquet_so) alone, a call from the other driver
        refused with a ValueError.

    Returns
    -------
    AveragedVertexBuilder

    Wrong arguments are refused with a ValueError naming the argument.

    Example: the box of imaginary time (Yirga and Campbell's central cluster
    on the Matsubara axis) at the static vertex's half-weight point,

        res = run_parquet(mf, 'os', max_diis_2b=5,
                          vertex_builder=averaged_vertex_builder('box'))
    """
    return AveragedVertexBuilder(window, axis, x_half=x_half, x_half_eh=x_half_eh,
                                 x_half_pp=x_half_pp, pp_center=pp_center, driver=driver)


class AveragedVertexBuilder:
    """The vertex builder of `averaged_vertex_builder`: called as
    builder(name, spectra, eta_2b) by its driver once per reducible vertex,
    it returns the static vertex of `name` with the window-averaged weights
    R_w (`averaged_inverse`) in place of the SRG's, contracted as the
    driver's default contracts (module docstring). Its `__name__` names
    every parameter, so the run's options record it
    (result.options['vertex_builder']); the eh and pp widths it records are
    None where they follow the run's eta_2b (`widths` gives them at an
    eta_2b).

    Attributes
    ----------
    window, axis, x_half_eh, x_half_pp, pp_center, driver
        As `averaged_vertex_builder` takes them (x_half_eh and x_half_pp
        with x_half in place of None).
    """

    def __init__(self, window, axis='imaginary', *, x_half=None, x_half_eh=None,
                 x_half_pp=None, pp_center=0.0, driver=None):
        self.axis = _checked_axis(axis)
        self.window = _checked_window(window, self.axis)
        x_half = _checked_half_weight('x_half', x_half)
        x_half_eh = _checked_half_weight('x_half_eh', x_half_eh)
        x_half_pp = _checked_half_weight('x_half_pp', x_half_pp)
        self.x_half_eh = x_half if x_half_eh is None else x_half_eh
        self.x_half_pp = x_half if x_half_pp is None else x_half_pp
        center = checked_real('pp_center', pp_center, 'a finite energy (Ha)')
        if not math.isfinite(center):
            raise ValueError(f'pp_center = {pp_center!r}: a finite energy (Ha)')
        self.pp_center = center
        if not (driver is None or (isinstance(driver, str) and driver in DRIVERS)):
            raise ValueError(f'driver = {driver!r}: None or one of {tuple(DRIVERS)}')
        self.driver = driver
        only = '' if driver is None else f', driver={driver!r}'
        self.__name__ = (f'averaged_vertex_builder({self.window!r}, {self.axis!r}, '
                         f'x_half_eh={self.x_half_eh!r}, x_half_pp={self.x_half_pp!r}, '
                         f'pp_center={self.pp_center!r}{only})')

    def __repr__(self):
        return self.__name__

    def widths(self, eta_2b):
        """{'eh': (x_half, parameter), 'pp': (x_half, parameter)} at this
        eta_2b: each source vertex's half-weight point (Hartree) and its
        window's parameter (`PARAMETERS`)."""
        return {'eh': self._width(self.x_half_eh, eta_2b),
                'pp': self._width(self.x_half_pp, eta_2b)}

    def _width(self, x_half, eta_2b):
        point = srg_half_weight_point(eta_2b) if x_half is None else x_half
        return point, window_parameter(self.window, self.axis, x_half=x_half, eta_2b=eta_2b)

    def _weights(self, x_half, d, eta_2b):
        return averaged_inverse(d, self.window, self.axis, x_half=x_half, eta_2b=eta_2b)

    def __call__(self, name, spectra, eta_2b):
        """The averaged vertex `name` of this iteration's `spectra`: a
        (norb,)*4 or (nso,)*4 array, as the driver's default returns."""
        if name in VERTEX_NAMES:
            driver = 'restricted'
        elif name in SO_VERTEX_NAMES:
            driver = 'spin_orbital'
        else:
            raise ValueError(f'unknown reducible vertex {name!r}: one of {VERTEX_NAMES} '
                             f'(closed shell) or {SO_VERTEX_NAMES} (spin orbitals)')
        if self.driver not in (None, driver):
            raise ValueError(f'vertex {name!r} comes from the {driver} driver; this '
                             f'builder was made for driver={self.driver!r}')
        if driver == 'restricted':
            return self._restricted(name, spectra, eta_2b)
        return self._spin_orbital(name, spectra, eta_2b)

    def _restricted(self, name, spectra, eta_2b):
        if name in ('eh_singlet', 'eh_triplet'):
            om, rho = getattr(spectra, name)
            om = np.asarray(om, dtype=float)
            w_res = self._weights(self.x_half_eh, 0.0 - om, eta_2b)
            w_anti = -self._weights(self.x_half_eh, 0.0 + om, eta_2b)
            return restricted_vertices.eh_vertex_from_weights(rho, w_res, w_anti)
        if name in ('pp_singlet', 'pp_triplet'):
            spin = name[3:]
            om_ee, rho_ee = getattr(spectra, 'ee_' + spin)
            om_hh, rho_hh = getattr(spectra, 'hh_' + spin)
            w_ee = self._weights(self.x_half_pp, self.pp_center - np.asarray(om_ee, dtype=float),
                                 eta_2b)
            w_hh = -self._weights(self.x_half_pp, self.pp_center - np.asarray(om_hh, dtype=float),
                                  eta_2b)
            return restricted_vertices.pp_vertex_from_weights(rho_ee, w_ee, rho_hh, w_hh)

    def _spin_orbital(self, name, spectra, eta_2b):
        if name == 'eh':
            om = np.asarray(spectra.eh.om, dtype=float)
            w_res = self._weights(self.x_half_eh, 0.0 - om, eta_2b)
            w_anti = self._weights(self.x_half_eh, 0.0 + om, eta_2b)
            return spin_orbital_two_body.eh_vertex_from_weights(spectra.eh.rho, w_res, w_anti)
        if name == 'pp':
            w_ee = self._weights(self.x_half_pp,
                                 self.pp_center - np.asarray(spectra.ee.om, dtype=float), eta_2b)
            w_hh = self._weights(self.x_half_pp,
                                 self.pp_center - np.asarray(spectra.hh.om, dtype=float), eta_2b)
            return spin_orbital_two_body.pp_vertex_from_weights(spectra.ee.rho, w_ee,
                                                                spectra.hh.rho, w_hh)


def iteration_gap(eps, nO):
    """The HOMO-LUMO gap min eps[nO:] - max eps[:nO] of one-body energies,
    Hartree (eps[nO] - eps[nO - 1] where they ascend): the half-weight point
    a per-insertion window with x_half 'gap' takes, from the energies of the
    BSE diagonals the driver hands the builder at each two-body iteration
    (the reference's in osPA, eOld in evPA, the current eQP in qsPA).

    Parameters
    ----------
    eps : array_like, shape (n,)
        One-body energies, occupied first (closed-shell orbitals or spin
        orbitals).
    nO : int
        The occupied ones, 0 < nO < n.
    """
    eps = np.asarray(eps, dtype=float)
    if eps.ndim != 1 or not 0 < int(nO) < eps.shape[0]:
        raise ValueError(f'eps of shape {eps.shape} with nO = {nO!r}: one-body energies, '
                         f'occupied first, and 0 < nO < their number')
    return float(eps[int(nO):].min() - eps[:int(nO)].max())


def insertion_vertex_builder(phbar_eh=None, eh_pp=None, pp_eh=None, *, pp_center=0.0,
                             driver=None):
    """A per-insertion `vertex_builder` that takes each role of the
    reducible vertices at a window of its own (`InsertionVertexBuilder`).

    Parameters
    ----------
    phbar_eh, eh_pp, pp_eh : None, str or tuple
        The window of each role (module docstring, Per insertion): the eh
        vertex crossed into the eh kernel (the transverse channel,
        'phbar_eh') and into the pp kernel (its direct and swapped terms
        alike, 'eh_pp'), the pp vertex into the eh kernel ('pp_eh'). None:
        the static vertex, the default's to the bit
        (`restricted.static_reducible_vertex`,
        `spin_orbital.static_reducible_vertex_so`). Otherwise
        (window, axis, x_half) as `averaged_inverse` takes them, axis
        'imaginary' and x_half None (the SRG's half-weight point at the
        run's eta_2b) where left out: a window name alone, or a tuple of
        one to three; x_half 'gap' follows the iteration
        (`iteration_gap`).
    pp_center : float
        The point c a windowed pp_eh weighs its poles about, R_w(c - Omega),
        Hartree, as `averaged_vertex_builder` takes it; with pp_eh None the
        pp vertex is the static one at 0, and another pp_center is refused.
    driver : {None, 'restricted', 'spin_orbital'}
        As `averaged_vertex_builder`.

    Returns
    -------
    InsertionVertexBuilder

    Wrong arguments are refused with a ValueError naming the argument.

    Example: the transverse insertion at the Matsubara point i gamma
    (gamma = 0.25 Ha), the other roles static,

        res = run_parquet(mf, 'os', max_diis_2b=5, vertex_builder=insertion_vertex_builder(
            phbar_eh=('lorentzian', 'real', 0.25)))
    """
    return InsertionVertexBuilder(phbar_eh, eh_pp, pp_eh, pp_center=pp_center, driver=driver)


def transverse_vertex_builder(window, axis='imaginary', x_half=None, *, driver=None):
    """The per-insertion builder of Kauch's prescription: the transverse
    insertion into the eh kernel ('phbar_eh') at the window (window, axis,
    x_half), the eh vertex in the pp kernel ('eh_pp') and the pp vertex
    ('pp_eh') static: `insertion_vertex_builder` with phbar_eh = (window,
    axis, x_half). In the pp kernel the transverse term is the exchange
    partner of the longitudinal one and shares its tensor (module docstring,
    Per insertion).

        res = run_parquet(mf, 'ev', max_diis_2b=5,
                          vertex_builder=transverse_vertex_builder('exponential', x_half='gap'))
    """
    return InsertionVertexBuilder(phbar_eh=(window, axis, x_half), driver=driver)


class InsertionVertexBuilder:
    """The per-insertion vertex builder of `insertion_vertex_builder`. Its
    attribute per_insertion is True: the drivers call it once per reducible
    vertex and role at every two-body iteration, as
    builder(name, role, spectra, eta_2b, eps=eps, nO=nO) (the roles of
    `spectra.EH_ROLES` for an eh vertex, `spectra.PP_ROLES` for a pp one),
    and it returns the vertex `name` of this iteration's spectra with the
    weights of that role's window: the default's static vertex for a role
    without one, else the vertex `AveragedVertexBuilder` gives for that
    window, its 'gap' resolved from eps and nO. Its `__name__` names every
    role's window, so the run's options record it.

    Attributes
    ----------
    windows : dict
        {role: None or (window, axis, x_half)}, as checked.
    pp_center, driver
        As `insertion_vertex_builder` takes them.
    """
    per_insertion = True

    def __init__(self, phbar_eh=None, eh_pp=None, pp_eh=None, *, pp_center=0.0, driver=None):
        given = dict(phbar_eh=phbar_eh, eh_pp=eh_pp, pp_eh=pp_eh)
        self.windows = {role: _checked_spec(role, given[role]) for role in ROLES}
        center = checked_real('pp_center', pp_center, 'a finite energy (Ha)')
        if not math.isfinite(center):
            raise ValueError(f'pp_center = {pp_center!r}: a finite energy (Ha)')
        if self.windows['pp_eh'] is None and center != 0.0:
            raise ValueError(f'pp_center = {pp_center!r}: the point a windowed pp_eh is taken '
                             f'about; with pp_eh None the pp vertex is the static one, at 0')
        self.pp_center = center
        if not (driver is None or (isinstance(driver, str) and driver in DRIVERS)):
            raise ValueError(f'driver = {driver!r}: None or one of {tuple(DRIVERS)}')
        self.driver = driver
        roles = ', '.join(f'{role}={self.windows[role]!r}' for role in ROLES)
        only = '' if driver is None else f', driver={driver!r}'
        self.__name__ = (f'insertion_vertex_builder({roles}, pp_center={self.pp_center!r}'
                         f'{only})')

    def __repr__(self):
        return self.__name__

    def widths(self, eta_2b, eps=None, nO=None):
        """{role: (x_half, parameter) or None} at this eta_2b: each windowed
        role's half-weight point (Hartree) and its window's parameter
        (`PARAMETERS`), a 'gap' one's at the one-body energies eps with nO
        occupied (needed then); None for a static role."""
        return {role: self._width(role, eta_2b, eps, nO) for role in ROLES}

    def _width(self, role, eta_2b, eps, nO):
        if self.windows[role] is None:
            return None
        window, axis, x_half = self.windows[role]
        if x_half == GAP:
            if eps is None or nO is None:
                raise ValueError(f"the {role} window follows the gap: give eps and nO")
            x_half = self._gap(role, eps, nO)
        point = srg_half_weight_point(eta_2b) if x_half is None else x_half
        return point, window_parameter(window, axis, x_half=x_half, eta_2b=eta_2b)

    def _gap(self, role, eps, nO):
        gap = iteration_gap(eps, nO)
        if not gap > 0.0:
            raise ValueError(f"the {role} window follows the gap, and the HOMO-LUMO gap of this "
                             f"iteration's one-body energies is {gap!r} Ha: 'gap' takes a "
                             f"positive one (a virtual level fell below an occupied one, or an "
                             f"energy is NaN)")
        return gap

    def __call__(self, name, role, spectra, eta_2b, *, eps, nO):
        """The vertex `name` of this iteration's `spectra` for its insertion
        `role`: a (norb,)*4 or (nso,)*4 array, as the driver's default
        returns."""
        if name in VERTEX_NAMES:
            driver = 'restricted'
        elif name in SO_VERTEX_NAMES:
            driver = 'spin_orbital'
        else:
            raise ValueError(f'unknown reducible vertex {name!r}: one of {VERTEX_NAMES} '
                             f'(closed shell) or {SO_VERTEX_NAMES} (spin orbitals)')
        if self.driver not in (None, driver):
            raise ValueError(f'vertex {name!r} comes from the {driver} driver; this '
                             f'builder was made for driver={self.driver!r}')
        roles = EH_ROLES if name in ('eh_singlet', 'eh_triplet', 'eh') else PP_ROLES
        if role not in roles:
            raise ValueError(f'role {role!r} of the {name!r} vertex: one of {roles}')
        if self.windows[role] is None:
            if driver == 'restricted':
                return static_reducible_vertex(name, spectra, eta_2b)
            return static_reducible_vertex_so(name, spectra, eta_2b)
        window, axis, x_half = self.windows[role]
        if x_half == GAP:
            x_half = self._gap(role, eps, nO)
        averaged = AveragedVertexBuilder(window, axis, x_half=x_half, pp_center=self.pp_center,
                                         driver=driver)
        return averaged(name, spectra, eta_2b)


def _checked_spec(role, spec):
    """The window of `role` as (window, axis, x_half), or None: a window name
    alone, or a tuple (or list) of one to three, axis 'imaginary' and x_half
    None where left out; x_half None, 'gap' or a half-weight point. A
    ValueError names the role."""
    if spec is None:
        return None
    parts = (spec,) if isinstance(spec, str) else spec
    if not isinstance(parts, (tuple, list)) or not 1 <= len(parts) <= 3:
        raise ValueError(f'{role} = {spec!r}: None (the static vertex), a window name, or '
                         f'(window, axis, x_half) with axis and x_half optional')
    window, axis, x_half = tuple(parts) + ('imaginary', None)[len(parts) - 1:]
    try:
        axis = _checked_axis(axis)
        window = _checked_window(window, axis)
    except ValueError as err:
        raise ValueError(f'{role} = {spec!r}: {err}') from None
    if not (isinstance(x_half, str) and x_half == GAP):
        try:
            x_half = _checked_half_weight('x_half', x_half)
        except ValueError as err:
            raise ValueError(f"{role} = {spec!r}: {err}, or 'gap' (the iteration's HOMO-LUMO "
                             f"gap)") from None
    return window, axis, x_half


def _checked_axis(axis):
    if not (isinstance(axis, str) and axis in WINDOWS):
        raise ValueError(f'axis = {axis!r}: one of {tuple(WINDOWS)}')
    return axis


def _checked_window(window, axis):
    if not (isinstance(window, str) and window in WINDOWS[axis]):
        raise ValueError(f'window = {window!r}: on the {axis} axis one of {WINDOWS[axis]}')
    return window


def _checked_half_weight(name, value):
    """None, or a half-weight point as a float: at least
    PARQUET_AVERAGED_MIN_X_HALF Hartree (narrower, a window's parameter and
    its arguments overflow), inf allowed."""
    if value is None:
        return None
    what = f'a half-weight point >= {PARQUET_AVERAGED_MIN_X_HALF:g} Ha (inf allowed), or None'
    number = checked_real(name, value, what)
    if not number >= PARQUET_AVERAGED_MIN_X_HALF:
        raise ValueError(f'{name} = {value!r}: {what}')
    return number
