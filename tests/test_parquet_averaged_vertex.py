"""The frequency-averaged reducible vertices (`Parquet.averaged_vertices`).

The window-averaged odd inverse R_w that takes the place of the SRG weight of
the static vertex, and the builders that hand the averaged vertices to the
`vertex_builder` hook of both drivers. Held:

* every window's R_w against an independent quadrature, at small, medium
  and large |x| of both signs and at two widths: the imaginary windows
  against the Matsubara integral x int w(nu) / (nu^2 + x^2) dnu of their
  frequency weight (Yirga and Campbell's sinc weight gives the box), the
  real Gaussian, Lorentzian and SRG against the principal value
  P.V. int w(y) / (x - y) dy of theirs (scipy's Cauchy weight; the SRG's is
  2 F(sqrt(2 s) y) / (pi^1.5 y), F Dawson's function, positive and
  normalized, of half width at half maximum 0.7517 / sqrt(s)), the real
  Fejer and sinc against the sine transform of their time window;
* R_w odd to the bit; x R_w -> 1 as |x| grows, within each window's leading
  correction, but for the real sinc, whose x R = 1 - cos(x T) swings between
  0 and 2 forever; rule H, x_half R_w(x_half) = 1/2 at the first crossing,
  for every window, and the registry's roots found again by bracketing;
* the zero-mode taper: every imaginary weight is exactly 0 at x = 0.0, -0.0,
  +-1e-16 and +-2.2e-16, where the untapered average is +-L(0); an
  infinitely wide window (eta_2b = 0, or x_half = inf) weighs nothing;
* the builders against an independent einsum pole sum of Eqs. (37) and (43)
  on the spectra a short osPA run handed its builder, in both drivers,
  singlet and triplet, eh and pp, with separate eh and pp widths and a pp
  centre; 'srg' matched to eta_2b gives each driver's static vertex to the
  bit (its weights and the contraction), and a run with it the default run;
  a very narrow imaginary window gives the static vertex at a huge eta_2b;
* runs: osPA, evPA and qsPA with the imaginary box and exponential windows
  on water/6-31G (TDA, s_1b = s_2b = 50) converge, finite and unflagged,
  away from the static run; the spin-orbital driver on that closed shell
  agrees with the closed-shell one with averaged windows (s_1b = 1e8,
  linearized, fixed two-body counts), as the static vertices agree
  (`test_parquet_spin_adaptation`); H/STO-3G's spin flip forced to 0.0,
  -0.0, +-1e-16 and +-2.2e-16 gives the same run to the bit with imaginary
  windows, its eh vertex zero (`test_parquet_small_systems` holds the
  default so; untapered, the flip, whose effective integral is about -J,
  would weigh +-L(0) J^2 a few ulps off, though H's energies would not
  move), and LiH/STO-3G's lowest triplet forced to 0.0, -0.0 and 1e-16
  against 2.2e-16 the same energies to rounding (untapered, eqp would move
  by 1 mHa);
* one builder serves both drivers, a builder made for one driver refuses
  the other's vertices; extreme widths and denominators give finite weights,
  the bare pole where the window is narrow, with no overflow warning; 'srg'
  is the default's expression at every denominator, +-inf and NaN included;
* the options record names the window and its widths, and wrong arguments
  are refused with a ValueError naming them.

Run: python tests/test_parquet_averaged_vertex.py, or under pytest.
"""
import functools
import math
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from scipy import integrate, optimize, special

from src.Base.numerics import constants
from src.Base.numerics.constants import HARTREE_TO_EV, PARQUET_VERTEX_DAMPING
from src.SingleReference.Parquet import restricted, restricted_vertices, spin_orbital
from src.SingleReference.Parquet import spin_orbital_two_body
from src.SingleReference.Parquet.averaged_vertices import (WINDOWS, averaged_inverse,
                                                           averaged_vertex_builder,
                                                           srg_half_weight_point,
                                                           window_parameter, zero_mode_taper)
from src.SingleReference.Parquet.restricted import static_reducible_vertex
from src.SingleReference.Parquet.spin_orbital import static_reducible_vertex_so
from tests.pins import assert_same
from tests.test_parquet_small_systems import (CLOSED_ZERO_MODE_GAP, CLOSED_ZERO_MODE_Z,
                                              SPIN_FLIP_VALUES, ULPS_OFF, ZERO_MODE_BELOW,
                                              ZERO_MODE_RUNS, flag_names, forced_triplet_rhf,
                                              lowest_triplet_at, open_shell, spin_flip_at)
from tests.test_parquet_small_systems import assert_same_run as assert_same_bits
from tests.test_parquet_spin_adaptation import (CROSS_DRIVER, CROSS_DRIVER_Z,
                                                assert_same_two_body, both_layouts, rhf)

#: Every (axis, window).
ALL = [(axis, window) for axis, windows in WINDOWS.items() for window in windows]
#: Ha. The SRG's half-weight points at s = 50 and s = 1 (0.0833 and 0.589 Ha).
X_HALF_50 = srg_half_weight_point(50.0)
X_HALF_1 = srg_half_weight_point(1.0)
#: Ha. Where each window is held to quadrature: small, medium and large, both
#: signs (eh excitations of water start near 0.3 Ha, pp ones near 0.7).
QUADRATURE_X = (1e-3, -1e-3, 0.05, -0.05, 0.3, -0.3, 2.0, -2.0, 30.0, -30.0)
#: Relative bar against quadrature: 1e-12 measured at most (the SRG's
#: principal value with its 1/y^2 tail), 3e-14 for the other real weights and
#: 2e-15 on the imaginary axis.
QUADRATURE_RTOL = 1e-10
#: |x_half R_w(x_half) - 1/2| allowed: a few ulps of 1/2 (4.4e-16 measured).
RULE_H_TOL = 4e-15
#: Half width at half maximum of the SRG weight times sqrt(s), Ha^(1/2): the
#: root of F(u) / u = 1/2 over sqrt(2) (1.0630611 / sqrt(2), mpmath).
SRG_HWHM = 0.7516976882353909
#: Relative bar of a builder against the einsum pole sum (other summation
#: orders): 3.5e-16 measured.
POLE_SUM_RTOL = 1e-12
#: Hartree^-2. The 'huge' s_2b at which the static vertex is the bare 1/Omega
#: one, and the half-weight point (Ha) of the narrow imaginary windows held
#: to it; their relative gap is at most about x_half / min |Omega| (the
#: exponential's level shift: 2.8e-9 measured, the triangle 1.8e-9, the
#: Gaussian 3e-16 and the box 0, each linear in x_half or faster).
HUGE_ETA_2B = 1e8
NARROW_X_HALF = 1e-9
NARROW_RTOL = 1e-8
#: Ha. How far a run's principal IP must lie from the static run's for the
#: window to count as having acted (8.7 meV is the least measured: the box in
#: osPA; 18 to 54 meV for the others).
ACTED = 1e-4
#: The averaged runs on water/6-31G: (scheme, options), TDA, s_1b = s_2b = 50
#: (Table III), two-body DIIS 5; the quasiparticle equation linearized and
#: qsPA with one-body DIIS 3 (4 one-body iterations), so that every run is
#: short (0.2 to 1.1 s).
RUNS = {
    'os': dict(eta_1b=50.0, eta_2b=50.0, max_diis_2b=5, qp_solver='linearized'),
    'ev': dict(eta_1b=50.0, eta_2b=50.0, max_diis_2b=5, qp_solver='linearized'),
    'qs': dict(eta_1b=50.0, eta_2b=50.0, max_diis_2b=5, max_diis_1b=3),
}
#: The fixed-count comparison of the two drivers (test_parquet_spin_adaptation's
#: 'os_tda_cap'), with averaged vertices.
TWO_DRIVERS = dict(tda_eh=True, tda_pp=True, eta_1b=1e8, eta_2b=1.0, qp_solver='linearized',
                   max_it_2b=5, conv_2b=1e-12, damping=PARQUET_VERTEX_DAMPING)
#: Builders of the comparisons with einsum and of the two drivers: the
#: factory's keywords (window first), with separate widths and pp centres.
BUILDERS = {
    'imaginary box': dict(window='box'),
    'imaginary exponential, own pp width, pp centre': dict(window='exponential',
                                                           x_half_pp=0.25, pp_center=-0.5),
    'imaginary gaussian': dict(window='gaussian', x_half=0.2),
    'imaginary triangle, own eh width': dict(window='triangle', x_half_eh=0.15),
    'real gaussian, pp centre': dict(window='gaussian', axis='real', pp_center=-0.4),
    'real lorentzian': dict(window='lorentzian', axis='real'),
    'real fejer': dict(window='fejer', axis='real', x_half_pp=0.3),
    'real sinc': dict(window='sinc', axis='real'),
    'real srg at its own width': dict(window='srg', axis='real', x_half=0.12),
}


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def quad(f, a, b, **kwargs):
    """scipy's quad to 1e-12 (an absolute floor where QUADPACK's infinite
    Fourier integrals need one), its warnings silenced: the result is held
    to the bar, not to quad's estimate."""
    fourier = kwargs.get('weight') in ('sin', 'cos') and b == np.inf
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return integrate.quad(f, a, b, epsabs=1e-15 if fourier else 0.0, epsrel=1e-12,
                              limit=2000, **kwargs)[0]


def matsubara(window, x, parameter):
    """x int w(nu) / (nu^2 + x^2) dnu over the whole Matsubara axis, w the
    window's normalized frequency weight: the box's sin(T nu) / (pi nu), the
    exponential's Lorentzian, the Gaussian, the triangle's Fejer weight
    (1 - cos T nu) / (pi T nu^2); the oscillating tails by QUADPACK's Fourier
    integrals."""
    p = abs(x)

    def lorentz(nu):
        return 2.0 * p / (nu * nu + p * p)
    if window == 'exponential':
        def w(nu):
            return parameter / math.pi / (nu * nu + parameter * parameter)
        value = quad(lambda nu: w(nu) * lorentz(nu), 0.0, 10 * p)
        value += quad(lambda nu: w(nu) * lorentz(nu), 10 * p, np.inf)
    elif window == 'gaussian':
        def w(nu):
            return math.exp(-0.5 * (nu / parameter) ** 2) / (parameter * math.sqrt(2 * math.pi))
        top = 40 * parameter
        cut = min(10 * p, top)
        value = quad(lambda nu: w(nu) * lorentz(nu), 0.0, cut)
        value += quad(lambda nu: w(nu) * lorentz(nu), cut, top) if top > cut else 0.0
    else:
        T = parameter
        far = max(50.0 / T, 10 * p)
        if window == 'box':
            def near(nu):
                return T / math.pi * np.sinc(T * nu / math.pi) * lorentz(nu)
        else:
            def near(nu):
                return T / (2 * math.pi) * np.sinc(T * nu / (2 * math.pi)) ** 2 * lorentz(nu)
        cut = min(10 * p, far)
        value = quad(near, 0.0, cut) + (quad(near, cut, far) if far > cut else 0.0)
        if window == 'box':
            value += quad(lambda nu: lorentz(nu) / (math.pi * nu), far, np.inf, weight='sin',
                          wvar=T)
        else:
            value += quad(lambda nu: lorentz(nu) / (math.pi * T * nu * nu), far, np.inf)
            value -= quad(lambda nu: lorentz(nu) / (math.pi * T * nu * nu), far, np.inf,
                          weight='cos', wvar=T)
    return math.copysign(value, x)


def srg_weight(y, s):
    """w_s(y) = 2 F(sqrt(2 s) y) / (pi^1.5 y), F Dawson's function."""
    c = math.sqrt(2.0 * s)
    u = c * y
    return 2.0 * c / math.pi ** 1.5 * (1.0 if u == 0.0 else special.dawsn(u) / u)


def principal_value(window, x, parameter):
    """P.V. int w(y) / (x - y) dy of the real Gaussian (std sigma),
    Lorentzian and SRG weights: scipy's Cauchy weight on [-L, L], and the
    tails int_L^inf w(y) 2 x / (x^2 - y^2) dy."""
    if window == 'gaussian':
        def w(y):
            return math.exp(-0.5 * (y / parameter) ** 2) / (parameter * math.sqrt(2 * math.pi))
        top, tails = abs(x) + 40 * parameter, False
    elif window == 'lorentzian':
        def w(y):
            return parameter / math.pi / (y * y + parameter * parameter)
        top, tails = abs(x) + 100 * parameter, True
    else:
        def w(y):
            return srg_weight(y, parameter)
        top, tails = abs(x) + 100 / math.sqrt(2.0 * parameter), True
    value = -quad(w, -top, top, weight='cauchy', wvar=x)
    if tails:
        value += quad(lambda y: w(y) * 2 * x / (x * x - y * y), top, np.inf)
    return value


def sine_transform(window, x, T):
    """int_0^T sin(x t) g(t) dt of the real Fejer (triangle window) and sinc
    (box window) weights."""
    g = (lambda t: 1.0 - t / T) if window == 'fejer' else (lambda t: 1.0)
    return math.copysign(quad(g, 0.0, T, weight='sin', wvar=abs(x)), x)


def independent(axis, window, x, parameter):
    if axis == 'imaginary':
        return matsubara(window, x, parameter)
    if window in ('fejer', 'sinc'):
        return sine_transform(window, x, parameter)
    return principal_value(window, x, parameter)


@pytest.mark.parametrize('axis, window', ALL)
def test_each_window_against_quadrature(axis, window):
    for x_half in (X_HALF_50, 0.25):
        parameter = window_parameter(window, axis, x_half=x_half)
        got = averaged_inverse(np.array(QUADRATURE_X), window, axis, x_half=x_half)
        for x, r in zip(QUADRATURE_X, got):
            want = independent(axis, window, x, parameter)
            assert abs(r - want) <= QUADRATURE_RTOL * abs(want), (axis, window, x_half, x, r,
                                                                  want)


def test_the_srg_weight():
    """The weight whose principal-value average is the SRG regularizer is
    positive, normalized, and 0.7517 / sqrt(s) Ha wide at half maximum."""
    for s in (1.0, 50.0):
        y = np.linspace(0.0, 50.0 / math.sqrt(s), 2001)
        assert all(srg_weight(v, s) > 0.0 for v in y)
        norm = 2.0 * quad(lambda v: srg_weight(v, s), 0.0, np.inf)
        assert abs(norm - 1.0) < 1e-10, (s, norm)
        top = srg_weight(0.0, s)
        half = optimize.brentq(lambda v: srg_weight(v, s) - top / 2, 1e-9, 10.0 / math.sqrt(s),
                               xtol=1e-15)
        assert abs(half * math.sqrt(s) - SRG_HWHM) < 1e-12, (s, half)


@pytest.mark.parametrize('axis, window', ALL)
def test_odd_and_the_bare_pole_far_out(axis, window):
    """R_w(-x) = -R_w(x) to the bit, 0 at 0 and at +-inf, NaN at NaN; and
    x R_w -> 1 within the window's leading correction (but the real sinc)."""
    x = np.concatenate([np.geomspace(1e-12, 1e4, 80), [np.inf]])
    for eta_2b in (1.0, 50.0):
        r = averaged_inverse(x, window, axis, eta_2b=eta_2b)
        assert np.array_equal(averaged_inverse(-x, window, axis, eta_2b=eta_2b), -r)
        assert averaged_inverse(0.0, window, axis, eta_2b=eta_2b) == 0.0
        assert r[-1] == 0.0 and np.isnan(averaged_inverse(np.nan, window, axis, eta_2b=eta_2b))
        p = window_parameter(window, axis, eta_2b=eta_2b)
        far = np.array([10.0, 100.0, 1e3, 1e4])
        off = far * averaged_inverse(far, window, axis, eta_2b=eta_2b) - 1.0
        bound = {('imaginary', 'box'): np.exp(-p * far),
                 ('imaginary', 'exponential'): p / far,
                 ('imaginary', 'gaussian'): (p / far) ** 2,
                 ('imaginary', 'triangle'): 1.0 / (p * far),
                 ('real', 'gaussian'): 2.0 * (p / far) ** 2,
                 ('real', 'lorentzian'): (p / far) ** 2,
                 ('real', 'fejer'): 1.0 / (p * far),
                 ('real', 'srg'): np.exp(-2.0 * p * far * far)}.get((axis, window))
        if bound is None:     # the real sinc: x R = 1 - cos(x T), 0 to 2 forever
            k = np.array([1e2, 1e3, 1e4])
            node, crest = 2 * math.pi * k / p, (2 * k + 1) * math.pi / p
            assert np.all(np.abs(node * averaged_inverse(node, window, axis, eta_2b=eta_2b))
                          < 1e-8)
            assert np.allclose(crest * averaged_inverse(crest, window, axis, eta_2b=eta_2b), 2.0)
            continue
        assert np.all(np.abs(off) <= bound + 4e-16), (axis, window, off, bound)


@pytest.mark.parametrize('axis, window', ALL)
def test_rule_h(axis, window):
    """x_half R_w(x_half) = 1/2 at every width, the first crossing; the
    default width is the SRG's half-weight point at eta_2b."""
    for x_half in (1e-3, X_HALF_50, 0.25, X_HALF_1, 3.0):
        r = averaged_inverse(x_half, window, axis, x_half=x_half)
        assert abs(x_half * r - 0.5) <= RULE_H_TOL, (axis, window, x_half, x_half * r)
        below = x_half * np.linspace(0.01, 0.999, 60)
        assert np.all(below * averaged_inverse(below, window, axis, x_half=x_half) < 0.5)
    for eta_2b in (1.0, 10.0, 50.0, 1e4):
        x_half = srg_half_weight_point(eta_2b)
        r = averaged_inverse(x_half, window, axis, eta_2b=eta_2b)
        assert abs(x_half * r - 0.5) <= RULE_H_TOL, (axis, window, eta_2b)
    if window == 'srg':
        assert window_parameter('srg', 'real', eta_2b=50.0) == 50.0


def test_the_registry_roots_are_found_again():
    """Rule H's roots without a closed form, found by bracketing to a few ulps;
    the triangle's also as 2 + W_0(-2 exp(-2))."""
    equations = {
        'PARQUET_RULE_H_IMAGINARY_GAUSSIAN': (
            lambda v: math.sqrt(math.pi) * v * special.erfcx(v) - 0.5, 0.3, 0.6),
        'PARQUET_RULE_H_IMAGINARY_TRIANGLE': (lambda u: -math.expm1(-u) / u - 0.5, 1.0, 2.0),
        'PARQUET_RULE_H_REAL_GAUSSIAN': (lambda u: 2.0 * u * special.dawsn(u) - 0.5, 0.3, 0.8),
        'PARQUET_RULE_H_REAL_FEJER': (lambda u: math.sin(u) / u - 0.5, 1.5, 2.5),
    }
    for name, (f, a, b) in equations.items():
        root = optimize.brentq(f, a, b, xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=200)
        value = getattr(constants, name)
        assert abs(root - value) <= 4 * np.spacing(value), (name, root, value)
    lambert = 2.0 + special.lambertw(-2.0 * math.exp(-2.0)).real
    assert abs(lambert - constants.PARQUET_RULE_H_IMAGINARY_TRIANGLE) <= 4 * np.spacing(lambert)


@pytest.mark.parametrize('window', WINDOWS['imaginary'])
def test_the_taper_takes_out_a_zero_mode(window):
    """Exactly 0 where rounding puts an exact zero mode, of either sign;
    just past the taper the one-sided limit L(0) the taper keeps from a
    mode a few ulps off."""
    zeros = np.array([0.0, -0.0, 1e-16, -1e-16, ULPS_OFF, -ULPS_OFF])
    for eta_2b in (1.0, 50.0):
        assert np.all(averaged_inverse(zeros, window, eta_2b=eta_2b) == 0.0)
        p = window_parameter(window, eta_2b=eta_2b)
        l0 = {'box': p, 'exponential': 1.0 / p, 'gaussian': math.sqrt(math.pi / 2) / p,
              'triangle': p / 2}[window]
        past = averaged_inverse(np.array([1e-6, -1e-6]), window, eta_2b=eta_2b)
        assert past == pytest.approx([l0, -l0], rel=1e-4), (window, eta_2b, past, l0)
    assert np.all(zero_mode_taper(np.array([7.4e-16, -7.4e-16])) == 0.0)
    assert np.all(zero_mode_taper(np.array([6.2e-7, -6.2e-7, 1e-3, np.inf])) == 1.0)


@pytest.mark.parametrize('axis, window', ALL)
def test_an_infinitely_wide_window_weighs_nothing(axis, window):
    x = np.array([-30.0, -0.3, -1e-3, -ULPS_OFF, 0.0, ULPS_OFF, 1e-3, 0.3, 30.0])
    assert not averaged_inverse(x, window, axis, eta_2b=0.0).any()
    assert not averaged_inverse(x, window, axis, x_half=np.inf).any()
    assert srg_half_weight_point(0.0) == np.inf


def capture(default, into):
    """A builder that is `default` and keeps the spectra it was handed last."""
    def build(name, spectra, eta_2b):
        into['spectra'] = spectra
        return default(name, spectra, eta_2b)
    return build


@functools.lru_cache(maxsize=None)
def captured_spectra(driver):
    """The spectra of the third two-body iteration of an osPA on water/6-31G
    (s_2b = 50), as `driver`'s builder is handed them."""
    (eps, eri, nO), (eps_so, eri_so, nO_so, pairs) = both_layouts('water')
    into = {}
    options = dict(eta_2b=50.0, max_it_2b=3, conv_2b=1e-12, qp_solver='linearized')
    if driver == 'restricted':
        quiet(restricted.ev_parquet, eps, eri, nO, 'os',
              vertex_builder=capture(static_reducible_vertex, into), **options)
    else:
        quiet(spin_orbital.ev_parquet_so, eps_so, eri_so, nO_so, 'os', spin_pairs=pairs,
              vertex_builder=capture(static_reducible_vertex_so, into), **options)
    return into['spectra']


def vertex_names(driver):
    return restricted_vertices.VERTEX_NAMES if driver == 'restricted' else spin_orbital.VERTEX_NAMES


def channels(spectra, name):
    """((om, rho), None) of an eh vertex, ((om_ee, rho_ee), (om_hh, rho_hh))
    of a pp vertex, in either driver's spectra."""
    if name == 'eh':
        return spectra.eh, None
    if name == 'pp':
        return spectra.ee, spectra.hh
    if name.startswith('eh'):
        return getattr(spectra, name), None
    return getattr(spectra, 'ee_' + name[3:]), getattr(spectra, 'hh_' + name[3:])


def pole_sum(spectra, name, eh_weight, pp_weight, pp_center):
    """Eqs. (37) and (43), the eh vertex at omega = 0 and the pp vertex at
    omega = pp_center, as einsum pole sums with the pole weights given (one
    formula for both drivers):

        Phi^eh_pqrs = sum_n M_pr,n M_sq,n w(0 - Omega_n) - M_rp,n M_qs,n w(0 + Omega_n)
        Phi^pp_pqrs = sum_m M^ee_pq,m M^ee_rs,m w(c - Omega^ee_m)
                      - M^hh_pq,m M^hh_rs,m w(c - Omega^hh_m)
    """
    first, second = channels(spectra, name)
    if second is None:
        om, rho = first
        return (np.einsum('prn,sqn,n->pqrs', rho, rho, eh_weight(0.0 - om), optimize=True)
                - np.einsum('rpn,qsn,n->pqrs', rho, rho, eh_weight(0.0 + om), optimize=True))
    (om_ee, rho_ee), (om_hh, rho_hh) = first, second
    return (np.einsum('pqm,rsm,m->pqrs', rho_ee, rho_ee, pp_weight(pp_center - om_ee),
                      optimize=True)
            - np.einsum('pqm,rsm,m->pqrs', rho_hh, rho_hh, pp_weight(pp_center - om_hh),
                        optimize=True))


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
@pytest.mark.parametrize('case', sorted(BUILDERS))
def test_the_builders_against_an_einsum_pole_sum(driver, case):
    spectra = captured_spectra(driver)
    keywords = dict(BUILDERS[case])
    builder = averaged_vertex_builder(keywords.pop('window'), driver=driver, **keywords)
    eta_2b = 50.0

    def weight(x_half):
        return functools.partial(averaged_inverse, window=builder.window, axis=builder.axis,
                                 x_half=x_half, eta_2b=eta_2b)
    acted = 0.0
    for name in vertex_names(driver):
        got = builder(name, spectra, eta_2b)
        want = pole_sum(spectra, name, weight(builder.x_half_eh), weight(builder.x_half_pp),
                        builder.pp_center)
        scale = np.abs(want).max()
        assert scale > 1e-3, (driver, case, name)
        gap = np.abs(got - want).max()
        assert gap <= POLE_SUM_RTOL * scale, (driver, case, name, gap, scale)
        default = (static_reducible_vertex if driver == 'restricted'
                   else static_reducible_vertex_so)(name, spectra, eta_2b)
        acted = max(acted, np.abs(got - default).max() / scale)
        # a width that follows eta_2b is infinite at eta_2b = 0: no vertex
        if (builder.x_half_eh if name.startswith('eh') else builder.x_half_pp) is None:
            assert not builder(name, spectra, 0.0).any(), (driver, case, name)
    # the static vertex differs: the window acted (on the eh vertices at least,
    # whose poles lie nearest zero; a window with a fast tail, the SRG or the box,
    # leaves the pp poles beyond 0.7 Ha bare, as the static vertex does)
    assert acted > 1e-6, (driver, case, acted)
    with pytest.raises(ValueError, match=rf'this builder was made for driver={driver!r}'):
        builder('eh' if driver == 'restricted' else 'eh_singlet', spectra, eta_2b)
    with pytest.raises(ValueError, match='unknown reducible vertex'):
        builder('ph_singlet', spectra, eta_2b)


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_srg_matched_to_eta_2b_is_the_static_vertex_to_the_bit(driver):
    """'srg' with x_half None: the default's weights (`regularized_inverse` of
    each driver) and vertex to the bit, at eta_2b = 0, 1, 50; and a run with
    it the default run, the options record naming it."""
    spectra = captured_spectra(driver)
    builder = averaged_vertex_builder('srg', 'real', driver=driver)
    default = static_reducible_vertex if driver == 'restricted' else static_reducible_vertex_so
    inverse = (restricted_vertices.regularized_inverse if driver == 'restricted'
               else spin_orbital_two_body.regularized_inverse)
    for eta_2b in (0.0, 1.0, 50.0):
        for name in vertex_names(driver):
            first, second = channels(spectra, name)
            denominators = ((0.0 - first.om, 0.0 + first.om) if second is None
                            else (0.0 - first.om, 0.0 - second.om))
            for d in denominators:
                assert (averaged_inverse(d, 'srg', 'real', eta_2b=eta_2b).tobytes()
                        == inverse(d, eta_2b).tobytes()), (driver, eta_2b, name)
            got, want = builder(name, spectra, eta_2b), default(name, spectra, eta_2b)
            assert got.tobytes() == want.tobytes(), (driver, eta_2b, name)
    options = dict(eta_2b=50.0, max_it_2b=6, conv_2b=1e-12, qp_solver='linearized')
    (eps, eri, nO), (eps_so, eri_so, nO_so, pairs) = both_layouts('water')
    if driver == 'restricted':
        def run(**more):
            return quiet(restricted.ev_parquet, eps, eri, nO, 'os', **options, **more)
    else:
        def run(**more):
            return quiet(spin_orbital.ev_parquet_so, eps_so, eri_so, nO_so, 'os',
                         spin_pairs=pairs, **options, **more)
    got, want = run(vertex_builder=builder), run()
    assert got.n_it_2b == want.n_it_2b
    assert_same(got.eqp, want.eqp, label=(driver, 'eqp'))
    assert_same(got.z, want.z, unit='1', label=(driver, 'z'))
    for x, y in zip(got.vertices, want.vertices):
        assert_same(x, y, quantity='root', label=(driver, 'vertices'))
    assert flag_names(got) == flag_names(want)
    assert got.options['vertex_builder'] == builder.__name__


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_a_narrow_imaginary_window_is_the_static_vertex_at_a_huge_eta_2b(driver):
    spectra = captured_spectra(driver)
    default = static_reducible_vertex if driver == 'restricted' else static_reducible_vertex_so
    for window in WINDOWS['imaginary']:
        gaps = []
        for x_half in (1e-3, 1e-6, NARROW_X_HALF):
            builder = averaged_vertex_builder(window, x_half=x_half, driver=driver)
            gap = 0.0
            for name in vertex_names(driver):
                want = default(name, spectra, HUGE_ETA_2B)
                got = builder(name, spectra, HUGE_ETA_2B)
                gap = max(gap, np.abs(got - want).max() / np.abs(want).max())
            gaps.append(gap)
        assert gaps[-1] <= NARROW_RTOL, (driver, window, gaps)
        assert gaps[0] >= gaps[-1], (driver, window, gaps)


@functools.lru_cache(maxsize=None)
def water_run(scheme, window):
    """run_parquet on water/6-31G with RUNS[scheme], the static vertex
    (window None) or the imaginary `window` at the SRG's half-weight point."""
    options = dict(RUNS[scheme])
    if window is not None:
        options['vertex_builder'] = averaged_vertex_builder(window)
    return quiet(restricted.run_parquet, rhf('water'), scheme, **options)


@pytest.mark.parametrize('scheme', sorted(RUNS))
@pytest.mark.parametrize('window', ['box', 'exponential'])
def test_averaged_runs_converge_away_from_the_static_run(scheme, window):
    result, static = water_run(scheme, window), water_run(scheme, None)
    assert result.converged and static.converged, (scheme, window)
    assert np.isfinite(result.eqp).all() and not flag_names(result), (scheme, window)
    homo = int(np.argmax(result.eqp[:result.nO]))
    shift = abs(result.eqp[homo] - static.eqp[homo])
    assert shift > ACTED, (scheme, window, shift * HARTREE_TO_EV)
    assert result.options['vertex_builder'] == averaged_vertex_builder(window).__name__


@pytest.mark.parametrize('case', ['imaginary box', 'real gaussian, pp centre',
                                  'imaginary exponential, own pp width, pp centre'])
def test_the_two_drivers_agree_on_a_closed_shell(case):
    """The spin-orbital driver is the closed-shell one spin-adapted with
    averaged vertices too: the spectra, the EcRPA, eqp and Z within the bars
    of the static comparison, the alpha/beta copies equal to the bit; and
    the window moved the spectra."""
    (eps, eri, nO), (eps_so, eri_so, nO_so, pairs) = both_layouts('water')
    keywords = dict(BUILDERS[case])
    window = keywords.pop('window')
    r = quiet(restricted.ev_parquet, eps, eri, nO, 'os', **TWO_DRIVERS,
              vertex_builder=averaged_vertex_builder(window, **keywords))
    g = quiet(spin_orbital.ev_parquet_so, eps_so, eri_so, nO_so, 'os', spin_pairs=pairs,
              **TWO_DRIVERS,
              vertex_builder=averaged_vertex_builder(window, driver='spin_orbital', **keywords))
    assert r.n_it_2b == g.n_it_2b == [TWO_DRIVERS['max_it_2b']], case
    assert_same_two_body(r, g, case)
    assert np.array_equal(g.eqp[0::2], g.eqp[1::2]), case
    assert np.abs(r.eqp - g.eqp[0::2]).max() <= CROSS_DRIVER, case
    assert np.abs(r.z - g.z[0::2]).max() <= CROSS_DRIVER_Z, case
    static = quiet(restricted.ev_parquet, eps, eri, nO, 'os', **TWO_DRIVERS)
    assert np.abs(r.spectra.eh_triplet.om - static.spectra.eh_triplet.om).max() > 1e-6, case


@pytest.mark.parametrize('run', ['flex', 'os', 'os_linearized', 'ev', 'qs'])
@pytest.mark.parametrize('window', ['box', 'exponential'])
def test_the_spin_flip_at_exactly_zero_with_an_imaginary_window(run, window, monkeypatch):
    """H/STO-3G's one eh root, the spin flip, forced to each of
    SPIN_FLIP_VALUES: the run with an imaginary window as computed to the
    bit, unflagged, its eh vertex zero. The flip has a residue (its
    effective integral is about -J, 0.77 Ha): untapered, its eh vertex
    would be +-L(0) J^2 a few ulps off zero and 0 at 0.0, so the vertex
    assertion is what holds the taper here. H's eqp and Z would not move
    (two spin orbitals, no pp pair), which is why LiH's triplet (below) is
    held for the energies."""
    scheme, options = ZERO_MODE_RUNS[run]
    mf, _ = open_shell('h_sto')
    builder = averaged_vertex_builder(window, driver='spin_orbital')
    computed = quiet(spin_orbital.run_parquet_so, mf, scheme, eta_2b=1.0,
                     vertex_builder=builder, **options)
    assert abs(computed.spectra.eh.om[0]) < ZERO_MODE_BELOW, computed.spectra.eh.om
    for value in SPIN_FLIP_VALUES:
        with monkeypatch.context() as patch:
            patch.setattr(spin_orbital, 'solve_eh_bse_so', spin_flip_at(value))
            forced = quiet(spin_orbital.run_parquet_so, mf, scheme, eta_2b=1.0,
                           vertex_builder=builder, **options)
        assert forced.spectra.eh.om.tobytes() == np.array([value]).tobytes(), (run, value)
        assert not flag_names(forced), (run, value, flag_names(forced))
        assert_same_bits(forced, computed, (run, window, value), None, None)
        assert not forced.vertices.eh.any(), (run, window, value)


@pytest.mark.parametrize('run', ['flex', 'os', 'os_linearized', 'ev'])
def test_a_closed_shells_zero_triplet_with_an_imaginary_window(run, monkeypatch):
    """LiH/STO-3G's lowest triplet, whose pole has a residue, forced to
    0.0, -0.0 and 1e-16 against ULPS_OFF (the closed-shell analogue of
    `test_parquet_small_systems`, whose bars these are; LiH's qsPA meets no
    bar between forced values there either): with the box window the same
    run to rounding. That is the taper's work: untapered, the triplet a few
    ulps off would weigh -L(0) in the vertex, and eqp would move by 1 mHa
    and Z by 6e-3 (measured)."""
    scheme, options = ZERO_MODE_RUNS[run]
    builder = averaged_vertex_builder('box')
    results = []
    for value in (ULPS_OFF, 0.0, -0.0, 1e-16):
        blocks, solve = lowest_triplet_at(value)
        with monkeypatch.context() as patch:
            patch.setattr(restricted, 'eh_bse_blocks', blocks)
            patch.setattr(restricted, 'solve_eh_bse', solve)
            result = quiet(restricted.run_parquet, forced_triplet_rhf('lih'), scheme,
                           eta_2b=1.0, vertex_builder=builder, **options)
        assert result.spectra.eh_triplet.om[0] == value, (run, value)
        results.append((value, result))
    (_, reference), *forced = results
    for value, result in forced:
        assert_same_bits(result, reference, ('lih', run, value), CLOSED_ZERO_MODE_GAP,
                         CLOSED_ZERO_MODE_Z)


def test_the_options_record_names_the_window_and_its_widths():
    builder = averaged_vertex_builder('box', x_half_eh=0.25, pp_center=-0.5)
    name = ("averaged_vertex_builder('box', 'imaginary', x_half_eh=0.25, x_half_pp=None, "
            "pp_center=-0.5)")
    assert builder.__name__ == repr(builder) == name
    same = averaged_vertex_builder('box', 'imaginary', x_half_eh=np.float64(0.25),
                                   pp_center=np.float64(-0.5))
    assert same.__name__ == name
    assert averaged_vertex_builder('box', x_half=0.25, x_half_pp=0.3).__name__ == (
        "averaged_vertex_builder('box', 'imaginary', x_half_eh=0.25, x_half_pp=0.3, "
        "pp_center=0.0)")
    eps, eri, nO = both_layouts('water')[0]
    result = quiet(restricted.ev_parquet, eps, eri, nO, 'os', eta_2b=50.0, max_it_2b=1,
                   qp_solver='linearized', vertex_builder=builder)
    assert result.options['vertex_builder'] == name
    (x_eh, t_eh), (x_pp, t_pp) = builder.widths(50.0).values()
    assert (x_eh, x_pp) == (0.25, X_HALF_50) and t_eh == math.log(2.0) / 0.25
    so = averaged_vertex_builder('gaussian', 'real', driver='spin_orbital')
    assert so.__name__ == ("averaged_vertex_builder('gaussian', 'real', x_half_eh=None, "
                           "x_half_pp=None, pp_center=0.0, driver='spin_orbital')")


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_one_builder_serves_both_drivers(driver):
    """A builder made without a driver gives, for either driver's vertex
    names, the vertex of the builder made for that driver, to the bit."""
    spectra = captured_spectra(driver)
    for window, axis in (('box', 'imaginary'), ('gaussian', 'real')):
        either = averaged_vertex_builder(window, axis, x_half_pp=0.3, pp_center=-0.4)
        only = averaged_vertex_builder(window, axis, x_half_pp=0.3, pp_center=-0.4,
                                       driver=driver)
        for name in vertex_names(driver):
            assert np.array_equal(either(name, spectra, 50.0), only(name, spectra, 50.0)), name


@pytest.mark.parametrize('axis, window', ALL)
def test_extreme_widths_and_denominators(axis, window):
    """eta_2b up to the largest float (x_half from it never 0) and the
    smallest explicit x_half give finite weights, the bare pole 1/x but for
    the real sinc; denominators up to 1e300 and inf give finite weights or
    the window's limit; none of it warns of an overflow."""
    x = np.array([1e-3, 0.3, 10.0, -10.0])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for eta_2b in (1e300, 9e307, np.finfo(float).max):
            r = averaged_inverse(x, window, axis, eta_2b=eta_2b)
            assert np.all(np.isfinite(r)), (axis, window, eta_2b, r)
            if window != 'sinc':
                assert np.allclose(x * r, 1.0, rtol=0.0, atol=1e-12), (axis, window, eta_2b, r)
        r = averaged_inverse(x, window, axis, x_half=constants.PARQUET_AVERAGED_MIN_X_HALF)
        assert np.all(np.isfinite(r)), (axis, window, r)
        if window != 'sinc':
            assert np.allclose(x * r, 1.0, rtol=0.0, atol=1e-12), (axis, window, r)
        big = np.array([1e150, 1e200, 1e300, -1e300, np.inf, -np.inf])
        for eta_2b in (1.0, 50.0):
            r = averaged_inverse(big, window, axis, eta_2b=eta_2b)
            # at most twice the bare pole (the real sinc's crest), 1/1e150 at the smallest
            assert np.all(np.isfinite(r)) and np.all(np.abs(r) <= 2.5e-150), (axis, window, r)


def test_srg_is_the_default_expression_at_every_input():
    """'srg' with s = eta_2b: each driver's regularized_inverse to the bit at
    every denominator, +-0, +-inf and NaN included, at eta_2b = 0 too, where
    the default's 0 * inf at x = +-inf gives NaN."""
    x = np.array([0.0, -0.0, 5e-324, -1e-300, 2.2e-16, -0.3, 0.3, 30.0, -1e200, 1e300,
                  np.inf, -np.inf, np.nan])
    for eta_2b in (0.0, 1e-300, 1.0, 50.0, 1e8, 1e300):
        got = averaged_inverse(x, 'srg', 'real', eta_2b=eta_2b)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for default in (restricted_vertices.regularized_inverse,
                            spin_orbital_two_body.regularized_inverse):
                assert got.tobytes() == default(x, eta_2b).tobytes(), (eta_2b, default)


#: Wrong arguments of the factory: (keywords, the argument the error names).
REFUSED = [
    (dict(window='sinc'), 'window'),
    (dict(window='lorentzian'), 'window'),
    (dict(window='box', axis='real'), 'window'),
    (dict(window=None), 'window'),
    (dict(window='box', axis='matsubara'), 'axis'),
    (dict(window='box', axis=None), 'axis'),
    (dict(window='box', x_half=0.0), 'x_half'),
    (dict(window='box', x_half=-0.1), 'x_half'),
    (dict(window='box', x_half=float('nan')), 'x_half'),
    (dict(window='box', x_half=True), 'x_half'),
    (dict(window='box', x_half='0.1'), 'x_half'),
    (dict(window='box', x_half=1e-320), 'x_half'),
    (dict(window='srg', axis='real', x_half=1e-151), 'x_half'),
    (dict(window='box', x_half_eh=0.0), 'x_half_eh'),
    (dict(window='box', x_half_pp=-1.0), 'x_half_pp'),
    (dict(window='box', pp_center=float('inf')), 'pp_center'),
    (dict(window='box', pp_center=float('nan')), 'pp_center'),
    (dict(window='box', pp_center=None), 'pp_center'),
    (dict(window='box', driver='so'), 'driver'),
    (dict(window='box', driver=['restricted']), 'driver'),
]


@pytest.mark.parametrize('keywords, argument', REFUSED)
def test_wrong_arguments_are_refused(keywords, argument):
    keywords = dict(keywords)
    window = keywords.pop('window')
    with pytest.raises(ValueError, match=rf'^{argument} = '):
        averaged_vertex_builder(window, **keywords)


def test_a_width_needs_x_half_or_eta_2b():
    with pytest.raises(ValueError, match='x_half and eta_2b are both None'):
        averaged_inverse(0.3, 'box')
    for eta_2b in (-1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match=r'^eta_2b = '):
            averaged_inverse(0.3, 'box', eta_2b=eta_2b)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
