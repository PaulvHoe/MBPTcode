"""Independent oracle for the 0-D tests of the Hedin cycles: the closed-form maps of the four vertex
schemes in zero dimensions, in exact arithmetic.

Source: the backup slide "the maps in zero dimensions" of the group-meeting talk and
src/zero_d_convergence.py of the theory repository (AutomaticEquationGenerator, working tree at
366bf6d), which evaluate the same maps with mpmath. Here every coefficient is an int or a
fractions.Fraction, so the counts are exact and a coupling given as a fraction gives exact rationals.
None of this code is shared with src/SingleReference/HedinCycles: the test holds the two against each
other (as power series and exactly at a coupling given as a fraction) and both against
tests/data/hedin_cycles_reference_values.json (the counts, and the 0-D values at finite coupling).

Notation, that of src/zero_d_convergence.py: the slide's coupling y is written z here, and the slide's
m = max(n - 2, 0) is written k. ell is the weight of a closed loop, z the coupling of the start line
(u = v h^2 from (H), w = v f^2 from (HF)), m = ell from (H) and 1 + ell from (HF),
R = 1/(1 - m z) the start response (RPA or TDHF), c0 = 0 from (H) and z from (HF),
kappa^(n) = G^(0) (Sigma^(n) - Sigma^(0)), gamma_n = 1/(1 - kappa^(n)) with gamma_0 = gamma_-1 = 1,
theta = 2z d/dz - 1, and the level functionals S_n(s) = s / (1 - ell s - theta_s S_(n-1)(s)) with
S_0 = 0 from (H) and S_0 = s from (HF).

  SG-I/L:      kappa^(n) = ell z (gamma_(n-1) - 1) + R z gamma_(n-1) (1 + theta kappa^(n-1)) - c0
  SG-Gamma/W:  kappa_xc = kappa^(n-1) - ell z (gamma_(n-2) - 1) + c0
               beta_n = 1 + ell z R gamma_k^2 (1 + theta kappa^(k)),  k = max(n - 2, 0)
               Gamma^(n) = 1 + R theta kappa_xc / beta_n
               kappa^(n) = ell z (gamma_(n-1) - 1) + z gamma_(n-1) Gamma^(n) / (1 - ell z gamma_(n-1)^2 Gamma^(n)) - c0
  cycle-based: kappa^(n) = ell z (gamma_(n-1) - 1) + S_n(z gamma_(n-1)^2) / gamma_(n-1) - c0

The two double-counting traps of the group note (section "two ways to count screening twice"), as
the note's group_note_counts.py defines them (sg_literal_update and pitfall_reducible):
'eq7_literal', SG's Eq. (7) used literally after the Hartree update, Gamma^(n) = 1 + theta kappa_xc,
the start's vertex legs with no Lambda_H factor, and 'along_l1', the vertex taken along the start's
L^(1) instead of its vertex legs with W also built from P, Gamma^(n) = 1 + R theta kappa_xc. The
reference file names the second one 'vertex taken with respect to the external potential instead of
the total one, with W also built from P'.
"""
from fractions import Fraction


def _reciprocal(a0):
    if isinstance(a0, int):
        return a0 if a0 in (1, -1) else Fraction(1, a0)
    return 1 / a0


class Jet:
    """f(c + e) = sum_k a[k] e^k + O(e^(N+1)), N = len(a) - 1 (zero_d.Jet of the theory repository).

    About c = 0, theta = 2u d/du - 1 is diagonal and loses no order. About c != 0 it loses one.
    """

    __slots__ = ('a', 'c')

    def __init__(self, a, c=0):
        self.a, self.c = list(a), c

    @classmethod
    def variable(cls, c, order):
        return cls(([c, 1] + [0] * order)[:order + 1], c)

    @property
    def value(self):
        return self.a[0]

    def _lift(self, other):
        return other if isinstance(other, Jet) else Jet([other] + [0] * (len(self.a) - 1), self.c)

    def __add__(self, other):
        other = self._lift(other)
        return Jet([p + q for p, q in zip(self.a, other.a)], self.c)

    __radd__ = __add__

    def __neg__(self):
        return Jet([-p for p in self.a], self.c)

    def __sub__(self, other):
        return self + (-self._lift(other))

    def __rsub__(self, other):
        return self._lift(other) - self

    def __mul__(self, other):
        if not isinstance(other, Jet):
            return Jet([other * p for p in self.a], self.c)
        a, b = self.a, other.a
        n = min(len(a), len(b))
        return Jet([sum(a[i] * b[k - i] for i in range(k + 1)) for k in range(n)], self.c)

    __rmul__ = __mul__

    def __pow__(self, k):
        out = self._lift(1)
        for _ in range(k):
            out = out * self
        return out

    def inv(self):
        a = self.a
        r = _reciprocal(a[0])
        b = [r]
        for k in range(1, len(a)):
            b.append(-r * sum(a[j] * b[k - j] for j in range(1, k + 1)))
        return Jet(b, self.c)

    def __truediv__(self, other):
        return self * self._lift(other).inv()

    def __rtruediv__(self, other):
        return self._lift(other) * self.inv()

    def theta(self):
        a = self.a
        if self.c == 0:
            return Jet([(2 * k - 1) * p for k, p in enumerate(a)], self.c)
        d = [(k + 1) * a[k + 1] for k in range(len(a) - 1)]
        return Jet([2 * self.c * d[k] + (2 * d[k - 1] if k else 0) - a[k] for k in range(len(d))], self.c)


def integer_coefficients(f):
    out = []
    for p in f.a:
        p = Fraction(p)
        if p.denominator != 1:
            raise ValueError(f'non-integer coefficient {p}')
        out.append(int(p))
    return out


def compose(coefficients, x):
    """f(x(u)) for a series x(u) = O(u)."""
    out = x._lift(coefficients[-1])
    for c in reversed(coefficients[:-1]):
        out = out * x + c
    return out


class Start:
    def __init__(self, name, ell):
        self.name, self.ell = name, ell
        self.m = ell if name == 'H' else 1 + ell

    def R(self, z):
        return 1 / (1 - self.m * z)

    def c0(self, z):
        return 0 * z if self.name == 'H' else z

    def S0(self, s):
        return 0 * s if self.name == 'H' else s


def sg_il(z, cycles, st):
    R, c0, ell = st.R(z), st.c0(z), st.ell
    kap, out = 0 * z, []
    for _ in range(cycles):
        gam = 1 / (1 - kap)
        kap = ell * z * (gam - 1) + R * z * gam * (1 + kap.theta()) - c0
        out.append(kap)
    return out


def sg_gw(z, cycles, st, vertex_rule='scheme'):
    """SG-Gamma/W, or with vertex_rule 'eq7_literal' or 'along_l1' one of the two traps."""
    R, c0, ell = st.R(z), st.c0(z), st.ell
    kaps, out = [0 * z], []
    gam = {-1: 1 + 0 * z, 0: 1 + 0 * z}
    for n in range(1, cycles + 1):
        gam[n - 1] = 1 / (1 - kaps[n - 1])
        k_xc = kaps[n - 1] - ell * z * (gam[n - 2] - 1) + c0
        if vertex_rule == 'scheme':
            k = max(n - 2, 0)
            beta = 1 + ell * z * R * gam[k] ** 2 * (1 + kaps[k].theta())
            vertex = 1 + R * k_xc.theta() / beta
        elif vertex_rule == 'eq7_literal':
            vertex = 1 + k_xc.theta()
        elif vertex_rule == 'along_l1':
            vertex = 1 + R * k_xc.theta()
        else:
            raise ValueError(vertex_rule)
        g = gam[n - 1]
        kap = ell * z * (g - 1) + z * g * vertex / (1 - ell * z * g ** 2 * vertex) - c0
        kaps.append(kap)
        out.append(kap)
    return out


def s_levels(s, n, st):
    """S_0 .. S_n as jets in s. About s != 0 every theta_s costs one order."""
    S = st.S0(s)
    levels = [S]
    for _ in range(n):
        S = s / (1 - st.ell * s - S.theta())
        levels.append(S)
    return levels


def cycle_based_series(cycles, order, st):
    u = Jet.variable(0, order)
    levels = s_levels(Jet.variable(0, order), cycles, st)
    kap, out = 0 * u, []
    c0 = st.c0(u)
    for n in range(1, cycles + 1):
        gam = 1 / (1 - kap)
        kap = st.ell * u * (gam - 1) + compose(levels[n].a, u * gam ** 2) / gam - c0
        out.append(kap)
    return out


def cycle_based_point(z0, cycles, st):
    kap, out = 0 * z0, []
    c0 = st.c0(z0)
    for n in range(1, cycles + 1):
        gam = 1 / (1 - kap)
        levels = s_levels(Jet.variable(z0 * gam ** 2, n), n, st)
        kap = st.ell * z0 * (gam - 1) + levels[n].value / gam - c0
        out.append(kap)
    return out


SCHEMES = ('SG-I/L', 'SG-Gamma/W', 'cycle-based')


def series(name, cycles, order, start, ell=1):
    """Integer coefficients of kappa^(1..cycles), z^0 .. z^order."""
    st = Start(start, ell)
    if name == 'cycle-based':
        return [integer_coefficients(k) for k in cycle_based_series(cycles, order, st)]
    fn = sg_il if name == 'SG-I/L' else sg_gw
    return [integer_coefficients(k) for k in fn(Jet.variable(0, order), cycles, st)]


def trap_series(rule, cycles, order, start, ell=1):
    return [integer_coefficients(k) for k in sg_gw(Jet.variable(0, order), cycles, Start(start, ell), rule)]


def point_values(name, z0, cycles, start, ell=1):
    """kappa^(1..cycles) at the coupling z0 (exact for a Fraction)."""
    st = Start(start, ell)
    if name == 'cycle-based':
        return cycle_based_point(z0, cycles, st)
    fn = sg_il if name == 'SG-I/L' else sg_gw
    return [k.value for k in fn(Jet.variable(z0, cycles), cycles, st)]


def _double_factorial_odd(k):
    out = 1
    for j in range(1, 2 * k, 2):
        out *= j
    return out


def exact_bare_series(order):
    """G/g of the 0-D complex boson (loop weight 1) in x = v g^2: 1, 2, 10, 74, 706, ...
    From Wick's theorem, Z = sum (2k - 1)!! x^k and G/g = (1 - 1/Z)/x."""
    Z = Jet([_double_factorial_odd(k) for k in range(order + 2)])
    return integer_coefficients(1 - Z.inv())[1:]


def exact_series(start, order):
    """The exact kappa of the start's line at loop weight 1: 1 - line/G with
    h = (1 + u) g, x = u/(1 + u)^2 from (H) and f = (1 + 2w) g, x = w/(1 + 2w)^2 from (HF)."""
    z = Jet.variable(0, order)
    m = 1 if start == 'H' else 2
    g_hat = compose(exact_bare_series(order), z / (1 + m * z) ** 2)
    return integer_coefficients(1 - (1 + m * z) / g_hat)
