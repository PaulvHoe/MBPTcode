"""The zero-dimensional backend of the Hedin cycles: diagram counts and 0-D values of the schemes.

In zero dimensions every space-time label collapses to one point (Molinari, PRB 71, 113102 (2005)),
the factors i and -1 are absorbed so that every Feynman diagram has weight +1, and a closed fermion
loop has the weight ell (the loop weight). Then P = ell G^2 Gamma, Sigma_H[G] = ell v G, Sigma_x[G] = v G
and L_0 = G^2.

The start line is set to G^(0) = 1, and the bare interaction to the coupling of the line,
v = y = v (G^(0))^2. Hedin's equations are unchanged under G^(0) -> lambda G^(0), v -> v / lambda^2,
so every object X of a cycle is (G^(0))^dim q(y), with dim the power of G^(0) it carries (G: 1,
Sigma: -1, v and W: -2, P, L and vertex legs: 2, vertices: 0, kernels: -2). A Quantity stores dim and
the truncated series q. Its derivative along the map, with respect to G^(0) at fixed v, is then
(dim + 2y d/dy) q with one power of G^(0) less. For a self-energy this is theta = 2y d/dy - 1, which
multiplies y^k by 2k - 1, the number of lines a diagram of order k has.

The derivative at fixed form, with respect to the argument of a level functional S[G], uses a second
variable eps: S is evaluated at G + eps and the eps-linear part is read off. Nested level functionals
evaluate the lower levels at the same seeded line, and each derivative costs one order in eps.

The series are expanded about y = y0 and eps = 0. With y0 = 0 and integer coefficients they are exact
power series, and the coefficient of y^k of kappa^(n) = G^(0) (Sigma^(n) - Sigma^(0)) is the number of
the scheme's diagrams at v^k (the counts). At a finite y0 they are Taylor jets, and each derivative
along the map costs one order in y. The coefficients may be any exact or fixed-precision numbers:
ints for the counts, decimal.Decimal or fractions.Fraction at a finite coupling (the two do not mix,
so a loop weight or coupling of one type cannot be combined with the other).
"""
from fractions import Fraction

from src.SingleReference.HedinCycles.schemes import run_cycles, validate


def _reciprocal(a0):
    if isinstance(a0, int):
        return a0 if a0 in (1, -1) else Fraction(1, a0)
    return 1 / a0


def _min_order(a, b):
    """The eps order of a result: None (no eps dependence) is exact and gives way to any order."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


class Quantity:
    """(G^(0))^dim q(y0 + dy, eps), q a series truncated at dy^ny and eps^ne.

    c[i][j] is the coefficient of dy^i eps^j. ne is None for a quantity that does not depend on eps
    (it is then exact in eps and stored in one column).
    """

    __slots__ = ('dim', 'c', 'y0', 'ny', 'ne')

    def __init__(self, dim, c, y0, ny, ne=None):
        self.dim, self.c, self.y0, self.ny, self.ne = dim, c, y0, ny, ne

    @classmethod
    def constant(cls, dim, value, y0, ny):
        return cls(dim, [[value]] + [[0] for _ in range(ny)], y0, ny)

    @classmethod
    def coupling(cls, dim, y0, ny):
        """The series of y itself, carrying (G^(0))^dim."""
        c = [[y0]] + [[1 if i == 1 else 0] for i in range(1, ny + 1)]
        return cls(dim, c, y0, ny)

    # ------------------------------------------------------------------ access
    @property
    def value(self):
        """The value at y = y0, eps = 0."""
        return self.c[0][0]

    def series(self):
        """The coefficients of dy^0 .. dy^ny at eps = 0."""
        return [row[0] for row in self.c]

    def _columns(self):
        return 1 if self.ne is None else self.ne + 1

    def _padded(self, ny, ncols):
        return [(row + [0] * (ncols - len(row)))[:ncols] for row in self.c[:ny + 1]]

    def _like(self, other):
        """other as a Quantity: a plain number is a constant, allowed with dim 0 or as zero."""
        if isinstance(other, Quantity):
            if other.y0 != self.y0:
                raise ValueError('quantities expanded about different couplings')
            return other
        if self.dim != 0 and other != 0:
            raise ValueError(f'cannot add the number {other} to a quantity carrying (G^(0))^{self.dim}')
        return Quantity.constant(self.dim, other, self.y0, self.ny)

    # -------------------------------------------------------------- arithmetic
    def __add__(self, other):
        other = self._like(other)
        if other.dim != self.dim:
            raise ValueError(f'cannot add quantities carrying (G^(0))^{self.dim} and (G^(0))^{other.dim}')
        ny, ne = min(self.ny, other.ny), _min_order(self.ne, other.ne)
        ncols = 1 if ne is None else ne + 1
        a, b = self._padded(ny, ncols), other._padded(ny, ncols)
        return Quantity(self.dim, [[p + q for p, q in zip(ra, rb)] for ra, rb in zip(a, b)],
                        self.y0, ny, ne)

    __radd__ = __add__

    def __neg__(self):
        return Quantity(self.dim, [[-p for p in row] for row in self.c], self.y0, self.ny, self.ne)

    def __sub__(self, other):
        return self + (-self._like(other))

    def __rsub__(self, other):
        return self._like(other) - self

    def __mul__(self, other):
        if not isinstance(other, Quantity):
            return Quantity(self.dim, [[other * p for p in row] for row in self.c],
                            self.y0, self.ny, self.ne)
        if other.y0 != self.y0:
            raise ValueError('quantities expanded about different couplings')
        ny, ne = min(self.ny, other.ny), _min_order(self.ne, other.ne)
        ncols = 1 if ne is None else ne + 1
        a, b = self._padded(ny, ncols), other._padded(ny, ncols)
        c = [[0] * ncols for _ in range(ny + 1)]
        for i1, ra in enumerate(a):
            for j1, p in enumerate(ra):
                if not p:
                    continue
                for i2 in range(ny + 1 - i1):
                    rb, rc = b[i2], c[i1 + i2]
                    for j2 in range(ncols - j1):
                        rc[j1 + j2] += p * rb[j2]
        return Quantity(self.dim + other.dim, c, self.y0, ny, ne)

    __rmul__ = __mul__

    def reciprocal(self):
        a = self.c
        ncols = self._columns()
        r = _reciprocal(a[0][0])
        b = [[0] * ncols for _ in range(self.ny + 1)]
        for i in range(self.ny + 1):
            for j in range(ncols):
                if i == 0 and j == 0:
                    b[0][0] = r
                    continue
                s = 0
                for i1 in range(i + 1):
                    for j1 in range(j + 1):
                        if (i1 or j1) and a[i1][j1]:
                            s += a[i1][j1] * b[i - i1][j - j1]
                b[i][j] = -r * s
        return Quantity(-self.dim, b, self.y0, self.ny, self.ne)

    def __truediv__(self, other):
        if not isinstance(other, Quantity):
            return self * _reciprocal(other)
        return self * other.reciprocal()

    def __rtruediv__(self, other):
        return self.reciprocal() * other

    # ------------------------------------------------------------- derivatives
    def d_line0(self):
        """The derivative with respect to the start line G^(0) at fixed v: (dim + 2y d/dy) q."""
        if self.ne is not None:
            raise ValueError('the derivative along the map acts on cycle quantities, which do not '
                             'carry the fixed-form seed')
        a, y0 = self.dim, self.y0
        if y0 == 0:
            c = [[(a + 2 * i) * row[0]] for i, row in enumerate(self.c)]
            return Quantity(a - 1, c, y0, self.ny)
        if self.ny == 0:
            raise ValueError('the jet in the coupling is exhausted: raise its order')
        c = [[a * self.c[i][0] + 2 * i * self.c[i][0] + 2 * y0 * (i + 1) * self.c[i + 1][0]]
             for i in range(self.ny)]
        return Quantity(a - 1, c, y0, self.ny - 1)

    def seeded(self, order):
        """self + eps, eps known to order `order`: the line of a fixed-form derivative."""
        if self.ne is not None:
            raise ValueError('the line already carries the fixed-form seed')
        if order < 1:
            raise ValueError('a fixed-form derivative needs an eps order of at least 1')
        c = [row[:1] + [0] * order for row in self.c]
        c[0][1] = 1
        return Quantity(self.dim, c, self.y0, self.ny, order)

    def d_eps(self):
        """The derivative with respect to the seed eps (one order in eps less)."""
        if self.ne is None:
            return Quantity.constant(self.dim - 1, 0, self.y0, self.ny)
        if self.ne == 0:
            raise ValueError('the fixed-form jet is exhausted: raise eps_order')
        c = [[(j + 1) * row[j + 1] for j in range(self.ne)] for row in self.c]
        return Quantity(self.dim - 1, c, self.y0, self.ny, self.ne - 1)

    def at_eps0(self):
        return Quantity(self.dim, [row[:1] for row in self.c], self.y0, self.ny)


class ZeroDBackend:
    """The operations of algebra.Backend in zero dimensions, with the loop weight ell.

    coupling: y0, the coupling of the start line about which the series are expanded (0 for exact
    power series, a number for Taylor jets at that coupling). order: the truncation in y (the highest
    power of the interaction kept about 0, or the jet order at a finite coupling, which must be at
    least the number of along-the-map derivatives, cycles - 1). eps_order: the order of the
    fixed-form seed, at least the number of cycles of a cycle-based scheme and at least 1.
    """

    def __init__(self, loop_weight=1, coupling=0, order=8, eps_order=1):
        self.ell = loop_weight
        self.y0, self.order, self.eps_order = coupling, order, eps_order
        self.line0 = Quantity.constant(1, 1, coupling, order)
        self.v = Quantity.coupling(-2, coupling, order)

    def zero_self_energy(self):
        return Quantity.constant(-1, 0, self.y0, self.order)

    def sigma_h(self, G):
        return self.ell * self.v * G

    def sigma_x(self, G):
        return self.v * G

    def dyson(self, d_sigma):
        return self.line0 / (1 - self.line0 * d_sigma)

    def vertex_legs(self, G, vertex):
        return G * G * vertex

    def polarizability(self, G, vertex):
        return self.ell * G * G * vertex

    def screen(self, P):
        return self.v / (1 - self.v * P)

    def sigma_gw(self, G, W, vertex):
        return G * W * vertex

    def solve_vertex(self, G, K):
        return 1 / (1 - K * G * G)

    def solve_l(self, G, K):
        L0 = G * G
        return L0 / (1 - L0 * K)

    def apply_kernel(self, K, L):
        return K * L

    def sigma_il(self, G, X):
        return self.v * G * X

    def sg_vertex(self, lam, lam_h):
        return 1 + lam / (1 + lam_h)

    def along_map(self, X, legs):
        return X.d_line0() * legs

    def kernel(self, S, G):
        if G.ne is None:           # the outermost derivative: seed the line, read off eps = 0
            return S(G.seeded(self.eps_order)).d_eps().at_eps0()
        return S(G).d_eps()        # nested: G is the seeded line G' + eps of an outer derivative


def _backend_for(scheme, cycles, loop_weight, coupling, order):
    eps_order = cycles if scheme in ('cb_gw', 'cb_il') else 1
    return ZeroDBackend(loop_weight, coupling, order, eps_order)


def _kappas(alg, hist):
    """kappa^(n) = G^(0) (Sigma^(n) - Sigma^(0)), n = 1 .. cycles."""
    return [alg.line0 * (s - hist.sigmas[0]) for s in hist.sigmas[1:]]


def kappa_series(scheme, start, cycles, order=8, loop_weight=1, lag=True):
    """kappa^(1 .. cycles) as exact integer coefficients of y^0 .. y^order, y = v (G^(0))^2.
    The coefficient of y^k is the number of the scheme's diagrams at v^k (at loop weight 1), and at
    an integer loop weight ell a polynomial in ell evaluated there. `order` is the caller's, and the
    order of the fixed-form seed is chosen from the cycles."""
    validate(scheme, start, cycles, lag)
    if isinstance(loop_weight, bool) or not isinstance(loop_weight, int):
        raise ValueError(f'counts need an integer loop weight, got {loop_weight!r}')
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError(f'order must be a positive integer, got {order!r}')
    alg = _backend_for(scheme, cycles, loop_weight, 0, order)
    out = []
    for k in _kappas(alg, run_cycles(alg, scheme, start, cycles, lag)):
        coefficients = []
        for p in k.series():
            p = Fraction(p)
            if p.denominator != 1:
                raise ArithmeticError(f'non-integer coefficient {p} in a count')
            coefficients.append(int(p))
        out.append(coefficients)
    return out


def counts(scheme, start, cycles, order=8, loop_weight=1):
    """The diagram counts of cycles 1 .. cycles at v^1 .. v^order: the coefficients of kappa^(n)
    relative to the start line, so a start line that holds a diagram (the Fock diagram of f) does not
    count it again."""
    return [row[1:] for row in kappa_series(scheme, start, cycles, order, loop_weight)]


def kappa_at(scheme, start, coupling, cycles, loop_weight=1, lag=True):
    """kappa^(1 .. cycles) at the coupling y0 = coupling of the start line (exact for a Fraction,
    to the working precision for a decimal.Decimal), with G^(n) = G^(0) / (1 - kappa^(n)). Both the
    order in y and the order of the fixed-form seed are chosen from the cycles."""
    validate(scheme, start, cycles, lag)
    order = 0 if scheme in ('cb_gw', 'cb_il') else cycles
    alg = _backend_for(scheme, cycles, loop_weight, coupling, order)
    return [k.value for k in _kappas(alg, run_cycles(alg, scheme, start, cycles, lag))]
