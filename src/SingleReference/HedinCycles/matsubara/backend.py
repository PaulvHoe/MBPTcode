"""The finite-basis Matsubara backend of the Hedin cycles.

Everything lives in the orbitals of the start line (its levels e on the diagonal), spin orbitals or, for a
closed shell, spatial orbitals with a factor ctx.spin = 2 on every closed fermion loop (systems.py), on the
Matsubara axis at the inverse temperature beta (grids.py). The operations of algebra.py take their
imaginary-time forms here, each with its own signs, and the cycle loop holds none:

    G^-1(i nu) = i nu + mu - h - Sigma(i nu)
    P(1,2) = G(1,3) G(4,1) Gamma(3,4;2)               P is negative at Omega = 0
    W = v + v P W
    Sigma_xc(1,2) = -G(1,3) W(4,1) Gamma(3,2;4)       or W(1,4), the option w_order (algebra.py)
    Gamma = 1 + K G G Gamma         L = G G + G G K L         Sigma = Sigma_HF - v G K L (I/L)

with K = delta Sigma / delta G. The Coulomb interaction is factorized, v = sum_P B^P B^P, so a boson is
a matrix over the factors: P_PQ(Omega) = (1/beta) sum_nu Tr[B^P G(nu + Omega) Gamma^Q(nu; Omega) G(nu)],
M = (1 - P)^-1 and W = sum_PQ B^P M_PQ B^Q. A three-point object X^Q(nu; Omega) (a vertex, the
response of a self-energy, a kernel applied to L) has its pair end at nu + Omega (out) and nu (in), and
its density end contracted with B^Q: it is the response to the source B^Q at the frequency Omega. It is
stored as the bare vertex B^Q (vertices only), a part independent of nu on every transfer of the boson
grid, and a part on the vertex box. The self-energy of a three-point object is

    Sigma(nu) = -(1/beta) sum_Omega sum_PQ M_QP(Omega) [B^P G(nu + Omega) Gamma^Q(nu; Omega)]

for W(4,1), with M_PQ(-Omega) in place of M_QP(Omega) for W(1,4), and with M = 1 for the bare
interaction of the I/L closure. Sums over all fermionic frequencies subtract the
model line (the static part of a line's inverse), whose sums are exact: densities, and the pair sums
(1/beta) sum_nu G(nu + Omega) X G(nu) of static X.

A static kernel (Hartree and exchange, the start's) makes the BSE a pair-space solve at every bosonic
frequency. A kernel at fixed form depends on frequency: it acts on a direction as the tangent of its level
functional at a leaf copy of the line (Kernel.apply), and its BSE is solved by GMRES.
"""
import itertools

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.special import expit, psi

from src.Base.numerical_flags import flag
from src.Base.numerics.constants import HEDIN_BSE_RESTART, HEDIN_CONFLUENT, HEDIN_DIAGNOSTICS_MAX_N
from src.Base.numerics.numerics import Numerics
from src.SingleReference.HedinCycles.matsubara.grids import Grids
from src.SingleReference.HedinCycles.matsubara.starts import start_line


def _trigamma(z):
    """psi_1(z) = sum_k 1/(z + k)^2 for complex z off the poles: the recurrence up to Re z >= 20, then the
    asymptotic series 1/z + 1/(2 z^2) + sum_k B_2k / z^(2k+1), whose terms beyond z^-11 are below 1e-16."""
    z = np.asarray(z, dtype=complex)
    out = np.zeros_like(z)
    shift = np.maximum(0, np.ceil(20 - z.real)).astype(int)
    for k in range(int(shift.max()) if shift.size else 0):
        mask = shift > k
        out[mask] += 1.0 / (z[mask] + k) ** 2
    w = z + shift
    inv = 1.0 / w
    inv2 = inv * inv
    series = inv + 0.5 * inv2 + inv * inv2 * (1 / 6 + inv2 * (-1 / 30 + inv2 * (1 / 42 + inv2 * (-1 / 30 + inv2 * 5 / 66))))
    return out + series


class HedinBSEWarning(UserWarning):
    """The BSE of a kernel at fixed form stopped above its tolerance."""


#: the readings of Sigma_xc = -G W Gamma: '4,1' meets W at the vertex's density end first, '1,4' at the
#: external point first (the printed form)
W_ORDERS = ('4,1', '1,4')

_ids = itertools.count()


class Context:
    """What every object of one run shares: the grids, mu, the factors in the start's orbitals, and spin, the
    factor of a closed fermion loop (1 in spin orbitals, 2 for a closed shell in spatial orbitals)."""

    def __init__(self, grids, mu, B, spin=1):
        self.grids, self.mu, self.B, self.spin = grids, mu, B, spin
        self.N, self.naux = B.shape[1], B.shape[0]


# ---------------------------------------------------------------------------------- one-particle
class Line:
    """G(i nu) = [i nu + mu - h - dyn(i nu)]^-1, dyn on the fermion grid (None for a model line)."""

    def __init__(self, ctx, h, dyn=None, recipe=('line',)):
        self.ctx, self.h, self.dyn, self.recipe = ctx, np.asarray(h), dyn, recipe
        self.uid = next(_ids)
        hh = 0.5 * (self.h + self.h.conj().T)
        self.w, self.V = np.linalg.eigh(hh)
        self.G = None
        self._pair_maps = {}
        if dyn is not None:
            g = ctx.grids
            iv = 1j * g.nu(g.fermion_n) + ctx.mu
            A = iv[:, None, None] * np.eye(ctx.N) - self.h[None] - dyn
            self.G = np.linalg.inv(A)

    @property
    def x(self):
        """Model levels relative to mu."""
        return self.w - self.ctx.mu

    def model(self, n):
        iv = 1j * self.ctx.grids.nu(np.asarray(n))
        d = 1.0 / (iv[:, None] - self.x[None, :])
        return np.einsum('ai,ni,bi->nab', self.V, d, self.V.conj())

    def at(self, n):
        n = np.asarray(n)
        out = self.model(n)
        if self.G is not None:
            nf = self.ctx.grids.n_fermion
            inside = (n >= -nf) & (n < nf)
            out[inside] = self.G[n[inside] + nf]
        return out

    def fermi(self):
        return expit(-self.ctx.grids.beta * self.x)

    def density(self):
        """rho_rs = <c_s^dag c_r>: the model's exactly, the rest summed on the grid."""
        rho = (self.V * self.fermi()) @ self.V.conj().T
        if self.G is not None:
            g = self.ctx.grids
            rho = rho + (self.G - self.model(g.fermion_n)).sum(axis=0) / g.beta
        return 0.5 * (rho + rho.conj().T)

    def model_pair_weights(self, m):
        """F_ab(Omega_m) = (1/beta) sum_nu 1/((i nu + i Omega - x_a)(i nu - x_b)) for the model levels."""
        g = self.ctx.grids
        x, f = self.x, self.fermi()
        iw = 1j * g.omega(m)
        den = iw + x[None, :] - x[:, None]
        num = f[None, :] - f[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            F = num / den
        tiny = np.abs(den) < 1e-12
        F[tiny] = np.broadcast_to(-g.beta * f * (1 - f), F.shape)[tiny]
        return F

    def pair_sum(self, X, m):
        """(1/beta) sum over all nu of G(nu + Omega_m) X G(nu) for static X (..., N, N)."""
        F = self.model_pair_weights(m)
        Xt = np.einsum('ia,...ij,jb->...ab', self.V.conj(), X, self.V)
        out = np.einsum('ai,...ij,bj->...ab', self.V, Xt * F, self.V.conj())
        if self.G is not None:
            g = self.ctx.grids
            n = np.arange(-g.n_fermion - abs(m), g.n_fermion + abs(m))
            Gp, Gn = self.at(n + m), self.at(n)
            Mp, Mn = self.model(n + m), self.model(n)
            out = out + (np.einsum('nac,...cd,ndb->...ab', Gp, X, Gn)
                         - np.einsum('nac,...cd,ndb->...ab', Mp, X, Mn)) / g.beta
        return out

    def pair_map(self, m):
        """The pair sum as a matrix: S_ab = sum_cd chi[a, b, c, d] X_cd (kept per transfer)."""
        m = int(m)
        if m not in self._pair_maps:
            N = self.ctx.N
            E = np.eye(N * N).reshape(N * N, N, N)
            S = self.pair_sum(E, m)                       # (N*N cd, N, N ab)
            self._pair_maps[m] = S.reshape(N * N, N * N).T.reshape(N, N, N, N)
        return self._pair_maps[m]

    def box_sum(self, Y, m):
        """(1/beta) sum over the vertex box of G(nu + Omega_m) Y(nu) G(nu), Y (..., 2 n_vertex, N, N)."""
        g = self.ctx.grids
        n = g.vertex_n
        return np.einsum('nac,...ncd,ndb->...ab', self.at(n + m), Y, self.at(n)) / g.beta


class LinearizedLine:
    """The density of the linearized line G^(0) + G^(0) dSigma G^(0): the start's density plus its first-order
    response to dSigma = Sigma^(n) - Sigma^(0), summed over every fermionic frequency (the static part of dSigma
    exactly on the model line, the rest on the grid). It has no Green's function, only density(), and serves as
    the source of a Hartree term (run_cycles' hartree='linearized'). The Dyson line of a cycle keeps the
    start's chemical potential and need not hold the electron count, and this density does: on H2/STO-3G from
    'H' its trace moves by 1e-11 where the cycle-1 line's moves by +0.019. The tangent engine has no rule for it
    yet (a later cycle that differentiates it raises NotImplementedError)."""

    def __init__(self, backend, d_sigma):
        self.ctx, self.recipe = backend.ctx, ('linearized', d_sigma)
        self.uid = next(_ids)
        G0, g = backend.line0, backend.ctx.grids
        rho = G0.density() + G0.pair_sum(d_sigma.static, 0)
        if d_sigma.dyn is not None:
            G = G0.at(g.fermion_n)
            rho = rho + np.einsum('nab,nbc,ncd->ad', G, d_sigma.dyn, G) / g.beta
        self.rho = 0.5 * (rho + rho.conj().T)

    def density(self):
        return self.rho


class SelfEnergy:
    """static (N, N) plus dyn (2 n_fermion, N, N) or None."""

    def __init__(self, ctx, static, dyn=None, recipe=('sigma',)):
        self.ctx, self.static, self.dyn, self.recipe = ctx, np.asarray(static, dtype=complex), dyn, recipe
        self.uid = next(_ids)

    def _combine(self, other, sign, op):
        dyn = self.dyn
        if other.dyn is not None:
            dyn = sign * other.dyn if dyn is None else dyn + sign * other.dyn
        return SelfEnergy(self.ctx, self.static + sign * other.static, dyn, (op, self, other))

    def __add__(self, other):
        return self._combine(other, 1, 'add')

    def __sub__(self, other):
        return self._combine(other, -1, 'sub')

    def __neg__(self):
        return SelfEnergy(self.ctx, -self.static, None if self.dyn is None else -self.dyn, ('neg', self))

    def at_grid(self):
        """The whole self-energy on the fermion grid."""
        n = 2 * self.ctx.grids.n_fermion
        out = np.broadcast_to(self.static, (n,) + self.static.shape).copy()
        return out if self.dyn is None else out + self.dyn


# ------------------------------------------------------------------------------------- bosons
class Boson:
    """A matrix over the factors on the boson grid: data (2 m_boson + 1, N_aux, N_aux). kind 'P' is
    a polarizability, kind 'M' the screened interaction W = sum_PQ B^P M_PQ B^Q."""

    def __init__(self, ctx, kind, data, recipe):
        self.ctx, self.kind, self.data, self.recipe = ctx, kind, data, recipe
        self.uid = next(_ids)

    def at(self, m):
        return self.data[self.ctx.grids.boson_index(m)]


# ------------------------------------------------------------------------------ three-point
class ThreePoint:
    """X^Q(nu; Omega_m): bare B^Q (vertices only), static (2 m_boson + 1, N_aux, N, N) independent of
    nu, and dyn (2 m_vertex + 1, N_aux, 2 n_vertex, N, N) on the vertex box, each None when absent."""

    def __init__(self, ctx, bare, static=None, dyn=None, recipe=('three-point',)):
        self.ctx, self.bare, self.static, self.dyn, self.recipe = ctx, bare, static, dyn, recipe
        self.uid = next(_ids)

    def static_at(self, m):
        """bare + static part at the transfer m, (N_aux, N, N)."""
        ctx = self.ctx
        out = np.zeros((ctx.naux, ctx.N, ctx.N), dtype=complex)
        if self.bare:
            out = out + ctx.B
        if self.static is not None:
            out = out + self.static[ctx.grids.boson_index(m)]
        return out

    def correction_at(self, m):
        """static part without the bare vertex, (N_aux, N, N)."""
        ctx = self.ctx
        if self.static is None:
            return np.zeros((ctx.naux, ctx.N, ctx.N), dtype=complex)
        return self.static[ctx.grids.boson_index(m)]

    def dyn_at(self, m):
        g = self.ctx.grids
        if self.dyn is None or abs(m) > g.m_vertex:
            return None
        return self.dyn[m + g.m_vertex]


class Legs:
    """G Gamma G: the vertex Gamma with the line G on each leg (also L = G Gamma_red G)."""

    def __init__(self, line, vertex, recipe=('legs',)):
        self.line, self.vertex, self.recipe = line, vertex, recipe
        self.uid = next(_ids)

    def pair_density(self, m):
        """(1/beta) sum_nu G(nu + Omega_m) Gamma^Q(nu; Omega_m) G(nu), (N_aux, N, N)."""
        out = self.line.pair_sum(self.vertex.static_at(m), m)
        d = self.vertex.dyn_at(m)
        if d is not None:
            out = out + self.line.box_sum(d, m)
        return out


# --------------------------------------------------------------------------------- kernels
def static_coefficients(obj, leaf):
    """(a_H, a_x) with obj = a_H Sigma_H[leaf] + a_x Sigma_x[leaf] + (terms without the leaf), or None
    when obj depends on the leaf in any other way."""
    op = obj.recipe[0]
    if op in ('sigma_h', 'sigma_x'):
        if obj.recipe[1] is leaf:
            return (1.0, 0.0) if op == 'sigma_h' else (0.0, 1.0)
        return None if depends_on(obj.recipe[1], leaf) else (0.0, 0.0)
    if op == 'zero':
        return (0.0, 0.0)
    if op in ('add', 'sub'):
        a, b = static_coefficients(obj.recipe[1], leaf), static_coefficients(obj.recipe[2], leaf)
        if a is None or b is None:
            return None
        s = 1 if op == 'add' else -1
        return (a[0] + s * b[0], a[1] + s * b[1])
    if op == 'neg':
        a = static_coefficients(obj.recipe[1], leaf)
        return None if a is None else (-a[0], -a[1])
    return None if depends_on(obj, leaf) else (0.0, 0.0)


def depends_on(obj, leaf, _seen=None):
    if obj is leaf:
        return True
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return False
    _seen.add(id(obj))
    return any(depends_on(a, leaf, _seen) for a in getattr(obj, 'recipe', ())[1:]
               if hasattr(a, 'recipe'))


class Kernel:
    """K = delta S[G] / delta G at G, from the level functional S evaluated at a leaf copy of G."""

    def __init__(self, backend, S, G):
        self.backend, self.line, self.recipe = backend, G, ('kernel', G)
        self.leaf = Line(G.ctx, G.h, G.dyn, recipe=('leaf',))
        self.out = S(self.leaf)
        self.coefficients = static_coefficients(self.out, self.leaf)

    def apply(self, Y):
        """K G Y G for an amputated direction Y (a ThreePoint over the directions R, its static part with
        any bare vertex included): the tangent of S at the leaf, a ThreePoint without a bare vertex,
        static on the boson grid and frequency dependent on the vertex box, or None where S does not
        depend on the line."""
        from src.SingleReference.HedinCycles.matsubara.tangents import Seed, Tangents
        return Tangents(self.backend, Seed(self.leaf, Y))(self.out)

    @property
    def is_static(self):
        return self.coefficients is not None

    def static_apply(self, rho, triplet=False):
        """K applied to a pair field whose sum over nu is rho (..., N, N): a_H Sigma_H[rho] + a_x Sigma_x[rho].
        triplet=True drops the Hartree term (the triplet channel of a closed shell, where the loop over both
        spins cancels)."""
        a_h, a_x = self.coefficients
        if triplet:
            a_h = 0.0
        B = self.line.ctx.B
        out = np.zeros_like(rho, dtype=complex)
        if a_h:
            s = self.line.ctx.spin                       # the Hartree term closes a loop over both spins
            out = out + a_h * s * np.einsum('Ppq,...P->...pq', B, np.einsum('Psr,...rs->...P', B, rho))
        if a_x:
            out = out - a_x * np.einsum('Ppr,...rs,Psq->...pq', B, rho, B)
        return out

    def static_matrix(self, triplet=False):
        """K as an (N*N, N*N) matrix on vec(rho) (row-major), the triplet channel with triplet=True."""
        N = self.line.ctx.N
        E = np.eye(N * N).reshape(N * N, N, N)
        return self.static_apply(E, triplet).reshape(N * N, N * N).T


# --------------------------------------------------------------------------------- backend
class MatsubaraBackend:
    """The operations of algebra.Backend on the Matsubara axis for one system, start and grids.

    w_order chooses how sigma_gw reads Sigma_xc(1,2) = -G(1,3) W Gamma(3,2;4) once W is not symmetric,
    which a truncated cycle allows (its P^(n) carries the vertex at one end). '4,1', the default, takes
    W(4,1), the order Hedin's derivation gives before the symmetry of W is used. It keeps the Gamma/W
    route an exact rewriting of the I/L route when the vertex is a response of the line, so SG-Gamma/W
    without its lag equals SG-I/L and the two cycle-based schemes coincide. '1,4' takes the printed form
    W(1,4). The two hold mirror-image diagrams with the same counts and agree wherever P is symmetric
    (zero dimensions, cycle 1, the particle-hole symmetric half-filled Hubbard dimer). A molecule has no
    such symmetry: H2/STO-3G, though half filled, has an asymmetric P^(2).

    numerics: a Numerics, whose hedin section gives the temperature, the grids, the BSE of a kernel at fixed
    form and the start line's SCF (Numerics() if None). A Grids is read as a hedin section with those grids
    and every other setting at its default, the shorthand of the tests."""

    def __init__(self, system, start, numerics=None, w_order='4,1'):
        if w_order not in W_ORDERS:
            raise ValueError(f"w_order must be one of {W_ORDERS}, got {w_order!r}")
        if isinstance(numerics, Grids):
            numerics = Numerics(hedin=numerics.section())
        self.numerics = Numerics() if numerics is None else numerics
        self.w_order = w_order
        self.system = system
        self.start = start_line(system, start, self.numerics.hedin)
        self.start_name = start
        self.grids = grids = Grids.from_section(self.numerics.hedin, self.start.gap)
        C = self.start.C
        B = np.einsum('ip,Pij,jq->Ppq', C, system.B, C)
        self.ctx = Context(grids, self.start.mu, B, system.spin)
        self.e = self.start.e
        self.line0 = Line(self.ctx, np.diag(self.e), recipe=('line0',))

    # ------------------------------------------------------------------ one-particle
    def zero_self_energy(self):
        N = self.ctx.N
        return SelfEnergy(self.ctx, np.zeros((N, N)), None, ('zero',))

    def sigma_h(self, G):
        rho = G.density()
        B = self.ctx.B
        static = self.ctx.spin * np.einsum('Ppq,Psr,rs->pq', B, B, rho)
        return SelfEnergy(self.ctx, static, None, ('sigma_h', G))

    def sigma_x(self, G):
        rho = G.density()
        B = self.ctx.B
        static = -np.einsum('Ppr,rs,Psq->pq', B, rho, B)
        return SelfEnergy(self.ctx, static, None, ('sigma_x', G))

    def dyson(self, d_sigma):
        h = np.diag(self.e).astype(complex) + d_sigma.static
        return Line(self.ctx, h, d_sigma.dyn, ('dyson', d_sigma))

    def linearized_line(self, d_sigma):
        """The start's density plus its first-order response to d_sigma (LinearizedLine), the source of the
        Hartree term with run_cycles' hartree='linearized'."""
        return LinearizedLine(self, d_sigma)

    # ------------------------------------------------------------------------ kernels
    def kernel(self, S, G):
        return Kernel(self, S, G)

    def solve_vertex(self, G, K):
        """Gamma = 1 + K G G Gamma. For a static K: Gamma^Q(Omega) = B^Q + c^Q(Omega) with
        (1 - K chi(Omega)) c = K chi(Omega) B, solved in pair space at every boson frequency. For a kernel
        that depends on frequency, _solve_vertex_dynamic."""
        if not K.is_static:
            return self._solve_vertex_dynamic(G, K)
        ctx = self.ctx
        if K.coefficients == (0.0, 0.0):
            return ThreePoint(ctx, True, None, None, ('solve_vertex', G, K))
        N, naux = ctx.N, ctx.naux
        Kmat = K.static_matrix()
        Bvec = ctx.B.reshape(naux, N * N).T
        static = np.empty((len(ctx.grids.boson_m), naux, N, N), dtype=complex)
        for i, m in enumerate(ctx.grids.boson_m):
            chi = G.pair_map(m).reshape(N * N, N * N)
            KX = Kmat @ chi
            c = np.linalg.solve(np.eye(N * N) - KX, KX @ Bvec)
            static[i] = c.T.reshape(naux, N, N)
        return ThreePoint(ctx, True, static, None, ('solve_vertex', G, K))

    def _solve_vertex_dynamic(self, G, K):
        """Gamma^Q = B^Q + x^Q with x = K G (B + x) G (Kernel.apply), solved by GMRES for every source and
        transfer at once. x has a static part on the boson grid and a part on the vertex box, so beyond the
        transfer box the vertex is static. A solve above Numerics.hedin.bse_tol after its bse_max_iter
        iterations keeps its result and raises the flag 'hedin_bse_not_converged'."""
        hedin = self.numerics.hedin
        ctx, g = self.ctx, self.grids
        N, naux = ctx.N, ctx.naux
        s_shape = (len(g.boson_m), naux, N, N)
        d_shape = (len(g.vertex_m), naux, 2 * g.n_vertex, N, N)
        ns = int(np.prod(s_shape))

        def three_point(v, bare):
            static = v[:ns].reshape(s_shape) + (ctx.B[None] if bare else 0)
            return ThreePoint(ctx, False, static, v[ns:].reshape(d_shape))

        def flat(t):
            out = np.zeros(ns + int(np.prod(d_shape)), dtype=complex)
            if t is not None and t.static is not None:
                out[:ns] = t.static.ravel()
            if t is not None and t.dyn is not None:
                out[ns:] = t.dyn.ravel()
            return out

        size = ns + int(np.prod(d_shape))
        rhs = flat(K.apply(three_point(np.zeros(size, dtype=complex), bare=True)))
        A = LinearOperator((size, size), dtype=complex,
                           matvec=lambda v: v - flat(K.apply(three_point(v, bare=False))))
        # scipy counts maxiter in restart cycles: bse_max_iter counts the inner iterations, so whole
        # cycles of HEDIN_BSE_RESTART first and the remainder as one shorter cycle from where they stopped
        restart = min(HEDIN_BSE_RESTART, hedin.bse_max_iter)
        cycles, rest = divmod(hedin.bse_max_iter, restart)
        x, info = gmres(A, rhs, rtol=hedin.bse_tol, atol=0.0, restart=restart, maxiter=cycles)
        if info != 0 and rest:
            x, info = gmres(A, rhs, x0=x, rtol=hedin.bse_tol, atol=0.0, restart=rest, maxiter=1)
        if info != 0:
            residual = np.linalg.norm(A.matvec(x) - rhs) / np.linalg.norm(rhs)
            flag('hedin_bse_not_converged',
                 f'the BSE of a kernel at fixed form stopped at the relative residual {residual:.1e} '
                 f'(tolerance {hedin.bse_tol:.0e})', HedinBSEWarning)
        sol = three_point(x, bare=False)
        return ThreePoint(ctx, True, sol.static, sol.dyn, ('solve_vertex', G, K))

    def solve_l(self, G, K):
        vertex = self.solve_vertex(G, K)
        return Legs(G, vertex, ('solve_l', G, K))

    def vertex_legs(self, G, vertex):
        return Legs(G, vertex, ('vertex_legs', G, vertex))

    def apply_kernel(self, K, L):
        if not K.is_static:
            ctx, g = self.ctx, self.grids
            v = L.vertex
            Y = ThreePoint(ctx, False, np.stack([v.static_at(m) for m in g.boson_m]), v.dyn)
            X = K.apply(Y)
            if X is None:
                X = ThreePoint(ctx, False, np.zeros((len(g.boson_m), ctx.naux, ctx.N, ctx.N), dtype=complex))
            return ThreePoint(ctx, False, X.static, X.dyn, ('apply_kernel', K, L))
        ctx = self.ctx
        static = np.stack([K.static_apply(L.pair_density(m)) for m in ctx.grids.boson_m])
        return ThreePoint(ctx, False, static, None, ('apply_kernel', K, L))

    # ------------------------------------------------------------------------- bosons
    def polarizability(self, G, vertex):
        ctx = self.ctx
        legs = Legs(G, vertex)
        data = ctx.spin * np.stack([np.einsum('Pba,Qab->PQ', ctx.B, legs.pair_density(m)) for m in ctx.grids.boson_m])
        return Boson(ctx, 'P', data, ('polarizability', G, vertex))

    def screen(self, P):
        naux = self.ctx.naux
        data = np.linalg.inv(np.eye(naux)[None] - P.data)
        return Boson(self.ctx, 'M', data, ('screen', P))

    # -------------------------------------------------------------------- self-energies
    def tail_weights(self, G, n, M, M_lo=None):
        """T(nu)_rs = (1/beta) sum over m > M and m < -M_lo (M_lo = M by default) of
        G(nu + Omega_m)_rs / (i Omega_m) at the fermionic indices n, with G replaced by its model line and
        the sums done exactly: for a model level x, z = i nu - x and c = beta z / (2 pi i),

            (1/beta) sum_{m > M} 1 / ((i Omega_m + z) i Omega_m) = [psi(M + 1 + c) - psi(M + 1)] / (2 pi i z),
            (1/beta) sum_{m < -M'} 1 / ((i Omega_m + z) i Omega_m) = [psi(M' + 1) - psi(M' + 1 - c)] / (2 pi i z)."""
        g = self.ctx.grids
        M_lo = M if M_lo is None else M_lo
        z = 1j * g.nu(np.asarray(n))[:, None] - G.x[None, :]
        c = g.beta * z / (2j * np.pi)
        t = (psi(M + 1 + c) - psi(M + 1) + psi(M_lo + 1) - psi(M_lo + 1 - c)) / (2j * np.pi * z)
        return np.einsum('ra,na,sa->nrs', G.V, t, G.V.conj())

    def tail_pair_weights(self, G, n, m, M):
        """D[n, a, b] = (1/beta) sum over |w| > M of 1 / ((i nu_n + i Omega_m + i w - x_a)(i nu_n + i w - x_b) i w)
        for the model levels x of G: the tail of a line perturbed at the transfer Omega_m, in the model's
        eigenbasis. By partial fractions it is minus the divided difference of the one-line weight
        t(z) = (1/beta) sum_{|w| > M} 1 / ((i w + z) i w) at z_a = i (nu + Omega) - x_a and z_b = i nu - x_b,
        with t'(z) where they meet."""
        g = self.ctx.grids
        nu = g.nu(np.asarray(n))
        za = 1j * (nu + g.omega(m))[:, None, None] - G.x[None, :, None]
        zb = 1j * nu[:, None, None] - G.x[None, None, :]

        def t(z):
            c = g.beta * z / (2j * np.pi)
            return (psi(M + 1 + c) - psi(M + 1 - c)) / (2j * np.pi * z)

        def dt(z):
            c = g.beta * z / (2j * np.pi)
            dpsi = (_trigamma(M + 1 + c) + _trigamma(M + 1 - c)) * g.beta / (2j * np.pi)
            return dpsi / (2j * np.pi * z) - t(z) / z

        za, zb = np.broadcast_arrays(za, zb)
        near = np.abs(za - zb) < HEDIN_CONFLUENT
        with np.errstate(divide='ignore', invalid='ignore'):
            out = -(t(za) - t(zb)) / (za - zb)
        out[near] = -dt(za[near])
        return out

    def _three_point_sigma(self, G, coupling, vertex):
        """-(1/beta) sum_Omega sum_Q [C^Q(Omega) G(nu + Omega) X^Q(nu; Omega)] on the fermion grid, for the
        coupling C^Q(Omega) = coupling(m) and the three-point X (its static part with or without the bare
        vertex as it is stored).

        The static part is summed over the boson grid and the frequency-dependent part over the transfer
        box, and each sum is completed by its tail. The coupling and X side by side,
        Z(nu; Omega) = sum_Q C^Q(Omega) (x) X^Q(nu; Omega), with G(nu + Omega) to go between them, decays
        as Z_1(nu) / (i Omega). Z_1 is taken from the two ends, (i Omega Z(Omega) + i Omega' Z(Omega')) / 2
        at m = +-M, where the next order cancels, and the rest of the sum is tail_weights. The tail is
        linear in the summand at the two ends, so two routes whose summands agree at every transfer (the
        Gamma/W and I/L forms of one self-energy) agree with their tails too."""
        ctx, g = self.ctx, self.ctx.grids
        n = g.fermion_n
        box = slice(g.n_fermion - g.n_vertex, g.n_fermion + g.n_vertex)
        out = np.zeros((len(n), ctx.N, ctx.N), dtype=complex)

        def static(m):
            return vertex.static_at(m) if vertex.bare else vertex.correction_at(m)

        for m in g.boson_m:
            CG = np.einsum('Qpr,nrs->nQps', coupling(m), G.at(n + m))
            out -= np.einsum('nQps,Qsq->npq', CG, static(m))
            d = vertex.dyn_at(m)
            if d is not None:
                out[box] -= np.einsum('nQps,Qnsq->npq', CG[box], d)
        out = out / g.beta
        if vertex.bare or vertex.static is not None:
            M = g.m_boson
            T = self.tail_weights(G, n, M)
            for s in (M, -M):
                out -= 0.5j * g.omega(s) * np.einsum('Qpr,nrs,Qsq->npq', coupling(s), T, static(s))
        if vertex.dyn is not None:
            M = g.m_vertex
            T = self.tail_weights(G, g.vertex_n, M)
            for s in (M, -M):
                out[box] -= 0.5j * g.omega(s) * np.einsum('Qpr,nrs,Qnsq->npq', coupling(s), T, vertex.dyn_at(s))
        return out

    def screened_coupling(self, W, m):
        """C^Q(Omega_m) = sum_P (M - 1)(...) B^P, the screened part of W met by the vertex end Q, for the
        reading w_order: M_QP(Omega) for W(4,1) (the boson's outgoing end Q at the vertex, its incoming end
        P at the external point), M_PQ(-Omega) for W(1,4)."""
        eye = np.eye(self.ctx.naux)
        if self.w_order == '4,1':
            return np.einsum('QP,Ppq->Qpq', W.at(m) - eye, self.ctx.B)
        return np.einsum('PQ,Ppq->Qpq', W.at(-m) - eye, self.ctx.B)

    def sigma_gw(self, G, W, vertex):
        """-G W Gamma = Sigma_x[G] - G (W - v) Gamma - G v (Gamma - 1), with W read as w_order says."""
        ctx, B = self.ctx, self.ctx.B
        correlation = self._three_point_sigma(G, lambda m: self.screened_coupling(W, m), vertex)
        if vertex.static is not None or vertex.dyn is not None:
            bare_part = ThreePoint(ctx, False, vertex.static, vertex.dyn)
            correlation = correlation + self._three_point_sigma(G, lambda m: B, bare_part)
        x = self.sigma_x(G)
        return SelfEnergy(ctx, x.static, correlation, ('sigma_gw', G, W, vertex))

    def sigma_il(self, G, X):
        """-v G X: the I/L closure without its Hartree-Fock part."""
        dyn = self._three_point_sigma(G, lambda m: self.ctx.B, X)
        N = self.ctx.N
        return SelfEnergy(self.ctx, np.zeros((N, N)), dyn, ('sigma_il', G, X))

    # ------------------------------------------------------------------ along the map
    def along_map(self, X, legs):
        """(delta X / delta G^(0)) . legs: the tangent of X with the start line perturbed along the legs'
        vertex (amputated: delta G^(0) = G^(0) Gamma_legs G^(0)), one source per Coulomb factor."""
        from src.SingleReference.HedinCycles.matsubara.tangents import LineTangent, Seed, Tangents
        if legs.line is not self.line0:
            raise ValueError('the legs of the derivative along the map must sit on the start line')
        g = self.grids
        Y = ThreePoint(self.ctx, False, np.stack([legs.vertex.static_at(m) for m in g.boson_m]),
                       legs.vertex.dyn)
        t = Tangents(self, Seed(self.line0, Y))(X)
        if isinstance(t, LineTangent):
            return t
        if t is None:
            N, naux = self.ctx.N, self.ctx.naux
            t = ThreePoint(self.ctx, False, np.zeros((len(g.boson_m), naux, N, N), dtype=complex))
            t.coeff = np.zeros((len(g.boson_m), naux, naux), dtype=complex)
        t.recipe = ('along_map', X, legs)     # a later tangent of t has no rule yet (cycle 3)
        return t

    # ---------------------------------------------------------------------- diagnostics
    def cycle_diagnostics(self, hist, n):
        """TEST_PLAN's per-cycle diagnostics of cycle n of a run (section 8, reported and not asserted): a dict
        of named numbers, each the worst over its grid, read from the run without changing it.

        excess_charge   tr rho(G^(n)) minus the electron count. The Dyson lines keep the start's mu and need
                        not conserve the particle number (H2/STO-3G from 'H': +0.019 at cycle 1).
        hartree_charge  the same for the density that cycle n's Hartree term is built on (n >= 1): the
                        excess charge of G^(n-1), or with hartree='linearized' that of its linearized density.
        sigma_change    max |Sigma^(n) - Sigma^(n-1)| over the fermion grid (n >= 1).
        dyson_min       the smallest singular value of 1 - G^(0) (Sigma^(n) - Sigma^(0)) over the fermion grid
                        (n >= 1), the distance of the cycle's Dyson equation from singular.
        screening_min   the smallest |eigenvalue| of 1 - P^(n)(Omega) over the boson grid, where the run keeps
                        W^(n) (SG-Gamma/W from cycle 2).
        chain_min       the smallest |eigenvalue| of 1 + Lambda_H over the boson grid, SG-Gamma/W's chain-rule
                        factor (from cycle 2; it is 1 at cycle 2 with the lag, where Lambda_H vanishes).
        bubble_charge   the largest charge tr(G^(n-1) Gamma^Q G^(n-1)) that a source Q induces at Omega_m,
                        m = 1, 2, 4, 8 (inside the boson grid), where the run keeps Gamma^(n) (SG-Gamma/W from
                        cycle 2). A bubble that conserves charge has none.
        start_bse_min   (n = 0) the smallest |eigenvalue| of 1 - K chi(0) for the start's two BSEs, as the pair
                        (I/L: Hartree plus the start's exchange; Gamma/W: the start's exchange, 1 from 'H'),
                        dense in pair space and None above HEDIN_DIAGNOSTICS_MAX_N orbitals. In spin orbitals
                        it holds every spin channel, for a closed shell in spatial orbitals the singlet.
        start_bse_min_triplet  (n = 0, spatial orbitals) the same in the triplet channel, where the Hartree
                        term drops out: the minimum of the two is the spin-orbital start_bse_min.

        The BSE of a kernel at fixed form (the cycle-based schemes) is solved by GMRES, and its spectrum is not
        formed."""
        from src.SingleReference.HedinCycles.schemes import start_level
        ctx, g = self.ctx, self.grids
        G = hist.lines[n]
        s = ctx.spin
        out = {'excess_charge': float(s * np.trace(G.density()).real - self.system.nelec)}
        if n == 0:
            out['start_bse_min'] = self._start_bse_min(start_level(self, self.start_name))
            if s == 2:
                out['start_bse_min_triplet'] = self._start_bse_min(start_level(self, self.start_name), triplet=True)
            return out
        out['sigma_change'] = float(np.abs(hist.sigmas[n].at_grid() - hist.sigmas[n - 1].at_grid()).max())
        if n < len(hist.hartree_lines):
            out['hartree_charge'] = float(s * np.trace(hist.hartree_lines[n].density()).real - self.system.nelec)
        d_sigma = (hist.sigmas[n] - hist.sigmas[0]).at_grid()
        D = np.eye(ctx.N)[None] - self.line0.at(g.fermion_n) @ d_sigma
        out['dyson_min'] = float(np.linalg.svd(D, compute_uv=False).min())
        W = hist.screened[n] if n < len(hist.screened) else None
        if W is not None:
            out['screening_min'] = float(min(np.abs(np.linalg.eigvals(np.linalg.inv(M))).min() for M in W.data))
        vertex = hist.vertices[n] if n < len(hist.vertices) else None
        if vertex is not None:
            lam_h = vertex.recipe[2] if vertex.recipe[0] == 'sg_vertex' else None
            if lam_h is not None and hasattr(lam_h, 'coeff'):
                factor = np.eye(ctx.naux)[None] + np.swapaxes(lam_h.coeff, 1, 2)
                out['chain_min'] = float(min(np.abs(np.linalg.eigvals(F)).min() for F in factor))
            legs = Legs(hist.lines[n - 1], vertex)
            ms = [m for m in (1, 2, 4, 8) if m <= g.m_boson]
            out['bubble_charge'] = float(s * max(np.abs(np.trace(legs.pair_density(m), axis1=-2, axis2=-1)).max()
                                             for m in ms))
        return out

    def _start_bse_min(self, S0, triplet=False):
        """The smallest |eigenvalue| of 1 - K chi(0) for the start's I/L and Gamma/W BSEs (None, None above
        HEDIN_DIAGNOSTICS_MAX_N orbitals), in the triplet channel with triplet=True."""
        N = self.ctx.N
        if N > HEDIN_DIAGNOSTICS_MAX_N:
            return (None, None)
        chi = self.line0.pair_map(0).reshape(N * N, N * N)
        out = []
        for S in (lambda G: self.sigma_h(G) + S0(G), S0):
            K = self.kernel(S, self.line0)
            if K.coefficients == (0.0, 0.0) or (triplet and K.coefficients[1] == 0.0):
                out.append(1.0)
                continue
            out.append(float(np.abs(np.linalg.eigvals(np.eye(N * N) - K.static_matrix(triplet) @ chi)).min()))
        return tuple(out)

    def sg_vertex(self, lam, lam_h):
        """Gamma^Q = B^Q + sum_R Lambda^R [(1 + D)^-1]_RQ, with D_PR the Hartree coefficients of Lambda_H
        (the response to the source R of a difference of Hartree self-energies is sum_P B^P D_PR)."""
        ctx, g = self.ctx, self.grids
        if not hasattr(lam_h, 'coeff'):
            raise ValueError('the second piece of the SG vertex must be the tangent of Hartree self-energies')
        D = np.swapaxes(lam_h.coeff, 1, 2)                                       # (w, P, R)
        inv = np.linalg.inv(np.eye(ctx.naux)[None] + D)                         # (w, R, Q)
        static = None
        if lam.static is not None:
            static = np.einsum('wRpq,wRQ->wQpq', lam.static, inv)
        dyn = None
        if lam.dyn is not None:
            iw = g.boson_index(g.vertex_m)
            dyn = np.einsum('kRnpq,kRQ->kQnpq', lam.dyn, inv[iw])
        return ThreePoint(ctx, True, static, dyn, ('sg_vertex', lam, lam_h))
