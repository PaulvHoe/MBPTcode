"""First-order tangents of the Matsubara backend: the derivative along the map.

A tangent is the response of an object of a cycle's history to a perturbation of a leaf line, one
transfer frequency Omega at a time, for a set of directions R. Frequencies follow backend.py: a
perturbed two-point object has nu + Omega at its outgoing end and nu at its incoming end, and a perturbed
boson is stored at its incoming frequency omega (outgoing omega + Omega). A perturbed line is kept
amputated, delta G(nu; Omega) = G(nu + Omega) Y(nu; Omega) G(nu), with Y a three-point object over the
directions. A Dyson line then has Y = Y_0 + delta(Sigma - Sigma^(0)), where Y_0 is the start line's.

The derivative along the map seeds the start line with the legs' vertex, Y_0 = Gamma_legs (the start's
vertex for SG-Gamma/W, the reducible vertex of L^(1) for SG-I/L). Every object of cycle 1 is built on the
start line, a model line, so the sums of three propagators are exact: with z_k = x_k - i Omega_k the
pole of the k-th propagator,

    (1/beta) sum_nu prod_k 1/(i nu + i Omega_k - x_k) = f[z_1, z_2, z_3],

the second divided difference of the Fermi function f (f(z_k) = f(x_k), since Omega_k is bosonic).
The parts of a tangent that do not depend on nu (Hartree and exchange of the perturbed density) are kept
on every transfer of the boson grid, the parts that do on the vertex box for the transfers of the vertex
box. Internal boson sums run over the boson grid, dropping the frequencies whose shifted partner leaves it,
and are completed beyond it by the tails of backend._three_point_sigma (the line term with the sum of two
model lines over 1/(i w), backend.tail_pair_weights). At Omega = 0 a tangent is then the derivative of the
discrete self-energy, tails included (finite differences to 1e-9). At Omega != 0 it is the same
construction, whose shifted sums lose the frequencies near the grid's edge.

Memory: the vertex and boson tangents are built one transfer at a time (PerTransfer), and the sums over
internal bosonic frequencies run in blocks that fit Numerics.layout.tile_gb (HEDIN_TILE_GB when None),
each block one batched matrix product where the contraction allows.

On a line with a frequency-dependent self-energy, or along a direction with a part on the vertex box, a
triangle is the model line's exact sum with the static part of the direction plus the rest, summed where
it lives: where a line differs from its model (the fermion grid) or the direction depends on nu (the vertex
box). The vertices inside a differentiated functional are static (those of cycle 1, the level that the
kernels at fixed form of cycle 2 differentiate).
"""
import numpy as np
from scipy.special import expit

from src.SingleReference.HedinCycles.matsubara.backend import ThreePoint, depends_on

from src.Base.numerics.constants import HEDIN_CONFLUENT as _CONFLUENT
from src.Base.numerics.constants import HEDIN_TILE_GB


class LineTangent:
    """delta G(nu; Omega) = G(nu + Omega) Y G(nu); Y None for a line the seed does not reach."""

    def __init__(self, line, Y):
        self.line, self.Y = line, Y


class PerTransfer:
    """A tangent built one transfer of the vertex box at a time: at(k) computes transfer k on demand and
    keeps only the last one, so that no tangent is held for every transfer at once. Its consumers run
    over the transfers in order."""

    def __init__(self, fn):
        self.fn, self._k, self._value = fn, None, None

    def at(self, k):
        if self._k != k:
            self._value, self._k = self.fn(k), k
        return self._value


class BosonTangent(PerTransfer):
    """at(k)[iomega, R, P, Q]: delta X_PQ(omega; Omega_k) at the incoming omega, over the boson grid."""


class VertexTangent(PerTransfer):
    """at(k)[iomega, R, Q, N, N]: the tangent of a vertex (or of K L) that does not depend on nu."""


class LegsTangent:
    def __init__(self, line_tangent, vertex_tangent, legs):
        self.line_tangent, self.vertex_tangent, self.legs = line_tangent, vertex_tangent, legs


class Seed:
    """The leaf line perturbed and its amputated direction Y (a ThreePoint over the directions R)."""

    def __init__(self, leaf, Y):
        self.leaf, self.Y = leaf, Y
        self.nsrc = (Y.static if Y.static is not None else Y.dyn).shape[1]


# ------------------------------------------------------------------------------ Fermi sums
def _f(x, beta):
    return expit(-beta * x)


def _df(x, beta):
    f = _f(x, beta)
    return -beta * f * (1 - f)


def _d2f(x, beta):
    f = _f(x, beta)
    return beta ** 2 * f * (1 - f) * (1 - 2 * f)


def _dd2(xa, za, xb, zb, beta):
    d = za - zb
    near = np.abs(d) < _CONFLUENT
    with np.errstate(divide='ignore', invalid='ignore'):
        regular = (_f(xa, beta) - _f(xb, beta)) / d
    return np.where(near, _df(xa, beta), regular)


def _dd3(xa, za, xb, zb, xc, zc, beta):
    """f[za, zb, zc] with confluent poles handled, arrays broadcast."""
    ac, ab = np.abs(za - zc), np.abs(za - zb)
    with np.errstate(divide='ignore', invalid='ignore'):
        r1 = (_dd2(xa, za, xb, zb, beta) - _dd2(xb, zb, xc, zc, beta)) / (za - zc)
        r2 = (_df(xa, beta) - _dd2(xc, zc, xb, zb, beta)) / (za - zb)
    r3 = 0.5 * _d2f(xa, beta)
    return np.where(ac >= _CONFLUENT, r1, np.where(ab >= _CONFLUENT, r2, r3))


def triangle_weights(line, shifts):
    """D[w, a, b, c] = (1/beta) sum_nu G_a(nu + i s_a) G_b(nu + i s_b) G_c(nu + i s_c) for the model
    levels of line, shifts = (s_a, s_b, s_c) bosonic indices (arrays over w broadcast together)."""
    g = line.ctx.grids
    x = line.x
    sa, sb, sc = (np.asarray(s)[:, None, None, None] for s in shifts)
    za = x[None, :, None, None] - 1j * g.omega(sa)
    zb = x[None, None, :, None] - 1j * g.omega(sb)
    zc = x[None, None, None, :] - 1j * g.omega(sc)
    xa, xb, xc = x[None, :, None, None], x[None, None, :, None], x[None, None, None, :]
    return _dd3(xa, za, xb, zb, xc, zc, g.beta)


def pair_weights(line, ms):
    """F[w, a, b] = (1/beta) sum_nu G_a(nu + Omega_w) G_b(nu) for the model levels, ms bosonic indices."""
    g = line.ctx.grids
    x = line.x
    om = g.omega(np.asarray(ms))[:, None, None]
    xa, xb = x[None, :, None], x[None, None, :]
    return _dd2(xa, xa - 1j * om, xb, xb + 0j * om, g.beta)


def _to_eig(line, X):
    return line.V.conj().T @ X @ line.V


def _from_eig(line, X):
    return line.V @ X @ line.V.conj().T


# ---------------------------------------------------------------------------------- engine
class Tangents:
    """tangent(obj) for one seed, memoized over the history."""

    def __init__(self, backend, seed):
        self.backend, self.seed, self.ctx = backend, seed, backend.ctx
        self.g = self.ctx.grids
        self.memo = {}
        self.transfers = self.g.vertex_m
        self.bosons = self.g.boson_m

    def __call__(self, obj):
        key = id(obj)
        if key not in self.memo:
            self.memo[key] = self._tangent(obj)
        return self.memo[key]

    # ---------------------------------------------------------------------- dispatch
    def _tangent(self, obj):
        op = obj.recipe[0]
        if op in ('line0', 'leaf', 'line'):
            return LineTangent(obj, self.seed.Y if obj is self.seed.leaf else None)
        if op == 'dyson':
            y0 = self(self.backend.line0).Y
            return LineTangent(obj, _add(y0, self(obj.recipe[1])))
        if op == 'zero':
            return None
        if op in ('add', 'sub'):
            a, b = self(obj.recipe[1]), self(obj.recipe[2])
            return _add(a, b) if op == 'add' else _add(a, _neg(b))
        if op == 'neg':
            return _neg(self(obj.recipe[1]))
        if op == 'sigma_h':
            return self._hartree(self(obj.recipe[1]))
        if op == 'sigma_x':
            return self._exchange(self(obj.recipe[1]))
        if op == 'sigma_gw':
            return self._sigma_gw(*obj.recipe[1:])
        if op == 'sigma_il':
            return self._sigma_il(*obj.recipe[1:])
        if op == 'screen':
            return self._screen(obj)
        if op == 'polarizability':
            return self._polarizability(*obj.recipe[1:])
        if op == 'solve_vertex':
            return self._solve_vertex(obj, *obj.recipe[1:])
        if op in ('solve_l', 'vertex_legs'):
            return LegsTangent(self(obj.line), self(obj.vertex), obj)
        if op == 'apply_kernel':
            return self._apply_kernel(*obj.recipe[1:])
        if not depends_on(obj, self.seed.leaf):
            return None
        raise NotImplementedError(f'no tangent rule for {op!r} yet')

    # ------------------------------------------------------------------ densities
    def _density(self, t):
        """delta rho(Omega) = (1/beta) sum over all nu of delta G(nu; Omega): (2 m_boson + 1, R, N, N)."""
        line, Y = t.line, t.Y
        out = np.zeros((len(self.bosons), self.seed.nsrc, self.ctx.N, self.ctx.N), dtype=complex)
        for i, m in enumerate(self.bosons):
            if Y.static is not None:
                out[i] = line.pair_sum(Y.static[i], m)
            d = Y.dyn_at(m)
            if d is not None:
                out[i] = out[i] + line.box_sum(d, m)
        return out

    def _hartree(self, t):
        if t is None or t.Y is None:
            return None
        B = self.ctx.B
        coeff = np.einsum('Psr,wRrs->wRP', B, self._density(t))
        out = ThreePoint(self.ctx, False, np.einsum('wRP,Ppq->wRpq', coeff, B))
        out.coeff = coeff
        return out

    def _exchange(self, t):
        if t is None or t.Y is None:
            return None
        B = self.ctx.B
        return ThreePoint(self.ctx, False, -np.einsum('Ppr,wRrs,Psq->wRpq', B, self._density(t), B))

    # ------------------------------------------------------------------- triangles
    def _direction(self, t, m):
        """The direction of a line tangent at the transfer m: its static part (R, N, N) and its part on the
        vertex box (R, 2 n_vertex, N, N) or None."""
        Y = t.Y
        R, N = self.seed.nsrc, self.ctx.N
        Ys = Y.static[self.g.boson_index(m)] if Y.static is not None else np.zeros((R, N, N), dtype=complex)
        return Ys, Y.dyn_at(m)

    def _shifted_direction(self, Ys, Yd, n):
        """Y(nu_n) = Y_s + Y_d(nu_n) at the fermionic indices n (any shape), Y_d zero off the vertex box:
        (*n.shape, R, N, N)."""
        n = np.asarray(n)
        Y = np.broadcast_to(Ys, n.shape + Ys.shape).copy()
        if Yd is not None:
            idx = n + self.g.n_vertex
            inside = (idx >= 0) & (idx < 2 * self.g.n_vertex)
            Y[inside] += np.moveaxis(Yd[:, idx[inside]], 0, 1)
        return Y

    def _perturbed(self, line, Ys, Yd, m, n, model=False):
        """D(nu) = G(nu + O) Y(nu) G(nu) at the fermionic indices n (1-D), O = m: (n, R, N, N). With
        model=True the model line and the static part of the direction alone."""
        at = line.model if model else line.at
        Y = np.broadcast_to(Ys, (len(n),) + Ys.shape) if model else self._shifted_direction(Ys, Yd, n)
        return np.einsum('nab,nRbc,ncd->nRad', at(n + m), Y, at(n))

    def triangle(self, t, vertex_static, m):
        """T[w, R, Q] = (1/beta) sum_nu [G(nu+w+O) Y(nu+w) G(nu+w) Gamma(w) G(nu) + G(nu+w+O) Gamma(w) G(nu+O) Y(nu) G(nu)]
        over the boson grid w, for the transfer O = m: (n_w, R, Q, N, N). The vertex Gamma(w) is static.
        With D(nu) = G(nu + O) Y(nu) G(nu) the perturbed line, the two terms are
        sum_nu' D(nu') Gamma(w) G(nu' - w) and sum_nu G(nu + w + O) Gamma(w) D(nu): the model line's exact
        sums with the static direction, plus D G - D_m G_m = (D - D_m) G + D_m (G - G_m) (and its mirror)
        summed where it lives, in blocks of w, each sum one batched matrix product (_pair_contract)."""
        line = t.line
        Ys, Yd = self._direction(t, m)
        out = self._triangle_model(line, Ys, vertex_static, m)
        if line.dyn is None and Yd is None:
            return out
        g = self.g
        nf, w = g.n_fermion, self.bosons
        N, R, Q = self.ctx.N, self.seed.nsrc, vertex_static.shape[1]
        # the perturbed line minus its model, where they differ
        nD = np.arange(-nf - abs(m), nf + abs(m))
        Dd = self._perturbed(line, Ys, Yd, m, nD) - self._perturbed(line, Ys, Yd, m, nD, model=True)
        dynamic = line.dyn is not None
        if dynamic:
            # the model's perturbed line wherever it meets a line that differs from its model
            span = nf + g.m_boson + abs(m) + 1
            Dm = self._perturbed(line, Ys, None, m, np.arange(-span, span), model=True)   # (nU, R, N, N)
            dG = line.G - line.model(g.fermion_n)                                          # (ng, N, N)
        acc = np.zeros_like(out)
        per_w = 16 * N * N * (2 * len(nD) * (Q + 1) + (2 * nf * (2 * Q + 2 * R) if dynamic else 0))
        for sl in self._wblocks(per_w):
            ww, gm = w[sl], vertex_static[sl]
            # sum_nu' (D - D_m)(nu') Gam(w) G(nu' - w)
            Gl = _at2(line, nD[None, :] - ww[:, None])                                     # (b, nD, N, N)
            acc[sl] += _pair_contract(Dd, gm[:, None] @ Gl[:, :, None])
            # sum_nu G(nu + w + O) Gam(w) (D - D_m)(nu)
            Gr = _at2(line, nD[None, :] + ww[:, None] + m)
            acc[sl] += np.swapaxes(_pair_contract(Gr[:, :, None] @ gm[:, None], Dd), -4, -3)
            if dynamic:
                # sum_ng D_m(ng + w) Gam(w) dG(ng) and sum_ng dG(ng) Gam(w) D_m(ng - w - O)
                Dl = np.stack([Dm[span + x - nf:span + x + nf] for x in ww])                # (b, ng, R, N, N)
                Dr = np.stack([Dm[span - x - m - nf:span - x - m + nf] for x in ww])
                acc[sl] += _pair_contract(Dl, gm[:, None] @ dG[None, :, None])
                acc[sl] += np.swapaxes(_pair_contract(dG[None, :, None] @ gm[:, None], Dr), -4, -3)
        return out + acc / g.beta

    def _wblocks(self, bytes_per_w):
        """Slices of the boson grid whose arrays of bytes_per_w each fit Numerics.layout.tile_gb
        (HEDIN_TILE_GB when it is None)."""
        cap = self.backend.numerics.layout.tile_gb
        cap = HEDIN_TILE_GB if cap is None else cap
        size = max(1, int(cap * 2 ** 30 // max(1, bytes_per_w)))
        nw = len(self.bosons)
        return [slice(s, min(s + size, nw)) for s in range(0, nw, size)]

    def _triangle_model(self, line, Ys, vertex_static, m):
        """The triangle on the model line with the static direction Y_s, summed exactly (triangle_weights)."""
        Y = _to_eig(line, Ys)                                                       # (R, N, N)
        Gam = _to_eig(line, vertex_static)                                          # (w, Q, N, N)
        w = self.bosons
        DA = triangle_weights(line, (w + m, w, np.zeros_like(w)))
        DB = triangle_weights(line, (w + m, np.full_like(w, m), np.zeros_like(w)))
        out = (np.einsum('wabc,Rab,wQbc->wRQac', DA, Y, Gam)
               + np.einsum('wabc,wQab,Rbc->wRQac', DB, Gam, Y))
        return _from_eig(line, out)

    def _vertex_static_all(self, vertex):
        return np.stack([vertex.static_at(m) for m in self.bosons])                 # (w, Q, N, N)

    # ----------------------------------------------------------------------- bosons
    def _polarizability(self, G, vertex):
        tG, tV = self(G), self(vertex)
        if (tG is None or tG.Y is None) and tV is None:
            return None
        B = self.ctx.B
        if vertex.dyn is not None:
            raise NotImplementedError('the polarizability tangent of a dynamic vertex is milestone 2c')
        gam = self._vertex_static_all(vertex)

        def at(k):
            m = self.transfers[k]
            S = np.zeros((len(self.bosons), self.seed.nsrc, self.ctx.naux, self.ctx.N, self.ctx.N), dtype=complex)
            if tG is not None and tG.Y is not None:
                S += self.triangle(tG, gam, m)
            if tV is not None:
                S += self._chi_shifted(G, tV.at(k), m)
            return np.einsum('Pba,wRQab->wRPQ', B, S)
        return BosonTangent(at)

    def _chi_shifted(self, line, X, m):
        """(1/beta) sum_nu G(nu + w + O) X(w) G(nu) for X (w, R, Q, N, N) on the boson grid, O = m."""
        if line.dyn is not None:
            return np.stack([np.einsum('abcd,...cd->...ab', line.pair_map(w + m), X[i])
                             for i, w in enumerate(self.bosons)])
        F = pair_weights(line, self.bosons + m)                                     # (w, N, N)
        Xt = _to_eig(line, X)
        return _from_eig(line, Xt * F[:, None, None])

    def _screen(self, M):
        tP = self(M.recipe[1])
        if tP is None:
            return None

        def at(k):
            m = self.transfers[k]
            dP = tP.at(k)
            out = np.zeros_like(dP)
            j = self.g.boson_index(self.bosons + m)
            ok = (j >= 0) & (j < len(self.bosons))
            out[ok] = (M.data[j[ok]][:, None] @ dP[ok]) @ M.data[ok][:, None]
            return out
        return BosonTangent(at)

    # ---------------------------------------------------------------------- vertices
    def _solve_vertex(self, vertex, G, K):
        if not K.is_static:
            raise NotImplementedError('the tangent of a vertex with a dynamic kernel is milestone 2c')
        tG = self(G)
        if K.coefficients == (0.0, 0.0) or tG is None or tG.Y is None:
            return None
        N, naux, R = self.ctx.N, self.ctx.naux, self.seed.nsrc
        Kmat = K.static_matrix()
        gam = self._vertex_static_all(vertex)

        def at(k):
            m = self.transfers[k]
            T = self.triangle(tG, gam, m)                                                # (w, R, Q, N, N)
            F = pair_weights(G, self.bosons + m) if G.dyn is None else None               # (w, N, N)
            out = np.empty((len(self.bosons), R, naux, N, N), dtype=complex)
            for i, w in enumerate(self.bosons):
                chi = _pair_matrix(G, F[i]) if F is not None else G.pair_map(w + m).reshape(N * N, N * N)
                A = np.eye(N * N) - Kmat @ chi
                rhs = Kmat @ T[i].reshape(R * naux, N * N).T
                out[i] = np.linalg.solve(A, rhs).T.reshape(R, naux, N, N)
            return out
        return VertexTangent(at)

    def _apply_kernel(self, K, L):
        if not K.is_static:
            raise NotImplementedError('the tangent of a dynamic kernel applied to L is milestone 2c')
        tL = self(L)
        tG, tV = tL.line_tangent, tL.vertex_tangent
        if (tG is None or tG.Y is None) and tV is None:
            return None
        gam = self._vertex_static_all(L.vertex)

        def at(k):
            m = self.transfers[k]
            S = np.zeros((len(self.bosons), self.seed.nsrc, self.ctx.naux, self.ctx.N, self.ctx.N), dtype=complex)
            if tG is not None and tG.Y is not None:
                S += self.triangle(tG, gam, m)
            if tV is not None:
                S += self._chi_shifted(L.line, tV.at(k), m)
            return K.static_apply(S)
        return VertexTangent(at)

    # ------------------------------------------------------------------ self-energies
    def _sigma_terms(self, G, coupling, vertex, dcoupling, t_vertex, bare_with_correction):
        """The part of the self-energy tangent that depends on nu, on the vertex box for the transfers of
        the vertex box: -(1/beta) sum_w [ C(w) dG(nu+w; O) Gam(w) + dC(w; O) G(nu+w) Gam(w)
        + C(w) G(nu+w+O) dGam(w; O) ], where C^Q(w) = coupling(w) (with the bare interaction added to its
        vertex correction when bare_with_correction) and dC = dcoupling(k, O) (_coupling_tangent).

        dC and dGam decay as 1/(i w) (a triangle whose third line carries w), so the last two terms
        are completed beyond the summed range as in backend._three_point_sigma: the two factors side by
        side decay as Z_1 / (i w), Z_1 is taken from the two ends of the range, and the line between them
        is summed exactly (backend.tail_weights). At Omega = 0 this is the derivative of the self-energy's
        own tail. Without it the tangent of H2's Sigma^(1) from (HF) misses its finite differences by
        1.2e-3 at m_boson = 128, falling as 1/m_boson (on the dimer the coefficient vanishes)."""
        ctx, g = self.ctx, self.g
        B, N, R = ctx.B, ctx.N, self.seed.nsrc
        n = g.vertex_n
        tG = self(G)
        gam = self._vertex_static_all(vertex)                                          # (w, Q, N, N)
        corr = gam - (B[None] if vertex.bare else 0)
        bare_part = bare_with_correction and (vertex.static is not None or not vertex.bare)
        C = np.stack([coupling(w) for w in self.bosons])                               # (w, Q, N, N)
        Q = C.shape[1]
        out = np.zeros((len(self.transfers), R, len(n), N, N), dtype=complex)
        tail = np.zeros_like(out)
        mb = g.m_boson
        Gw = _at2(G, n[None, :] + self.bosons[:, None])                                 # (w, n, N, N)
        blocks = self._wblocks(16 * N * N * len(n) * (Q * R * 2 + 2 * Q + R))
        for k, m in enumerate(self.transfers):
            Gwm = _at2(G, n[None, :] + self.bosons[:, None] + m)                        # (w, n, N, N)
            if tG is not None and tG.Y is not None:
                Ys, Yd = self._direction(tG, m)
                for sl in blocks:
                    Y = self._shifted_direction(Ys, Yd, n[None, :] + self.bosons[sl, None])     # (b, n, R, N, N)
                    left = C[sl, None] @ Gwm[sl, :, None]                               # (b, n, Q, N, N)
                    right = Gw[sl, :, None] @ gam[sl, None]
                    out[k] -= ((left[:, :, :, None] @ Y[:, :, None]) @ right[:, :, :, None]).sum(axis=(0, 2)).swapaxes(0, 1)
                    if bare_part:
                        left = B[None, None] @ Gwm[sl, :, None]
                        right = Gw[sl, :, None] @ corr[sl, None]
                        out[k] -= ((left[:, :, :, None] @ Y[:, :, None]) @ right[:, :, :, None]).sum(axis=(0, 2)).swapaxes(0, 1)
                # beyond the grid the perturbed line carries the static direction alone
                Yt = _to_eig(G, Ys)                                                     # (R, N, N)
                D = self.backend.tail_pair_weights(G, n, m, mb)                         # (n, N, N)
                T2 = _from_eig(G, Yt[:, None] * D[None])                                # (R, n, N, N)
                for e in (mb, -mb):
                    i = g.boson_index(e)
                    edge = np.einsum('Qpr,Rnrs,Qsq->Rnpq', C[i], T2, gam[i])
                    if bare_part:
                        edge = edge + np.einsum('Qpr,Rnrs,Qsq->Rnpq', B, T2, corr[i])
                    tail[k] -= 0.5j * g.omega(e) * edge
            if dcoupling is not None:
                dC = dcoupling(k, m)                                                    # (w, R, Q, N, N)
                # the sums over w, Q and the inner index as one matrix product per block of w
                for sl in blocks:
                    GG = Gw[sl, :, None] @ gam[sl, None]                                # (b, n, Q, N, N)
                    lhs = dC[sl].transpose(1, 3, 0, 2, 4).reshape(R * N, -1)           # (R a, w Q b)
                    rhs = GG.transpose(0, 2, 3, 1, 4).reshape(-1, len(n) * N)          # (w Q b, n d)
                    out[k] -= (lhs @ rhs).reshape(R, N, len(n), N).transpose(0, 2, 1, 3)
                hi, lo = min(mb, mb + m), max(-mb, -mb + m)          # the range where dC is known
                T = self.backend.tail_weights(G, n, hi, -lo)
                for e in (hi, lo):
                    i = g.boson_index(e)
                    tail[k] -= 0.5j * g.omega(e) * np.einsum('RQpr,nrs,Qsq->Rnpq', dC[i], T, gam[i])
            if t_vertex is not None:
                full = C + (B[None] if bare_with_correction else 0)
                dV = t_vertex.at(k)                                                     # (w, R, Q, N, N)
                # the sums over w, Q and the inner index as one matrix product per block of w
                for sl in blocks:
                    FG = full[sl, None] @ Gwm[sl, :, None]                              # (b, n, Q, N, N)
                    lhs = FG.transpose(1, 3, 0, 2, 4).reshape(len(n) * N, -1)          # (n a, w Q s)
                    rhs = dV[sl].transpose(0, 2, 3, 1, 4).reshape(-1, R * N)           # (w Q s, R q)
                    out[k] -= (lhs @ rhs).reshape(len(n), N, R, N).transpose(2, 0, 1, 3)
                T = self.backend.tail_weights(G, n + m, mb)
                for e in (mb, -mb):
                    i = g.boson_index(e)
                    tail[k] -= 0.5j * g.omega(e) * np.einsum('Qpr,nrs,RQsq->Rnpq', full[i], T, dV[i])
        return ThreePoint(ctx, False, None, out / g.beta + tail)

    def _sigma_gw(self, G, W, vertex):
        if vertex.dyn is not None:
            raise NotImplementedError('the self-energy tangent of a dynamic vertex is milestone 2c')
        tG = self(G)
        x = self._exchange(tG)
        tW = self(W)
        dcoupling = None if tW is None else (lambda k, m: self._coupling_tangent(tW, k, m))
        corr = self._sigma_terms(G, lambda w: self.backend.screened_coupling(W, w), vertex, dcoupling,
                                 self(vertex), bare_with_correction=True)
        return _add(x, corr)

    def _coupling_tangent(self, tW, k, m):
        """dC^Q(w; O) of the screened coupling for the transfer O = m (index k): (w, R, Q, N, N). The
        perturbed boson of W(4,1) leaves the external point P at w - O and reaches the vertex end Q at w,
        dC^Q = sum_P dM_QP(w - O; O) B^P, dropped where w - O leaves the boson grid. That of W(1,4) is
        sum_P dM_PQ(-w; O) B^P. The two agree wherever the perturbed W is symmetric (every cycle-1 level)."""
        g, B = self.g, self.ctx.B
        if self.backend.w_order == '4,1':
            j = g.boson_index(self.bosons - m)
            ok = (j >= 0) & (j < len(self.bosons))
            dW = tW.at(k)
            dM = np.zeros_like(dW)                                                                # (w, R, Q, P)
            dM[ok] = dW[j[ok]]
            return np.einsum('wRQP,Ppq->wRQpq', dM, B)
        return np.einsum('wRPQ,Ppq->wRQpq', tW.at(k)[g.boson_index(-self.bosons)], B)

    def _sigma_il(self, G, X):
        """-v G X: the coupling is the bare interaction, and X carries no bare vertex."""
        if X.dyn is not None:
            raise NotImplementedError('the self-energy tangent of an X with a frequency-dependent part '
                                      '(SG-I/L from cycle 3 on) is not written yet')
        return self._sigma_terms(G, lambda w: self.ctx.B, X, None, self(X), bare_with_correction=False)


# --------------------------------------------------------------------------------- helpers
def _pair_contract(L, R):
    """out[..., X, Y, a, d] = sum over n and b of L[..., n, X, a, b] R[..., n, Y, b, d], one batched matrix
    product over the leading axes (which broadcast)."""
    n, X, a, b = L.shape[-4:]
    Y, d = R.shape[-3], R.shape[-1]
    Lm = np.moveaxis(L, -4, -2).reshape(L.shape[:-4] + (X * a, n * b))
    Rm = np.moveaxis(R, -3, -2).reshape(R.shape[:-4] + (n * b, Y * d))
    out = (Lm @ Rm).reshape(np.broadcast_shapes(L.shape[:-4], R.shape[:-4]) + (X, a, Y, d))
    return np.moveaxis(out, -2, -3)


def _at2(line, n):
    """line.at for a 2-D array of fermionic indices: (*n.shape, N, N)."""
    return line.at(n.ravel()).reshape(n.shape + (line.ctx.N, line.ctx.N))


def _pair_matrix(line, F):
    """The pair sum with weights F (N, N) in the line's orbitals as an (N*N, N*N) matrix."""
    N = F.shape[0]
    V = line.V
    return np.einsum('ai,ci,ij,dj,bj->abcd', V, V.conj(), F, V, V.conj()).reshape(N * N, N * N)


def _neg(t):
    if t is None:
        return None
    out = ThreePoint(t.ctx, False, None if t.static is None else -t.static, None if t.dyn is None else -t.dyn)
    if hasattr(t, 'coeff'):
        out.coeff = -t.coeff
    return out


def _add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    static = a.static if b.static is None else (b.static if a.static is None else a.static + b.static)
    dyn = a.dyn if b.dyn is None else (b.dyn if a.dyn is None else a.dyn + b.dyn)
    out = ThreePoint(a.ctx, False, static, dyn)
    if hasattr(a, 'coeff') and hasattr(b, 'coeff'):
        out.coeff = a.coeff + b.coeff
    return out
