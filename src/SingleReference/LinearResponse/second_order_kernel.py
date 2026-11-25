"""The second-order Bethe-Salpeter kernel of BSE(2)@HF, BSE(2)@GW and BSE(1)@GF2
(van Hoegaerden and Foerster): A(2)(w), its frequency derivative and B(2) on
density-fitting factors, at any orbital energies.

The twelve terms of A(2)(w) (A_TERMS) are c (n1)(n2) / (D - w) summed over the
two indices of the resolvent that are not output indices (i, a, j, b); n1 and
n2 are chemist integrals (pq|rs), and in the six exchange-type terms the second
one carries the W slot, (pq|K|rs). With s = 1 for singlets and 0 for triplets:

    ph (kc):  2s(ac|kb)(ic|jk) - (ab|kc)(ic|K|jk) - (ac|kb)(kc|K|ij) + 2(ab|kc)(kc|ij)
              over D = e_b - e_i + e_c - e_k, and the same four with (b, i) -> (a, j)
              over D = e_a - e_j + e_c - e_k;
    pp (kl):  (kb|il)(ka|K|jl) - 2s(kb|il)(la|jk)   over D = e_a + e_b - e_k - e_l;
    hh (cd):  (cb|id)(ca|K|jd) - 2s(cb|id)(ad|jc)   over D = e_c + e_d - e_i - e_j.

These are the effective denominators D - w of the paper Fortran's
SecondorderADynamic, whose w = 0 limit is the static kernel of the paper's
Eq. 25; power 2 gives the table with 1/(D - w)^2, which is dA(2)/dw. The pp and hh
channels are the Fortran's lpp and lhh sums; the paper's 'pp off' drops both.

The fourth ph term of each resolvent, 2(ab|kc)(kc|ij) (RING_TERMS), is the ring:
the second-order piece v P0 v of the RPA screening of the direct term. An A(1)
with -(ij|W0|ab) holds it again, as 4 sum_kc (ab|kc)(kc|ij) / (e_c - e_k) (the
paper's appendix on double counting), and a B(1) with -(ib|W0|ja) its B
counterpart. rings=False drops both terms from the table, and with them from
A(2)(w), dA(2)/dw, B(2) (its relabellings) and the mirror.

T is the table, its mirror T~ the table with i <-> a and j <-> b relabelled in the
integrals (not in the resolvent), and B(2) the table at w = 0 with j <-> b
(B_p) or i <-> a (B_c) relabelled. The W slots follow one of three rules:

    'v'        K = v:          A(2) = T,               B(2) = B_c
    'average'  K = W0:         A(2) = (T + T~) / 2,    B(2) = (B_p + B_c) / 2
    '2sosex'   K = 2 W0 - v:   A(2) = (T + T~) / 2,    B(2) = (B_p + B_c) / 2

'v' is the kernel of BSE(2)@HF, BSE(1)@GF2 and BSE(2)_Wv@GW; '2sosex' is
BSE(2)@GW, equal to T(W0) + T~(W0) - T(v) by linearity (paper Eq. 28). W0 is the
full static screened interaction, v included. A(2) and B(2) are not symmetric
once W0 enters; the caller adds their symmetric parts, (M + M^T) / 2, as the
Fortran's SymmetrizeMatrix does. Matrices are indexed (ia),(jb), ia = i * nvirt + a.

Every term is evaluated in tiles over one output index from the three-index
factors, so no four-index tensor larger than one tile is held: B(2)'s hh terms
carry (vv|vv), which is never formed whole.
"""
import numpy as np

#: s of the singlet and triplet kernels.
SPIN_FACTOR = {'singlet': 1.0, 'triplet': 0.0}
#: The W-slot rules (module docstring).
SLOT_RULES = ('v', 'average', '2sosex')
#: The kernel's sums: the kc, kl and cd terms.
CHANNELS = ('ph', 'pp', 'hh')

#: (coefficient, n1, n2, whether n2 is a W slot, resolvent indices, channel);
#: '2s' and '-2s' carry the spin factor.
A_TERMS = (
    ('2s', 'ackb', 'icjk', False, 'ibkc', 'ph'),
    ('-1', 'abkc', 'icjk', True, 'ibkc', 'ph'),
    ('-1', 'ackb', 'kcij', True, 'ibkc', 'ph'),
    ('2', 'abkc', 'kcij', False, 'ibkc', 'ph'),
    ('2s', 'bcka', 'jcik', False, 'jakc', 'ph'),
    ('-1', 'abkc', 'jcik', True, 'jakc', 'ph'),
    ('-1', 'bcka', 'kcij', True, 'jakc', 'ph'),
    ('2', 'abkc', 'kcij', False, 'jakc', 'ph'),
    ('1', 'kbil', 'kajl', True, 'abkl', 'pp'),
    ('-2s', 'kbil', 'lajk', False, 'abkl', 'pp'),
    ('-2s', 'cbid', 'adjc', False, 'cdij', 'hh'),
    ('1', 'cbid', 'cajd', True, 'cdij', 'hh'),
)
#: The two ph ring terms 2(ab|kc)(kc|ij), one per resolvent (module docstring).
RING_TERMS = (A_TERMS[3], A_TERMS[7])
MIRROR = (('i', 'a'), ('j', 'b'))
SWAP_P = (('j', 'b'),)
SWAP_C = (('i', 'a'),)

_OCC = frozenset('ijkl')
_OUT = 'iajb'
_AUX = 'Z'


def _coefficient(code, s):
    return {'2s': 2.0 * s, '-2s': -2.0 * s, '-1': -1.0, '1': 1.0, '2': 2.0}[code]


def _relabel(letters, swaps):
    swap = {}
    for x, y in swaps:
        swap[x], swap[y] = y, x
    return ''.join(swap.get(ch, ch) for ch in letters)


def static_screening_aux(eps, coeff_df, nocc, tda=False):
    """The static RPA screened interaction W0 = v + v chi(0) v in the whitened
    auxiliary basis, (pq|W0|rs) = B_pq . W_aux . B_rs, v included.

    With P = B_ov D^-1 B_ov^T and D = e_a - e_i, the full RPA gives
    W_aux = (1 + 4P)^-1 (singlet chi(0) = -4 (A + B)^-1) and the Tamm-Dancoff
    one W_aux = 1 - 4P (1 + 2P)^-1 (chi(0) = -4 A^-1); the paper's TDA runs screen
    with the latter in every cycle. eps are the energies W0 is screened at.
    """
    eps = np.asarray(eps, dtype=float)
    B_ov = np.asarray(coeff_df)[:, :nocc, nocc:]
    naux = B_ov.shape[0]
    gaps = eps[nocc:][None, :] - eps[:nocc, None]
    B_ov = B_ov.reshape(naux, -1)
    P = (B_ov / gaps.reshape(-1)) @ B_ov.T
    unit = np.eye(naux)
    if tda:
        return unit - 4.0 * np.linalg.solve(unit + 2.0 * P, P)
    return np.linalg.inv(unit + 4.0 * P)


class SecondOrderKernel:
    """A(2)(w), dA(2)/dw and B(2) of one spin channel (module docstring).

    eps: orbital energies of the denominators, (nmo,), occupied first.
    coeff_df: three-index factors (naux, nmo, nmo) with sum_P B_P,pq B_P,rs = (pq|rs).
    spin: 'singlet' or 'triplet'. rule: one of SLOT_RULES. W_aux: the static W0
    in the whitened auxiliary basis (static_screening_aux) for the screened rules,
    None for 'v'. channels: a subset of CHANNELS. rings: False drops RING_TERMS
    from every table (module docstring).
    """

    def __init__(self, eps, coeff_df, nocc, spin='singlet', rule='v', W_aux=None,
                 channels=CHANNELS, rings=True):
        if spin not in SPIN_FACTOR:
            raise ValueError(f'spin {spin!r} is not one of {tuple(SPIN_FACTOR)}')
        if rule not in SLOT_RULES:
            raise ValueError(f'rule {rule!r} is not one of {SLOT_RULES}')
        if (rule == 'v') != (W_aux is None):
            raise ValueError("rule 'v' takes no W_aux and the screened rules need one")
        unknown = set(channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f'unknown channels {sorted(unknown)}; choose from {CHANNELS}')
        eps = np.asarray(eps, dtype=float)
        self.nocc, self.nvirt = nocc, eps.size - nocc
        self.npair = self.nocc * self.nvirt
        self.s = SPIN_FACTOR[spin]
        self.rule = rule
        self.channels = tuple(channels)
        self.rings = bool(rings)
        self._eps = {'o': eps[:nocc], 'v': eps[nocc:]}
        coeff_df = np.asarray(coeff_df)
        self._factors = {'v': self._split(coeff_df)}
        if rule == 'v':
            self._factors['w'] = self._factors['v']
        else:
            K = W_aux if rule == 'average' else 2.0 * W_aux - np.eye(W_aux.shape[0])
            self._factors['w'] = self._split(np.tensordot(K, coeff_df, axes=(1, 0)))

    def _split(self, factor):
        o, v = slice(0, self.nocc), slice(self.nocc, None)
        return {pat: np.ascontiguousarray(factor[:, rows, cols])
                for pat, rows, cols in (('oo', o, o), ('ov', o, v), ('vo', v, o), ('vv', v, v))}

    def _size(self, letter):
        return self.nocc if letter in _OCC else self.nvirt

    def _pattern(self, letters):
        return ''.join('o' if ch in _OCC else 'v' for ch in letters)

    def _block(self, n, which, tile=None, t0=None):
        """(n0 n1|n2 n3), the right factor from `which` ('v' or the W slot 'w'),
        over the letters of n with `tile` fixed at t0."""
        left = self._factors['v'][self._pattern(n[:2])]
        right = self._factors[which][self._pattern(n[2:])]
        ls, rs = n[:2], n[2:]
        if tile is not None and tile in ls:
            left, ls = left.take(t0, axis=1 + ls.index(tile)), ls.replace(tile, '')
        if tile is not None and tile in rs:
            right, rs = right.take(t0, axis=1 + rs.index(tile)), rs.replace(tile, '')
        return np.einsum(f'{_AUX}{ls},{_AUX}{rs}->{ls}{rs}', left, right, optimize=True)

    def _resolvent(self, letters, omega, power, tile=None, t0=None):
        """1/(D - w)^power over `letters` with `tile` fixed at t0, D the sum of the
        virtual energies minus the occupied ones."""
        shift, axes = -omega, []
        for ch in letters:
            sign = -1.0 if ch in _OCC else 1.0
            e = self._eps['o' if ch in _OCC else 'v']
            if ch == tile:
                shift += sign * e[t0]
            else:
                axes.append(sign * e)
        D = np.full([a.size for a in axes], shift)
        for k, a in enumerate(axes):
            D = D + a.reshape([-1 if m == k else 1 for m in range(len(axes))])
        return letters.replace(tile, '') if tile else letters, D ** (-power)

    def _tile_letter(self, n1, n2, rs):
        """The output index to tile over: one in the larger integral block, so no
        whole (vv|vv) is formed, and in the resolvent if possible."""
        size = lambda n: np.prod([self._size(ch) for ch in n])
        big = n1 if size(n1) >= size(n2) else n2
        candidates = [ch for ch in _OUT if ch in big] or list(_OUT)
        preferred = [ch for ch in candidates if ch in rs]
        return (preferred or candidates)[0]

    def _term(self, coef, n1, n2, slot, rs, omega, power, x=None, y=None):
        """One relabelled term: the (o, v, o, v) array, or x^T term y with
        x, y as (o, v) arrays."""
        tile = self._tile_letter(n1, n2, rs)
        which = 'w' if slot else 'v'
        fixed1 = None if tile in n1 else self._block(n1, 'v')
        fixed2 = None if tile in n2 else self._block(n2, which)
        fixedR = None if tile in rs else self._resolvent(rs, omega, power)
        s1, s2 = n1.replace(tile, ''), n2.replace(tile, '')
        out_letters = _OUT.replace(tile, '')
        if x is None:
            total = np.zeros((self.nocc, self.nvirt, self.nocc, self.nvirt))
            place = [slice(None)] * 4
        else:
            total = 0.0
        for t0 in range(self._size(tile)):
            b1 = fixed1 if fixed1 is not None else self._block(n1, 'v', tile, t0)
            b2 = fixed2 if fixed2 is not None else self._block(n2, which, tile, t0)
            sR, R = fixedR if fixedR is not None else self._resolvent(rs, omega, power, tile, t0)
            if x is None:
                place[_OUT.index(tile)] = t0
                total[tuple(place)] += coef * np.einsum(
                    f'{s1},{s2},{sR}->{out_letters}', b1, b2, R, optimize=True)
            else:
                xs = x.take(t0, axis='ia'.index(tile)) if tile in 'ia' else x
                ys = y.take(t0, axis='jb'.index(tile)) if tile in 'jb' else y
                xl, yl = 'ia'.replace(tile, ''), 'jb'.replace(tile, '')
                total += coef * np.einsum(f'{s1},{s2},{sR},{xl},{yl}->', b1, b2, R, xs, ys,
                                          optimize=True)
        return total

    def _table(self, swaps, omega, power, x=None, y=None):
        total = 0.0 if x is not None else np.zeros((self.nocc, self.nvirt, self.nocc, self.nvirt))
        for term in A_TERMS:
            code, n1, n2, slot, rs, channel = term
            coef = _coefficient(code, self.s)
            if channel not in self.channels or coef == 0.0 or (not self.rings and term in RING_TERMS):
                continue
            total = total + self._term(coef, _relabel(n1, swaps), _relabel(n2, swaps), slot, rs,
                                       omega, power, x, y)
        return total

    def A(self, omega=0.0, power=1):
        """A(2)(w) (power 1) or dA(2)/dw (power 2), (npair, npair), not symmetrized."""
        T = self._table((), omega, power)
        if self.rule != 'v':
            T = 0.5 * (T + self._table(MIRROR, omega, power))
        return T.reshape(self.npair, self.npair)

    def B(self):
        """B(2), (npair, npair), not symmetrized; independent of w."""
        if self.rule == 'v':
            T = self._table(SWAP_C, 0.0, 1)
        else:
            T = 0.5 * (self._table(SWAP_P, 0.0, 1) + self._table(SWAP_C, 0.0, 1))
        return T.reshape(self.npair, self.npair)

    def form(self, x, omega=0.0, power=1, y=None):
        """x^T A(2)(w) y (power 1) or x^T dA(2)/dw y (power 2) for flat (npair,)
        vectors, without forming the matrix; y defaults to x."""
        xm = np.asarray(x).reshape(self.nocc, self.nvirt)
        ym = xm if y is None else np.asarray(y).reshape(self.nocc, self.nvirt)
        value = self._table((), omega, power, xm, ym)
        if self.rule != 'v':
            value = 0.5 * (value + self._table(MIRROR, omega, power, xm, ym))
        return float(value)
