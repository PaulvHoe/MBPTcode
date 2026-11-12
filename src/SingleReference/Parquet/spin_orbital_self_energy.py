"""The spin-orbital parquet self-energy, as Quack's G branch computes it: the
diagonal of evPA (G_Parquet_self_energy_diag), the same diagonal at any
frequency for a root search, and the static matrix of qsPA
(G_Parquet_self_energy_SRG_dp), of Marie and Loos, J. Chem. Phys. 163, 194115
(2025), Eqs. (21), (48), (50), (51) and (53), paper-era Quack tree.

Under the parquet approximation the correlation self-energy is the second-order
one plus an eh and a pp part (Eq. (21)),

    Sigma_c = Sigma^(2) + Sigma^eh + Sigma^pp,

the last two built from the spin-orbital BSE spectra alone
(`spectra.SpinOrbitalSpectra`): the eh eigenvalues Om_n and effective
integrals rho[p, q, n] (M^eh of Eq. (38)), and the 2p (ee, Om^ee_m, nVV
roots) and 2h (hh, Om^hh_m, nOO roots) branches of the pp BSE with theirs
(M^ee, M^hh of Eq. (44)).

THE TERMS. Quack does not use the partial fractions of Eqs. (50) and (51): it
writes each term as a product of two regularized denominators,

    num * R(D1) * R(D2),        R(D) = (1 - exp(-2 eta D^2)) / D,

with the regularizer of Eq. (53) on both, eta the one-body strength (eta_1b,
s_1b of the paper; eta -> infinity is unregularized). With <pq||rs> = <pq|rs>
- <pq|sr>, i, j, k occupied, a, b, c virtual, n an eh root and m an ee or hh
root, the terms of orbital p at frequency w (Quack's column numbers):

    Sigma^(2)  1/2 <pa||ji>^2                R(w + e_a - e_i - e_j)
               1/2 <pi||ba>^2                R(w + e_i - e_a - e_b)
    eh 1       <pa||ij> rho_ia,n rho_pj,n    R(w - e_j + Om_n) R(w - e_i - e_j + e_a)
    eh 2       <pi||aj> rho_ai,n rho_pj,n    R(e_a - e_i + Om_n) R(w - e_j + Om_n)
    eh 3       <pa||ij> rho_ai,n rho_jp,n    R(e_a - e_i + Om_n) R(w - e_i - e_j + e_a)
    eh 4      -<pi||ab> rho_ia,n rho_bp,n    R(w - e_b - Om_n) R(w + e_i - e_a - e_b)
    eh 5       <pa||ib> rho_ai,n rho_bp,n    R(e_a - e_i + Om_n) R(w - e_b - Om_n)
    eh 6       <pi||ab> rho_ai,n rho_pb,n    R(e_a - e_i + Om_n) R(w + e_i - e_a - e_b)
    pp 1      -<pc|ij> hh_ij,m hh_pc,m       R(w + e_c - Om^hh_m) R(w + e_c - e_i - e_j)
    pp 2       <pc|ab> hh_ab,m hh_pc,m       R(Om^hh_m - e_a - e_b) R(w + e_c - Om^hh_m)
    pp 3      -<pc|ij> ee_ij,m ee_pc,m       R(Om^ee_m - e_i - e_j) R(w + e_c - e_i - e_j)
    pp 4       <pk|ab> ee_ab,m ee_pk,m       R(w + e_k - e_a - e_b) R(w + e_k - Om^ee_m)
    pp 5      -<pk|ij> ee_ij,m ee_pk,m       R(Om^ee_m - e_i - e_j) R(w + e_k - Om^ee_m)
    pp 6       <pk|ab> hh_ab,m hh_pk,m       R(Om^hh_m - e_a - e_b) R(w + e_k - e_a - e_b)

summed over every internal index; the pp pairs run over all ordered pairs
(i, j) and (a, b), and the plain <pc|ij> against the pair-antisymmetric rho
is Eq. (51)'s 1/2 <pc||ij>. G_Parquet_self_energy_diag evaluates them at
w = eps[p]; here w is free, every denominator that holds eps[p] in Quack's
expression taking w instead (the convention of the restricted
R_Parquet_self_energy_diag_omega), so the two agree at w = eps[p]. Every
denominator is formed in Quack's order of operations (w = eps[p] gives its
dem to the bit); the sums are not taken in its order, which moves a result in
its last bits.

TWO KINDS OF TERMS. In ten terms one denominator holds w: they are simple
regularized poles c_k R(w - x_k), whose residues c_k (every integral,
amplitude and frequency-independent R summed) `DiagonalSelfEnergySO` builds
once, as matrix products. In eh 1, eh 4, pp 1 and pp 4 both denominators hold
w, and under the product form no partial fraction separates them, so at each
frequency their integrals, regularized by the contracted side's R(w - x), are
contracted again with the BSE amplitudes (rho_ia,n, hh_ij,m, ee_ab,m). That
contraction is the cost of one more frequency.

DERIVATIVE AND Z. Quack differentiates term by term holding the regularizers
fixed: each R(D) that holds w contributes -(1 - exp(-2 eta D^2)) / D^2, the
product terms both of theirs, and Z_p = 1 / (1 - dSigma_pp/dw). It is not the
derivative of the regularized Sigma (that adds 4 eta exp(-2 eta D^2) terms);
this is kept. G_Parquet_self_energy_diag's EcGM is always 0 (no Galitskii-
Migdal energy is computed; G_evParquet's call of G_Parquet_Galitskii_Migdal
is commented out), and none is returned here.

THE STATIC MATRIX (qsPA). G_Parquet_self_energy_SRG_dp builds Sigma~_pq from
the same thirteen terms. In each, the integral is orbital p's, the second
amplitude orbital q's (rho_qj,n, rho_jq,n, rho_bq,n, rho_qb,n, ee_qc,m,
hh_qc,m, ...; the second-order terms 1/2 <pa||ji><qa||ji>, ...), and the
denominator that holds w becomes the two-orbital SRG kernel

    K(D_p, D_q) = (1 - exp(-eta (D_p^2 + D_q^2))) (D_p + D_q) / (D_p^2 + D_q^2)

of D_p = D(eps[p]) and D_q = D(eps[q]) (the qsGW form of Marie and Loos, J.
Chem. Theory Comput. 2023); K(D, D) = R(D). In eh 1, eh 4, pp 1 and pp 4 the
other denominator (the contracted side: w - e_i - e_j + e_a, w + e_i - e_a -
e_b, w + e_c - e_i - e_j, w + e_k - e_a - e_b) is R(D(eps[p])) of p alone, the
'dp' of its name. The matrix is therefore not symmetric (G_qsParquet takes
(Sigma~ + Sigma~^T)/2 after the call; it is returned here as the routine
returns it), its diagonal is the diagonal self-energy at w = eps[p], and its Z
is 1. Since exp(-eta (D_p^2 + D_q^2)) = exp(-eta D_p^2) exp(-eta D_q^2), the
exponentials are taken once per orbital and pole; K itself couples p and q,
so the matrix costs one multiply-add chain per (p, q, pole), over the pairs
q >= p (K is symmetric in p, q).

CONVENTIONS. eri[p, q, r, s] = <pq|rs> over spin orbitals (physicist, not
antisymmetrized, real); eps[p] occupied first, nO occupied and nV = norb - nO
virtual spin orbitals, no frozen core or virtuals; orbital indices are 0-based
(Quack's p is 1-based). rho[p, q, n] is indexed as Quack stores it: eh has
nS = nO nV roots, ee nVV = nV (nV - 1) / 2 and hh nOO = nO (nO - 1) / 2. The
self-energy routines read only the lengths of the pair spaces, not their
order.

NOT THE SELF-ENERGY OF QUACK'S RESTRICTED CODE. R_evParquet's
R_Parquet_self_energy_diag (and R_Parquet_QP_graph's
R_Parquet_self_energy_diag_omega) split the terms whose two denominators hold
w into partial fractions, regularizing the frequency-independent difference
of the two; the same tree's R_Parquet_self_energy_diag_dp and
R_Parquet_self_energy_diag_omega_dp keep the product form, and this module
equals them (spin orbital 2p or 2p + 1 against spatial p) to 1e-12 on
converged spectra. At finite eta the two forms differ: on converged
water/6-31G spectra (TDA, s_2b = 50) the HOMO's linearized quasiparticle
energy differs from R_evParquet's by 8.6 meV at s_1b = 50 and by 130 meV at
s_1b = 10, a core orbital's Sigma_c by up to 0.2 eV. They coincide as
eta -> infinity (at 1e8 too, except within about 1e-4 Ha of a pole, where
the regularizer still acts). The static matrix differs more: G_qsParquet
calls G_Parquet_self_energy_SRG_dp (`static_self_energy_srg_so`), while
R_qsParquet calls R_Parquet_self_energy_SRG_sp (its SRG_dp call is commented
out, R_qsParquet.f90:838-839), whose kernel is another matrix off the
diagonal at every s_1b: on the same converged water/6-31G spectra the two
symmetrized matrices differ by 1.0e-2 Ha off the diagonal at s_1b = 50 and
by 9.7e-2 Ha at s_1b = 1e8, where their diagonals agree to 5e-12, and the
whole qsPA at the paper's settings (TDA, s_1b = s_2b = 50) puts the water
HOMO 21 meV apart. Closed-shell qsPA as the paper runs it is therefore the
restricted driver's (`restricted.run_parquet`, `restricted_static_self_energy`).

COST AND MEMORY at norb = 138 spin orbitals (nO = 10, nV = 128: nS = 1280,
nVV = 8128, nOO = 45), timed on one core (Apple M4, OpenBLAS). Building the
residues and the product terms' data of all 138 orbitals is 9.2e10
multiply-adds in matrix products (eh 5 and eh 6 2.9e10 each, pp 3 1.4e10,
pp 2 1.3e10), 9 s, and holds 1.4e6 numbers per orbital, 1.6 GB for all.
Evaluating the four product terms (value and derivative) at one frequency
per orbital is 2 nV nS^2 + 2 nO nV^2 nVV = 4.2e8 + 2.7e9 multiply-adds per
orbital, 4.3e11 for all, 40 s; one orbital alone 0.34 s per frequency (pp 4
runs over rho_ee's (a, b) block one virtual a at a time, so no copy of that
1.1 GB block is made). The simple poles cost 8e5 exponentials per orbital.
diagonal_self_energy_so is both, in blocks of orbitals under
PARQUET_SO_DIAGONAL_BLOCK_ELEMENTS: 45 s, 0.6 GB beyond its inputs. The
static matrix adds the contracted sides at eps (2.2e11) and the kernel sums
over 4.4e9 (p <= q, pole) triples (4.5e5 poles), 53 s, 1.7 GB beyond its
inputs. The inputs are the ERI (2.9 GB), rho_ee (1.2 GB) and rho_eh
(0.2 GB). Quack's own loops evaluate about 3e11 terms (two exponentials each)
for the diagonal and N times as many for the static matrix.
"""
from typing import NamedTuple

import numpy as np

from src.Base.numerical_flags import flag
from src.Base.numerics.constants import (PARQUET_SO_DIAGONAL_BLOCK_ELEMENTS,
                                         PARQUET_SO_KERNEL_CACHE_ELEMENTS,
                                         PARQUET_SO_PRODUCT_BLOCK_ELEMENTS,
                                         PARQUET_SO_WORK_BLOCK_ELEMENTS)
from src.SingleReference.Parquet.events import (FLAG_SELF_ENERGY_NOT_FINITE,
                                                FLAG_STATIC_SIGMA_NOT_FINITE, ParquetNotFinite)
from src.SingleReference.Parquet.pairs import pp_pair_count
from src.SingleReference.Parquet.spectra import Channel, SpinOrbitalSpectra

#: The 2 of the one-body regularizer 1 - exp(-2 eta D^2) of every denominator
#: (Eq. (53), Quack's reg); the static kernel's 1 - exp(-eta (D_p^2 + D_q^2))
#: is the same at p = q.
SRG_EXPONENT_FACTOR = 2.0

#: Quack's column of each eh and pp term in its Sig_eh(:, 1:6) and
#: Sig_pp(:, 1:6), and of the two second-order sums (2h1p, 2p1h) of Sig_2.
TERM_COLUMNS = {'sig_2': ('2h1p', '2p1h'), 'sig_eh': (1, 2, 3, 4, 5, 6),
                'sig_pp': (1, 2, 3, 4, 5, 6)}


class _Term(NamedTuple):
    """One of Quack's terms over the orbitals P of a build.

    A simple-pole term has x and y of shape (P,) + pole shape: x carries the
    integral of orbital p, y the amplitude of orbital q (q = p on the
    diagonal), and pole names its w-dependent denominator. A product term
    (x None) has a (P, R) + L, its integrals times sign, contracted over L
    with m (L + (C,)), the BSE amplitudes, and regularized by its own
    w-dependent denominator a_pole (shape (R,) + L); y (P, R, C) is then the
    amplitude of orbital q and pole (shape (R, C)) the other denominator.
    """
    part: str
    column: int
    pole: str
    x: object
    y: np.ndarray
    a: object = None
    m: object = None
    a_pole: object = None


def _reg_inverse(d, eta):
    """(1 - exp(-2 eta d^2)) / d, Quack's reg/dem, as Quack writes it (1 -
    exp, not expm1)."""
    return (1.0 - np.exp(-SRG_EXPONENT_FACTOR * eta * d * d)) / d


def _checked(eta, eps, eri, nO, spectra):
    """The inputs as float arrays (each rho C-contiguous), refused when they
    do not fit together."""
    eta = float(eta)
    if not eta >= 0.0:
        raise ValueError(f'eta = {eta}: the one-body regularization strength is >= 0 '
                         f'(inf is the unregularized self-energy)')
    eps = np.asarray(eps, dtype=float)
    if eps.ndim != 1:
        raise ValueError(f'eps must be one-dimensional, got shape {eps.shape}')
    norb = len(eps)
    eri = np.asarray(eri, dtype=float)
    if eri.shape != (norb,) * 4:
        raise ValueError(f'eri has shape {eri.shape}, expected {(norb,) * 4}')
    nO = int(nO)
    if not 0 < nO < norb:
        raise ValueError(f'nO = {nO} must lie between 1 and norb - 1 = {norb - 1}')
    nV = norb - nO
    counts = {'eh': nO * nV, 'ee': pp_pair_count(nV, 'spin-orbital'),
              'hh': pp_pair_count(nO, 'spin-orbital')}
    chans = {}
    for name, count in counts.items():
        try:
            om, rho = getattr(spectra, name)
        except (AttributeError, TypeError, ValueError):
            raise ValueError(f'spectra.{name} is not an (om, rho) Channel of a '
                             f'SpinOrbitalSpectra') from None
        om = np.asarray(om, dtype=float)
        rho = np.ascontiguousarray(rho, dtype=float)
        if om.shape != (count,) or rho.shape != (norb, norb, count):
            raise ValueError(f'spectra.{name}: om {om.shape}, rho {rho.shape}; expected '
                             f'({count},) and {(norb, norb, count)}')
        chans[name] = Channel(om, rho)
    return eta, eps, eri, nO, SpinOrbitalSpectra(**chans)


def _pole_steps(eps, nO, spectra):
    """Each w-dependent denominator as Quack's expression forms it: a pole
    class name -> the energies (sign, array) added one after the other to w,
    ((w + s1 e1) + s2 e2) + ..., the arrays broadcasting to the class's
    pole shape (written beside it)."""
    eo, ev = eps[:nO], eps[nO:]
    om_eh, om_ee, om_hh = spectra.eh.om, spectra.ee.om, spectra.hh.om
    return {
        # [i, j, a]  w + e_a - e_i - e_j: second order 2h1p, pp 3
        'ooV+': ((1, ev[None, None, :]), (-1, eo[:, None, None]), (-1, eo[None, :, None])),
        # [i, a, j]  w - e_i - e_j + e_a: eh 3
        'ooV-': ((-1, eo[:, None, None]), (-1, eo[None, None, :]), (1, ev[None, :, None])),
        # [i, a, b]  w + e_i - e_a - e_b: second order 2p1h, eh 6, pp 6;
        #            the contracted side of pp 4 ([k, a, b])
        'oVV': ((1, eo[:, None, None]), (-1, ev[None, :, None]), (-1, ev[None, None, :])),
        # [j, n]     w - e_j + Om_n: eh 2, eh 1
        'eh-o': ((-1, eo[:, None]), (1, om_eh[None, :])),
        # [b, n]     w - e_b - Om_n: eh 5, eh 4
        'eh-v': ((-1, ev[:, None]), (-1, om_eh[None, :])),
        # [k, m]     w + e_k - Om^ee_m: pp 5, pp 4
        'ee': ((1, eo[:, None]), (-1, om_ee[None, :])),
        # [c, m]     w + e_c - Om^hh_m: pp 2, pp 1
        'hh': ((1, ev[:, None]), (-1, om_hh[None, :])),
        # [j, i, a]  w - e_i - e_j + e_a: the contracted side of eh 1
        'eh1': ((-1, eo[None, :, None]), (-1, eo[:, None, None]), (1, ev[None, None, :])),
        # [b, i, a]  w + e_i - e_a - e_b: the contracted side of eh 4
        'eh4': ((1, eo[None, :, None]), (-1, ev[None, None, :]), (-1, ev[:, None, None])),
        # [c, i, j]  w + e_c - e_i - e_j: the contracted side of pp 1
        'pp1': ((1, ev[:, None, None]), (-1, eo[None, :, None]), (-1, eo[None, None, :])),
    }


def _delta(w, steps):
    """The denominators of a pole class for the frequencies w, one per orbital:
    shape w.shape + the class's pole shape."""
    w = np.asarray(w, dtype=float)
    ndim = max(np.ndim(arr) for _, arr in steps)
    d = w.reshape(w.shape + (1,) * ndim)
    for sign, arr in steps:
        d = d + arr if sign > 0 else d - arr
    return d


def _row_blocks(n, per_row, budget=PARQUET_SO_WORK_BLOCK_ELEMENTS):
    """Slices of range(n), each of at most max(1, budget // per_row) rows."""
    size = max(1, int(budget) // max(1, int(per_row)))
    return [slice(r0, min(n, r0 + size)) for r0 in range(0, n, size)]


def _contract(x, m):
    """sum_L x[r, L] m[L, c] for x of shape (rows,) + L and m of shape L + (C,).

    A C-contiguous m is one matrix product; a strided one (the (a, b) block of
    rho_ee, kept as a view) is one product per first index of L, each m[a]
    contiguous, so no copy of it is made. The sizes are explicit: C is 0 for
    an empty pp branch (nO = 1 or nV = 1), where nothing can be inferred.
    """
    rows, ncol = x.shape[0], m.shape[-1]
    inner = int(np.prod(m.shape[:-1]))
    if m.flags.c_contiguous:
        return x.reshape(rows, inner) @ m.reshape(inner, ncol)
    out = np.zeros((rows, ncol))
    for a in range(m.shape[0]):
        out += x[:, a].reshape(rows, inner // m.shape[0]) @ m[a]
    return out


def _contracted_side(term, w, eta, poles, derivative):
    """sum_L a[p, r, L] R(D_a[p, r, L]) m[L, c] of a product term at the
    frequencies w (one per orbital): shape (P, R, C), and with derivative
    also the same with R(D)/D, stacked first, shape (2, P, R, C)."""
    a, m = term.a, term.m
    P, R = a.shape[:2]
    per = R * int(np.prod(a.shape[2:]))
    ncol = m.shape[-1]
    k = 2 if derivative else 1
    out = np.empty((k, P, R, ncol))
    for blk in _row_blocks(P, k * per, PARQUET_SO_PRODUCT_BLOCK_ELEMENTS):
        ab = a[blk]
        d = _delta(w[blk], poles[term.a_pole]).reshape(ab.shape)
        g = _reg_inverse(d, eta)
        x = np.stack((ab * g, ab * (g / d))) if derivative else (ab * g)[None]
        nb = ab.shape[0]
        out[:, blk] = _contract(x.reshape((k * nb * R,) + ab.shape[2:]),
                                m).reshape(k, nb, R, ncol)
    return out if derivative else out[0]


def _terms(eta, eps, eri, nO, spectra, ps):
    """Quack's thirteen terms (Sig_2 as its two sums) for the orbitals ps,
    one at a time, grouped by the pole class of their w-dependent (q-side)
    denominator: 'ooV+', 'ooV-', 'oVV', 'eh-o', 'eh-v', 'ee', 'hh'."""
    norb = len(eps)
    nV = norb - nO
    nS = nO * nV
    P = len(ps)
    o, v = slice(0, nO), slice(nO, norb)
    eo, ev = eps[o], eps[v]
    om_eh, rho_eh = spectra.eh
    om_ee, rho_ee = spectra.ee
    om_hh, rho_hh = spectra.hh

    def eri_block(s1, s2, s3):
        # <p s1|s2 s3> of the orbitals ps: a copy of the block alone
        return eri[:, s1, s2, s3][ps]

    # the frequency-independent factors, with their regularizers:
    # rt[ia, n] = rho_ai,n R(e_a - e_i + Om_n), yee[i, j, m] = ee_ij,m
    # R(Om^ee_m - e_i - e_j), yhh[a, b, m] = hh_ab,m R(Om^hh_m - e_a - e_b)
    d_eh = (ev[None, :, None] - eo[:, None, None]) + om_eh[None, None, :]
    rt = (rho_eh[v, o, :].transpose(1, 0, 2) * _reg_inverse(d_eh, eta)).reshape(nS, nS)
    yee = rho_ee[o, o, :] * _reg_inverse((om_ee[None, None, :] - eo[:, None, None])
                                         - eo[None, :, None], eta)
    yhh = rho_hh[v, v, :] * _reg_inverse((om_hh[None, None, :] - ev[:, None, None])
                                         - ev[None, :, None], eta)
    del d_eh

    # ---- poles w + e_a - e_i - e_j, [i, j, a] ----
    pvoo = eri_block(v, o, o)                                    # [p, a, i, j] = <pa|ij>
    # Sigma^(2) 2h1p: 1/2 <pa||ji>^2
    diff = pvoo.transpose(0, 3, 2, 1) - pvoo.transpose(0, 2, 3, 1)
    yield _Term('sig_2', 0, 'ooV+', 0.5 * diff, diff)
    del diff
    # pp 3: -<pc|ij> ee_ij,m ee_pc,m R(Om^ee_m - e_i - e_j)
    ee_pv = rho_ee[:, v, :][ps]                                  # [p, c, m]
    y = (yee.reshape(nO * nO, -1) @ ee_pv.reshape(P * nV, -1).T)
    y = y.reshape(nO, nO, P, nV).transpose(2, 0, 1, 3)
    yield _Term('sig_pp', 2, 'ooV+', -pvoo.transpose(0, 2, 3, 1), y)
    del ee_pv, y

    # ---- poles w - e_i - e_j + e_a, [i, a, j] ----
    # eh 3: <pa||ij> rho_ai,n R(e_a - e_i + Om_n) rho_jp,n
    eh_op = rho_eh[o][:, ps, :]                                  # [j, p, n]
    y = (rt @ eh_op.reshape(nO * P, nS).T).reshape(nO, nV, nO, P).transpose(3, 0, 1, 2)
    yield _Term('sig_eh', 2, 'ooV-', pvoo.transpose(0, 2, 1, 3) - pvoo.transpose(0, 3, 1, 2), y)
    del eh_op, y

    # ---- poles w + e_i - e_a - e_b, [i, a, b] ----
    povv = eri_block(o, v, v)                                    # [p, i, a, b] = <pi|ab>
    swap = povv.transpose(0, 1, 3, 2)                            # <pi|ba>
    # Sigma^(2) 2p1h: 1/2 <pi||ba>^2
    diff = swap - povv
    yield _Term('sig_2', 1, 'oVV', 0.5 * diff, diff)
    del diff
    # eh 6: <pi||ab> rho_ai,n R(e_a - e_i + Om_n) rho_pb,n
    eh_pv = rho_eh[:, v, :][ps]                                  # [p, b, n]
    y = (rt @ eh_pv.reshape(P * nV, nS).T).reshape(nO, nV, P, nV).transpose(2, 0, 1, 3)
    yield _Term('sig_eh', 5, 'oVV', povv - swap, y)
    del eh_pv, y
    # pp 6: <pk|ab> hh_ab,m R(Om^hh_m - e_a - e_b) hh_pk,m
    hh_po = rho_hh[:, o, :][ps]                                  # [p, k, m]
    y = (yhh.reshape(nV * nV, -1) @ hh_po.reshape(P * nO, -1).T)
    y = y.reshape(nV, nV, P, nO).transpose(2, 3, 0, 1)
    yield _Term('sig_pp', 5, 'oVV', povv, y)
    del hh_po, y

    # ---- poles w - e_j + Om_n, [j, n] ----
    eh_po = rho_eh[:, o, :][ps]                                  # [p, j, n] = rho_pj,n
    # eh 2: <pi||aj> rho_ai,n R(e_a - e_i + Om_n) rho_pj,n
    povo = eri_block(o, v, o)                                    # [p, i, a, j] = <pi|aj>
    poov = eri_block(o, o, v)                                    # [p, i, j, a] = <pi|ja>
    diff = povo.transpose(0, 3, 1, 2) - poov.transpose(0, 2, 1, 3)   # [p, j, i, a]
    del povo, poov
    x = (diff.reshape(P * nO, nS) @ rt).reshape(P, nO, nS)
    del diff
    yield _Term('sig_eh', 1, 'eh-o', x, eh_po)
    del x
    # eh 1: <pa||ij> rho_ia,n rho_pj,n, contracted side w - e_i - e_j + e_a
    rm = np.ascontiguousarray(rho_eh[o, v, :]).reshape(nS, nS)   # [ia, n] = rho_ia,n
    a = (pvoo.transpose(0, 3, 2, 1) - pvoo.transpose(0, 2, 3, 1)).reshape(P, nO, nS)
    yield _Term('sig_eh', 0, 'eh-o', None, eh_po, a=a, m=rm, a_pole='eh1')
    del eh_po, a

    # ---- poles w - e_b - Om_n, [b, n] ----
    eh_vp = np.ascontiguousarray(rho_eh[v][:, ps, :].transpose(1, 0, 2))   # [p, b, n] = rho_bp,n
    # eh 5: <pa||ib> rho_ai,n R(e_a - e_i + Om_n) rho_bp,n
    pvov = eri_block(v, o, v)                                    # [p, a, i, b] = <pa|ib>
    pvvo = eri_block(v, v, o)                                    # [p, a, b, i] = <pa|bi>
    diff = pvov.transpose(0, 3, 2, 1) - pvvo.transpose(0, 2, 3, 1)   # [p, b, i, a]
    del pvov, pvvo
    x = (diff.reshape(P * nV, nS) @ rt).reshape(P, nV, nS)
    del diff
    yield _Term('sig_eh', 4, 'eh-v', x, eh_vp)
    del x
    # eh 4: -<pi||ab> rho_ia,n rho_bp,n, contracted side w + e_i - e_a - e_b
    a = (povv.transpose(0, 2, 1, 3) - povv.transpose(0, 3, 1, 2)).reshape(P, nV, nS)
    yield _Term('sig_eh', 3, 'eh-v', None, eh_vp, a=a, m=rm, a_pole='eh4')
    del eh_vp, a, rm, swap

    # ---- poles w + e_k - Om^ee_m, [k, m] ----
    ee_po = rho_ee[:, o, :][ps]                                  # [p, k, m] = ee_pk,m
    # pp 5: -<pk|ij> ee_ij,m R(Om^ee_m - e_i - e_j) ee_pk,m
    pooo = eri_block(o, o, o)                                    # [p, k, i, j]
    x = -(pooo.reshape(P * nO, nO * nO) @ yee.reshape(nO * nO, -1)).reshape(P, nO, -1)
    del pooo
    yield _Term('sig_pp', 4, 'ee', x, ee_po)
    del x
    # pp 4: <pk|ab> ee_ab,m ee_pk,m, contracted side w + e_k - e_a - e_b
    m_ee = rho_ee[v, v, :]
    if m_ee.size <= PARQUET_SO_WORK_BLOCK_ELEMENTS:
        m_ee = np.ascontiguousarray(m_ee)
    yield _Term('sig_pp', 3, 'ee', None, ee_po, a=povv, m=m_ee, a_pole='oVV')
    del ee_po, m_ee, povv

    # ---- poles w + e_c - Om^hh_m, [c, m] ----
    hh_pv = rho_hh[:, v, :][ps]                                  # [p, c, m] = hh_pc,m
    # pp 2: <pc|ab> hh_ab,m R(Om^hh_m - e_a - e_b) hh_pc,m, the vvv integrals
    # of one orbital at a time
    yv = yhh.reshape(nV * nV, -1)
    x = np.empty(hh_pv.shape)
    for r, p in enumerate(ps):
        x[r] = eri[p, v, v, v].reshape(nV, nV * nV) @ yv
    yield _Term('sig_pp', 1, 'hh', x, hh_pv)
    del x
    # pp 1: -<pc|ij> hh_ij,m hh_pc,m, contracted side w + e_c - e_i - e_j
    m_hh = np.ascontiguousarray(rho_hh[o, o, :])
    yield _Term('sig_pp', 0, 'hh', None, hh_pv, a=-pvoo, m=m_hh, a_pole='pp1')


def _precomputed_per_orbital(norb, nO, nOO, nVV):
    """Numbers DiagonalSelfEnergySO holds per orbital (residues, product-term
    integrals and amplitudes), doubled for the transient factors of its build."""
    nV = norb - nO
    nS = nO * nV
    held = (3 * nO * nO * nV + 3 * nO * nV * nV + nO * nS + nV * nS + nO * nVV + nV * nOO
            + 2 * nO * nS + 2 * nV * nS + nO * nV * nV + nO * nVV + 2 * nV * nO * nO + nV * nOO)
    return 2 * held


class DiagonalSelfEnergySO:
    """Sigma_pp(w) and its derivative for a set of spin orbitals, the parts
    that do not depend on w built once (G_Parquet_self_energy_diag with every
    eps[p] of a w-dependent denominator taken at w).

    Parameters
    ----------
    eta : float
        One-body regularization strength (Quack's eta_1b, the paper's s_1b);
        1e8 is effectively unregularized, inf is allowed.
    eps : ndarray, shape (norb,)
        The spin-orbital energies of every denominator (Quack's eOld: HF in
        osPA, the previous iterate in evPA), occupied first.
    eri : ndarray, shape (norb,)*4
        <pq|rs> over spin orbitals.
    nO : int
        Occupied spin orbitals.
    spectra : spectra.SpinOrbitalSpectra
        The eh, ee and hh channels (om, rho) of the two-body problem.
    orbitals : sequence of int, optional
        The spin orbitals p (0-based) whose Sigma_pp is wanted; default all.
        Every per-orbital array runs over these, in this order.

    The simple poles keep their residues, (len(orbitals),) + pole shape for
    each of the ten terms; the four product terms keep their integrals and
    the orbital's amplitudes. One frequency then costs the exponentials of
    the poles and, per orbital, 2 nV nS^2 + 2 nO nV^2 nVV multiply-adds for
    the product terms (see the module notes).
    """

    def __init__(self, eta, eps, eri, nO, spectra, orbitals=None):
        eta, eps, eri, nO, spectra = _checked(eta, eps, eri, nO, spectra)
        norb = len(eps)
        ps = (np.arange(norb) if orbitals is None
              else np.atleast_1d(np.asarray(orbitals, dtype=int)))
        if ps.ndim != 1 or (ps.size and (ps.min() < 0 or ps.max() >= norb)):
            raise ValueError(f'orbitals must be spin-orbital indices in [0, {norb})')
        self.eta = eta
        self.eps = eps
        self.nO = nO
        self.orbitals = ps
        self._rows = {}
        for r, p in enumerate(ps):
            self._rows.setdefault(int(p), r)
        self._poles = _pole_steps(eps, nO, spectra)
        self._simple, self._product = [], []
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            for term in _terms(eta, eps, eri, nO, spectra, ps):
                if term.x is None:
                    self._product.append(term)
                else:
                    self._simple.append((term.part, term.column, term.pole,
                                         term.x * term.y))

    def _parts(self, w):
        """Quack's decomposition at the frequencies w (one per orbital):
        part name -> (sig, dsig), each (len(w), ncol)."""
        w = np.asarray(w, dtype=float)
        if w.shape != self.orbitals.shape:
            raise ValueError(f'w has shape {w.shape}: one frequency per orbital, '
                             f'shape {self.orbitals.shape}')
        P = len(w)
        parts = {name: (np.zeros((P, len(cols))), np.zeros((P, len(cols))))
                 for name, cols in TERM_COLUMNS.items()}
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            for name, column, pole, res in self._simple:
                sig, dsig = parts[name]
                # npole is 0 for a pole class over an empty pp branch (nO = 1: no
                # 2h root, nV = 1: no 2p root); its term is then an exact 0
                npole = int(np.prod(res.shape[1:]))
                for blk in _row_blocks(P, 3 * npole):
                    rb = res[blk].reshape(blk.stop - blk.start, npole)
                    d = _delta(w[blk], self._poles[pole]).reshape(rb.shape)
                    g = _reg_inverse(d, self.eta)
                    sig[blk, column] = np.einsum('pk,pk->p', rb, g)
                    g /= d
                    dsig[blk, column] = -np.einsum('pk,pk->p', rb, g)
            for term in self._product:
                sig, dsig = parts[term.part]
                R, C = term.y.shape[1:]
                per = 2 * R * int(np.prod(term.a.shape[2:])) + 4 * R * C
                for blk in _row_blocks(P, per, PARQUET_SO_PRODUCT_BLOCK_ELEMENTS):
                    t = _contracted_side(term._replace(a=term.a[blk]), w[blk], self.eta,
                                         self._poles, derivative=True)
                    t *= term.y[blk]
                    d = _delta(w[blk], self._poles[term.pole])
                    g = _reg_inverse(d, self.eta)
                    sig[blk, term.column] = np.einsum('prc,prc->p', t[0], g)
                    dsig[blk, term.column] = -np.einsum('prc,prc->p', t[1], g)
                    g /= d
                    dsig[blk, term.column] -= np.einsum('prc,prc->p', t[0], g)
        return parts

    def _evaluate(self, w):
        """(sig, dsig, parts) at the frequencies w, summed as Quack sums SigC."""
        parts = self._parts(w)
        sig2, dsig2 = parts['sig_2']
        sig, dsig = sig2[:, 0] + sig2[:, 1], dsig2[:, 0] + dsig2[:, 1]
        for name in ('sig_eh', 'sig_pp'):
            s, ds = parts[name]
            for c in range(s.shape[1]):
                sig = sig + s[:, c]
                dsig = dsig + ds[:, c]
        return sig, dsig, parts

    def at_each(self, w, parts=None):
        """(sig, dsig) over `orbitals`, orbital orbitals[r] at frequency w[r]
        (Hartree): Sigma_pp(w) and dSigma_pp/dw with the regularizers held
        constant. dsig is the derivative, not 1/(1 - dsig).

        parts : dict, optional
            Filled with Quack's decomposition (dump_GParquet_self_energy):
            'sig_2' (len(w), 2), its 2h1p and 2p1h sums (Sig_2 is their sum);
            'sig_eh' and 'sig_pp' (len(w), 6), Quack's Sig_eh(:, 1:6) and
            Sig_pp(:, 1:6); 'dsig_2', 'dsig_eh', 'dsig_pp' the same of the
            derivative (Quack's Z_2, Z_eh, Z_pp before Z = 1/(1 - Z)).
        """
        sig, dsig, terms = self._evaluate(w)
        _flag_not_finite(np.concatenate((sig, dsig)), FLAG_SELF_ENERGY_NOT_FINITE,
                         'the spin-orbital parquet self-energy at w')
        if parts is not None:
            for name, (s, ds) in terms.items():
                parts[name], parts['d' + name] = s, ds
        return sig, dsig

    def at(self, p, w):
        """(Sigma_pp(w), dSigma_pp/dw) of spin orbital p (0-based) at frequency
        w (Hartree); w may be an array, then both are arrays of its shape,
        all its frequencies taken together (one pass over the BSE amplitudes
        per block of PARQUET_SO_PRODUCT_BLOCK_ELEMENTS)."""
        try:
            r = self._rows[int(p)]
        except KeyError:
            raise ValueError(f'orbital {p} is not among the orbitals this '
                             f'self-energy was built for') from None
        ws = np.asarray(w, dtype=float)
        flat = ws.reshape(-1)
        sig, dsig, _ = self._row_view(r, len(flat))._evaluate(flat)
        _flag_not_finite(np.concatenate((sig, dsig)), FLAG_SELF_ENERGY_NOT_FINITE,
                         f'the spin-orbital parquet self-energy of orbital {p} at w')
        if ws.ndim == 0:
            return float(sig[0]), float(dsig[0])
        return sig.reshape(ws.shape), dsig.reshape(ws.shape)

    def diagonal(self, parts=None):
        """(sigc, z) over `orbitals`, each at w = eps[p]: G_Parquet_self_energy_diag's
        SigC and Z = 1/(1 - dSigma_pp/dw). parts as for at_each."""
        sig, dsig = self.at_each(self.eps[self.orbitals], parts=parts)
        with np.errstate(divide='ignore', invalid='ignore'):
            z = 1.0 / (1.0 - dsig)
        return sig, z

    def _row_view(self, r, count):
        """This self-energy's row r repeated count times, one row per
        frequency (broadcast views, no copy of the data)."""
        def rows(arr):
            return np.broadcast_to(arr[r:r + 1], (count,) + arr.shape[1:])

        one = object.__new__(DiagonalSelfEnergySO)
        one.eta, one.eps, one.nO = self.eta, self.eps, self.nO
        one.orbitals = np.full(count, self.orbitals[r])
        one._rows = {int(self.orbitals[r]): 0}
        one._poles = self._poles
        one._simple = [(name, col, pole, rows(res)) for name, col, pole, res in self._simple]
        one._product = [t._replace(a=rows(t.a), y=rows(t.y)) for t in self._product]
        return one


def _flag_not_finite(values, name, where):
    """Warn `events.ParquetNotFinite` and raise the named flag `name`
    ('parquet_self_energy_not_finite' for the diagonal,
    'parquet_static_sigma_not_finite' for the static matrix) if values hold a
    NaN or an inf: a denominator vanished exactly (Quack's reg/dem is 0/0
    there too) or a spectrum is not finite."""
    bad = ~np.isfinite(values)
    if bad.any():
        flag(name, f'{where}: {int(bad.sum())} of {bad.size} values are not finite (a '
                   f'denominator of Quack\'s formula is exactly zero, or a spectrum is not '
                   f'finite); returned as computed',
             ParquetNotFinite, stacklevel=3)


def diagonal_self_energy_so(eta, eps, eri, nO, spectra, parts=None):
    """(sigc, z): the diagonal spin-orbital parquet self-energy of every spin
    orbital and its renormalization factor, Quack's G_Parquet_self_energy_diag
    (the routine G_evParquet calls; Eqs. (21), (48), (50), (51), regularized
    as Eq. (53)).

    Parameters
    ----------
    eta : float
        One-body regularization strength (eta_1b, s_1b).
    eps : ndarray, shape (norb,)
        The energies of every denominator (G_evParquet's eOld); orbital p is
        evaluated at w = eps[p].
    eri : ndarray, shape (norb,)*4
        <pq|rs> over spin orbitals.
    nO : int
        Occupied spin orbitals.
    spectra : spectra.SpinOrbitalSpectra
    parts : dict, optional
        Filled with Quack's decomposition over every orbital, as
        DiagonalSelfEnergySO.at_each fills it.

    Returns
    -------
    sigc : ndarray, shape (norb,)
        Sigma_pp(eps[p]).
    z : ndarray, shape (norb,)
        1 / (1 - dSigma_pp/dw) at w = eps[p], the regularizers held constant.

    Quack's third output, EcGM, is always 0 and is not returned. The
    orbitals run in blocks whose precomputed data stay under
    PARQUET_SO_DIAGONAL_BLOCK_ELEMENTS.
    """
    eta, eps, eri, nO, spectra = _checked(eta, eps, eri, nO, spectra)
    norb = len(eps)
    per = _precomputed_per_orbital(norb, nO, len(spectra.hh.om), len(spectra.ee.om))
    sig, dsig = np.empty(norb), np.empty(norb)
    pieces = {name: [] for name in TERM_COLUMNS}
    for blk in _row_blocks(norb, per, PARQUET_SO_DIAGONAL_BLOCK_ELEMENTS):
        ps = np.arange(norb)[blk]
        se = DiagonalSelfEnergySO(eta, eps, eri, nO, spectra, orbitals=ps)
        sig[blk], dsig[blk], terms = se._evaluate(eps[ps])
        for name in TERM_COLUMNS:
            pieces[name].append(terms[name])
    _flag_not_finite(np.concatenate((sig, dsig)), FLAG_SELF_ENERGY_NOT_FINITE,
                     'the diagonal spin-orbital parquet self-energy')
    if parts is not None:
        for name, chunks in pieces.items():
            parts[name] = np.concatenate([s for s, _ in chunks])
            parts['d' + name] = np.concatenate([ds for _, ds in chunks])
    with np.errstate(divide='ignore', invalid='ignore'):
        z = 1.0 / (1.0 - dsig)
    return sig, z


def self_energy_at_so(p, w, eta, eps, eri, nO, spectra):
    """(sig, dsig): Sigma_pp(w) and dSigma_pp/dw of spin orbital p (0-based) at
    frequency w, every other denominator on eps.

    The frequency-dependent counterpart of G_Parquet_self_energy_diag that the
    G branch lacks (G_evParquet stops unless the quasiparticle equation is
    linearized), in the convention of R_Parquet_self_energy_diag_omega: the
    denominators that hold eps[p] take w, and the derivative holds the
    regularizers constant (dsig, not 1/(1 - dsig)). At w = eps[p] it is
    diagonal_self_energy_so's sigc[p] and 1 - 1/z[p].

    w may be an array (both outputs then have its shape). This builds the
    residues of orbital p alone; for a root search over many orbitals and
    frequencies, build a DiagonalSelfEnergySO once and call `at_each` or `at`.
    """
    return DiagonalSelfEnergySO(eta, eps, eri, nO, spectra, orbitals=[p]).at(p, w)


def _kernel_sum(sig, d, pairs, eta):
    """sig[p, q] += sum_k K(d[p, k], d[q, k]) sum_t x_t[p, k] y_t[q, k] over
    every p, q, K the static SRG kernel, taken once per pair q >= p.

    The poles run in chunks under PARQUET_SO_KERNEL_CACHE_ELEMENTS, each
    chunk's columns of d and of every factor copied out once, so they stay in
    cache while
    every orbital p takes its rows q >= p; exp(-eta (d_p^2 + d_q^2)) is the
    product of the two orbitals' exp(-eta d^2).
    """
    norb = d.shape[0]
    d = d.reshape(norb, -1)
    xs = [x.reshape(norb, -1) for x, _ in pairs]
    ys = [y.reshape(norb, -1) for _, y in pairs]
    npole = d.shape[1]
    chunk = max(1, PARQUET_SO_KERNEL_CACHE_ELEMENTS // ((6 + 2 * len(pairs)) * norb))
    for k0 in range(0, npole, chunk):
        ks = slice(k0, min(npole, k0 + chunk))
        dk = np.ascontiguousarray(d[:, ks])
        ek = np.exp(-eta * dk * dk)
        d2k = dk * dk
        xk = [np.ascontiguousarray(x[:, ks]) for x in xs]
        yk = [np.ascontiguousarray(y[:, ks]) for y in ys]
        f, s, t = np.empty(dk.shape), np.empty(dk.shape), np.empty(dk.shape)
        for p in range(norb):
            n = norb - p
            fv, sv, tv = f[:n], s[:n], t[:n]
            # K(d_p, d_q) = (1 - e_p e_q) (d_p + d_q) / (d_p^2 + d_q^2), q >= p
            np.multiply(ek[p], ek[p:], out=fv)
            np.subtract(1.0, fv, out=fv)
            np.add(dk[p], dk[p:], out=sv)
            fv *= sv
            np.add(d2k[p], d2k[p:], out=sv)
            fv /= sv
            for x, y in zip(xk, yk):
                np.multiply(fv, y[p:], out=tv)
                sig[p, p:] += tv @ x[p]
                np.multiply(fv[1:], x[p + 1:], out=tv[1:])
                sig[p + 1:, p] += tv[1:] @ y[p]


def static_self_energy_srg_so(eta, eps, eri, nO, spectra):
    """(sigc, z): the static spin-orbital parquet self-energy matrix of qsPA,
    Quack's G_Parquet_self_energy_SRG_dp (the routine G_qsParquet calls; not
    R_qsParquet's R_Parquet_self_energy_SRG_sp, which differs from it off the
    diagonal at every s_1b: module notes).

    Sigma~_pq is the thirteen terms with orbital p's integral, orbital q's
    second amplitude and the w-dependent denominator D replaced by the SRG
    kernel K(D(eps[p]), D(eps[q])); in the product terms eh 1, eh 4, pp 1 and
    pp 4 the contracted side keeps R(D(eps[p])) of p alone. See the module
    notes.

    Parameters
    ----------
    eta : float
        One-body regularization strength (eta_1b, s_1b).
    eps : ndarray, shape (norb,)
        The energies of every denominator (G_qsParquet's eQP).
    eri : ndarray, shape (norb,)*4
        <pq|rs> over spin orbitals, in the basis of eps (G_qsParquet's ERI_MO).
    nO : int
        Occupied spin orbitals.
    spectra : spectra.SpinOrbitalSpectra

    Returns
    -------
    sigc : ndarray, shape (norb, norb)
        Sigma~_pq as the routine returns it: NOT symmetric (G_qsParquet uses
        (sigc + sigc.T) / 2); its diagonal is diagonal_self_energy_so's sigc.
    z : ndarray, shape (norb,)
        Ones, as the routine sets them.
    """
    eta, eps, eri, nO, spectra = _checked(eta, eps, eri, nO, spectra)
    norb = len(eps)
    ps = np.arange(norb)
    poles = _pole_steps(eps, nO, spectra)
    sig = np.zeros((norb, norb))
    key, pairs = None, []
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        for term in _terms(eta, eps, eri, nO, spectra, ps):
            if term.pole != key:
                if pairs:
                    _kernel_sum(sig, _delta(eps, poles[key]), pairs, eta)
                key, pairs = term.pole, []
            x = (term.x if term.x is not None
                 else _contracted_side(term, eps, eta, poles, derivative=False))
            for k, (xk, yk) in enumerate(pairs):
                if yk is term.y:
                    pairs[k] = (xk + x, yk)
                    break
            else:
                pairs.append((x, term.y))
            del term, x
        if pairs:
            _kernel_sum(sig, _delta(eps, poles[key]), pairs, eta)
    _flag_not_finite(sig, FLAG_STATIC_SIGMA_NOT_FINITE,
                     'the static spin-orbital parquet self-energy')
    return sig, np.ones(norb)
