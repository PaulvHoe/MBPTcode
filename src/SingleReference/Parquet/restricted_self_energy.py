"""The diagonal parquet self-energy of osPA and evPA, and its quasiparticle
equation, as Quack's closed-shell code poses them (Marie and Loos,
J. Chem. Phys. 163, 194115 (2025), Eqs. (21), (48), (50), (51) and (53);
Quack's R_Parquet_self_energy_diag and R_Parquet_self_energy_diag_omega of
the paper-era tree, and the equation its R_Parquet_QP_graph solves).

Under the parquet approximation the correlation self-energy is the second-order
one plus an eh and a pp part (Eq. (21)),

    Sigma_c = Sigma^(2) + Sigma^eh + Sigma^pp,

and the eh and pp parts are built from the BSE spectra alone: the eigenvalues
Omega and the generalized effective integrals rho[p, q, n] (M^eh, M^ee, M^hh
of Eqs. (38) and (44)) of the singlet and triplet eh BSE and of the 2p (ee)
and 2h (hh) branches of the singlet and triplet pp BSE
(`spectra.RestrictedSpectra`). The second-order part is the closed-shell form
of Eq. (48),

    Sigma^(2)_pp(w) = sum_ija [2 <pa|ji><ji|pa> - <pa|ji><ji|ap>] R(w + e_a - e_i - e_j)
                    + sum_iab [2 <pi|ba><ba|pi> - <pi|ba><ba|ip>] R(w + e_i - e_a - e_b)

and each channel of Eqs. (50) and (51) gives eight terms, each a product

    num * R(Delta) * R(w - x),        R(D) = (1 - exp(-2 eta D^2)) / D,

of integrals and amplitudes (num), a frequency-independent denominator Delta
(e_a - e_i -+ Omega^eh_n, Omega^ee_m - e_i - e_j, e_i + e_j - Omega^hh_m, ...)
and a frequency-dependent one w - x. The closed-shell spin weights are Quack's:
0.5 <pa|ij> - <pa|ji> (eh singlet), 1.5 <pa|ij> (eh triplet), 0.5 and 1.5
<pa|ij> (pp singlet and triplet). Every denominator, the frequency-independent
ones included, carries the energy-dependent regularizer of Eq. (53) with the
one-body strength eta (s_1b of the paper); eta -> infinity is the unregularized
self-energy. The form 1 - exp(...) is kept as Quack writes it (not expm1).

POLES. The frequency-dependent denominators are w - x with x of six kinds,

    2h1p          e_i + e_j - e_a      (second order, eh and pp terms)
    2p1h          e_a + e_b - e_i      (second order, eh and pp terms)
    eh hole       e_j - Omega^eh_n     eh particle   e_b + Omega^eh_n
    ee            Omega^ee_m - e_k     hh            Omega^hh_m - e_c

(one eh pair of families per eh spin, one ee and one hh family per pp spin),
so a diagonal element is a sum over simple regularized poles,

    Sigma_pp(w) = sum_k c_pk R(w - x_k).

The positions x_k do not depend on p and the residues c_pk (every integral,
amplitude and R(Delta) summed over the internal indices) do not depend on w.
`DiagonalSelfEnergy` computes the residues of the orbitals asked for once, as
matrix products, after which one more frequency costs one exp per pole: this
is what makes the root search of every orbital at every evPA iteration
affordable.

ROUNDING. Each frequency-dependent denominator is formed from its energies in
the order Quack's expression adds them (w + e_a - e_i - e_j in most 2h1p terms,
w - e_i - e_j + e_a in the second eh term, hence the separate '2h1p_alt'
family), so every R(w - x) is Quack's to the last bit, also beside a pole,
where w - x carries a large condition number (on it, w - x exactly zero,
Quack's is 0/0: ZERO DENOMINATORS below). Only the terms a pole collects
are summed in another order than Quack sums them, which moves Sigma and its
derivative in their last bits (1e-15 relative to the sum of the terms'
magnitudes); Z = 1/(1 - dSigma/dw) magnifies that by Z^2 where dSigma/dw is
close to 1.

DERIVATIVE AND Z. Quack differentiates term by term holding the regularizer
1 - exp(-2 eta (w - x)^2) constant,

    dSigma_pp/dw = - sum_k c_pk R(w - x_k) / (w - x_k),   Z_p = 1 / (1 - dSigma_pp/dw),

which is not the derivative of the regularized Sigma (that one would add
c_pk 4 eta exp(-2 eta (w - x)^2)); this is kept. The Galitskii-Migdal
correlation energy of R_Parquet_self_energy_diag is zero (Quack's EcGM = 0:
none is computed), and none is returned here.

ZERO DENOMINATORS. Where a denominator D is exactly zero (w on a pole, or a
frequency-independent Delta that vanishes), Quack's R(D) and its slope
R(D) / D are 0/0 = NaN, and so is the whole Sigma_pp. Here both are 0 at
every finite eta (eta = inf, the bare 1/D, keeps its pole): R(0) = 0 is R's
limit, and the slope 0 is what Quack's own formula gives a few ulps off
zero, where 1 - exp(-2 eta D^2) rounds to 0 (|D| below 5e-13 at eta = 1e8),
not the slope's limit 2 eta. A zero mode's pole thus adds the same,
nothing, to Sigma and to its derivative whether rounding puts the mode at 0
or a few ulps off. This is the module's one departure from Quack's
arithmetic, every other denominator being evaluated as Quack writes it.

THE QUASIPARTICLE EQUATION. R_evParquet evaluates the self-energy with every
denominator on the energies of the previous one-body iteration eOld (the HF
energies at the first), each orbital p at w = eOld_p, and takes the linearized
energies eQPlin = eHF + Z Sigma_c (anchored on eHF, not on eOld, also after the
first iteration: `linearized_qp_energies`). R_Parquet_QP_graph then searches
w = eHF_p + Sigma_pp(w) from eQPlin_p, with the other denominators still on
eOld; that search is not ported (the tests carry it for their comparisons
with Quack: tests/parquet_quack_newton.py).

The drivers' `qp_solver` says how the equation is solved
(`options.QP_SOLVERS`, or a function: `options.checked_qp_solver`):
'linearized' (eQP = eQPlin, Quack's lin_parquet and G_evParquet's only
branch) or 'find_qp_root', the drivers' default, the house's one root finder
on that equation with Quack's slope (`DiagonalSelfEnergy.solve_qp_scan`): at
the first one-body iteration the root of largest valid weight in its scan
windows about eHF_p (narrowest first, the first whose largest weight exceeds
QP_WINDOW_Z_MIN), from the second on the root continuing the previous
iteration's eQP_p where that root's Z lay in (0, 1] (`options.roots_to_follow`;
otherwise the orbital is searched as at the first iteration). It takes 150 to
14 000 evaluations of Sigma_pp per orbital without a root to follow, on water
and ammonia 6-31G and water/aug-cc-pVDZ, and about as many with one, whose
scan window it searches whole for the branch report (150 to 3 200 in the
evPA of water and ammonia 6-31G at s_1b = 50 and s_2b = 1). Every route
flags an energy kept with Z outside (0, 1] ('z_outside'), the linearized
one and a qp_solver function's included.

R AND G. This is Quack's closed-shell self-energy. It and the spin-orbital
G_Parquet_self_energy_diag (`spin_orbital_self_energy`) write each term as a
product of regularized denominators, mostly R(Delta) R(w - x), a
frequency-independent one times a frequency-dependent one. They part where
both bare denominators hold w (the spin-orbital eh 1, eh 4, pp 1 and pp 4
terms): this one splits such a term into partial fractions first, two terms
R(Delta) R(w - x), while G regularizes it as it stands, R(w - x1) R(w - x2)
(as does the unused R_Parquet_self_energy_diag_dp); so the two differ at
finite eta and agree as eta -> infinity. On the same
spectra (FLEX RPA, s_1b = 50, water and ammonia 6-31G) the quasiparticle
energies of the two differ by 2.4 and 1.6 meV at the HOMO, 23 and 114 meV at
the 2a1 level and 0.18 and 0.53 eV at most, and by 2e-11 eV at s_1b = 1e8:
the paper's closed-shell FLEX, osPA and evPA come from this module.

CONVENTIONS. eri[p, q, r, s] = <pq|rs> (MO, physicist, not antisymmetrized,
real orbitals); eps[p] occupied first, nO occupied and nV = norb - nO virtual
orbitals, no frozen core or virtuals; orbital indices are 0-based (Quack's p is
1-based). rho[p, q, n] is indexed as Quack stores it; eh_* channels have
nS = nO nV states, ee_* nVV and hh_* nOO (singlet pairs p <= q, triplet
p < q), as `pairs` lays them out, though only the lengths matter here.

COST AND MEMORY at norb = 138 (CH4/aug-cc-pVTZ, nO = 5, nS = 665,
nVV = 8911 / 8778 and nOO = 15 / 10 for the singlet / triplet). There are
370 405 poles per orbital; the residues of all 138 orbitals take 409 MB.
Building them (`DiagonalSelfEnergy`, hence `diagonal_self_energy`) is
1.0e12 flops, all in matrix products: 4.3e11 for each pp
spin (the 3p2h terms sum_ab <pi|ab> rho^ee_ab,m R and its transpose, streamed
over blocks of the ee states of PARQUET_SELF_ENERGY_BLOCK_BYTES), 6.7e10 for each eh
spin, and 3.1e8 exponentials (the ee regularizers, once each). Beside its
inputs (ERI 2.9 GB, the ee rho 1.4 GB per spin) the build peaks at 1.0 GB,
the residues included; on an Apple M4 shared with other jobs it took 65 s on
one thread and 30 s on eight. One more frequency for one orbital (`at`) is
370 405 exps and a few elementwise passes, 3.5 ms on any number of threads;
the diagonal is 138 of them (0.45 s), and each evaluation of an orbital's
quasiparticle equation one.
`self_energy_at` builds one orbital's residues, which still needs every ee
regularizer: 5 s. Quack's own loops run about 3.6e9 terms, each with two
exps, per orbital and per self-energy evaluation, 5e11 for the diagonal and as
many again for every step of Quack's quasiparticle search on every orbital.
"""
import math

import numpy as np

from src.Base.numerical_flags import flag, flags_into_out
from src.Base.numerics.constants import PARQUET_SELF_ENERGY_BLOCK_BYTES
from src.Solvers.qp_equation import solve_qp_equation_scan
from src.SingleReference.Parquet.events import FLAG_SELF_ENERGY_NOT_FINITE, ParquetNotFinite
from src.SingleReference.Parquet.options import flag_branch
from src.SingleReference.Parquet.pairs import EH_SPIN_WEIGHTS, PP_SPIN_WEIGHTS

#: The parts of the self-energy Quack's dump_RParquet_self_energy prints:
#: second order direct and exchange (Sig2d, Sig2x), singlet and triplet eh
#: (Sig1eh, Sig3eh), singlet and triplet pp (Sig1pp, Sig3pp).
TERMS = ('2d', '2x', 'eh_singlet', 'eh_triplet', 'pp_singlet', 'pp_triplet')

#: The pole families, in the order the residue matrix holds them. '2h1p' and
#: '2h1p_alt' are the same poles e_i + e_j - e_a, whose denominator Quack
#: writes w + e_a - e_i - e_j in most terms and w - e_i - e_j + e_a in the
#: second eh term; the three-energy families come first.
POLE_GROUPS = ('2h1p', '2h1p_alt', '2p1h',
               'eh_singlet_hole', 'eh_singlet_particle',
               'eh_triplet_hole', 'eh_triplet_particle',
               'ee_singlet', 'hh_singlet', 'ee_triplet', 'hh_triplet')

#: Closed-shell spin sums of the direct and exchange second-order terms
#: (2 <pa|ji><ji|pa> - <pa|ji><ji|ap>, Eq. (48) summed over spin).
SECOND_ORDER_WEIGHTS = {'2d': 2.0, '2x': -1.0}

#: Quack's IEEE arithmetic, silently: a vanishing denominator's 0/0 (then
#: replaced by 0, ZERO DENOMINATORS), inf or NaN from inputs that are not
#: finite, and an exp(-2 eta D^2) far below the smallest double, 0.
_IEEE_SILENT = dict(divide='ignore', invalid='ignore', over='ignore', under='ignore')


def _srg_over(d, eta):
    """(1 - exp(-2 eta d^2)) / d elementwise (a new array): reg/dem of every
    Quack denominator, in Quack's order of operations; an exactly zero d
    gives the limit 0 (ZERO DENOMINATORS)."""
    t = np.multiply(-2.0 * eta, d)
    t *= d
    np.exp(t, out=t)
    np.subtract(1.0, t, out=t)
    t /= d
    if eta < np.inf:
        zero = d == 0.0
        if zero.any():
            t[zero] = 0.0                   # Quack's 0/0: the limit, 2 eta d -> 0
    return t


def _channel_arrays(channel, norb, name):
    """(om, rho) of one spectra.Channel, checked against norb."""
    try:
        om, rho = channel
    except (TypeError, ValueError):
        raise ValueError(f'spectra.{name} is not an (om, rho) Channel') from None
    om = np.asarray(om, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if om.ndim != 1 or rho.shape != (norb, norb, len(om)):
        raise ValueError(f'spectra.{name}: rho has shape {rho.shape}, expected '
                         f'(norb, norb, len(om)) = ({norb}, {norb}, {len(om)})')
    return om, rho


def _second_order(eri, nO, P, terms, hhp, pph):
    """Add the second-order residues of `terms` ('2d', '2x') of orbitals P to
    hhp[p, i, j, a] (pole e_i + e_j - e_a) and pph[p, i, a, b]
    (pole e_a + e_b - e_i): R_Parquet_self_energy_diag's 2h1p and 2p1h sums."""
    o, v = slice(0, nO), slice(nO, eri.shape[0])
    A = eri[P, v, o, o].transpose(0, 3, 2, 1)           # <pa|ji> at [p,i,j,a]
    if '2d' in terms:
        hhp += (SECOND_ORDER_WEIGHTS['2d'] * A) * eri[o, o, P, v].transpose(2, 1, 0, 3)
    if '2x' in terms:
        hhp += (SECOND_ORDER_WEIGHTS['2x'] * A) * eri[o, o, v, P].transpose(3, 1, 0, 2)
    A = eri[P, o, v, v].transpose(0, 1, 3, 2)           # <pi|ba> at [p,i,a,b]
    if '2d' in terms:
        pph += (SECOND_ORDER_WEIGHTS['2d'] * A) * eri[v, v, P, o].transpose(2, 3, 1, 0)
    if '2x' in terms:
        pph += (SECOND_ORDER_WEIGHTS['2x'] * A) * eri[v, v, o, P].transpose(3, 2, 1, 0)


def _spin_combination(wd, direct, wx, exchange, rows):
    """wd * direct + wx * exchange (exchange left out when wx is 0), as a
    C-ordered matrix of `rows` rows; wx is signed (pairs.EH_SPIN_WEIGHTS)."""
    k = wd * direct
    if wx:
        k = k + wx * exchange
    return np.ascontiguousarray(k).reshape(rows, -1)


def _eh_channel(eps, eri, nO, P, om, rho, weights, eta,
                hole, particle, hhp, hhp_alt, pph):
    """Add one eh channel's residues (R_Parquet_self_energy_diag, singlet or
    triplet eh part) of orbitals P to hole[p, j, n] (pole e_j - Om_n),
    particle[p, b, n] (pole e_b + Om_n), hhp[p, i, j, a] and hhp_alt[p, i, j, a]
    (both pole e_i + e_j - e_a) and pph[p, i, a, b] (pole e_a + e_b - e_i).

    With U_ia,n = rho_ia,n R(e_a - e_i - Om_n), W_ia,n = rho_ai,n
    R(e_a - e_i + Om_n) and the spin-weighted integrals (wx signed,
    pairs.EH_SPIN_WEIGHTS) K1_pj,ia = wd <pa|ij> + wx <pa|ji>,
    K3_pj,ia = wd <pi|aj> + wx <pi|ja>, K5_pb,ia = wd <pi|ab> + wx <pi|ba>,
    K7_pb,ia = wd <pa|ib> + wx <pa|bi>:

        hole     rho_pj,n [K1 U + K3 W]_pj,n                 (Quack's terms 1, 3)
        particle rho_bp,n [K5 U + K7 W]_pb,n                 (terms 5, 7)
        2h1p      K1_pj,ia sum_n W_ia,n rho_jp,n             (term 4)
        2h1p_alt -K1_pj,ia sum_n U_ia,n rho_pj,n             (term 2)
        2p1h     K5_pb,ia sum_n [W_ia,n rho_pb,n - U_ia,n rho_bp,n]   (8, 6)
    """
    wd, wx = weights
    norb = len(eps)
    nV, nP, nS = norb - nO, len(P), len(om)
    o, v = slice(0, nO), slice(nO, norb)
    gap = (eps[None, v] - eps[o, None])[:, :, None]     # e_a - e_i at [i,a,1]
    U = np.ascontiguousarray(rho[o, v, :] * _srg_over(gap - om, eta))
    U = U.reshape(nO * nV, nS)
    W = np.ascontiguousarray(rho[v, o, :].transpose(1, 0, 2) * _srg_over(gap + om, eta))
    W = W.reshape(nO * nV, nS)

    E = eri[P, v, o, o]                                 # <pa|xy> at [p,a,x,y]
    K1 = _spin_combination(wd, E.transpose(0, 3, 2, 1), wx, E.transpose(0, 2, 3, 1),
                           nP * nO)
    K3 = _spin_combination(wd, eri[P, o, v, o].transpose(0, 3, 1, 2),
                           wx, eri[P, o, o, v].transpose(0, 2, 1, 3), nP * nO)
    rho_po = rho[P, o, :]                               # rho_pj,n
    hole += (K1 @ U + K3 @ W).reshape(nP, nO, nS) * rho_po
    del K3
    K1 = K1.reshape(nP, nO, nO, nV).transpose(0, 2, 1, 3)       # [p,i,j,a]
    G = W @ rho[o, P, :].transpose(1, 0, 2).reshape(nP * nO, nS).T  # [ia, pj]
    hhp += K1 * G.reshape(nO, nV, nP, nO).transpose(2, 0, 3, 1)
    G = U @ rho_po.reshape(nP * nO, nS).T                         # [ia, pj]
    hhp_alt -= K1 * G.reshape(nO, nV, nP, nO).transpose(2, 0, 3, 1)
    del K1, G, rho_po

    E = eri[P, o, v, v]                                 # <pi|ab> at [p,i,a,b]
    K5 = _spin_combination(wd, E.transpose(0, 3, 1, 2), wx, E.transpose(0, 2, 1, 3),
                           nP * nV)
    del E
    K7 = _spin_combination(wd, eri[P, v, o, v].transpose(0, 3, 2, 1),
                           wx, eri[P, v, v, o].transpose(0, 2, 3, 1), nP * nV)
    rho_vp = rho[v, P, :].transpose(1, 0, 2)            # rho_bp,n at [p,b,n]
    S = K5 @ U
    S += K7 @ W
    del K7
    particle += S.reshape(nP, nV, nS) * rho_vp
    del S
    G = W @ rho[P, v, :].reshape(nP * nV, nS).T
    G -= U @ rho_vp.reshape(nP * nV, nS).T              # [ia, pb]
    pph += (K5.reshape(nP, nV, nO, nV).transpose(0, 2, 3, 1)
            * G.reshape(nO, nV, nP, nV).transpose(2, 0, 1, 3))


def _pp_channel(eps, eri, nO, P, om_ee, rho_ee, om_hh, rho_hh, f, eta, block_bytes,
                ee, hh, hhp, pph):
    """Add one pp channel's residues (R_Parquet_self_energy_diag, singlet or
    triplet pp part, weight f) of orbitals P to ee[p, k, m] (pole
    Om^ee_m - e_k), hh[p, c, m] (pole Om^hh_m - e_c), hhp[p, i, j, c] and
    pph[p, k, a, b].

    With the regularized amplitudes
    Bee_ij,m = rho^ee_ij,m R(Om^ee_m - e_i - e_j),
    Bhh_ij,m = rho^hh_ij,m R(e_i + e_j - Om^hh_m),
    Bhh_ab,m = rho^hh_ab,m R(Om^hh_m - e_a - e_b) and
    Bee_ab,m = rho^ee_ab,m R(e_a + e_b - Om^ee_m),

        ee    -f rho^ee_pk,m [sum_ij <pk|ij> Bee_ij,m + sum_ab <pk|ab> Bee_ab,m]
        hh     f rho^hh_pc,m [sum_ij <pc|ij> Bhh_ij,m + sum_ab <pc|ab> Bhh_ab,m]
        2h1p  -f <pc|ij> sum_m [Bee_ij,m rho^ee_pc,m + Bhh_ij,m rho^hh_pc,m]
        2p1h   f <pk|ab> sum_m [Bhh_ab,m rho^hh_pk,m + Bee_ab,m rho^ee_pk,m]

    (Quack's 4h1p and second 3p2h terms; first 3h2p and 4p1h; the other
    3h2p; the other 3p2h).

    The ee virtual block (nV^2 by nVV) is streamed in blocks of the ee states,
    the <pc|ab> slab in blocks of orbitals.
    """
    norb = len(eps)
    nV, nP, nE, nH = norb - nO, len(P), len(om_ee), len(om_hh)
    o, v = slice(0, nO), slice(nO, norb)
    eo, ev = eps[o], eps[v]
    d = (om_ee - eo[:, None, None]) - eo[None, :, None]         # Om - e_i - e_j
    b_ee_o = np.ascontiguousarray(rho_ee[o, o, :] * _srg_over(d, eta))
    d = (eo[:, None] + eo[None, :])[:, :, None] - om_hh         # e_i + e_j - Om
    b_hh_o = np.ascontiguousarray(rho_hh[o, o, :] * _srg_over(d, eta))
    d = (om_hh - ev[:, None, None]) - ev[None, :, None]         # Om - e_a - e_b
    b_hh_v = np.ascontiguousarray(rho_hh[v, v, :] * _srg_over(d, eta))
    del d
    b_ee_o = b_ee_o.reshape(nO * nO, nE)
    b_hh_o = b_hh_o.reshape(nO * nO, nH)
    b_hh_v = b_hh_v.reshape(nV * nV, nH)

    # ee poles and the 2p1h sums, streamed over the ee states
    S = eri[P, o, o, o].reshape(nP * nO, nO * nO) @ b_ee_o      # <pk|ij> Bee_ij,m
    E_ov = eri[P, o, v, v].reshape(nP * nO, nV * nV)            # <pk|ab>
    rho_po = rho_ee[P, o, :].reshape(nP * nO, nE)               # rho^ee_pk,m
    H = b_hh_v @ rho_hh[P, o, :].reshape(nP * nO, nH).T         # [ab, pk]
    ab = (ev[:, None] + ev[None, :])[:, :, None]                # e_a + e_b
    step = max(1, block_bytes // (8 * nV * nV))
    for m0 in range(0, nE, step):
        m1 = min(m0 + step, nE)
        B = rho_ee[v, v, m0:m1] * _srg_over(ab - om_ee[m0:m1], eta)
        B = np.ascontiguousarray(B).reshape(nV * nV, m1 - m0)
        S[:, m0:m1] += E_ov @ B
        H += B @ rho_po[:, m0:m1].T
        del B
    ee += (-f * rho_po.reshape(nP, nO, nE)) * S.reshape(nP, nO, nE)
    pph += ((f * E_ov.reshape(nP, nO, nV, nV))
            * H.reshape(nV, nV, nP, nO).transpose(2, 3, 0, 1))
    del S, E_ov, rho_po, H

    # hh poles: <pc|ij> for all orbitals, <pc|ab> in blocks of orbitals
    E_vo = eri[P, v, o, o]                                      # <pc|ij> at [p,c,i,j]
    T = E_vo.reshape(nP * nV, nO * nO) @ b_hh_o
    step = max(1, block_bytes // (8 * nV ** 3))
    for s0 in range(0, nP, step):
        s1 = min(s0 + step, nP)
        E = eri[P[s0:s1], v, v, v].reshape((s1 - s0) * nV, nV * nV)
        T[s0 * nV:s1 * nV] += E @ b_hh_v
        del E
    hh += (f * rho_hh[P, v, :]) * T.reshape(nP, nV, nH)
    del T

    # 2h1p: one orbital at a time, rho^ee_pc,m and rho^hh_pc,m are contiguous
    M = np.empty((nP, nO * nO, nV))
    for k, p in enumerate(P):
        M[k] = b_ee_o @ rho_ee[p, v, :].T + b_hh_o @ rho_hh[p, v, :].T
    hhp += (-f * E_vo.transpose(0, 2, 3, 1)) * M.reshape(nP, nO, nO, nV)


class DiagonalSelfEnergy:
    """Sigma_pp(w) of a set of orbitals as a sum over regularized simple
    poles, built once from the BSE spectra (R_Parquet_self_energy_diag and
    R_Parquet_self_energy_diag_omega, Eqs. (21), (48), (50), (51), (53)).

    Parameters
    ----------
    eta : float
        One-body regularization strength (Quack's eta_1b, the paper's s_1b);
        a large value (1e8) is effectively unregularized.
    eps : ndarray, shape (norb,)
        The energies of every denominator (Quack's eOld: the HF energies in
        osPA, the previous iterate in evPA), occupied first.
    eri : ndarray, shape (norb, norb, norb, norb)
        <pq|rs> in the MO basis.
    nO : int
        Doubly occupied orbitals.
    spectra : spectra.RestrictedSpectra
        The six channels (om, rho) of the converged two-body problem.
    orbitals : sequence of int, optional
        The orbitals p (0-based) whose Sigma_pp is wanted; default all. Every
        per-orbital array a method returns runs over these, in this order.
    terms : sequence of str, optional
        The parts of TERMS to include; default all (the full self-energy).
    block_bytes : int, optional
        Size of the blocks the build streams (PARQUET_SELF_ENERGY_BLOCK_BYTES).

    Attributes
    ----------
    positions : ndarray, shape (npoles,)
        x_k, Sigma_pp(w) = sum_k residues[row(p), k] R(w - x_k).
    offsets : ndarray, shape (3, npoles)
        The energies (A, B, C) whose sum ((w + A) + B) + C is the denominator
        w - x_k in Quack's order of operations (C is 0 for the two-energy
        families).
    residues : ndarray, shape (len(orbitals), npoles)
    groups : dict
        POLE_GROUPS name -> (slice of the poles, index shape), e.g.
        '2h1p' -> (slice, (nO, nO, nV)) for x_ija = e_i + e_j - e_a.
    """

    def __init__(self, eta, eps, eri, nO, spectra, orbitals=None, terms=TERMS,
                 block_bytes=PARQUET_SELF_ENERGY_BLOCK_BYTES):
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
        terms = tuple(terms)
        unknown = set(terms) - set(TERMS)
        if unknown or not terms:
            raise ValueError(f'terms {terms!r}: choose a non-empty subset of {TERMS}')
        if orbitals is None:
            P = np.arange(norb)
        else:
            P = np.atleast_1d(np.asarray(orbitals, dtype=int))
        if P.ndim != 1 or P.size == 0 or P.min() < 0 or P.max() >= norb:
            raise ValueError(f'orbitals must be indices in [0, {norb})')
        chans = {name: _channel_arrays(getattr(spectra, name, None), norb, name)
                 for name in ('eh_singlet', 'eh_triplet', 'ee_singlet', 'ee_triplet',
                              'hh_singlet', 'hh_triplet')}
        nV = norb - nO
        self.eta = float(eta)
        self.eps = eps
        self.nO = nO
        self.orbitals = P
        self.terms = terms
        self._rows = {int(p): k for k, p in enumerate(P)}

        # The frequency-dependent denominator of a pole is ((w + A) + B) + C,
        # the energies added in the order Quack's expression adds them, so that
        # every denominator is Quack's to the last bit.
        eo, ev = eps[:nO], eps[nO:]
        zero = np.zeros(())
        ija, iab = (nO, nO, nV), (nO, nV, nV)
        families = {
            # w + e_a - e_i - e_j  (second order; eh terms 4; pp 3h2p terms)
            '2h1p': (ev[None, None, :], -eo[:, None, None], -eo[None, :, None], ija),
            # w - e_i - e_j + e_a  (eh terms 2)
            '2h1p_alt': (-eo[:, None, None], -eo[None, :, None], ev[None, None, :],
                         ija),
            # w + e_i - e_a - e_b  (second order; eh terms 6, 8; pp 3p2h terms)
            '2p1h': (eo[:, None, None], -ev[None, :, None], -ev[None, None, :], iab)}
        for spin in ('singlet', 'triplet'):
            if f'eh_{spin}' in terms:
                om = chans[f'eh_{spin}'][0][None, :]
                # w - e_j + Om (eh terms 1, 3), w - e_b - Om (terms 5, 7)
                families[f'eh_{spin}_hole'] = (-eo[:, None], om, zero, (nO, om.size))
                families[f'eh_{spin}_particle'] = (-ev[:, None], -om, zero,
                                                   (nV, om.size))
            if f'pp_{spin}' in terms:
                om_ee = chans[f'ee_{spin}'][0][None, :]
                om_hh = chans[f'hh_{spin}'][0][None, :]
                # w + e_k - Om^ee, w + e_c - Om^hh
                families[f'ee_{spin}'] = (eo[:, None], -om_ee, zero, (nO, om_ee.size))
                families[f'hh_{spin}'] = (ev[:, None], -om_hh, zero, (nV, om_hh.size))
        if not any(f'eh_{spin}' in terms for spin in ('singlet', 'triplet')):
            del families['2h1p_alt']
        self.groups, start = {}, 0
        for name in POLE_GROUPS:
            if name in families:
                shape = families[name][3]
                size = int(np.prod(shape))
                self.groups[name] = (slice(start, start + size), shape)
                start += size
        self.offsets = np.empty((3, start))
        for name, (sl, shape) in self.groups.items():
            for k in range(3):
                self.offsets[k, sl] = np.broadcast_to(families[name][k], shape).ravel()
        # the families with a third energy come first
        self._n3 = max(self.groups[g][0].stop
                       for g in ('2h1p', '2h1p_alt', '2p1h') if g in self.groups)
        self.positions = -((self.offsets[0] + self.offsets[1]) + self.offsets[2])
        self.residues = np.zeros((len(P), start))

        with np.errstate(**_IEEE_SILENT):
            hhp, pph = self.group_residues('2h1p'), self.group_residues('2p1h')
            if '2d' in terms or '2x' in terms:
                _second_order(eri, nO, P, terms, hhp, pph)
            for spin in ('singlet', 'triplet'):
                if f'eh_{spin}' in terms:
                    om, rho = chans[f'eh_{spin}']
                    _eh_channel(eps, eri, nO, P, om, rho, EH_SPIN_WEIGHTS[spin],
                                self.eta, self.group_residues(f'eh_{spin}_hole'),
                                self.group_residues(f'eh_{spin}_particle'), hhp,
                                self.group_residues('2h1p_alt'), pph)
                if f'pp_{spin}' in terms:
                    om_ee, rho_ee = chans[f'ee_{spin}']
                    om_hh, rho_hh = chans[f'hh_{spin}']
                    _pp_channel(eps, eri, nO, P, om_ee, rho_ee, om_hh, rho_hh,
                                PP_SPIN_WEIGHTS[spin], self.eta, int(block_bytes),
                                self.group_residues(f'ee_{spin}'),
                                self.group_residues(f'hh_{spin}'), hhp, pph)
        self._d, self._r, self._t = np.empty(start), np.empty(start), np.empty(start)
        self._zero = np.empty(start, dtype=bool)

    def group_residues(self, group):
        """The residues of one pole family (a POLE_GROUPS name) as a writable
        view of shape (len(orbitals),) + the family's index shape."""
        sl, shape = self.groups[group]
        return self.residues[:, sl].reshape((len(self.orbitals),) + shape)

    def row(self, p):
        """The residue row of orbital p (0-based)."""
        try:
            return self._rows[int(p)]
        except KeyError:
            raise ValueError(f'orbital {p} is not among the orbitals this '
                             f'self-energy was built for') from None

    def at(self, p, w):
        """(Sigma_pp(w), dSigma_pp/dw) of orbital p (0-based) at frequency w
        (Hartree): R_Parquet_self_energy_diag_omega's SigC and Z.

        The derivative holds the regularizers 1 - exp(-2 eta (w - x)^2)
        constant, as Quack's does; it is not 1/(1 - ...). A pole hit exactly
        (w - x_k = 0), Quack's 0/0, adds 0 to Sigma and 0 to the derivative,
        as that pole a few ulps off does (ZERO DENOMINATORS). One exp per pole;
        the sums over the poles are numpy's pairwise ones, not BLAS dot
        products, whose threads cost more than the sum between the
        elementwise passes of a frequency loop.
        """
        c = self.residues[self.row(p)]
        n3 = self._n3
        d, r, t, zero = self._d, self._r, self._t, self._zero
        with np.errstate(**_IEEE_SILENT):
            np.add(w, self.offsets[0], out=d)
            d += self.offsets[1]
            d[:n3] += self.offsets[2, :n3]
            np.multiply(-2.0 * self.eta, d, out=r)
            r *= d
            np.exp(r, out=r)
            np.subtract(1.0, r, out=r)
            r /= d                                  # reg/dem
            sig = np.multiply(c, r, out=t).sum()
            # an exactly zero d makes its reg/dem 0/0 and so sig NaN: only then are the
            # zeros looked for and given 0 (ZERO DENOMINATORS), and sig summed again
            hit = (not math.isfinite(sig) and self.eta < np.inf
                   and np.equal(d, 0.0, out=zero).any())
            if hit:
                r[zero] = 0.0                       # Quack's 0/0: the limit, 2 eta d -> 0
                sig = np.multiply(c, r, out=t).sum()
            np.divide(r, d, out=t)                  # reg/dem/dem
            if hit:
                t[zero] = 0.0                       # Quack's slope a few ulps off zero
            t *= c
            dsig = -t.sum()
        return sig, dsig

    def diagonal(self):
        """(sigc, z) over `orbitals`, each orbital p at w = eps[p]:
        R_Parquet_self_energy_diag's SigC and Z = 1/(1 - dSigma_pp/dw)."""
        sig = np.empty(len(self.orbitals))
        dsig = np.empty(len(self.orbitals))
        for k, p in enumerate(self.orbitals):
            sig[k], dsig[k] = self.at(p, self.eps[p])
        with np.errstate(divide='ignore', invalid='ignore'):
            z = 1.0 / (1.0 - dsig)
        bad = ~(np.isfinite(sig) & np.isfinite(z))
        if bad.any():
            flag(FLAG_SELF_ENERGY_NOT_FINITE,
                 f'the diagonal parquet self-energy is not finite for orbitals '
                 f'{self.orbitals[bad].tolist()} (0-based): an energy, a spectrum or an '
                 f'integral that is not finite, an overflow, a pole hit at eta = inf (the '
                 f'bare 1/d), or dSigma/dw exactly 1 (Z = 1/0); the values are returned as '
                 f'computed',
                 ParquetNotFinite, stacklevel=2)
        return sig, z

    @flags_into_out
    def solve_qp_scan(self, eps_hf, follow=None, out=None):
        """(eqp, z, found) over `orbitals`: each orbital's quasiparticle
        equation solved by the house's one root finder (not Quack's).

        The equation is f(w) = w - eps_hf[p] - Sigma_pp(w) with the other
        denominators on `eps` (Sigma_pp from `at`), handed to
        src.Solvers.qp_equation.solve_qp_equation_scan about the bare level
        a = eps_hf[p] with that finder's own tolerance (QP_ROOT_TOL): each
        bracket of its scan is refined by a Newton safeguarded by the
        bracket on Quack's slope f'(w) = 1 - dSigma_pp/dw (the regularizers
        held constant, as in Quack's Z), and z = 1/f' at the root, where the
        scan's residual |f| reaches 5e-12 Ha. positive_weight is off: the
        parquet self-energy has residues of both signs, so a bracket where f
        falls may hold a root, and no
        sum rule on the weights ends a search early. The pole-sum route
        (`PoleSum`) is not taken: its largest weight over the whole axis
        picks roots far from the quasiparticle here (water orbital 1 at
        +47 eV at s_1b 1e8).

        Which root. Without `follow` (the first one-body iteration) the scan
        windows about a (QP_SEARCH_WINDOW_TIERS) are taken narrowest first,
        every bracket of a window refined, and the first window whose
        largest valid weight (0 < Z <= 1) exceeds QP_WINDOW_Z_MIN ends the
        search with that root; failing that, the widest window's largest
        valid weight is kept, or, with no valid weight at all, the root
        nearest a, with its Z outside (0, 1]. With `follow` (evPA from the
        second one-body iteration on: the previous iteration's eQP) the root
        kept is the one of valid weight nearest follow[p], on the narrowest
        window about a that holds follow[p] (a wider one where that root
        lies past the window's edge), whatever its weight, so that the
        orbital stays on its branch; where the branch has gone (no root of
        valid weight in any window that holds follow[p]) the largest weight
        is kept again, as without `follow`, after the scans that looked for
        the branch. An orbital whose follow[p] is not finite (NaN) is solved
        as without `follow`: the drivers hand NaN for an orbital whose
        previous root has Z outside (0, 1] (`options.roots_to_follow`), a
        root the finder follows only to a valid one beside it, falling back
        otherwise to the search without `follow` at twice its cost.
        Re-deciding by weight at every iteration made the evPA of
        ammonia/6-31G (TDA, s_2b 1, s_1b 50) cycle through three N 1s roots
        from the third iteration on (weights 0.70, 0.83 and 0.66);
        following, it converges, as water does.

        Cost, in evaluations of Sigma_pp per orbital, one serving f and f'
        at the same w: without `follow` 150 to 10 000 on water and ammonia
        6-31G and 150 to 14 000 on water/aug-cc-pVDZ (norb 41, FLEX TDA:
        7.7 s per one-body step at s_1b 50 and 33 s at s_1b 1e8);
        following, the cells about the root followed and then the rest of
        its scan window, every bracket refined for the branch report, about
        what a search without `follow` costs in that window (evPA, s_2b 1, on
        water and ammonia 6-31G: 150 to 3 200 at s_1b 50, an iteration after
        the first 13 600 to 16 400 against the first's 13 300 and 18 100;
        160 to 3 700 at s_1b 1e8 with two-body DIIS 5). A branch gone costs
        about twice a search without `follow` (water/6-31G orbital 11,
        Z = 1.067 at s_1b 50, followed from its own root: 11 000 against
        5 500). In the evPA runs above the roots with Z outside (0, 1] are
        those of the first iteration, and the second, which does not follow
        them, searches those orbitals afresh, at what following them would
        cost (ammonia/6-31G orbitals 12 and 13, their Z back in (0, 1]
        there: 2 302 evaluations each either way, the same roots). How the
        finder compares with Quack's own search is in
        tests/parquet_quack_newton.py, which carries that search for the
        tests.

        Parameters
        ----------
        eps_hf : ndarray, shape (norb,)
            The bare levels a = eHF_p of the equations, indexed by orbital.
        follow : ndarray, shape (norb,), optional
            The roots to follow, indexed by orbital (the previous evPA
            iteration's eQP), NaN (not finite) for an orbital not followed,
            whose largest weight is kept; None follows none.
        out : dict, optional
            Receives 'n_eval' (evaluations of Sigma_pp per orbital), one bool
            per orbital in 'no_root', 'z_outside', 'followed' (the branch
            followed continued; False where nothing is followed) and
            'flipped' (the finder found a larger valid weight at another
            root in the scan window of the root followed, every bracket of
            which it refines, or the branch has gone and a root of largest
            weight was kept again; an equation left without a root is
            'no_root' instead), the finder's `out` record, and
            'numerical_flags', the flags raised.

        Returns
        -------
        eqp, z : ndarray, shape (len(orbitals),)
            The roots and 1/f' there. An orbital without a root in the
            widest window is NaN, with the finder's QPNoRoot warning and
            named flag 'no_root'; a root kept with z outside (0, 1] carries
            the finder's QPWeightOutside warning and named flag 'z_outside',
            as on every quasiparticle route. A root followed that the finder
            reports 'flipped' is warned as the house warns for a followed
            root (RuntimeWarning, qp_equation.branch_warning) and flagged
            'parquet_qp_branch_flipped' (`options.flag_branch`).
        found : ndarray of bool, shape (len(orbitals),)
            Whether the orbital's equation had a root.
        """
        eps_hf = np.asarray(eps_hf, dtype=float)
        norb = len(self.eps)
        if eps_hf.shape != (norb,):
            raise ValueError(f'eps_hf has shape {eps_hf.shape}, expected ({norb},)')
        if follow is not None:
            follow = np.asarray(follow, dtype=float)
            if follow.shape != (norb,):
                raise ValueError(f'follow has shape {follow.shape}, expected ({norb},)')
        P = self.orbitals
        eqp, z = np.empty(len(P)), np.empty(len(P))
        n_eval = np.zeros(len(P), dtype=int)
        no_root, z_out, followed, flipped = (np.zeros(len(P), dtype=bool) for _ in range(4))
        for k, p in enumerate(P):
            residual, slope, count = self._equation(p, eps_hf[p])
            record = {}
            label = f'the parquet quasiparticle equation of orbital {p} (0-based)'
            w_follow = (None if follow is None or not np.isfinite(follow[p])
                        else float(follow[p]))
            eqp[k], z[k] = solve_qp_equation_scan(
                residual, slope, float(eps_hf[p]), follow=w_follow, label=label,
                out=record, positive_weight=False)
            n_eval[k] = count[0]
            no_root[k], z_out[k] = record['no_root'], record['z_outside']
            followed[k] = bool(record['followed'])
            if w_follow is not None:
                flipped[k] = flag_branch(label, eqp[k], z[k], record, w_follow)
        if out is not None:
            out.update(n_eval=n_eval, no_root=no_root, z_outside=z_out, followed=followed,
                       flipped=flipped)
        return eqp, z, ~no_root

    def _equation(self, p, a):
        """(f, f', count) of orbital p's equation f(w) = w - a - Sigma_pp(w)
        for the root finder: f and f' at one w share one evaluation of `at`,
        and count[0] counts the evaluations."""
        memo = {'w': None}
        count = [0]

        def sigma(w):
            w = float(w)
            if memo['w'] != w:
                memo['sig'], memo['dsig'] = self.at(p, w)
                memo['w'] = w
                count[0] += 1
            return memo['sig'], memo['dsig']

        def f(w):
            return w - a - sigma(w)[0]

        def fprime(w):
            return 1.0 - sigma(w)[1]

        return f, fprime, count


def diagonal_self_energy(eta, eps, eri, nO, spectra):
    """(sigc, z): the diagonal parquet self-energy of every orbital and its
    renormalization factor, as Quack's R_Parquet_self_energy_diag (the
    simple-pole version R_evParquet calls).

    Parameters
    ----------
    eta : float
        One-body regularization strength (eta_1b, s_1b of Eq. (53)).
    eps : ndarray, shape (norb,)
        The energies of every denominator (eOld); orbital p is evaluated at
        w = eps[p].
    eri : ndarray, shape (norb,)*4
        <pq|rs>.
    nO : int
        Doubly occupied orbitals.
    spectra : spectra.RestrictedSpectra

    Returns
    -------
    sigc : ndarray, shape (norb,)
        Sigma_pp(eps[p]), Eqs. (21), (48), (50), (51) regularized as Eq. (53).
    z : ndarray, shape (norb,)
        1 / (1 - dSigma_pp/dw) at w = eps[p], the regularizers held constant.

    Quack's EcGM output is always 0 and is not returned.
    """
    return DiagonalSelfEnergy(eta, eps, eri, nO, spectra).diagonal()


def self_energy_at(p, w, eta, eps, eri, nO, spectra):
    """(sig, dsig): Sigma_pp(w) and dSigma_pp/dw of one orbital p (0-based) at
    frequency w, every other denominator on eps, as Quack's
    R_Parquet_self_energy_diag_omega (whose p is 1-based and whose Z output
    is this dsig, not 1/(1 - dsig)).

    Builds the residues of orbital p alone; to evaluate many frequencies,
    build a DiagonalSelfEnergy once and call its `at`.
    """
    return DiagonalSelfEnergy(eta, eps, eri, nO, spectra, orbitals=[p]).at(p, w)


def linearized_qp_energies(eps_hf, sigc, z):
    """R_evParquet's eQPlin = eHF + Z SigC: anchored on the HF energies at
    every one-body iteration, while SigC and Z are evaluated at w = eOld (so
    beyond the first evPA iteration this is not the linearization about
    eOld, eOld + Z (eHF + SigC - eOld))."""
    z, sigc = np.asarray(z, dtype=float), np.asarray(sigc, dtype=float)
    return np.asarray(eps_hf, dtype=float) + z * sigc


def self_energy_parts(eta, eps, eri, nO, spectra, orbitals=None):
    """The decomposition dump_RParquet_self_energy prints, each orbital at
    w = eps[p]: a dict TERMS name -> Sigma_pp over `orbitals` (default all),

        '2d', '2x'                    Sig2d, Sig2x   (second order, Eq. (48))
        'eh_singlet', 'eh_triplet'    Sig1eh, Sig3eh (Eq. (50))
        'pp_singlet', 'pp_triplet'    Sig1pp, Sig3pp (Eq. (51))

    whose sum is diagonal_self_energy's sigc (Quack forms the parts as
    differences of its running sum; each is built here on its own).
    """
    return {t: DiagonalSelfEnergy(eta, eps, eri, nO, spectra, orbitals=orbitals,
                                  terms=(t,)).diagonal()[0]
            for t in TERMS}
