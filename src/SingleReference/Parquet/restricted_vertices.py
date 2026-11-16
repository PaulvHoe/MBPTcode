"""The closed-shell parquet two-body step: generalized effective integrals and
reducible vertices.

One two-body iteration of the parquet loop (Marie and Loos, J. Chem. Phys.
163, 194115 (2025), Fig. 8, steps 5 to 7) takes the eigenvectors of the eh
and pp BSEs, contracts them with the irreducible vertices into the
generalized effective integrals M (Eqs. (38) and (44)), and builds from M and
the BSE eigenvalues the reducible vertices Phi (Eqs. (37) and (43)), which
make the kernels of the next iteration through Eq. (47). This module is that
step for a closed shell, ported from Quack (git e3528a40): the routines of
R_screened_integrals.f90 and R_{eh,pp}_{singlet,triplet}_Phi.f90.

Generalized effective integrals
-------------------------------
The eh integrals of an excitation n (`eh_effective_integrals`) are

    M^eh_{pq,n} = sum_{jb} K_{pbqj} X_{jb,n} + K_{pjqb} Y_{jb,n},
    X = (XpY + XmY) / 2,  Y = (XpY - XmY) / 2,

with K = i Gamma^eh, the closed-shell irreducible eh vertex of Eq. (47a) as
Quack spin-adapts it (index order p x q y, Phi read as below):

    singlet  K_{pxqy} = 2 <px|qy> - <px|yq>
                        - 1/2 Phi^{eh,s}_{pxyq} - 3/2 Phi^{eh,t}_{pxyq}
                        + 1/2 Phi^{pp,s}_{pxqy} + 3/2 Phi^{pp,t}_{pxqy}
    triplet  K_{pxqy} = - <px|yq>
                        - 1/2 Phi^{eh,s}_{pxyq} + 1/2 Phi^{eh,t}_{pxyq}
                        - 1/2 Phi^{pp,s}_{pxqy} + 1/2 Phi^{pp,t}_{pxqy}

The pp integrals of a double attachment m (M^ee) and a double removal m
(M^hh) (`pp_effective_integrals`) are

    M^ee_{pq,m} = sum_{c<=d} K_{pq,cd} X1_{cd,m} + sum_{k<=l} K_{pq,kl} Y1_{kl,m},
    M^hh_{pq,m} = sum_{c<=d} K_{pq,cd} X2_{cd,m} + sum_{k<=l} K_{pq,kl} Y2_{kl,m},

(c < d and k < l for the triplet) with K = i Gamma^pp of Eq. (47b), which
holds no pp vertex:

    singlet  K_{pq,rs} = [<pq|rs> + <pq|sr>
                          + 1/2 Phi^{eh,s}_{pqrs} - 3/2 Phi^{eh,t}_{pqrs}
                          + 1/2 Phi^{eh,s}_{pqsr} - 3/2 Phi^{eh,t}_{pqsr}]
                         / sqrt(1 + delta_rs)
    triplet  K_{pq,rs} = <pq|rs> - <pq|sr>
                         + 1/2 (Phi^{eh,s} + Phi^{eh,t})_{pqrs}
                         - 1/2 (Phi^{eh,s} + Phi^{eh,t})_{pqsr}

Only the summed pair (r, s) carries the singlet norm; (p, q) is any pair of
orbitals.

Reducible vertices
------------------
With the excitation energies Omega and the integrals M of one channel,

    Phi^eh_{pqrs}(omega) = sum_n [ M_{pr,n} M_{sq,n} R(omega - Omega_n)
                                 - M_{rp,n} M_{qs,n} R(omega + Omega_n) ]       (37)
    Phi^pp_{pqrs}(omega) = sum_m M^ee_{pq,m} M^ee_{rs,m} R(omega - Omega^ee_m)
                         - sum_m M^hh_{pq,m} M^hh_{rs,m} R(omega - Omega^hh_m)  (43)

where R(d) = (1 - exp(-2 eta d^2)) / d is Quack's SRG-regularized 1/d
(`regularized_inverse`): eta is the two-body flow parameter s_2b of Eq. (52)
(Quack's eta_2b), eta = 0 makes every Phi vanish (FLEX) and eta -> infinity
is the unregularized 1/d. At omega = 0 these are Eqs. (39) and (45) with the
regularizer of Eq. (52). Quack regularizes the whole denominator, so away
from omega = 0 the regulator is exp(-2 eta (omega -+ Omega)^2), not the
exp(-2 eta Omega^2) of Eq. (52). Omega^hh are the hh eigenvalues as the pp BSE
returns them (negative for a bound pair). The singlet and triplet channels
share these formulas. Phi is the i*Phi of the paper (real for real orbitals),
stored Phi[p, q, r, s] as Quack stores it.

Frequencies: where the static kernel approximation lives
--------------------------------------------------------
`omega` is the bosonic frequency of the vertex: the eh transfer frequency of
Phi^eh (Eq. (29)) and the pp total frequency of Phi^pp (pp convention,
Eq. (30)). Each depends on that one frequency only because the irreducible
vertices are taken static (Eq. (28)). Quack's two-body loop (R_evParquet)
evaluates all four vertices at omega = 0 (it passes 0d0) at every iteration;
the result, damped (1/4 new + 3/4 old: the drivers' default damping 0.75)
and DIIS-extrapolated, is the Phi that the next iteration reads through
Eq. (47) and nowhere else:

* the eh BSE kernels (R_eh_*_Gamma_A/B) read the eh vertices crossed,
  Phi^eh_{pqsr} of Eq. (47a) (order p x y q above), and the pp vertices
  direct, Phi^pp_{pqrs};
* the pp BSE kernels (R_pp_*_Gamma_B/C/D) read the eh vertices in both
  orders, Phi^eh_{pqrs} - Phi^eh_{pqsr} of Eq. (47b), and no pp vertex;
* the effective integrals here contract the same irreducible vertices, built
  from the same (previous) Phi, as the BSE whose eigenvectors they take.

So every reducible vertex inside a kernel is its zero-frequency value. In the
frequency-dependent equations a Phi inside another channel's Gamma (Eq. (46))
is evaluated at its own bosonic frequency, which in that channel's variables
is a combination of its fermionic and bosonic frequencies; the static kernel
approximation (Eq. (28)) takes every Gamma at zero frequency, and Quack
realizes it by this omega = 0 evaluation. The self-energy (Eqs. (31), (50),
(51); R_Parquet_self_energy_diag) does not read these tensors: it takes
(Omega, M) of each channel, the whole frequency dependence of Eqs. (37) and
(43). A frequency-averaged vertex sum_k c_k Phi(omega_k) is the same
contraction with the weights sum_k c_k R(omega_k -+ Omega) in place of
R(0 -+ Omega): `eh_vertex_from_weights` and `pp_vertex_from_weights` take the
weights, so an average costs one contraction, as the static vertex does.
Through the drivers' `vertex_builder` hook such an average can only be a
real one (a complex tensor is refused). A builder of the default protocol
gives one tensor per source channel, which feeds both the eh and the pp
kernels (and their effective integrals), and sees neither eps nor nO; a
per-insertion builder (per_insertion = True) gives each insertion role its
own tensor, the eh vertex crossed into the eh kernel ('phbar_eh') and into
the pp kernel ('eh_pp', its direct and swapped terms alike, which exchange
turns into each other), the pp vertex into the eh kernel ('pp_eh'), and is
handed the iteration's eps and nO (docs/methods/parquet.md, One tensor per
insertion). The functions here read whichever tensors they are handed.

Conventions
-----------
eri[p, q, r, s] = <pq|rs> (MO, physicist, not antisymmetrized); occupied
orbitals first, nO of them, nV = norb - nO virtual; no frozen orbitals
(Quack's nC = nR = 0). Pair orders are Quack's (`pairs`): eh pairs jb with j
outer and b inner, pp pairs upper triangles row by row. The eh eigenvectors
are XpY[n, jb], XmY[n, jb] (row n an excitation, as phRLR returns them); the
pp ones X1[cd, m] (nVV, nVV), Y1[kl, m] (nOO, nVV), X2[cd, m] (nVV, nOO),
Y2[kl, m] (nOO, nOO), as ppRLR returns them. rho[p, q, n] holds M_{pq,n}.
`vertices` is a `RestrictedVertices` (eh_singlet, eh_triplet, pp_singlet,
pp_triplet) or any sequence in that order.

Every contraction is a GEMM over the pair or excitation index: M^eh one GEMM
with K = 2 nS, M^ee and M^hh two each (Quack's four dgemm) on the irreducible
vertex gathered a block of p at a time, Phi^pp two (ee, a block of (pq) rows
at a time, then hh accumulated in place), Phi^eh one with K = 2 nS formed a
block of p at a time (it is not a product in its own (pq),(rs) layout). A
block is at most PARQUET_EH_VERTEX_BLOCK_BYTES, PARQUET_PP_VERTEX_BLOCK_BYTES
or PARQUET_PP_INTEGRAL_BLOCK_BYTES and 1/PARQUET_MIN_BLOCKS_PER_VERTEX of a
vertex. With OpenBLAS (checked on 0.3.21) the blocks change no bit at one
thread: its GEMM sums each element over K alike for any number of rows from
two on (numpy takes a one-row product as a GEMV, so no block has one row).
The vertex products (N = norb^2 columns) stay bitwise at 2 and 4 threads too;
the narrow M^hh products (N = nOO columns, K = nVV) do not, since OpenBLAS
rounds them differently with the thread count and the number of rows (4e-15
relative on random entries at norb 41, as the whole product moves between 1
and 2 threads). Another BLAS may round any block differently in the last
bit.

Quack's own behaviour, kept: 1 - exp(-x) is evaluated as written, not as
-expm1(-x); the singlet pair norm is Quack's literal 0.7071067811865475d0,
which is 1/np.sqrt(2.0) bitwise (`pp_pair_norm`). Departed from: a
denominator omega -+ Omega that vanishes exactly, where Quack's
(1 - exp(-2 eta d^2)) / d is 0/0 = NaN in the whole vertex, gives the
weight's limit 0 (`regularized_inverse`), so a zero mode weighs nothing
whether rounding puts it at 0 or a few ulps off; a weight that is not
finite (a NaN energy) makes the vertex NaN, as in Quack, warned about
(`events.ParquetNotFinite`) and flagged, 'parquet_vertex_pole'.
Where Quack's closed-shell code may give no number, both cases are
accepted: with one occupied orbital (H2, He; nO = 1) the triplet pp space has
no occupied pair, and with one virtual orbital (nV = 1, water/6-31G cut to
six orbitals) it has no virtual pair. Either way
R_pp_triplet_screened_integral hands an empty pair count to dgemm as a
leading dimension (nOO = 0 for Y1 and Y2, R_screened_integrals.f90:525 and
:531; nVV = 0 for X1 and X2, :523 and :529), a BLAS argument error that
Accelerate and the reference BLAS answer by ending the run (OpenBLAS and MKL
print it and skip the product, which is empty anyway). Here that
channel's M^hh simply has no column and M^ee no occupied-pair term (nO = 1),
or M^ee no column and no virtual-pair term (nV = 1). Quack's spin-orbital
G_evParquet, whose pp pairs are pairs of spin orbitals, has no empty pair
count there and runs these closed shells, and the restricted port
reproduces it (H2, He and the cut water: the spectra and EcRPA to 4e-13 Ha;
`restricted`), so the numbers have that reference.
"""
import numpy as np
from scipy.linalg import blas

from src.Base.numerical_flags import flag
from src.Base.numerics.constants import (PARQUET_EH_VERTEX_BLOCK_BYTES,
                                         PARQUET_MIN_BLOCKS_PER_VERTEX,
                                         PARQUET_PP_INTEGRAL_BLOCK_BYTES,
                                         PARQUET_PP_VERTEX_BLOCK_BYTES)
from src.SingleReference.Parquet.events import FLAG_VERTEX_POLE, ParquetNotFinite
from src.SingleReference.Parquet.pairs import EH_SPINS, pp_pair_norm, pp_pairs

#: The closed-shell eh irreducible vertex K_{pxqy} = i Gamma^eh of Eq. (47a)
#: that R_eh_{singlet,triplet}_screened_integral contracts, term by term in
#: Quack's order: (coefficient, tensor, read), 'direct' reading T[p, x, q, y]
#: and 'crossed' T[p, x, y, q].
EH_KERNEL_TERMS = {
    'singlet': ((2.0, 'eri', 'direct'), (-1.0, 'eri', 'crossed'),
                (-0.5, 'eh_singlet', 'crossed'), (-1.5, 'eh_triplet', 'crossed'),
                (0.5, 'pp_singlet', 'direct'), (1.5, 'pp_triplet', 'direct')),
    'triplet': ((-1.0, 'eri', 'crossed'),
                (-0.5, 'eh_singlet', 'crossed'), (0.5, 'eh_triplet', 'crossed'),
                (-0.5, 'pp_singlet', 'direct'), (0.5, 'pp_triplet', 'direct')),
}

#: The closed-shell pp irreducible vertex K_{pq,rs} = i Gamma^pp of Eq. (47b)
#: that R_pp_{singlet,triplet}_screened_integral contracts, before the singlet
#: pair norm, term by term in Quack's order: (coefficient, tensor, read),
#: 'direct' reading T[p, q, r, s] and 'swapped' T[p, q, s, r].
PP_KERNEL_TERMS = {
    'singlet': ((1.0, 'eri', 'direct'), (1.0, 'eri', 'swapped'),
                (0.5, 'eh_singlet', 'direct'), (-1.5, 'eh_triplet', 'direct'),
                (0.5, 'eh_singlet', 'swapped'), (-1.5, 'eh_triplet', 'swapped')),
    'triplet': ((1.0, 'eri', 'direct'), (-1.0, 'eri', 'swapped'),
                (0.5, 'eh_singlet', 'direct'), (0.5, 'eh_triplet', 'direct'),
                (-0.5, 'eh_singlet', 'swapped'), (-0.5, 'eh_triplet', 'swapped')),
}

#: The tensors of a `RestrictedVertices`, in its order.
VERTEX_NAMES = ('eh_singlet', 'eh_triplet', 'pp_singlet', 'pp_triplet')


def eh_effective_integrals(eri, vertices, XpY, XmY, nO, spin):
    """Generalized effective eh integrals M^eh_{pq,n} of one closed-shell eh
    BSE, Eq. (38).

    Ports R_eh_singlet_screened_integral and R_eh_triplet_screened_integral
    (Quack, R_screened_integrals.f90): the irreducible eh vertex of
    `EH_KERNEL_TERMS`, built from the reducible `vertices` the BSE was solved
    with, contracted with X = (XpY + XmY)/2 and Y = (XpY - XmY)/2 in one GEMM.

    Cost: a gather of 2 norb^2 nS kernel elements, then one GEMM of
    4 norb^2 nS^2 flops; memory: the kernel (2 norb^2 nS) beside rho. At
    norb = 138, nO = 5: 0.034 TFLOP, peak 0.32 GB.

    Parameters
    ----------
    eri : ndarray, shape (norb,)*4
        <pq|rs>, MO, physicist.
    vertices : RestrictedVertices
        The reducible vertices of the kernel (the previous iteration's).
    XpY, XmY : ndarray, shape (nS, nS)
        X + Y and X - Y of the BSE, row n an excitation, column jb.
    nO : int
        Doubly occupied orbitals.
    spin : {'singlet', 'triplet'}

    Returns
    -------
    rho : ndarray, shape (norb, norb, nS)
        rho[p, q, n] = M^eh_{pq,n}.
    """
    terms = _channel_terms(EH_KERNEL_TERMS, spin)
    eri = _four_index(eri, 'eri')
    norb = eri.shape[0]
    nV, nS = _occupation(norb, nO)
    tensors = _tensors(eri, vertices, norb)
    XpY = _matrix(XpY, (nS, nS), 'XpY')
    XmY = _matrix(XmY, (nS, nS), 'XmY')
    o, v = slice(0, nO), slice(nO, norb)
    # gam[p, q, 0, j, b] = K_{pbqj}, gam[p, q, 1, j, b] = K_{pjqb}, gathered one
    # p at a time: T[p] read (x, q, y) or (x, y, q), transposed to (q, j, b)
    gam = np.empty((norb, norb, 2, nO, nV))
    reads = ((v, o, (1, 2, 0), (2, 1, 0)),
             (o, v, (1, 0, 2), (2, 0, 1)))
    for p in range(norb):
        for block, (x, y, direct_axes, crossed_axes) in enumerate(reads):
            out = gam[p, :, block]
            for i, (coef, name, read) in enumerate(terms):
                tp = tensors[name][p]
                view = (tp[x, :, y].transpose(direct_axes) if read == 'direct'
                        else tp[x, y, :].transpose(crossed_axes))
                if i == 0:
                    np.multiply(view, coef, out=out)
                else:
                    out += coef * view
    X = 0.5 * (XpY + XmY)
    Y = 0.5 * (XpY - XmY)
    rho = gam.reshape(norb * norb, 2 * nS) @ np.concatenate((X, Y), axis=1).T
    return rho.reshape(norb, norb, nS)


def pp_effective_integrals(eri, vertices, X1, Y1, X2, Y2, nO, spin):
    """Generalized effective pp integrals M^ee_{pq,m} and M^hh_{pq,m} of one
    closed-shell pp BSE, Eq. (44).

    Ports R_pp_singlet_screened_integral and R_pp_triplet_screened_integral
    (Quack, R_screened_integrals.f90): the irreducible pp vertex of
    `PP_KERNEL_TERMS` (eh vertices only), its singlet pairs normalized by
    1/sqrt(1 + delta), contracted with the eigenvectors in Quack's four GEMMs.
    The vertex is gathered onto the pairs one orbital p at a time, and
    contracted a block of p at a time (at most PARQUET_PP_INTEGRAL_BLOCK_BYTES
    and 1/PARQUET_MIN_BLOCKS_PER_VERTEX of a vertex) into the rows p of M^ee
    and M^hh, so the whole gathered vertex is never held.

    Cost: GEMMs of 2 norb^2 (nVV + nOO)^2 flops after a gather of
    6 norb^2 (nVV + nOO) elements; memory: M^ee and M^hh, norb^2 (nVV + nOO),
    and one block of the gathered vertex. At norb = 138, nO = 5 (singlet):
    3.0 TFLOP, peak 1.6 GB (the integrals 1.36 GB, a block 0.27 GB).

    Parameters
    ----------
    eri : ndarray, shape (norb,)*4
        <pq|rs>, MO, physicist.
    vertices : RestrictedVertices
        The reducible vertices of the kernel (the previous iteration's); only
        eh_singlet and eh_triplet are read.
    X1, Y1 : ndarray, shapes (nVV, nVV) and (nOO, nVV)
        The double-attachment (ee) eigenvectors, column m.
    X2, Y2 : ndarray, shapes (nVV, nOO) and (nOO, nOO)
        The double-removal (hh) eigenvectors, column m.
    nO : int
        Doubly occupied orbitals.
    spin : {'singlet', 'triplet'}
        Pairs p <= q (nVV = nV(nV+1)/2, nOO = nO(nO+1)/2) or p < q.

    Returns
    -------
    rho_ee : ndarray, shape (norb, norb, nVV)
        rho_ee[p, q, m] = M^ee_{pq,m}.
    rho_hh : ndarray, shape (norb, norb, nOO)
        rho_hh[p, q, m] = M^hh_{pq,m}.
    """
    terms = _channel_terms(PP_KERNEL_TERMS, spin)
    eri = _four_index(eri, 'eri')
    norb = eri.shape[0]
    _occupation(norb, nO)
    tensors = _tensors(eri, vertices, norb)
    vir1, vir2 = pp_pairs(nO, norb, spin)
    occ1, occ2 = pp_pairs(0, nO, spin)
    nVV, nOO = vir1.size, occ1.size
    X1 = _matrix(X1, (nVV, nVV), 'X1')
    Y1 = _matrix(Y1, (nOO, nVV), 'Y1')
    X2 = _matrix(X2, (nVV, nOO), 'X2')
    Y2 = _matrix(Y2, (nOO, nOO), 'Y2')
    # the vv pairs, then the oo pairs, as flat (r, s) and (s, r) of an (r, s) axis
    r, s = np.concatenate((vir1, occ1)), np.concatenate((vir2, occ2))
    columns = {'direct': r * norb + s, 'swapped': s * norb + r}
    norm = np.concatenate((pp_pair_norm(vir1, vir2, spin),
                           pp_pair_norm(occ1, occ2, spin)))
    npair = nVV + nOO
    rho_ee = np.empty((norb, norb, nVV))
    rho_hh = np.empty((norb, norb, nOO))
    # gam[p - p0, q, :nVV] = K_{pq,cd}, gam[p - p0, q, nVV:] = K_{pq,kl} for one
    # block of p (each block has norb rows or more, so no GEMM is a GEMV)
    block_bytes = min(PARQUET_PP_INTEGRAL_BLOCK_BYTES,
                      rho_ee.itemsize * norb**4 // PARQUET_MIN_BLOCKS_PER_VERTEX)
    step = max(1, block_bytes // (rho_ee.itemsize * norb * max(1, npair)))
    gam = np.empty((min(step, norb), norb, npair))
    taken = np.empty((norb, npair))
    read_names = sorted({name for _, name, _ in terms})
    for p0 in range(0, norb, step):
        p1 = min(norb, p0 + step)
        for p in range(p0, p1):
            row = gam[p - p0]
            flat = {name: tensors[name][p].reshape(norb, norb * norb)
                    for name in read_names}
            for i, (coef, name, read) in enumerate(terms):
                # the columns are in range by construction: 'clip' only spares the copy
                np.take(flat[name], columns[read], axis=1, out=taken, mode='clip')
                if i == 0:
                    np.multiply(taken, coef, out=row)
                else:
                    taken *= coef
                    row += taken
            row *= norm
        g2d = gam[:p1 - p0].reshape((p1 - p0) * norb, npair)
        g_vv, g_oo = g2d[:, :nVV], g2d[:, nVV:]
        for rho, x, y in ((rho_ee, X1, Y1), (rho_hh, X2, Y2)):
            r2d = rho[p0:p1].reshape((p1 - p0) * norb, rho.shape[2])
            np.matmul(g_vv, x, out=r2d)
            _add_product(r2d, g_oo, y)
    return rho_ee, rho_hh


def regularized_inverse(d, eta):
    """Quack's SRG-regularized 1/d: (1 - exp(-2 eta d^2)) / d, elementwise.

    The regulator of every reducible vertex (R_*_Phi, Eq. (52) at omega = 0).
    It vanishes for eta = 0 and tends to 1/d as eta grows. An exactly zero d
    gives the limit 0 (the weight is 2 eta d to first order) at every finite
    eta, where Quack's (rho rho / d)(1 - exp(-2 eta d^2)) is 0/0 = NaN: the
    port departs from Quack there alone, so that a zero mode weighs nothing
    whether rounding puts it at 0 or a few ulps off (where 1 - exp rounds to
    0). Every other d is evaluated by the formula as written. At eta = inf,
    the bare 1/d, a zero d stays a pole (NaN). A NaN d (a NaN energy) gives
    NaN; every weight that is not finite is warned about
    (`events.ParquetNotFinite`) and flagged ('parquet_vertex_pole'), as the
    spin-orbital vertices count theirs. A d so large that d^2 overflows (|d|
    above about 1e154 Ha, eigenvalues of a diverging two-body loop) gives
    exp(-inf) = 0 and the finite 1/d, without a warning.

    Parameters
    ----------
    d : array_like
        Denominators omega - Omega (or omega + Omega), Hartree.
    eta : float
        The two-body flow parameter s_2b (Quack's eta_2b), >= 0.

    Returns
    -------
    ndarray, shaped as d.
    """
    if not eta >= 0.0:
        raise ValueError(f'the SRG flow parameter eta is {eta!r}; it must be >= 0')
    d = np.asarray(d, dtype=float)
    # past |d| ~ 1e154 (a diverging loop's eigenvalues) d * d overflows: exp(-inf) is 0
    # and w = 1/d, finite, so the overflow is no event; a weight that is not finite is
    # flagged below
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        w = (1.0 - np.exp(-2.0 * eta * d * d)) / d
    if eta < np.inf:
        zero = d == 0.0
        if zero.any():
            w = np.where(zero, 0.0, w)      # Quack's 0/0: the limit, 2 eta d -> 0
    poles = int(np.count_nonzero(~np.isfinite(w)))
    if poles:
        flag(FLAG_VERTEX_POLE,
             f'{poles} reducible-vertex weight(s) not finite (an energy that is not '
             'finite, or a zero denominator at eta = inf, the bare 1/d): the vertex is '
             'NaN there, as in Quack', ParquetNotFinite)
    return w


def eh_reducible_vertex(om, rho, eta, omega=0.0):
    """The eh reducible vertex Phi^eh_{pqrs}(omega) of one closed-shell eh
    channel, Eq. (37); at omega = 0 the static vertex of Eq. (39) with the
    regularizer of Eq. (52).

    Ports R_eh_singlet_Phi and R_eh_triplet_Phi (Quack, R_eh_*_Phi.f90),
    which are one formula: the weights R(omega - Omega_n) and
    -R(omega + Omega_n) of `regularized_inverse`, contracted by
    `eh_vertex_from_weights`. R_evParquet calls it at omega = 0.

    Parameters
    ----------
    om : ndarray, shape (nS,)
        The eh excitation energies Omega_n.
    rho : ndarray, shape (norb, norb, nS)
        M^eh_{pq,n} (`eh_effective_integrals`).
    eta : float
        The two-body SRG flow parameter s_2b, >= 0.
    omega : float
        The eh bosonic (transfer) frequency, Hartree; 0 is the static kernel
        approximation.

    Returns
    -------
    phi : ndarray, shape (norb,)*4
        Phi^eh[p, q, r, s].
    """
    omega = float(omega)
    om = np.asarray(om, dtype=float)
    w_res = regularized_inverse(omega - om, eta)
    w_anti = -regularized_inverse(omega + om, eta)
    return eh_vertex_from_weights(rho, w_res, w_anti)


def eh_vertex_from_weights(rho, w_res, w_anti):
    """Phi^eh_{pqrs} = sum_n M_{pr,n} M_{sq,n} w_res[n] + M_{rp,n} M_{qs,n} w_anti[n].

    The contraction of `eh_reducible_vertex` (R_eh_*_Phi) with the weights
    given: w_res = R(omega - Omega) and w_anti = -R(omega + Omega) at one
    frequency, or their average over several. One GEMM over the 2 nS stacked
    excitations, [(p, r), (q, s)], formed a block of p at a time (at most
    PARQUET_EH_VERTEX_BLOCK_BYTES and 1/PARQUET_MIN_BLOCKS_PER_VERTEX of Phi)
    and transposed into Phi[p, q, r, s]; the weighted (left) integrals of a
    block are stacked with it, the unweighted (right) ones once.

    Cost: 4 norb^4 nS flops; memory: Phi, the stacked right integrals
    (2 norb^2 nS) and one block with its left integrals. At norb = 138,
    nO = 5: 0.97 TFLOP, peak 3.4 GB (Phi 2.9 GB, the right integrals 0.2 GB,
    a block 0.25 GB).

    Parameters
    ----------
    rho : ndarray, shape (norb, norb, nS)
        M^eh_{pq,n}.
    w_res, w_anti : ndarray, shape (nS,)
        Weights of the resonant (M_pr M_sq) and antiresonant (M_rp M_qs)
        terms.

    Returns
    -------
    phi : ndarray, shape (norb,)*4
    """
    rho = _integrals(rho, 'rho')
    norb, _, nS = rho.shape
    w_res = _vector(w_res, nS, 'w_res')
    w_anti = _vector(w_anti, nS, 'w_anti')
    rho_t = rho.transpose(1, 0, 2)
    # right[q, s] = (M_sq, M_qs); left[p, r] = (M_pr w_res, M_rp w_anti), a block of p
    right = np.concatenate((rho_t, rho), axis=2).reshape(norb * norb, 2 * nS)
    phi = np.empty((norb,) * 4)
    block_bytes = min(PARQUET_EH_VERTEX_BLOCK_BYTES, phi.nbytes // PARQUET_MIN_BLOCKS_PER_VERTEX)
    step = max(1, block_bytes // (phi.itemsize * norb**3))
    for p0 in range(0, norb, step):
        p1 = min(norb, p0 + step)
        left = np.concatenate((rho[p0:p1] * w_res, rho_t[p0:p1] * w_anti), axis=2)
        block = left.reshape((p1 - p0) * norb, 2 * nS) @ right.T
        del left
        phi[p0:p1] = block.reshape(p1 - p0, norb, norb, norb).transpose(0, 2, 1, 3)
        del block               # before the next block is made: one block at a time
    return phi


def pp_reducible_vertex(om_ee, rho_ee, om_hh, rho_hh, eta, omega=0.0):
    """The pp reducible vertex Phi^pp_{pqrs}(omega) of one closed-shell pp
    channel, Eq. (43); at omega = 0 the static vertex of Eq. (45) with the
    regularizer of Eq. (52).

    Ports R_pp_singlet_Phi and R_pp_triplet_Phi (Quack, R_pp_*_Phi.f90),
    which are one formula: the weights R(omega - Omega^ee_m) and
    -R(omega - Omega^hh_m) of `regularized_inverse`, contracted by
    `pp_vertex_from_weights`. R_evParquet calls it at omega = 0.

    Parameters
    ----------
    om_ee : ndarray, shape (nVV,)
        Double-attachment eigenvalues Omega^ee_m.
    rho_ee : ndarray, shape (norb, norb, nVV)
        M^ee_{pq,m} (`pp_effective_integrals`).
    om_hh : ndarray, shape (nOO,)
        Double-removal eigenvalues Omega^hh_m, as the pp BSE returns them.
    rho_hh : ndarray, shape (norb, norb, nOO)
        M^hh_{pq,m}.
    eta : float
        The two-body SRG flow parameter s_2b, >= 0.
    omega : float
        The pp bosonic (total) frequency, Hartree; 0 is the static kernel
        approximation.

    Returns
    -------
    phi : ndarray, shape (norb,)*4
        Phi^pp[p, q, r, s].
    """
    omega = float(omega)
    w_ee = regularized_inverse(omega - np.asarray(om_ee, dtype=float), eta)
    w_hh = -regularized_inverse(omega - np.asarray(om_hh, dtype=float), eta)
    return pp_vertex_from_weights(rho_ee, w_ee, rho_hh, w_hh)


def pp_vertex_from_weights(rho_ee, w_ee, rho_hh, w_hh):
    """Phi^pp_{pqrs} = sum_m M^ee_{pq,m} M^ee_{rs,m} w_ee[m]
                     + sum_m M^hh_{pq,m} M^hh_{rs,m} w_hh[m].

    The contraction of `pp_reducible_vertex` (R_pp_*_Phi) with the weights
    given: w_ee = R(omega - Omega^ee) and w_hh = -R(omega - Omega^hh) at one
    frequency, or their average over several. Quack's two GEMMs in the
    (pq),(rs) layout: the ee one into Phi, a block of (pq) rows of the
    weighted M^ee at a time (at most PARQUET_PP_VERTEX_BLOCK_BYTES and
    1/PARQUET_MIN_BLOCKS_PER_VERTEX of Phi), the hh one added in place.

    Cost: 2 norb^4 (nVV + nOO) flops, the dearest step of a two-body
    iteration; memory: Phi and one block of the weighted M^ee. At norb = 138,
    nO = 5 (singlet): 6.5 TFLOP, peak 3.1 GB (Phi 2.9 GB; the whole weighted
    M^ee would be 1.36 GB).

    Parameters
    ----------
    rho_ee : ndarray, shape (norb, norb, nVV)
    w_ee : ndarray, shape (nVV,)
    rho_hh : ndarray, shape (norb, norb, nOO)
    w_hh : ndarray, shape (nOO,)

    Returns
    -------
    phi : ndarray, shape (norb,)*4
    """
    rho_ee = _integrals(rho_ee, 'rho_ee')
    rho_hh = _integrals(rho_hh, 'rho_hh')
    norb = rho_ee.shape[0]
    if rho_hh.shape[:2] != (norb, norb):
        raise ValueError(f'rho_hh is {rho_hh.shape}, rho_ee {rho_ee.shape}: '
                         'they must share their orbital axes')
    w_ee = _vector(w_ee, rho_ee.shape[2], 'w_ee')
    w_hh = _vector(w_hh, rho_hh.shape[2], 'w_hh')
    r_ee = rho_ee.reshape(norb * norb, rho_ee.shape[2])
    r_hh = rho_hh.reshape(norb * norb, rho_hh.shape[2])
    phi = np.empty((norb,) * 4)
    phi2d = phi.reshape(norb * norb, norb * norb)
    # equal blocks of (pq) rows, at most the cap and never one row: numpy takes
    # a one-row product as a GEMV, which sums in another order than the GEMM
    n_rows = norb * norb
    block_bytes = min(PARQUET_PP_VERTEX_BLOCK_BYTES, phi.nbytes // PARQUET_MIN_BLOCKS_PER_VERTEX)
    cap = max(2, block_bytes // (phi.itemsize * max(1, r_ee.shape[1])))
    n_blocks = max(1, min(-(-n_rows // cap), n_rows // 2))
    bounds = [n_rows * i // n_blocks for i in range(n_blocks + 1)]
    for r0, r1 in zip(bounds[:-1], bounds[1:]):
        np.matmul(r_ee[r0:r1] * w_ee, r_ee.T, out=phi2d[r0:r1])
    _add_product(phi2d, r_hh * w_hh, r_hh.T)
    return phi


def _channel_terms(table, spin):
    """The kernel terms of `spin` in `table`; an unknown channel is refused."""
    if spin not in table:
        raise ValueError(f'unknown closed-shell spin channel {spin!r}; '
                         f'one of {EH_SPINS}')
    return table[spin]


def _four_index(t, name):
    """`t` as a float64 (n, n, n, n) array (no copy when it already is)."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 4 or len(set(t.shape)) != 1:
        raise ValueError(f'{name} is {t.shape}; it must be (norb, norb, norb, norb)')
    return t


def _tensors(eri, vertices, norb):
    """{'eri': eri, 'eh_singlet': ..., ...}: the tensors a kernel term reads."""
    vertices = tuple(vertices)
    if len(vertices) != len(VERTEX_NAMES):
        raise ValueError(f'vertices holds {len(vertices)} tensors; a closed shell '
                         f'has {len(VERTEX_NAMES)}: {VERTEX_NAMES}')
    tensors = {'eri': eri}
    for name, t in zip(VERTEX_NAMES, vertices):
        t = _four_index(t, name)
        if t.shape[0] != norb:
            raise ValueError(f'{name} is {t.shape}, eri {eri.shape}')
        tensors[name] = t
    return tensors


def _occupation(norb, nO):
    """(nV, nS) of nO doubly occupied orbitals out of norb."""
    if not 0 < nO < norb:
        raise ValueError(f'nO = {nO} of {norb} orbitals: both the occupied and '
                         'the virtual space must be non-empty')
    return norb - nO, nO * (norb - nO)


def _matrix(m, shape, name):
    """`m` as a float64 array of `shape`, refused otherwise."""
    m = np.asarray(m, dtype=float)
    if m.shape != shape:
        raise ValueError(f'{name} is {m.shape}; expected {shape}')
    return m


def _vector(w, n, name):
    """`w` as a float64 (n,) array, refused otherwise."""
    w = np.asarray(w, dtype=float)
    if w.shape != (n,):
        raise ValueError(f'{name} is {w.shape}; expected ({n},)')
    return w


def _integrals(rho, name):
    """`rho` as a float64 (norb, norb, n) array, refused otherwise."""
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 3 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f'{name} is {rho.shape}; it must be (norb, norb, n)')
    return rho


def _transposed_operand(m):
    """(a, trans) with a Fortran-ordered and op(a) = m^T for BLAS, without a
    copy when m is C- or Fortran-contiguous."""
    if m.flags.c_contiguous:
        return m.T, 0
    if m.flags.f_contiguous:
        return m, 1
    return np.ascontiguousarray(m).T, 0


def _add_product(c2d, a2d, b2d):
    """c2d += a2d @ b2d in place, c2d C-contiguous: BLAS dgemm with beta = 1
    on c2d^T = b2d^T a2d^T (Fortran order), so no c2d-sized temporary."""
    if a2d.shape[1] == 0 or c2d.size == 0:
        return
    first, trans_first = _transposed_operand(b2d)
    second, trans_second = _transposed_operand(a2d)
    out = blas.dgemm(1.0, first, second, beta=1.0, c=c2d.T, trans_a=trans_first,
                     trans_b=trans_second, overwrite_c=1)
    if not np.may_share_memory(out, c2d):
        c2d[...] = out.T
