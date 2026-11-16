"""The two-body part of the spin-orbital parquet loop (Quack's G branch).

One two-body iteration of the parquet approximation of Marie and Loos,
J. Chem. Phys. 163, 194115 (2025), Sec. IV B, in spin orbitals and in the
static kernel approximation, ported routine by routine from the paper-era
Quack (git e3528a40): its G_evParquet two-body loop, lines 209-421.

Given the reducible vertices i*Phi^eh and i*Phi^pp of the previous iteration,
the irreducible kernels of the parquet approximation (Eq. 47)

    i Gamma^eh_pqrs = <pq||rs> - i Phi^eh_pqsr + i Phi^pp_pqrs,
    i Gamma^pp_pqrs = <pq||rs> + i Phi^eh_pqrs - i Phi^eh_pqsr,

taken at omega = 0, define two Bethe-Salpeter problems.

The eh BSE (Eqs. 34-35) over the 1h1p pairs ia,

    A_ia,jb = (e_a - e_i) d_ij d_ab + i Gamma^eh_ajib,  B_ia,jb = i Gamma^eh_abij,

is solved as Casida's problem (A - B)^1/2 (A + B) (A - B)^1/2 Z = Om^2 Z, with
X + Y = Om^-1/2 Z^T (A - B)^1/2 and X - Y = Om^1/2 Z^T (A - B)^-1/2 (phGLR).
The pp BSE (Eqs. 40-41) over the 2p pairs a < b and the 2h pairs i < j,

    C_ab,cd = (e_a + e_b) d_ac d_bd + i Gamma^pp_abcd,  B_ab,ij = i Gamma^pp_abij,
    D_ij,kl = -(e_i + e_j) d_ik d_jl + i Gamma^pp_ijkl,
    [[C, B], [-B^T, -D]] (X; Y) = (X; Y) Om,

is solved as a general eigenproblem whose eigenvalues above zero form the 2p
(ee) branch and the others the 2h (hh) branch, each branch normalized in the
metric diag(1, -1) (ppGLR, sort_ppRPA). The eigenvectors give the generalized
effective integrals M^eh_pq,n, M^ee_pq,m and M^hh_pq,m (Eqs. 38 and 44), and
these the reducible vertices of the next iteration (Eqs. 37 and 43, with the
energy-dependent regularizer of Eq. 52):

    i Phi^eh_pqrs(w) = sum_n [M_pr,n M_sq,n g(w - Om_n) - M_rp,n M_qs,n g(w + Om_n)],
    i Phi^pp_pqrs(w) = sum_m M^ee_pq,m M^ee_rs,m g(w - Om^ee_m)
                     - sum_m M^hh_pq,m M^hh_rs,m g(w - Om^hh_m),
    g(x) = (1 - exp(-2 eta x^2)) / x,

and g(0) = 0, its limit, where Quack's formula is 0/0 (`regularized_inverse`).

Quack's G_eh_Phi and G_pp_Phi are the static vertices, w = 0. The frequency
w enters as in the restricted R_eh_singlet_Phi and R_pp_singlet_Phi, so a
frequency average of the vertex is an average of the pole weights g: build
the weights yourself and hand them to `eh_vertex_from_weights` or
`pp_vertex_from_weights`. Through the driver's `vertex_builder` hook a builder
of the default protocol gives one tensor per source channel, inserted at
several places, the eh tensor in both the eh and the pp kernels (and their
effective integrals), the pp tensor in the eh kernel and its effective
integrals (Eq. 47), and sees neither eps nor nO; a per-insertion builder
(per_insertion = True) gives each insertion role its own tensor, the eh
vertex crossed into the eh kernel ('phbar_eh') and into the pp kernel
('eh_pp', its direct and swapped terms alike, which exchange turns into each
other), the pp vertex into the eh kernel ('pp_eh'), and is handed eps and nO
(docs/methods/parquet.md, One tensor per insertion). The functions here read
whichever tensors they are handed.

The loop around one iteration (damping, DIIS, the convergence test, the
one-body update) is the driver's (`spin_orbital.two_body_spectra_so` chains
the BSEs and their integrals, its vertex builder the vertices).

Conventions
-----------
Arrays are indexed exactly as Quack's (0-based): eri[p, q, r, s] = <pq|rs>
(MO, physicist, not antisymmetrized) over spin orbitals, occupied first (any
spin order: a closed shell interleaved alpha, beta keeps energy order, as
Quack's GHF does); eps[p] the one-body energies of the BSE diagonals;
vertices a `spectra.SpinOrbitalVertices` (eh, pp), each [p, q, r, s], the
real i*Phi of the paper. Pair orders are those of `pairs`: ia with i outer
(eh_pairs), a < b and i < j row by row (pp_pairs, 'spin-orbital'). The eh
eigenvectors are rows, XpY[n, ia] and XmY[n, ia], as phGLR returns them; the
pp eigenvectors are columns, X1[ab, m] and Y1[ij, m] of the 2p branch,
X2[ab, m] and Y2[ij, m] of the 2h branch (Quack's names: X2 is the 2p part of
a 2h eigenvector, the paper's Y^hh, and Y2 its 2h part, the paper's X^hh).
The 2h energies om_hh are the signed eigenvalues (near e_i + e_j < 0). The
effective integrals are rho[p, q, n] over every orbital pair. Quack's frozen
core and virtuals (nC, nR) are not carried over: every orbital is active.

Faithful to Quack, quirks included
----------------------------------
- phGLR takes sqrt(Om^2) where phRLR takes sqrt(|Om^2|): an imaginary eh
  excitation gives NaN (its rows of X +- Y, and Ec) here, a real |Om| in the
  restricted code. A - B not positive definite gives NaN in (A - B)^1/2 in
  both, so the whole eh solution is NaN. Both are flagged.
- sort_ppRPA splits the pp spectrum at the eigenvalue 0 (Quack's eF = 0),
  not by the sign of the metric norm; beyond nVV positive (nOO other)
  eigenvalues the rest are dropped in dgeev's output order, which differs
  between LAPACK libraries (so does Quack's choice), and missing ones are
  left at Om = 0 with eigenvector columns Quack leaves uninitialized (zero
  here). dgeev's imaginary parts are discarded: a complex pair enters as its
  real part twice, with the real and imaginary parts of its eigenvector,
  whose metric overlap is indefinite. Either way the Lowdin step meets a
  zero or negative overlap and that branch's eigenvectors are NaN, in Quack
  and here. All of these are flagged.
- Each pp branch is M-orthonormalized by the Lowdin S^-1/2 of its metric
  overlap over the whole branch, as sort_ppRPA does; every symmetric
  eigensolve reads the upper triangle only, as Quack's dsyev('V', 'U').
- Under the TDA no B block is built: the block functions return None for
  it, as the closed-shell ones do (Quack adds an uninitialized Gamma_B to
  its B, which the TDA solve never reads).
- An empty pair space is kept as Quack keeps it: with one occupied spin
  orbital (nO = 1, any one-electron system) the pp problem has no 2h pair,
  with one virtual (nV = 1) no 2p pair; that branch has no roots, its
  effective integrals no column, and Quack's G_evParquet runs on, whatever
  the BLAS: its G_pp_screened_integral sums in loops (unlike the closed-shell
  R_pp_triplet_screened_integral, whose dgemm with the empty pair count as
  a leading dimension ends the run under Accelerate or the reference BLAS;
  OpenBLAS and MKL print the argument error and skip the empty product).
- A collinear open-shell reference (a UHF, or a GHF that stayed collinear,
  with unpaired electrons) has an exact zero mode in the full eh problem:
  rotating its spin axis (and, for a spatially degenerate open shell, its
  open orbitals) costs no energy, so A - B and A + B share an eigenvalue
  that only the SCF residual moves off zero (1e-9 Ha at conv_tol 1e-12,
  either sign). Below zero, A - B is not positive definite and the whole
  eh solution is NaN, in Quack and here (OH UHF/6-31G); above zero, Om of
  1e-7 Ha and X + Y growing as Om^-1/2 take over the vertices and the
  self-energy, so the energies are finite and meaningless (NH2 UHF/6-31G
  FLEX: up to 18 eV from the TDA; Quack, on the other side by roundoff,
  NaN). An eh root below PARQUET_EH_ZERO_MODE_TOL is flagged
  'parquet_eh_zero_mode'. The TDA has no Om^-1/2 (A alone, X + Y = U^T)
  and runs open shells as Quack does; a one-electron system's spin flip
  is an exact zero root of A there, which roundoff can put just below
  zero, so the TDA flags a negative root only below
  -PARQUET_EH_ZERO_MODE_TOL ('parquet_eh_tda_negative_excitations', which
  Quack does not print). The TDA does not take an open shell's soft modes
  away, though: they are small rather than zero there, and nothing flags
  them (OH UHF/6-31G: 7.5e-4 Ha for the rotation of its pi hole about the
  bond, 3.0e-3 and 7.4e-3 Ha for spin flips), and the static vertex weighs
  such a root by g(Om) = (1 - exp(-2 eta Om^2)) / Om, about 1/Om once
  2 eta Om^2 nears 1 (1338 Ha^-1 for OH's lowest): OH's FLEX eh vertex
  grows from 2.2 at s_2b = 50 to 40, 347 and 516 at 1e4, 1e6 and 1e8
  (water's stays at 1.85), and its osPA diverges at s_2b = 1e6 and 1e8.

Eigenvectors are defined up to sign and, within a degenerate set (a closed
shell's triplets come in threes), up to rotation; every vertex built from
them is not, so two eigensolvers agree on Phi, not on X and Y.

Cost at N spin orbitals (nS = nO nV, nVV = nV(nV-1)/2, nOO = nO(nO-1)/2):
the BSE blocks are nS^2 and nVV^2 matrices, the eh solve O(nS^3), the pp solve
O(nVV^3) (dgeev, the largest step without the TDA), the effective integrals
N^2 nS and N^2 (nVV + nOO) arrays from GEMMs of 4 N^2 nS^2 and
2 N^2 (nVV + nOO)^2 flops, and each vertex one N^4 array from GEMMs of
4 N^4 nS (eh) and 2 N^4 (nVV + nOO) (pp) flops, built in blocks of at most
PARQUET_SO_BLOCK_BYTES so no second N^4 array is held.

The numerical events are warned about and raised as the named flags of
`events` (ParquetBSEUnstable, ParquetNotFinite), under the names the
closed-shell modules use for the same events.
"""
from typing import NamedTuple

import numpy as np
import scipy.linalg as la
from scipy.linalg import lapack

from src.Base.numerical_flags import flag
from src.Base.numerics.constants import (PARQUET_EH_ZERO_MODE_TOL, PARQUET_EIGH_DRIVER,
                                         PARQUET_LOWDIN_SMALL_EIGENVALUE,
                                         PARQUET_PP_BRANCH_SPLIT, PARQUET_PP_COMPLEX_TOL,
                                         PARQUET_PP_EC_CONSISTENCY_TOL, PARQUET_SO_BLOCK_BYTES)
from src.SingleReference.Parquet.events import (FLAG_BSE_DIAGONALIZATION_FAILED,
                                                FLAG_BSE_NOT_FINITE,
                                                FLAG_EH_A_MINUS_B_NOT_POSITIVE,
                                                FLAG_EH_NEGATIVE_EXCITATIONS,
                                                FLAG_EH_TDA_NEGATIVE_EXCITATIONS, FLAG_EH_ZERO_MODE,
                                                FLAG_PP_BRANCH_COUNT, FLAG_PP_COMPLEX_ROOTS,
                                                FLAG_PP_DGEEV_FAILED, FLAG_PP_EC_INCONSISTENT,
                                                FLAG_PP_LOWDIN_SMALL_EIGENVALUE, FLAG_VERTEX_POLE,
                                                ParquetBSEUnstable, ParquetNotFinite)
from src.SingleReference.Parquet.pairs import eh_pairs, pp_pairs


class EhBSESolution(NamedTuple):
    """phGLR's output: excitation energies om (nS,), ascending; eigenvectors
    as rows, XpY[n, ia] = (X + Y) and XmY[n, ia] = (X - Y); the correlation
    energy ec = (sum om - Tr A) / 2."""
    om: np.ndarray
    XpY: np.ndarray
    XmY: np.ndarray
    ec: float


class PPBSESolution(NamedTuple):
    """ppGLR's output: the 2p branch om_ee (nVV,), X1 (nVV, nVV), Y1 (nOO, nVV)
    and the 2h branch om_hh (nOO,), X2 (nVV, nOO), Y2 (nOO, nOO), each
    ascending, eigenvectors as columns; the correlation energy
    ec = (sum om_ee - sum om_hh - Tr C - Tr D) / 2."""
    om_ee: np.ndarray
    X1: np.ndarray
    Y1: np.ndarray
    om_hh: np.ndarray
    X2: np.ndarray
    Y2: np.ndarray
    ec: float


def _orbital_count(nO, *tensors):
    """N of the (N,)*k arrays handed in, after checking them against each
    other and 0 < nO < N."""
    n = tensors[0].shape[0]
    for t in tensors:
        if t.shape != (n,) * t.ndim:
            raise ValueError(f'expected arrays over {n} orbitals on every axis, '
                             f'got shape {t.shape}')
    if not 0 < nO < n:
        raise ValueError(f'nO = {nO} occupied of {n} spin orbitals: need 0 < nO < N')
    return n


def _vertex_arrays(vertices, n=None):
    """The eh and pp vertices as float arrays, checked to be (n,)*4."""
    eh = np.asarray(vertices.eh, dtype=float)
    pp = np.asarray(vertices.pp, dtype=float)
    for name, t in (('eh', eh), ('pp', pp)):
        if t.ndim != 4 or (n is not None and t.shape != (n,) * 4):
            raise ValueError(f'the {name} vertex has shape {t.shape}; expected ({n},)*4')
    return eh, pp


def _pair_block(t, rows, cols):
    """t[p, q, r, s] - t[p, q, s, r] over the pairs rows = (p, q) and
    cols = (r, s): an antisymmetrized pair block, gathered without any
    N^4 temporary."""
    p, q = rows
    r, s = cols
    out = t[p[:, None], q[:, None], r[None, :], s[None, :]]
    out -= t[p[:, None], q[:, None], s[None, :], r[None, :]]
    return out


def _nan_eigenpairs(n):
    return np.full(n, np.nan), np.full((n, n), np.nan)


def _eigh_upper(a, what, stacklevel=3):
    """Eigenpairs of the symmetric `a` from its upper triangle, ascending,
    eigenvectors as columns: Quack's diagonalize_matrix (dsyev('V', 'U')).
    NaN or inf in that triangle ('parquet_bse_not_finite'), or a LAPACK
    failure ('parquet_bse_diagonalization_failed'), gives NaN, flagged
    (`stacklevel` is handed to `flag`); below the diagonal, which dsyev('V',
    'U') never reads, it is passed over, as in Quack and the closed-shell
    `restricted_bse._eigh_upper`."""
    n = a.shape[0]
    finite = np.isfinite(a)
    if not finite.all() and np.triu(~finite).any():
        flag(FLAG_BSE_NOT_FINITE, f'{what}: the matrix is not finite; its eigenpairs are NaN',
             ParquetNotFinite, stacklevel=stacklevel)
        return _nan_eigenpairs(n)
    del finite
    try:
        w, v = la.eigh(a, lower=False, driver=PARQUET_EIGH_DRIVER, check_finite=False)
    except la.LinAlgError as err:
        flag(FLAG_BSE_DIAGONALIZATION_FAILED, f'{what}: {err}; its eigenpairs are NaN',
             ParquetBSEUnstable, stacklevel=stacklevel)
        return _nan_eigenpairs(n)
    return w, v


def eh_bse_blocks_so(eps, eri, nO, tda):
    """Bare eh BSE blocks (A, B) over spin orbitals: the eh RPA with exchange.

    Ports Quack's phGLR_A and phGLR_B with dRPA = .false. and lambda = 1
    (Eqs. 35a-b with the bare kernel <pq||rs>):

        A[ia, jb] = (e_a - e_i) d_ij d_ab + <bi|ja> - <bj|ai>,
        B[ia, jb] = <ij|ab> - <ij|ba>,

    element by element as Quack indexes the ERIs, and the diagonal added as
    (A + e_a) - e_i, Quack's rounding.

    Parameters
    ----------
    eps : ndarray (N,)
        One-body energies of the BSE diagonal (Quack's eOld).
    eri : ndarray (N, N, N, N)
        <pq|rs>, not antisymmetrized.
    nO : int
        Occupied spin orbitals (the first nO).
    tda : bool
        Tamm-Dancoff: B is not built (None).

    Returns
    -------
    A : ndarray (nS, nS)
        Over the eh pairs ia = i nV + (a - nO).
    B : ndarray (nS, nS), or None when tda
    """
    eps = np.asarray(eps, dtype=float)
    eri = np.asarray(eri, dtype=float)
    n = _orbital_count(nO, eps, eri)
    nV = n - nO
    nS = nO * nV
    o, v = slice(0, nO), slice(nO, n)
    A = (eri[v, o, o, v].transpose(1, 3, 2, 0)
         - eri[v, o, v, o].transpose(3, 2, 1, 0)).reshape(nS, nS)
    i, a = eh_pairs(nO, nV)
    d = np.arange(nS)
    A[d, d] = A[d, d] + eps[a] - eps[i]
    if tda:
        B = None
    else:
        B = (eri[o, o, v, v].transpose(0, 2, 1, 3)
             - eri[o, o, v, v].transpose(0, 3, 1, 2)).reshape(nS, nS)
    return A, B


def eh_kernel_blocks_so(vertices, nO, tda):
    """Vertex part of the eh BSE kernel, (Gamma_A, Gamma_B).

    Ports Quack's G_eh_Gamma_A and G_eh_Gamma_B: the Phi terms of
    i Gamma^eh_ajib and i Gamma^eh_abij (Eqs. 35 and 47a),

        Gamma_A[ia, jb] = -Phi^eh_ajbi + Phi^pp_ajib,
        Gamma_B[ia, jb] = -Phi^eh_abji + Phi^pp_abij,

    added to the bare blocks by the caller (A + Gamma_A, B + Gamma_B), as
    G_evParquet does. Zero vertices (the first iteration, Quack's FLEX
    start) give zero blocks.

    Parameters
    ----------
    vertices : SpinOrbitalVertices
        The reducible vertices of the previous iteration, each (N,)*4.
    nO : int
        Occupied spin orbitals.
    tda : bool
        Tamm-Dancoff: Gamma_B is not built (None).

    Returns
    -------
    Gamma_A : ndarray (nS, nS)
    Gamma_B : ndarray (nS, nS), or None when tda
    """
    eh, pp = _vertex_arrays(vertices)
    n = _orbital_count(nO, eh, pp)
    nS = nO * (n - nO)
    o, v = slice(0, nO), slice(nO, n)
    gam_a = (pp[v, o, o, v].transpose(2, 0, 1, 3)
             - eh[v, o, v, o].transpose(3, 0, 1, 2)).reshape(nS, nS)
    if tda:
        gam_b = None
    else:
        gam_b = (pp[v, v, o, o].transpose(2, 0, 3, 1)
                 - eh[v, v, o, o].transpose(3, 0, 2, 1)).reshape(nS, nS)
    return gam_a, gam_b


def solve_eh_bse_so(A, B, tda):
    """Eigenpairs of the eh BSE: Quack's phGLR.

    Under the TDA, A = U Om U^T and X + Y = X - Y = U^T. Otherwise Casida's
    form (Eq. 34): with (A - B) = U lam U^T, (A - B)^+-1/2 = U lam^+-1/2 U^T,
    Z diagonalizing (A - B)^1/2 (A + B) (A - B)^1/2 = Z Om^2 Z^T,

        XpY = Om^-1/2 Z^T (A - B)^1/2,    XmY = Om^1/2 Z^T (A - B)^-1/2,

    rows n, Om = sqrt(Om^2) ascending. Each symmetric eigensolve reads the
    upper triangle, as Quack's dsyev('V', 'U'), and A + B enters its product
    whole: a NaN or inf in A's upper triangle (TDA), or anywhere in A or B
    (otherwise), gives NaN eigenpairs flagged 'parquet_bse_not_finite', as
    in the closed-shell solver, except where A - B, finite in its upper
    triangle, is not positive definite: the product is then not
    diagonalized, and only that is flagged (below).

    Quack's instabilities are kept: A - B not positive definite makes
    (A - B)^1/2 NaN, hence the whole solution; a negative Om^2 makes that
    Om, its rows of XpY and XmY, and Ec NaN (phGLR's sqrt; phRLR would take
    |Om^2|). Each is warned as `ParquetBSEUnstable` under the named flag
    'parquet_eh_a_minus_b_not_positive' or 'parquet_eh_negative_excitations',
    where Quack prints its warning. A root that passes both but lies below
    PARQUET_EH_ZERO_MODE_TOL, whose X + Y ~ Om^-1/2 then takes over the
    vertices, is kept and warned as well ('parquet_eh_zero_mode', which
    Quack does not print): the exact zero mode of a collinear open shell
    (module docstring), or an instability. Under the TDA a root below
    -PARQUET_EH_ZERO_MODE_TOL, which Quack lets pass silently, is kept and
    flagged as the closed-shell solver flags a negative one
    ('parquet_eh_tda_negative_excitations': a stretched RHF's triplet
    instability, H2/6-31G at 2.0 Angstrom -0.0633 Ha); a root between that
    and zero is not, since there lies a one-electron system's spin flip, an
    exact zero that roundoff puts on either side (the H atom in 6-31G:
    -1.4e-15 Ha). The TDA has no zero-mode test: its amplitudes do not grow
    with 1/Om, though a small root still weighs about 1/Om in a weakly
    regularized vertex (module docstring).

    Parameters
    ----------
    A, B : ndarray (nS, nS)
        Bare blocks plus the vertex kernel (eh_bse_blocks_so +
        eh_kernel_blocks_so); B may be None when tda.
    tda : bool
        Tamm-Dancoff: B is not read.

    Returns
    -------
    EhBSESolution
        om (nS,), XpY (nS, nS), XmY (nS, nS), ec = (sum om - Tr A) / 2.
    """
    A = np.asarray(A, dtype=float)
    nS = A.shape[0]
    if A.shape != (nS, nS):
        raise ValueError(f'A has shape {A.shape}; expected a square matrix')
    if tda:
        om, U = _eigh_upper(A, 'eh BSE (TDA)')
        XpY = np.ascontiguousarray(U.T)
        XmY = XpY.copy()
        negative = om < -PARQUET_EH_ZERO_MODE_TOL
        if negative.any():
            flag(FLAG_EH_TDA_NEGATIVE_EXCITATIONS,
                 f'spin-orbital eh BSE (TDA): {int(negative.sum())} negative excitation '
                 f'energies below -PARQUET_EH_ZERO_MODE_TOL = -{PARQUET_EH_ZERO_MODE_TOL:g} '
                 f'Ha, the lowest {np.min(om[negative]):.6f}; kept as they are (Quack does '
                 f'not check)', ParquetBSEUnstable)
    else:
        B = np.asarray(B, dtype=float)
        if B.shape != (nS, nS):
            raise ValueError(f'B has shape {B.shape}; expected {(nS, nS)}')
        ApB = A + B
        AmB = A - B
        lam, U = _eigh_upper(AmB, 'eh BSE, A - B')
        if np.min(lam, initial=np.inf) < 0.0:
            flag(FLAG_EH_A_MINUS_B_NOT_POSITIVE,
                 f'spin-orbital eh BSE: A - B is not positive definite (lowest eigenvalue '
                 f'{np.min(lam):.3e}); (A - B)^1/2 is NaN and so is the whole eh solution, '
                 f'as in Quack', ParquetBSEUnstable)
        with np.errstate(invalid='ignore', divide='ignore'):
            sq = np.sqrt(lam)
            AmBSq = U @ (sq[:, None] * U.T)
            AmBIv = U @ ((1.0 / sq)[:, None] * U.T)
            Zm = AmBSq @ (ApB @ AmBSq)
        if np.isfinite(lam).all() and np.min(lam, initial=np.inf) >= 0.0:
            # A - B passed (its upper triangle); A + B is read whole, as Quack's
            # products read it, so a NaN or inf anywhere in A or B reaches Zm and
            # is flagged by this diagonalization, as in the closed-shell solver
            om2, Z = _eigh_upper(Zm, 'eh BSE, (A - B)^1/2 (A + B) (A - B)^1/2')
        else:
            om2, Z = _nan_eigenpairs(nS)        # flagged above, at A - B
        if np.min(om2, initial=np.inf) < 0.0:
            flag(FLAG_EH_NEGATIVE_EXCITATIONS,
                 f'spin-orbital eh BSE: {int(np.sum(om2 < 0.0))} imaginary excitation(s) '
                 f'(lowest Om^2 {np.min(om2):.3e}); their Om, X + Y, X - Y and Ec are NaN, '
                 f'as in Quack (phGLR takes sqrt(Om^2))', ParquetBSEUnstable)
        with np.errstate(invalid='ignore', divide='ignore'):
            om = np.sqrt(om2)
            XpY = (Z.T @ AmBSq) * (1.0 / np.sqrt(om))[:, None]
            XmY = (Z.T @ AmBIv) * np.sqrt(om)[:, None]
        with np.errstate(invalid='ignore'):
            zero = om < PARQUET_EH_ZERO_MODE_TOL
        if zero.any():
            flag(FLAG_EH_ZERO_MODE,
                 f'spin-orbital eh BSE: {int(zero.sum())} excitation energy(ies) below '
                 f'PARQUET_EH_ZERO_MODE_TOL = {PARQUET_EH_ZERO_MODE_TOL:g} Ha (lowest '
                 f'{np.min(om[zero]):.3e} Ha), a zero mode (the spin rotation of a collinear '
                 f'open shell, or an instability) whose X + Y ~ Om^-1/2 takes over the vertices '
                 f'and the self-energy; kept, as in Quack. Open shells run in the TDA '
                 f'(tda_eh=True)', ParquetBSEUnstable)
    ec = 0.5 * (np.sum(om) - np.trace(A))
    return EhBSESolution(om, XpY, XmY, float(ec))


def pp_bse_blocks_so(eps, eri, nO, tda):
    """Bare pp BSE blocks (B, C, D) over spin orbitals: the pp RPA.

    Ports Quack's ppGLR_B, ppGLR_C and ppGLR_D with lambda = 1 and eF = 0
    (Eqs. 41a-c with the bare kernel <pq||rs>):

        B[ab, ij] = <ab|ij> - <ab|ji>,
        C[ab, cd] = (e_a + e_b) d_ac d_bd + <ab|cd> - <ab|dc>,
        D[ij, kl] = -(e_i + e_j) d_ik d_jl + <ij|kl> - <ij|lk>,

    over the pairs a < b and i < j in pp_pairs order.

    Parameters
    ----------
    eps : ndarray (N,)
        One-body energies of the BSE diagonal.
    eri : ndarray (N, N, N, N)
        <pq|rs>, not antisymmetrized.
    nO : int
        Occupied spin orbitals.
    tda : bool
        Tamm-Dancoff: B is not built (None).

    Returns
    -------
    B : ndarray (nVV, nOO), or None when tda
    C : ndarray (nVV, nVV)
    D : ndarray (nOO, nOO)
        Either pair space may be empty (nO = 1: nOO = 0; nV = 1: nVV = 0).
    """
    eps = np.asarray(eps, dtype=float)
    eri = np.asarray(eri, dtype=float)
    n = _orbital_count(nO, eps, eri)
    vv = pp_pairs(nO, n, 'spin-orbital')
    oo = pp_pairs(0, nO, 'spin-orbital')
    nVV, nOO = len(vv[0]), len(oo[0])
    C = _pair_block(eri, vv, vv)
    dv = np.arange(nVV)
    C[dv, dv] = C[dv, dv] + (eps[vv[0]] + eps[vv[1]])
    D = _pair_block(eri, oo, oo)
    do = np.arange(nOO)
    D[do, do] = D[do, do] - (eps[oo[0]] + eps[oo[1]])
    B = None if tda else _pair_block(eri, vv, oo)
    return B, C, D


def pp_kernel_blocks_so(vertices, nO, tda):
    """Vertex part of the pp BSE kernel, (Gamma_B, Gamma_C, Gamma_D).

    Ports Quack's G_pp_Gamma_B, G_pp_Gamma_C and G_pp_Gamma_D: the Phi^eh
    terms of i Gamma^pp (Eqs. 41 and 47b; Phi^pp does not enter),

        Gamma_B[ab, ij] = Phi^eh_abij - Phi^eh_abji,
        Gamma_C[ab, cd] = Phi^eh_abcd - Phi^eh_abdc,
        Gamma_D[ij, kl] = Phi^eh_ijkl - Phi^eh_ijlk,

    added to the bare blocks by the caller.

    Parameters
    ----------
    vertices : SpinOrbitalVertices
        The reducible vertices of the previous iteration (pp is not read).
    nO : int
        Occupied spin orbitals.
    tda : bool
        Tamm-Dancoff: Gamma_B is not built (None).

    Returns
    -------
    Gamma_B : ndarray (nVV, nOO), or None when tda
    Gamma_C : ndarray (nVV, nVV)
    Gamma_D : ndarray (nOO, nOO)
    """
    eh = np.asarray(vertices.eh, dtype=float)
    n = _orbital_count(nO, eh)
    vv = pp_pairs(nO, n, 'spin-orbital')
    oo = pp_pairs(0, nO, 'spin-orbital')
    gam_c = _pair_block(eh, vv, vv)
    gam_d = _pair_block(eh, oo, oo)
    gam_b = None if tda else _pair_block(eh, vv, oo)
    return gam_b, gam_c, gam_d


def _lowdin(S, branch):
    """S^-1/2 of a pp branch's metric overlap: Quack's orthogonalize_matrix
    with ortho_type 1 (dsyev, 1/sqrt of each eigenvalue, U D U^T)."""
    w, U = _eigh_upper(S, f'pp BSE {branch} metric overlap', stacklevel=5)
    small = w < PARQUET_LOWDIN_SMALL_EIGENVALUE
    if small.any():
        flag(FLAG_PP_LOWDIN_SMALL_EIGENVALUE,
             f'spin-orbital pp BSE: {int(small.sum())} eigenvalue(s) of the {branch} metric '
             f'overlap below {PARQUET_LOWDIN_SMALL_EIGENVALUE:g} (lowest {np.min(w):.3e}) in the '
             f'Lowdin normalization; kept as Quack does (NaN below zero)',
             ParquetBSEUnstable, stacklevel=4)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        d = 1.0 / np.sqrt(w)
        return U @ (d[:, None] * U.T)


def _pp_general(B, C, D):
    """The pp BSE without the TDA: Quack's ppGLR (dgeev of
    [[C, B], [-B^T, -D]]) and sort_ppRPA (the sign split, the ascending
    sort, the Lowdin normalization of each branch in the metric
    diag(1, -1))."""
    nVV, nOO = B.shape
    nPP = nVV + nOO
    if nPP == 0:
        # no pp pair at all (one occupied and one virtual spin orbital): no root
        return (np.zeros(0), np.zeros((0, 0)), np.zeros((0, 0)),
                np.zeros(0), np.zeros((0, 0)), np.zeros((0, 0)))
    M = np.empty((nPP, nPP), order='F')
    M[:nVV, :nVV] = C
    M[nVV:, nVV:] = -D
    M[:nVV, nVV:] = B
    M[nVV:, :nVV] = -B.T
    if not np.isfinite(M).all():
        flag(FLAG_BSE_NOT_FINITE,
             'spin-orbital pp BSE: the pp matrix is not finite; its eigenpairs are NaN',
             ParquetNotFinite, stacklevel=3)
        wr, wi, vr = np.full(nPP, np.nan), np.zeros(nPP), np.full((nPP, nPP), np.nan)
    else:
        work, info = lapack.dgeev_lwork(nPP, compute_vl=0, compute_vr=1)
        lwork = max(int(np.real(work)), 4 * nPP, 1)
        wr, wi, _, vr, info = lapack.dgeev(M, compute_vl=0, compute_vr=1, lwork=lwork,
                                           overwrite_a=1)
        if info != 0:
            flag(FLAG_PP_DGEEV_FAILED,
                 f'spin-orbital pp BSE: dgeev returned info = {info}; kept as Quack does',
                 ParquetBSEUnstable, stacklevel=3)
    # Quack discards wi whatever it is; a pair split off the real axis by
    # roundoff alone (a degenerate 2p or 2h pair) is no instability, so only
    # imaginary parts above PARQUET_PP_COMPLEX_TOL are flagged.
    scale = max(1.0, float(np.max(np.abs(wr), initial=0.0)))
    cplx = np.abs(wi) > PARQUET_PP_COMPLEX_TOL * scale
    if cplx.any():
        flag(FLAG_PP_COMPLEX_ROOTS,
             f'spin-orbital pp BSE: {int(cplx.sum())} complex eigenvalue(s) (largest '
             f'|Im| {np.max(np.abs(wi)):.3e}); their real parts and the real and imaginary '
             f'parts of their eigenvectors are used as roots, as in Quack',
             ParquetBSEUnstable, stacklevel=3)
    upper = wr > PARQUET_PP_BRANCH_SPLIT
    i_ee = np.flatnonzero(upper)
    i_hh = np.flatnonzero(~upper)
    if len(i_ee) != nVV or len(i_hh) != nOO:
        flag(FLAG_PP_BRANCH_COUNT,
             f'spin-orbital pp BSE: {len(i_ee)} eigenvalues above {PARQUET_PP_BRANCH_SPLIT:g} for '
             f'{nVV} 2p roots and {len(i_hh)} others for {nOO} 2h roots; the surplus is '
             f'dropped and missing roots are left at 0 with zero eigenvectors, as in Quack',
             ParquetBSEUnstable, stacklevel=3)
    om_ee, z_ee = np.zeros(nVV), np.zeros((nPP, nVV))
    om_hh, z_hh = np.zeros(nOO), np.zeros((nPP, nOO))
    k = min(len(i_ee), nVV)
    om_ee[:k], z_ee[:, :k] = wr[i_ee[:k]], vr[:, i_ee[:k]]
    k = min(len(i_hh), nOO)
    om_hh[:k], z_hh[:, :k] = wr[i_hh[:k]], vr[:, i_hh[:k]]
    order = np.argsort(om_ee, kind='stable')
    om_ee, z_ee = om_ee[order], z_ee[:, order]
    order = np.argsort(om_hh, kind='stable')
    om_hh, z_hh = om_hh[order], z_hh[:, order]
    metric = np.concatenate([np.ones(nVV), -np.ones(nOO)])[:, None]
    with np.errstate(invalid='ignore', over='ignore'):
        if nVV:
            z_ee = z_ee @ _lowdin(z_ee.T @ (metric * z_ee), '2p')
        if nOO:
            z_hh = z_hh @ _lowdin(z_hh.T @ (metric * -z_hh), '2h')
    return (om_ee, np.ascontiguousarray(z_ee[:nVV]), np.ascontiguousarray(z_ee[nVV:]),
            om_hh, np.ascontiguousarray(z_hh[:nVV]), np.ascontiguousarray(z_hh[nVV:]))


def solve_pp_bse_so(B, C, D, tda):
    """Eigenpairs of the pp BSE, split into its 2p and 2h branches: Quack's
    ppGLR with sort_ppRPA.

    Under the TDA the branches decouple: C = X1 om_ee X1^T, -D = Y2 om_hh
    Y2^T, Y1 = X2 = 0. Otherwise [[C, B], [-B^T, -D]] (Eq. 40) is
    diagonalized by dgeev; eigenvalues above PARQUET_PP_BRANCH_SPLIT (0) are
    the 2p roots and the rest the 2h roots, each branch sorted ascending and
    Lowdin-normalized in the metric diag(1, -1):
    X1^T X1 - Y1^T Y1 = 1 and Y2^T Y2 - X2^T X2 = 1.

    Kept as Quack has them, and flagged (`ParquetBSEUnstable`): complex
    roots, 'parquet_pp_complex_roots' (their real parts, and the real and
    imaginary parts of their eigenvectors, are used as roots; a pair whose
    |Im| is below PARQUET_PP_COMPLEX_TOL, a degenerate real pair split by
    roundoff, is treated the same way without a flag), and a sign split other
    than nVV / nOO, 'parquet_pp_branch_count' (surplus roots dropped, missing
    ones at 0). A near-singular Lowdin step is flagged
    'parquet_pp_lowdin_small_eigenvalue', and 2p and 2h correlation energies
    departing from their mean by more than PARQUET_PP_EC_CONSISTENCY_TOL
    'parquet_pp_ec_inconsistent', where Quack prints.

    Parameters
    ----------
    B : ndarray (nVV, nOO), or None when tda
    C : ndarray (nVV, nVV)
    D : ndarray (nOO, nOO)
        Bare blocks plus the vertex kernel (pp_bse_blocks_so +
        pp_kernel_blocks_so). B is not read under the TDA. Either pair
        space may be empty (nVV = 0 or nOO = 0): that branch has no root.
    tda : bool
        Tamm-Dancoff.

    Returns
    -------
    PPBSESolution
        om_ee (nVV,), X1 (nVV, nVV), Y1 (nOO, nVV), om_hh (nOO,),
        X2 (nVV, nOO), Y2 (nOO, nOO), and
        ec = (sum om_ee - sum om_hh - Tr C - Tr D) / 2.
    """
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    nVV, nOO = C.shape[0], D.shape[0]
    if C.shape != (nVV, nVV) or D.shape != (nOO, nOO):
        raise ValueError(f'C {C.shape} and D {D.shape} must be square')
    if tda:
        empty = (np.zeros(0), np.zeros((0, 0)))
        om_ee, X1 = _eigh_upper(C, 'pp BSE 2p block (TDA)') if nVV else empty
        om_hh, Y2 = _eigh_upper(-D, 'pp BSE 2h block (TDA)') if nOO else empty
        X1, Y2 = np.ascontiguousarray(X1), np.ascontiguousarray(Y2)
        Y1, X2 = np.zeros((nOO, nVV)), np.zeros((nVV, nOO))
    else:
        B = np.asarray(B, dtype=float)
        if B.shape != (nVV, nOO):
            raise ValueError(f'B has shape {B.shape}; expected {(nVV, nOO)}')
        om_ee, X1, Y1, om_hh, X2, Y2 = _pp_general(B, C, D)
    tr_c, tr_d = np.trace(C), np.trace(D)
    ec = 0.5 * (np.sum(om_ee) - np.sum(om_hh) - tr_c - tr_d)
    ec_ee = np.sum(om_ee) - tr_c
    ec_hh = -np.sum(om_hh) - tr_d
    if (abs(ec - ec_ee) > PARQUET_PP_EC_CONSISTENCY_TOL
            or abs(ec - ec_hh) > PARQUET_PP_EC_CONSISTENCY_TOL):
        flag(FLAG_PP_EC_INCONSISTENT,
             f'spin-orbital pp BSE: the 2p and 2h correlation energies {ec_ee:.10f} and '
             f'{ec_hh:.10f} differ from their mean {ec:.10f} by more than '
             f'{PARQUET_PP_EC_CONSISTENCY_TOL:g} (Quack: RPA1 != RPA2)', ParquetBSEUnstable)
    return PPBSESolution(om_ee, X1, Y1, om_hh, X2, Y2, float(ec))


def eh_effective_integrals_so(eri, vertices, XpY, XmY, nO):
    """Generalized eh effective integrals M^eh_pq,n (Eq. 38): Quack's
    G_eh_screened_integral.

        rho[p, q, n] = sum_jb (i Gamma^eh_pbqj X[n, jb] + i Gamma^eh_pjqb Y[n, jb]),
        i Gamma^eh_pbqj = <pb|qj> - <pb|jq> - Phi^eh_pbjq + Phi^pp_pbqj,
        i Gamma^eh_pjqb = <pj|qb> - <pj|bq> - Phi^eh_pjbq + Phi^pp_pjqb,

    X = (XpY + XmY) / 2, Y = (XpY - XmY) / 2, over every orbital pair pq.
    The vertices are those the eh kernel was built with (the previous
    iteration's).

    Parameters
    ----------
    eri : ndarray (N, N, N, N)
        <pq|rs>.
    vertices : SpinOrbitalVertices
        Reducible vertices of the kernel.
    XpY, XmY : ndarray (nS, nS)
        Eigenvectors as rows (solve_eh_bse_so).
    nO : int
        Occupied spin orbitals.

    Returns
    -------
    rho : ndarray (N, N, nS)

    Cost: 4 N^2 nS^2 flops (two GEMMs); memory a few N^2 nS arrays.
    """
    eri = np.asarray(eri, dtype=float)
    n = _orbital_count(nO, eri)
    eh, pp = _vertex_arrays(vertices, n)
    nS = nO * (n - nO)
    XpY = np.asarray(XpY, dtype=float)
    XmY = np.asarray(XmY, dtype=float)
    if XpY.shape != (nS, nS) or XmY.shape != (nS, nS):
        raise ValueError(f'XpY {XpY.shape} and XmY {XmY.shape}: expected {(nS, nS)}')
    o, v = slice(0, nO), slice(nO, n)
    X = 0.5 * (XpY + XmY)
    Y = 0.5 * (XpY - XmY)
    # kernel[p, q, j, b]
    k1 = (eri[:, v, :, o].transpose(0, 2, 3, 1) - eri[:, v, o, :].transpose(0, 3, 2, 1)
          - eh[:, v, o, :].transpose(0, 3, 2, 1) + pp[:, v, :, o].transpose(0, 2, 3, 1))
    rho = k1.reshape(n * n, nS) @ X.T
    del k1
    k2 = (eri[:, o, :, v].transpose(0, 2, 1, 3) - eri[:, o, v, :].transpose(0, 3, 1, 2)
          - eh[:, o, v, :].transpose(0, 3, 1, 2) + pp[:, o, :, v].transpose(0, 2, 1, 3))
    rho += k2.reshape(n * n, nS) @ Y.T
    return rho.reshape(n, n, nS)


def pp_effective_integrals_so(eri, vertices, X1, Y1, X2, Y2, nO):
    """Generalized pp effective integrals M^ee_pq,m and M^hh_pq,m (Eq. 44):
    Quack's G_pp_screened_integral.

        rho_ee[p, q, m] = sum_{c<d} G_pqcd X1[cd, m] + sum_{k<l} G_pqkl Y1[kl, m],
        rho_hh[p, q, m] = sum_{c<d} G_pqcd X2[cd, m] + sum_{k<l} G_pqkl Y2[kl, m],
        G_pqrs = i Gamma^pp_pqrs = <pq|rs> - <pq|sr> + Phi^eh_pqrs - Phi^eh_pqsr,

    over every orbital pair pq (Phi^pp does not enter, Eq. 47b).

    Parameters
    ----------
    eri : ndarray (N, N, N, N)
        <pq|rs>.
    vertices : SpinOrbitalVertices
        Reducible vertices of the kernel (pp is not read).
    X1, Y1, X2, Y2 : ndarray
        The pp eigenvectors (solve_pp_bse_so), (nVV, nVV), (nOO, nVV),
        (nVV, nOO), (nOO, nOO).
    nO : int
        Occupied spin orbitals.

    Returns
    -------
    rho_ee : ndarray (N, N, nVV)
    rho_hh : ndarray (N, N, nOO)

    Cost: 2 N^2 (nVV + nOO)^2 flops, in blocks of the first index whose
    gathered kernel is at most PARQUET_SO_BLOCK_BYTES; memory the two outputs.
    """
    eri = np.asarray(eri, dtype=float)
    n = _orbital_count(nO, eri)
    eh = np.asarray(vertices.eh, dtype=float)
    if eh.shape != (n,) * 4:
        raise ValueError(f'the eh vertex has shape {eh.shape}; expected ({n},)*4')
    c, d = pp_pairs(nO, n, 'spin-orbital')
    k, l = pp_pairs(0, nO, 'spin-orbital')
    nVV, nOO = len(c), len(k)
    X1, Y1, X2, Y2 = (np.asarray(x, dtype=float) for x in (X1, Y1, X2, Y2))
    for name, x, shape in (('X1', X1, (nVV, nVV)), ('Y1', Y1, (nOO, nVV)),
                           ('X2', X2, (nVV, nOO)), ('Y2', Y2, (nOO, nOO))):
        if x.shape != shape:
            raise ValueError(f'{name} has shape {x.shape}; expected {shape}')
    rho_ee = np.empty((n, n, nVV))
    rho_hh = np.empty((n, n, nOO))
    step = max(1, PARQUET_SO_BLOCK_BYTES // (8 * n * max(nVV + nOO, 1)))
    for p0 in range(0, n, step):
        sl = slice(p0, min(p0 + step, n))
        nb = sl.stop - sl.start
        e, h = eri[sl], eh[sl]
        g_vv = (e[:, :, c, d] - e[:, :, d, c]) + (h[:, :, c, d] - h[:, :, d, c])
        g_oo = (e[:, :, k, l] - e[:, :, l, k]) + (h[:, :, k, l] - h[:, :, l, k])
        # explicit sizes: an empty pair space (nVV or nOO = 0) has nothing to infer -1 from
        g_vv = g_vv.reshape(nb * n, nVV)
        g_oo = g_oo.reshape(nb * n, nOO)
        rho_ee[sl] = (g_vv @ X1 + g_oo @ Y1).reshape(nb, n, nVV)
        rho_hh[sl] = (g_vv @ X2 + g_oo @ Y2).reshape(nb, n, nOO)
    return rho_ee, rho_hh


def regularized_inverse(dem, eta):
    """g(dem) = (1 - exp(-2 eta dem^2)) / dem: the pole weight of the
    regularized reducible vertices (Eq. 52, Quack's two-body regularizer).

    eta = 0 is complete regularization (g = 0: the vertices vanish, FLEX);
    large eta leaves 1/dem. dem = 0 exactly (omega on a pole: a zero mode at
    omega = 0, such as a one-electron system's spin flip) gives the limit 0
    (g is 2 eta dem to first order), where Quack's formula is 0/0 = NaN: the
    port departs from Quack there alone, so that such a mode weighs nothing
    whether rounding puts it at 0 or a few ulps off (where 1 - exp rounds to
    0); every other dem is evaluated by the formula as written. A NaN dem
    gives NaN (the callers count the weights that are not finite,
    `_flag_poles`). A dem so large that dem^2 overflows (above about 1e154
    Ha, eigenvalues of a diverging two-body loop) gives exp(-inf) = 0 and
    the finite 1/dem, without a warning.

    Parameters
    ----------
    dem : array_like
        omega - Om (or omega + Om) per pole.
    eta : float
        The two-body regularization s_2b, >= 0 (Quack's eta_2b).
    """
    eta = float(eta)
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError(f'eta = {eta}: the two-body regularization must be finite and >= 0')
    dem = np.asarray(dem, dtype=float)
    # past |dem| ~ 1e154 (a diverging loop's eigenvalues) dem * dem overflows: exp(-inf) is
    # 0 and g = 1/dem, finite, so the overflow is no event (the callers count what is not
    # finite, `_flag_poles`)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        g = (1.0 - np.exp(-2.0 * eta * dem * dem)) / dem
    zero = dem == 0.0
    if zero.any():
        g = np.where(zero, 0.0, g)          # Quack's 0/0: the limit, 2 eta dem -> 0
    return g


def _flag_poles(weights, what):
    bad = sum(int(np.sum(~np.isfinite(w))) for w in weights)
    if bad:
        flag(FLAG_VERTEX_POLE,
             f'{what}: {bad} pole weight(s) not finite (an eigenvalue or omega that is '
             f'not finite); the vertex is NaN there, as in Quack', ParquetNotFinite,
             stacklevel=3)


def eh_vertex_from_weights(rho, w_res, w_anti):
    """The eh reducible vertex from its pole weights:

        Phi[p, q, r, s] = sum_n (rho[p, r, n] rho[s, q, n] w_res[n]
                                 - rho[r, p, n] rho[q, s, n] w_anti[n])

    (Eq. 37 with w_res[n] = g(omega - Om_n), w_anti[n] = g(omega + Om_n));
    a frequency average of the vertex is this with averaged weights.

    Parameters
    ----------
    rho : ndarray (N, N, nS)
        Effective integrals.
    w_res, w_anti : ndarray (nS,)
        Weights of the resonant (+Om) and antiresonant (-Om) poles.

    Returns
    -------
    phi : ndarray (N, N, N, N)

    Cost: 4 N^4 nS flops in blocks of the first index of at most
    PARQUET_SO_BLOCK_BYTES each; memory the N^4 output and two blocks.
    """
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 3 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f'rho has shape {rho.shape}; expected (N, N, nS)')
    n, n2, nS = rho.shape[0], rho.shape[0] ** 2, rho.shape[2]
    w_res = np.asarray(w_res, dtype=float)
    w_anti = np.asarray(w_anti, dtype=float)
    if w_res.shape != (nS,) or w_anti.shape != (nS,):
        raise ValueError(f'weights {w_res.shape}, {w_anti.shape}; expected ({nS},)')
    R = rho.reshape(n2, nS)
    r_res = R * w_res
    r_anti = (R * w_anti).reshape(n, n, nS)
    phi = np.empty((n, n, n, n))
    step = max(1, PARQUET_SO_BLOCK_BYTES // (8 * n ** 3))
    for p0 in range(0, n, step):
        p1 = min(p0 + step, n)
        nb = p1 - p0
        # t1[p, r, s, q] = sum_n rho[p, r, n] w_res[n] rho[s, q, n]
        t1 = (r_res[p0 * n:p1 * n] @ R.T).reshape(nb, n, n, n)
        # t2[r, p, q, s] = sum_n rho[r, p, n] w_anti[n] rho[q, s, n]
        t2 = (r_anti[:, p0:p1].reshape(n * nb, nS) @ R.T).reshape(n, nb, n, n)
        np.subtract(t1.transpose(0, 3, 1, 2), t2.transpose(1, 2, 0, 3), out=phi[p0:p1])
    return phi


def pp_vertex_from_weights(rho_ee, w_ee, rho_hh, w_hh):
    """The pp reducible vertex from its pole weights:

        Phi[p, q, r, s] = sum_m rho_ee[p, q, m] rho_ee[r, s, m] w_ee[m]
                        - sum_m rho_hh[p, q, m] rho_hh[r, s, m] w_hh[m]

    (Eq. 43 with w_ee[m] = g(omega - Om^ee_m), w_hh[m] = g(omega - Om^hh_m));
    a frequency average of the vertex is this with averaged weights.

    Parameters
    ----------
    rho_ee : ndarray (N, N, nVV)
    w_ee : ndarray (nVV,)
    rho_hh : ndarray (N, N, nOO)
    w_hh : ndarray (nOO,)

    Returns
    -------
    phi : ndarray (N, N, N, N)

    Cost: 2 N^4 (nVV + nOO) flops in row blocks of at most PARQUET_SO_BLOCK_BYTES;
    memory the N^4 output and one block.
    """
    rho_ee = np.asarray(rho_ee, dtype=float)
    rho_hh = np.asarray(rho_hh, dtype=float)
    n = rho_ee.shape[0] if rho_ee.ndim == 3 else -1
    if (rho_ee.ndim != 3 or rho_hh.ndim != 3 or rho_ee.shape[1] != n
            or rho_hh.shape[:2] != (n, n)):
        raise ValueError(f'rho_ee {rho_ee.shape} and rho_hh {rho_hh.shape}: expected (N, N, n)')
    n2 = n * n
    nVV, nOO = rho_ee.shape[2], rho_hh.shape[2]
    w_ee = np.asarray(w_ee, dtype=float)
    w_hh = np.asarray(w_hh, dtype=float)
    if w_ee.shape != (nVV,) or w_hh.shape != (nOO,):
        raise ValueError(f'weights {w_ee.shape}, {w_hh.shape}; expected ({nVV},), ({nOO},)')
    r_ee = rho_ee.reshape(n2, nVV)
    r_hh = rho_hh.reshape(n2, nOO)
    phi = np.empty((n2, n2))
    step = max(1, PARQUET_SO_BLOCK_BYTES // (8 * max(n2, nVV)))
    for r0 in range(0, n2, step):
        r1 = min(r0 + step, n2)
        np.matmul(r_ee[r0:r1] * w_ee, r_ee.T, out=phi[r0:r1])
        if nOO:
            phi[r0:r1] -= (r_hh[r0:r1] * w_hh) @ r_hh.T
    return phi.reshape(n, n, n, n)


def eh_reducible_vertex_so(om, rho, eta, omega=0.0):
    """The regularized eh reducible vertex i Phi^eh_pqrs(omega): Quack's
    G_eh_Phi at omega = 0, and at any omega the frequency dependence of
    R_eh_singlet_Phi (Eq. 37 with the regularizer of Eq. 52):

        Phi[p, q, r, s] = sum_n [rho[p, r, n] rho[s, q, n] g(omega - Om_n)
                                 - rho[r, p, n] rho[q, s, n] g(omega + Om_n)],

    g = regularized_inverse; at omega = 0 this is Eq. 52a,
    -sum_n (rho_pr,n rho_sq,n + rho_rp,n rho_qs,n) (1 - exp(-2 eta Om_n^2)) / Om_n.

    Parameters
    ----------
    om : ndarray (nS,)
        eh excitation energies.
    rho : ndarray (N, N, nS)
        eh effective integrals (eh_effective_integrals_so).
    eta : float
        Two-body regularization s_2b (Quack's eta_2b), >= 0.
    omega : float
        The frequency of the vertex (0: the static vertex the loop uses).

    Returns
    -------
    phi : ndarray (N, N, N, N)

    A pole hit (omega = +-Om_n exactly) weighs 0, the limit of g; a NaN
    energy gives NaN, flagged 'parquet_vertex_pole'. Cost: 4 N^4 nS flops;
    memory the N^4 output and two blocks of at most PARQUET_SO_BLOCK_BYTES.
    """
    om = np.asarray(om, dtype=float)
    w_res = regularized_inverse(omega - om, eta)
    w_anti = regularized_inverse(omega + om, eta)
    _flag_poles((w_res, w_anti), f'spin-orbital eh reducible vertex at omega = {omega}')
    return eh_vertex_from_weights(rho, w_res, w_anti)


def pp_reducible_vertex_so(om_ee, rho_ee, om_hh, rho_hh, eta, omega=0.0):
    """The regularized pp reducible vertex i Phi^pp_pqrs(omega): Quack's
    G_pp_Phi at omega = 0, and at any omega the frequency dependence of
    R_pp_singlet_Phi (Eq. 43 with the regularizer of Eq. 52):

        Phi[p, q, r, s] = sum_m rho_ee[p, q, m] rho_ee[r, s, m] g(omega - Om^ee_m)
                        - sum_m rho_hh[p, q, m] rho_hh[r, s, m] g(omega - Om^hh_m),

    g = regularized_inverse, Om^hh the signed (negative) 2h roots; at
    omega = 0 this is Eq. 52b.

    Parameters
    ----------
    om_ee : ndarray (nVV,)
    rho_ee : ndarray (N, N, nVV)
    om_hh : ndarray (nOO,)
    rho_hh : ndarray (N, N, nOO)
        The pp spectrum and effective integrals (solve_pp_bse_so,
        pp_effective_integrals_so).
    eta : float
        Two-body regularization s_2b, >= 0.
    omega : float
        The frequency of the vertex (0: the static vertex).

    Returns
    -------
    phi : ndarray (N, N, N, N)

    A pole hit (omega = Om exactly) weighs 0, the limit of g; a NaN energy
    gives NaN, flagged 'parquet_vertex_pole'. Cost: 2 N^4 (nVV + nOO) flops;
    memory the N^4 output and one block of at most PARQUET_SO_BLOCK_BYTES.
    """
    om_ee = np.asarray(om_ee, dtype=float)
    om_hh = np.asarray(om_hh, dtype=float)
    w_ee = regularized_inverse(omega - om_ee, eta)
    w_hh = regularized_inverse(omega - om_hh, eta)
    _flag_poles((w_ee, w_hh), f'spin-orbital pp reducible vertex at omega = {omega}')
    return pp_vertex_from_weights(rho_ee, w_ee, rho_hh, w_hh)
