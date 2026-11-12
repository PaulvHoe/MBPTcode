"""The closed-shell eh and pp Bethe-Salpeter problems of the parquet two-body loop.

Each two-body iteration of the static-kernel parquet approximation (Marie and
Loos, J. Chem. Phys. 163, 194115 (2025), Sec. IV) solves four BSEs, built on
the one-body energies eps of the current one-body iteration and the static
irreducible kernels of the previous two-body iteration: the eh BSE (Eq. (34))
and the pp BSE (Eq. (40)), each in its singlet and triplet channel. This
module builds their blocks and solves them as Quack's R_evParquet does; the
kernels enter through the four reducible vertices of the loop
(`spectra.RestrictedVertices`, the real i*Phi of the paper), spin-adapted as
in the paper's supplementary material.

eh BSE, pairs ia (occupied i outer, virtual a inner: `pairs.eh_pairs`),
<pq|rs> physicist integrals:

    A_{ia,jb} = (eps_a - eps_i) d_ij d_ab + 2 <bi|ja> - <bj|ai> + GamA   (singlet)
                (eps_a - eps_i) d_ij d_ab           - <bj|ai> + GamA   (triplet)
    B_{ia,jb} = 2 <bi|ja> - <bj|ia> + GamB                              (singlet)
                          - <bj|ia> + GamB                              (triplet)

    GamA_{ia,jb} = c1 Phi^eh_s[a,j,b,i] + c2 Phi^eh_t[a,j,b,i]
                 + c3 Phi^pp_s[a,j,i,b] + c4 Phi^pp_t[a,j,i,b],
    GamB_{ia,jb} = the same at [a,b,j,i] (eh) and [a,b,i,j] (pp),

with (c1, c2, c3, c4) = (-1/2, -3/2, +1/2, +3/2) in the singlet channel and
(-1/2, +1/2, -1/2, +1/2) in the triplet one (Eqs. (35) and (47a)). The
spectrum comes from A alone in the Tamm-Dancoff approximation, otherwise from
the symmetric (A-B)^{1/2} (A+B) (A-B)^{1/2} = Z Omega^2 Z^T, with
X+Y = Omega^{-1/2} Z^T (A-B)^{1/2} and X-Y = Omega^{1/2} Z^T (A-B)^{-1/2}.

pp BSE, pairs (p, q) with p <= q (singlet) or p < q (triplet), p outer
(`pairs.pp_pairs`), n_pq = 1/sqrt(1 + d_pq) in the singlet channel and 1 in
the triplet one, + in the singlet channel and - in the triplet one:

    C_{ab,cd} = (eps_a + eps_b) d_ac d_bd + n_ab n_cd [<ab|cd> +- <ab|dc>] + GamC
    B_{ab,ij} =                             n_ab n_ij [<ab|ij> +- <ab|ji>] + GamB
    D_{ij,kl} = -(eps_i + eps_j) d_ik d_jl + n_ij n_kl [<ij|kl> +- <ij|lk>] + GamD

    GamX_{pq,rs} = n_pq n_rs [W_pqrs +- W_pqsr],
    W = Phi^eh_s / 2 - 3 Phi^eh_t / 2 (singlet), (Phi^eh_s + Phi^eh_t) / 2 (triplet)

(Eqs. (41) and (47b)); only the eh vertices enter. In the Tamm-Dancoff
approximation the 2p (ee) energies are the eigenvalues of C and the 2h (hh)
ones those of -D. Otherwise H = [[C, B], [-B^T, -D]] is diagonalized as a
general real matrix (LAPACK dgeev), its eigenpairs are split into the ee
branch (eigenvalue > 0) and the hh branch (eigenvalue <= 0), each sorted
ascending, and each branch is Loewdin-orthonormalized in the metric
eta = diag(1, -1), so that X1^T X1 - Y1^T Y1 = 1 and X2^T X2 - Y2^T Y2 = -1.

Conventions. Arrays are indexed as Quack's Fortran arrays, 0-based: eri[p, q,
r, s] = <pq|rs> (MO, physicist, not antisymmetrized), occupied orbitals
first; the eh eigenvectors are the ROWS of XpY[n, ia] and XmY[n, ia] (as
phRLR returns them); the pp ones are the COLUMNS of X1[cd, m] (nVV, nVV),
Y1[kl, m] (nOO, nVV), X2[cd, m] (nVV, nOO), Y2[kl, m] (nOO, nOO). Only the
all-electron problem (Quack's nC = nR = 0) at full coupling (lambda = 1) with
exchange (dRPA false) is built, as R_evParquet calls it.

What is Quack's and kept as such, oddities included (each function says
which it carries):

  * the symmetric eigenproblems read the upper triangle only (dsyev 'U'); the
    LAPACK driver differs (divide and conquer here, PARQUET_EIGH_DRIVER),
    which leaves the eigenvalues and eigenspaces as they are, to rounding;
  * full eh: an A - B that is not positive definite makes (A-B)^{1/2} NaN,
    and with it the whole spectrum (Quack prints a warning and carries on);
    a negative Omega^2 is replaced by |Omega^2| (an imaginary excitation
    becomes a real one) and the energies stay in the ascending order of
    Omega^2, so they are not ascending themselves;
  * full pp: the branches are told apart by the sign of the eigenvalue with
    the chemical potential at zero (Quack's file switch
    'use_chem_pot_in_ppRPA', eF = eps_HOMO + eps_LUMO, is not ported), not
    by the metric norm; dgeev's imaginary parts are dropped (a complex pair
    is taken as two real vectors); roots beyond nVV (nOO) on one side are
    dropped, the first ones in dgeev's order kept (an order that differs
    between LAPACK libraries), and missing ones left at zero energy with a
    zero vector (uninitialized memory in Quack), which makes the Loewdin
    step of that branch inf/NaN; the Loewdin step runs over the whole
    branch, not per eigenvalue;
  * the singlet C block scales a diagonal pair by Quack's literal
    0.7071067811865475 (twice for a = b and c = d: 0.4999999999999999),
    where B, D and the kernels divide by sqrt((1 + d_pq)(1 + d_rs)).

Each of Quack's printed warnings is a warning here (`events.ParquetBSEUnstable`)
with a named flag (`src.Base.numerical_flags`; the names are the FLAG_*
constants of `events`), and the result is what Quack's arithmetic gives. Three
events are flagged that Quack lets pass: a negative Tamm-Dancoff eh energy
('parquet_eh_tda_negative_excitations', numbers kept), a full eh excitation
energy below PARQUET_EH_ZERO_MODE_TOL ('parquet_eh_zero_mode', numbers kept:
the edge of an instability, where the self-energy grows as 1/Omega), and NaN
or inf entering a diagonalization ('parquet_bse_not_finite', `events.ParquetNotFinite`),
whose eigenpairs are NaN here where Quack's are what its LAPACK makes of them
(Accelerate's dgeev: zero pp energies).

Cost, at the paper's largest problem (CH4/aug-cc-pVTZ: norb 138, nO 5,
nS 665, nVV 8911 singlet and 8778 triplet, nOO 15 and 10): the eh blocks
and solver are O(nS^2) memory and O(nS^3) time, a fraction of a second.
The pp blocks are gathered one row orbital at a time, O(nVV^2) memory (the
result, 0.64 GB for C) plus (norb, norb, norb)-sized temporaries, a few
seconds; the Tamm-Dancoff pp solver is one symmetric diagonalization of C
(about three nVV^2 doubles at its peak, 1.9 GB), the full one dgeev on
nVV + nOO plus the Loewdin steps (about six, 3.8 GB), both O(nVV^3) time,
minutes at nVV 8911.
"""
import numpy as np
import scipy.linalg as la
from scipy.linalg import lapack

from src.Base.numerical_flags import flag
from src.Base.numerics.constants import (PARQUET_EH_ZERO_MODE_TOL, PARQUET_EIGH_DRIVER,
                                         PARQUET_LOWDIN_SMALL_EIGENVALUE,
                                         PARQUET_PP_BRANCH_SPLIT, PARQUET_PP_EC_CONSISTENCY_TOL)
from src.SingleReference.Parquet.events import (FLAG_BSE_DIAGONALIZATION_FAILED,
                                                FLAG_BSE_NOT_FINITE,
                                                FLAG_EH_A_MINUS_B_NOT_POSITIVE,
                                                FLAG_EH_NEGATIVE_EXCITATIONS,
                                                FLAG_EH_TDA_NEGATIVE_EXCITATIONS,
                                                FLAG_EH_ZERO_MODE,
                                                FLAG_PP_BRANCH_COUNT, FLAG_PP_DGEEV_FAILED,
                                                FLAG_PP_EC_INCONSISTENT,
                                                FLAG_PP_LOWDIN_SMALL_EIGENVALUE,
                                                ParquetBSEUnstable, ParquetNotFinite)
from src.SingleReference.Parquet.pairs import EH_SPINS, eh_pairs, pp_pairs

#: The spin channels of the closed-shell eh and pp BSEs.
SPINS = EH_SPINS

#: Quack's spelled-out 1/sqrt(2) of the singlet C block (ppRLR_C), multiplied
#: rather than divided, so the block matches Quack's to the bit.
QUACK_INV_SQRT2 = 0.7071067811865475

#: (Phi^eh_s, Phi^eh_t, Phi^pp_s, Phi^pp_t) coefficients of the eh kernel,
#: R_eh_singlet_Gamma_A/B and R_eh_triplet_Gamma_A/B.
EH_KERNEL_COEFFICIENTS = {'singlet': (-0.5, -1.5, 0.5, 1.5),
                          'triplet': (-0.5, 0.5, -0.5, 0.5)}

#: (Phi^eh_s, Phi^eh_t) coefficients of W in the pp kernel,
#: R_pp_singlet_Gamma_B/C/D and R_pp_triplet_Gamma_B/C/D.
PP_KERNEL_COEFFICIENTS = {'singlet': (0.5, -1.5), 'triplet': (0.5, 0.5)}


def eh_bse_blocks(eps, eri, nO, spin, tda):
    """(A, B): the bare blocks of the closed-shell eh BSE.

    Ports Quack's phRLR_A and phRLR_B (ispin 1 singlet, 2 triplet; dRPA
    false, lambda 1, nC = nR = 0), Eq. (35) of Marie and Loos without the
    kernel:

        A_{ia,jb} = (eps_a - eps_i) d_ij d_ab + 2 <bi|ja> - <bj|ai>  (singlet)
                    (eps_a - eps_i) d_ij d_ab           - <bj|ai>  (triplet)
        B_{ia,jb} = 2 <bi|ja> - <bj|ia>  (singlet),  -<bj|ia>  (triplet)

    Parameters
    ----------
    eps : ndarray, shape (norb,)
        One-body energies of the diagonal (R_evParquet's eOld), occupied
        first.
    eri : ndarray, shape (norb, norb, norb, norb)
        eri[p, q, r, s] = <pq|rs>, MO, physicist.
    nO : int
        Doubly occupied orbitals.
    spin : {'singlet', 'triplet'}
    tda : bool
        Tamm-Dancoff: B is not built.

    Returns
    -------
    A : ndarray, shape (nS, nS)
    B : ndarray, shape (nS, nS), or None when tda
        nS = nO nV pairs ia in `pairs.eh_pairs` order.
    """
    eps, eri, norb = _orbital_inputs(eps, eri, nO)
    _check_spin(spin)
    nV = norb - nO
    nS = nO * nV
    o, v = slice(0, nO), slice(nO, norb)
    # [i, a, j, b] views of <bi|ja>, <bj|ai> and <bj|ia>
    bija = eri[v, o, o, v].transpose(1, 3, 2, 0)
    bjai = eri[v, o, v, o].transpose(3, 2, 1, 0)
    A = np.empty((nO, nV, nO, nV))
    if spin == 'singlet':
        np.multiply(2.0, bija, out=A)
        A -= bjai
    else:
        np.negative(bjai, out=A)
    A = A.reshape(nS, nS)
    i, a = eh_pairs(nO, nV)
    # Quack adds e(a), then subtracts e(i)
    A.flat[::nS + 1] = (A.flat[::nS + 1] + eps[a]) - eps[i]
    if tda:
        return A, None
    bjia = eri[v, o, o, v].transpose(2, 3, 1, 0)
    B = np.empty((nO, nV, nO, nV))
    if spin == 'singlet':
        np.multiply(2.0, bija, out=B)
        B -= bjia
    else:
        np.negative(bjia, out=B)
    return A, B.reshape(nS, nS)


def eh_kernel_blocks(vertices, nO, spin, tda):
    """(GamA, GamB): the static kernel of the closed-shell eh BSE.

    Ports Quack's R_eh_singlet_Gamma_A/B and R_eh_triplet_Gamma_A/B: the
    spin-adapted iGamma^eh of Eq. (47a) less the bare integrals, on the
    reducible vertices of the previous two-body iteration,

        GamA_{ia,jb} = c1 Phi^eh_s[a,j,b,i] + c2 Phi^eh_t[a,j,b,i]
                     + c3 Phi^pp_s[a,j,i,b] + c4 Phi^pp_t[a,j,i,b],
        GamB_{ia,jb} = c1 Phi^eh_s[a,b,j,i] + c2 Phi^eh_t[a,b,j,i]
                     + c3 Phi^pp_s[a,b,i,j] + c4 Phi^pp_t[a,b,i,j],

    with `EH_KERNEL_COEFFICIENTS[spin]` = (c1, c2, c3, c4). Each block reads an
    (nO nV)^2 corner of each vertex; nothing of norb^4 size is formed.

    Parameters
    ----------
    vertices : spectra.RestrictedVertices
        eh_singlet, eh_triplet, pp_singlet, pp_triplet, each (norb,)*4.
    nO : int
    spin : {'singlet', 'triplet'}
    tda : bool
        Tamm-Dancoff: GamB is not built.

    Returns
    -------
    GamA : ndarray, shape (nS, nS)
    GamB : ndarray, shape (nS, nS), or None when tda
    """
    ehs, eht, pps, ppt = _vertex_tensors(vertices, nO, 4)
    _check_spin(spin)
    norb = ehs.shape[0]
    nV = norb - nO
    nS = nO * nV
    o, v = slice(0, nO), slice(nO, norb)
    coef = EH_KERNEL_COEFFICIENTS[spin]
    # [i, a, j, b] views: eh at [a,j,b,i], pp at [a,j,i,b]
    GamA = _combine(coef, (ehs[v, o, v, o].transpose(3, 0, 1, 2),
                           eht[v, o, v, o].transpose(3, 0, 1, 2),
                           pps[v, o, o, v].transpose(2, 0, 1, 3),
                           ppt[v, o, o, v].transpose(2, 0, 1, 3)))
    if tda:
        return GamA.reshape(nS, nS), None
    # eh at [a,b,j,i], pp at [a,b,i,j]
    GamB = _combine(coef, (ehs[v, v, o, o].transpose(3, 0, 2, 1),
                           eht[v, v, o, o].transpose(3, 0, 2, 1),
                           pps[v, v, o, o].transpose(2, 0, 3, 1),
                           ppt[v, v, o, o].transpose(2, 0, 3, 1)))
    return GamA.reshape(nS, nS), GamB.reshape(nS, nS)


def solve_eh_bse(A, B, tda):
    """(om, XpY, XmY, ec): the spectrum of the closed-shell eh BSE.

    Ports Quack's phRLR. Tamm-Dancoff: om and the rows of XpY = XmY are the
    eigenpairs of A. Otherwise, with A - B = U d U^T,

        Z = (A-B)^{1/2} (A+B) (A-B)^{1/2} = W Omega^2 W^T,
        XpY = Omega^{-1/2} W^T (A-B)^{1/2},  XmY = Omega^{1/2} W^T (A-B)^{-1/2},

    om = sqrt(|Omega^2|) in the ascending order of Omega^2. As in Quack: an
    A - B that is not positive definite makes (A-B)^{1/2} NaN, and with it
    om, XpY, XmY and ec (flag 'parquet_eh_a_minus_b_not_positive'); a
    negative Omega^2 is taken by its modulus (flag
    'parquet_eh_negative_excitations'), so om is then not ascending. A
    negative Tamm-Dancoff energy, which Quack lets pass silently, is flagged
    too ('parquet_eh_tda_negative_excitations'), the numbers kept. So is a
    full-problem om below PARQUET_EH_ZERO_MODE_TOL ('parquet_eh_zero_mode',
    as the spin-orbital solver flags it): a closed shell has no zero mode of
    its own (it breaks no continuous symmetry), but at the edge of an eh
    instability its lowest Omega goes to zero, X + Y grows as Omega^-1/2 and
    the self-energy as 1/Omega (H2/6-31G 1e-6 Angstrom before its triplet
    instability, FLEX at s_1b = 50: Omega 3.6e-4 Ha, Sigma_c of the HOMO
    -5346 eV, every quasiparticle root the house's root finder keeps of Z
    below 0.012, the HOMO's at +22.9 eV); the numbers are kept, as in Quack.
    The TDA has no such test: its amplitudes do not grow with 1/Omega (a
    small TDA root still weighs about 1/Omega in a weakly regularized
    vertex, `spin_orbital_two_body`).

    Parameters
    ----------
    A, B : ndarray, shape (nS, nS)
        The full blocks (bare + kernel); B is not read (may be None) when tda.
        As in Quack, the TDA diagonalization reads the upper triangle of A;
        otherwise A - B is read by its upper triangle and A + B whole.
    tda : bool

    Returns
    -------
    om : ndarray, shape (nS,)
    XpY, XmY : ndarray, shape (nS, nS)
        Row n is the eigenvector of om[n] (X+Y and X-Y; equal under TDA).
    ec : float
        Quack's EcRPA = (sum(om) - Tr A) / 2 (rounding noise under TDA).
    """
    A = _square(A, 'A')
    n = A.shape[0]
    if tda:
        om, vecs = _eigh_upper(A, 'the eh A block (TDA)')
        xpy = np.ascontiguousarray(vecs.T)
        xmy = xpy.copy()
        if n and om[0] < 0.0:
            flag(FLAG_EH_TDA_NEGATIVE_EXCITATIONS,
                 f'eh BSE (TDA): {int(np.sum(om < 0))} negative excitation '
                 f'energies, the lowest {om[0]:.6f}; kept as they are (Quack '
                 f'does not check)', ParquetBSEUnstable)
        return om, xpy, xmy, 0.5 * (om.sum() - np.trace(A))
    if B is None:
        raise ValueError('solve_eh_bse without TDA needs the B block')
    B = _square(B, 'B')
    if B.shape != A.shape:
        raise ValueError(f'B has shape {B.shape}, A {A.shape}')
    apb = A + B
    amb = A - B
    d, u = _eigh_upper(amb, 'A - B of the eh BSE')
    if n and d[0] < 0.0:
        flag(FLAG_EH_A_MINUS_B_NOT_POSITIVE,
             f'eh BSE: A - B is not positive definite (lowest eigenvalue '
             f'{d[0]:.6e}); its square root is NaN, and so are the excitation '
             f'energies, X+Y, X-Y and EcRPA, as in Quack', ParquetBSEUnstable)
        return (np.full(n, np.nan), np.full((n, n), np.nan), np.full((n, n), np.nan),
                np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        sq = np.sqrt(d)
        amb_sq = u @ (sq[:, None] * u.T)
        amb_iv = u @ ((1.0 / sq)[:, None] * u.T)
    del u, amb
    z = amb_sq @ (apb @ amb_sq)
    del apb
    om2, w = _eigh_upper(z, '(A-B)^1/2 (A+B) (A-B)^1/2 of the eh BSE')
    del z
    if n and om2[0] < 0.0:
        neg = om2[om2 < 0.0]
        flag(FLAG_EH_NEGATIVE_EXCITATIONS,
             f'eh BSE: {len(neg)} negative squared excitation energies '
             f'{np.array2string(neg, precision=5)}; taken as sqrt|Omega^2|, '
             f'as in Quack', ParquetBSEUnstable)
    om = np.sqrt(np.abs(om2))
    zero = om < PARQUET_EH_ZERO_MODE_TOL
    if zero.any():
        flag(FLAG_EH_ZERO_MODE,
             f'eh BSE: {int(zero.sum())} excitation energy(ies) below '
             f'PARQUET_EH_ZERO_MODE_TOL = {PARQUET_EH_ZERO_MODE_TOL:g} Ha (lowest '
             f'{np.min(om[zero]):.3e} Ha): the closed shell is at the edge of an eh '
             f'instability, where X + Y ~ Omega^-1/2 takes over the vertices and the '
             f'self-energy (Sigma ~ 1/Omega, Z -> 0); kept, as in Quack', ParquetBSEUnstable)
    with np.errstate(divide='ignore', invalid='ignore'):
        xpy = w.T @ amb_sq
        xpy *= (1.0 / np.sqrt(om))[:, None]
        xmy = w.T @ amb_iv
        xmy *= np.sqrt(om)[:, None]
    return om, xpy, xmy, 0.5 * (om.sum() - np.trace(A))


def pp_bse_blocks(eps, eri, nO, spin, tda):
    """(B, C, D): the bare blocks of the closed-shell pp BSE.

    Ports Quack's ppRLR_B, ppRLR_C and ppRLR_D (ispin 1 singlet, 2 triplet;
    lambda 1, nC = nR = 0, chemical potential eF = 0), Eq. (41) of Marie and
    Loos without the kernel:

        C_{ab,cd} = (eps_a + eps_b) d_ac d_bd + n_ab n_cd [<ab|cd> +- <ab|dc>]
        B_{ab,ij} =                             n_ab n_ij [<ab|ij> +- <ab|ji>]
        D_{ij,kl} = -(eps_i + eps_j) d_ik d_jl + n_ij n_kl [<ij|kl> +- <ij|lk>]

    (+ and n_pq = 1/sqrt(1 + d_pq) singlet; - and n = 1 triplet). The
    singlet C applies n_ab n_cd as Quack's ppRLR_C does, multiplying by
    `QUACK_INV_SQRT2` once per diagonal pair; B and D divide by
    sqrt((1 + d)(1 + d)). Quack's file switch 'use_chem_pot_in_ppRPA', a
    file of that name in the working directory that makes ppRLR_C and
    ppRLR_D subtract eF = eps_HOMO + eps_LUMO from the pair energies of the
    diagonals, is not ported (eF = 0).

    The blocks are gathered from eri one row orbital at a time: besides the
    result no temporary is larger than (norb, norb, norb).

    Parameters
    ----------
    eps, eri, nO, spin : as for `eh_bse_blocks`
    tda : bool
        Tamm-Dancoff: B is not built.

    Returns
    -------
    B : ndarray, shape (nVV, nOO), or None when tda
    C : ndarray, shape (nVV, nVV)
    D : ndarray, shape (nOO, nOO)
        Virtual pairs (a, b) and occupied pairs (i, j) in `pairs.pp_pairs`
        order: a <= b (singlet) or a < b (triplet), a outer.
    """
    eps, eri, norb = _orbital_inputs(eps, eri, nO)
    _check_spin(spin)
    occ, vir = (0, nO), (nO, norb)
    terms = ((1.0, eri),)
    C = _pp_pair_block(terms, vir, vir, spin, 'literal')
    a, b = pp_pairs(nO, norb, spin)
    C.flat[::C.shape[0] + 1] += eps[a] + eps[b]
    D = _pp_pair_block(terms, occ, occ, spin, 'sqrt')
    i, j = pp_pairs(0, nO, spin)
    D.flat[::D.shape[0] + 1] += -(eps[i] + eps[j])
    B = None if tda else _pp_pair_block(terms, vir, occ, spin, 'sqrt')
    return B, C, D


def pp_kernel_blocks(vertices, nO, spin, tda):
    """(GamB, GamC, GamD): the static kernel of the closed-shell pp BSE.

    Ports Quack's R_pp_singlet_Gamma_B/C/D and R_pp_triplet_Gamma_B/C/D: the
    spin-adapted iGamma^pp of Eq. (47b) less the bare integrals,

        GamX_{pq,rs} = n_pq n_rs [W_pqrs +- W_pqsr],
        W = c1 Phi^eh_s + c2 Phi^eh_t,

    (c1, c2) = `PP_KERNEL_COEFFICIENTS[spin]`, + and n_pq = 1/sqrt(1 + d_pq)
    (divided as sqrt((1 + d_pq)(1 + d_rs))) singlet, - and n = 1 triplet; X =
    C over virtual pairs, B virtual by occupied, D occupied pairs. Only the eh
    vertices enter. Gathered one row orbital at a time, as `pp_bse_blocks`.

    Parameters
    ----------
    vertices : spectra.RestrictedVertices
        Only eh_singlet and eh_triplet, each (norb,)*4, are read.
    nO : int
    spin : {'singlet', 'triplet'}
    tda : bool
        Tamm-Dancoff: GamB is not built.

    Returns
    -------
    GamB : ndarray, shape (nVV, nOO), or None when tda
    GamC : ndarray, shape (nVV, nVV)
    GamD : ndarray, shape (nOO, nOO)
    """
    ehs, eht = _vertex_tensors(vertices, nO, 2)
    _check_spin(spin)
    norb = ehs.shape[0]
    occ, vir = (0, nO), (nO, norb)
    c1, c2 = PP_KERNEL_COEFFICIENTS[spin]
    terms = ((c1, ehs), (c2, eht))
    GamC = _pp_pair_block(terms, vir, vir, spin, 'sqrt')
    GamD = _pp_pair_block(terms, occ, occ, spin, 'sqrt')
    GamB = None if tda else _pp_pair_block(terms, vir, occ, spin, 'sqrt')
    return GamB, GamC, GamD


def solve_pp_bse(B, C, D, tda):
    """(om_ee, X1, Y1, om_hh, X2, Y2, ec): the spectrum of the closed-shell
    pp BSE.

    Ports Quack's ppRLR, with diagonalize_general_matrix, sort_ppRPA and its
    Loewdin orthogonalize_matrix. Tamm-Dancoff: om_ee and the columns of X1
    are the eigenpairs of C (Y1 = 0), om_hh and the columns of Y2 those of -D
    (X2 = 0). Otherwise

        H = [[C, B], [-B^T, -D]],  H R = R w  (LAPACK dgeev, real parts),

    the eigenpairs with w > 0 form the ee branch and the others the hh
    branch (Quack's chemical potential 0), each sorted ascending, and each
    branch Z is Loewdin-orthonormalized in eta = diag(1, -1):

        Z1 <- Z1 S1^{-1/2},  S1 = Z1^T eta Z1;   Z2 <- Z2 S2^{-1/2},  S2 = -Z2^T eta Z2,

    so X1^T X1 - Y1^T Y1 = 1 and X2^T X2 - Y2^T Y2 = -1. As in Quack, a
    branch with other than nVV (nOO) eigenvalues keeps the first nVV (nOO)
    of dgeev's order (an order that differs between LAPACK libraries) and
    pads with zero energies and zero vectors (flag 'parquet_pp_branch_count';
    the padded branch's Loewdin step is then inf/NaN), a complex pair enters
    as its real and imaginary parts, and a Loewdin eigenvalue below
    PARQUET_LOWDIN_SMALL_EIGENVALUE is reported
    ('parquet_pp_lowdin_small_eigenvalue'; a negative one makes the branch
    NaN).

    ec is Quack's EcRPA = (sum om_ee - sum om_hh - Tr C - Tr D) / 2; Quack's
    check that sum om_ee - Tr C and -sum om_hh - Tr D agree with it to
    PARQUET_PP_EC_CONSISTENCY_TOL raises 'parquet_pp_ec_inconsistent' when it
    fails.

    Parameters
    ----------
    B : ndarray, shape (nVV, nOO)
        Not read (may be None) when tda.
    C : ndarray, shape (nVV, nVV)
    D : ndarray, shape (nOO, nOO)
        The full blocks (bare + kernel); the Tamm-Dancoff diagonalizations
        read the upper triangles of C and -D.
    tda : bool

    Returns
    -------
    om_ee : ndarray, shape (nVV,)
    X1 : ndarray, shape (nVV, nVV)
    Y1 : ndarray, shape (nOO, nVV)
        Column m: the ee eigenvector of om_ee[m].
    om_hh : ndarray, shape (nOO,)
    X2 : ndarray, shape (nVV, nOO)
    Y2 : ndarray, shape (nOO, nOO)
        Column m: the hh eigenvector of om_hh[m].
    ec : float
    """
    C = _square(C, 'C')
    D = _square(D, 'D')
    nvv, noo = C.shape[0], D.shape[0]
    if tda:
        om1, x1 = _eigh_upper(C, 'the pp C block (TDA)')
        om2, y2 = _eigh_upper(-D, 'minus the pp D block (TDA)')
        y1 = np.zeros((noo, nvv))
        x2 = np.zeros((nvv, noo))
    else:
        if B is None:
            raise ValueError('solve_pp_bse without TDA needs the B block')
        B = np.asarray(B, dtype=float)
        if B.shape != (nvv, noo):
            raise ValueError(f'B has shape {B.shape}, expected {(nvv, noo)}')
        om1, x1, y1, om2, x2, y2 = _pp_full(B, C, D)
    ec = 0.5 * (om1.sum() - om2.sum() - np.trace(C) - np.trace(D))
    ec1 = om1.sum() - np.trace(C)
    ec2 = -om2.sum() - np.trace(D)
    if (abs(ec - ec1) > PARQUET_PP_EC_CONSISTENCY_TOL
            or abs(ec - ec2) > PARQUET_PP_EC_CONSISTENCY_TOL):
        flag(FLAG_PP_EC_INCONSISTENT,
             f'pp BSE: EcRPA = {ec:.10f} but sum(om_ee) - Tr C = {ec1:.10f} '
             f'and -sum(om_hh) - Tr D = {ec2:.10f} (Quack: RPA1 != RPA2)',
             ParquetBSEUnstable)
    return om1, x1, y1, om2, x2, y2, ec


# ---------------------------------------------------------------------------
# helpers


def _check_spin(spin):
    if spin not in SPINS:
        raise ValueError(f'unknown spin channel {spin!r}; one of {SPINS}')


def _orbital_inputs(eps, eri, nO):
    """eps and eri as float arrays (no copy of a float64 eri) and norb."""
    eps = np.asarray(eps, dtype=float)
    if eps.ndim != 1:
        raise ValueError(f'eps must be one-dimensional, got shape {eps.shape}')
    norb = eps.shape[0]
    eri = np.asarray(eri, dtype=float)
    if eri.shape != (norb,) * 4:
        raise ValueError(f'eri has shape {eri.shape}, expected {(norb,) * 4}')
    if not 0 < int(nO) < norb:
        raise ValueError(f'nO = {nO} must lie between 1 and norb - 1 = {norb - 1}')
    return eps, eri, norb


def _vertex_tensors(vertices, nO, count):
    """The first `count` vertices (eh_singlet, eh_triplet, pp_singlet,
    pp_triplet), checked to share one (norb,)*4 shape."""
    names = ('eh_singlet', 'eh_triplet', 'pp_singlet', 'pp_triplet')[:count]
    tensors = tuple(np.asarray(getattr(vertices, name), dtype=float) for name in names)
    norb = tensors[0].shape[0] if tensors[0].ndim else 0
    for name, t in zip(names, tensors):
        if t.shape != (norb,) * 4:
            raise ValueError(f'vertex {name} has shape {t.shape}, expected {(norb,) * 4}')
    if not 0 < int(nO) < norb:
        raise ValueError(f'nO = {nO} must lie between 1 and norb - 1 = {norb - 1}')
    return tensors


def _square(m, name):
    m = np.asarray(m, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f'{name} must be a square matrix, got shape {m.shape}')
    return m


def _combine(coef, views):
    """c0 v0 + c1 v1 + ... in Quack's left-to-right order, C-ordered."""
    out = np.empty(views[0].shape)
    np.multiply(coef[0], views[0], out=out)
    for c, x in zip(coef[1:], views[1:]):
        out += c * x
    return out


def _pp_pair_block(terms, rows, cols, spin, norm):
    """G[(p,q), (r,s)] = W[p,q,r,s] +- W[p,q,s,r], W = sum_k c_k T_k.

    Rows over the pairs of orbitals rows = (start, stop), columns over those
    of cols, both in `pairs.pp_pairs` order; + for the singlet, - for the
    triplet. norm (singlet only): 'sqrt' divides by sqrt((1 + d_pq)(1 + d_rs))
    (ppRLR_B/D and every kernel block), 'literal' multiplies by
    QUACK_INV_SQRT2 once per diagonal pair (ppRLR_C). Built one row orbital p
    at a time: a lone tensor T[p, q0:stop] is gathered from in place, a sum
    is formed on the column block only, so the temporaries are
    (n_q, norb, norb) at most besides the result.
    """
    r0, r1 = rows
    c0, c1 = cols
    singlet = spin == 'singlet'
    cp, cq = pp_pairs(c0, c1, spin)
    nrow = len(pp_pairs(r0, r1, spin)[0])
    out = np.empty((nrow, len(cp)))
    if nrow == 0 or len(cp) == 0:
        return out
    norb = terms[0][1].shape[0]
    lone = len(terms) == 1 and terms[0][0] == 1.0
    if lone:
        # flat (r, s) of a whole (norb, norb) plane
        flat_rs, flat_sr = cp * norb + cq, cq * norb + cp
    else:
        # flat (r, s) of the column block alone
        nc = c1 - c0
        flat_rs, flat_sr = (cp - c0) * nc + (cq - c0), (cq - c0) * nc + (cp - c0)
    col_diag = cp == cq
    start = 0
    for p in range(r0, r1):
        q0 = p if singlet else p + 1
        nq = r1 - q0
        if nq <= 0:
            continue
        if lone:
            w = terms[0][1][p, q0:r1].reshape(nq, norb * norb)
        else:
            w = np.empty((nq, nc, nc))
            np.multiply(terms[0][0], terms[0][1][p, q0:r1, c0:c1, c0:c1], out=w)
            for c, t in terms[1:]:
                w += c * t[p, q0:r1, c0:c1, c0:c1]
            w = w.reshape(nq, nc * nc)
        g = np.take(w, flat_rs, axis=1)
        if singlet:
            g += np.take(w, flat_sr, axis=1)
            row_diag = np.arange(q0, r1) == p
            if norm == 'sqrt':
                g /= np.sqrt(np.multiply.outer(1.0 + row_diag, 1.0 + col_diag))
            else:
                f = np.where(row_diag, QUACK_INV_SQRT2, 1.0)[:, None]
                g *= np.where(col_diag[None, :], QUACK_INV_SQRT2 * f, f)
        else:
            g -= np.take(w, flat_sr, axis=1)
        out[start:start + nq] = g
        start += nq
    return out


def _eigh_upper(a, what):
    """(w, v) of the symmetric matrix held in the upper triangle of `a`: w
    ascending, v[:, k] the eigenvector of w[k] (Quack's diagonalize_matrix,
    dsyev 'U'). NaN or inf in that triangle, or a LAPACK failure, gives NaN
    eigenpairs and a flag: Quack prints its 'Problem in diagonalize_matrix'
    and carries on with what dsyev left."""
    n = a.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros((0, 0))
    finite = np.isfinite(a)
    if not finite.all() and np.triu(~finite).any():
        flag(FLAG_BSE_NOT_FINITE, f'{what} holds NaN or inf; its eigenpairs are '
             f'returned as NaN', ParquetNotFinite)
        return np.full(n, np.nan), np.full((n, n), np.nan)
    del finite
    try:
        return la.eigh(a, lower=False, driver=PARQUET_EIGH_DRIVER, check_finite=False)
    except la.LinAlgError as err:
        flag(FLAG_BSE_DIAGONALIZATION_FAILED, f'diagonalization of {what} failed ({err}); its '
             f'eigenpairs are returned as NaN', ParquetBSEUnstable)
        return np.full(n, np.nan), np.full((n, n), np.nan)


def _lowdin(s, branch):
    """S^{-1/2} = U diag(u^{-1/2}) U^T (Quack's orthogonalize_matrix, type 1):
    a negative eigenvalue u gives NaN and a zero one inf, as in Quack."""
    u, vecs = _eigh_upper(s, f'the metric overlap of the pp {branch} branch')
    small = u < PARQUET_LOWDIN_SMALL_EIGENVALUE
    if small.any():
        flag(FLAG_PP_LOWDIN_SMALL_EIGENVALUE,
             f'pp BSE: the metric overlap of the {branch} branch has '
             f'{int(small.sum())} eigenvalues below {PARQUET_LOWDIN_SMALL_EIGENVALUE} '
             f'({np.array2string(u[small][:8], precision=3)}); Quack takes '
             f'u^-1/2 regardless', ParquetBSEUnstable)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / np.sqrt(u)
    return vecs @ (inv[:, None] * vecs.T)


def _pp_full(B, C, D):
    """ppRLR without TDA: dgeev of [[C, B], [-B^T, -D]], then sort_ppRPA."""
    nvv, noo = C.shape[0], D.shape[0]
    npp = nvv + noo
    shapes = (nvv,), (nvv, nvv), (noo, nvv), (noo,), (nvv, noo), (noo, noo)
    if npp == 0:
        return tuple(np.zeros(s) for s in shapes)
    m = np.empty((npp, npp), order='F')
    m[:nvv, :nvv] = C
    m[nvv:, nvv:] = -D
    m[:nvv, nvv:] = B
    m[nvv:, :nvv] = -B.T
    if not np.isfinite(m).all():
        flag(FLAG_BSE_NOT_FINITE, 'the pp BSE matrix holds NaN or inf; its '
             'eigenpairs are returned as NaN', ParquetNotFinite)
        return tuple(np.full(s, np.nan) for s in shapes)
    # Quack's workspace query: scipy's default (4 npp) forces unblocked LAPACK
    lwork, info = lapack.dgeev_lwork(npp, compute_vl=0, compute_vr=1)
    lwork = int(lwork) if info == 0 else max(4 * npp, 1)
    wr, wi, _, vr, info = lapack.dgeev(m, compute_vl=0, compute_vr=1, lwork=lwork,
                                       overwrite_a=1)
    del m, _
    if info < 0:
        raise RuntimeError(f'dgeev: illegal argument {-info}')
    if info > 0:
        flag(FLAG_PP_DGEEV_FAILED, f'pp BSE: dgeev did not converge (info {info}); the '
             f'eigenpairs are what it returned, as in Quack', ParquetBSEUnstable)
    ee = np.flatnonzero(wr > PARQUET_PP_BRANCH_SPLIT)
    hh = np.flatnonzero(~(wr > PARQUET_PP_BRANCH_SPLIT))
    if len(ee) != nvv or len(hh) != noo:
        flag(FLAG_PP_BRANCH_COUNT,
             f'pp BSE: {len(ee)} eigenvalues above zero for {nvv} pairs of '
             f'virtuals and {len(hh)} at or below zero for {noo} pairs of '
             f'occupied (instabilities; {int(np.sum(wi != 0.0))} complex, '
             f'their imaginary parts dropped as in Quack)', ParquetBSEUnstable)
    metric = np.concatenate([np.ones(nvv), -np.ones(noo)])
    branches = []
    for idx, size, sign, name in ((ee, nvv, 1.0, 'ee'), (hh, noo, -1.0, 'hh')):
        src = np.full(size, -1)
        om = np.zeros(size)
        k = min(len(idx), size)
        src[:k] = idx[:k]
        om[:k] = wr[idx[:k]]
        order = np.argsort(om, kind='stable')
        om, src = om[order], src[order]
        if k == size:
            z = vr[:, src]
        else:
            z = np.zeros((npp, size))
            z[:, src >= 0] = vr[:, src[src >= 0]]
        if size:
            # Quack: S1 = Z1^T (M Z1), S2 = Z2^T (M (-Z2)), M = diag(1, -1)
            s = z.T @ (z * (sign * metric)[:, None])
            z = z @ _lowdin(s, name)
            del s
        branches.append((om, z))
    del vr
    (om1, z1), (om2, z2) = branches
    return (om1, np.ascontiguousarray(z1[:nvv]), np.ascontiguousarray(z1[nvv:]),
            om2, np.ascontiguousarray(z2[:nvv]), np.ascontiguousarray(z2[nvv:]))
