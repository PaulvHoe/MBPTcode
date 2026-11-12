"""BSE(2) excitation energies (van Hoegaerden and Foerster): the second-order
kernel of `second_order_kernel` on the bare first-order BSE, and the dBSE(2)
dynamical correction.

    A = A(1) + sym A(2)(0),   B = B(1) + sym B(2),   sym M = (M + M^T) / 2.

A(1) and B(1) are the TDHF blocks with bare v at the energies eps, as in the
paper's second cycle; first_order='screened' puts W0 in their direct term, the
double-counting 'exch' variant of the SI. eps are the quasiparticle energies of
the sequence's previous cycle: Sigma_TDHF@L_TDHF for BSE(2)@HF, G0F2 for
BSE(1)@GF2 and Sigma_BSE@L_BSE for BSE(2)@GW, the orbitals staying RHF; naming
the sequence runs it first (`second_order_qp.qp_sequence`, AMS's cycles and
root rule). W0 is the static RPA at those same energies, with Tamm-Dancoff
screening when tda is set, as the paper's TDA runs screen in every cycle.

dBSE(2) takes one renormalized Newton step from each static root (Omega, X, Y),
normalized as X^T X - Y^T Y = 1:

    dOmega    = X^T [A(2)(+Omega) - A(2)(0)] X + Y^T [A(2)(-Omega) - A(2)(0)] Y
    slope     = X^T A(2)'(+Omega) X - Y^T A(2)'(-Omega) Y,    zeta = 1 / (1 - slope)
    Omega_dyn = Omega + zeta dOmega

with Y = 0 under the TDA. B(2) does not depend on omega, so it adds nothing.
These are the signs of the paper's corrected Eqs. 29 and 33 and of the Fortran
Dynamicperturbation. The paper's runs corrected the lowest 30 roots.
"""
import numpy as np
from pyscf import scf

from src.Base.numerical_flags import collecting, flag, gathered
from src.Base.pyscf.pyscf_interface import get_density_fitting_coefficients
from src.SingleReference.BSE.second_order_qp import qp_sequence
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.second_order_kernel import (
    CHANNELS, SecondOrderKernel, static_screening_aux)

#: How A(1) and B(1) treat the direct term: bare v (the paper) or W0 (the SI's 'exch').
FIRST_ORDER = ('bare', 'screened')


def _sym(M):
    return 0.5 * (M + M.T)


def second_order_bse(mf, eps=None, kernel='2sosex', spin='singlet', tda=False,
                     channels=CHANNELS, first_order='bare', W_aux=None,
                     dynamical_roots=30):
    """Static BSE(2) roots and their dBSE(2) correction for a closed-shell RHF.

    mf: the converged RHF; its orbitals are kept. eps: the energies on the
    diagonal and in the denominators (Ha, all orbitals), the mean field's when
    None, or the name of a quasiparticle sequence of `second_order_qp.
    QP_SEQUENCES` ('Sigma_TDHF@L_TDHF', 'Sigma_BSE@L_BSE', 'G0F2', 'G0W0'),
    run first with the same tda and its last cycle's energies used.
    kernel: the W-slot rule of SecondOrderKernel ('v' for BSE(2)@HF and
    BSE(1)@GF2, '2sosex' for BSE(2)@GW, 'average'). spin: 'singlet' or
    'triplet'. tda: drop B, and screen W0 in the Tamm-Dancoff approximation.
    channels: the kernel sums kept, 'pp' and 'hh' both off being the paper's
    'pp off'. first_order: one of FIRST_ORDER. W_aux: the static W0 in the
    whitened auxiliary basis, built at eps when None and the kernel or
    first_order needs it. dynamical_roots: how many of the lowest roots get the
    dBSE(2) step.

    Returns a dict: 'omega' (static roots, Ha), 'X', 'Y' (columns are roots),
    'dynamical' (per corrected root: 'omega' = Omega_dyn, 'delta' = dOmega,
    'slope', 'zeta', 'y_norm2' = |Y|^2), the inputs 'eps', 'W_aux', 'qp',
    the sequence's record (`qp_sequence`) when eps names one, else None, and
    'numerical_flags', the named flags raised on the way.
    """
    flags = []
    with collecting(flags):
        out = _second_order_bse(mf, eps, kernel, spin, tda, channels, first_order,
                                W_aux, dynamical_roots)
    out['numerical_flags'] = gathered(flags)
    return out


def _second_order_bse(mf, eps, kernel, spin, tda, channels, first_order, W_aux,
                      dynamical_roots):
    if isinstance(mf, scf.uhf.UHF):
        raise NotImplementedError('BSE(2) is implemented for closed-shell RHF references only')
    if first_order not in FIRST_ORDER:
        raise ValueError(f'first_order {first_order!r} is not one of {FIRST_ORDER}')
    nocc = int(np.count_nonzero(mf.mo_occ > 0))
    coeff_df = np.asarray(get_density_fitting_coefficients(mf.mol, mf))
    qp = None
    if isinstance(eps, str):
        qp = qp_sequence(mf, eps, tda=tda, coeff_df=coeff_df)
        eps = qp['eps']
    eps = np.asarray(mf.mo_energy if eps is None else eps, dtype=float)
    if W_aux is None and (kernel != 'v' or first_order == 'screened'):
        W_aux = static_screening_aux(eps, coeff_df, nocc, tda=tda)

    second = SecondOrderKernel(eps, coeff_df, nocc, spin=spin, rule=kernel,
                               W_aux=None if kernel == 'v' else W_aux, channels=channels)
    A1, B1 = LinearResponseSolver(eps, coeff_df=coeff_df).build_casida_matrices(
        nocc, lBSE=True, W_aux=W_aux if first_order == 'screened' else None,
        triplet=spin == 'triplet')
    A = A1 + _sym(second.A())
    if tda:
        omega, X, Y = CasidaSolver(A, np.zeros_like(A)).solve(tda=True)
    else:
        omega, X, Y = CasidaSolver(A, B1 + _sym(second.B())).solve()
    omega, X, Y = np.asarray(omega), np.asarray(X), np.asarray(Y)
    return {'omega': omega, 'X': X, 'Y': Y, 'eps': eps, 'W_aux': W_aux, 'qp': qp,
            'dynamical': dynamical_correction(second, omega, X, Y, dynamical_roots, tda)}


def dynamical_correction(second, omega, X, Y, count, tda=False):
    """dBSE(2) for the lowest `count` roots (module docstring): a dict of
    per-root arrays 'omega', 'delta', 'slope', 'zeta', 'y_norm2'.

    second: the SecondOrderKernel the static problem was built from, so that
    A(2)(0) is the static reference. A root with zeta <= 0 (the Fortran's
    BREAKDOWN) raises the named flag 'second_order_bse_zeta_nonpositive'.
    """
    n = min(int(count), omega.size)
    out = {key: np.zeros(n) for key in ('omega', 'delta', 'slope', 'zeta', 'y_norm2')}
    for m in range(n):
        w, x = omega[m], X[:, m]
        delta = second.form(x, w) - second.form(x)
        slope = second.form(x, w, power=2)
        if not tda:
            y = Y[:, m]
            delta += second.form(y, -w) - second.form(y)
            slope -= second.form(y, -w, power=2)
            out['y_norm2'][m] = y @ y
        zeta = 1.0 / (1.0 - slope)
        out['delta'][m], out['slope'][m], out['zeta'][m] = delta, slope, zeta
        out['omega'][m] = w + zeta * delta
    bad = np.flatnonzero(out['zeta'] <= 0.0)
    if bad.size:
        flag('second_order_bse_zeta_nonpositive',
             f'the dBSE(2) renormalization zeta = 1/(1 - slope) is not positive for roots '
             f'{(bad + 1).tolist()}: the first-order step is meaningless there (a kernel pole '
             'lies close to the static root)', RuntimeWarning)
    return out
