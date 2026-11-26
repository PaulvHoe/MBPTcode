"""The start lines of the Hedin cycles on a System: self-consistent Hartree ('H') or Hartree-Fock ('HF').

Both are solved at zero temperature by aufbau: the nocc lowest levels of the Fock matrix F = h + s J[D]
(Hartree) or F = h + s J[D] - K[D] (Hartree-Fock) are occupied, with D the density matrix of one spin block,
s the system's spin (1 for spin orbitals, 2 for a closed shell in spatial orbitals, where nocc = nelec / 2),
J[D]_pq = sum_P B^P_pq tr(B^P D) and K[D] = sum_P B^P D B^P. The chemical potential mu lies in
the middle of the gap between the highest occupied and the lowest empty level. A start line without a
gap (a degenerate, partly filled level, as the Hartree line of a one-electron system has) cannot be
put on the Matsubara axis at low temperature and is refused.
"""
import numpy as np

from src.Base.numerics.constants import HEDIN_GAP_MIN
from src.Base.numerics.numerics import HedinNumerics


class StartLine:
    """The converged start: e (levels), C (their orbitals in the system's basis), nocc, mu, gap,
    and the Fock matrix F in the system's basis."""

    def __init__(self, start, e, C, nocc, F):
        self.start, self.e, self.C, self.nocc, self.F = start, e, C, nocc, F
        self.gap = e[nocc] - e[nocc - 1]
        self.mu = 0.5 * (e[nocc] + e[nocc - 1])


def hartree(B, D):
    return np.einsum('Ppq,Prs,sr->pq', B, B, D)


def exchange(B, D):
    return np.einsum('Ppr,rs,Psq->pq', B, D, B)


def fock(system, D, start):
    F = system.h + system.spin * hartree(system.B, D)
    if start == 'HF':
        F = F - exchange(system.B, D)
    return F


def start_line(system, start, hedin=None):
    """Solve the start's mean field by aufbau with DIIS on the commutator FD - DF, to the density change
    hedin.scf_tol in at most hedin.scf_max_iter iterations (a Numerics hedin section, its defaults if None)."""
    hedin = HedinNumerics() if hedin is None else hedin
    if start not in ('H', 'HF'):
        raise ValueError(f"unknown start {start!r}: choose 'H' or 'HF'")
    nocc = system.nocc
    e, C = np.linalg.eigh(system.h)
    D = C[:, :nocc] @ C[:, :nocc].T
    focks, errors = [], []
    for _ in range(hedin.scf_max_iter):
        F = fock(system, D, start)
        err = F @ D - D @ F
        focks.append(F)
        errors.append(err)
        focks, errors = focks[-8:], errors[-8:]
        if len(focks) > 1:
            k = len(focks)
            A = np.zeros((k + 1, k + 1))
            A[:k, :k] = [[np.vdot(a, b).real for b in errors] for a in errors]
            A[:k, k] = A[k, :k] = -1
            rhs = np.zeros(k + 1)
            rhs[k] = -1
            try:
                coef = np.linalg.solve(A, rhs)[:k]
                F = sum(c * f for c, f in zip(coef, focks))
            except np.linalg.LinAlgError:
                pass
        e, C = np.linalg.eigh(F)
        D_new = C[:, :nocc] @ C[:, :nocc].T
        change = np.abs(D_new - D).max()
        D = D_new
        if change < hedin.scf_tol and np.abs(err).max() < np.sqrt(hedin.scf_tol):
            break
    else:
        raise RuntimeError(f'the {start} start did not converge in {hedin.scf_max_iter} iterations '
                           f'(last density change {change:.1e})')
    F = fock(system, D, start)
    e, C = np.linalg.eigh(F)
    line = StartLine(start, e, C, nocc, F)
    if line.gap < HEDIN_GAP_MIN:
        raise ValueError(f'the {start} start of {system.name!r} has no gap ({line.gap:.1e} Ha between the '
                         f'highest occupied and the lowest empty level): the chemical potential cannot be '
                         f'put between them, so the Matsubara cycles at low temperature are refused')
    return line
