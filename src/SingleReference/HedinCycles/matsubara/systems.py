"""Systems for the Matsubara backend of the Hedin cycles, in spin orbitals or, for a closed shell, in spatial
orbitals.

A System holds the one-body Hamiltonian h (an N x N matrix over N orbitals), the Coulomb factor vectors B^P
with (pq|rs) = sum_P B^P_pq B^P_rs (each B^P an N x N matrix), the number of electrons and spin, the number
of electrons an orbital takes. With spin = 1 the orbitals are spin orbitals in two blocks: the spatial
orbitals with spin up, then the same ones with spin down (the factors are the same in both blocks). With
spin = 2 they are the spatial orbitals of a closed shell (spin adaptation). Every source of the cycles is a
charge density, so for a closed shell every object of the cycles is the same for both spins (the singlet
channel), and the spatial form holds one spin block with a factor 2 on every closed fermion loop: the Hartree
term, the polarizability and their tangents. The factors are exact: an eigendecomposition of the two-electron
integrals, or the on-site vectors of a Hubbard model. A model Hamiltonian enters through its integral arrays,
so the coupling can be scaled (v -> lambda v) by scaling the factors.
"""
import numpy as np

from src.Base.numerics.constants import HEDIN_FACTOR_TOL


class System:
    """h: (N, N) one-body Hamiltonian, B: (N_aux, N, N) Coulomb factors, nelec: electrons, name, and spin: 1 for
    spin orbitals, 2 for a closed shell in spatial orbitals (an even nelec, nelec / 2 orbitals occupied)."""

    def __init__(self, h, B, nelec, name='', spin=1):
        h, B = np.asarray(h, dtype=float), np.asarray(B, dtype=float)
        if h.ndim != 2 or h.shape[0] != h.shape[1]:
            raise ValueError(f'h must be a square matrix, got shape {h.shape}')
        if B.ndim != 3 or B.shape[1:] != h.shape:
            raise ValueError(f'B must have shape (N_aux, {h.shape[0]}, {h.shape[0]}), got {B.shape}')
        if spin not in (1, 2) or isinstance(spin, bool):
            raise ValueError(f'spin must be 1 (spin orbitals) or 2 (a closed shell in spatial orbitals), got {spin!r}')
        if not isinstance(nelec, (int, np.integer)) or isinstance(nelec, bool):
            raise ValueError(f'nelec must be an integer, got {nelec!r}')
        if spin == 2 and nelec % 2:
            raise ValueError(f'a closed shell in spatial orbitals needs an even number of electrons, got {nelec}')
        if not 0 < nelec // spin < h.shape[0]:
            raise ValueError(f'nelec must fill between 1 and {h.shape[0] - 1} orbitals of {spin} electron(s) each, '
                             f'got {nelec!r}')
        self.h, self.B, self.nelec, self.name, self.spin = h, B, int(nelec), name, spin

    @property
    def norb(self):
        return self.h.shape[0]

    @property
    def nocc(self):
        """The number of occupied orbitals (spin orbitals for spin = 1, spatial orbitals for spin = 2)."""
        return self.nelec // self.spin

    def scaled(self, lam):
        """The same system with the interaction scaled, v -> lam v (the factors by sqrt(lam))."""
        if lam < 0:
            raise ValueError('a negative coupling scale has no real Coulomb factors')
        return System(self.h, np.sqrt(lam) * self.B, self.nelec, f'{self.name}, v x {lam}', self.spin)

    def eri(self):
        """(pq|rs) over the system's orbitals, rebuilt from the factors (for small systems and tests)."""
        return np.einsum('Ppq,Prs->pqrs', self.B, self.B)


def spin_orbital(h_spatial, factors_spatial, nelec, name=''):
    """A System from spatial h (n x n) and spatial factors (N_aux, n, n): both spin blocks."""
    n = h_spatial.shape[0]
    h = np.zeros((2 * n, 2 * n))
    h[:n, :n] = h[n:, n:] = h_spatial
    B = np.zeros((factors_spatial.shape[0], 2 * n, 2 * n))
    B[:, :n, :n] = B[:, n:, n:] = factors_spatial
    return System(h, B, nelec, name)


def spatial(h_spatial, factors_spatial, nelec, name=''):
    """A closed shell in spatial orbitals (spin = 2): the spin-adapted form of spin_orbital(...)."""
    return System(h_spatial, factors_spatial, nelec, name, spin=2)


def coulomb_factors(eri_spatial):
    """Exact factors B^P_ij of (ij|kl) by eigendecomposition of the (ij),(kl) matrix, keeping the
    eigenvalues above HEDIN_FACTOR_TOL times the largest."""
    n = eri_spatial.shape[0]
    V = eri_spatial.reshape(n * n, n * n)
    w, U = np.linalg.eigh(0.5 * (V + V.T))
    if w.size == 0 or w.max() <= 0:
        return np.zeros((0, n, n))
    keep = w > HEDIN_FACTOR_TOL * w.max()
    return (U[:, keep] * np.sqrt(w[keep])).T.reshape(-1, n, n)


def hubbard_dimer(t=1.0, U=1.0, eps=(0.0, 0.0), nelec=2, spin_adapted=False):
    """Two sites with hopping t, on-site interaction U and site energies eps: h = [[e1, -t], [-t, e2]],
    (ii|ii) = U. The factors are the on-site vectors sqrt(U) |i><i|. spin_adapted=True gives the closed shell
    in spatial orbitals (an even nelec)."""
    if U < 0:
        raise ValueError('a negative U has no real Coulomb factors')
    h = np.array([[eps[0], -t], [-t, eps[1]]], dtype=float)
    B = np.zeros((2, 2, 2))
    B[0, 0, 0] = B[1, 1, 1] = np.sqrt(U)
    name = f'Hubbard dimer t={t} U={U} eps={tuple(eps)} N={nelec}'
    return spatial(h, B, nelec, name) if spin_adapted else spin_orbital(h, B, nelec, name)


def from_pyscf(mf, name='', spin_adapted=False):
    """A molecule in the orthonormal basis of a pyscf mean field's orbitals (all electrons). spin_adapted=True
    gives the closed shell in spatial orbitals (an even number of electrons)."""
    from pyscf import ao2mo
    mol = mf.mol
    C = np.asarray(mf.mo_coeff)
    if C.ndim != 2:
        raise ValueError('from_pyscf needs a restricted mean field (one set of orbitals)')
    n = C.shape[1]
    h = C.T @ mf.get_hcore() @ C
    eri = ao2mo.restore(1, ao2mo.full(mol, C), n)
    make = spatial if spin_adapted else spin_orbital
    return make(h, coulomb_factors(eri), mol.nelectron, name or mol.atom.__str__()[:40])
