"""What the two-body problems hand the one-body one, and what the two-body loop
carries from one iteration to the next.

A `Channel` is one BSE's spectrum: its eigenvalues `om` and the generalized
effective integrals `rho[p, q, n]` of each eigenvector n (M^eh_{pq,n},
M^ee_{pq,m} or M^hh_{pq,m} of Marie and Loos, Eqs. (38) and (44)). The
self-energy is built from these alone.

The closed-shell problem has six channels (`RestrictedSpectra`): the singlet
and triplet eh BSE, and the 2p (ee) and 2h (hh) branches of the singlet and
triplet pp BSE. The spin-orbital problem has three (`SpinOrbitalSpectra`).

The reducible vertices Phi[p, q, r, s] the loop iterates are
`RestrictedVertices` (singlet and triplet, eh and pp) and
`SpinOrbitalVertices`. They are the i*Phi of the paper (real for real
orbitals), in the index order Quack stores them.
"""
from typing import NamedTuple

import numpy as np


class Channel(NamedTuple):
    """One BSE spectrum: eigenvalues om (n,) and effective integrals
    rho (norb, norb, n)."""
    om: np.ndarray
    rho: np.ndarray


class RestrictedSpectra(NamedTuple):
    """The six channels of the closed-shell parquet self-energy, in the
    argument order of Quack's R_Parquet_self_energy_diag."""
    eh_singlet: Channel
    eh_triplet: Channel
    ee_singlet: Channel
    ee_triplet: Channel
    hh_singlet: Channel
    hh_triplet: Channel


class SpinOrbitalSpectra(NamedTuple):
    """The three channels of the spin-orbital parquet self-energy."""
    eh: Channel
    ee: Channel
    hh: Channel


class RestrictedVertices(NamedTuple):
    """The four closed-shell reducible vertices, each (norb,)*4."""
    eh_singlet: np.ndarray
    eh_triplet: np.ndarray
    pp_singlet: np.ndarray
    pp_triplet: np.ndarray

    @classmethod
    def zeros(cls, norb):
        """Vanishing reducible vertices: the FLEX starting point."""
        return cls(*(np.zeros((norb,) * 4) for _ in range(4)))


class SpinOrbitalVertices(NamedTuple):
    """The two spin-orbital reducible vertices, each (nso,)*4."""
    eh: np.ndarray
    pp: np.ndarray

    @classmethod
    def zeros(cls, nso):
        """Vanishing reducible vertices: the FLEX starting point."""
        return cls(np.zeros((nso,) * 4), np.zeros((nso,) * 4))
