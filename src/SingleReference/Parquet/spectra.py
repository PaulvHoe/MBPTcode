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

Each vertex enters the kernels of the next iteration (Eq. (47)) in roles,
its insertions: an eh vertex crossed into the eh kernel, the transverse
particle-hole channel ph-bar ('phbar_eh'), and into the pp kernel ('eh_pp'),
where it enters twice, direct (the longitudinal channel ph) and swapped
(ph-bar), two terms that exchanging the pair's two particles turns into each
other and that so take one tensor, or the pp kernel would lose its
antisymmetry; a pp vertex direct into the eh kernel ('pp_eh'). A
per-insertion vertex builder gives each role its own tensor, which the loop
carries as `RestrictedInsertions` (six tensors) or `SpinOrbitalInsertions`
(three); the default carries one tensor per vertex, inserted in all of its
roles.
"""
from typing import NamedTuple

import numpy as np

#: The roles of an eh vertex: crossed into the eh kernel (the transverse
#: ph-bar channel), and into the pp kernel (its direct term, the longitudinal
#: ph channel, and its swapped one, ph-bar, exchange partners: one tensor).
EH_ROLES = ('phbar_eh', 'eh_pp')
#: The role of a pp vertex: direct into the eh kernel.
PP_ROLES = ('pp_eh',)
#: (vertex, role) of each tensor of a `RestrictedInsertions`, in its order.
RESTRICTED_INSERTIONS = (('eh_singlet', 'phbar_eh'), ('eh_singlet', 'eh_pp'),
                         ('eh_triplet', 'phbar_eh'), ('eh_triplet', 'eh_pp'),
                         ('pp_singlet', 'pp_eh'), ('pp_triplet', 'pp_eh'))
#: (vertex, role) of each tensor of a `SpinOrbitalInsertions`, in its order.
SPIN_ORBITAL_INSERTIONS = (('eh', 'phbar_eh'), ('eh', 'eh_pp'), ('pp', 'pp_eh'))


def insertion_weights(stored):
    """The two-body DIIS weight of each stored tensor, (vertex, role) in
    `stored`: one over the number of tensors of its vertex (an eh vertex's
    two a half each), so that each vertex counts once in the Gram matrix, as
    in Quack's vector of vertices."""
    names = [name for name, _ in stored]
    return tuple(1.0 / names.count(name) for name in names)


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


class RestrictedInsertions(NamedTuple):
    """The closed-shell reducible vertices per insertion, each (norb,)*4: a
    field <vertex>_<role> per entry of RESTRICTED_INSERTIONS, in its order
    (each eh vertex in its two roles, each pp vertex in its one)."""
    eh_singlet_phbar_eh: np.ndarray
    eh_singlet_eh_pp: np.ndarray
    eh_triplet_phbar_eh: np.ndarray
    eh_triplet_eh_pp: np.ndarray
    pp_singlet_pp_eh: np.ndarray
    pp_triplet_pp_eh: np.ndarray

    def eh_kernel(self):
        """The vertices the eh kernels and their effective integrals read:
        the eh ones in their 'phbar_eh' role, the pp ones in 'pp_eh'."""
        return RestrictedVertices(self.eh_singlet_phbar_eh, self.eh_triplet_phbar_eh,
                                  self.pp_singlet_pp_eh, self.pp_triplet_pp_eh)

    def pp_kernel(self):
        """The vertices the pp kernels and their effective integrals read: the
        eh ones in their 'eh_pp' role, in the direct and the swapped terms
        alike; the pp ones, which no pp kernel reads, are the 'pp_eh' ones."""
        return RestrictedVertices(self.eh_singlet_eh_pp, self.eh_triplet_eh_pp,
                                  self.pp_singlet_pp_eh, self.pp_triplet_pp_eh)


class SpinOrbitalInsertions(NamedTuple):
    """The spin-orbital reducible vertices per insertion, each (nso,)*4: a
    field <vertex>_<role> per entry of SPIN_ORBITAL_INSERTIONS, in its order."""
    eh_phbar_eh: np.ndarray
    eh_eh_pp: np.ndarray
    pp_pp_eh: np.ndarray

    def eh_kernel(self):
        """The vertices the eh kernel and its effective integrals read: eh in
        its 'phbar_eh' role, pp in 'pp_eh'."""
        return SpinOrbitalVertices(self.eh_phbar_eh, self.pp_pp_eh)

    def pp_kernel(self):
        """The vertices the pp kernel and its effective integrals read: eh in
        its 'eh_pp' role, in the direct and the swapped term alike; pp, which
        the pp kernel does not read, the 'pp_eh' one."""
        return SpinOrbitalVertices(self.eh_eh_pp, self.pp_pp_eh)
