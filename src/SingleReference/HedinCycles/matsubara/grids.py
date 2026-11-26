"""Matsubara grids of the Hedin cycles.

beta is the inverse temperature, nu_n = (2n + 1) pi / beta a fermionic and Omega_m = 2 m pi / beta a
bosonic Matsubara frequency. One-particle quantities (G, Sigma) live on n in [-n_fermion, n_fermion),
two-point bosons (P, W) and the transfers of every vertex's frequency-independent part on
m in [-m_boson, m_boson], and the frequency-dependent part of a vertex on n in [-n_vertex, n_vertex)
for transfers m in [-m_vertex, m_vertex] (zero outside that box). Outside its grid a line is replaced
by its model line, the static part of its inverse alone, whose sums are done exactly.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grids:
    beta: float
    n_fermion: int
    n_vertex: int
    m_boson: int
    m_vertex: int

    def __post_init__(self):
        for name in ('n_fermion', 'n_vertex', 'm_boson', 'm_vertex'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer, got {value!r}')
        if not self.beta > 0:
            raise ValueError(f'beta must be positive, got {self.beta!r}')
        if self.n_vertex > self.n_fermion or self.m_vertex > self.m_boson:
            raise ValueError('the vertex box must lie inside the one-particle and boson grids')

    @classmethod
    def from_section(cls, hedin, gap):
        """The grids of a Numerics hedin section, beta from HEDIN_BETA_GAP / gap when the section leaves it
        unset (gap the start line's, in Hartree)."""
        from src.Base.numerics.constants import HEDIN_BETA_GAP
        beta = hedin.beta if hedin.beta is not None else HEDIN_BETA_GAP / gap
        return cls(beta=float(beta), n_fermion=int(hedin.n_fermion), n_vertex=int(hedin.n_vertex),
                   m_boson=int(hedin.m_boson), m_vertex=int(hedin.m_vertex))

    def section(self):
        """These grids as the fields of a Numerics hedin section."""
        return {'beta': self.beta, 'n_fermion': self.n_fermion, 'n_vertex': self.n_vertex,
                'm_boson': self.m_boson, 'm_vertex': self.m_vertex}

    def nu(self, n):
        return (2 * np.asarray(n) + 1) * np.pi / self.beta

    def omega(self, m):
        return 2 * np.asarray(m) * np.pi / self.beta

    @property
    def fermion_n(self):
        return np.arange(-self.n_fermion, self.n_fermion)

    @property
    def vertex_n(self):
        return np.arange(-self.n_vertex, self.n_vertex)

    @property
    def boson_m(self):
        return np.arange(-self.m_boson, self.m_boson + 1)

    @property
    def vertex_m(self):
        return np.arange(-self.m_vertex, self.m_vertex + 1)

    def boson_index(self, m):
        """Position of the bosonic index m on the boson grid."""
        return np.asarray(m) + self.m_boson
