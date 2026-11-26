"""The operations a backend of the Hedin cycles provides, each with its equation.

The cycle loop (schemes.py) is written against these operations and against + and - on the objects
they return, so the same loop runs on every backend: the zero-dimensional one (zero_d.py), where
every object is a number or a power series in the coupling, and the finite-basis Matsubara one
(matsubara/).
Each operation carries its own factors of i and its signs, so the loop holds none. In zero dimensions
every factor is +1, the counting convention in which every diagram has weight +1 and a closed fermion
loop the weight ell.

Objects and their indices (Hedin's notation, a number is a space-time-spin point, 1+ is 1 a moment
later, and the integral over repeated points is implied):
- G(1,2) a line (a one-particle Green's function), G^(0) the start line, Sigma(1,2) a self-energy,
  v(1,2) the bare Coulomb interaction, P(1,2) a polarizability and W(1,2) a screened interaction.
- Gamma(1,2;3) a vertex. (1,2) is its pair end and 3 its density end, where v or W attaches.
- L(1,2;3) = L(1,2;3,3+) the two-particle correlation function with its density end closed, and
  L_0(1,2;3,4) = -i G(1,3) G(4,2).
- K(1,2;3,4) = delta Sigma(1,2) / delta G(3,4) a kernel: always the plain derivative of a
  self-energy with respect to a line. Its first pair belongs to Sigma, its second acts on pair ends.
- The legs of a scheme along the map: L^(1)(1,2;3), the start's two-particle correlation function, or
  G^(0) Gamma^(1) G^(0)(1,2;3), the start's vertex with a start line on each leg.

Derivatives are taken with respect to lines only:
- along_map(X, legs)(1,2;3) = [delta X(1,2) / delta G^(0)(4,5)] legs(4,5;3) differentiates a cycle
  quantity X, which is an explicit functional of the start line, and contracts the derivative with the
  pair end of the legs.
- kernel(S, G)(1,2;3,4) = delta S[G](1,2) / delta G(3,4) differentiates a level functional S, the
  formula of one cycle as a function of a generic line, with respect to its own argument.

Two requirements go beyond the list of operations, and the 0-D backend meets both by construction:
- along_map must differentiate X through everything X was built from, the outputs of earlier
  along_map and kernel calls and the legs' own dependence on G^(0) included. From cycle 3 on, SG-I/L
  differentiates a Sigma^(n-1) that already holds along_map(Sigma^(n-2), L^(1)), which takes tangents
  of tangents.
- kernel(S, G) at a line G that is itself the perturbed line of an outer derivative must return the
  kernel at that perturbed line. The level functional S_k calls kernel(S_(k-1), G) inside, so the
  kernel of the next level needs delta K_(k-1) / delta G (a second derivative of S_(k-1)).
"""
from typing import Callable, Protocol

#: Every operation of Backend by name. run_cycles refuses a backend that lacks one.
OPERATIONS = ('line0', 'zero_self_energy', 'sigma_h', 'sigma_x', 'dyson', 'vertex_legs',
              'polarizability', 'screen', 'sigma_gw', 'solve_vertex', 'solve_l', 'apply_kernel',
              'sigma_il', 'sg_vertex', 'along_map', 'kernel')


class Backend(Protocol):
    """What the cycle loop needs of a backend."""

    #: G^(0), the start line: self-consistent Hartree (start 'H') or Hartree-Fock (start 'HF').
    line0: object

    def zero_self_energy(self):
        """A self-energy equal to zero."""

    def sigma_h(self, G):
        """Sigma_H[G](1,2) = -i delta(1,2) v(1,3) G(3,3+), the Hartree self-energy of the line G."""

    def sigma_x(self, G):
        """Sigma_x[G](1,2) = i G(1,2) v(1+,2), the exchange self-energy of the line G."""

    def dyson(self, d_sigma):
        """G(1,2) = G^(0)(1,2) + G^(0)(1,3) d_sigma(3,4) G(4,2), the Dyson equation with the start line."""

    def vertex_legs(self, G, vertex):
        """(G Gamma G)(1,2;3) = G(1,4) Gamma(4,5;3) G(5,2): a vertex with a line on each leg."""

    def polarizability(self, G, vertex):
        """P(1,2) = -i G(1,3) G(4,1+) Gamma(3,4;2)."""

    def screen(self, P):
        """W(1,2) = v(1,2) + v(1,3) P(3,4) W(4,2)."""

    def sigma_gw(self, G, W, vertex):
        """i G(1,3) W(4,1+) Gamma(3,2;4), the exchange-correlation self-energy of the Gamma/W route.

        W meets the density end of the vertex first, as Hedin's derivation gives it before the symmetry
        W(1,4) = W(4,1) of the exact theory is used. A truncated cycle has P^(n) = G G Gamma^(n) with its
        vertex at one end only, so its W is not symmetric, and only this order keeps the Gamma/W route an
        exact rewriting of the I/L route when the vertex is a response of the line. A backend may offer the
        printed order W(1+,4) as an option (the Matsubara backend's w_order). In zero dimensions the two
        coincide."""

    def solve_vertex(self, G, K):
        """Gamma(1,2;3) = delta(1,2) delta(1,3) + K(1,2;4,5) G(4,6) G(7,5) Gamma(6,7;3),
        the BSE for Hedin's vertex with the kernel K."""

    def solve_l(self, G, K):
        """L(1,2;3) = L_0(1,2;3,3+) + L_0(1,2;4,5) Xi(4,5;6,7) L(6,7;3) with Xi = i K and
        L_0(1,2;4,5) = -i G(1,4) G(5,2): the BSE of the I/L route."""

    def apply_kernel(self, K, L):
        """(K L)(1,2;3) = K(1,2;4,5) L(4,5;3): the kernel acts on the pair end of L, and the result
        has the index structure of along_map(Sigma, L^(1))."""

    def sigma_il(self, G, X):
        """i v(1,3) G(1,4) [i X](4,2;3): the I/L closure i v G Xi L with Xi L = i X, where X is a plain
        derivative contracted with L, K L or along_map(Sigma, L^(1))."""

    def sg_vertex(self, lam, lam_h):
        """delta(1,2) delta(1,3) + lam(1,2;4) [1 + lambda_H]^-1(4,3): the vertex of SG-Gamma/W.

        lam is the three-point along-the-map derivative of an exchange-correlation self-energy.
        lam_h is the along-the-map derivative of a difference of Hartree self-energies, which is local,
        lam_h(1,2;3) = delta(1,2) lambda_H(1;3). The two-point lambda_H(4;3) is what is inverted (no
        BSE), and the inverse acts on the density end of lam from the right. In a factorized Coulomb
        basis lambda_H is the matrix of the Hartree tangent's coefficients on the factor vectors, not
        its projection onto them (which would multiply it by the factors' overlap).
        """

    def along_map(self, X, legs):
        """[delta X(1,2) / delta G^(0)(4,5)] legs(4,5;3): the derivative along the map, contracted
        with the pair end of legs. It sees through every dependence of X on G^(0) (see the module)."""

    def kernel(self, S: Callable, G):
        """delta S[G](1,2) / delta G(3,4) at G: the derivative at fixed form of a level functional S.
        At a perturbed line it is the kernel at that line (see the module)."""


def missing_operations(backend):
    """The names of OPERATIONS a backend does not provide."""
    return [name for name in OPERATIONS if not hasattr(backend, name)]
