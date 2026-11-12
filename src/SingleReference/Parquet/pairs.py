"""The pair orderings of the parquet two-body problems, and the spin weights
the closed-shell modules share.

Every pair space is laid out as Quack lays it out (Marie and Loos,
J. Chem. Phys. 163, 194115 (2025); the Quack routines named below), so a
matrix or an amplitude built here lines up element by element with its Quack
counterpart:

    eh (1h1p) pairs      ia = i * nV + (a - nO): occupied i outer, virtual a
                         inner (phRLR_A, phGLR_A)
    pp pairs, singlet    (p, q) with p <= q over one block (virtual or
                         occupied), p outer and q inner (ppRLR_C/D, ispin 1)
    pp pairs, triplet    (p, q) with p < q, same order (ppRLR_C/D, ispin 2)
    pp pairs, spin       p < q, same order (ppGLR_C/D)
    orbitals

The pp orders are numpy's upper triangle, row by row (`np.triu_indices` with
k = 0 or 1). Indices returned here are absolute orbital indices (a virtual
index counts from 0, not from the first virtual), ready to index an ERI or a
reducible-vertex tensor.

Only the all-electron problem is laid out: Quack's frozen core (nC) and frozen
virtuals (nR) are not carried over, and a caller asking for them is refused
by the drivers.

The closed-shell spin weights Quack hard-codes in its self-energies
(R_Parquet_self_energy_diag, R_Parquet_self_energy_SRG_sp,
R_Parquet_self_energy_eta) are held once here, signed and applied additively
everywhere: an eh term carries wd <pa|ij> + wx <pa|ji> (`EH_SPIN_WEIGHTS`),
a pp term w <pa|ij> (`PP_SPIN_WEIGHTS`); and so are the copies of a triplet
spectrum in the spin-orbital one (`TRIPLET_MULTIPLICITY`), which make a
closed-shell channel energy the spin-orbital one.
"""
import numpy as np

#: The pp spin channels of the closed-shell problem, and the spin-orbital one
#: (antisymmetric pairs p < q).
PP_SPINS = ('singlet', 'triplet', 'spin-orbital')

#: The eh spin channels of the closed-shell problem.
EH_SPINS = ('singlet', 'triplet')

#: (wd, wx) of the direct <pa|ij> and the exchange <pa|ji> integral in Quack's
#: closed-shell eh self-energy terms, applied as wd direct + wx exchange:
#: singlet 0.5 <pa|ij> - <pa|ji>, triplet 1.5 <pa|ij> (its exchange weight is an
#: exact 0, Quack's '- 0d0*ERI').
EH_SPIN_WEIGHTS = {'singlet': (0.5, -1.0), 'triplet': (1.5, 0.0)}

#: Weight of the integral in Quack's closed-shell pp self-energy terms: singlet
#: 0.5, triplet 1.5 (its -0.5 / -1.5 on the 2h1p-like terms is this times -1).
PP_SPIN_WEIGHTS = {'singlet': 0.5, 'triplet': 1.5}

#: Copies of a closed-shell triplet BSE's spectrum in the spin-orbital one
#: (M_s = -1, 0, +1): the spin-orbital channel energy is Ec_singlet +
#: 3 Ec_triplet, as Quack's R_print_parquet_1b prints it, and as G_evParquet's
#: own Ec_eh and Ec_pp come out (water/6-31G FLEX, full BSEs: to 1.6e-12 and
#: 3e-13 Ha).
TRIPLET_MULTIPLICITY = 3.0


def eh_pairs(nO, nV):
    """(i, a) of the 1h1p pairs in Quack order, a absolute (nO <= a < nO + nV)."""
    i = np.repeat(np.arange(nO), nV)
    a = np.tile(np.arange(nO, nO + nV), nO)
    return i, a


def pp_pairs(start, stop, spin):
    """(p, q) of the pairs over orbitals start..stop-1 in Quack order.

    spin: 'singlet' (p <= q) or 'triplet' / 'spin-orbital' (p < q).
    """
    if spin not in PP_SPINS:
        raise ValueError(f'unknown pp spin channel {spin!r}; one of {PP_SPINS}')
    k = 0 if spin == 'singlet' else 1
    p, q = np.triu_indices(stop - start, k=k)
    return p + start, q + start


def pp_pair_count(n, spin):
    """Number of pairs over n orbitals: n(n+1)/2 (singlet) or n(n-1)/2."""
    if spin not in PP_SPINS:
        raise ValueError(f'unknown pp spin channel {spin!r}; one of {PP_SPINS}')
    return n * (n + 1) // 2 if spin == 'singlet' else n * (n - 1) // 2


def pp_pair_norm(p, q, spin):
    """1/sqrt(1 + delta_pq) of a singlet pair (its spatially symmetric
    normalization); 1 for a triplet or spin-orbital pair."""
    if spin == 'singlet':
        return 1.0 / np.sqrt(1.0 + (np.asarray(p) == np.asarray(q)))
    return np.ones(np.shape(p))
