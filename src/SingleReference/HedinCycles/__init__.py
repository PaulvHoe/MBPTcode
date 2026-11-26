"""Hedin's equations iterated for a few cycles, with four vertex schemes.

Every cycle n builds Sigma^(n) from the Green's function G^(n-1) of the previous cycle, and every
Dyson equation uses the same start line G^(0): the self-consistent Hartree line h (start 'H') or
the self-consistent Hartree-Fock line f (start 'HF'). The four schemes differ in two choices: the
derivative in the vertex is taken along the map that the cycles compute (through G^(0)) or at fixed
form (of the previous cycle's formula, at its own line), and the self-energy is closed on the
Gamma/W route (Hedin's) or on the I/L route (with the two-particle correlation function L):

    'sg_il'   SG-I/L               along the map, I/L route
    'sg_gw'   SG-Gamma/W           along the map, Gamma/W route
    'cb_gw'   cycle-based Gamma/W  at fixed form, Gamma/W route
    'cb_il'   cycle-based I/L      at fixed form, I/L route

The map of the package:

    algebra.py   the operations a backend provides, each with its equation
    schemes.py   the four schemes, the two starts and the cycle loop, written only against them
    zero_d.py    the zero-dimensional backend: exact diagram counts and 0-D values
    matsubara/   the finite-basis backend on the Matsubara axis, in spin orbitals (see its own map)

See docs/methods/hedin-cycles.md.
"""
from src.SingleReference.HedinCycles.schemes import (HARTREE, SCHEME_NAMES, SCHEMES, STARTS,
                                                     CycleHistory, run_cycles)

__all__ = ['SCHEMES', 'SCHEME_NAMES', 'STARTS', 'HARTREE', 'CycleHistory', 'run_cycles']
