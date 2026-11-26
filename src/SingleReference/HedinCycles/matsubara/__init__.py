"""The finite-basis backend of the Hedin cycles on the Matsubara axis, in spin orbitals.

The map of the package:

    systems.py    a system: its one-electron Hamiltonian in spin orbitals, its Coulomb factors B^P
                  (v = sum_P B^P B^P) and electron count; the Hubbard dimer, a pyscf mean field
    starts.py     the start line: the aufbau Hartree or Hartree-Fock SCF with DIIS, its levels and gap
    grids.py      the Matsubara grids: the one-particle and boson grids and the vertex box
    backend.py    MatsubaraBackend, the operations of algebra.py: lines and their model sums, the
                  three-point objects, the tails of the frequency sums, the BSEs, Kernel, the w_order
    tangents.py   the tangent engine: the derivative along the map and the kernels at fixed form

See docs/methods/hedin-cycles.md, section "The Matsubara backend".
"""
