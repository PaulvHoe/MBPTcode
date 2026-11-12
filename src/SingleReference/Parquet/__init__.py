"""The parquet approximation for molecules (Marie and Loos, J. Chem. Phys. 163,
194115 (2025)), ported from the paper-era Quack (git e3528a40): FLEX, osPA,
evPA and qsPA in the static kernel approximation, closed shell and in spin
orbitals. docs/methods/parquet.md says what is computed and how to run it;
this docstring maps the port onto Quack: which routine each function ports,
what is not ported, what of Quack's behaviour is kept on purpose, where the
port departs from it, and how it was checked against it.

The two front doors:

    restricted.run_parquet      a closed-shell RHF: Quack's R_evParquet (FLEX,
                                osPA, evPA; `ev_parquet` on given MO
                                integrals) and R_qsParquet (qsPA;
                                `qs_parquet` on AO integrals); the paper's
                                closed-shell numbers are this driver's
    spin_orbital.run_parquet_so an RHF, UHF or GHF in spin orbitals: Quack's
                                G_evParquet (`ev_parquet_so`) and G_qsParquet
                                (`qs_parquet_so`); open shells

The modules:

    restricted.py               the closed-shell drivers: the two nested
                                loops, the two-body DIIS and damping, the
                                qsPA Fock step, and the cost and memory of a
                                run (its COST AND MEMORY)
    restricted_bse.py           the closed-shell eh and pp BSEs: bare blocks,
                                static kernels and solvers
    restricted_vertices.py      the closed-shell effective integrals and
                                reducible vertices
    restricted_self_energy.py   the closed-shell diagonal self-energy at any
                                frequency, and its quasiparticle equation by
                                the house's root finder
                                (qp_solver='find_qp_root')
    restricted_static_self_energy.py
                                the static self-energies of the closed-shell
                                qsPA
    spin_orbital.py             the spin-orbital drivers, their quasiparticle
                                solvers and the spin-orbital layout of a mean
                                field
    spin_orbital_two_body.py    one spin-orbital two-body iteration: BSEs,
                                kernels, effective integrals and vertices
    spin_orbital_self_energy.py the spin-orbital self-energies, diagonal at any
                                frequency and static
    options.py                  the options both drivers take and their
                                checks (refused before any work), QP_SOLVERS
                                and the contract of a qp_solver function,
                                and the roots find_qp_root follows
    spectra.py                  what the two-body loop hands the one-body
                                step: each BSE's eigenvalues and effective
                                integrals, and the reducible vertices
    pairs.py                    Quack's orderings of the eh and pp pairs, the
                                closed-shell spin weights and the triplet
                                multiplicity
    diis.py                     Quack's DIIS, accumulated in blocks
    events.py                   the three warning classes and the name of
                                every flag (docs/warnings.md)

Every numerical setting is a PARQUET_* constant of `Base.numerics.constants`.

QUACK'S ROUTINES (the paper tree) AND THE PORT:

    R_evParquet                 restricted.ev_parquet;
                                run_parquet(mf, 'flex' / 'os' / 'ev')
    R_qsParquet, with omega = 0d0 in its four R_*_Phi calls
                                restricted.qs_parquet; run_parquet(mf, 'qs')
    phRLR_A, phRLR_B            restricted_bse.eh_bse_blocks
    R_eh_singlet_Gamma_A/B, R_eh_triplet_Gamma_A/B
                                restricted_bse.eh_kernel_blocks
    phRLR                       restricted_bse.solve_eh_bse
    ppRLR_B, ppRLR_C, ppRLR_D   restricted_bse.pp_bse_blocks
    R_pp_singlet_Gamma_B/C/D, R_pp_triplet_Gamma_B/C/D
                                restricted_bse.pp_kernel_blocks
    ppRLR, with sort_ppRPA and orthogonalize_matrix
                                restricted_bse.solve_pp_bse
    R_eh_singlet_screened_integral, R_eh_triplet_screened_integral
                                restricted_vertices.eh_effective_integrals
    R_pp_singlet_screened_integral, R_pp_triplet_screened_integral
                                restricted_vertices.pp_effective_integrals
    R_eh_singlet_Phi, R_eh_triplet_Phi
                                restricted_vertices.eh_reducible_vertex (at
                                omega = 0: restricted.static_reducible_vertex)
    R_pp_singlet_Phi, R_pp_triplet_Phi
                                restricted_vertices.pp_reducible_vertex (at
                                omega = 0: the same)
    R_Parquet_self_energy_diag  restricted_self_energy.diagonal_self_energy,
                                DiagonalSelfEnergy.diagonal
    R_Parquet_self_energy_diag_omega
                                restricted_self_energy.self_energy_at,
                                DiagonalSelfEnergy.at
    R_Parquet_QP_graph          not ported (below)
    R_Parquet_self_energy_SRG_sp
                                restricted_static_self_energy
                                .static_self_energy_srg
    R_Parquet_self_energy_eta   restricted_static_self_energy
                                .static_self_energy_eta
    dump_RParquet_self_energy   restricted_self_energy.self_energy_parts
    Hartree_matrix_AO_basis, exchange_matrix_AO_basis
                                restricted.hartree_exchange; per spin block
                                spin_orbital.ghf_fock
    AOtoMO_ERI_RHF              restricted.ao_to_mo_eri
    orthogonalization_matrix    restricted.orthogonalization_matrix,
                                spin_orbital.orthogonalization_matrix
    MOtoAO, AOtoMO (the static self-energy to the AO basis and back);
    diagonalize_matrix (dsyev) on the Fock matrix
                                inside qs_parquet and qs_parquet_so
                                (PARQUET_QS_EIGH_DRIVER)
    DIIS_extrapolation, with linear_solve's dsysvx
                                diis.QuackDIIS
    G_evParquet                 spin_orbital.ev_parquet_so;
                                run_parquet_so(mf, 'flex' / 'os' / 'ev')
    G_qsParquet                 spin_orbital.qs_parquet_so;
                                run_parquet_so(mf, 'qs')
    phGLR_A, phGLR_B; G_eh_Gamma_A/B; phGLR
                                spin_orbital_two_body.eh_bse_blocks_so;
                                eh_kernel_blocks_so; solve_eh_bse_so
    ppGLR_B/C/D; G_pp_Gamma_B/C/D; ppGLR
                                spin_orbital_two_body.pp_bse_blocks_so;
                                pp_kernel_blocks_so; solve_pp_bse_so
    G_eh_screened_integral, G_pp_screened_integral
                                spin_orbital_two_body
                                .eh_effective_integrals_so,
                                pp_effective_integrals_so
    G_eh_Phi, G_pp_Phi          spin_orbital_two_body.eh_reducible_vertex_so,
                                pp_reducible_vertex_so (at omega = 0:
                                spin_orbital.static_reducible_vertex_so)
    G_Parquet_self_energy_diag  spin_orbital_self_energy
                                .diagonal_self_energy_so,
                                DiagonalSelfEnergySO
    (no G routine; R_Parquet_self_energy_diag_omega's convention)
                                spin_orbital_self_energy.self_energy_at_so,
                                DiagonalSelfEnergySO.at
    G_Parquet_self_energy_SRG_dp
                                spin_orbital_self_energy
                                .static_self_energy_srg_so
    dump_GParquet_self_energy   DiagonalSelfEnergySO.at_each with `parts`
    AOtoMO_ERI_GHF              spin_orbital.spin_orbital_eri

NOT PORTED: the frozen core and virtuals (nC, nR); R_Parquet_QP_graph,
Quack's own search on the quasiparticle equation (below); R_evParquet's
plotting blocks (R_Parquet_plot_self_energy, which calls
R_Parquet_self_energy_iieta); the print routines (R_print_parquet_1b, ...:
the result holds what they print); Quack's 'use_chem_pot_in_ppRPA' switch;
the unused _dp variants of the closed-shell self-energy (R_qsParquet's call
of R_Parquet_self_energy_SRG_dp is commented out); G_Parquet_Galitskii_Migdal
(commented out in Quack, whose EcGM is always 0); RParquet_v2 and the
R_Parquet_self_energy only it calls; and G_evParquet_v2, which the
paper-era GQuAcK calls for evParquet and which mixes and extrapolates the
BSE matrices instead of the vertices: this port is G_evParquet.

QUACK'S BEHAVIOUR, KEPT ON PURPOSE:

- The order of operations. The BSEs in the order eh singlet, eh triplet, pp
  singlet, pp triplet; the effective integrals from the irreducible vertices
  of the OLD reducible vertices, the ones the kernels were built from; the
  new vertices from those integrals and the NEW eigenvalues; the change
  before the damping. FLEX builds and damps vertices that nothing reads, as
  Quack does, so its timings are the algorithm's.
- Damping before DIIS, error damped - old, the damping evaluated as Quack
  evaluates it (docs/methods/parquet.md, Options).
- 'Failed' is n_it == max_it: a loop that meets its threshold at its last
  allowed iteration still counts as not converged, as in Quack. FLEX and
  osPA, whose single one-body step Quack reports as a failed one-body loop,
  get no one-body verdict here (converged_1b None).
- gfortran's max and maxval with NaN. The change of a vertex is gfortran's
  maxval, which skips NaN elements unless every element is NaN, and the
  change of the loop its max over the vertices, which skips a NaN argument
  (arm64 fmaxnm): a NaN in one vertex does not end the loop. An all-NaN
  change ends it, and Quack reports success; here converged_2b is False and
  the flag 'parquet_two_body_not_finite' says why
  ('parquet_one_body_not_finite' for the one-body change).
- The linearized energy is anchored on eHF at every evPA iteration,
  eQPlin = eHF + Z SigC(eOld).
- Two-body DIIS is decided by rounding. Its error vectors become nearly
  linearly dependent (condition numbers of 1e14 to 1e18), and a system below
  rcond = eps is solved without a warning, as in Quack: rounding sets the
  weights, and hardly the extrapolated vertices, since the errors they
  combine are nearly one vector. Converged runs with two-body DIIS agree
  with Quack's to about 1e-12 Ha in eQP and SigC (at 1 to 4 threads at
  most 4.5e-13 in eQP and 1.3e-12 in SigC), with the same iteration
  counts, where each two-body loop starts its history afresh and no
  one-body DIIS acts (osPA, and qsPA, whose history R_qsParquet empties
  at every one-body iteration; water, ammonia and OH 6-31G at s_2b = 1
  and conv_2b 1e-7 to 1e-10); to about 1e-11 Ha with one-body DIIS as
  well, whose systems also lie below rcond = eps (the spin-orbital qsPA of
  water, ammonia and OH: 1.2e-11 at most); and to about 1e-9 Ha where the
  history carries over between evPA one-body iterations, whose later
  two-body counts then differ by up to about ten. The counts move with the
  number of BLAS threads too (docs/methods/parquet.md, Options).
- The BSE arithmetic: an A - B that is not positive definite makes the whole
  eh solution NaN; a negative Omega^2 becomes sqrt|Omega^2| in the
  closed-shell solver (phRLR) and NaN in the spin-orbital one (phGLR); the
  pp branches are split at zero energy, extra roots dropped in dgeev's order
  and missing ones left at zero; the singlet pp block's literal
  0.7071067811865475. Each event Quack prints is flagged here, with Quack's
  numbers.
- qsPA: the printed Sigma_c is the diagonal in the NEW orbitals, Z is 1, the
  commutator error is taken before DIIS, DIIS extrapolates F and not the
  density, and the Fock matrix is diagonalized by dsyev.
- No Galitskii-Migdal energy (Quack's EcGM is 0).

WHERE THE PORT DEPARTS FROM QUACK:

- Quack's stop is a flag. R_qsParquet ends the program when its one-body
  loop fails; here the run warns, raises 'parquet_one_body_not_converged'
  and returns its last iterate. G_evParquet stops unless the quasiparticle
  equation is linearized ('not implemented'); here the spin-orbital driver
  solves it as the closed-shell one does (G_evParquet's own branch is
  qp_solver='linearized').
- Quack's quasiparticle search is not ported. R_evParquet solves the
  equation by R_Parquet_QP_graph, a Newton from eQPlin; the drivers solve it
  by the house's root finder (qp_solver='find_qp_root', the default) or take
  it linearized. The tests carry Quack's search for their comparisons with
  Quack and hand it to the drivers as a qp_solver function
  (tests/parquet_quack_newton.py, whose docstring has its arithmetic and how
  it compares with the finder).
- Events Quack lets pass are flagged, the numbers kept as Quack's: a
  negative Tamm-Dancoff eh excitation (both drivers; over spin orbitals only
  below -1e-3 Ha, since a one-electron system's spin flip is an exact zero
  that roundoff puts on either side), a full eh excitation below 1e-3 Ha (an
  open shell's spin rotation, or a closed shell at the edge of an
  instability), and NaN or inf entering a diagonalization, the self-energy
  or a vertex.
- R_qsParquet's missing omega. The paper tree calls R_eh_singlet_Phi and
  R_eh_triplet_Phi with eight arguments for nine and R_pp_singlet_Phi and
  R_pp_triplet_Phi with eleven for twelve: omega is missing, and with
  implicit interfaces the vertex array is read as omega and the new vertex
  written through an undefined address (undefined behaviour; the unpatched
  routine prints other numbers). The port builds the vertices at omega = 0,
  as R_evParquet does, and was verified against R_qsParquet_fixed, the paper
  routine with only 0d0 added to those four calls.
- DIIS where Quack's weights are undefined. Quack's linear_solve calls
  dsysvx and ignores its info. The port calls dsysvx the same way and keeps
  its rcond; a NaN or inf in the error history, where dsysvx stops at the
  NaN pivot and Quack sums uninitialized weights, gives a NaN iterate
  ('parquet_diis_not_finite'), and an exactly singular system (a zero
  pivot), where Quack's weights are undefined, takes the least-squares
  weights (lstsq) instead ('parquet_diis_singular').
- Alpha/beta pairing. For an RHF, run_parquet_so solves each orbital's alpha
  and beta copy once and copies the root (`spin_orbital`); G_evParquet
  solves both, and at large s_1b the two copies, equal only to roundoff, can
  part (water/6-31G evPA at s_1b = 1e8, linearized, without the pairing: an
  orbital split by 4.7 meV after 18 iterations).
- The core Hamiltonian of qsPA is the mean field's own, mf.get_hcore(),
  where Quack takes T + V. They are the same for an all-electron
  nonrelativistic molecule; with an ECP, X2C or an external field qsPA
  starts here from the reference's own Fock matrix, in both drivers. A
  GHF's is taken as its alpha block where it is spin-free (to 1e-10 Ha), and
  refused otherwise.
- nO = 1 or nV = 1. With one doubly occupied or one virtual orbital (H2, He)
  the triplet pp space is empty, and R_pp_triplet_screened_integral hands
  dgemm a zero leading dimension, which ends the run with Accelerate or the
  reference BLAS. The closed-shell port accepts it; on H2/6-31G, He/6-31G and
  water/6-31G cut to six orbitals it reproduces G_evParquet, which runs
  these cases, to 4e-13 Ha in the spectra and, at s_1b = 1e8, 4e-14 Ha in
  eQP. The spin-orbital port runs one occupied or one virtual spin orbital
  (the H atom, H2- in STO-3G) as G_evParquet does.
- The symmetric BSE diagonalizations use LAPACK's divide and conquer
  (PARQUET_EIGH_DRIVER) where Quack calls dsyev: the same eigenvalues and
  eigenspaces to rounding, six times faster at n = 3000.
- A canonical orthogonalization that drops basis functions is refused for
  qsPA: R_qsParquet's arrays are shape-consistent only when it keeps them
  all, and Quack runs on. (PySCF's one-electron HF1e is refused too, a
  pitfall of PySCF rather than of Quack: it drops the two-electron terms;
  build the UHF with pyscf.scf.uhf.UHF(mol).)

CHECKED AGAINST QUACK. The tests hold both drivers to Quack's own routines,
run from Python on small molecules, their numbers recorded in the test
files. Quack's printed tables for water and ammonia in 6-31G (FLEX with and
without the TDA and linearized, at s_1b = 1e8; osPA at s_2b = 1 and
s_1b = 1e8, and at s_2b = 10 and s_1b = 50 with two-body DIIS 5; evPA at
s = 50 with one-body and two-body DIIS 5; two-body loops converged to 1e-9,
evPA's one-body loop to 1e-7) are reproduced by the closed-shell driver,
given Quack's search, to their printed digits, 5e-7 eV in eQP and 5e-7 in Z,
with the same two-body iteration counts where there is no DIIS, but for two
high virtuals of ammonia at s_1b = 1e8, where that search is chaotic
(tests/parquet_quack_newton.py).
"""
