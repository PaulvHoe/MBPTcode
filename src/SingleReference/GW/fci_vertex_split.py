"""The exact FCI vertex split of G0W quasiparticle energies on a frozen-core active space.

For an occupied orbital p of a converged closed-shell RHF, on the active space of every MO above
the ncore lowest (h1 and the full four-index (pq|rs) from pyscf's CASCI, whose Fock matrix is
diag(e) with e = mo_energy[ncore:]), the ionization energy of p is computed exactly and on four
rungs of G0W@HF that share G0 and differ only in the correlation part W_c of the screened
interaction:

    rung    W_c from the density response of
    rpa     direct RPA (Hartree kernel, singlet): G0W0@HF
    tdhf    TDHF (Hartree + exchange kernel, singlet)
    bse     static BSE@HF (the TDHF kernel with its exchange-type terms screened by W_RPA(w = 0))
    fci     the active space's FCI ground state: the exact W_c on this Hamiltonian

Each rung gives Sigma^c_pp in pole form,

    Sigma^c_pp(w) = sum_q<m sum_k (W^k_pq)^2 / (w - e_q + Omega_k)
                  + sum_q>=m sum_k (W^k_pq)^2 / (w - e_q - Omega_k),
    W^k_pq        = sum_rs (pq|rs) <k|E_rs|0>,  E_rs = sum_sigma a+_r,sigma a_s,sigma,

m the doubly occupied active orbitals. For the Casida rungs W^k_pq = sqrt2 sum_ia (pq|ia)
(X + Y)_ia,k in the house normalization X^T X - Y^T Y = 1 (`LinearResponse.casida.CasidaSolver`
on `LinearResponse.linear_response.LinearResponseSolver.build_casida_matrices`; the static BSE
kernel is that builder's with W_aux from `static_screening_aux`, the house's BSE@HF). For the FCI
rung the k-sum of each q is the spectral measure of O_pq|Psi0> = sum_rs (pq|rs) E_rs |Psi0>,
projected off |Psi0>: Lanczos of depth D on the projected Hamiltonian Q H Q gives Ritz values
theta_k, Omega_k = theta_k - E0, and weights |O_pq Psi0|^2 U_0k^2, a Gauss quadrature of the
resolvent with D nodes, exact once D reaches the Krylov dimension (equal to the dense Lehmann sum
over every FCI state, which tests/test_fci_vertex_split.py holds). It runs without
reorthogonalization: a ghost copy of a converged pole shares that pole's weight and leaves the
quadrature as it is. Each Lanczos vector is projected off |Psi0> as it is made, as the pilot
does. Q H Q holds |Psi0> at eigenvalue 0. Where the active-space spectrum lies below 0 (the core
energy left out: N2/STO-3G, and Ne, HF and H2O in 6-31G), that eigenvalue lies above it, and the
three-term recurrence alone lets rounding seed that direction and amplify it at every step until
a Ritz value converges at 0, which spends Gauss nodes at steps rounding decides (N2/STO-3G's
2sigma_g: 20 such Ritz values in its 8 runs at D = 40, and a D/2 root 2.2 meV from the D/2-node
quadrature). In cc-pVDZ the spectrum reaches above 0 (He, H2, LiH, Ne), so 0 lies inside it.
Projecting every Lanczos vector off |Psi0> removes the direction in either case. States the
start vectors do not couple to (triplets, other symmetries) are still seeded by rounding and
converge with relative weights of 6e-21 or less (Ne/6-31G's HOMO the largest measured: 7e-24 at
D = 40, 5.5e-21 at D = 60 and 80), so the rung's lowest excitation is the lowest Ritz value whose
weight exceeds FCI_SPLIT_LOWEST_POLE_WEIGHT of the total, which differs from a Casida rung's
lowest excitation (its spectrum's, coupled to p or not). The FCI rung is checked against its own
leading half (depth D/2, read from the same tridiagonal matrix): that change bounds the error of
the D/2 value, so the check is conservative for D. A start vector whose norm is below
ACTIVE_SPACE_VERTEX_TOL carries no pole; with none left, Sigma^c_pp of the rung is zero (its root
e_p, Z = 1).

The quasiparticle equation w = e_p + Sigma^c_pp(w), where e_p already holds the exchange
(Sigma_x - v_xc = 0 on HF), is solved for every rung by the one root finder
(`Solvers.qp_equation.solve_qp_equation_poles`, the root of largest weight), and IP = -w. A root
whose weight is QP_WINDOW_Z_MIN or less is no quasiparticle
(`Solvers.qp_equation.quasiparticle_weight`): it is kept, its splits included, and flagged.
Beside it the linearized solution w = e_p + Z(e_p) Sigma^c_pp(e_p),
Z(e_p) = 1 / (1 - Sigma^c_pp'(e_p)), one Newton step of the same equation from e_p
(`Solvers.qp_equation.secular`).

The exact ionization energy is the main pole of the one-hole spectral function of p: Lanczos with
full reorthogonalization (without it the main pole splits into ghost copies, which would halve
its weight) on a_p,alpha|Psi0> in the (m - 1, m) sector gives the poles IP_s = E_s(N - 1) - E0 and
their weights |<s|a_p,alpha|0>|^2, which sum to the alpha occupation D_pp / 2. The main pole is the
one of largest weight, its weight the exact Z. It is read off Ritz values, not solved from a
quasiparticle equation. Each Ritz value theta_k lies within its residual bound beta_D |U_D-1,k| of
a cation energy, so a pole whose bound is below FCI_SPLIT_HOLE_RESIDUAL_BAR is certified as a
cation state to within that bound; the main pole's bound is checked against it. Above the bar a
pole is not certified: it may be a quadrature node standing for unresolved satellites (HF/6-31G's
HOMO, second pole bound 0.20 Ha) or a converged state whose bound overstates its error, which is
about bound^2 / gap (N2/STO-3G's 2sigma_g, second pole bound 1.9e-5 Ha, 3e-10 Ha from the dense
cation state). The bounds depend on rounding and on Psi0's own error: a factor of 2 between two
equally valid runs at the 1e-6 Ha level.

The split (eV): the W side IP[G0W_FCI] - IP[G0W0], the Sigma side IP_exact - IP[G0W_FCI] (the two
add up to minus the G0W0@HF error), and IP[G0W_TDHF] - IP[G0W_FCI], IP[G0W_BSE] - IP[G0W_FCI].

The density side, on the HF reference: static shifts s of the G0W0 quasiparticle energy, w -> w + s,
whose IP is then IP[G0W0] - s:

    dH            <p|J[D_FCI - D_HF]|p>              the exact density's Hartree change (oracle)
    dHx           <p|(J - K/2)[D_FCI - D_HF]|p>      with its exchange (oracle)
    T_c           <p|W_RPA(0)[dD_lin]|p>             the screened tadpole
    dHx_lin_scr   <p|(J - K/2)[dD_lin]|p> + its TDHF (A + B) static response

dD_lin is the linearized-GW correlation density of Sigma_c alone on the RPA rung's poles,
spin-summed (the house's unrelaxed GW density, whose ov block is
[Sigma^virt_ia(e_i) + Sigma^occ_ia(e_a)] / (e_i - e_a)), and W_RPA(0) = v - 4 v (A + B)^-1 v
(`LinearResponse.linear_response.static_screened_coulomb_chemist`). T_c is the correlation part of
the screened tadpole of a large-N_f counting (v -> (2 / N_f) v, N_f = 2 the molecule): through
O(1/N_f) the quasiparticle energy on the Hartree reference is Sigma_x + Sigma_c(e) + T, with
T = <p|v dn|p> the static Hartree response to the linearized-GW density dn_bare of
G (Sigma_x + Sigma_c) G screened by the static RPA, dn = (1 - P0(0) v)^-1 dn_bare. On HF, G0W0
holds every term of that order but T_c, the part sourced by Sigma_c, so G0W0@HF + T_c is exact
through O(1/N_f). The counting is the study's own, checked against exact N_f-flavour
diagonalizations with one occupied orbital per flavour.

A rung whose Casida problem is unstable (omega^2 <= 0 or an indefinite A - B or A + B) is not
repaired: it is NaN and flagged. Energies are in Hartree inside, in eV in the record. Both
half-depth changes in the record are IP(D) - IP(D/2).

The reference state. |Psi0> is the state pyscf's direct_spin1 reaches from its default guess,
the determinant of lowest diagonal energy, taken as it is (an open-shell one is not
spin-adapted). For every panel system that is the closed-shell HF determinant, by 0.05 Ha or more
(MgO/6-31G the least), and the Davidson keeps that guess's spatial and alpha-beta symmetry, so
|Psi0> is the lowest singlet of the HF determinant's symmetry, not necessarily the ground state of
the active space: MgO/STO-3G with ncore = 6 has its 3Pi 0.234 eV below it. With 400 determinants
or fewer and a non-degenerate lowest level pyscf diagonalizes exactly, and |Psi0> is the lowest
M_s = 0 state; with a degenerate one it runs the Davidson from the guess. A triplet the solver
does reach is flagged: O2 as an RHF singlet in STO-3G with ncore = 4 (diagonalized exactly, its
lowest M_s = 0 state the triplet) and C2/STO-3G with ncore = 4 (an open-shell guess and a
degenerate lowest triplet pair). The ground-state fields of the record are |Psi0>'s, and
'fci_split_ground_not_singlet' reads |Psi0>'s own <S^2>; a lower state of another spin or
symmetry is not looked for. The Davidson stops once its residual is below pyscf's default, the
square root of FCI_CONV_TOL (3e-7 Ha; 5e-8 to 2e-7 Ha measured), so |Psi0> carries a vector
error of about 1e-7 relative (2.6e-8 for H2O/STO-3G, 2e-7 for MgO/STO-3G against the dense
state), and every field built on it about as much: up to 5e-7 eV in the oracle shifts and the
G0W_FCI root, 4e-8 in Z and the occupations. With 400 determinants or fewer and a non-degenerate
lowest level it is exact to rounding.

The engines. The FCI ground state is pyscf's direct_spin1 Davidson, the engine
`Solvers.active_space_solver.PyscfFCISolver` wraps, called directly for its converged flag,
max_cycle and max_memory, which the wrapper does not expose (it also takes direct_spin0 for a
singlet, whose contraction assumes an exactly symmetric CI vector that the Lanczos iterates are
only to rounding). The Lanczos runs are the house's (`SingleReference.ADC.solve.lanczos_extend`),
one step at a time where the FCI rung projects each new Lanczos vector off |Psi0>. The FCI
rung's depth is the driver's `depth`, a numerical option on purpose (as BSE(2)'s
dynamical_roots): the cluster panel runs it at more than one depth, the default and twice the
pilot's 16, to compare at the pilot's own depth through the D/2 value.
"""
import math
import time

import numpy as np
from pyscf import ao2mo, fci, mcscf, scf

from src.Base.environment_seam import NoEnvironment, environment_of
from src.Base.numerical_flags import collecting, flag, gathered
from src.Base.numerics.constants import (ACTIVE_SPACE_VERTEX_TOL, FCI_CONV_TOL,
                                         FCI_SPLIT_DAVIDSON_VECTORS, FCI_SPLIT_FOCK_BAR,
                                         FCI_SPLIT_HALF_DEPTH_BAR, FCI_SPLIT_HOLE_COPIES,
                                         FCI_SPLIT_HOLE_DEPTH, FCI_SPLIT_HOLE_RESIDUAL_BAR,
                                         FCI_SPLIT_LOWEST_POLE_WEIGHT, FCI_SPLIT_MAX_CYCLE,
                                         FCI_SPLIT_MAX_MEMORY_MB, FCI_SPLIT_SCREENING_DEPTH,
                                         FCI_SPLIT_SECOND_POLE_RATIO,
                                         FCI_SPLIT_SECOND_POLE_WINDOW,
                                         FCI_SPLIT_SPIN_SQUARE_BAR, FLOAT64_PER_GB,
                                         HARTREE_TO_EV, MB_PER_GB, QP_DEGENERACY_TOL,
                                         QP_WINDOW_Z_MIN)
from src.Base.pyscf.pyscf_interface import occupied_count
from src.Base.utils.parallel.memory import allocation_max_memory_mb
from src.SingleReference.ADC.solve import LanczosState, lanczos_extend
from src.SingleReference.GW.qp_selection import degenerate_blocks
from src.SingleReference.LinearResponse.casida import CasidaInstability, CasidaSolver
from src.SingleReference.LinearResponse.linear_response import (LinearResponseSolver,
                                                                static_screened_coulomb_chemist)
from src.Solvers.qp_equation import quasiparticle_weight, secular, solve_qp_equation_poles

#: The four screenings of G0W@HF, in the order the record lists them.
RUNGS = ('rpa', 'tdhf', 'bse', 'fci')


class FCISplitShort(RuntimeWarning):
    """A step of the FCI vertex split stopped short of what it was asked: the FCI ground state did
    not converge or is not a singlet, the active Fock matrix is not diagonal, a Lanczos run did
    not settle, the exact main pole is ambiguous, a G0W root is no quasiparticle, or a Casida rung
    is unstable (that rung is NaN). Everything else is kept."""


def _plain(value):
    """value as a plain Python int or float where it is a number, for a message or the record."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def _occupations(occ):
    """mf.mo_occ through its last occupied orbital and the one after, for a refusal message."""
    occupied = np.flatnonzero(occ > 0)
    end = int(occupied[-1]) + 2 if occupied.size else 1
    shown = ', '.join(f'{float(x):g}' for x in occ[:end])
    return f'[{shown}{", ..." if end < occ.size else ""}]'


def check_reference(mf):
    """The doubly occupied orbital count of mf; refuses (ValueError) a mean field the split is not
    defined for: anything but a converged closed-shell RHF on exact integrals, in the gas phase
    (no solvent, no environment but `Base.environment_seam.NoEnvironment`), with an aufbau
    occupation and at least one virtual orbital."""
    if not isinstance(mf, scf.hf.RHF) or isinstance(mf, scf.rohf.ROHF):
        raise ValueError(f'mf = {type(mf).__name__}: the FCI vertex split needs a closed-shell RHF')
    if getattr(mf, 'xc', None) is not None:
        raise ValueError(f'mf.xc = {mf.xc!r}: the FCI vertex split needs a Hartree-Fock reference '
                         '(its G0W rungs read Sigma_x - v_xc = 0)')
    if getattr(mf, 'with_df', None) is not None:
        raise ValueError(f'mf.with_df = {type(mf.with_df).__name__}: the FCI vertex split needs '
                         'exact integrals (pyscf would build the active space on the fitted ones)')
    solvent = getattr(mf, 'with_solvent', None)
    if solvent is not None:
        raise ValueError(f'mf.with_solvent = {type(solvent).__name__}: the FCI vertex split is for '
                         'an isolated molecule')
    environment = environment_of(mf)
    if not isinstance(environment, NoEnvironment):
        raise ValueError(f'mf.with_screening = {type(environment).__name__}: the FCI vertex split '
                         'is for an isolated molecule')
    if not getattr(mf, 'converged', False):
        raise ValueError('mf.converged = False: the FCI vertex split needs a converged RHF')
    if mf.mol.spin != 0:
        raise ValueError(f'mf.mol.spin = {_plain(mf.mol.spin)!r}: the FCI vertex split needs a '
                         'closed shell')
    occ = np.asarray(mf.mo_occ)
    nocc = occupied_count(mf)
    if not (np.all(occ[:nocc] == 2) and np.all(occ[nocc:] == 0)):
        raise ValueError(f'mf.mo_occ = {_occupations(occ)}: the FCI vertex split needs an aufbau '
                         'closed-shell occupation')
    if nocc == occ.size:
        raise ValueError(f'mf.mo_occ = {_occupations(occ)}: all {nocc} orbitals occupied; the FCI '
                         'vertex split needs a virtual orbital to screen with')
    return nocc


def check_arguments(nocc, ncore, orbitals, depth):
    """(orbitals, depth) checked: orbitals as a list of plain ints (None kept), depth the
    screening depth (None for FCI_SPLIT_SCREENING_DEPTH). Refuses (ValueError) an ncore outside
    0 <= ncore < nocc, orbitals that are not distinct occupied active orbitals (full-MO
    numbering, ncore <= p < nocc), and a depth that is not an even integer of at least 2."""
    if isinstance(ncore, (bool, np.bool_)) or not isinstance(ncore, (int, np.integer)) or not (
            0 <= ncore < nocc):
        raise ValueError(f'ncore = {_plain(ncore)!r}: an integer with 0 <= ncore < nocc = {nocc}')
    if orbitals is not None:
        given = orbitals
        if isinstance(given, (int, np.integer)) and not isinstance(given, (bool, np.bool_)):
            given = [given]
        try:
            given = [_plain(p) for p in given]
        except TypeError:
            given = None
        if (not given or any(isinstance(p, bool) or not isinstance(p, int) for p in given)
                or len(set(given)) != len(given)
                or any(not ncore <= p < nocc for p in given)):
            raise ValueError(f'orbitals = {orbitals!r}: distinct occupied active orbitals in '
                             f'full-MO numbering, {int(ncore)} <= p < {nocc}')
        orbitals = given
    if depth is None:
        depth = FCI_SPLIT_SCREENING_DEPTH
    elif (isinstance(depth, (bool, np.bool_)) or not isinstance(depth, (int, np.integer))
          or depth < 2 or depth % 2):
        raise ValueError(f'depth = {_plain(depth)!r}: an even integer >= 2 (None for '
                         f'FCI_SPLIT_SCREENING_DEPTH = {FCI_SPLIT_SCREENING_DEPTH})')
    return orbitals, int(depth)


def check_core(mo_energy, nocc, ncore):
    """Refuses (ValueError) an ncore that splits a degenerate occupied level, a block of
    `SingleReference.GW.qp_selection.degenerate_blocks` (neighbours closer than
    QP_DEGENERACY_TOL, Ha) with members on both sides of the cut: freezing part of a level and
    keeping the rest active breaks the molecule's symmetry and changes every number."""
    for block in degenerate_blocks(mo_energy, range(nocc), QP_DEGENERACY_TOL):
        if block[0] < ncore <= block[-1]:
            raise ValueError(f'ncore = {int(ncore)}: it splits the degenerate occupied level '
                             f'{[int(i) for i in block]} (MO energies within QP_DEGENERACY_TOL = '
                             f'{QP_DEGENERACY_TOL:g} Ha of their neighbours); freeze the whole '
                             f'level or none of it')


def check_memory(nmo, nocc, ncore):
    """(peak, allowed, dets_n, dets_n_minus_1): the run's estimated peak memory (MB), the MB this
    job may give it (`Base.utils.parallel.memory.allocation_max_memory_mb`, its SLURM share, else
    FCI_SPLIT_MAX_MEMORY_MB) and the determinants in N and N - 1. The peak is the larger of the
    ground state's Davidson (FCI_SPLIT_DAVIDSON_VECTORS N-sector vectors) and the hole's Lanczos
    basis (FCI_SPLIT_HOLE_COPIES copies of FCI_SPLIT_HOLE_DEPTH (N - 1)-sector vectors). Refuses
    (ValueError, before any work) an active space whose peak exceeds the allowance."""
    n, m = nmo - ncore, nocc - ncore
    na, nh = math.comb(n, m), math.comb(n, m - 1)
    dets_n, dets_h = na * na, nh * na
    peak = max(FCI_SPLIT_DAVIDSON_VECTORS * dets_n,
               FCI_SPLIT_HOLE_COPIES * FCI_SPLIT_HOLE_DEPTH * dets_h) / FLOAT64_PER_GB * MB_PER_GB
    allowed = float(allocation_max_memory_mb(default=FCI_SPLIT_MAX_MEMORY_MB))
    if peak > allowed:
        raise ValueError(f'ncore = {int(ncore)}: the active space of {n} orbitals and {2 * m} '
                         f'electrons ({dets_n} determinants in N, {dets_h} in N - 1) needs about '
                         f'{peak:.0f} MB at its peak, above the {allowed:.0f} MB this job may use '
                         f'(its SLURM share, else FCI_SPLIT_MAX_MEMORY_MB); freeze more core '
                         f'orbitals or give the job more memory')
    return peak, allowed, dets_n, dets_h


def active_hamiltonian(mf, ncore):
    """(h1, eri, ecore, e, m, miss): the frozen-core active space of every MO above the ncore
    lowest, from pyscf's CASCI (h1 its effective one-body matrix, eri the full four-index (pq|rs)
    in chemists' notation, ecore the nuclear and core energy, Ha), e = mo_energy[ncore:], m the
    doubly occupied active orbitals, and miss (Ha) the largest element of the active Fock matrix
    h1 + 2 J - K minus diag(e). A miss above FCI_SPLIT_FOCK_BAR is flagged
    ('fci_split_fock_not_diagonal') and the run goes on."""
    n = mf.mo_coeff.shape[1] - ncore
    nelec = mf.mol.nelectron - 2 * ncore
    mc = mcscf.CASCI(mf, n, nelec)
    mc.verbose = 0
    h1, ecore = mc.get_h1eff()
    eri = ao2mo.restore(1, mc.get_h2eff(), n)
    e = np.asarray(mf.mo_energy)[ncore:].copy()
    m = nelec // 2
    fock = (h1 + 2.0 * np.einsum('pqii->pq', eri[:, :, :m, :m])
            - np.einsum('piiq->pq', eri[:, :m, :m, :]))
    miss = float(np.abs(fock - np.diag(e)).max())
    if miss > FCI_SPLIT_FOCK_BAR:
        flag('fci_split_fock_not_diagonal',
             f'the active Fock matrix of the CASCI integrals differs from diag(mo_energy[ncore:]) '
             f'by {miss:.2e} Ha, above FCI_SPLIT_FOCK_BAR = {FCI_SPLIT_FOCK_BAR:g} Ha: the '
             f'orbitals are not canonical Hartree-Fock to that accuracy; the run goes on with '
             f'mo_energy', FCISplitShort)
    return np.asarray(h1), eri, float(ecore), e, m, miss


def level_members(e, m, p):
    """The degenerate level of p among the occupied active orbitals: the block of
    `SingleReference.GW.qp_selection.degenerate_blocks` (neighbours closer than QP_DEGENERACY_TOL,
    Ha, the house's one separation) that holds p."""
    return next(list(block) for block in degenerate_blocks(e, range(m), QP_DEGENERACY_TOL)
                if p in block)


def default_orbitals(e, m, ncore):
    """The HOMO and, if there is one, the next lower occupied level outside the HOMO's degenerate
    set (`level_members`), represented by its highest orbital: HOMO-1 by level, not by index.
    Full-MO numbering."""
    homo = m - 1
    below = [i for i in range(m) if i not in level_members(e, m, homo)]
    return [homo + ncore] + ([max(below) + ncore] if below else [])


def ground_state(h1, eri, n, m):
    """(E0, c0, D, converged, spin_square): the reference state |Psi0> of the split in the
    (m, m) sector by pyscf's direct_spin1 Davidson from its default guess, the determinant of
    lowest diagonal energy (FCI_CONV_TOL within FCI_SPLIT_MAX_CYCLE iterations, holding what this
    job may give it in memory: `Base.utils.parallel.memory.allocation_max_memory_mb`, its SLURM
    share, else FCI_SPLIT_MAX_MEMORY_MB), E0 without the core energy, D its spin-summed 1-RDM.
    Where that guess is the closed-shell HF determinant (every panel system), the Davidson keeps
    its spatial and alpha-beta symmetry, so |Psi0> is the lowest singlet of the HF determinant's
    symmetry, which need not be the active space's ground state (MgO/STO-3G, ncore = 6: its 3Pi
    lies 0.234 eV lower); pyscf diagonalizes 400 determinants or fewer exactly when the lowest
    level is not degenerate, and |Psi0> is then the lowest M_s = 0 state. That exact path and a
    Davidson from an open-shell guess can end on a triplet: O2 as an RHF singlet and C2, both
    STO-3G with ncore = 4. Not converged is flagged ('fci_split_ground_not_converged'), <S^2>
    above FCI_SPLIT_SPIN_SQUARE_BAR too ('fci_split_ground_not_singlet', as for those two); the
    state is kept."""
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = FCI_CONV_TOL
    solver.max_cycle = FCI_SPLIT_MAX_CYCLE
    solver.max_memory = allocation_max_memory_mb(default=FCI_SPLIT_MAX_MEMORY_MB)
    solver.verbose = 0
    e0, c0 = solver.kernel(h1, eri, n, (m, m))
    converged = bool(solver.converged)
    ndet = int(c0.size)
    if not converged:
        flag('fci_split_ground_not_converged',
             f'the FCI ground state ({ndet} determinants) did not converge to FCI_CONV_TOL = '
             f'{FCI_CONV_TOL:g} Ha in FCI_SPLIT_MAX_CYCLE = {FCI_SPLIT_MAX_CYCLE} cycles; its '
             f'last iterate is kept', FCISplitShort)
    ss = float(solver.spin_square(c0, n, (m, m))[0])
    if ss > FCI_SPLIT_SPIN_SQUARE_BAR:
        flag('fci_split_ground_not_singlet',
             f'<S^2> of the FCI ground state is {ss:.3e}, above FCI_SPLIT_SPIN_SQUARE_BAR = '
             f'{FCI_SPLIT_SPIN_SQUARE_BAR:g}: it is not the singlet the split assumes; the run '
             f'goes on', FCISplitShort)
    D = solver.make_rdm1(c0, n, (m, m))
    return float(e0), np.asarray(c0), np.asarray(D), converged, ss


def casida_rungs(e, eri, m):
    """{'rpa', 'tdhf', 'bse'}: (omega, X + Y) of each singlet Casida screening (Ha; house
    normalization), None for an unstable one (`LinearResponse.casida.CasidaInstability`, flagged
    'fci_split_rung_unstable'), and 'tdhf_apb', the TDHF A + B for the exchange-screened density
    shift."""
    lr = LinearResponseSolver(e, eri_chemist=eri, spin_mode='restricted', eta=0.0)
    builds = {'rpa': lambda: lr.build_casida_matrices(m, lBSE=False),
              'tdhf': lambda: lr.build_casida_matrices(m, lBSE=True),
              'bse': lambda: lr.build_casida_matrices(m, lBSE=True,
                                                      W_aux=lr.static_screening_aux(m))}
    out = {}
    for rung, build in builds.items():
        A, B = build()
        if rung == 'tdhf':
            out['tdhf_apb'] = A + B
        try:
            omega, X, Y = CasidaSolver(A, B).solve()
        except CasidaInstability as exc:
            flag('fci_split_rung_unstable',
                 f'the {rung.upper()} screening is unstable ({exc}): its G0W rung is NaN, not '
                 f'repaired', FCISplitShort)
            out[rung] = None
            continue
        out[rung] = (np.asarray(omega), np.asarray(X) + np.asarray(Y))
    return out


def casida_residues(eri, m, xpy, rows):
    """w[r, q, k] = sqrt2 sum_ia (p_r q|ia) (X + Y)_ia,k for the orbitals p_r in rows (active
    numbering): the singlet density-fluctuation couplings of a Casida screening, w^2 the residues
    of Sigma^c."""
    n = eri.shape[0]
    ov = eri[np.asarray(rows)][:, :, :m, m:].reshape(len(rows) * n, -1)
    return (np.sqrt(2.0) * (ov @ xpy)).reshape(len(rows), n, -1)


def pole_arrays(e, m, couplings, omega):
    """(num, pol) of Sigma^c_pp from couplings[q, k] (W^k_pq, Ha) and excitation energies
    omega[k] (Ha) shared by every q: pol = e_q - Omega_k for occupied q, e_q + Omega_k for
    virtual q."""
    sign = np.where(np.arange(len(e)) < m, -1.0, 1.0)
    return ((couplings ** 2).ravel(),
            (e[:, None] + sign[:, None] * np.asarray(omega)[None, :]).ravel())


def _qp_root(num, pol, anchor, label):
    """(w, Z, w_lin, Z_lin) of w = anchor + sum num / (w - pol): the root of largest weight by
    the one root finder (`Solvers.qp_equation.solve_qp_equation_poles`; its 'no_root' and
    'z_outside' flags reach the driver's record) and the linearized solution, one Newton step
    from the anchor (`Solvers.qp_equation.secular`)."""
    w, z = solve_qp_equation_poles(num, pol, anchor, label=label)
    f, slope = secular(num, pol, anchor, anchor)
    return w, z, anchor - f / slope, 1.0 / slope


def spectral_measure(matvec, start, depth, reorth, deflate=None):
    """((theta, weight, residual), (theta_half, weight_half, residual_half), steps, breakdown):
    the Lanczos spectral measure of `start` under the symmetric operator `matvec` at depth
    `depth` and at depth // 2, the second from the leading block of the same tridiagonal matrix,
    which is bitwise what a separate depth // 2 run gives (the house Lanczos,
    `SingleReference.ADC.solve.lanczos_extend`, with full reorthogonalization if reorth). With
    `deflate`, a unit eigenvector of the operator that `start` is orthogonal to, every new Lanczos
    vector is projected off it as it is made (one step of the house Lanczos at a time): the
    three-term recurrence alone lets rounding seed that direction, and where its eigenvalue lies
    outside the rest of the spectrum amplifies it at every step until a Ritz value converges there.
    Ritz values theta ascending, weights |start|^2 U_0k^2, residual the bound beta_k |U_k-1,j| of
    each Ritz pair (0 at an invariant subspace); steps the depth reached (less at an invariant
    subspace, breakdown True)."""
    state = LanczosState(start, keep_vectors=reorth)
    if deflate is None:
        lanczos_extend(matvec, state, depth, reorth=reorth)
    else:
        for _ in range(depth):
            if state.breakdown:
                break
            lanczos_extend(matvec, state, 1, reorth=reorth)
            state.v_curr = state.v_curr - deflate * np.dot(deflate, state.v_curr)
    a, b = state.arrays()
    norm2 = float(np.dot(start, start))

    def measure(k):
        T = np.diag(a[:k]) + np.diag(b[1:k], 1) + np.diag(b[1:k], -1)
        theta, U = np.linalg.eigh(T)
        coupling = abs(float(b[k])) if k < len(b) else 0.0
        return theta, norm2 * U[0] ** 2, coupling * np.abs(U[-1])
    return measure(len(a)), measure(min(len(a), depth // 2)), len(a), bool(state.breakdown)


def linearized_gw_density(e, m, omega, w):
    """The spin-summed linearized-GW correlation density dD (n, n) of Sigma_c on G0, from the
    couplings w[p, q, k] (sqrt2 included) and excitation energies omega[k] of a screening:

        dD_ij = -2 sum_k,a w_iak w_jak / ((e_a + O_k - e_i)(e_a + O_k - e_j))
        dD_ab =  2 sum_k,i w_aik w_bik / ((e_i - O_k - e_a)(e_i - O_k - e_b))
        dD_ia =  2 [Sigma^virt_ia(e_i) + Sigma^occ_ia(e_a)] / (e_i - e_a)

    Sigma^occ and Sigma^virt the occupied and virtual pole sums of Sigma^c_ia. Its trace is 0."""
    o, v = slice(0, m), slice(m, len(e))
    eo, ev, om = e[o], e[v], np.asarray(omega)
    t = w[o, v, :] / (ev[None, :, None] + om[None, None, :] - eo[:, None, None])
    oo = -np.einsum('iak,jak->ij', t, t)
    t = w[v, o, :] / (eo[None, :, None] - om[None, None, :] - ev[:, None, None])
    vv = np.einsum('aik,bik->ab', t, t)
    virt = np.einsum('ibk,abk->ia', w[o, v, :] / (eo[:, None, None] - ev[None, :, None]
                                                  - om[None, None, :]), w[v, v, :])
    occ = np.einsum('ijk,ajk->ia', w[o, o, :], w[v, o, :] / (ev[:, None, None] - eo[None, :, None]
                                                             + om[None, None, :]))
    ov = (virt + occ) / (eo[:, None] - ev[None, :])
    dD = np.zeros((len(e), len(e)))
    dD[o, o], dD[v, v], dD[o, v], dD[v, o] = oo, vv, ov, ov.T
    return 2.0 * dD


def _density_side(e, eri, m, rungs, dD_fci):
    """The density-side pieces shared by every orbital: dD_lin (None without an RPA rung), the
    static RPA W(0), J and K of dD_lin, the TDHF static response y to (J - K/2)[dD_lin] (None when
    the TDHF A + B is not positive definite, flagged 'fci_split_rung_unstable'), and the oracle
    density change D_FCI - D_HF."""
    n = len(e)
    side = {'dD_fci': dD_fci - np.diag(np.where(np.arange(n) < m, 2.0, 0.0)),
            'dD_lin': None, 'W0': None, 'J': None, 'K': None, 'y': None}
    if rungs['rpa'] is None:
        return side
    omega, xpy = rungs['rpa']
    w = casida_residues(eri, m, xpy, list(range(n)))
    dD = linearized_gw_density(e, m, omega, w)
    J = np.einsum('pqrs,rs->pq', eri, dD)
    K = np.einsum('prsq,rs->pq', eri, dD)
    side.update(dD_lin=dD, W0=static_screened_coulomb_chemist(e, eri, m), J=J, K=K)
    try:
        chol = np.linalg.cholesky(rungs['tdhf_apb'])
    except np.linalg.LinAlgError:
        flag('fci_split_rung_unstable',
             'the TDHF A + B is not positive definite: the exchange-screened density shift '
             'dhx_lin_scr is NaN, not repaired', FCISplitShort)
        return side
    rhs = (J - 0.5 * K)[:m, m:].ravel()
    side['y'] = -2.0 * np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    return side


def _density_shifts(eri, m, p, side, w_g0w0):
    """The density-side record of active orbital p (eV): the oracle shifts dh and dhx, T_c and
    its bare part, dhx_lin_scr, the trace of dD_lin, and each shift's G0W0@HF IP,
    IP[G0W0] - s."""
    to_ev = HARTREE_TO_EV
    dD = side['dD_fci']
    dh = float(np.einsum('rs,rs->', eri[p, p], dD))
    dhx = dh - 0.5 * float(np.einsum('rs,rs->', eri[p, :, :, p], dD))
    out = {'dh_oracle_ev': dh * to_ev, 'dhx_oracle_ev': dhx * to_ev}
    tc = tc_bare = dhx_scr = trace = np.nan
    if side['dD_lin'] is not None:
        dl = side['dD_lin']
        tc = float(np.einsum('rs,rs->', side['W0'][p, p], dl))
        tc_bare = float(np.einsum('rs,rs->', eri[p, p], dl))
        trace = float(np.trace(dl))
        if side['y'] is not None:
            n = eri.shape[0]
            y = side['y'].reshape(m, n - m)
            induced = float(np.sum((2.0 * eri[p, p][:m, m:] - eri[p, :m, m:, p]) * y))
            dhx_scr = float((side['J'] - 0.5 * side['K'])[p, p]) + induced
    out.update(tc_ev=tc * to_ev, tc_bare_ev=tc_bare * to_ev, dhx_lin_scr_ev=dhx_scr * to_ev,
               trace_dd_lin=trace)
    for key, shift in (('dh_oracle', dh), ('dhx_oracle', dhx), ('tc', tc),
                       ('dhx_lin_scr', dhx_scr)):
        out[f'ip_g0w0_{key}_ev'] = -(w_g0w0 + shift) * to_ev
    return out


def _hole(h1, eri, n, m, c0, e0, p, p_mo):
    """The exact one-hole record of active orbital p (Lanczos with full reorthogonalization on
    a_p,alpha|Psi0> in the (m - 1, m) sector, FCI_SPLIT_HOLE_DEPTH steps): the main pole (IP in eV,
    Z) and its Ritz residual bound (Ha), the second largest weight, its pole and bound, the weight
    of the other poles within FCI_SPLIT_SECOND_POLE_WINDOW of the main one, the weights' sum, and
    the main pole at half depth (a diagnostic). A main pole whose bound exceeds
    FCI_SPLIT_HOLE_RESIDUAL_BAR is flagged ('fci_split_depth_unconverged'), an ambiguous one too
    ('fci_split_main_pole_ambiguous'): weight QP_WINDOW_Z_MIN or less, or the poles within the
    window carrying FCI_SPLIT_SECOND_POLE_RATIO of its weight or more."""
    nelec = (m - 1, m)
    link = (fci.cistring.gen_linkstr_index_trilidx(range(n), m - 1),
            fci.cistring.gen_linkstr_index_trilidx(range(n), m))
    h2 = fci.direct_spin1.absorb_h1e(h1, eri, n, nelec, 0.5)
    shape = (fci.cistring.num_strings(n, m - 1), fci.cistring.num_strings(n, m))

    def matvec(x):
        return fci.direct_spin1.contract_2e(h2, x.reshape(shape), n, nelec,
                                            link_index=link).ravel()
    start = np.ascontiguousarray(fci.addons.des_a(c0, n, (m, m), p)).ravel()
    depth = FCI_SPLIT_HOLE_DEPTH
    (theta, weight, residual), (theta_h, weight_h, _), steps, breakdown = spectral_measure(
        matvec, start, depth, reorth=True)
    ips, ips_h = theta - e0, theta_h - e0
    order = np.argsort(weight)[::-1]
    main = order[0]
    second = order[1] if order.size > 1 else None
    ip, z, bound = float(ips[main]), float(weight[main]), float(residual[main])
    ip_h = float(ips_h[np.argmax(weight_h)])
    near = np.abs(ips - ip) <= FCI_SPLIT_SECOND_POLE_WINDOW
    near[main] = False
    window = float(weight[near].sum())
    rec = {'ip_ev': ip * HARTREE_TO_EV, 'z': z, 'residual_ha': bound,
           'second_ip_ev': None if second is None else float(ips[second]) * HARTREE_TO_EV,
           'second_z': None if second is None else float(weight[second]),
           'second_residual_ha': None if second is None else float(residual[second]),
           'window_weight': window, 'weight_sum': float(weight.sum()),
           'ip_half_depth_ev': ip_h * HARTREE_TO_EV,
           'half_depth_change_ev': (ip - ip_h) * HARTREE_TO_EV, 'depth': depth, 'steps': steps,
           'breakdown': breakdown}
    if bound > FCI_SPLIT_HOLE_RESIDUAL_BAR:
        flag('fci_split_depth_unconverged',
             f'orbital {p_mo}: the exact main pole at {ip * HARTREE_TO_EV:.4f} eV has the Ritz '
             f'residual bound {bound:.2e} Ha after {steps} Lanczos steps, above '
             f'FCI_SPLIT_HOLE_RESIDUAL_BAR = {FCI_SPLIT_HOLE_RESIDUAL_BAR:g} Ha: a cation state '
             f'lies that close, which one is not settled; the depth-{depth} pole is kept',
             FCISplitShort)
    crowded = window >= FCI_SPLIT_SECOND_POLE_RATIO * z
    if z <= QP_WINDOW_Z_MIN or crowded:
        what = (f'its weight {z:.4f} is QP_WINDOW_Z_MIN = {QP_WINDOW_Z_MIN:g} or less'
                if z <= QP_WINDOW_Z_MIN else
                f'the other poles within FCI_SPLIT_SECOND_POLE_WINDOW = '
                f'{FCI_SPLIT_SECOND_POLE_WINDOW:g} Ha of it carry {window:.4f}, '
                f'FCI_SPLIT_SECOND_POLE_RATIO = {FCI_SPLIT_SECOND_POLE_RATIO:g} of its weight or '
                f'more')
        flag('fci_split_main_pole_ambiguous',
             f'orbital {p_mo}: the exact main pole at {ip * HARTREE_TO_EV:.4f} eV (Z = {z:.4f}) '
             f'is ambiguous: {what}; the pole of largest weight is kept', FCISplitShort)
    return rec


class _Screening:
    """The FCI rung's operator: H in the (m, m) sector projected off |Psi0> (Q H Q, which holds
    |Psi0> at eigenvalue 0; the Lanczos runs keep their vectors off it, `spectral_measure`), and
    the start vectors O_pq|Psi0> = sum_rs (pq|rs) E_rs |Psi0>, projected too."""

    def __init__(self, h1, eri, n, m, c0):
        self.eri, self.n, self.nelec = eri, n, (m, m)
        self.link = (fci.cistring.gen_linkstr_index_trilidx(range(n), m),) * 2
        self.h2 = fci.direct_spin1.absorb_h1e(h1, eri, n, self.nelec, 0.5)
        self.c0 = np.ascontiguousarray(c0)
        self.c0f = self.c0.ravel()

    def matvec(self, x):
        y = fci.direct_spin1.contract_2e(self.h2, x.reshape(self.c0.shape), self.n, self.nelec,
                                         link_index=self.link).ravel()
        y -= self.c0f * np.dot(self.c0f, y)
        return y

    def start(self, p, q):
        v = fci.direct_spin1.contract_1e(np.ascontiguousarray(self.eri[p, q]), self.c0, self.n,
                                         self.nelec, link_index=self.link).ravel()
        v -= self.c0f * np.dot(self.c0f, v)
        return v


def _fci_rung(screening, e, m, e0, p, depth):
    """The FCI rung of active orbital p: (num, pol) at depth and at depth // 2 (Lanczos vectors
    kept off |Psi0>, `spectral_measure`), the lowest excitation energy with weight (Ha; above
    FCI_SPLIT_LOWEST_POLE_WEIGHT of the total weight sum_q |Q O_pq Psi0|^2 of Sigma_pp), the
    deepest Lanczos run and the start vectors skipped (norm below ACTIVE_SPACE_VERTEX_TOL: no
    pole). With every start vector skipped the pole arrays are empty (Sigma^c_pp = 0) and the
    lowest excitation is None."""
    full, half, lows, total, steps, skipped = ([], []), ([], []), [], 0.0, 0, 0
    for q in range(len(e)):
        start = screening.start(p, q)
        norm2 = float(np.dot(start, start))
        if np.sqrt(norm2) < ACTIVE_SPACE_VERTEX_TOL:
            skipped += 1
            continue
        measures = spectral_measure(screening.matvec, start, depth, reorth=False,
                                    deflate=screening.c0f)
        sign = -1.0 if q < m else 1.0
        for into, (theta, weight, _) in zip((full, half), measures[:2]):
            into[0].append(weight)
            into[1].append(e[q] + sign * (theta - e0))
        lows.append(measures[0][:2])
        total += norm2
        steps = max(steps, measures[2])
    if not full[0]:
        empty = (np.zeros(0), np.zeros(0))
        return empty, empty, None, steps, skipped
    floor = FCI_SPLIT_LOWEST_POLE_WEIGHT * total
    omega_min = min(float(theta[weight > floor].min()) for theta, weight in lows
                    if np.any(weight > floor)) - e0
    return ((np.concatenate(full[0]), np.concatenate(full[1])),
            (np.concatenate(half[0]), np.concatenate(half[1])), omega_min, steps, skipped)


def _rung_root(num, pol, e_p, label, rung, p_mo):
    """A stable rung's (w, Z, record) for active orbital energy e_p: the quasiparticle root and
    its linearized solution (`_qp_root`); a root of weight QP_WINDOW_Z_MIN or less is flagged
    ('fci_split_rung_no_quasiparticle') and kept."""
    w, z, w_lin, z_lin = _qp_root(num, pol, e_p, label)
    if np.isfinite(z) and z <= QP_WINDOW_Z_MIN:
        flag('fci_split_rung_no_quasiparticle',
             f'orbital {p_mo}: the G0W_{rung.upper()} root at {-w * HARTREE_TO_EV:.4f} eV has '
             f'Z = {z:.4f}, QP_WINDOW_Z_MIN = {QP_WINDOW_Z_MIN:g} or less: no quasiparticle by '
             f'weight; the root of largest weight is kept, and so are its splits', FCISplitShort)
    rec = {'ip_ev': -w * HARTREE_TO_EV, 'z': z,
           'quasiparticle': quasiparticle_weight(z, QP_WINDOW_Z_MIN),
           'ip_linearized_ev': -w_lin * HARTREE_TO_EV, 'z_linearized': z_lin}
    return w, z, rec


def _unstable_rung():
    """The record of a rung whose screening is unstable: NaN throughout."""
    return {'ip_ev': np.nan, 'z': np.nan, 'quasiparticle': False, 'ip_linearized_ev': np.nan,
            'z_linearized': np.nan, 'min_omega_ev': np.nan, 'stable': False}


def fci_vertex_split(mf, ncore, orbitals=None, depth=None):
    """The exact FCI vertex split of the G0W@HF ionization energies of occupied orbitals, on the
    frozen-core active space of every MO above the ncore lowest: a JSON-serializable record (eV)
    with

    method            'fci_vertex_split'
    parameters        ncore, the orbitals (full-MO numbering), the screening and hole depths, and
                      the MB pyscf's FCI may hold (`max_memory_mb`)
    sizes             nmo, nocc, n_act, nelec_act, n_ov, the determinants in N and N - 1, and the
                      estimated peak memory (`memory_estimate_mb`, `check_memory`)
    reference         the RHF energy and the active Fock matrix's largest miss of diag(e) (Ha)
    ground_state      |Psi0>'s energy (with the core, Ha), its correlation energy, converged, <S^2>
                      (|Psi0> is the state pyscf's direct_spin1 reaches from its default guess:
                      from the HF determinant, the lowest singlet of its symmetry;
                      `ground_state`)
    orbitals          per orbital: its HF level, its FCI occupation, the exact one-hole main pole
                      ('exact': IP, Z, its Ritz residual bound, the second weight, its pole and
                      bound, the weight within the ambiguity window, the half-depth pole), the
                      four rungs ('rungs': rpa, tdhf, bse, fci, each IP, Z, quasiparticle, the
                      linearized IP and Z, the lowest excitation with weight, stable; the fci
                      rung also at half depth), the splits ('splits_ev') and the density side
                      ('density': the shifts s and the IPs IP[G0W0] - s, a shift moving the
                      quasiparticle energy w to w + s); both half-depth changes are
                      IP(D) - IP(D/2)
    timings_s         wall seconds of each stage
    numerical_flags   a list of {'flag', 'message'}: 'fci_split_fock_not_diagonal',
                      'fci_split_ground_not_converged', 'fci_split_ground_not_singlet',
                      'fci_split_depth_unconverged', 'fci_split_main_pole_ambiguous',
                      'fci_split_rung_no_quasiparticle', 'fci_split_rung_unstable', and the root
                      finder's 'no_root' and 'z_outside'

    orbitals: occupied orbitals ncore <= p < nocc in full-MO numbering (default the HOMO and the
    next lower occupied level outside the HOMO's degenerate set, `default_orbitals`); depth: the
    FCI rung's Lanczos depth D, an even integer (default FCI_SPLIT_SCREENING_DEPTH), checked
    against D/2. It is a numerical option on purpose (see the module docstring); the hole runs
    FCI_SPLIT_HOLE_DEPTH steps, judged by its residual bound. A wrong input (an ncore that splits a
    degenerate occupied level included), or an active space larger than the memory this job may
    use, is refused before any work (ValueError, `check_reference`, `check_arguments`,
    `check_core`, `check_memory`). An unstable Casida rung is NaN and flagged. NaN is written by
    json.dumps as the token NaN, which Python's json reads back; a value that does not apply (no
    second pole, no FCI-rung pole) is None.
    """
    nocc = check_reference(mf)
    orbitals, depth = check_arguments(nocc, ncore, orbitals, depth)
    ncore = int(ncore)
    check_core(mf.mo_energy, nocc, ncore)
    peak_mb, allowed_mb, dets_n, dets_h = check_memory(int(mf.mo_coeff.shape[1]), nocc, ncore)
    clock = {}
    t_start = time.perf_counter()
    flags = []
    with collecting(flags):
        t0 = time.perf_counter()
        h1, eri, ecore, e, m, fock_miss = active_hamiltonian(mf, ncore)
        n = len(e)
        if orbitals is None:
            orbitals = default_orbitals(e, m, ncore)
        clock['active_space'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        e0, c0, D, converged, ss = ground_state(h1, eri, n, m)
        clock['ground_state'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        rungs = casida_rungs(e, eri, m)
        clock['casida_rungs'] = time.perf_counter() - t0

        t0 = time.perf_counter()
        side = _density_side(e, eri, m, rungs, D)
        clock['density_side'] = time.perf_counter() - t0

        screening = _Screening(h1, eri, n, m, c0)
        per_orbital, clock['hole'], clock['fci_rung'] = [], 0.0, 0.0
        for p_mo in orbitals:
            p = p_mo - ncore
            t0 = time.perf_counter()
            exact = _hole(h1, eri, n, m, c0, e0, p, p_mo)
            clock['hole'] += time.perf_counter() - t0

            rec_rungs, roots = {}, {}
            for rung in ('rpa', 'tdhf', 'bse'):
                got = rungs[rung]
                if got is None:
                    roots[rung] = np.nan
                    rec_rungs[rung] = _unstable_rung()
                    continue
                omega, xpy = got
                num, pol = pole_arrays(e, m, casida_residues(eri, m, xpy, [p])[0], omega)
                roots[rung], _, rec_rungs[rung] = _rung_root(
                    num, pol, e[p], f'G0W_{rung.upper()} orbital {p_mo}', rung, p_mo)
                rec_rungs[rung].update(min_omega_ev=float(omega.min()) * HARTREE_TO_EV,
                                       stable=True)

            t0 = time.perf_counter()
            (num, pol), (num_h, pol_h), omega_min, steps, skipped = _fci_rung(
                screening, e, m, e0, p, depth)
            w_f, _, rec_fci = _rung_root(num, pol, e[p], f'G0W_FCI orbital {p_mo}', 'fci', p_mo)
            w_h = solve_qp_equation_poles(
                num_h, pol_h, e[p], label=f'G0W_FCI orbital {p_mo}, depth {depth // 2}')[0]
            clock['fci_rung'] += time.perf_counter() - t0
            if abs(w_f - w_h) > FCI_SPLIT_HALF_DEPTH_BAR:
                flag('fci_split_depth_unconverged',
                     f'orbital {p_mo}: the G0W_FCI quasiparticle energy moves by '
                     f'{abs(w_f - w_h):.2e} Ha between Lanczos depth {depth // 2} and {depth}, '
                     f'above FCI_SPLIT_HALF_DEPTH_BAR = {FCI_SPLIT_HALF_DEPTH_BAR:g} Ha (a bound '
                     f'on the depth-{depth // 2} error, the depth-{depth} one smaller); the '
                     f'depth-{depth} value is kept', FCISplitShort)
            roots['fci'] = w_f
            rec_fci.update(min_omega_ev=None if omega_min is None else omega_min * HARTREE_TO_EV,
                           stable=True,
                           ip_half_depth_ev=-w_h * HARTREE_TO_EV,
                           half_depth_change_ev=-(w_f - w_h) * HARTREE_TO_EV, depth=depth,
                           steps=steps, skipped_starts=skipped)
            rec_rungs['fci'] = rec_fci

            ip = {rung: -roots[rung] * HARTREE_TO_EV for rung in RUNGS}
            splits = {'w_side': ip['fci'] - ip['rpa'], 'sigma_side': exact['ip_ev'] - ip['fci'],
                      'tdhf_minus_fci': ip['tdhf'] - ip['fci'],
                      'bse_minus_fci': ip['bse'] - ip['fci'],
                      'g0w0_minus_exact': ip['rpa'] - exact['ip_ev']}
            per_orbital.append({
                'orbital': int(p_mo), 'active_orbital': int(p),
                'level': [int(i + ncore) for i in level_members(e, m, p)],
                'eps_hf_ev': float(e[p]) * HARTREE_TO_EV, 'occupation_fci': float(D[p, p]),
                'exact': exact, 'rungs': rec_rungs, 'splits_ev': splits,
                'density': _density_shifts(eri, m, p, side, roots['rpa'])})
    clock['total'] = time.perf_counter() - t_start
    record = {
        'method': 'fci_vertex_split',
        'parameters': {'ncore': ncore, 'orbitals': [int(p) for p in orbitals],
                       'screening_depth': depth, 'hole_depth': FCI_SPLIT_HOLE_DEPTH,
                       'max_memory_mb': allowed_mb},
        'sizes': {'nmo': int(mf.mo_coeff.shape[1]), 'nocc': nocc, 'n_act': n,
                  'nelec_act': 2 * m, 'n_ov': m * (n - m), 'dets_n': int(dets_n),
                  'dets_n_minus_1': int(dets_h), 'memory_estimate_mb': peak_mb},
        'reference': {'e_hf_ha': float(mf.e_tot), 'fock_miss_ha': fock_miss},
        'ground_state': {'e_fci_ha': e0 + ecore, 'e_corr_ha': e0 + ecore - float(mf.e_tot),
                         'converged': converged, 'spin_square': ss},
        'orbitals': per_orbital,
        'timings_s': {k: float(v) for k, v in clock.items()},
        'numerical_flags': gathered(flags),
    }
    return record
