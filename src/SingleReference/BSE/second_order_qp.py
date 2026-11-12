"""The quasiparticle energies that feed the final kernels of the BSE(2) paper
(van Hoegaerden and Foerster): its AMS sequences, cycle by cycle.

====================  ========================================================
sequence              cycles, each at the energies the one before left
====================  ========================================================
'G0W0'                G0W0@HF: the RPA at the HF energies, plain GW
'Sigma_TDHF@L_TDHF'   TDHF at the HF energies and GWGammaInf with the bare
                      vertex: the HF sequence's cycle 1, feeding BSE(2)@HF
'Sigma_BSE@L_BSE'     G0W0@HF; then the static RPA W0 and BSE(1) (W0 in the
                      direct term of A(1) and B(1)) at those energies and
                      GWGammaInf with W0 in the vertex: the GW sequence's
                      cycles 1 and 2, feeding BSE(2)@GW and BSE(2)_Wv@GW
'G0F2'                Sigma(2) at the HF energies, unbroadened, as AMS
                      evaluates it: the GF2 sequence's cycle 1, feeding
                      BSE(1)@GF2
====================  ========================================================

Every cycle keeps the RHF orbitals, puts the poles of Sigma_c (G) and the
screening at the energies of every orbital the previous cycle left (the HF
ones for the first), and solves, for every orbital p,

    w = eps_HF,p + Re Sigma_c,pp(w),

anchored on the reference energies in every cycle as AMS anchors it (Sigma_x
- v_x vanishes on HF). Re Sigma_c = sum_k r_k x_k / (x_k^2 + eta^2), with
x_k = w - P_k over the poles P_k and residues r_k of MBPTcode's self-energy
(`GW.self_energy`; `ADC.second_order.sigma2_poles` for G0F2), whose terms are
AMS's. The cycles screen with singlet responses only; tda puts every response
of the sequence in the Tamm-Dancoff approximation, the RPA behind W0
included, as the paper's TDA runs did.

Each equation's root is AMS's (moGWmodule selfEnergy): one scan of
AMS_QP_GRID_POINTS points over eps_HF,p -+ h, h by |eps_HF,p - mu| from
AMS_QP_WINDOW_TIERS with mu halfway between the HOMO and LUMO the poles sit at;
every sign change between neighbouring points bisected as AMS bisects it, to
AMS_QP_BISECTION_WIDTH; and among those midpoints the one of largest weight Z
= 1/(1 - dSigma/dw) > 0 kept, Z > 1 included; an orbital without one keeps
its HF energy. The scan and the bisection are `Solvers.qp_equation.
SampledEquation`'s, the choice AMS's own (`ams_qp_root`). Two things differ
from AMS on purpose. The root kept is converged inside the bracket AMS's
bisection left it in, where AMS takes that bracket's midpoint, on a lattice of
2.15 meV about eps_HF,p: at most 1.07 meV from AMS's on the same self-energy.
And in the unbroadened G0F2 a sign change across a pole of Sigma(2) is dropped
as a pole, where AMS keeps it as a root of vanishing weight; no kept root
moves.
"""
import contextlib

import numpy as np

from src.Base.numerical_flags import collecting, flag, gathered
from src.Base.numerics.constants import (
    AMS_QP_BISECTION_STEPS, AMS_QP_BISECTION_WIDTH, AMS_QP_GRID_POINTS,
    AMS_QP_WINDOW_TIERS, DEFAULT_BROADENING_ETA, QP_POLE_SUM_BLOCK,
    QP_ROOT_MAX_ITER, QP_ROOT_TOL, QP_WINDOW_Z_MIN)
from src.Base.pyscf.pyscf_interface import get_density_fitting_coefficients
from src.SingleReference.ADC.second_order import sigma2_poles
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.second_order_kernel import static_screening_aux
from src.Solvers.qp_equation import SampledEquation

#: The cycles of each sequence, in order (module docstring).
QP_SEQUENCES = {
    'G0W0': ('G0W0',),
    'Sigma_TDHF@L_TDHF': ('Sigma_TDHF@L_TDHF',),
    'Sigma_BSE@L_BSE': ('G0W0', 'Sigma_BSE@L_BSE'),
    'G0F2': ('G0F2',),
}
CYCLES = ('G0W0', 'Sigma_TDHF@L_TDHF', 'Sigma_BSE@L_BSE', 'G0F2')


def pole_sum(residues, poles, w, eta):
    """(Re Sigma(w), dRe Sigma/dw) of Sigma(w) = sum_k r_k / (w - P_k) broadened
    by eta: sum_k r_k x_k / (x_k^2 + eta^2) and its slope sum_k r_k (eta^2 -
    x_k^2) / (x_k^2 + eta^2)^2, x_k = w - P_k; eta = 0 is the pole sum itself,
    inf or NaN on a pole, which the scan reads as one (`SampledEquation.solve`).
    w: a frequency or an array of them (Ha); the frequency x pole products are
    formed QP_POLE_SUM_BLOCK elements at a time.
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    value, slope = np.empty(w.size), np.empty(w.size)
    step = max(1, QP_POLE_SUM_BLOCK // max(1, poles.size))
    on_pole = (np.errstate(divide='ignore', invalid='ignore') if eta == 0.0
               else contextlib.nullcontext())
    with on_pole:
        for i in range(0, w.size, step):
            x = w[i:i + step, None] - poles[None, :]
            d = 1.0 / (x * x + eta * eta)
            value[i:i + step] = (x * d) @ residues
            slope[i:i + step] = ((eta * eta - x * x) * d * d) @ residues
    return value, slope


def ams_window(a, mu):
    """AMS's one scan window for an orbital anchored at a: a -+ h, h the half
    width of the last AMS_QP_WINDOW_TIERS tier whose depth |a - mu| exceeds."""
    half = AMS_QP_WINDOW_TIERS[0][1]
    for depth, width in AMS_QP_WINDOW_TIERS:
        if abs(a - mu) > depth:
            half = width
    return a - half, a + half


def ams_bracket_width(lo, hi):
    """The width of the bracket AMS's bisection leaves a sign change of its
    scan over [lo, hi] in: one grid cell, halved until it is within
    AMS_QP_BISECTION_WIDTH, AMS_QP_BISECTION_STEPS times at most."""
    width = (hi - lo) / (AMS_QP_GRID_POINTS - 1)
    for _ in range(AMS_QP_BISECTION_STEPS):
        width *= 0.5
        if width <= AMS_QP_BISECTION_WIDTH:
            break
    return width


def _equation(residues, poles, a, eta):
    """f(w) = w - a - Re Sigma(w), on the grid's array or at one point, and
    f'(w) at one point: the bisection and the Newton step ask f and f' at one
    point in turn, so the last point's pole sum is kept for both."""
    residues, poles = np.asarray(residues, float), np.asarray(poles, float)
    last = {}

    def at(w):
        if w not in last:
            last.clear()
            last[w] = pole_sum(residues, poles, w, eta)
        return last[w]

    def f(w):
        if np.ndim(w):
            return np.asarray(w, dtype=float) - a - pole_sum(residues, poles, w, eta)[0]
        return float(w) - a - float(at(float(w))[0][0])

    def fprime(w):
        return 1.0 - float(at(float(w))[1][0])
    return f, fprime


def _weight(fprime, w):
    """Z = 1/f'(w), infinite where f' vanishes, as AMS's quotient makes it."""
    slope = fprime(w)
    return 1.0 / slope if slope != 0.0 else np.inf


def ams_roots(residues, poles, a, mu, eta):
    """Every (w, Z) of w = a + Re Sigma(w) that AMS's scan finds, in increasing
    order: each sign change between neighbouring points of `ams_window`'s grid
    of AMS_QP_GRID_POINTS points, no cell halved, bisected as AMS bisects it
    (the right end first, to AMS_QP_BISECTION_WIDTH or AMS_QP_BISECTION_STEPS
    halvings) and taken at its bracket's midpoint, with Z = 1/(1 -
    dSigma/dw) there from the closed-form slope, of any sign. With eta = 0 a
    sign change across a pole is dropped as a pole (`SampledEquation.solve`),
    where AMS keeps a root of vanishing weight.

    residues, poles: Sigma(w) = sum_k r_k / (w - P_k) before broadening; eta:
    its broadening (`pole_sum`).
    """
    f, fprime = _equation(residues, poles, a, eta)
    lo, hi = ams_window(a, mu)
    scan = SampledEquation(f, vectorized=True, n_omega=AMS_QP_GRID_POINTS,
                           tol=AMS_QP_BISECTION_WIDTH, max_iter=AMS_QP_BISECTION_STEPS,
                           windows=[(lo, hi)], split_depth=0)
    brackets, _ = scan.brackets(lo, hi)
    return tuple((w, _weight(fprime, w)) for bracket in brackets
                 for w, _ in scan.solve(bracket))


def ams_qp_root(residues, poles, a, mu, eta):
    """(w, Z) of AMS's quasiparticle root: among `ams_roots`, the one of
    largest Z > 0, Z > 1 included, the lowest of equal weights (moGWmodule
    selfEnergy keeps a root whose Z exceeds the best so far, from 0),
    converged by Newton inside the bracket AMS's bisection left it in
    (`ams_bracket_width`), Z then at the converged root; (a, NaN) when no root
    has a positive weight, the anchor AMS keeps then. AMS chooses among the
    midpoints, so the choice is made there too.
    """
    positive = [(w, z) for w, z in ams_roots(residues, poles, a, mu, eta) if z > 0.0]
    if not positive:
        return float(a), np.nan
    w, z = max(positive, key=lambda root: root[1])
    half = 0.5 * ams_bracket_width(*ams_window(a, mu))
    f, fprime = _equation(residues, poles, a, eta)
    polish = SampledEquation(f, fprime=fprime, vectorized=True, n_omega=2,
                             tol=QP_ROOT_TOL, max_iter=QP_ROOT_MAX_ITER,
                             windows=[(w - half, w + half)], split_depth=0)
    brackets, _ = polish.brackets(w - half, w + half)
    converged = [root for bracket in brackets for root in polish.solve(bracket)]
    if converged:
        w, z = converged[0]
    return float(w), float(z)


def fermi_level(eps, nocc):
    """AMS's mu: halfway between the highest occupied and the lowest virtual
    of `eps`, refused where they cross (the poles of Sigma would change side)."""
    homo, lumo = np.max(eps[:nocc]), np.min(eps[nocc:])
    if not homo < lumo:
        raise ValueError(f'the quasiparticle HOMO {homo!r} Ha lies at or above the '
                         f'LUMO {lumo!r} Ha: occupied and virtual levels crossed, '
                         'so the poles of the next self-energy cannot be placed')
    return 0.5 * (homo + lumo)


def cycle_self_energy(cycle, eps_prev, coeff_df, nocc, tda=False, eta=None):
    """(poles_of, eta) of one cycle of CYCLES (module docstring): poles_of(p)
    gives the (residues, poles) of its Sigma_c,pp before broadening, with
    the poles and the screening at eps_prev, and eta the broadening its
    equations take (DEFAULT_BROADENING_ETA when None; G0F2 is unbroadened, as
    AMS evaluates it, and refuses an eta).

    eps_prev: (norb,) Ha; coeff_df: (naux, norb, norb), (pq|rs) = sum_Q
    B_Qpq B_Qrs; tda: the cycle's responses, the RPA behind W0 included, in
    the Tamm-Dancoff approximation.
    """
    if cycle not in CYCLES:
        raise ValueError(f'cycle {cycle!r} is not one of {CYCLES}')
    eps_prev = np.asarray(eps_prev, dtype=float)
    if cycle == 'G0F2':
        if eta is not None:
            raise ValueError('G0F2 is unbroadened, as AMS evaluates it: no eta')

        def poles_of(p):
            res_D, res_X, poles = sigma2_poles(eps_prev, nocc, p, B_aa=coeff_df)
            return res_D + res_X, poles
        return poles_of, 0.0
    eta = DEFAULT_BROADENING_ETA if eta is None else float(eta)
    W0 = (static_screening_aux(eps_prev, coeff_df, nocc, tda=tda)
          if cycle == 'Sigma_BSE@L_BSE' else None)
    A, B = LinearResponseSolver(eps_prev, coeff_df=coeff_df).build_casida_matrices(
        nocc, lBSE=cycle != 'G0W0', W_aux=W0)
    omega, X, Y = (np.asarray(v) for v in CasidaSolver(A, B).solve(tda=tda))
    del A, B
    mode = 'GW' if cycle == 'G0W0' else 'GWGammaInf'
    sigma = SelfEnergySolver(eps_prev, df_coeff=coeff_df, spin_mode='restricted', eta=eta)

    def poles_of(p):
        chi_a = sigma.get_chi_a(nocc, X, Y, spin_channel='alpha', p_state=p)
        chi_b = (None if mode == 'GW' else sigma.get_chi_b_vertex(
            nocc, X, Y, spin_channel='alpha', eri_w=W0, p_state=p))
        return sigma.self_energy_evaluator(p, nocc, omega, chi_a, chi_b,
                                           vertex_mode=mode).poles()
    return poles_of, eta


def quasiparticle_cycle(cycle, eps_hf, eps_prev, coeff_df, nocc, tda=False,
                        eta=None):
    """(eps, Z, mu) of one cycle of CYCLES (module docstring): every orbital's
    root of w = eps_HF,p + Re Sigma_c,pp(w) by AMS's rule (`ams_qp_root`), with
    Sigma_c's poles and screening at eps_prev (`cycle_self_energy`, which
    takes tda and eta), and AMS's mu of eps_prev, which sets the windows.

    eps_hf, eps_prev: (norb,) Ha. Z is NaN for an orbital that keeps its HF
    energy, which raises the named flag 'qp_sequence_no_root'; a root kept
    with Z <= QP_WINDOW_Z_MIN raises 'qp_sequence_low_weight', one with Z > 1
    'qp_sequence_weight_above_one'.
    """
    eps_hf = np.asarray(eps_hf, dtype=float)
    eps_prev = np.asarray(eps_prev, dtype=float)
    mu = fermi_level(eps_prev, nocc)
    poles_of, eta = cycle_self_energy(cycle, eps_prev, coeff_df, nocc, tda=tda, eta=eta)
    eps, Z = np.empty(eps_hf.size), np.empty(eps_hf.size)
    for p in range(eps_hf.size):
        residues, poles = poles_of(p)
        eps[p], Z[p] = ams_qp_root(residues, poles, eps_hf[p], mu, eta)
    kept = np.flatnonzero(np.isnan(Z))
    if kept.size:
        flag('qp_sequence_no_root',
             f'{cycle}: orbitals {kept.tolist()} have no root of positive weight in '
             "AMS's window and keep their HF energies, as AMS keeps them",
             RuntimeWarning)
    low = np.flatnonzero((Z > 0.0) & (Z <= QP_WINDOW_Z_MIN))
    if low.size:
        flag('qp_sequence_low_weight',
             f'{cycle}: the roots kept for orbitals {low.tolist()} have Z <= '
             f'{QP_WINDOW_Z_MIN} ({np.round(Z[low], 3).tolist()}): no quasiparticle, '
             'kept as AMS keeps them', RuntimeWarning)
    above = np.flatnonzero(Z > 1.0)
    if above.size:
        flag('qp_sequence_weight_above_one',
             f'{cycle}: the roots kept for orbitals {above.tolist()} have Z > 1 '
             f'({np.round(Z[above], 3).tolist()}): AMS keeps the largest weight, '
             'one above 1 included, which no positive-weight self-energy has',
             RuntimeWarning)
    return eps, Z, mu


def qp_sequence(mf, name, tda=False, eta=None, coeff_df=None):
    """The quasiparticle sequence `name` of QP_SEQUENCES on a closed-shell RHF
    (module docstring): a dict with 'name', 'tda', 'eps' and 'Z' of its last
    cycle (Ha; Z NaN where the HF energy was kept), 'cycles', one dict per
    cycle with its 'name', 'eps', 'Z' and 'mu', and 'numerical_flags', the
    named flags its cycles raised.

    tda: every response of the sequence in the Tamm-Dancoff approximation.
    eta: the broadening of Sigma_c, DEFAULT_BROADENING_ETA when None (G0F2,
    unbroadened as in AMS, refuses one).
    coeff_df: the three-index factors of `mf` (`get_density_fitting_coefficients`),
    built when None.
    """
    if name not in QP_SEQUENCES:
        raise ValueError(f'quasiparticle sequence {name!r} is not one of '
                         f'{tuple(QP_SEQUENCES)}')
    if getattr(mf, 'xc', None) is not None:
        raise NotImplementedError(
            "the paper's quasiparticle sequences start from a closed-shell RHF; "
            'a Kohn-Sham reference has no place in them')
    if getattr(mf, 'mo_occ', None) is None:
        raise ValueError('the reference has no orbitals: run the RHF first')
    if np.ndim(mf.mo_occ) != 1:
        raise NotImplementedError(
            "the paper's quasiparticle sequences start from a closed-shell RHF; "
            'an unrestricted reference has no place in them')
    occ = np.asarray(mf.mo_occ)
    nocc = int(np.count_nonzero(occ > 0))
    if not (np.all(occ[:nocc] == 2.0) and np.all(occ[nocc:] == 0.0)):
        raise ValueError('the reference is not a closed-shell aufbau RHF')
    if coeff_df is None:
        coeff_df = get_density_fitting_coefficients(mf.mol, mf)
    coeff_df = np.asarray(coeff_df)
    eps_hf = np.asarray(mf.mo_energy, dtype=float)
    eps, cycles, flags = eps_hf, [], []
    with collecting(flags):
        for cycle in QP_SEQUENCES[name]:
            eps, Z, mu = quasiparticle_cycle(cycle, eps_hf, eps, coeff_df, nocc,
                                             tda=tda, eta=eta)
            cycles.append({'name': cycle, 'eps': eps, 'Z': Z, 'mu': mu})
    return {'name': name, 'tda': bool(tda), 'eps': eps, 'Z': cycles[-1]['Z'],
            'cycles': cycles, 'numerical_flags': gathered(flags)}
