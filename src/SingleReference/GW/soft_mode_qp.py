"""Quasiparticle energies of the triplet-carrying vertex self-energies on TDHF screening (PSD2,
PSD4, PSD7, PSD9: `GW.self_energy.TRIPLET_VERTEX_MODES`) with the soft triplet modes replaced
by their collective levels (`LinearResponse.soft_mode.soft_mode_triplets`).

The restricted PSD2 on TDHF is the spin-orbital electron-hole pair self-energy
(tests/test_tdhf_vertex_is_bare.py),

    Sigma_pp(w) = 1/2 sum_S (2 a_S - b_S)^2 g_S(w) + 3/2 sum_T b_T^2 g_T(w),

whose triplet sum runs over the TDHF triplet excitations. At the RHF->UHF instability the lowest
of them goes to zero and b_T^2 grows as 1/Omega_T, so the house route
(`GW.qp_energy.calc_qp_energy` with selfenergy='PSD2', polarizability='TDHF') loses its
quasiparticle before the instability (H2/cc-pVDZ at 1.19 A: Z = 0.36) and refuses the reference
past it (`LinearResponse.casida.CasidaInstability`). Here the triplet list is the soft-mode
split's; everything else is that route's own: the singlet TDHF modes from
`LinearResponse.casida.CasidaSolver`, the amplitudes and the pole form from
`GW.self_energy.SelfEnergySolver` with a bare vertex, and the root from the one root finder
(`Solvers.qp_equation.solve_qp_equation_scan`, which scans about the HF energy as the Casida
route does, here with the pole form's exact slope). With model=None the triplet list is the
plain TDHF one, the house route's spectrum.

Energies are in Hartree.
"""
import numpy as np

from src.Base.numerical_flags import collecting, gathered
from src.SingleReference.common.eri_blocks import mo_eri
from src.SingleReference.GW.self_energy import TRIPLET_VERTEX_MODES, SelfEnergySolver
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.SingleReference.LinearResponse.soft_mode import (MODELS, check_reference,
                                                          soft_mode_triplets)
from src.Solvers.qp_equation import solve_qp_equation_scan


def _pole_equation(res, pol, a):
    """(f, f') of the quasiparticle equation f(w) = w - a - sum res / (w - pol) = 0."""
    def residual(w):
        return w - a - np.sum(res / (w - pol))

    def slope(w):
        return 1.0 + np.sum(res / (w - pol) ** 2)
    return residual, slope


def soft_mode_qp(mf, selfenergy='PSD2', states=None, model='1d'):
    """Quasiparticle energies (Ha) of a triplet-carrying PSD self-energy on TDHF screening, its
    soft triplet modes replaced by their collective levels: a dict with

    states            the orbitals solved (default the HOMO)
    eps, Z            their quasiparticle energies (Ha) and pole strengths
    triplets          the `LinearResponse.soft_mode.soft_mode_triplets` record (None for model=None)
    numerical_flags   the root finder's ('no_root', 'z_outside') and the split's

    selfenergy is one of `GW.self_energy.TRIPLET_VERTEX_MODES`; model is '1d' or '3d', or None
    for the plain TDHF triplets (refused past the instability, as the house route refuses).
    """
    if selfenergy not in TRIPLET_VERTEX_MODES:
        raise ValueError(f'selfenergy = {selfenergy!r}: one of {TRIPLET_VERTEX_MODES}')
    if model is not None and model not in MODELS:
        raise ValueError(f'model = {model!r}: one of {MODELS}, or None')
    nocc = check_reference(mf)
    eps = np.asarray(mf.mo_energy)
    states = [nocc - 1] if states is None else [int(p) for p in states]
    eri = mo_eri(mf, mf.mol)
    lr = LinearResponseSolver(eps, eri_chemist=eri, spin_mode='restricted')
    om_s, X_s, Y_s = (np.asarray(v) for v in
                      CasidaSolver(*lr.build_casida_matrices(nocc, lBSE=True)).solve())
    flags = []
    with collecting(flags):
        if model is None:
            rec = None
            om_t, X_t, Y_t = (np.asarray(v) for v in CasidaSolver(
                *lr.build_casida_matrices(nocc, lBSE=True, triplet=True)).solve())
        else:
            rec = soft_mode_triplets(mf, model)
            om_t, X_t, Y_t = rec['omega'], rec['X'], rec['Y']
        se = SelfEnergySolver(eps, eri_chemist=eri, spin_mode='restricted', eta=0.0)
        w_qp, z_qp = [], []
        for p in states:
            ev = se.self_energy_evaluator(
                p, nocc, om_s, se.get_chi_a(nocc, X_s, Y_s, p_state=p),
                se.get_chi_b_vertex(nocc, X_s, Y_s, eri_w=None, p_state=p),
                eigenvalues_casida_t=om_t,
                chiXYb_t=se.get_chi_b_vertex(nocc, X_t, Y_t, eri_w=None, p_state=p),
                vertex_mode=selfenergy)
            res, pol = ev.poles()
            residual, slope = _pole_equation(res, pol, eps[p])
            w, z = solve_qp_equation_scan(residual, slope, eps[p],
                                          label=f'{selfenergy} orbital {p}')
            w_qp.append(w)
            z_qp.append(z)
    return dict(states=states, eps=np.array(w_qp), Z=np.array(z_qp), triplets=rec,
                numerical_flags=gathered(flags))
