"""The soft-mode split of a closed shell's TDHF triplet instability (`LinearResponse.soft_mode`)
and the PSD self-energies on its levels (`GW.soft_mode_qp`): what is held.

1. The triplet matrices built from MO blocks are the house's (build_casida_matrices with
   triplet=True) to CROSS_PATH_RTOL: H2 and H2O, cc-pVDZ.
2. The split is exact: mu / M = Omega_s^2 to 1e-12 of the largest Omega_n^2 and d^T e = 1 to
   1e-12, and the soft mode put back at its harmonic level is the house Casida's root 0,
   (Omega, X, Y) to 1e-10 (H2 at 1.0 A; N2, whose soft direction spans two degenerate pi pairs).
3. The RHF->UHF path: V''(0) = mu to 1e-5 relative (h and 2h Richardson, h = 4e-3; 1.3e-6
   measured), D(Q) =
   sin Q for one occupied orbital to 1e-12, and N2's half-period pi / sigma_1 = 1.42 pi with
   sigma_1 = sigma_2 to 1e-9 (its two pi pairs).
4. The collective solvers on a harmonic V give Omega and <1|Q|0>^2 = 1/(2 M Omega) to 1e-5
   relative: 1D periodic, 1D between walls, 3D (5e-7, 5e-7, 5.4e-6 measured).
5. PSD2 with the soft mode at its harmonic level is PSD2@TDHF: the pole sums agree to
   CROSS_PATH_RTOL at five frequencies (H2/cc-pVDZ, 1.0 A), and with model=None the driver gives
   `calc_qp_energy`'s PSD2@TDHF HOMO to 1e-6 eV (2.6e-8 eV measured).
6. Through the instability (H2/cc-pVDZ, R_CF = 1.2104 A): Omega_T^2 > 0 at 1.20 A and < 0 at
   1.212 A, where the house triplet Casida refuses (CasidaInstability) and the split gives a
   finite collective triplet.
7. Against FCI (the house's PyscfFCISolver; H2/cc-pVDZ at 0.75, 1.0, 1.212, 1.5 and 2.0 A): the
   collective 1D triplet within 1.0 eV (0.83 eV the worst, at 1.5 A), and closer than TDA from
   1.0 A on (TDA 0.8 to 1.4 eV low, negative at 2.0 A).
8. PSD2 through the instability: PSD2@TDHF has no quasiparticle at 1.19 A (Z = 0.36, IP 26 eV)
   and is refused at 1.212 A; with the soft mode split off Z > 0.7 at 0.75 to 1.212 A, the IP
   moves by 0.03 eV from 1.19 to 1.212 A, and it lies closer to FCI than PSD2@TDHF at 0.75, 1.0
   and 1.19 A.
9. Molecules at their MP2/cc-pVDZ minima, against the recorded EOM-CCSD of
   tests/soft_mode_references.py: the collective 1D triplet of the soft modes of N2, H2CO,
   ethylene and O3 within 1.2 eV (-0.48, +0.69, +0.52, +1.15 eV measured) and closer than the TDA
   root of the same mode (TDA misses by -1.60, -1.20, -0.93 eV, and is negative for O3); H2O, whose
   triplet is stiff (TDHF 8.09 eV), is the control: within 1.1 eV (+1.00 measured), where TDHF is
   closer (+0.65).
10. Refusals before any work: a UHF, a DFT, a density-fitted and an unconverged reference; an
    unknown model, boundary or self-energy; an A - B that is not positive definite
    (CasidaInstability).
11. Flags: O3's three soft modes raise 'soft_modes_coupled' (SoftModesCoupled); an edge-weight bar
    of zero raises 'soft_mode_edge_weight' (SoftModeEdgeWeight); both reach numerical_flags.

The reference numbers come from pyscf FCI run here and from EOM-CCSD recorded on a workstation
(provenance in tests/soft_mode_references.py); the bars hold measured misses at 1 to 4 threads.

Run: python tests/test_soft_mode_triplet.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import ao2mo, gto, scf

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.common.eri_blocks import mo_eri
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.self_energy import SelfEnergySolver
from src.SingleReference.GW.soft_mode_qp import soft_mode_qp
from src.SingleReference.LinearResponse import soft_mode as SM
from src.SingleReference.LinearResponse.casida import CasidaInstability, CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from src.Solvers.active_space_solver import PyscfFCISolver
from tests.pins import CROSS_PATH_RTOL
from tests.soft_mode_references import EOM_CCSD, GEOMETRIES, rhf

#: Bar of the split's exact identities: mu / M = Omega_s^2 relative to the largest |Omega_n^2|,
#: the scale the rounding of T = (A - B)^1/2 (A + B) (A - B)^1/2 goes with (332 Ha^2 for N2,
#: whose Omega_s^2 is 5.1e-3: relative to Omega_s^2 itself N2 misses by up to 4e-11), and
#: d^T e = 1. Measured over 40 orders of the pairs, six OpenBLAS kernels and 1 to 4 threads on a
#: workstation: at most 6e-16 (N2) and 7e-16.
SPLIT_RTOL = 1e-12
#: Ha. The harmonic collective level against the house Casida root: 3e-13 (N2) measured, 4e-12
#: at most over six OpenBLAS kernels and 1 to 4 threads.
HARMONIC_ROOT_TOL = 1e-10
#: N2's sigma_1 = sigma_2 (its two pi pairs): 2e-13 measured, 9e-12 at most over 40 orders of the
#: pairs, six OpenBLAS kernels and 1 to 4 threads.
SIGMA_PAIR_TOL = 1e-9
#: Relative bar of V''(0) = mu, Richardson from h and 2h (h = 4e-3, V(0) = 0 and V even); a
#: single difference at h = 1e-3 loses 1e-5 to rounding for N2, whose V(h) is 6e-9 Ha. Measured:
#: 4e-10 (H2) and 1.3e-6 (N2).
CURVATURE_RTOL = 1e-5
#: D(Q) = sin Q for one occupied orbital: 1e-16 measured.
SIN_TOL = 1e-12
#: Relative bar of the collective solvers on a harmonic V: 5.4e-6 measured (3D).
SOLVER_RTOL = 1e-5
#: eV. model=None against calc_qp_energy's PSD2@TDHF: 2.6e-8 measured (bisection stop 1e-8 Ha).
DEFAULT_ROUTE_TOL = 1e-6
#: eV. Collective 1D triplet against FCI for H2: 0.83 measured at 1.5 A.
FCI_TRIPLET_TOL = 1.0
#: eV. The soft-mode PSD2 IP between 1.19 and 1.212 A: 0.034 measured.
ACROSS_TOL = 0.1
#: Pole strength the soft-mode PSD2 HOMO keeps from 0.75 to 1.212 A: 0.84 measured.
Z_MIN_SOFT = 0.7
#: eV. Collective 1D triplet against EOM-CCSD for the soft modes: 1.15 measured (O3).
EOM_TRIPLET_TOL = 1.2
#: eV. The same for H2O's stiff triplet: 1.00 measured.
EOM_STIFF_TOL = 1.1

H2_DISTANCES = (0.75, 1.0, 1.212, 1.5, 2.0)
SOFT_MOLECULES = ('N2', 'H2CO', 'ethylene', 'O3')


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def relative(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(1.0, np.max(np.abs(b))))


def h2(R, basis='cc-pvdz'):
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {R}', basis=basis, unit='Angstrom', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def fci_h2(mf):
    """(triplet - singlet gap, IP) of H2 by the house's FCI solver on the RHF MO integrals (Ha)."""
    C, n = mf.mo_coeff, mf.mo_coeff.shape[1]
    h1 = C.T @ mf.get_hcore() @ C
    eri = ao2mo.restore(1, ao2mo.kernel(mf.mol, C), n)
    fci = PyscfFCISolver()
    e_s = fci.roots(h1, eri, n, (1, 1), spin=0)[0][0]
    e_t = fci.roots(h1, eri, n, (1, 1), spin=2)[0][0]
    e_c = fci.roots(h1, eri, n, (1, 0), spin=None)[0][0]
    return e_t - e_s, e_c - e_s


def triplet_casida(mf):
    nocc = mf.mol.nelectron // 2
    lr = LinearResponseSolver(mf.mo_energy, eri_chemist=mo_eri(mf, mf.mol), spin_mode='restricted')
    return lr.build_casida_matrices(nocc, lBSE=True, triplet=True)


def check_matrices():
    ok = True
    for label, mf in (('H2/cc-pVDZ 1.0 A', h2(1.0)), ('H2O/cc-pVDZ', rhf(GEOMETRIES['H2O']))):
        A, B = SM.triplet_tdhf_matrices(mf)
        Ah, Bh = triplet_casida(mf)
        miss = max(relative(A, Ah), relative(B, Bh))
        ok &= check(miss < CROSS_PATH_RTOL, f'{label}: triplet A, B from MO blocks = the house builder',
                    f'{miss:.1e}')
    return ok


def check_split_exact():
    ok = True
    for label, mf in (('H2 1.0 A', h2(1.0)), ('N2', rhf(GEOMETRIES['N2']))):
        A, B = SM.triplet_tdhf_matrices(mf)
        split = SM.soft_split(A, B)
        mode = split['modes'][0]
        miss = (abs(mode['stiffness'] / mode['mass'] - mode['omega2'])
                / np.max(np.abs(split['omega2'])))
        ok &= check(miss < SPLIT_RTOL and abs(mode['direction'] @ mode['conjugate'] - 1) < SPLIT_RTOL,
                    f'{label}: mu / M = Omega_s^2 and d^T e = 1', f'{miss:.1e}')
        om = np.sqrt(mode['omega2'])
        c1 = np.array([1.0 / np.sqrt(2 * mode['mass'] * om)])
        dE, X, Y = SM.collective_modes(np.array([om]), c1, mode['mass'], mode['direction'],
                                       mode['conjugate'])
        w, Xh, Yh = (np.asarray(v) for v in CasidaSolver(*triplet_casida(mf)).solve())
        sign = np.sign((X[:, 0] + Y[:, 0]) @ (Xh[:, 0] + Yh[:, 0]))
        miss = max(abs(dE[0] - w[0]), np.abs(sign * X[:, 0] - Xh[:, 0]).max(),
                   np.abs(sign * Y[:, 0] - Yh[:, 0]).max())
        ok &= check(miss < HARMONIC_ROOT_TOL,
                    f'{label}: the harmonic level of the soft mode = the house Casida root 0',
                    f'{miss:.1e}')
    return ok


def check_path():
    ok = True
    h = 4e-3
    for label, mf in (('H2 1.0 A', h2(1.0)), ('N2', rhf(GEOMETRIES['N2']))):
        nocc = mf.mol.nelectron // 2
        mode = SM.soft_split(*SM.triplet_tdhf_matrices(mf))['modes'][0]
        q, V, D = SM.path_samples(mf, mode['direction'], h * (SM.SOFT_MODE_PATH_POINTS - 1))
        curv = (4 * (2 * (V[1] - V[0]) / h ** 2) - (V[2] - V[0]) / (2 * h ** 2)) / 3
        miss = abs(curv - mode['stiffness']) / abs(mode['stiffness'])
        ok &= check(miss < CURVATURE_RTOL, f"{label}: V''(0) = mu", f'{miss:.1e}')
        L, sigma = SM.path_period(mode['direction'], nocc)
        if nocc == 1:
            q, V, D = SM.path_samples(mf, mode['direction'], L)
            ok &= check(np.abs(D - np.sin(q)).max() < SIN_TOL, f'{label}: D(Q) = sin Q, period 2 pi',
                        f'{np.abs(D - np.sin(q)).max():.1e}')
        else:
            ok &= check(abs(sigma[0] - sigma[1]) < SIGMA_PAIR_TOL and abs(L / np.pi - 1.4227) < 1e-3,
                        f'{label}: half-period pi / sigma_1 with sigma_1 = sigma_2 (two pi pairs)',
                        f'L = {L / np.pi:.4f} pi, |sigma_1 - sigma_2| = {abs(sigma[0] - sigma[1]):.1e}')
    return ok


def check_solvers():
    ok = True
    M, mu = 2.0, 2.0
    q = np.linspace(0.0, np.pi, SM.SOFT_MODE_PATH_POINTS)
    for model, boundary in (('1d', 'periodic'), ('1d', 'walls'), ('3d', 'periodic')):
        dE, cm, _ = SM.collective_levels(M, q, 0.5 * mu * q ** 2, q, np.pi, model, boundary)
        miss = max(abs(dE[0] - 1.0), abs(cm[0] ** 2 - 0.25) / 0.25)
        ok &= check(miss < SOLVER_RTOL, f'harmonic V, {model} {boundary}: Omega and 1/(2 M Omega)',
                    f'{miss:.1e}')
    return ok


def psd2_poles(mf, triplets, p):
    """(res, pol) of PSD2@TDHF for orbital p with the given triplet list (omega, X, Y)."""
    nocc = mf.mol.nelectron // 2
    eps = mf.mo_energy
    eri = mo_eri(mf, mf.mol)
    lr = LinearResponseSolver(eps, eri_chemist=eri, spin_mode='restricted')
    om_s, X_s, Y_s = (np.asarray(v) for v in CasidaSolver(*lr.build_casida_matrices(nocc, lBSE=True)).solve())
    om_t, X_t, Y_t = triplets
    se = SelfEnergySolver(eps, eri_chemist=eri, spin_mode='restricted', eta=0.0)
    ev = se.self_energy_evaluator(p, nocc, om_s, se.get_chi_a(nocc, X_s, Y_s, p_state=p),
                                  se.get_chi_b_vertex(nocc, X_s, Y_s, eri_w=None, p_state=p),
                                  eigenvalues_casida_t=om_t,
                                  chiXYb_t=se.get_chi_b_vertex(nocc, X_t, Y_t, eri_w=None, p_state=p),
                                  vertex_mode='PSD2')
    return ev.poles()


def check_psd_identity():
    ok = True
    mf = h2(1.0)
    split = SM.soft_split(*SM.triplet_tdhf_matrices(mf))
    mode = split['modes'][0]
    om = np.sqrt(mode['omega2'])
    harmonic = SM.collective_modes(np.array([om]), np.array([1.0 / np.sqrt(2 * mode['mass'] * om)]),
                                   mode['mass'], mode['direction'], mode['conjugate'])
    swapped = tuple(np.concatenate([r, h], axis=-1) if r.ndim == 1 else np.hstack([r, h])
                    for r, h in zip(split['regular'], harmonic))
    plain = tuple(np.asarray(v) for v in CasidaSolver(*triplet_casida(mf)).solve())
    w = mf.mo_energy[0] + np.array([-0.3, -0.1, 0.05, 0.2, 0.6])
    sig = []
    for trip in (swapped, plain):
        res, pol = psd2_poles(mf, trip, 0)
        sig.append(np.array([np.sum(res / (x - pol)) for x in w]))
    ok &= check(relative(sig[0], sig[1]) < CROSS_PATH_RTOL,
                'H2 1.0 A: PSD2 with the soft mode at its harmonic level = PSD2@TDHF (pole sums)',
                f'{relative(sig[0], sig[1]):.1e}')
    ref = -calc_qp_energy(mf, selfenergy='PSD2', polarizability='TDHF', mode='casida', df=False,
                          eta=0.0, state='homo')
    got = -soft_mode_qp(mf, 'PSD2', model=None)['eps'][0] * HARTREE_TO_EV
    ok &= check(abs(got - ref) < DEFAULT_ROUTE_TOL, "model=None: calc_qp_energy's PSD2@TDHF HOMO",
                f'{got:.8f} vs {ref:.8f} eV')
    return ok


def check_instability():
    ok = True
    om2 = [SM.soft_split(*SM.triplet_tdhf_matrices(h2(R)))['omega2'][0] for R in (1.20, 1.212)]
    ok &= check(om2[0] > 0 > om2[1], 'H2: Omega_T^2 changes sign between 1.20 and 1.212 A',
                f'{om2[0]:.2e}, {om2[1]:.2e} Ha^2')
    mf = h2(1.212)
    try:
        CasidaSolver(*triplet_casida(mf)).solve()
        refused = False
    except CasidaInstability:
        refused = True
    rec = SM.soft_mode_triplets(mf, '1d')
    exc = rec['soft'][0]['excitation'] * HARTREE_TO_EV
    ok &= check(refused and np.isfinite(exc) and exc > 0,
                'H2 1.212 A: the house triplet Casida refuses, the collective triplet is finite',
                f'{exc:.3f} eV')
    return ok


def check_fci():
    ok = True
    for R in H2_DISTANCES:
        mf = h2(R)
        gap, _ = fci_h2(mf)
        rec = SM.soft_mode_triplets(mf, '1d')
        e1 = rec['soft'][0]['excitation']
        tda = rec['omega_tda_soft']
        miss = (e1 - gap) * HARTREE_TO_EV
        line = f'1D {e1 * HARTREE_TO_EV:.3f}, TDA {tda * HARTREE_TO_EV:.3f}, FCI {gap * HARTREE_TO_EV:.3f} eV'
        ok &= check(abs(miss) < FCI_TRIPLET_TOL, f'H2 {R} A: collective 1D triplet within '
                    f'{FCI_TRIPLET_TOL} eV of FCI', line)
        if R >= 1.0:
            ok &= check(abs(e1 - gap) < abs(tda - gap) or tda < 0, f'H2 {R} A: closer than TDA')
    return ok


def check_psd_through():
    ok = True
    ips = {}
    for R in (0.75, 1.0, 1.19, 1.212):
        mf = h2(R)
        _, ip_fci = fci_h2(mf)
        soft = soft_mode_qp(mf, 'PSD2', model='1d')
        ip_soft, z_soft = -soft['eps'][0], soft['Z'][0]
        try:
            plain = soft_mode_qp(mf, 'PSD2', model=None)
            ip_plain, z_plain = -plain['eps'][0], plain['Z'][0]
        except CasidaInstability:
            ip_plain = z_plain = None
        ips[R] = ip_soft
        ok &= check(z_soft > Z_MIN_SOFT, f'H2 {R} A: soft-mode PSD2 has a quasiparticle',
                    f'Z = {z_soft:.2f}')
        if R == 1.19:
            ok &= check(z_plain is not None and z_plain < 0.5,
                        'H2 1.19 A: PSD2@TDHF has no quasiparticle', f'Z = {z_plain:.2f}')
        if R == 1.212:
            ok &= check(ip_plain is None, 'H2 1.212 A: PSD2@TDHF is refused (CasidaInstability)')
        else:
            ok &= check(abs(ip_soft - ip_fci) < abs(ip_plain - ip_fci),
                        f'H2 {R} A: soft-mode PSD2 IP closer to FCI than PSD2@TDHF',
                        f'{ip_soft * HARTREE_TO_EV:.3f} / {ip_plain * HARTREE_TO_EV:.3f} / FCI '
                        f'{ip_fci * HARTREE_TO_EV:.3f} eV')
    moved = abs(ips[1.212] - ips[1.19]) * HARTREE_TO_EV
    ok &= check(moved < ACROSS_TOL, 'soft-mode PSD2 IP continuous across R_CF', f'{moved:.3f} eV')
    return ok


def check_molecules():
    ok = True
    flags = {}
    for name in SOFT_MOLECULES + ('H2O',):
        mf = rhf(GEOMETRIES[name])
        rec = SM.soft_mode_triplets(mf, '1d')
        flags[name] = [f['flag'] for f in rec['numerical_flags']]
        eom = EOM_CCSD[name][0]
        e1, tda = rec['soft'][0]['excitation'], rec['omega_tda_soft']
        miss = (e1 - eom) * HARTREE_TO_EV
        line = (f'1D {e1 * HARTREE_TO_EV:.3f}, TDA {tda * HARTREE_TO_EV:.3f}, '
                f'EOM-CCSD {eom * HARTREE_TO_EV:.3f} eV')
        if name == 'H2O':
            ok &= check(abs(miss) < EOM_STIFF_TOL, 'H2O (stiff control): collective 1D triplet '
                        f'within {EOM_STIFF_TOL} eV of EOM-CCSD', line)
        else:
            ok &= check(abs(miss) < EOM_TRIPLET_TOL and (abs(e1 - eom) < abs(tda - eom) or tda < 0),
                        f'{name}: collective 1D triplet within {EOM_TRIPLET_TOL} eV of EOM-CCSD '
                        'and closer than TDA', line)
    ok &= check('soft_modes_coupled' in flags['O3'] and not flags['N2'],
                "O3's three soft modes are flagged 'soft_modes_coupled', N2 raises nothing",
                f"O3 {flags['O3']}")
    return ok


def check_refusals():
    ok = True
    mol = gto.M(atom='H 0 0 0; H 0 0 1.0', basis='cc-pvdz', verbose=0)
    uhf = scf.UHF(mol).run()
    rks = scf.RKS(mol).run(xc='pbe')
    dfr = scf.RHF(mol).density_fit().run()
    raw = scf.RHF(mol)
    cases = [('UHF', lambda: SM.soft_mode_triplets(uhf), NotImplementedError),
             ('RKS', lambda: SM.soft_mode_triplets(rks), ValueError),
             ('density fitting', lambda: SM.soft_mode_triplets(dfr), ValueError),
             ('unconverged', lambda: SM.soft_mode_triplets(raw), ValueError),
             ("model '2d'", lambda: SM.soft_mode_triplets(h2(1.0), '2d'), ValueError),
             ("boundary 'open'", lambda: SM.collective_levels(1.0, np.linspace(0, 1, 5),
                                                             np.zeros(5), np.zeros(5), 1.0, '1d',
                                                             'open'), ValueError),
             ("selfenergy 'GW'", lambda: soft_mode_qp(h2(1.0), 'GW'), ValueError),
             ('A - B indefinite', lambda: SM.soft_split(np.diag([1.0, -0.5]), np.zeros((2, 2))),
              CasidaInstability)]
    for label, call, kind in cases:
        try:
            call()
            raised = None
        except Exception as exc:   # noqa: BLE001 -- the class is what is checked
            raised = exc
        ok &= check(isinstance(raised, kind), f'refused: {label}', type(raised).__name__)
    return ok


def check_edge_flag():
    saved = SM.SOFT_MODE_EDGE_WEIGHT_MAX
    SM.SOFT_MODE_EDGE_WEIGHT_MAX = 0.0
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            rec = SM.soft_mode_triplets(h2(1.0), '1d')
    finally:
        SM.SOFT_MODE_EDGE_WEIGHT_MAX = saved
    names = [f['flag'] for f in rec['numerical_flags']]
    warned = any(issubclass(w.category, SM.SoftModeEdgeWeight) for w in caught)
    return check(names == ['soft_mode_edge_weight'] and warned,
                 "an edge-weight bar of zero raises 'soft_mode_edge_weight'", f'{names}')


def run():
    warnings.simplefilter('ignore')
    ok = True
    for title, fn in (('1. triplet matrices', check_matrices), ('2. the split is exact', check_split_exact),
                      ('3. the RHF->UHF path', check_path), ('4. collective solvers', check_solvers),
                      ('5. PSD2 identities', check_psd_identity),
                      ('6. through the instability', check_instability),
                      ('7. H2 triplet against FCI', check_fci),
                      ('8. PSD2 through the instability', check_psd_through),
                      ('9. molecules against EOM-CCSD', check_molecules),
                      ('10. refusals', check_refusals), ('11. flags', check_edge_flag)):
        print(f'\n-- {title}')
        ok &= fn()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_soft_mode_triplet():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
