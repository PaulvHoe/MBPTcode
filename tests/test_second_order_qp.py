"""The quasiparticle sequences of the BSE(2) paper, as its AMS runs did them
(`BSE.second_order_qp`).

1. AMS's graphical rule on model self-energies, against a line-for-line
   transcription of moGWmodule selfEnergy (its grid, its bisection, its
   1e-10 Ha central-difference Z and its choice): the roots of the scan are
   AMS's bracket midpoints with AMS's weights, and the root kept is AMS's,
   converged inside the bracket AMS left it in -- also where two weights tie
   to 1e-5 and the converged weights would choose the other root. The largest
   weight is kept, Z > 1 included; a window with no root of positive weight
   (none, or only Z <= 0) keeps the anchor; the window's tier follows
   |a - mu|, not |a|; a root pair inside one cell of the grid is not seen; a
   lone root and its Z are the exact pole sum's; the broadened pole sum and
   its slope are the closed forms.
2. The cycles on H2O/6-31G against MBPTcode's own routes: the broadened sum of
   the Sigma_TDHF@L_TDHF cycle's poles is SelfEnergySolver's evaluator and the
   G0F2 cycle's is `sigma2`, slope included; every quasiparticle (Z > 0.5) of
   G0W0@HF, Sigma_TDHF@L_TDHF (full and TDA) and Sigma_BSE@L_BSE (full) is
   `calc_qp_energy`'s, G on the cycle before (`shifted_mean_field`) and the
   equation anchored on HF; under the TDA Sigma_BSE@L_BSE screens with the
   TDA-RPA W0 where `calc_qp_energy` keeps the full one, and moves the HOMO by
   tens of meV; and the G0F2 cycle is `g0f2_energies` wherever AMS's grid
   brackets that routine's root, the orbitals where a pole of Sigma(2) shares
   the root's cell (and hides it from AMS) aside.
3. H2O/aug-cc-pVTZ at the paper's geometry against the paper's AMS runs: in
   every cycle of the HF, GW and GF2 sequences, full and TDA, HOMO-2..LUMO
   keep AMS's root, within 5 meV in the shift eps_QP - eps_HF and 0.002 in
   Z, and the roots of 0 < Z < 1 AMS printed for orbitals 3-12 (satellites
   included) are as many as the scan here finds. The chained TDA
   Sigma_BSE@L_BSE is held to 12 meV and its HOMO-1 and HOMO, one AMS
   satellite each, are left out of the count: fed AMS's own first-cycle
   shifts its HOMO-2..HOMO land within 3 meV, so the rest comes from the
   first cycle's higher levels, whose HF energies already differ between the
   codes (BAND's fit and grid). The full-RPA W0 in the TDA Sigma_BSE@L_BSE
   misses by 40-83 meV, far outside these bars.
4. second_order_bse(mf, eps=<sequence>) is second_order_bse on that
   sequence's energies, carries its record, and collects its named flags;
   the refusals.

Run: python tests/test_second_order_qp.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import dft, gto, scf

from src.Base.numerics.constants import HARTREE_TO_EV
from src.Base.pyscf.pyscf_interface import get_density_fitting_coefficients
from src.SingleReference.ADC.second_order import g0f2_energies, sigma2
from src.SingleReference.BSE import second_order_qp as SQ
from src.SingleReference.BSE.second_order_bse import second_order_bse
from src.SingleReference.GW.evGW import shifted_mean_field
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.Solvers.qp_equation import solve_qp_equation_poles
from tests.pins import CROSS_PATH_RTOL

SMALL_H2O = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
#: The paper's water (the AMS inputs, Angstrom), aug-cc-pVTZ, all electrons.
PAPER_H2O = 'O 0 0 0; H 0.9591 0 0; H -0.2373 0.9293 0'
#: AMS's own Hartree in eV (1/0.0367493245), which its printed roots are in.
AMS_EV = 1.0 / 0.0367493245
#: AMS's HF energies of orbitals 3-6 (HOMO-2..LUMO), Ha.
AMS_HF = [-0.716447, -0.584953, -0.510269, 0.029353]
#: Per AMS run and iteration: the root of largest weight of orbitals 3-6
#: (eV, Z) and how many roots of 0 < Z < 1 it printed for orbitals 3-12
#: (BSE2_port: the wd*_H2O outputs of rerun_2026-10).
AMS_RUNS = {
    ('Sigma_TDHF@L_TDHF', False, 0): (
        [(-18.809432, 0.912962), (-14.965028, 0.914059), (-12.806065, 0.914083),
         (0.640911, 0.993452)], [1] * 10),
    ('Sigma_TDHF@L_TDHF', True, 0): (
        [(-18.225349, 0.898053), (-14.136146, 0.894617), (-11.899878, 0.893028),
         (0.612995, 0.992717)], [2, 2] + [1] * 8),
    ('Sigma_BSE@L_BSE', False, 0): (
        [(-19.127242, 0.939139), (-15.104606, 0.932125), (-12.883370, 0.930258),
         (0.681711, 0.996660)], [1] * 10),
    ('Sigma_BSE@L_BSE', False, 1): (
        [(-18.805137, 0.902066), (-14.756734, 0.894476), (-12.503287, 0.891809),
         (0.649500, 0.993387)], [2, 3] + [1] * 8),
    ('Sigma_BSE@L_BSE', True, 0): (
        [(-18.942568, 0.926386), (-14.700902, 0.917406), (-12.402361, 0.915046),
         (0.666679, 0.996179)], [1] * 10),
    ('Sigma_BSE@L_BSE', True, 1): (
        [(-18.178107, 0.880528), (-13.751768, 0.866365), (-11.363037, 0.861270),
         (0.621585, 0.992407)], [3, None, None] + [1] * 7),
    ('G0F2', False, 0): (
        [(-18.139455, 0.899096), (-13.835515, 0.884184), (-11.554153, 0.880163),
         (0.647353, 0.995209)], [1] * 10),
}
#: Ha. The shift eps_QP - eps_HF, AMS against here, and the bar of the chained
#: TDA Sigma_BSE@L_BSE (module docstring).
SHIFT_TOL = 5e-3 / HARTREE_TO_EV
CHAINED_TDA_TOL = 12e-3 / HARTREE_TO_EV
Z_TOL = 2e-3
#: eV: calc_qp_energy bisects to a 1e-8 Ha bracket, the cycles here converge
#: their roots: half that bracket, with room.
ROUTE_TOL = 3e-7


def check(ok, label, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {label}{(': ' + detail) if detail else ''}")
    return bool(ok)


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def rhf(atom, basis, auxbasis=False):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    if auxbasis is not False:
        mf = mf.density_fit(auxbasis=auxbasis)
    return mf.run()


def ams_self_energy(res, pol, a, mu, eta):
    """moGWmodule selfEnergy for one orbital, line for line on a model pole
    sum (eigKS = a - mu, both frames shifted by mu): the window tier, a
    100-point grid, each strict sign change bisected from the right end to
    eps = 1e-4 or 30 steps, the midpoint's Z by a 1e-10 central difference,
    and the root kept whose Z exceeds the best so far, from 0. Returns
    (kept root, its Z or NaN, [(midpoint, Z)], last bracket width)."""
    sigma = lambda w: float(SQ.pole_sum(res, pol - mu, w, eta)[0][0])
    eig_ks = a - mu
    omega_min = -0.5
    if abs(eig_ks) > 1.5:
        omega_min = -1.0
    if abs(eig_ks) > 4.0:
        omega_min = -2.0
    grid = np.linspace(omega_min + eig_ks, -omega_min + eig_ks, 100)
    shift = [w - eig_ks - sigma(w) for w in grid]
    kept, z_old, roots, width = eig_ks, 0.0, [], None
    for i in range(1, grid.size):
        if shift[i] * shift[i - 1] < 0.0:
            ra, rb = grid[i], grid[i - 1]
            for _ in range(30):
                c = (rb + ra) / 2.0
                if (ra - eig_ks - sigma(ra)) * (c - eig_ks - sigma(c)) <= 0.0:
                    rb = c
                else:
                    ra = c
                if abs(rb - ra) <= 1e-4:
                    break
            q = (rb + ra) / 2.0
            derivative = (sigma(q + 1e-10) - sigma(q - 1e-10)) / 2e-10
            z = 1.0 / (1.0 - derivative)
            roots.append((q + mu, z))
            width = abs(rb - ra)
            if z > z_old:
                kept, z_old = q, z
    return kept + mu, (z_old if z_old > 0.0 else np.nan), roots, width


def check_ams_rule():
    ok = True
    eta = 1e-3
    rng = np.random.default_rng(11)
    models = {
        # -0.005/(w - 0.2) + 0.02/(w + 0.4): a root of Z < 1 and one of Z > 1
        'a weight above 1 beats one below': (np.array([-0.005, 0.02]),
                                             np.array([0.2, -0.4]), 0.0, 0.0),
        # two midpoint weights 1.1e-5 apart (0.480799 against 0.480788) whose
        # converged weights rank the other way round
        'a near tie, decided on the midpoints': (
            np.array([0.05, 0.0016628, 0.0017554, 0.0010784]),
            np.array([-1.6, -0.459303, -0.453531, -0.454187]), -0.5, 0.0),
        'a signed model': (rng.normal(size=60) * 2e-3, rng.uniform(-1.5, 1.5, size=60),
                           0.1, -0.2),
        'a deep level, tier by |a - mu|': (np.abs(rng.normal(size=60)) * 2e-3,
                                           rng.uniform(-3.5, -1.0, size=60), -1.9, -0.3),
    }
    for label, (res, pol, a, mu) in models.items():
        mine = SQ.ams_roots(res, pol, a, mu, eta)
        w, z = SQ.ams_qp_root(res, pol, a, mu, eta)
        kept, z_kept, roots, width = ams_self_energy(res, pol, a, mu, eta)
        same = (len(mine) == len(roots) and np.allclose(
            [m[0] for m in mine], [r[0] for r in roots], rtol=0, atol=1e-12)
            and np.allclose([m[1] for m in mine], [r[1] for r in roots], rtol=1e-6))
        value, slope = SQ.pole_sum(res, pol, w, eta)
        ok &= check(same and abs(w - kept) <= 0.5 * width and abs(w - a - value[0]) < 1e-12,
                    f"{label}: AMS's roots, and its kept one converged in its bracket",
                    f'{len(mine)} roots; kept {w:.9f} against AMS {kept:.9f} (Z {z:.6f} / '
                    f'{z_kept:.6f})')
    # the weight at the kept root against a central difference of f
    res, pol, a, mu = models['a weight above 1 beats one below']
    w, z = SQ.ams_qp_root(res, pol, a, mu, eta)
    h = 1e-7
    f = lambda x: x - a - SQ.pole_sum(res, pol, x, eta)[0][0]
    ok &= check(z > 1.0 and abs(z * (f(w + h) - f(w - h)) / (2 * h) - 1.0) < 1e-6,
                'Z = 1/(1 - dSigma/dw) at the kept root, Z > 1 kept', f'Z = {z:.6f}')
    # no root at all (10/(w - 10) is about -1 near 0), or only one of Z < 0
    # (a strong pole at the level): the anchor stays
    w0, z0 = SQ.ams_qp_root(np.array([10.0]), np.array([10.0]), 0.3, 0.0, eta)
    roots = SQ.ams_roots(np.array([1.0]), np.array([0.0]), 0.0, 0.3, eta)
    w1, z1 = SQ.ams_qp_root(np.array([1.0]), np.array([0.0]), 0.0, 0.3, eta)
    ok &= check(w0 == 0.3 and np.isnan(z0) and len(roots) == 1 and roots[0][1] < 0.0
                and w1 == 0.0 and np.isnan(z1),
                'no root of positive weight, none or only Z < 0: the anchor stays')
    # tier by |a - mu|: 1.3 Ha from mu at a = 1.7 is the narrow tier, 1.7 is not
    ok &= check(SQ.ams_window(1.7, 0.4) == (1.2, 2.2)
                and SQ.ams_window(1.7, 0.0) == (0.7, 2.7)
                and SQ.ams_window(-5.0, 0.2) == (-7.0, -3.0),
                "AMS's window tier follows |eps_HF - mu|")
    # one positive-weight pole far from the level: one root, the exact one
    res, pol = np.array([0.03]), np.array([-1.1])
    w, z = SQ.ams_qp_root(res, pol, -0.4, 0.0, 0.0)
    w_ref, z_ref = solve_qp_equation_poles(res, pol, -0.4)
    ok &= check(abs(w - w_ref) < 1e-11 and abs(z - z_ref) < 1e-10,
                'a lone root and its Z are the exact pole sum\'s',
                f'{w:.12f} vs {w_ref:.12f}')
    # a root pair inside one cell of the grid is not seen: a pole of weight
    # 2e-5 at 0.204 Ha, 0.004 Ha above the level, pulls f through zero three
    # times within 0.01 Ha, and only the last crossing falls in another cell
    res, pol = np.array([2e-5]), np.array([0.204])

    def crossings(n):
        grid = np.linspace(*SQ.ams_window(0.2, 0.0), n)
        f = grid - 0.2 - SQ.pole_sum(res, pol, grid, eta)[0]
        return int(np.count_nonzero(f[1:] * f[:-1] < 0))
    roots = SQ.ams_roots(res, pol, 0.2, 0.0, eta)
    ok &= check(crossings(100001) == 3 and crossings(100) == len(roots) == 1,
                "only the grid's own sign changes bracket",
                f'{len(roots)} root found, {crossings(100001)} on a fine grid')
    # the broadened sum and its slope against a central difference
    rng = np.random.default_rng(3)
    res, pol = rng.normal(size=40) * 1e-3, rng.uniform(-2, 2, size=40)
    w = np.linspace(-1, 1, 7)
    h = 1e-6
    v, s = SQ.pole_sum(res, pol, w, 0.05)
    fd = (SQ.pole_sum(res, pol, w + h, 0.05)[0] - SQ.pole_sum(res, pol, w - h, 0.05)[0]) / (2 * h)
    direct = np.array([np.sum(res * (x - pol) / ((x - pol) ** 2 + 0.05 ** 2)) for x in w])
    ok &= check(np.allclose(v, direct, rtol=CROSS_PATH_RTOL, atol=0)
                and np.allclose(s, fd, rtol=1e-6, atol=1e-9),
                'the broadened pole sum and its slope')
    return ok


def check_against_calc_qp_energy(mf):
    ok = True
    nocc = mf.mol.nelectron // 2
    hf = np.asarray(mf.mo_energy)
    states = list(range(hf.size))
    B = np.asarray(get_density_fitting_coefficients(mf.mol, mf))

    def route(m, selfenergy, polarizability, tda, **kw):
        r = quiet(calc_qp_energy, m, selfenergy=selfenergy, polarizability=polarizability,
                  state=states, tda=tda, **kw)
        return np.array([r[p][selfenergy] for p in states])

    # each cycle's broadened pole sum is the evaluator's Sigma
    poles_of, eta = SQ.cycle_self_energy('Sigma_TDHF@L_TDHF', hf, B, nocc)
    sigma = SQ.SelfEnergySolver(hf, df_coeff=B, spin_mode='restricted', eta=eta)
    A, Bm = SQ.LinearResponseSolver(hf, coeff_df=B).build_casida_matrices(nocc, lBSE=True)
    om, X, Y = (np.asarray(v) for v in SQ.CasidaSolver(A, Bm).solve())
    w = np.linspace(-1.0, 1.0, 9)
    worst = 0.0
    for p in (0, nocc - 1, nocc, hf.size - 1):
        ev = sigma.self_energy_evaluator(
            p, nocc, om, sigma.get_chi_a(nocc, X, Y, p_state=p),
            sigma.get_chi_b_vertex(nocc, X, Y, eri_w=None, p_state=p),
            vertex_mode='GWGammaInf')
        mine = SQ.pole_sum(*poles_of(p), w, eta)[0]
        worst = max(worst, np.max(np.abs(mine - ev(w)) / np.abs(ev(w)).max()))
    ok &= check(worst < CROSS_PATH_RTOL, "the cycle's Sigma is SelfEnergySolver's evaluator",
                f'{worst:.1e} relative')

    cases = []
    for tda in (False, True):
        g = quiet(SQ.qp_sequence, mf, 'Sigma_BSE@L_BSE', tda=tda, coeff_df=B)
        t = quiet(SQ.qp_sequence, mf, 'Sigma_TDHF@L_TDHF', tda=tda, coeff_df=B)
        cases.append((f'G0W0@HF {"TDA" if tda else "full"}', g['cycles'][0],
                      route(mf, 'GW', 'RPA', tda)))
        cases.append((f'Sigma_TDHF@L_TDHF {"TDA" if tda else "full"}', t['cycles'][0],
                      route(mf, 'GWGammaInf', 'TDHF', tda)))
        view = shifted_mean_field(mf, g['cycles'][0]['eps'])
        bse = route(view, 'GWGammaInf', 'BSE', tda, eps_anchor=hf)
        if tda:
            homo = (g['eps'][nocc - 1] * HARTREE_TO_EV - bse[nocc - 1]) * 1e3
            ok &= check(abs(homo) > 10.0,
                        'TDA Sigma_BSE@L_BSE screens with the TDA-RPA W0, not the full one',
                        f'HOMO {homo:+.1f} meV from the full-W0 route')
        else:
            cases.append(('Sigma_BSE@L_BSE full', g['cycles'][1], bse))
    for label, cycle, ref in cases:
        qp = cycle['Z'] > 0.5
        dev = np.max(np.abs(cycle['eps'][qp] * HARTREE_TO_EV - ref[qp]))
        ok &= check(qp.sum() >= hf.size - 2 and dev < ROUTE_TOL,
                    f'{label} equals calc_qp_energy', f'{qp.sum()} quasiparticles, '
                    f'max {dev:.1e} eV')

    # G0F2: the unbroadened pole sum is sigma2's, slope included, and the cycle
    # is g0f2_energies (the largest weight over the whole axis) wherever AMS's
    # grid brackets that root, i.e. where no pole of Sigma(2) shares its cell
    poles_of, eta = SQ.cycle_self_energy('G0F2', hf, B, nocc)
    worst = 0.0
    for p in (0, nocc - 1, nocc, hf.size - 1):
        for x in (-0.83, 0.37, 1.91):
            mine = SQ.pole_sum(*poles_of(p), x, eta)
            ref = sigma2(hf, nocc, p, x, B_aa=B)
            worst = max(worst, abs(mine[0][0] - ref[0]) / abs(ref[0]),
                        abs(mine[1][0] - ref[1]) / abs(ref[1]))
    ok &= check(eta == 0.0 and worst < CROSS_PATH_RTOL,
                "the G0F2 cycle's unbroadened Sigma(2) and its slope are sigma2's",
                f'{worst:.1e} relative')
    seq = quiet(SQ.qp_sequence, mf, 'G0F2', coeff_df=B)
    whole, _ = g0f2_energies(hf, nocc, B_aa=B)
    mu = SQ.fermi_level(hf, nocc)
    hidden = []
    for p in range(hf.size):
        lo, hi = SQ.ams_window(hf[p], mu)
        grid = np.linspace(lo, hi, 100)
        k = np.searchsorted(grid, whole[p])
        _, poles = poles_of(p)
        if 0 < k < grid.size and np.any((poles > grid[k - 1]) & (poles < grid[k])):
            hidden.append(p)
    differ = [p for p in range(hf.size) if abs(seq['eps'][p] - whole[p]) * HARTREE_TO_EV > 1e-8]
    ok &= check(len(differ) <= 2 and set(differ) <= set(hidden),
                "the G0F2 cycle is g0f2_energies wherever AMS's grid brackets its root",
                f'differ at {differ}, each beside a pole of Sigma(2) in its cell '
                f'(such cells: {hidden})')
    return ok


def check_against_ams(mf):
    ok = True
    nocc = mf.mol.nelectron // 2
    hf = np.asarray(mf.mo_energy)
    B = np.asarray(get_density_fitting_coefficients(mf.mol, mf))
    done = {}
    for (name, tda, k), (kept, counts) in AMS_RUNS.items():
        if (name, tda) not in done:
            done[name, tda] = quiet(SQ.qp_sequence, mf, name, tda=tda, coeff_df=B)
        seq = done[name, tda]
        cycle = seq['cycles'][k]
        prev = hf if k == 0 else seq['cycles'][k - 1]['eps']
        label = f'{cycle["name"]} {"TDA" if tda else "full"}'
        shift = (cycle['eps'][2:6] - hf[2:6]) - (
            np.array([e for e, _ in kept]) / AMS_EV - np.array(AMS_HF))
        dz = cycle['Z'][2:6] - np.array([z for _, z in kept])
        tol = CHAINED_TDA_TOL if (name == 'Sigma_BSE@L_BSE' and tda and k == 1) else SHIFT_TOL
        ok &= check(np.max(np.abs(shift)) < tol and np.max(np.abs(dz)) < Z_TOL,
                    f"{label}: HOMO-2..LUMO keep AMS's root",
                    'shift - AMS ' + ' '.join(f'{x * HARTREE_TO_EV * 1e3:+.1f}' for x in shift)
                    + f' meV, |dZ| <= {np.max(np.abs(dz)):.4f}')
        poles_of, eta = SQ.cycle_self_energy(cycle['name'], prev, B, nocc, tda=tda)
        mu = SQ.fermi_level(prev, nocc)
        mine = [sum(0.0 < z < 1.0 for _, z in SQ.ams_roots(*poles_of(p), hf[p], mu, eta))
                for p in range(2, 12)]
        same = all(c is None or c == m for c, m in zip(counts, mine))
        ok &= check(same, f'{label}: the roots AMS printed for orbitals 3-12',
                    f'AMS {counts}, here {mine}')
    return ok


def check_sequence_into_bse(mf):
    ok = True
    for name, kernel, tda in (('Sigma_TDHF@L_TDHF', 'v', True),
                              ('Sigma_BSE@L_BSE', '2sosex', False)):
        named = quiet(second_order_bse, mf, eps=name, kernel=kernel, tda=tda,
                      dynamical_roots=3)
        seq = quiet(SQ.qp_sequence, mf, name, tda=tda)
        given = quiet(second_order_bse, mf, eps=seq['eps'], kernel=kernel, tda=tda,
                      dynamical_roots=3)
        ok &= check(np.array_equal(named['omega'], given['omega'])
                    and np.array_equal(named['dynamical']['omega'], given['dynamical']['omega'])
                    and named['qp']['name'] == name and given['qp'] is None
                    and np.array_equal(named['eps'], seq['eps'])
                    and len(named['qp']['cycles']) == len(SQ.QP_SEQUENCES[name]),
                    f'second_order_bse(eps={name!r}) runs the sequence first')
    # a kept root of low weight raises its flag into both records
    res = quiet(second_order_bse, mf, eps='Sigma_TDHF@L_TDHF', kernel='v', tda=True,
                dynamical_roots=1)
    names = [e['flag'] for e in res['numerical_flags']]
    ok &= check('qp_sequence_low_weight' in names
                and res['qp']['numerical_flags'][0] in res['numerical_flags'],
                "the cycles' named flags reach second_order_bse's record", f'{names}')
    B = get_density_fitting_coefficients(mf.mol, mf)
    refusals = [
        (lambda: SQ.qp_sequence(mf, 'evGW'), ValueError, 'an unknown sequence'),
        (lambda: SQ.quasiparticle_cycle('G0W2', mf.mo_energy, mf.mo_energy, B,
                                        mf.mol.nelectron // 2), ValueError, 'an unknown cycle'),
        (lambda: SQ.qp_sequence(mf, 'G0F2', eta=1e-3), ValueError,
         'a broadening of the unbroadened G0F2'),
        (lambda: SQ.qp_sequence(dft.RKS(mf.mol, xc='pbe'), 'G0W0'),
         NotImplementedError, 'a Kohn-Sham reference'),
        (lambda: SQ.qp_sequence(scf.RHF(mf.mol), 'G0W0'), ValueError,
         'a reference not run'),
        (lambda: SQ.fermi_level(np.array([-0.5, 0.2, 0.1]), 2), ValueError,
         'a HOMO above the LUMO'),
    ]
    for call, exc, what in refusals:
        try:
            call()
            raised = False
        except exc:
            raised = True
        ok &= check(raised, f'refuses {what}')
    return ok


def run():
    small = rhf(SMALL_H2O, '6-31g')
    paper = rhf(PAPER_H2O, 'aug-cc-pvtz', auxbasis='aug-cc-pv5z-ri')
    ok = check_ams_rule()
    ok &= check_against_calc_qp_energy(small)
    ok &= check_against_ams(paper)
    ok &= check_sequence_into_bse(small)
    print('ALL OK' if ok else 'SOME CHECKS FAILED')
    return ok


def test_second_order_qp():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
