"""The relaxed path and the generator-coordinate model of the soft-mode split on the larger cases,
and BF's PSD quasiparticles on their levels (`LinearResponse.soft_mode`, `GW.soft_mode_qp`; heavy,
run on a cluster).

Three variants side by side (tests/test_soft_mode_triplet.py holds each on the small systems):

    straight   soft_mode_triplets(mf, '1d'): the straight RHF->UHF path, P^2 / 2M (the default)
    relaxed    soft_mode_triplets(mf, '1d', 'relaxed'): the relaxed path with its inertia M(Q)
    gcm        soft_mode_triplets(mf, 'gcm'): the generator-coordinate model on the straight path

Triplet rows, cc-pVDZ at the geometries of tests/soft_mode_references.py: the first collective level
of the lowest soft mode against the references recorded there (`reference_ev`): EOM-CCSD for
butadiene and benzene (EOM_CCSD) and for hexatriene and octatetraene (EOM_CCSD_ADA_EV), spin-flip
EOM-CCSD for O3 (SPIN_FLIP_EV), ethylene twisted by 0 to 70 degrees in steps of 5 (spin-flip
EOM-CCSD at 45, 60 and 70 degrees, where it is recorded, EOM-CCSD at the others) and its C=C
stretched from 1.22 to 1.46 A through the instability at 1.3345 A (EOM-CCSD: EOM_CCSD to 1.30 A,
EOM_CCSD_ADA_EV beyond), 38 rows. BF rows, aug-cc-pVDZ and aug-cc-pVTZ at the QUEST geometry:
the HOMO of PSD2, PSD4, PSD7 and PSD9 (`GW.soft_mode_qp`) with the plain TDHF triplets
(model=None, the house route's spectrum) and with each variant, against CIPSI (11.054 and 11.149
eV, frozen core), with G0W0@HF and Sigma_TDHF@L_TDHF (`GW.qp_energy.calc_qp_energy`, Casida route,
exact integrals, its default broadening) as controls. The quasiparticle equations are the house
root finder's, and a plain TDHF spectrum past the instability is refused as the house route
refuses it (printed, not held).

Held, each a claim that stands before the run:

1. Every variant returns a finite first level on every row (the triplet rows, and BF's lowest soft
   mode in both bases), and every relaxed path is followed through its antipode as a constrained
   minimum: none is frozen or unconverged or takes a sample from a fresh start, the lowest
   curvature its probe finds across the constraint plane is not below
   -SOFT_MODE_RELAXED_SADDLE_BAR, and its loop closes (the alpha determinants at -+Q* overlap to
   1e-3 of 1, |D(Q*)| < 1e-3), so that it raises no 'soft_mode_path_*' flag. On the rows a
   workstation can run (O3, the twist and the stretch, 48 basis functions at most) and on BF in
   cc-pVDZ and aug-cc-pVDZ all of this holds, with the overlap within 2.4e-10 of 1 and |D(Q*)| below
   5.7e-7, and no path holds any mode but the soft ones (H2CO and N2 hold theirs in
   tests/test_soft_mode_triplet.py). The polyenes and BF/aug-cc-pVTZ were not run here; benzene,
   whose lowest triplet comes from its degenerate e1g/e2u orbitals, is the likeliest to need held
   modes or to fail this.
2. BF in both bases, a stable mode (two degenerate soft modes; Omega_s = 1.52 eV in cc-pVDZ, 1.39
   in aug-cc-pVDZ): the relaxed path relaxes and stays a minimum through Q*, the lowest eigenvalue
   of its dense constrained Hessian at the relaxed sample nearest Q* positive
   (`lowest_constrained_curvature`, central differences of the analytic gradient across the plane:
   +0.090 and +0.052 Ha in cc-pVDZ and aug-cc-pVDZ) and
   V_relaxed below V_straight at Q = L / 2 by more than MID_PATH_DROP (3.8 and 3.9 mHa); and it
   keeps the TDHF harmonic limit, V''(0) = mu to 1e-5 relative (Richardson from fresh constrained
   minima at Q = HARMONIC_STEP and twice it, minimised to SOFT_MODE_RELAXED_TANGENT_GRADIENT) and
   M(0) = M to 1e-8 relative (read only on an unfrozen path with more than SOFT_MODE_RELAXED_TRUST
   relaxed samples; 1.3e-15 and 2.4e-15 measured). The harmonic part holds the RHF and the split
   rather than the relaxation: there the straight point Q d is the constrained minimum to second
   order ((A + B) d = Omega_s^2 M e), so a wrong conjugate e or a wrong A + B fails it and a broken
   minimisation does not. Its scatter is the RHF's: over five RHF initial guesses (conv_tol 1e-10)
   V''(0) / mu - 1 is -2.1e-6 to +2.1e-6 in cc-pVDZ and aug-cc-pVDZ at Q = 1e-2, against -2.5e-6 to
   +4.1e-6 at the fast test's 4e-3 (BF/cc-pVTZ -1.8e-6 to +1.8e-6 there), and it follows the RHF
   solution, not the step or the minimiser's bar; the Richardson's O(Q^4) remainder is 3e-8 at
   1e-2.
3. Over the triplet rows the relaxed path's mean absolute error against the references lies below
   the straight path's. The relaxed path lowers every level (V_relaxed <= V_straight and
   M(Q) >= M), by 0.02 to 0.38 eV on the 22 small systems measured (the tweak study's 14 and eight
   ethylene points), so it helps where the straight path is too high and hurts where it is too
   low: the mean absolute error goes from 0.505 to 0.372 eV on the 14 and from 0.511 to 0.289 eV on
   the eight, while N2 and H2 at 0.75 A, already low, go lower, as benzene, 0.06 eV low on the
   straight path (tests/test_soft_mode_triplet_slow.py), will.

Printed for the record, not held: every level and its error, the GCM's (with its ranks, <S^2> of
the even ground state, coupling c_1, sigma_2 / sigma_1 and the levels over its overlap cuts), the
relaxed paths' antipode Q* / L, M(Q*) / M, closure overlap, convergence, fresh starts, held modes,
lowest curvature, freeze and energy calls, every variant's flags and seconds, the mean signed, mean
absolute and largest errors by group, and BF's IPs and pole strengths Z.

Memory per row, estimated (nothing above 60 basis functions runs on a workstation): about twelve
dense n_ov x n_ov float64 matrices at once (n_ov = n_occ n_virt: the TDHF triplet A and B, the split
of (A - B)^1/2 (A + B) (A - B)^1/2, the regular Casida vectors, and the relaxed path's basis Z and
Cholesky factors; a hold round adds the split's matrices once more), plus pyscf's in-core AO
integrals (nao^4 bytes, held by the RHF and by the UHF of a path): octatetraene (nao 162, n_ov 3857,
119 MB a matrix) about 3 GB, hexatriene (124, 2244) 1 GB, benzene (114, 1953) 0.7 GB, butadiene
(86, 1065) 0.2 GB, O3 and the ethylene rows under 0.1 GB. BF is set by the n_mo n_ov^2 float64
intermediates of the PSD and control routes (measured on BF at 28, 46 and 60 basis functions, each
piece in a fresh process: 6.6 n_mo n_ov^2 for a PSD2 call with the plain TDHF triplets, about 10
for the relaxed PSD and the G0W0@HF control): about 3 GB at its peak in aug-cc-pVTZ (n_mo 92, n_ov
595) and 0.3 GB in aug-cc-pVDZ (46, 273).

Run: python tests/test_soft_mode_variants_full.py, or under pytest (about an hour on 16 cores,
estimated from the small systems' costs, not timed: a relaxed path takes 200 to 470 UHF energies and
gradients, 1200 and 2400 for H2CO and N2 with their hold rounds, and BF/aug-cc-pVTZ's dense
constrained Hessian about 1200, each costing as nao^4, so octatetraene's soft modes dominate).
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.numerics.constants import (HARTREE_TO_EV, SOFT_MODE_RELAXED_SADDLE_BAR,
                                         SOFT_MODE_RELAXED_TANGENT_GRADIENT,
                                         SOFT_MODE_RELAXED_TRUST)
from src.SingleReference.GW.qp_energy import calc_qp_energy
from src.SingleReference.GW.self_energy import TRIPLET_VERTEX_MODES
from src.SingleReference.GW.soft_mode_qp import soft_mode_qp
from src.SingleReference.LinearResponse import soft_mode as SM
from src.SingleReference.LinearResponse.casida import CasidaInstability
from tests.soft_mode_references import (CIPSI_IP_EV, EOM_CCSD, EOM_CCSD_ADA_EV, G0W0_HF_IP_EV,
                                        QUEST_GEOMETRIES, SPIN_FLIP_EV, geometry, rhf)
from tests.test_soft_mode_triplet import lowest_constrained_curvature

#: Relative bar of V''(0) = mu on BF's relaxed path (Richardson from fresh constrained minima at
#: HARMONIC_STEP and twice it): five times the scatter over five RHF initial guesses (minao, atom,
#: huckel, 1e, vsap; conv_tol 1e-10), -2.1e-6 to +2.1e-6 in cc-pVDZ and aug-cc-pVDZ at Q = 1e-2.
#: The scatter follows the RHF solution, not the step or the minimiser's bar (the same guess gives
#: the same offset at Q = 1e-2 and 2e-2, minimised to 1e-6 or 1e-9), so a larger Q or a tighter
#: minimum does not shrink it; at 4e-3 (the fast test's step) it reached +4.1e-6 in aug-cc-pVDZ.
HARMONIC_CURVATURE_RTOL = 1e-5
#: Relative bar of the path inertia M(0) = M: the error of the spline's tangent at Q = 0 lies on
#: e^T kappa' = 0, where it moves M at second order only. 9e-16 (H2) and 3e-13 (ethylene twisted by
#: 60 degrees) measured in tests/test_soft_mode_triplet.py, whose bar is 1e-10, and 1.3e-15 and
#: 2.4e-15 on BF/cc-pVDZ and aug-cc-pVDZ; looser here for aug-cc-pVTZ, which was not run.
HARMONIC_INERTIA_RTOL = 1e-8
#: Q of the Richardson curvature (h and 2h): V(h) is 7e-7 Ha there for BF (1.1e-7 at the fast
#: test's 4e-3), and the O(h^4) remainder 3e-8 relative (8e-6 at 4e-2).
HARMONIC_STEP = 1e-2
#: Ha. How far the relaxation lowers BF's V at Q = L / 2 at least: 3.8 and 3.9 mHa measured in
#: cc-pVDZ and aug-cc-pVDZ.
MID_PATH_DROP = 1e-3
#: The closure of a relaxed loop: 1 minus the overlap of the alpha determinants at -+Q*, and
#: |D(Q*)|; 2.4e-10 and 5.7e-7 at most on the rows run on a workstation and BF in cc-pVDZ and
#: aug-cc-pVDZ.
CLOSURE_TOL = 1e-3

#: The variants by name: (model, path) of `soft_mode_triplets` and `soft_mode_qp`.
VARIANTS = {'straight': ('1d', 'straight'), 'relaxed': ('1d', 'relaxed'),
            'gcm': ('gcm', 'straight')}
#: The PSD columns of the BF rows: the plain TDHF triplets, then the variants.
PSD_COLUMNS = (('TDHF', (None, 'straight')),) + tuple(VARIANTS.items())

POLYENES = ('butadiene', 'benzene', 'hexatriene', 'octatetraene')
TWIST = ('ethylene',) + tuple(f'ethylene@{t}' for t in range(5, 75, 5))
STRETCH = tuple(sorted(k for k in (*EOM_CCSD, *EOM_CCSD_ADA_EV) if k.startswith('ethylene~')))
TRIPLET_ROWS = POLYENES + ('O3',) + TWIST + STRETCH
BF_BASES = ('aug-cc-pvdz', 'aug-cc-pvtz')
#: The groups of the printed summary.
GROUPS = (('polyenes, EOM-CCSD', POLYENES),
          ('O3 and ethylene at 45, 60, 70 degrees, spin-flip',
           tuple(k for k in TRIPLET_ROWS if k in SPIN_FLIP_EV)),
          ('the other twisted ethylenes, EOM-CCSD', tuple(k for k in TWIST
                                                          if k not in SPIN_FLIP_EV)),
          ('C=C stretch 1.22-1.46 A, EOM-CCSD', STRETCH))


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def reference_ev(label):
    """(reference in eV, its kind) of a triplet row: spin-flip EOM-CCSD where it is recorded, else
    EOM-CCSD."""
    if label in SPIN_FLIP_EV:
        return SPIN_FLIP_EV[label], 'SF-EOM-CCSD'
    if label in EOM_CCSD_ADA_EV:
        return EOM_CCSD_ADA_EV[label], 'EOM-CCSD'
    return EOM_CCSD[label][0] * HARTREE_TO_EV, 'EOM-CCSD'


def omega_text(omega2):
    """Omega_s (eV) from Omega_s^2 (Ha^2), with an 'i' past the instability."""
    return f'{np.sqrt(abs(omega2)) * HARTREE_TO_EV:.3f}' + ('i' if omega2 < 0 else '')


def flag_names(flags):
    return [f['flag'] if isinstance(f, dict) else f for f in flags]


def variant_text(name, soft):
    """The diagnostics of one variant's lowest soft mode, as printed."""
    if name == 'relaxed':
        frozen = soft['frozen_from']
        frozen = '-' if frozen is None else f"{frozen / soft['half_period']:.3f}"
        curvature = soft['curvature']
        curvature = '-' if curvature is None else f'{curvature:+.3f}'
        return (f"Q*/L {soft['antipode'] / soft['half_period']:.3f}, M(Q*)/M "
                f"{soft['mass_ratio_antipode']:.3f}, closure {soft['closure_overlap']:.4f}, D(Q*) "
                f"{soft['coupling_antipode']:+.1e}, converged {soft['converged']}, fresh starts "
                f"{soft['cold_restarts']} tried, {len(soft['restarted'])} taken, held modes "
                f"{soft['held_modes']}, lowest curvature {curvature}, frozen from Q/L {frozen}, "
                f"{soft['energy_calls']} energies")
    if name == 'gcm':
        sigma = soft['sigma']
        ratio = sigma[1] / sigma[0] if sigma.size > 1 else 0.0
        cuts = [v[0] * HARTREE_TO_EV for v in soft['cut_scan'].values()]
        return (f"ranks {soft['ranks']}, <S^2> {soft['spin_square'][0]:.3f}, c_1 "
                f"{soft['coupling']:.3f}, sigma_2/sigma_1 {ratio:.3f}, over the cuts "
                f"{min(cuts):.3f} to {max(cuts):.3f} eV")
    return (f"walls {soft['excitation_walls'] * HARTREE_TO_EV:.3f} eV, edge weight "
            f"{soft['edge_weight']:.1e}")


def triplet_row(label):
    """The first collective level (eV) of the lowest soft mode in every variant for one triplet row,
    with its diagnostics, printed as a block of the table."""
    mf = rhf(geometry(label))
    ref, kind = reference_ev(label)
    row = dict(label=label, ref=ref, kind=kind, nao=mf.mol.nao_nr())
    for name, (model, path) in VARIANTS.items():
        start = time.perf_counter()
        rec = SM.soft_mode_triplets(mf, model, path)
        soft = rec['soft'][0]
        row[name] = dict(level=soft['excitation'] * HARTREE_TO_EV, soft=soft, softs=rec['soft'],
                         flags=flag_names(rec['numerical_flags']),
                         seconds=time.perf_counter() - start)
        row['omega2'], row['nsoft'] = rec['omega2'][0], len(rec['soft'])
    print(f"  {label} ({row['nao']} basis functions): TDHF Omega_s {omega_text(row['omega2'])} "
          f"eV, {row['nsoft']} soft mode(s); {kind} {ref:.3f} eV", flush=True)
    for name in VARIANTS:
        v = row[name]
        print(f"    {name:9s} {v['level']:7.3f} ({v['level'] - ref:+.3f})  "
              f"{variant_text(name, v['soft'])}; flags {v['flags']}; {v['seconds']:.1f} s",
              flush=True)
    return row


def harmonic_limit(mf):
    """BF's relaxed path of the lowest soft mode, as claim 2 reads it: V''(0) / mu - 1 by
    Richardson from fresh constrained minima at HARMONIC_STEP and twice it (minimised to
    SOFT_MODE_RELAXED_TANGENT_GRADIENT), M(0) / M - 1 from `LinearResponse.soft_mode.relaxed_path`'s
    inertia at Q = 0 (None on a frozen path or one with SOFT_MODE_RELAXED_TRUST relaxed samples or
    fewer, whose M(0) is M by construction), the lowest eigenvalue (Ha) of the dense constrained
    Hessian at its relaxed sample nearest Q* (`lowest_constrained_curvature`), V_straight -
    V_relaxed (Ha) at Q = L / 2; with Omega_s^2 (Ha^2) and the number of soft modes."""
    nocc = mf.mol.nelectron // 2
    A, B = SM.triplet_tdhf_matrices(mf)
    split = SM.soft_split(A, B)
    s = split['soft'][0]
    mode = split['modes'][s]
    held = [split['modes'][t]['conjugate'] for t in split['soft'] if t != s]
    path = SM.relaxed_path(mf, A, B, mode, held)
    energy = SM._TripletEnergy(mf, nocc)
    minima = SM._ConstrainedMinima(energy, mode['direction'], mode['conjugate'], held, A + B)
    h = HARMONIC_STEP
    v1, v2 = (minima.minimum(Q, np.zeros(minima.size), SOFT_MODE_RELAXED_TANGENT_GRADIENT)[1]
              for Q in (h, 2 * h))
    curvature = (4 * (2 * v1 / h ** 2) - v2 / (2 * h ** 2)) / 3
    unfrozen = path['frozen_from'] is None and path['relaxed'] > SOFT_MODE_RELAXED_TRUST
    middle = (SM.SOFT_MODE_RELAXED_POINTS - 1) // 2
    drop = energy(path['q'][middle] * mode['direction'])[0] - path['V'][middle]
    return dict(curvature=curvature / mode['stiffness'] - 1,
                inertia=path['mass'][0] / mode['mass'] - 1 if unfrozen else None,
                lowest=lowest_constrained_curvature(mf, (nocc, A, B, mode, held), path),
                drop=drop, omega2=mode['omega2'], nsoft=len(split['soft']))


def bf_row(basis):
    """BF's HOMO (IP in eV, Z, flags) by PSD2, PSD4, PSD7 and PSD9 with the plain TDHF triplets and
    with every variant, and by the two controls; its lowest soft mode in every variant and the
    relaxed path's harmonic limit, printed as a block of the table."""
    mf = rhf(QUEST_GEOMETRIES['BF'], basis)
    key = ('BF', basis)
    out = dict(basis=basis, nao=mf.mol.nao_nr(), cipsi=CIPSI_IP_EV.get(key),
               g0w0_published=G0W0_HF_IP_EV.get(key), soft={}, softs={}, ip={})
    out['harmonic'] = harmonic_limit(mf)
    for name, (model, path) in PSD_COLUMNS:
        for se in TRIPLET_VERTEX_MODES:
            try:
                q = soft_mode_qp(mf, se, model=model, path=path)
            except CasidaInstability:
                out['ip'][se, name] = None
                continue
            out['ip'][se, name] = (-q['eps'][0] * HARTREE_TO_EV, q['Z'][0],
                                   flag_names(q['numerical_flags']))
            if q['triplets'] is not None:
                out['soft'][name] = q['triplets']['soft'][0]
                out['softs'][name] = q['triplets']['soft']
    out['levels'] = {name: soft['excitation'] * HARTREE_TO_EV
                     for name, soft in out['soft'].items()}
    for name, se, pol in (('G0W0@HF', 'GW', 'RPA'), ('Sigma_TDHF@L_TDHF', 'GWGammaInf', 'TDHF')):
        rec = {}
        e = calc_qp_energy(mf, selfenergy=se, polarizability=pol, mode='casida', df=False,
                           state='homo', out=rec)
        out[name] = (-e, flag_names(rec.get('numerical_flags', [])))
    print_bf(out)
    return out


def print_bf(out):
    """BF's block of the table."""
    ref = out['cipsi']

    def ev(x):
        return '-' if x is None else f'{x:.3f}'

    def miss(ip):
        return f'{ip - ref:+.3f}' if ref is not None and np.isfinite(ip) else '-'

    hl = out['harmonic']
    inertia = '-' if hl['inertia'] is None else f"{hl['inertia']:.1e}"
    print(f"  BF/{out['basis']} ({out['nao']} basis functions): TDHF Omega_s "
          f"{omega_text(hl['omega2'])} eV, {hl['nsoft']} soft mode(s); CIPSI {ev(ref)} (frozen "
          f"core), published G0W0@HF {ev(out['g0w0_published'])} eV", flush=True)
    print(f"    relaxed path: lowest constrained curvature near Q* {hl['lowest']:+.4f} Ha, "
          f"V_straight - V_relaxed at L / 2 {hl['drop'] * 1e3:.2f} mHa; V''(0)/mu - 1 "
          f"{hl['curvature']:.1e}, M(0)/M - 1 {inertia}")
    for name, soft in out['soft'].items():
        print(f"    {name:9s} level {out['levels'][name]:7.3f} eV  {variant_text(name, soft)}")
    print('    IP, IP - CIPSI, Z  ' + ''.join(f'{name:>24s}' for name, _ in PSD_COLUMNS))
    for se in TRIPLET_VERTEX_MODES:
        cells = []
        for name, _ in PSD_COLUMNS:
            got = out['ip'][se, name]
            cells.append('refused' if got is None
                         else f'{got[0]:7.3f} {miss(got[0]):>6s} {got[1]:5.2f}')
        print(f'    {se:18s} ' + ''.join(f'{c:>24s}' for c in cells), flush=True)
    for name in ('G0W0@HF', 'Sigma_TDHF@L_TDHF'):
        ip, flags = out[name]
        print(f'    {name:18s} {ip:7.3f} {miss(ip):>6s}  flags {flags}')
    flags = {f'{se} {name}': got[2] for (se, name), got in out['ip'].items()
             if got is not None and got[2]}
    print(f'    PSD flags: {flags}', flush=True)


def errors(rows, name):
    """Signed errors (eV) of one variant over rows."""
    return np.array([r[name]['level'] - r['ref'] for r in rows])


def print_groups(rows):
    by = {r['label']: r for r in rows}
    print(f"  {'group':46s} {'n':>3s}" + ''.join(f'{name:>24s}' for name in VARIANTS))
    for title, labels in GROUPS + (('all triplet rows', tuple(by)),):
        chosen = [by[k] for k in labels if k in by]
        if not chosen:
            continue
        cells = []
        for name in VARIANTS:
            e = errors(chosen, name)
            cells.append(f'{e.mean():+.3f} {np.abs(e).mean():.3f} {np.abs(e).max():.3f}')
        print(f'  {title:46s} {len(chosen):3d}' + ''.join(f'{c:>24s}' for c in cells))


def path_clean(soft):
    """Whether a relaxed record is a constrained minimum through its antipode: unfrozen, converged,
    no sample taken from a fresh start, no curvature below -SOFT_MODE_RELAXED_SADDLE_BAR at its
    probes, and a closed loop (CLOSURE_TOL)."""
    curvature = soft['curvature']
    return bool(soft['frozen_from'] is None and soft['converged'] and not len(soft['restarted'])
                and (curvature is None or curvature >= -SOFT_MODE_RELAXED_SADDLE_BAR)
                and 1.0 - soft['closure_overlap'] < CLOSURE_TOL
                and abs(soft['coupling_antipode']) < CLOSURE_TOL)


def check_finite(rows, bf):
    ok = True
    cases = [(r['label'], [r[name]['level'] for name in VARIANTS], r['relaxed']['softs'],
              r['relaxed']['flags']) for r in rows]
    cases += [(f"BF/{b['basis']}", [b['levels'].get(name, np.nan) for name in VARIANTS],
               b['softs'].get('relaxed'), (b['ip'].get(('PSD2', 'relaxed')) or (0, 0, []))[2])
              for b in bf]
    for label, levels, softs, flags in cases:
        clean = bool(softs) and all(path_clean(soft) for soft in softs)
        detail = ', '.join(f'{n} {x:.3f}' for n, x in zip(VARIANTS, levels))
        ok &= check(np.all(np.isfinite(levels)) and clean,
                    f"{label}: every variant's level is finite, every relaxed path a closed "
                    'minimum through Q*', f'{detail} eV; relaxed flags {flags}')
    return ok


def check_harmonic(bf):
    ok = True
    for b in bf:
        hl = b['harmonic']
        inertia = '-' if hl['inertia'] is None else f"{hl['inertia']:.1e}"
        ok &= check(hl['lowest'] > 0 and hl['drop'] > MID_PATH_DROP
                    and abs(hl['curvature']) < HARMONIC_CURVATURE_RTOL
                    and hl['inertia'] is not None and abs(hl['inertia']) < HARMONIC_INERTIA_RTOL,
                    f"BF/{b['basis']}: the relaxed path is a minimum below the straight one, and "
                    "keeps the TDHF harmonic limit (V''(0) = mu, M(0) = M)",
                    f"lowest curvature {hl['lowest']:+.4f} Ha, {hl['drop'] * 1e3:.2f} mHa below at "
                    f"L / 2; {hl['curvature']:.1e}, {inertia}; Omega_s^2 {hl['omega2']:.3e} Ha^2")
    return ok


def check_mean_absolute(rows):
    straight, relaxed = errors(rows, 'straight'), errors(rows, 'relaxed')
    return check(np.abs(relaxed).mean() < np.abs(straight).mean(),
                 "the relaxed path's mean absolute error lies below the straight path's",
                 f'{np.abs(relaxed).mean():.3f} against {np.abs(straight).mean():.3f} eV over '
                 f'{len(rows)} rows (mean signed {relaxed.mean():+.3f} and {straight.mean():+.3f})')


def run():
    warnings.simplefilter('ignore')
    ok = True
    print('\n-- the triplet rows: first level, eV (level - reference)')
    rows = [triplet_row(label) for label in TRIPLET_ROWS]
    print('\n-- BF: the PSD HOMO, IP in eV (IP - CIPSI) and Z')
    bf = [bf_row(basis) for basis in BF_BASES]
    print('\n-- mean signed, mean absolute and largest absolute error by group (eV)')
    print_groups(rows)
    print('\n-- 1. every variant is finite on every row, every relaxed path a closed minimum')
    ok &= check_finite(rows, bf)
    print('\n-- 2. BF: the relaxed path relaxes, is a minimum, and keeps the TDHF harmonic limit')
    ok &= check_harmonic(bf)
    print("\n-- 3. the relaxed path's mean absolute error lies below the straight path's")
    ok &= check_mean_absolute(rows)
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_soft_mode_variants_full():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
