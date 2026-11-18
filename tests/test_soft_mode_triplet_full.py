"""The soft-mode split against live EOM-CCSD on the larger cases (`LinearResponse.soft_mode`,
`GW.soft_mode_qp`; heavy twin of tests/test_soft_mode_triplet.py, run on a cluster).

All cc-pVDZ at the MP2/cc-pVDZ minima of tests/soft_mode_references.py. The EOM-CCSD triplet is
the root whose singles overlap the soft direction most, its Davidson seeded with d
(`eom_ccsd_references`); spin-flip EOM-CCSD from the M_s = 1 ROHF seeded with d's leading pair
(`spin_flip_gap`) is the second reference where EOM-CCSD from the RHF is known to fall short (O3,
ethylene twisted by 45 degrees or more). What is held:

1. Every EOM-CCSD reference is well defined: CCSD and the matched EOM triplet root converged, the
   matched triplet overlaps d by at least 0.9, the matched IP root has a HOMO weight of at least
   0.5. So is
   every spin-flip reference: d's leading pair carries at least 0.9 of d, the M_s = 1 ROHF
   seeded with that pair keeps its singly occupied orbitals on it (at least 0.9), ROHF, UCCSD
   and the spin-flip roots converged, and the singlet lies below the triplet.
2. The recorded references of butadiene, benzene, O3 and the twisted ethylenes come back to
   1e-5 Ha (another machine and BLAS).
3. The collective 1D triplet lies within 1.2 eV of EOM-CCSD for hexatriene, octatetraene,
   butadiene, benzene and the C=C stretch from 1.3145 to 1.46 A (the small molecules and the
   twist stay within 1.15 eV in the fast and slow files), and within 1.2 eV of spin-flip
   EOM-CCSD for O3 and the twisted ethylenes.
4. Along the stretch, through the instability at 1.3345 A, the offset collective 1D - EOM-CCSD
   moves by less than 0.15 eV (0.07 eV from 1.22 to 1.30 A in the slow file).
5. The soft-mode PSD2 HOMO equation has a root (finite energy) for butadiene, benzene,
   hexatriene and octatetraene; its Z is printed, not held (PSD2 itself is far from EOM-IP-CCSD
   for these molecules).

The full table (triplets, TDA, 3D, IPs, pole strengths, overlaps, flags) is printed for the record.
Measured on a workstation for butadiene and benzene: collective 1D +0.35 and -0.06 eV from the
recorded EOM-CCSD.

Run: python tests/test_soft_mode_triplet_full.py, or under pytest (about two hours on 16 cores,
dominated by octatetraene's EOM-CCSD).
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.GW.soft_mode_qp import soft_mode_qp
from src.SingleReference.LinearResponse import soft_mode as SM
from tests.soft_mode_references import (EOM_CCSD, eom_ccsd_references, geometry, rhf,
                                        spin_flip_gap)

#: Matched EOM triplet overlap with d and IP root HOMO weight: 0.96 and 0.99 the lowest recorded.
MIN_OVERLAP, MIN_HOMO_WEIGHT = 0.9, 0.5
#: Spin-flip references: the share of d on its leading pair (sigma_1^2: 0.974 to 0.985 for O3 and
#: the twisted ethylenes in cc-pVDZ), and the share of the seeded ROHF's singly occupied orbitals
#: on that pair (0.96 for O3, 0.97 to 0.98 for the ethylenes; 0.48 for O3's default-guess
#: triplet), both at the mean-field level on a workstation.
MIN_LEADING_PAIR, MIN_SF_MATCH = 0.9, 0.9
#: Ha. Recorded references against the live ones.
RECORDED_TOL = 1e-5
#: eV. Collective 1D triplet against (spin-flip) EOM-CCSD: 1.15 the worst measured so far (O3
#: against EOM-CCSD from the RHF).
EOM_TRIPLET_TOL = 1.2
#: eV. Movement of the offset collective 1D - EOM-CCSD along the stretch.
STRETCH_SPREAD = 0.15

POLYENES = ('butadiene', 'benzene', 'hexatriene', 'octatetraene')
STRETCH = ('ethylene~1.3145', 'ethylene~1.3200', 'ethylene~1.3245', 'ethylene~1.3295',
           'ethylene~1.3325', 'ethylene~1.3365', 'ethylene~1.3400', 'ethylene~1.3600',
           'ethylene~1.3800', 'ethylene~1.4000', 'ethylene~1.4200', 'ethylene~1.4400',
           'ethylene~1.4600')
SPIN_FLIP = ('O3', 'ethylene@45', 'ethylene@60', 'ethylene@70')


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def study(label, psd=False, spin_flip=False):
    """Every number of one system (Ha), and its row of the printed table."""
    mf = rhf(geometry(label))
    rec1 = SM.soft_mode_triplets(mf, '1d')
    rec3 = SM.soft_mode_triplets(mf, '3d')
    d = SM.soft_split(*SM.triplet_tdhf_matrices(mf))['modes'][rec1['soft'][0]['index']]['direction']
    t, ip, ov, hw, conv = eom_ccsd_references(mf, d)
    out = dict(label=label, omega2=rec1['omega2'][0], tda=rec1['omega_tda_soft'],
               e1d=rec1['soft'][0]['excitation'], e3d=rec3['soft'][0]['excitation'],
               eom=t, ip=ip, overlap=ov, homo_weight=hw, converged=conv,
               flags=[f['flag'] for f in rec1['numerical_flags']])
    if spin_flip:
        out['sf'], out['sf_converged'], out['sf_match'] = spin_flip_gap(mf, d)
        out['leading_pair'] = float(rec1['soft'][0]['sigma'][0] ** 2)
    if psd:
        q = soft_mode_qp(mf, 'PSD2', model='1d')
        out['psd2'], out['z'] = -q['eps'][0], q['Z'][0]
    om = out['omega2']
    om_txt = f'{np.sqrt(om) * HARTREE_TO_EV:.3f}' if om > 0 else f'{np.sqrt(-om) * HARTREE_TO_EV:.3f}i'
    ev = lambda x: f'{x * HARTREE_TO_EV:7.3f}' if x is not None and np.isfinite(x) else '      -'
    print(f"  {label:17s} TDHF {om_txt:>7s} TDA {ev(out['tda'])} 1D {ev(out['e1d'])} 3D {ev(out['e3d'])} "
          f"EOM {ev(t)} SF {ev(out.get('sf'))} | IP EOM {ev(ip)} PSD2 {ev(out.get('psd2'))} "
          f"Z {out.get('z', np.nan):.2f} | ovl {ov:.3f} hw {hw:.2f} conv {conv} {out['flags']}",
          flush=True)
    return out


def run():
    warnings.simplefilter('ignore')
    ok = True
    print('\n-- the table (eV)')
    rows = [study(n, psd=True) for n in POLYENES]
    rows += [study(s, spin_flip=True) for s in SPIN_FLIP]
    rows += [study(s) for s in STRETCH]
    by = {r['label']: r for r in rows}
    print('\n-- 1. the references are well defined')
    for r in rows:
        ok &= check(r['converged'] and r['overlap'] >= MIN_OVERLAP and r['homo_weight'] >= MIN_HOMO_WEIGHT,
                    f"{r['label']}: converged, matched", f"overlap {r['overlap']:.3f}, HOMO weight "
                    f"{r['homo_weight']:.2f}")
        if 'sf' in r:
            ok &= check(r['leading_pair'] >= MIN_LEADING_PAIR and r['sf_match'] >= MIN_SF_MATCH
                        and r['sf_converged'] and r['sf'] > 0,
                        f"{r['label']}: spin-flip reference on d's triplet, converged, singlet below",
                        f"leading pair {r['leading_pair']:.3f}, match {r['sf_match']:.3f}, "
                        f"gap {r['sf'] * HARTREE_TO_EV:.3f} eV")
    print('\n-- 2. the recorded references come back')
    for name in ('butadiene', 'benzene', 'O3', 'ethylene@45', 'ethylene@60', 'ethylene@70'):
        t_rec, ip_rec = EOM_CCSD[name][:2]
        miss = max(abs(by[name]['eom'] - t_rec), abs(by[name]['ip'] - ip_rec))
        ok &= check(miss < RECORDED_TOL, f'{name}: recorded EOM-CCSD reproduced', f'{miss:.1e} Ha')
    print('\n-- 3. the collective triplet against EOM-CCSD')
    for r in rows:
        ref = r.get('sf', r['eom'])
        miss = (r['e1d'] - ref) * HARTREE_TO_EV
        ok &= check(abs(miss) < EOM_TRIPLET_TOL, f"{r['label']}: within {EOM_TRIPLET_TOL} eV of "
                    f"{'spin-flip ' if 'sf' in r else ''}EOM-CCSD", f'{miss:+.3f} eV')
    print('\n-- 4. the stretch through the instability')
    off = np.array([(by[s]['e1d'] - by[s]['eom']) * HARTREE_TO_EV for s in STRETCH])
    ok &= check(np.ptp(off) < STRETCH_SPREAD, f'offset moves by less than {STRETCH_SPREAD} eV',
                f'{off.min():+.3f} to {off.max():+.3f} eV')
    print('\n-- 5. the soft-mode PSD2 has a root')
    for n in POLYENES:
        ok &= check(np.isfinite(by[n]['psd2']), f'{n}: PSD2 HOMO root found',
                    f"IP {by[n]['psd2'] * HARTREE_TO_EV:.3f} eV, Z {by[n]['z']:.2f}")
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_soft_mode_triplet_full():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
