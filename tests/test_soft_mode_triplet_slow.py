"""The soft-mode split along ethylene's twist and C=C stretch, and on butadiene and benzene
(`LinearResponse.soft_mode`, `GW.soft_mode_qp`; slow twin of tests/test_soft_mode_triplet.py).

All cc-pVDZ, from the MP2/cc-pVDZ minima of tests/soft_mode_references.py (ethylene's second CH2
group turned or both groups moved rigidly), against the EOM-CCSD recorded there.

1. Twist, 0 to 70 degrees in steps of 5 (planar ethylene at its minimum is already past the
   RHF->UHF instability, Omega_T = 0.94i eV): the collective 1D triplet lies above EOM-CCSD by an
   offset that stays between 0.4 and 0.7 eV (+0.52 to +0.55 measured) and moves by less than 0.1
   eV along the whole scan (0.03 measured), while TDA misses by -0.93 to -1.15 eV and goes negative
   at 70 degrees.
2. C=C stretch, 1.22 to 1.46 A, through the instability at 1.3345 A: Omega_T^2 changes sign
   between 1.3325 and 1.3365 A; PSD2@TDHF has no quasiparticle from 1.30 A on (Z 0.46 there,
   0.08 at 1.3325 A, IP 47 eV) and is refused past the instability; with the soft mode split
   off, the PSD2 HOMO keeps Z > 0.65 (0.70 to 0.72 measured) and its IP stays within a 0.3 eV
   band (11.03 to 11.22 eV measured) over the whole scan; the collective 1D triplet falls
   monotonically through the instability, within 0.6 eV of EOM-CCSD where it is recorded
   (+0.42 to +0.49 eV at 1.22 to 1.30 A).
3. Butadiene and benzene (one soft mode each, both past the instability): the collective 1D
   triplet within 0.5 eV of EOM-CCSD (+0.35 and -0.06 eV measured) and closer than the TDA root of
   the same mode (-0.70 and -0.63 eV).

The bars hold the measured values at 1 to 4 threads.

Run: python tests/test_soft_mode_triplet_slow.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.GW.soft_mode_qp import soft_mode_qp
from src.SingleReference.LinearResponse import soft_mode as SM
from src.SingleReference.LinearResponse.casida import CasidaInstability
from tests.soft_mode_references import EOM_CCSD, GEOMETRIES, ethylene_geometry, rhf

#: eV. The twist offset (collective 1D - EOM-CCSD) band: +0.52 to +0.55 measured.
TWIST_OFFSET = (0.4, 0.7)
#: eV. How far that offset may move along the twist: 0.03 measured.
TWIST_OFFSET_SPREAD = 0.1
#: Pole strength of the soft-mode PSD2 HOMO along the stretch: 0.70 to 0.72 measured.
STRETCH_Z_MIN = 0.65
#: eV. Band of the soft-mode PSD2 IP along the stretch: 0.19 measured.
STRETCH_IP_BAND = 0.3
#: eV. Collective 1D triplet against EOM-CCSD along the stretch: 0.49 measured.
STRETCH_EOM_TOL = 0.6
#: eV. Collective 1D triplet against EOM-CCSD for butadiene and benzene: 0.35 measured.
POLYENE_EOM_TOL = 0.5

TWISTS = tuple(range(0, 75, 5))
STRETCHES = (1.22, 1.26, 1.30, 1.32, 1.3245, 1.3295, 1.3325, 1.3365, 1.34, 1.36, 1.40, 1.46)


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def check_twist():
    ok = True
    offsets, tdas = [], []
    for t in TWISTS:
        rec = SM.soft_mode_triplets(rhf(ethylene_geometry(twist=t)), '1d')
        eom = EOM_CCSD['ethylene' if t == 0 else f'ethylene@{t}'][0]
        offsets.append((rec['soft'][0]['excitation'] - eom) * HARTREE_TO_EV)
        tdas.append((rec['omega_tda_soft'] - eom) * HARTREE_TO_EV)
    offsets, tdas = np.array(offsets), np.array(tdas)
    ok &= check(TWIST_OFFSET[0] < offsets.min() and offsets.max() < TWIST_OFFSET[1],
                f'twist: collective 1D - EOM-CCSD within {TWIST_OFFSET} eV',
                f'{offsets.min():+.3f} to {offsets.max():+.3f} eV')
    ok &= check(np.ptp(offsets) < TWIST_OFFSET_SPREAD, 'twist: the offset moves by less than '
                f'{TWIST_OFFSET_SPREAD} eV', f'{np.ptp(offsets):.3f} eV')
    ok &= check(np.all(np.abs(offsets) < np.abs(tdas)), 'twist: closer to EOM-CCSD than TDA at '
                'every angle', f'TDA {tdas.min():+.2f} to {tdas.max():+.2f} eV')
    return ok


def check_stretch():
    ok = True
    om2, exc, ip_soft, z_soft, plain = [], [], [], [], {}
    for r in STRETCHES:
        mf = rhf(ethylene_geometry(rcc=r))
        q = soft_mode_qp(mf, 'PSD2', model='1d')
        rec = q['triplets']
        om2.append(rec['omega2'][0])
        exc.append(rec['soft'][0]['excitation'])
        ip_soft.append(-q['eps'][0] * HARTREE_TO_EV)
        z_soft.append(q['Z'][0])
        try:
            p = soft_mode_qp(mf, 'PSD2', model=None)
            plain[r] = (-p['eps'][0] * HARTREE_TO_EV, p['Z'][0])
        except CasidaInstability:
            plain[r] = None
        key = f'ethylene~{r:.4f}'
        if key in EOM_CCSD:
            miss = (exc[-1] - EOM_CCSD[key][0]) * HARTREE_TO_EV
            ok &= check(abs(miss) < STRETCH_EOM_TOL, f'stretch {r} A: collective 1D triplet within '
                        f'{STRETCH_EOM_TOL} eV of EOM-CCSD', f'{miss:+.3f} eV')
    om2 = np.array(om2)
    k = STRETCHES.index(1.3325)
    ok &= check(om2[k] > 0 > om2[k + 1], 'stretch: Omega_T^2 changes sign between 1.3325 and 1.3365 A')
    ok &= check(all(plain[r] is None for r in STRETCHES[k + 1:]),
                'stretch: PSD2@TDHF refused past the instability')
    ok &= check(all(plain[r][1] < 0.5 for r in STRETCHES[2:k + 1]),
                'stretch: PSD2@TDHF has no quasiparticle from 1.30 A on',
                f'Z {plain[1.30][1]:.2f} at 1.30, {plain[1.3325][1]:.2f} at 1.3325 A; IP '
                f'{plain[1.3325][0]:.1f} eV')
    ok &= check(min(z_soft) > STRETCH_Z_MIN and np.ptp(ip_soft) < STRETCH_IP_BAND,
                'stretch: the soft-mode PSD2 HOMO keeps its quasiparticle in a narrow band',
                f'Z >= {min(z_soft):.2f}, IP {min(ip_soft):.2f} to {max(ip_soft):.2f} eV')
    ok &= check(np.all(np.diff(exc) < 0), 'stretch: the collective triplet falls monotonically '
                'through the instability', f'{exc[0] * HARTREE_TO_EV:.2f} to '
                f'{exc[-1] * HARTREE_TO_EV:.2f} eV')
    return ok


def check_polyenes():
    ok = True
    for name in ('butadiene', 'benzene'):
        rec = SM.soft_mode_triplets(rhf(GEOMETRIES[name]), '1d')
        eom = EOM_CCSD[name][0]
        e1, tda = rec['soft'][0]['excitation'], rec['omega_tda_soft']
        miss = (e1 - eom) * HARTREE_TO_EV
        ok &= check(abs(miss) < POLYENE_EOM_TOL and abs(e1 - eom) < abs(tda - eom),
                    f'{name}: collective 1D triplet within {POLYENE_EOM_TOL} eV of EOM-CCSD and '
                    'closer than TDA', f'{miss:+.3f} eV, TDA {(tda - eom) * HARTREE_TO_EV:+.3f} eV')
    return ok


def run():
    warnings.simplefilter('ignore')
    ok = True
    for title, fn in (('1. ethylene twist', check_twist), ('2. ethylene C=C stretch', check_stretch),
                      ('3. butadiene and benzene', check_polyenes)):
        print(f'\n-- {title}')
        ok &= fn()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_soft_mode_triplet_slow():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
