"""BSE(2): the second-order kernel of van Hoegaerden and Foerster on the DF route
(`second_order_kernel`, `second_order_bse`).

Exact three-index factors (no with_df: the full ERI tensor decomposed), RHF orbitals.

1. The tiled DF kernel equals a dense einsum of its twelve-term table on the
   four-index v and W0 built from the same factors -- A(2)(w) at w = 0 and +-0.37
   Ha, dA(2)/dw, the quadratic forms x^T A(2) y, B(2), the ph channel alone -- for
   the three W-slot rules and both spins, also without the ring terms
   (rings=False), whose removal takes exactly 2(ab|kc)(kc|ij) over both ph
   resolvents out of A(2)(0); and static_screening_aux equals
   LinearResponseSolver's static W0 (full RPA) and -4 A^-1 (TDA).
2. H2O/6-31G at HF energies with the static RPA W0, full BSE: the lowest roots of
   the verbatim paper Fortran (makeABmat_BSE and the routines it calls, compiled
   in the port's harness, BSE2_port/verification/sosex2_harness): 2W0 - v kernel
   singlets 10.784482, 12.810068, 13.189377 and triplets 10.025630, 11.644652,
   12.696080 eV; v kernel 10.710253 / 9.918701; W0-averaged 10.750023 / 9.975161.
3. The dBSE(2) step of the driver equals the matrix formula X^T [A(2)(+O) -
   A(2)(0)] X + Y^T [A(2)(-O) - A(2)(0)] Y and X^T A(2)'(+O) X - Y^T A(2)'(-O) Y,
   full and TDA, and the Casida vectors carry X^T X - Y^T Y = 1.
4. H2O/cc-pVDZ at QuAcK's geometry and HF energies, v kernel: the full static
   singlets and triplets and the dTDA step on them (X only) of QuAcK's BSE@GF2
   routines with the effective static reference (BSE2_port/quack_dbse_check.py,
   which reproduces the QuAcK binary): an independent code, to its 6 decimals.
5. Refusals: an unknown rule, spin or channel, W_aux with the v rule or without a
   screened one, an unrestricted reference.
6. G0F2 (`g0f2_energies`) on the same QuAcK input: every orbital equals QuAcK's
   graphical G0F2 table, HOMO -11.008134 and LUMO 4.530804 eV among them, except
   the high virtuals where QuAcK's Newton step from eHF stops on a satellite of
   weight 0.01-0.2; there the root of largest weight over the whole axis is
   kept.
   On QuAcK's own G0F2 energies the v kernel gives QuAcK's BSE@GF2 full static
   roots with the effective static reference (singlets 7.477928, 9.561180, ...).
7. first_order='screened' with rings=False is A(1) and B(1) with W0 plus the
   ring-free A(2) and B(2), full and TDA, with its dBSE(2) step on the
   ring-free kernel, and differs from both the bare first order and the exch
   variant.

Run: python tests/test_second_order_bse.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf

from src.Base.numerics.constants import HARTREE_TO_EV
from src.Base.pyscf.pyscf_interface import get_density_fitting_coefficients
from src.SingleReference.ADC.second_order import g0f2_energies
from src.SingleReference.BSE.second_order_bse import dynamical_correction, second_order_bse
from src.SingleReference.LinearResponse import second_order_kernel as SOK
from src.SingleReference.LinearResponse.casida import CasidaSolver
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver
from tests.pins import CROSS_PATH_RTOL

HARNESS_H2O = 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692'
QUACK_H2O = 'O 0 0 0; H 0.7571 0 0.5861; H -0.7571 0 0.5861'
#: eV, the verbatim-Fortran harness roots (6 decimals printed).
HARNESS_ROOTS = {('singlet', '2sosex'): [10.784482, 12.810068, 13.189377],
                 ('triplet', '2sosex'): [10.025630, 11.644652, 12.696080],
                 ('singlet', 'v'): [10.710253], ('triplet', 'v'): [9.918701],
                 ('singlet', 'average'): [10.750023], ('triplet', 'average'): [9.975161]}
#: eV, QuAcK same input: full static roots and the dTDA step (X only) on them.
QUACK_ROOTS = {
    'singlet': ([10.303798, 12.299852, 12.603900, 14.595165, 15.849076, 18.851955],
                [0.213220, 0.287434, 0.190433, 0.261018, 0.218949, 0.194875]),
    'triplet': ([9.615032, 11.608129, 11.890523, 13.582339, 14.825224, 16.508165],
                [0.238191, 0.267215, 0.319769, 0.376007, 0.283622, 0.380177])}
#: eV, QuAcK's graphical G0F2@RHF table (all 24 orbitals) and its weights, and
#: the BSE@GF2 full static roots on those energies.
QUACK_G0F2 = [-539.816787, -32.735281, -17.906378, -13.418362, -11.008134, 4.530804, 6.533658,
              20.069323, 21.464997, 29.735820, 30.490734, 33.392050, 37.773965, 38.837907,
              44.204819, 48.062383, 50.887295, 66.014695, 65.769323, 88.050537, 90.584852,
              95.479244, 104.703443, 112.985043]
QUACK_G0F2_Z = [0.774596, 0.618298, 0.929210, 0.914210, 0.907790, 0.983422, 0.982295, 0.958559,
                0.945716, 0.918680, 0.927600, 0.201977, 0.946923, 0.962951, 0.957558, 0.740510,
                0.877995, 0.009914, 0.244501, 0.244196, 0.033036, 0.157395, 0.028454, 0.042096]
QUACK_BSE_GF2 = {'singlet': [7.477928, 9.561180, 10.153010, 12.235785, 14.258184, 17.264197],
                 'triplet': [6.801796, 9.160650, 9.183159, 11.320779, 13.140528, 14.947115]}
#: eV: half a unit of the sixth decimal the harness prints, and the QuAcK
#: binary's own 6 decimals (it runs on its integrals, 1e-6 eV apart from these).
PRINTED_TOL = 5e-7
QUACK_TOL = 5e-6


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def rhf(atom, basis):
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf


class DenseTable:
    """The kernel's term table as plain einsums over four-index v and W0."""

    def __init__(self, mf, W_aux):
        self.no = mf.mol.nelectron // 2
        eps = mf.mo_energy
        self.eo, self.ev = eps[:self.no], eps[self.no:]
        B = np.asarray(get_density_fitting_coefficients(mf.mol, mf))
        self.g = {'v': np.einsum('Ppq,Prs->pqrs', B, B),
                  'w': np.einsum('Ppq,PQ,Qrs->pqrs', B, W_aux, B)}

    def block(self, n, which):
        O, V = slice(0, self.no), slice(self.no, None)
        return self.g[which][tuple(O if ch in 'ijkl' else V for ch in n)]

    def resolvent(self, rs, w, p):
        e = [(-self.eo if ch in 'ijkl' else self.ev) for ch in rs]
        D = e[0][:, None, None, None] + e[1][None, :, None, None] \
            + e[2][None, None, :, None] + e[3][None, None, None, :]
        return 1.0 / (D - w) ** p

    def table(self, s, swaps=(), screen=True, w=0.0, p=1, channels=SOK.CHANNELS, rings=True):
        nv = self.ev.size
        K = np.zeros((self.no, nv, self.no, nv))
        for code, n1, n2, slot, rs, channel in SOK.A_TERMS:
            if channel not in channels:
                continue
            if not rings and code == '2' and (n1, n2) == ('abkc', 'kcij'):
                continue    # the paper's fourth ph term, +2 (ab|kc)(kc|ij)
            n1r, n2r = SOK._relabel(n1, swaps), SOK._relabel(n2, swaps)
            K += SOK._coefficient(code, s) * np.einsum(
                f'{n1r},{n2r},{rs}->iajb', self.block(n1r, 'v'),
                self.block(n2r, 'w' if slot and screen else 'v'), self.resolvent(rs, w, p))
        return K.reshape(self.no * nv, self.no * nv)

    def A(self, rule, s, w=0.0, p=1, channels=SOK.CHANNELS, rings=True):
        if rule == 'v':
            return self.table(s, (), False, w, p, channels, rings)
        T = self.table(s, (), True, w, p, channels, rings) \
            + self.table(s, SOK.MIRROR, True, w, p, channels, rings)
        if rule == 'average':
            return 0.5 * T
        return T - self.table(s, (), False, w, p, channels, rings)

    def B(self, rule, s, rings=True):
        if rule == 'v':
            return self.table(s, SOK.SWAP_C, False, rings=rings)
        T = self.table(s, SOK.SWAP_P, rings=rings) + self.table(s, SOK.SWAP_C, rings=rings)
        return 0.5 * T if rule == 'average' else T - self.table(s, SOK.SWAP_C, False, rings=rings)


def relative(a, b):
    return float(np.abs(np.asarray(a) - b).max() / max(1.0, np.abs(b).max()))


def sym(M):
    return 0.5 * (M + M.T)


def check_kernel_against_the_dense_table(mf):
    nocc = mf.mol.nelectron // 2
    B = np.asarray(get_density_fitting_coefficients(mf.mol, mf))
    W_aux = SOK.static_screening_aux(mf.mo_energy, B, nocc)
    ok = check(relative(W_aux, LinearResponseSolver(mf.mo_energy, coeff_df=B).static_screening_aux(nocc))
               < CROSS_PATH_RTOL, 'static W0 equals LinearResponseSolver.static_screening_aux')
    dense = DenseTable(mf, W_aux)
    rng = np.random.default_rng(7)
    x, y = rng.standard_normal((2, nocc * (mf.mo_energy.size - nocc)))
    worst = 0.0
    for spin, s in SOK.SPIN_FACTOR.items():
        for rule in SOK.SLOT_RULES:
            K = SOK.SecondOrderKernel(mf.mo_energy, B, nocc, spin, rule,
                                      None if rule == 'v' else W_aux)
            for w, p in ((0.0, 1), (0.37, 1), (-0.37, 1), (0.37, 2)):
                ref = dense.A(rule, s, w, p)
                worst = max(worst, relative(K.A(w, p), ref),
                            relative(K.form(x, w, p, y), x @ ref @ y))
            worst = max(worst, relative(sym(K.B()), sym(dense.B(rule, s))))
        K = SOK.SecondOrderKernel(mf.mo_energy, B, nocc, spin, '2sosex', W_aux, channels=('ph',))
        worst = max(worst, relative(K.A(), dense.A('2sosex', s, channels=('ph',))))
    ok &= check(worst < CROSS_PATH_RTOL, 'tiled DF kernel = dense table (A, dA/dw, forms, B, ph only)',
                f'max rel {worst:.1e}')
    # rings=False: the same without the paper's fourth ph term in both resolvents
    worst = 0.0
    for spin, s in SOK.SPIN_FACTOR.items():
        for rule in SOK.SLOT_RULES:
            K = SOK.SecondOrderKernel(mf.mo_energy, B, nocc, spin, rule,
                                      None if rule == 'v' else W_aux, rings=False)
            for w, p in ((0.0, 1), (0.37, 1), (-0.37, 1), (0.37, 2)):
                ref = dense.A(rule, s, w, p, rings=False)
                worst = max(worst, relative(K.A(w, p), ref),
                            relative(K.form(x, w, p, y), x @ ref @ y))
            worst = max(worst, relative(sym(K.B()), sym(dense.B(rule, s, rings=False))))
        K = SOK.SecondOrderKernel(mf.mo_energy, B, nocc, spin, '2sosex', W_aux, channels=('ph',),
                                  rings=False)
        worst = max(worst, relative(K.A(), dense.A('2sosex', s, channels=('ph',), rings=False)))
    ok &= check(worst < CROSS_PATH_RTOL, 'rings=False kernel = dense table without the ring terms',
                f'max rel {worst:.1e}')
    # what rings=False removes from A(2)(0): 2 sum_kc (ab|kc)(kc|ij) [1/(e_b - e_i + e_c - e_k)
    # + 1/(e_a - e_j + e_c - e_k)], the second-order piece of -(ij|W0|ab) up to its denominator
    eo, ev = mf.mo_energy[:nocc], mf.mo_energy[nocc:]
    g = np.einsum('Ppq,Prs->pqrs', B, B)
    O, V = slice(0, nocc), slice(nocc, None)
    gap = ev[None, :] - eo[:, None]                         # [k, c] = e_c - e_k
    D1 = gap[:, :, None, None] + gap[None, None, :, :]      # [i, b, k, c] = e_b - e_i + e_c - e_k
    ring = 2.0 * (np.einsum('abkc,kcij,ibkc->iajb', g[V, V, O, V], g[O, V, O, O], 1.0 / D1)
                  + np.einsum('abkc,kcij,jakc->iajb', g[V, V, O, V], g[O, V, O, O], 1.0 / D1))
    ring = ring.reshape(nocc * ev.size, -1)
    worst = 0.0
    for spin in SOK.SPIN_FACTOR:
        for rule in SOK.SLOT_RULES:
            Wr = None if rule == 'v' else W_aux
            full = SOK.SecondOrderKernel(mf.mo_energy, B, nocc, spin, rule, Wr).A()
            ringless = SOK.SecondOrderKernel(mf.mo_energy, B, nocc, spin, rule, Wr, rings=False).A()
            worst = max(worst, relative(full - ringless, ring))
    ok &= check(worst < CROSS_PATH_RTOL, 'rings=False removes 2(ab|kc)(kc|ij) over both ph '
                'resolvents from A(2)(0), every rule and spin', f'max rel {worst:.1e}')
    # TDA screening: W_aux = 1 + B_ov (-4 A^-1) B_ov^T, A = D + 2 B_ov^T B_ov
    B_ov = B[:, :nocc, nocc:].reshape(B.shape[0], -1)
    gaps = (mf.mo_energy[nocc:][None, :] - mf.mo_energy[:nocc, None]).ravel()
    A_tda = np.diag(gaps) + 2.0 * B_ov.T @ B_ov
    explicit = np.eye(B.shape[0]) - 4.0 * B_ov @ np.linalg.solve(A_tda, B_ov.T)
    ok &= check(relative(SOK.static_screening_aux(mf.mo_energy, B, nocc, tda=True), explicit)
                < CROSS_PATH_RTOL, 'TDA static W0 equals 1 - 4 B_ov A^-1 B_ov^T')
    return ok


def check_fortran_harness_roots(mf):
    ok = True
    for (spin, rule), pin in HARNESS_ROOTS.items():
        roots = second_order_bse(mf, kernel=rule, spin=spin, dynamical_roots=0)['omega']
        d = np.abs(roots[:len(pin)] * HARTREE_TO_EV - pin).max()
        ok &= check(d < PRINTED_TOL, f'{spin:7s} {rule:7s} full BSE roots = paper Fortran',
                    f'max |d| {d:.1e} eV')
    return ok


def check_dynamical_correction(mf):
    nocc = mf.mol.nelectron // 2
    B = np.asarray(get_density_fitting_coefficients(mf.mol, mf))
    ok = True
    for spin in SOK.SPIN_FACTOR:
        for tda in (False, True):
            r = second_order_bse(mf, kernel='2sosex', spin=spin, tda=tda, dynamical_roots=3)
            K = SOK.SecondOrderKernel(mf.mo_energy, B, nocc, spin, '2sosex', r['W_aux'])
            A0, dyn, worst = K.A(), r['dynamical'], 0.0
            for m in range(3):
                O, x, yv = r['omega'][m], r['X'][:, m], r['Y'][:, m]
                delta = x @ (K.A(O) - A0) @ x + yv @ (K.A(-O) - A0) @ yv
                slope = x @ K.A(O, 2) @ x - yv @ K.A(-O, 2) @ yv
                zeta = 1.0 / (1.0 - slope)
                worst = max(worst, abs(dyn['delta'][m] - delta), abs(dyn['slope'][m] - slope),
                            abs(dyn['omega'][m] - O - zeta * delta))
            norm = np.abs(np.diag(r['X'].T @ r['X'] - r['Y'].T @ r['Y']) - 1.0).max()
            ok &= check(worst < CROSS_PATH_RTOL and norm < CROSS_PATH_RTOL,
                        f"{spin:7s} {'TDA' if tda else 'full'}: dBSE(2) = matrix formula, X^T X - Y^T Y = 1",
                        f'{worst:.1e}, {norm:.1e}')
    return ok


def check_quack_same_input(mf):
    nocc = mf.mol.nelectron // 2
    B = np.asarray(get_density_fitting_coefficients(mf.mol, mf))
    ok = True
    for spin, (static, steps) in QUACK_ROOTS.items():
        r = second_order_bse(mf, kernel='v', spin=spin, dynamical_roots=0)
        K = SOK.SecondOrderKernel(mf.mo_energy, B, nocc, spin, 'v')
        roots = r['omega'][:len(static)]
        dtda = []
        for O, x in zip(roots, r['X'].T):
            dtda.append((K.form(x, O) - K.form(x)) / (1.0 - K.form(x, O, power=2)) * HARTREE_TO_EV)
        d_static = np.abs(roots * HARTREE_TO_EV - static).max()
        d_step = np.abs(np.array(dtda) - steps).max()
        ok &= check(d_static < QUACK_TOL and d_step < QUACK_TOL,
                    f'{spin:7s} v kernel: static roots and dTDA steps = QuAcK',
                    f'max |d| {d_static:.1e} / {d_step:.1e} eV')
    return ok


def check_refusals(mf):
    nocc = mf.mol.nelectron // 2
    B = np.asarray(get_density_fitting_coefficients(mf.mol, mf))
    W_aux = SOK.static_screening_aux(mf.mo_energy, B, nocc)
    bad = [dict(rule='w0'), dict(spin='quintet'), dict(channels=('ph', 'ring')),
           dict(rule='v', W_aux=W_aux), dict(rule='2sosex')]
    ok = True
    for kwargs in bad:
        try:
            SOK.SecondOrderKernel(mf.mo_energy, B, nocc, **kwargs)
            ok &= check(False, f'refuses {kwargs}')
        except ValueError:
            ok &= check(True, f'refuses {sorted(kwargs)}')
    mol = gto.M(atom='O 0 0 0; H 0 0 0.97', basis='sto-3g', spin=1, verbose=0)
    try:
        second_order_bse(scf.UHF(mol))
        ok &= check(False, 'refuses an unrestricted reference')
    except NotImplementedError:
        ok &= check(True, 'refuses an unrestricted reference')
    return ok


def check_g0f2_against_quack(mf):
    nocc = mf.mol.nelectron // 2
    B = np.asarray(get_density_fitting_coefficients(mf.mol, mf))
    eps_qp, Z = g0f2_energies(mf.mo_energy, nocc, B_aa=B)
    d = eps_qp * HARTREE_TO_EV - QUACK_G0F2
    same = np.abs(d) < QUACK_TOL
    heavier = Z > np.array(QUACK_G0F2_Z)
    ok = check(same[nocc - 1] and same[nocc], 'G0F2 HOMO and LUMO = QuAcK',
               f'{d[nocc - 1]:+.1e}, {d[nocc]:+.1e} eV')
    ok &= check(np.all(same | heavier) and same.sum() >= 19,
                'every other orbital = QuAcK, or a root of larger weight kept',
                f'{same.sum()} of {d.size} equal; Z {np.round(Z[~same], 2).tolist()} against '
                f'{np.round(np.array(QUACK_G0F2_Z)[~same], 2).tolist()}')
    for spin, roots in QUACK_BSE_GF2.items():
        got = second_order_bse(mf, eps=np.array(QUACK_G0F2) / HARTREE_TO_EV, kernel='v', spin=spin,
                               dynamical_roots=0)['omega'][:len(roots)] * HARTREE_TO_EV
        dd = np.abs(got - roots).max()
        ok &= check(dd < QUACK_TOL, f'{spin:7s} BSE@GF2 on QuAcK G0F2 energies = QuAcK', f'max |d| {dd:.1e} eV')
    return ok


def check_screened_without_rings(mf):
    """first_order='screened' with rings=False is A(1) and B(1) with W0 plus the
    kernel without its ring terms, and differs from both the paper's bare first
    order and the SI's exch variant (W0 in A(1) with the rings kept)."""
    nocc = mf.mol.nelectron // 2
    B = np.asarray(get_density_fitting_coefficients(mf.mol, mf))
    ok = True
    for tda in (False, True):
        res = second_order_bse(mf, kernel='2sosex', first_order='screened', rings=False, tda=tda,
                               dynamical_roots=3)
        W, eps = res['W_aux'], res['eps']
        K = SOK.SecondOrderKernel(eps, B, nocc, 'singlet', '2sosex', W, rings=False)
        A1, B1 = LinearResponseSolver(eps, coeff_df=B).build_casida_matrices(
            nocc, lBSE=True, W_aux=W, triplet=False)
        A = A1 + sym(K.A())
        Bm = np.zeros_like(A) if tda else B1 + sym(K.B())
        by_hand = np.asarray(CasidaSolver(A, Bm).solve(tda=tda)[0])
        dyn = dynamical_correction(K, res['omega'], res['X'], res['Y'], 3, tda)['omega']
        exch = second_order_bse(mf, kernel='2sosex', first_order='screened', tda=tda,
                                dynamical_roots=0)['omega']
        bare = second_order_bse(mf, kernel='2sosex', tda=tda, dynamical_roots=0)['omega']
        new = res['omega']
        ok &= check(relative(new[:5], by_hand[:5]) < CROSS_PATH_RTOL
                    and relative(res['dynamical']['omega'], dyn) < CROSS_PATH_RTOL
                    and np.abs(new[:3] - exch[:3]).min() > 1e-4 and np.abs(new[:3] - bare[:3]).min() > 1e-4,
                    f"first_order='screened', rings=False ({'TDA' if tda else 'full'}) = A(1)[W0] "
                    '+ ring-free A(2), B(2), and dBSE(2) on the ring-free kernel',
                    'S1-S3 eV bare / exch / screened without rings: '
                    + ' / '.join(' '.join(f'{x:.3f}' for x in v[:3] * HARTREE_TO_EV)
                                 for v in (bare, exch, new)))
    return ok


def run():
    warnings.simplefilter('ignore')
    harness = rhf(HARNESS_H2O, '6-31g')
    print('\n-- 1. the tiled kernel against the dense table')
    ok = check_kernel_against_the_dense_table(harness)
    print('\n-- 2. against the paper Fortran')
    ok &= check_fortran_harness_roots(harness)
    print('\n-- 3. the dBSE(2) step')
    ok &= check_dynamical_correction(harness)
    print('\n-- 4. against QuAcK')
    quack = rhf(QUACK_H2O, 'cc-pvdz')
    ok &= check_quack_same_input(quack)
    print('\n-- 5. refusals')
    ok &= check_refusals(harness)
    print('\n-- 6. G0F2 and BSE@GF2 against QuAcK')
    ok &= check_g0f2_against_quack(quack)
    print('\n-- 7. W0 in the first order without the double-counted rings')
    ok &= check_screened_without_rings(harness)
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_second_order_bse():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
