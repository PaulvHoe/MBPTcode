"""The open-shell soft-mode split on the paper's own open-shell systems (Burth, Kitsaras and Loos,
arXiv:2610.36154): the spin-flip TDHF of cyclobutadiene's high-spin triplet UHF in cc-pVTZ, and the
core-ionized BF+ (F1s) and N2+ (1s) in aug-cc-pCVTZ
(`LinearResponse.soft_mode.soft_mode_open_shell`, `LinearResponse.soft_mode.noci_levels`; heavy,
run on a cluster, never on a workstation).

Geometries from the paper's supporting information (A): cyclobutadiene D2h with C at (+-0.783,
+-0.6715, 0) and H at (+-1.54123883, +-1.43212465, 0), D4h with C at (+-0.7195, +-0.7195, 0) and H
at (+-1.47822558, +-1.47822558, 0) (CCSD(T)/cc-pVTZ, Levchenko and Krylov 2004); BF at 1.2685 A and
N2 at 1.1007 A (Marie, Burth and Loos 2026). Values quoted from the paper: Tables 1 to 3 and Tables
S1 and S2.

Held, each a claim that stands before the run:

1. Cyclobutadiene, cc-pVTZ, on the high-spin (M_s = 1) triplet UHF: the solution from pyscf's
   guess and, if another, the one reached by following its internal instabilities (pyscf's
   stability analysis, up to SCF_FOLLOW times; at D2h in STO-3G the guess's solution is unstable
   and the followed one 5.3 mHa lower). One of them, the one the paper used, holds a and b:
   a. SF-TDA, the eigenvalues of the spin-flip block's alpha -> beta sector of A
      (`uhf_tdhf_matrices`): each of the paper's SF-TDA-RPAx energies relative to the singlet X
      (1.523, 3.845, 5.508 eV at D2h; 0.318, 3.131, 2.657 eV at D4h) is matched by a root relative
      to the lowest one to TDA_TOL.
   b. Full spin-flip TDHF with the spin rotation taken out (`_spin_rotation_split`): its complex
      Omega^2 form one quartet, sqrt(Omega^2) = 3.556 + 2.552i eV (D2h) and 3.157 + 2.084i eV
      (D4h) to RPAX_TOL. In the paper's Tables S1 and S2 the triplet's row is -Re Omega_X, its
      M_s = 0 component being the spin-rotation pole at 0, and every row's |Im| is Im Omega_X; its
      other rows less Re Omega_X (2.386 and 3.939 eV at D2h, 2.999 and 2.405 eV at D4h) are matched
      by real roots to RPAX_TOL.
   c. On that solution (the guess's if none holds them) the default split refuses the reference
      (CasidaInstability), and collisions='gcm' (model='gcm') gives every pole real and finite,
      the pair's strongest de-excitation negative (the singlet below the triplet), at both
      geometries. Printed against
      EOM-SF-CCSD(fT) (the triplet 1.516 and 0.163 eV above X), SF-TDA-RPAx (1.523, 0.318) and
      SF-BSE (1.678, 0.224).
2. N2+ 1s^-1, aug-cc-pCVTZ, maximum-overlap UHFs from the neutral RHF's orbitals:
   a. Delta-UHF of the hole localized on one atom (guess (1sigma_g +- 1sigma_u) / sqrt2) and of the
      symmetric hole (guess 1sigma_g) within DELTA_TOL of the paper's 409.42 and 419.25 eV.
   b. The localized hole's two blocks are real: no complex Omega^2 in either (the spin rotation
      taken out of the spin-flip block); the symmetric hole's spin-flip block has a complex
      Omega^2 (the paper's footnote b). Checked on the blocks themselves, since the soft-mode
      driver can refuse a real block for other reasons (a soft mode of the localized hole that is
      totally symmetric, whose path is not even: aug-cc-pVDZ, NotImplementedError).
   c. noci_levels of the hole on either atom (the two localized solutions): two states (rank 2),
      split by less than NOCI_SPLIT_MAX, the lower below the localized Delta-UHF. Printed against
      Delta-CVS-FCI (409.88 eV).
3. BF+ F1s^-1, aug-cc-pCVTZ:
   a. The first solution (maximum overlap, the neutral's F1s emptied in beta) within DELTA_TOL of
      694.32 eV, its spin-conserving block with a complex Omega^2 (footnote a).
   b. The second: the orbitals of the first turned by STEP along the lowest eigenvector of its CVS
      Hessian (A + B of the spin-conserving block over the pairs that do not fill the hole), which
      is negative, and reconverged with the same occupations, within DELTA_TOL of 694.20 eV; its
      spin-conserving block real and its spin-flip block complex or with a negative A + B
      eigenvalue (footnote b: unstable toward GHF).
   c. noci_levels of the two solutions: printed.

Printed besides, for every reference the split does not refuse: its soft modes (level, mass,
flags) in both blocks with model='gcm', and the times.

Measured on a workstation in smaller bases (not this file's rows): cyclobutadiene in STO-3G (24
basis functions): full spin-flip TDHF complex at both geometries (3.751 + 4.538i and 3.245 + 4.411i
eV), the two-mode GCM's triplet 1.584 and 0.554 eV above the singlet (pi CASSCF(4,4): 2.217 and
0.777 eV; SF-TDA 2.42 and 0.88), 140 s a pair. N2+ in cc-pVDZ: Delta-UHF 412.19 (localized) and
420.71 eV (symmetric); the localized blocks real, the symmetric ones complex in both blocks with
quartets in degenerate (pi) pairs, which the two-mode model refuses; NOCI 412.17 and 412.21 eV.
BF+ in aug-cc-pVDZ: 697.26 eV, its spin-conserving block real, its CVS Hessian negative
(-0.045 Ha, a pi pair), followed (STEP 0.3) to a second solution 0.128 eV lower (the paper: 0.12)
whose spin-conserving block is real and spin-flip block complex.

aug-cc-pCVTZ is not shipped with pyscf (it loads it from basis-set-exchange only when that package
is installed, and the cluster's environment does not install it): `aug_cc_pcvtz` assembles it as
cc-pCVTZ plus the diffuse shells of aug-cc-pVTZ (one s, p, d and f per atom), checked on a
workstation by building the four molecules (no SCF): 118 basis functions for N2 and for BF, as
pyscf's aug-cc-pwCVTZ, and 1449 spin-conserving and 1450 spin-flip pairs for the cations. Each
system runs in its own guard (`guarded`): an exception is reported as that system's failure and
the systems after it still run; a refusal of the two-mode GCM is a failure of claim 1c.

Memory, estimated from cyclobutadiene in STO-3G and 6-31G at the file's geometries (282 and 842
pairs, traced): about 6 GB at the peak, so size the job at 8 GB or more (12 to 16 GB leaves room).
Cyclobutadiene's spin-flip block has 4538 pairs (15 x 163 + 13 x 161, 176 basis functions), 165 MB
a dense matrix: the split holds about 22 of them at its peak (A, B, their reduced copies, the
coordinates and momenta, and the non-symmetric eigenproblem and Schur form of (A - B)(A + B) in
`krein_pairs`), about 3.6 GB, and about 13 while the two-mode GCM runs; each UHF solution keeps
pyscf's in-core AO integrals, nao^4 bytes (0.96 GB at nao 176), and both candidate solutions are
alive, about 1.9 GB; the GCM's active space of about 41 + 43 spin orbitals holds about 0.2 GB of
kernels and some 0.5 GB of temporaries; the D2h record (0.33 GB) is still alive during D4h. A
scheduler's out-of-memory kill is not an exception that `guarded` can catch, and cyclobutadiene
runs first. The core holes' blocks have about 1450 pairs.
Run: python tests/test_soft_mode_open_shell_full.py, or under pytest (estimated 30 to 60 minutes on
16 cores, dominated by the spin-flip eigenproblems and two-mode GCMs of cyclobutadiene; not timed).
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyscf import gto, scf
from scipy.linalg import expm

from src.Base.numerics.constants import HARTREE_TO_EV, SOFT_MODE_KREIN_IMAG_TOL
from src.SingleReference.LinearResponse import soft_mode as SM
from src.SingleReference.LinearResponse.casida import CasidaInstability

#: eV. The paper's SF-TDA-RPAx and SF-RPAx energies, printed to 1e-3 eV.
TDA_TOL = 0.01
RPAX_TOL = 0.01
#: eV. Delta-UHF against the paper's, printed to 0.01 eV.
DELTA_TOL = 0.05
#: eV. The g/u splitting of N2+'s 1s hole by the non-orthogonal CI: 0.041 eV in cc-pVDZ.
NOCI_SPLIT_MAX = 0.2
#: Internal UHF instabilities followed for the second candidate solution.
SCF_FOLLOW = 3
#: Rotation (in |kappa|) along the CVS Hessian's lowest eigenvector toward BF+'s second solution
#: (0.3 reached it in aug-cc-pVDZ).
STEP = 0.3

CYCLOBUTADIENE = {'D2h': (0.783, 0.6715, 1.54123883, 1.43212465),
                  'D4h': (0.7195, 0.7195, 1.47822558, 1.47822558)}
SF_TDA = {'D2h': (1.523, 3.845, 5.508), 'D4h': (0.318, 3.131, 2.657)}
SF_RPAX = {'D2h': (3.556, 2.552, (5.942, 7.495)), 'D4h': (3.157, 2.084, (6.156, 5.562))}
EOM_SF = {'D2h': 1.516, 'D4h': 0.163}
SF_BSE = {'D2h': 1.678, 'D4h': 0.224}
N2_DELTA = {'localized': 409.42, 'symmetric': 419.25}
N2_CVS_FCI = 409.88
BF_DELTA = (694.32, 694.20)


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''),
          flush=True)
    return bool(ok)


def aug_cc_pcvtz(*elements):
    """aug-cc-pCVTZ as a pyscf basis dict: cc-pCVTZ plus the diffuse shells of aug-cc-pVTZ (the
    shells aug-cc-pVTZ has beyond cc-pVTZ, one s, p, d and f for B, N and F), which is how the
    augmented core-valence sets are made. pyscf does not ship aug-cc-pCVTZ, and loads it from
    basis-set-exchange only when that package is installed."""
    out = {}
    for element in elements:
        valence = gto.basis.load('cc-pvtz', element)
        diffuse = [shell for shell in gto.basis.load('aug-cc-pvtz', element)
                   if shell not in valence]
        out[element] = gto.basis.load('cc-pcvtz', element) + diffuse
    return out


def guarded(check_system, label):
    """check_system() for one system, an exception reported as its failure, so that the systems
    after it still run."""
    try:
        return check_system()
    except Exception as exc:  # noqa: BLE001 -- reported as a failure of this system
        print(f'  [FAIL] {label}: {type(exc).__name__}: {exc}', flush=True)
        return False


def uhf_solutions(mol):
    """[the UHF from pyscf's guess, and the one reached by following its internal instabilities up
    to SCF_FOLLOW times, if that is another (lower) solution]."""
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 300
    mf.kernel()
    out = [mf]
    followed = scf.UHF(mol)
    followed.conv_tol = 1e-10
    followed.max_cycle = 300
    followed.kernel(dm0=mf.make_rdm1())
    for _ in range(SCF_FOLLOW):
        mo, _, stable, _ = followed.stability(return_status=True)
        if stable:
            break
        followed.kernel(dm0=followed.make_rdm1(mo, followed.mo_occ))
    if followed.e_tot < mf.e_tot - 1e-8:
        out.append(followed)
    return out


def mom(mol, C, occ):
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 300
    mf = scf.addons.mom_occ(mf, C, occ)
    mf.kernel(dm0=mf.make_rdm1(C, occ))
    return mf


def complex_count(A, B):
    lam = np.linalg.eigvals((A - B) @ (A + B))
    return int(np.count_nonzero(np.abs(lam.imag) > SOFT_MODE_KREIN_IMAG_TOL * np.max(np.abs(lam))))


def reduced(ref, block):
    """The block's (A, B), with the spin rotation taken out of a spin-flip block."""
    A, B = SM.uhf_tdhf_matrices(ref, block)
    if block == 'spin_flip':
        n_alpha, n_beta = (int(np.count_nonzero(o)) for o in ref.mo_occ)
        g = SM.spin_rotation_direction(ref)
        A, B = SM._spin_rotation_split(A, B, g, n_alpha * (ref.mo_coeff.shape[2] - n_beta))[:2]
    return A, B


def print_split(mf, label):
    for block in SM.BLOCKS:
        t0 = time.time()
        try:
            rec = SM.soft_mode_open_shell(mf, block, 'gcm', collisions='gcm')
        except Exception as exc:  # noqa: BLE001 -- printed, not held
            print(f'    {label} {block}: {type(exc).__name__}: {str(exc)[:100]}', flush=True)
            continue
        soft = [f"{s['excitation'] * HARTREE_TO_EV:+.3f} eV (M {s['mass']:+.2f})"
                for s in rec['soft']]
        pairs = [f"{p['kind']}: {p['deexcitation']} / {p['excitation']} Ha" for p in rec['pairs']]
        flags = [f['flag'] for f in rec['numerical_flags']]
        print(f'    {label} {block}: soft {soft}, pairs {pairs}, flags {flags}, '
              f'{time.time() - t0:.0f} s', flush=True)


def paper_match(mf, label):
    """(TDA miss, quartet (eV), SF-RPAx rows miss) of one UHF against the paper's Tables 1, 2, S1
    and S2."""
    ref = SM.open_shell_reference(mf)[0]
    A, B = SM.uhf_tdhf_matrices(ref, 'spin_flip')
    n_alpha, n_beta = (int(np.count_nonzero(o)) for o in ref.mo_occ)
    sector = n_alpha * (ref.mo_coeff.shape[2] - n_beta)
    tda = np.linalg.eigvalsh(A[:sector, :sector]) * HARTREE_TO_EV
    tda_miss = max(float(np.min(np.abs(tda - tda[0] - e))) for e in SF_TDA[label])
    Ar, Br = reduced(ref, 'spin_flip')
    lam, vec = np.linalg.eig((Ar - Br) @ (Ar + Br))
    bad = np.abs(lam.imag) > SOFT_MODE_KREIN_IMAG_TOL * np.max(np.abs(lam))
    quartet = np.sqrt(lam[bad].astype(complex)) * HARTREE_TO_EV
    re, im, rows = SF_RPAX[label]
    omega = np.sqrt(lam[~bad].real.astype(complex)) * HARTREE_TO_EV
    real = np.abs(omega[np.isreal(omega)].real)
    rows_miss = (max(float(np.min(np.abs(real - (e - re)))) for e in rows) if real.size
                 else np.inf)
    hit = (bad.sum() == 2 and abs(abs(quartet[0].real) - re) < RPAX_TOL
           and abs(abs(quartet[0].imag) - im) < RPAX_TOL)
    return tda_miss, quartet, rows_miss, hit


def check_cyclobutadiene():
    ok = True
    for label, (cx, cy, hx, hy) in CYCLOBUTADIENE.items():
        t0 = time.time()
        atoms = [f'C {sx * cx} {sy * cy} 0' for sx in (-1, 1) for sy in (-1, 1)]
        atoms += [f'H {sx * hx} {sy * hy} 0' for sx in (-1, 1) for sy in (-1, 1)]
        mol = gto.M(atom='; '.join(atoms), basis='cc-pvtz', spin=2, verbose=0)
        chosen, found, solutions = None, [], uhf_solutions(mol)
        for k, mf in enumerate(solutions):
            tda_miss, quartet, rows_miss, hit = paper_match(mf, label)
            match = tda_miss < TDA_TOL and hit and rows_miss < RPAX_TOL
            found.append(match)
            print(f"    {label} UHF {'from the guess' if k == 0 else 'followed'}: "
                  f'{mf.e_tot:.8f} Ha, <S^2> {mf.spin_square()[0]:.4f}; SF-TDA miss '
                  f'{tda_miss:.4f} eV, quartet {np.round(quartet, 3).tolist()} eV, SF-RPAx rows '
                  f'miss {rows_miss:.4f} eV', flush=True)
            if chosen is None and match:
                chosen = mf
        ok &= check(any(found), f"{label}: a UHF solution reproduces the paper's SF-TDA-RPAx "
                    'energies, its SF-RPAx quartet (Omega_X) and its other SF-RPAx rows',
                    f'{time.time() - t0:.0f} s')
        mf = chosen if chosen is not None else solutions[0]
        try:
            SM.soft_mode_open_shell(mf, 'spin_flip', 'gcm')
            refused = False
        except CasidaInstability:
            refused = True
        t1 = time.time()
        try:
            rec = SM.soft_mode_open_shell(mf, 'spin_flip', 'gcm', collisions='gcm')
        except (CasidaInstability, NotImplementedError) as exc:
            ok &= check(False, f"{label}: collisions='gcm' gives real poles",
                        f'refused: {type(exc).__name__}: {str(exc)[:200]}')
            continue
        pair = rec['pairs'][0] if rec['pairs'] else None
        real = bool(np.all(np.isfinite(rec['omega'])) and np.all(np.isreal(rec['omega'])))
        down = None if pair is None or pair['deexcitation'] is None else \
            pair['deexcitation'] * HARTREE_TO_EV
        ok &= check(refused and real and down is not None and down < 0,
                    f"{label}: the default refuses; collisions='gcm' gives real poles, the singlet "
                    'below the triplet',
                    f'triplet {-down if down is not None else None} eV above the singlet '
                    f'(EOM-SF-CCSD(fT) {EOM_SF[label]}, SF-TDA-RPAx {SF_TDA[label][0]}, SF-BSE '
                    f'{SF_BSE[label]}); overlap {pair["reference_overlap"] if pair else None}, '
                    f'flags {[f["flag"] for f in rec["numerical_flags"]]}, '
                    f'{time.time() - t1:.0f} s')
    return ok


def check_n2():
    ok = True
    atom = 'N 0 0 0; N 0 0 1.1007'
    basis = aug_cc_pcvtz('N')
    neutral = scf.RHF(gto.M(atom=atom, basis=basis, verbose=0)).run(conv_tol=1e-10)
    mol = gto.M(atom=atom, basis=basis, charge=1, spin=1, verbose=0)
    C = neutral.mo_coeff
    occ = np.zeros(C.shape[1])
    occ[:7] = 1
    solutions = {}
    for label, sign in (('localized', 1.0), ('mirror', -1.0), ('symmetric', 0.0)):
        guess = C.copy()
        if sign:
            guess[:, 0] = (C[:, 0] + sign * C[:, 1]) / np.sqrt(2.0)
            guess[:, 1] = (C[:, 0] - sign * C[:, 1]) / np.sqrt(2.0)
        hole = occ.copy()
        hole[0] = 0
        solutions[label] = mom(mol, (guess, guess), (occ, hole))
    for label in ('localized', 'symmetric'):
        ip = (solutions[label].e_tot - neutral.e_tot) * HARTREE_TO_EV
        ok &= check(abs(ip - N2_DELTA[label]) < DELTA_TOL, f'N2+ {label} hole: Delta-UHF',
                    f'{ip:.3f} eV (paper {N2_DELTA[label]})')
    loc = SM.open_shell_reference(solutions['localized'])[0]
    loc_complex = [complex_count(*reduced(loc, block)) for block in SM.BLOCKS]
    sym = SM.open_shell_reference(solutions['symmetric'])[0]
    sc = complex_count(*reduced(sym, 'spin_conserving'))
    ok &= check(loc_complex == [0, 0] and complex_count(*reduced(sym, 'spin_flip')) > 0,
                'N2+: the localized hole\'s blocks real, the symmetric hole\'s spin-flip block '
                'complex', f'localized complex {loc_complex}, symmetric spin-conserving complex: '
                f'{sc}')
    noci = SM.noci_levels([solutions['localized'], solutions['mirror']])
    ips = (noci['energies'] - neutral.e_tot) * HARTREE_TO_EV
    loc_ip = (solutions['localized'].e_tot - neutral.e_tot) * HARTREE_TO_EV
    ok &= check(noci['rank'] == 2 and ips[1] - ips[0] < NOCI_SPLIT_MAX and ips[0] < loc_ip,
                'N2+: the non-orthogonal CI of the hole on either atom, g and u',
                f'{np.round(ips, 3).tolist()} eV (Delta-CVS-FCI {N2_CVS_FCI}), overlap '
                f"{noci['overlap'][0, 1]:.2e}")
    print_split(solutions['localized'], 'N2+ localized')
    return ok


def check_bf():
    ok = True
    atom = 'B 0 0 0; F 0 0 1.2685'
    basis = aug_cc_pcvtz('B', 'F')
    neutral = scf.RHF(gto.M(atom=atom, basis=basis, verbose=0)).run(conv_tol=1e-10)
    mol = gto.M(atom=atom, basis=basis, charge=1, spin=1, verbose=0)
    C = neutral.mo_coeff
    occ = np.zeros(C.shape[1])
    occ[:7] = 1
    hole = occ.copy()
    hole[0] = 0
    first = mom(mol, (C, C), (occ, hole))
    ref, orders = SM.open_shell_reference(first)
    ip = (first.e_tot - neutral.e_tot) * HARTREE_TO_EV
    A, B = SM.uhf_tdhf_matrices(ref, 'spin_conserving')
    ok &= check(abs(ip - BF_DELTA[0]) < DELTA_TOL and complex_count(A, B) > 0,
                'BF+ first solution: Delta-UHF, spin-conserving block complex',
                f'{ip:.3f} eV (paper {BF_DELTA[0]}), {complex_count(A, B)} complex')
    n_alpha, n_beta = 7, 6
    nmo = ref.mo_coeff.shape[2]
    hole_at = list(orders[1]).index(0) - n_beta
    keep = np.ones(A.shape[0], dtype=bool)
    for i in range(n_beta):
        keep[n_alpha * (nmo - n_alpha) + i * (nmo - n_beta) + hole_at] = False
    w, v = np.linalg.eigh((A + B)[np.ix_(keep, keep)])
    kappa = np.zeros(A.shape[0])
    kappa[keep] = v[:, 0]
    pairs = SM._open_shell_pairs(ref, 'spin_conserving', kappa)
    turned = []
    for s, n in ((0, n_alpha), (1, n_beta)):
        K = np.zeros((nmo, nmo))
        K[n:, :n] = STEP * pairs[s, s].T
        turned.append(ref.mo_coeff[s] @ expm(K - K.T))
    second = mom(mol, tuple(turned), tuple(ref.mo_occ))
    ref2 = SM.open_shell_reference(second)[0]
    ip2 = (second.e_tot - neutral.e_tot) * HARTREE_TO_EV
    sc = complex_count(*reduced(ref2, 'spin_conserving'))
    A2, B2 = reduced(ref2, 'spin_flip')
    unstable = complex_count(A2, B2) > 0 or np.linalg.eigvalsh(A2 + B2)[0] < 0
    ok &= check(w[0] < 0 and abs(ip2 - BF_DELTA[1]) < DELTA_TOL and sc == 0 and unstable,
                'BF+ second solution along the CVS Hessian: Delta-UHF, spin-conserving real, '
                'spin-flip unstable', f'CVS lowest {w[0]:+.4f} Ha, {ip2:.3f} eV (paper '
                f'{BF_DELTA[1]}), spin-flip complex {complex_count(A2, B2)}')
    noci = SM.noci_levels([first, second])
    ips = np.round((noci['energies'] - neutral.e_tot) * HARTREE_TO_EV, 3).tolist()
    print(f"    BF+ NOCI of the two solutions: {ips} eV, overlap {noci['overlap'][0, 1]:+.4f}",
          flush=True)
    print_split(second, 'BF+ second')
    return ok


def run():
    warnings.simplefilter('ignore')
    ok = True
    print('\n-- 1. cyclobutadiene, cc-pVTZ: spin-flip TDHF on the high-spin triplet', flush=True)
    ok &= guarded(check_cyclobutadiene, 'cyclobutadiene')
    print('\n-- 2. N2+ 1s hole, aug-cc-pCVTZ', flush=True)
    ok &= guarded(check_n2, 'N2+')
    print('\n-- 3. BF+ F1s hole, aug-cc-pCVTZ', flush=True)
    ok &= guarded(check_bf, 'BF+')
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_soft_mode_open_shell_full():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
