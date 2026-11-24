"""The open-shell soft-mode split (`LinearResponse.soft_mode.soft_mode_open_shell`): the spin-
conserving and spin-flip TDHF blocks of a UHF reference, their soft modes on collinear (UHF) and
non-collinear (GHF) paths, their Krein collisions by the two-mode GCM, the spin rotation taken out
of the spin-flip block, and the non-orthogonal CI of broken-symmetry solutions (`noci_levels`):
what is held.

1. The blocks (`uhf_tdhf_matrices`): the spin-conserving one gives pyscf's TDHF roots on a UHF
   (the H2O triplet in 6-31G, to MATRIX_TOL), and both are the matrices of
   `LinearResponse.linear_response.LinearResponseSolver`'s unrestricted builders
   (build_casida_matrices with lBSE=True, build_spin_flip_casida_matrices); on a closed shell
   (H2O/6-31G through scf.addons.convert_to_uhf) the spin-flip block holds the TDHF triplet
   Omega^2 twice and the spin-conserving one the singlet's and the triplet's (to SPECTRUM_TOL);
   D (A + B) D = A - B in the spin-flip block.
2. The reference (`open_shell_reference`): a maximum-overlap core hole (H2O+ O1s in 6-31G, beta
   hole) is reordered occupied-first in each spin, its orders map back, and it is stationary;
   the same occupation set without reconverging is refused.
3. The kernels (`_OpenShellKernels`): N, H, a random one-body operator and S^2 between random GHF
   determinants (every spin orbital turned by a random real rotation of all 2n spin orbitals, so
   that M_s is mixed) of the H4/STO-3G triplet and of He2+/6-31G against the exact Fock-space
   vectors of the determinants (coefficients the minors of their spin-orbital matrices, H and S^2
   from second quantization), to KERNEL_TOL; the per-spin active space of a spin-flip and a
   spin-conserving plane (H4/6-31G triplet) against the complete basis; the reference
   determinant's energy and <S^2> are the UHF's.
4. The paths (`open_shell_path`): V(0) = 0, V''(0) = d^T (A + B) d and D'(0) = 1 along random
   directions of both blocks (H4/6-31G triplet, central differences, to PATH_TOL); on a closed
   shell the spin-conserving triplet direction (d, -d) / sqrt2 and the spin-flip direction
   (d, d) / sqrt2 give `path_samples`' V and D (a spin rotation takes one path to the other).
5. The closed-shell limit: soft_mode_open_shell on H2/cc-pVDZ at 1.0 and 1.5 A converted to a UHF
   (spin-conserving block) gives soft_mode_triplets' levels, 1D (to LIMIT_1D_TOL eV, the spline of
   two path samplings) and GCM (to LIMIT_GCM_TOL eV), and its couplings.
6. The spin rotation (`spin_rotation_direction`, `_spin_rotation_split`): (A + B) g and
   (A - B) D g vanish to the SCF's convergence (below SPIN_RESIDUAL_MAX relative), g^T D g =
   n_alpha - n_beta; the reduced Omega^2 are the block's without the zero mode (He2+ at 2.0 A,
   to REDUCED_TOL, the size of the residual of g);
   the driver's spin-rotation pole sits at omega = 0 with X'X - Y'Y = 1 and solves the RPA
   equations, every regular column solves them with X'X - Y'Y = 1 (to RPA_TOL); the harmonic pole
   of a soft spin-flip mode built through the reduced split is the dense RPA eigenpair of the full
   block (He2+ at 1.08 A, to POLE_TOL).
7. He2+/cc-pVDZ (sigma_g^2 sigma_u UHF, three electrons) through its hole-localization
   instability, against FCI's 2Sigma_g+ - 2Sigma_u+ gap: TDHF turns imaginary from 1.75 A on; the
   spin-conserving soft mode's GCM level is within GCM_TOL of FCI on every row (1.08 to 4.0 A), the
   1D level within ONE_D_TOL from 1.75 A on (the stiff mode at 1.08 to 1.5 A is 0.6 to 1.3 eV high,
   flagged 'soft_mode_edge_weight', printed); the non-orthogonal CI of the two localized UHF
   solutions, where they exist (1.75 A on), is within NOCI_TOL of FCI and closer than the GCM; at
   1.5 A the localized guesses fall back to the symmetric solution and the CI of the two is
   flagged 'soft_mode_noci_rank' (SoftModeNOCIShort). In STO-3G (1.08, 2.0 and 3.0 A) the soft
   direction turns one beta spin orbital, whose path spans exactly one even and one odd state:
   the GCM keeps both (ranks (1, 1)), equals FCI's gap to EXACT_TOL and raises no
   'soft_mode_gcm_few_states'.
8. Square and rectangular H4/6-31G on their high-spin triplet UHF, the four-electron model of
   cyclobutadiene's spin-flip problem: full spin-flip TDHF is complex (a Krein quartet of the
   negative transition to the singlet and the positive one to the quintet), refused by default;
   collisions='gcm' gives real poles: the de-excitation to the singlet within SINGLET_TOL of FCI's
   singlet-triplet gap (too small: the model's singlet is spin-contaminated), the excitation to the
   quintet within QUINTET_TOL; the stretched square (2.0 A) flagged
   'soft_mode_collision_reference' (overlap 0.65), no flag on the others. SF-TDA is printed.
9. Refusals: an unknown block, model '3d' or collisions option; RHF, ROHF and GHF references, a
   functional, density fitting, an unconverged SCF, n_beta > n_alpha; the spin-flip block with
   n_alpha = n_beta, before any Fock build (its get_veff never called); a spin-conserving
   collision plane no symmetry takes to minus itself (two totally symmetric directions of H2O+,
   NotImplementedError); noci_levels with one solution or two molecules, and over two bases with
   the same AO labels (He2+ in 6-31G and 3-21G), before any work. The unknown block and the one
   solution are refused on SCF objects not yet run, by their messages ("block = 'triplet'", 'needs
   two or more'), so that a refusal that came after the reference's checks would show as
   'mf.converged = False' instead; so is the mixed-basis case.
10. A complex instability of a ground-state determinant: O2/STO-3G's default triplet UHF at
   1.21 A (aufbau in both spins) has one spin-conserving soft mode with M < 0 and mu > 0 (M =
   -36.0, mu = +0.330; A + B indefinite too), which the inversion of a negative mass, validated on
   excited references, does not describe (FCI has no state below the reference): in 1D and in the
   GCM it raises 'soft_mode_complex_instability' once (SoftModeExcitedShort), not for its modes of
   positive mass.

FCI is pyscf's on the reference's orbitals; the bars hold measured misses.
Run: python tests/test_soft_mode_open_shell.py, or under pytest.
"""
import itertools
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.linalg
from pyscf import ao2mo, dft, fci, gto, scf, tdscf

from src.Base.numerics.constants import HARTREE_TO_EV
from src.SingleReference.LinearResponse import soft_mode as SM
from src.SingleReference.LinearResponse.casida import CasidaInstability
from src.SingleReference.LinearResponse.linear_response import LinearResponseSolver

#: Ha. The spin-conserving block's roots against pyscf's TDHF (2.2e-13 measured) and both blocks
#: against the house's unrestricted builders (3.6e-15).
MATRIX_TOL = 1e-8
#: Ha^2. A closed shell's blocks against its triplet and singlet Omega^2: 1.6e-12 measured.
SPECTRUM_TOL = 1e-10
#: Ha. The kernels against exact Fock-space vectors and the complete basis: 1.6e-14 measured (the
#: active-space check).
KERNEL_TOL = 1e-11
#: Relative. V''(0) = mu and D'(0) = 1 by central differences at Q = 1e-3: 1.1e-7 measured.
PATH_TOL = 1e-5
#: eV. The closed-shell limit: 2.6e-10 (1D, the splines of two samplings) and 1.2e-14 (GCM)
#: measured at 1.0 A (2.4e-11 and 3.8e-15 at 1.5 A).
LIMIT_1D_TOL = 1e-8
LIMIT_GCM_TOL = 1e-10
#: Ha^2. The reduced Omega^2 against the block's: 3.5e-10 measured on He2+ at 2.0 A, the size of
#: (A + B) g there (4e-10 relative), which the split's exactness rests on.
REDUCED_TOL = 1e-8
#: |(A + B) g| / |g| relative to the largest |A + B| eigenvalue: 1e-9 to 2e-8 at conv_tol 1e-12.
SPIN_RESIDUAL_MAX = 1e-6
#: The RPA equations of the regular and spin-rotation columns: 1e-8 measured (H4/6-31G triplet;
#: 9.4e-10 on He2+, the residual of g).
RPA_TOL = 1e-6
#: The harmonic soft pole against the dense RPA eigenpair: 1e-12 measured.
POLE_TOL = 1e-8
#: eV. He2+ against FCI: GCM +0.08 to +0.27, 1D -0.03 to +0.38 from 1.75 A, NOCI +0.000 to +0.13.
GCM_TOL = 0.3
ONE_D_TOL = 0.45
NOCI_TOL = 0.15
#: eV. H4 against FCI: the singlet de-excitation +0.26 to +0.72, the quintet -0.02 to -0.07.
SINGLET_TOL = 0.8
QUINTET_TOL = 0.1
#: eV. He2+/STO-3G's GCM (one spin-orbital pair, its exact span) against FCI: 2e-14 measured.
EXACT_TOL = 1e-8

HE2_ROWS = (1.08, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
HE2_MINIMAL_ROWS = (1.08, 2.0, 3.0)
H4_ROWS = ((1.5, 1.5), (1.4, 1.6), (1.3, 1.7), (2.0, 2.0))


def check(ok, label, detail=''):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f'   ({detail})' if detail else ''))
    return bool(ok)


def uhf(mol, conv_tol=1e-12):
    mf = scf.UHF(mol)
    mf.conv_tol = conv_tol
    mf.max_cycle = 200
    mf.kernel()
    return mf


def he2(R, basis='cc-pvdz', symmetry=True):
    """He2+ on its symmetric (sigma_g^2 sigma_u) UHF."""
    return uhf(gto.M(atom=f'He 0 0 0; He 0 0 {R}', basis=basis, charge=1, spin=1, verbose=0,
                     symmetry=symmetry))


def he2_localized(R, sign):
    """He2+'s UHF from a guess with the beta electron on one atom, (sigma_g +- sigma_u) / sqrt2."""
    sym = he2(R, symmetry=False)
    Ca, Cb = sym.mo_coeff
    cb = (Cb[:, 0] + sign * Cb[:, 1]) / np.sqrt(2.0)
    mf = scf.UHF(sym.mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.kernel(dm0=(Ca[:, :2] @ Ca[:, :2].T, np.outer(cb, cb)))
    return mf


def he2_fci_gap(mf):
    """FCI's 2Sigma_g+ - 2Sigma_u+ gap (Ha) of He2+ on mf's alpha orbitals (symmetry-adapted)."""
    energies = {}
    for irrep in ('A1u', 'A1g'):
        cis = fci.FCI(mf.mol, mf.mo_coeff[0])
        cis.wfnsym = irrep
        cis.conv_tol = 1e-12
        energies[irrep] = cis.kernel(nelec=(2, 1))[0]
    return energies['A1g'] - energies['A1u']


def h4(a, b, basis, spin):
    return gto.M(atom=f'H 0 0 0; H {a} 0 0; H {a} {b} 0; H 0 {b} 0', basis=basis, spin=spin,
                 verbose=0)


def h2o(basis, charge=0, spin=0):
    return gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis=basis, charge=charge,
                 spin=spin, verbose=0)


def rpa_omega2(A, B):
    lam = np.linalg.eigvals((A - B) @ (A + B))
    return lam[np.lexsort((lam.imag, lam.real))]


# ------------------------------------------------------------------ exact Fock-space reference
def fock_space(nso, nel):
    """(determinants, E): the sorted occupation tuples of nel electrons in nso spin orbitals and
    E[p, q] = <D'|a+_p a_q|D> (a+_{d1} ... a+_{dN}|0> ascending: the sign of each operator is the
    parity of the occupied spin orbitals before it)."""
    dets = list(itertools.combinations(range(nso), nel))
    index = {d: i for i, d in enumerate(dets)}
    E = np.zeros((nso, nso, len(dets), len(dets)))
    for j, det in enumerate(dets):
        for q in det:
            sign_q = (-1) ** det.index(q)
            rest = [o for o in det if o != q]
            for p in range(nso):
                if p in rest:
                    continue
                sign_p = (-1) ** sum(1 for o in rest if o < p)
                E[p, q, index[tuple(sorted(rest + [p]))], j] = sign_q * sign_p
    return dets, E


def exact_operators(h, eri, nso_half, E):
    """(H, S^2) in the determinant basis: h (n x n) and eri (n^4, chemists') on the spatial basis
    shared by both spins (spin orbital sigma * n + mu)."""
    n = nso_half
    hs = np.kron(np.eye(2), h)
    g = np.zeros((2 * n,) * 4)
    for s in (0, 1):
        for t in (0, 1):
            g[s * n:(s + 1) * n, s * n:(s + 1) * n, t * n:(t + 1) * n, t * n:(t + 1) * n] = eri
    one = np.einsum('pq,pqij->ij', hs, E)
    two = 0.5 * np.einsum('prqs,prij,qsjk->ik', g, E, E, optimize=True)
    two -= 0.5 * np.einsum('pqqs,psij->ij', g, E)
    up = sum(E[mu, n + mu] for mu in range(n))
    z = 0.5 * (sum(E[mu, mu] for mu in range(n)) - sum(E[n + mu, n + mu] for mu in range(n)))
    return one + two, up.T @ up + z @ z + z


def check_matrices():
    ok = True
    mf = uhf(h2o('6-31g', spin=2))
    ref = SM.open_shell_reference(mf)[0]
    A, B = SM.uhf_tdhf_matrices(ref, 'spin_conserving')
    omega = np.sqrt(np.sort(rpa_omega2(A, B).real)[:6])
    td = tdscf.TDHF(mf)
    td.nstates = 6
    td.conv_tol = 1e-12
    td.kernel()
    err = float(np.max(np.abs(omega - td.e[:6])))
    ok &= check(err < MATRIX_TOL, 'spin-conserving block = pyscf TDHF on a UHF (H2O triplet)',
                f'{err:.1e} Ha')
    (Ca, Cb), mol = ref.mo_coeff, ref.mol
    eri = tuple(ao2mo.general(mol, c, compact=False).reshape((Ca.shape[1],) * 4)
                for c in ((Ca,) * 4, (Cb,) * 4, (Ca, Ca, Cb, Cb)))
    house = LinearResponseSolver(tuple(ref.mo_energy), eri_chemist=eri, spin_mode='unrestricted')
    nocc = tuple(int(np.count_nonzero(o)) for o in ref.mo_occ)
    worst = 0.0
    for block, built in (('spin_conserving', house.build_casida_matrices(nocc, lBSE=True)),
                         ('spin_flip', house.build_spin_flip_casida_matrices(nocc, lBSE=True))):
        mine = SM.uhf_tdhf_matrices(ref, block)
        worst = max(worst, max(float(np.max(np.abs(x - y))) for x, y in zip(mine, built)))
    ok &= check(worst < MATRIX_TOL, "both blocks = LinearResponseSolver's unrestricted builders",
                f'{worst:.1e} Ha')
    rhf = scf.RHF(h2o('6-31g')).run(conv_tol=1e-12)
    ref = SM.open_shell_reference(scf.addons.convert_to_uhf(rhf))[0]
    triplet = np.sort(rpa_omega2(*SM.triplet_tdhf_matrices(rhf)).real)
    singlet = np.sort(rpa_omega2(*SM.singlet_tdhf_matrices(rhf)).real)
    A, B = SM.uhf_tdhf_matrices(ref, 'spin_flip')
    err_sf = float(np.max(np.abs(np.sort(rpa_omega2(A, B).real)
                                 - np.sort(np.r_[triplet, triplet]))))
    n1 = A.shape[0] // 2
    D = np.r_[np.ones(n1), -np.ones(A.shape[0] - n1)]
    err_d = float(np.max(np.abs(D[:, None] * (A + B) * D[None, :] - (A - B))))
    A, B = SM.uhf_tdhf_matrices(ref, 'spin_conserving')
    err_sc = float(np.max(np.abs(np.sort(rpa_omega2(A, B).real)
                                 - np.sort(np.r_[triplet, singlet]))))
    ok &= check(err_sf < SPECTRUM_TOL and err_sc < SPECTRUM_TOL,
                'closed shell: spin flip = triplet twice, spin conserving = singlet + triplet',
                f'{err_sf:.1e}, {err_sc:.1e} Ha^2')
    ok &= check(err_d < 1e-14, 'spin-flip block: D (A + B) D = A - B', f'{err_d:.1e}')
    return ok


def check_reference():
    ok = True
    neutral = scf.RHF(h2o('6-31g')).run(conv_tol=1e-11)
    C = neutral.mo_coeff
    occa = np.zeros(C.shape[1])
    occa[:5] = 1
    occb = occa.copy()
    occb[0] = 0
    mf = scf.UHF(h2o('6-31g', charge=1, spin=1))
    mf.conv_tol = 1e-11
    mf.max_cycle = 300
    mf = scf.addons.mom_occ(mf, (C, C), (occa, occb))
    mf.kernel(dm0=mf.make_rdm1((C, C), (occa, occb)))
    ref, orders = SM.open_shell_reference(mf)
    hole = int(np.flatnonzero(np.asarray(mf.mo_occ[1]) == 0)[0])
    first = all(np.all(o[:n] == 1) and np.all(o[n:] == 0)
                for o, n in zip(ref.mo_occ, (5, 4)))
    mapped = all(np.allclose(ref.mo_coeff[s], np.asarray(mf.mo_coeff[s])[:, orders[s]])
                 for s in (0, 1))
    ok &= check(mf.converged and first and mapped and orders[1][4] == hole,
                'a maximum-overlap O1s hole of H2O+ reordered occupied-first, orders map back',
                f'IP {(mf.e_tot - neutral.e_tot) * HARTREE_TO_EV:.2f} eV, beta hole at '
                f'orbital {hole} -> pair position {list(orders[1]).index(hole)}')
    swapped = scf.UHF(h2o('6-31g', charge=1, spin=1)).run(conv_tol=1e-11)
    occ = np.array(swapped.mo_occ)
    occ[1][[3, 4]] = occ[1][[4, 3]]
    swapped.mo_occ = occ
    try:
        SM.open_shell_reference(swapped)
        raised = None
    except ValueError as exc:
        raised = exc
    ok &= check(raised is not None, 'an occupation set without reconverging is refused',
                str(raised)[:60])
    return ok


def check_kernels():
    ok = True
    rng = np.random.default_rng(7)
    for label, mol in (('H4/STO-3G triplet', h4(1.5, 1.5, 'sto-3g', 2)),
                       ('He2+/6-31G', gto.M(atom='He 0 0 0; He 0 0 1.6', basis='6-31g',
                                            charge=1, spin=1, verbose=0))):
        mf = uhf(mol)
        n = mol.nao
        S = mol.intor('int1e_ovlp')
        W = mf.mo_coeff[0]
        na, nb = mol.nelec
        kernels = SM._OpenShellKernels(mol, (W, W), mf.get_hcore(), 0.0)
        C0 = np.zeros((2 * n, na + nb))
        C0[:n, :na] = W.T @ S @ mf.mo_coeff[0][:, :na]
        C0[n:, na:] = W.T @ S @ mf.mo_coeff[1][:, :nb]
        op = rng.standard_normal((2 * n, 2 * n))
        op = op + op.T
        dets, E = fock_space(2 * n, na + nb)
        eri = ao2mo.restore(1, ao2mo.kernel(mol, W), n)
        Hx, S2x = exact_operators(W.T @ mf.get_hcore() @ W, eri, n, E)
        Hx += mol.energy_nuc() * np.eye(len(dets))
        Ox = np.einsum('pq,pqij->ij', op, E)
        worst = 0.0
        for _ in range(6):
            pair = []
            for _ in range(2):
                K = rng.standard_normal((2 * n, 2 * n)) * 0.6
                pair.append(scipy.linalg.expm(K - K.T) @ C0)
            N, H, (O,), S2 = kernels.elements(pair[0][None], pair[1][None], [op])
            c = [np.array([np.linalg.det(D[list(d), :]) for d in dets]) for D in pair]
            worst = max(worst, abs(N[0] - c[0] @ c[1]), abs(H[0] - c[0] @ Hx @ c[1]),
                        abs(O[0] - c[0] @ Ox @ c[1]), abs(S2[0] - c[0] @ S2x @ c[1]))
        N, H, _, S2 = kernels.elements(C0[None], C0[None])
        ok &= check(worst < KERNEL_TOL and abs(H[0] - mf.e_tot) < KERNEL_TOL
                    and abs(S2[0] - mf.spin_square()[0]) < KERNEL_TOL,
                    f'kernels = exact Fock-space vectors at random GHF pairs ({label})',
                    f'{worst:.1e} Ha; reference E and <S^2> to '
                    f'{abs(H[0] - mf.e_tot):.0e}, {abs(S2[0] - mf.spin_square()[0]):.0e}')
    mf = uhf(h4(1.5, 1.5, '6-31g', 2))
    ref = SM.open_shell_reference(mf)[0]
    S = mf.mol.intor('int1e_ovlp')
    R = mf.mo_coeff[0]
    for block in SM.BLOCKS:
        size = SM.uhf_tdhf_matrices(ref, block)[0].shape[0]
        e1, e2 = np.linalg.qr(rng.standard_normal((size, 2)))[0].T
        space = SM._OpenShellSpace(ref, block, [e1, e2])
        gens = [space.generator(e) / np.sqrt(2.0) for e in (e1, e2)]
        bra = space.orbitals(gens, rng.standard_normal((5, 2)) * 1.5)
        ket = space.orbitals(gens, rng.standard_normal((5, 2)) * 1.5)
        op = space.operator(e1)
        N, H, (O,), S2 = space.kernels.elements(bra, ket, [op])
        ma, mb = space.kernels.m
        T = np.zeros((2 * R.shape[1], ma + mb))
        T[:R.shape[1], :ma] = R.T @ S @ space.bases[0]
        T[R.shape[1]:, ma:] = R.T @ S @ space.bases[1]
        full = SM._OpenShellKernels(mf.mol, (R, R), mf.get_hcore(), mf.e_tot)
        Nf, Hf, (Of,), S2f = full.elements(np.einsum('pq,eqn->epn', T, bra),
                                           np.einsum('pq,eqn->epn', T, ket), [T @ op @ T.T])
        worst = max(float(np.max(np.abs(x - y))) for x, y in ((N, Nf), (H, Hf), (O, Of), (S2, S2f)))
        ok &= check(worst < KERNEL_TOL, f'per-spin active space = complete basis ({block} plane, '
                    f'm = {space.kernels.m})', f'{worst:.1e}')
    return ok


def check_paths():
    ok = True
    rng = np.random.default_rng(11)
    ref = SM.open_shell_reference(uhf(h4(1.5, 1.5, '6-31g', 2)))[0]
    h = 1e-3
    for block in SM.BLOCKS:
        A, B = SM.uhf_tdhf_matrices(ref, block)
        d = rng.standard_normal(A.shape[0])
        d /= np.linalg.norm(d)
        space = SM._OpenShellSpace(ref, block, [d])
        orbs = space.orbitals([space.generator(d) / np.sqrt(2.0)], [[-h], [0.0], [h]])
        _, V, (D,), _ = space.kernels.elements(orbs, orbs, [space.operator(d)])
        mu = d @ (A + B) @ d
        curvature = (V[0] - 2 * V[1] + V[2]) / h ** 2
        slope = (D[2] - D[0]) / (2 * h)
        ok &= check(abs(V[1]) < 1e-12 and abs(curvature / mu - 1) < PATH_TOL
                    and abs(slope - 1) < PATH_TOL,
                    f"{block}: V(0) = 0, V''(0) = d^T (A + B) d, D'(0) = 1 (random d)",
                    f"V''/mu - 1 = {curvature / mu - 1:+.1e}, D' - 1 = {slope - 1:+.1e}")
    rhf = scf.RHF(gto.M(atom='H 0 0 0; H 0 0 1.5', basis='cc-pvdz', verbose=0)).run(conv_tol=1e-12)
    d = SM.soft_split(*SM.triplet_tdhf_matrices(rhf))['modes'][0]['direction']
    L = SM.path_period(d, 1)[0]
    q, V, D = SM.path_samples(rhf, d, L)
    ref = SM.open_shell_reference(scf.addons.convert_to_uhf(rhf))[0]
    for block, sign in (('spin_conserving', -1.0), ('spin_flip', 1.0)):
        so = np.r_[d, sign * d] / np.sqrt(2.0)
        Lo = SM._open_shell_half_period(ref, block, so)[0]
        qo, Vo, Do = SM.open_shell_path(ref, block, so, Lo)
        err = max(abs(Lo - L), float(np.max(np.abs(Vo - V))), float(np.max(np.abs(Do - D))))
        ok &= check(err < KERNEL_TOL, f'closed shell: the {block} triplet direction gives '
                    "path_samples' path", f'{err:.1e}')
    return ok


def check_closed_shell_limit():
    ok = True
    for R in (1.0, 1.5):
        rhf = scf.RHF(gto.M(atom=f'H 0 0 0; H 0 0 {R}', basis='cc-pvdz', verbose=0)).run(
            conv_tol=1e-12)
        mf = scf.addons.convert_to_uhf(rhf)
        for model, bar in (('1d', LIMIT_1D_TOL), ('gcm', LIMIT_GCM_TOL)):
            rec = SM.soft_mode_open_shell(mf, 'spin_conserving', model)
            closed = SM.soft_mode_triplets(rhf, model)
            a, b = rec['soft'][0], closed['soft'][0]
            err = abs(a['excitation'] - b['excitation']) * HARTREE_TO_EV
            extra = (abs(a['coupling'] - b['coupling']) if model == 'gcm'
                     else abs(a['mass'] - b['mass']) / b['mass'])
            ok &= check(err < bar and extra < 1e-9,
                        f'H2 {R} A as a UHF, {model}: spin-conserving soft mode = '
                        'soft_mode_triplets',
                        f'{a["excitation"] * HARTREE_TO_EV:.6f} eV, {err:.1e} eV off')
    return ok


def check_spin_rotation():
    ok = True
    for label, mf in (('He2+ 2.0 A', he2(2.0)),
                      ('H4/6-31G triplet', uhf(h4(1.5, 1.5, '6-31g', 2)))):
        ref = SM.open_shell_reference(mf)[0]
        na, nb = (int(np.count_nonzero(o)) for o in ref.mo_occ)
        A, B = SM.uhf_tdhf_matrices(ref, 'spin_flip')
        g = SM.spin_rotation_direction(ref)
        sectors = na * (ref.mo_coeff.shape[2] - nb)
        D = np.r_[np.ones(sectors), -np.ones(g.size - sectors)]
        scale = float(np.max(np.abs(np.linalg.eigvalsh(A + B))))
        res = max(np.linalg.norm((A + B) @ g), np.linalg.norm((A - B) @ (D * g))) / (
            np.linalg.norm(g) * scale)
        ok &= check(res < SPIN_RESIDUAL_MAX and abs(g @ (D * g) - (na - nb)) < 1e-10,
                    f'{label}: (A + B) g = (A - B) D g = 0, g^T D g = n_alpha - n_beta',
                    f'residual {res:.1e}, g^T D g = {g @ (D * g):.12f}')
        rec = SM.soft_mode_open_shell(mf, 'spin_flip', 'gcm', collisions='gcm')
        X, Y, omega = rec['X'], rec['Y'], rec['omega']
        col = rec['spin_rotation']['column']
        regular = np.arange(col + 1)
        resid = max(float(np.max(np.abs((A + B) @ (X + Y)[:, regular] - (X - Y)[:, regular]
                                        * omega[regular]))),
                    float(np.max(np.abs((A - B) @ (X - Y)[:, regular] - (X + Y)[:, regular]
                                        * omega[regular]))))
        norms = np.sum(X * X - Y * Y, axis=0)[regular]
        ok &= check(omega[col] == 0.0 and resid < RPA_TOL and np.allclose(norms, 1.0),
                    f'{label}: the regular columns and the spin-rotation pole solve the RPA '
                    'equations with X\'X - Y\'Y = 1', f'{resid:.1e}')
        if label.startswith('He2+'):
            full = np.sort(rpa_omega2(A, B).real)
            reduced = np.sort(np.real(rec['omega2']))
            drop = int(np.argmin(np.abs(full)))
            err = float(np.max(np.abs(np.delete(full, drop) - reduced)))
            ok &= check(err < REDUCED_TOL, f'{label}: the reduced Omega^2 = the block\'s without '
                        'the zero mode', f'{err:.1e} Ha^2, zero mode {full[drop]:.1e}')
    mf = he2(1.08)
    ref = SM.open_shell_reference(mf)[0]
    A, B = SM.uhf_tdhf_matrices(ref, 'spin_flip')
    g = SM.spin_rotation_direction(ref)
    Ar, Br, coordinates, momenta, _ = SM._spin_rotation_split(A, B, g, 2 * (ref.mo_coeff.shape[2]
                                                                          - 1))
    split = SM.krein_split(Ar, Br)
    mode = split['modes'][split['soft'][0]]
    om = np.sqrt(mode['omega2'])
    pole = SM.collective_modes(np.array([om]), np.array([1.0 / np.sqrt(2 * mode['mass'] * om)]),
                               mode['mass'], coordinates @ mode['direction'],
                               momenta @ mode['conjugate'])
    n = A.shape[0]
    w, v = np.linalg.eig(np.block([[A, B], [-B, -A]]))
    k = int(np.argmin(np.abs(w - om)))
    x, y = np.real(v[:n, k]), np.real(v[n:, k])
    x, y = x / np.sqrt(x @ x - y @ y), y / np.sqrt(x @ x - y @ y)
    s = np.sign(x @ pole[1][:, 0])
    err = max(float(np.max(np.abs(s * x - pole[1][:, 0]))),
              float(np.max(np.abs(s * y - pole[2][:, 0]))))
    ok &= check(err < POLE_TOL, 'He2+ 1.08 A spin flip: the harmonic soft pole through the reduced '
                'split = the dense RPA eigenpair', f'{err:.1e}, Omega {om * HARTREE_TO_EV:.3f} eV')
    return ok


def check_he2():
    ok = True
    print('    R      TDHF        1D      GCM     NOCI     FCI   (eV)')
    rows = []
    for R in HE2_ROWS:
        mf = he2(R)
        r1 = SM.soft_mode_open_shell(mf, 'spin_conserving', '1d')
        rg = SM.soft_mode_open_shell(mf, 'spin_conserving', 'gcm')
        fci_gap = he2_fci_gap(mf) * HARTREE_TO_EV
        a, b = he2_localized(R, +1.0), he2_localized(R, -1.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            noci = SM.noci_levels([a, b])
        broken = noci['rank'] == 2
        gap = (noci['energies'][1] - noci['energies'][0]) * HARTREE_TO_EV if broken else None
        om2 = r1['soft'][0]['omega2']
        tdhf = (f'{np.sqrt(om2) * HARTREE_TO_EV:.3f}' if om2 > 0
                else f'{np.sqrt(-om2) * HARTREE_TO_EV:.3f}i')
        level_1d = r1['soft'][0]['excitation'] * HARTREE_TO_EV
        level_gcm = rg['soft'][0]['excitation'] * HARTREE_TO_EV
        flags = [f['flag'] for f in r1['numerical_flags']]
        rows.append(dict(R=R, om2=om2, d1=level_1d - fci_gap, gcm=level_gcm - fci_gap,
                         noci=None if gap is None else gap - fci_gap, flags=flags,
                         noci_flags=[f['flag'] for f in noci['numerical_flags']],
                         warned=any(issubclass(w.category, SM.SoftModeNOCIShort) for w in caught)))
        print(f'    {R:4.2f}  {tdhf:>8s}  {level_1d:7.3f}  {level_gcm:7.3f}  '
              f'{"-" if gap is None else f"{gap:7.3f}":>7s}  {fci_gap:7.3f}   {flags}')
    past = [r for r in rows if r['R'] >= 1.75]
    ok &= check(all(r['om2'] < 0 for r in past)
                and all(r['om2'] > 0 for r in rows if r['R'] < 1.75),
                'TDHF turns imaginary from 1.75 A on')
    worst = max(abs(r['gcm']) for r in rows)
    ok &= check(worst < GCM_TOL, 'GCM within GCM_TOL of FCI on every row', f'{worst:.3f} eV')
    worst = max(abs(r['d1']) for r in past)
    ok &= check(worst < ONE_D_TOL, '1D within ONE_D_TOL of FCI from 1.75 A on', f'{worst:.3f} eV')
    stiff = [r for r in rows if r['R'] < 1.75]
    ok &= check(all('soft_mode_edge_weight' in r['flags'] for r in stiff),
                "the stiff mode's 1D levels (1.08 to 1.5 A) flagged 'soft_mode_edge_weight'",
                ', '.join(f"{r['d1']:+.2f}" for r in stiff))
    worst = max(abs(r['noci']) for r in past)
    closer = all(abs(r['noci']) < abs(r['gcm']) for r in past)
    ok &= check(all(r['noci'] is not None for r in past) and worst < NOCI_TOL and closer,
                'NOCI of the localized solutions within NOCI_TOL of FCI and closer than the GCM '
                '(1.75 A on)', f'{worst:.3f} eV')
    collapsed = [r for r in rows if r['R'] == 1.5][0]
    ok &= check(collapsed['noci'] is None and 'soft_mode_noci_rank' in collapsed['noci_flags']
                and collapsed['warned'],
                "1.5 A: the localized guesses fall back to the symmetric solution, "
                "'soft_mode_noci_rank'")
    misses, ranks, flags = [], [], []
    for R in HE2_MINIMAL_ROWS:
        mf = he2(R, 'sto-3g')
        rec = SM.soft_mode_open_shell(mf, 'spin_conserving', 'gcm')
        misses.append(abs(rec['soft'][0]['excitation'] - he2_fci_gap(mf)) * HARTREE_TO_EV)
        ranks.append(rec['soft'][0]['ranks'])
        flags += [f['flag'] for f in rec['numerical_flags']]
    ok &= check(max(misses) < EXACT_TOL and all(r == (1, 1) for r in ranks)
                and 'soft_mode_gcm_few_states' not in flags,
                'He2+/STO-3G, one beta spin-orbital pair: the GCM keeps its exact span, one even '
                "and one odd state, equals FCI's gap and raises no 'soft_mode_gcm_few_states'",
                f'{max(misses):.1e} eV; ranks {ranks}; {flags}')
    return ok


def check_h4():
    ok = True
    print('    geometry   TDHF (complex)    GCM down    up   | FCI singlet quintet | SF-TDA  '
          'overlap')
    rows = []
    for a, b in H4_ROWS:
        mf = uhf(h4(a, b, '6-31g', 2))
        try:
            SM.soft_mode_open_shell(mf, 'spin_flip', 'gcm')
            refused = False
        except CasidaInstability:
            refused = True
        rec = SM.soft_mode_open_shell(mf, 'spin_flip', 'gcm', collisions='gcm')
        pair = rec['pairs'][0]
        ref = SM.open_shell_reference(mf)[0]
        A, B = SM.uhf_tdhf_matrices(ref, 'spin_flip')
        tda = np.linalg.eigvalsh(A[:3 * (ref.mo_coeff.shape[2] - 1), :3 * (ref.mo_coeff.shape[2]
                                                                          - 1)])[0]
        lam = rpa_omega2(A, B)
        worst = np.sqrt(complex(lam[np.argmax(np.abs(lam.imag))]))
        rhf = scf.RHF(h4(a, b, '6-31g', 0)).run(conv_tol=1e-12)
        norb = rhf.mo_coeff.shape[1]
        c1 = fci.FCI(rhf)
        c1.conv_tol = 1e-12
        et = c1.kernel(nelec=(3, 1))[0]
        c0 = fci.FCI(rhf)
        c0.nroots = 8
        c0.conv_tol = 1e-12
        e0, v0 = c0.kernel(nelec=(2, 2))
        s2 = [fci.spin_op.spin_square0(v, norb, (2, 2))[0] for v in v0]
        singlet = (min(e for e, s in zip(e0, s2) if s < 0.5) - et) * HARTREE_TO_EV
        quintet = (min(e for e, s in zip(e0, s2) if s > 5.5) - et) * HARTREE_TO_EV
        down, up = pair['deexcitation'] * HARTREE_TO_EV, pair['excitation'] * HARTREE_TO_EV
        real = np.all(np.isreal(rec['omega'])) and np.all(np.isfinite(rec['omega']))
        rows.append(dict(geometry=(a, b), refused=refused, real=real, down=down - singlet,
                         up=up - quintet, flags=[f['flag'] for f in rec['numerical_flags']],
                         complex=abs(worst.imag) > 0))
        print(f'    {a}x{b}  {worst * HARTREE_TO_EV:.3f}  {down:+.3f}  {up:+.3f} | {singlet:+.3f}  '
              f'{quintet:+.3f} | {tda * HARTREE_TO_EV:+.3f}  {pair["reference_overlap"]:.3f}  '
              f'{rows[-1]["flags"]}')
    ok &= check(all(r['complex'] and r['refused'] for r in rows),
                'full spin-flip TDHF complex on every row, refused by default')
    ok &= check(all(r['real'] for r in rows), "collisions='gcm': every pole real and finite")
    worst = max(abs(r['down']) for r in rows)
    ok &= check(worst < SINGLET_TOL and all(r['down'] > 0 for r in rows),
                'the de-excitation to the singlet within SINGLET_TOL of FCI, the gap too small',
                ', '.join(f"{r['down']:+.3f}" for r in rows))
    worst = max(abs(r['up']) for r in rows)
    ok &= check(worst < QUINTET_TOL, 'the excitation to the quintet within QUINTET_TOL of FCI',
                ', '.join(f"{r['up']:+.3f}" for r in rows))
    flagged = [r['geometry'] for r in rows if 'soft_mode_collision_reference' in r['flags']]
    others = {r['geometry']: r['flags'] for r in rows if r['geometry'] != (2.0, 2.0)}
    ok &= check(flagged == [(2.0, 2.0)] and not any(others.values()),
                "only the stretched square flagged 'soft_mode_collision_reference', no flag on "
                'the others', f'{flagged}; others {others}')
    return ok


def check_refusals():
    ok = True
    mf = he2(2.0)
    rhf = scf.RHF(gto.M(atom='H 0 0 0; H 0 0 1.5', basis='cc-pvdz', verbose=0)).run(conv_tol=1e-12)
    rohf = scf.ROHF(h4(1.5, 1.5, 'sto-3g', 2)).run()
    ghf = scf.GHF(h4(1.5, 1.5, 'sto-3g', 2)).run()
    uks = dft.UKS(h4(1.5, 1.5, 'sto-3g', 2))
    uks.xc = 'lda'
    uks.run()
    dfuhf = scf.UHF(h4(1.5, 1.5, 'sto-3g', 2)).density_fit().run()
    cold = scf.UHF(h4(1.5, 1.5, 'sto-3g', 2))
    cold.max_cycle = 1
    cold.kernel()
    mirror = uhf(gto.M(atom='H 0 0 0; H 1.5 0 0; H 1.5 1.5 0; H 0 1.5 0', basis='sto-3g', spin=-2,
                       verbose=0))
    ionized = uhf(h2o('6-31g', charge=1, spin=1))
    ref = SM.open_shell_reference(ionized)[0]
    A, B = SM.uhf_tdhf_matrices(ref, 'spin_conserving')
    lam, vec = np.linalg.eig((A - B) @ (A + B))
    order = np.argsort(lam.real)
    even = []
    for k in order:
        d = np.real(vec[:, k]) / np.linalg.norm(np.real(vec[:, k]))
        L = SM._open_shell_half_period(ref, 'spin_conserving', d)[0]
        if SM._open_shell_parity(ref, 'spin_conserving', d, L) > SM.SOFT_MODE_OPEN_SHELL_PARITY_MAX:
            even.append(d)
        if len(even) == 2:
            break
    plane = np.linalg.qr(np.array(even).T)[0]
    closed = scf.addons.convert_to_uhf(rhf)
    builds, veff = [], closed.get_veff

    def counted(*args, **kwargs):
        builds.append(1)
        return veff(*args, **kwargs)
    closed.get_veff = counted
    cases = [('unknown block (an SCF not yet run)',
              lambda: SM.soft_mode_open_shell(scf.UHF(mf.mol), 'triplet'), ValueError,
              "block = 'triplet'"),
             ("model '3d'", lambda: SM.soft_mode_open_shell(mf, model='3d'), ValueError),
             ('unknown collisions', lambda: SM.soft_mode_open_shell(mf, collisions='all'),
              ValueError),
             ('an RHF', lambda: SM.soft_mode_open_shell(rhf), NotImplementedError),
             ('an ROHF', lambda: SM.soft_mode_open_shell(rohf), NotImplementedError),
             ('a GHF', lambda: SM.soft_mode_open_shell(ghf), NotImplementedError),
             ('a functional', lambda: SM.soft_mode_open_shell(uks), ValueError),
             ('density fitting', lambda: SM.soft_mode_open_shell(dfuhf), ValueError),
             ('an unconverged UHF', lambda: SM.soft_mode_open_shell(cold), ValueError),
             ('n_beta > n_alpha', lambda: SM.soft_mode_open_shell(mirror), ValueError),
             ('the spin-flip block with n_alpha = n_beta',
              lambda: SM.soft_mode_open_shell(closed, 'spin_flip'), NotImplementedError,
              'n_alpha = n_beta'),
             ('a spin-conserving plane no symmetry takes to minus itself (H2O+)',
              lambda: SM.open_shell_collision_levels(ref, 'spin_conserving', plane),
              NotImplementedError),
             ('noci_levels with one solution (an SCF not yet run)',
              lambda: SM.noci_levels([scf.UHF(mf.mol)]), ValueError, 'needs two or more'),
             ('noci_levels over two molecules', lambda: SM.noci_levels([mf, he2(2.5)]),
              ValueError)]
    for label, call, kind, *text in cases:
        try:
            call()
            raised = None
        except Exception as exc:   # noqa: BLE001 -- the class (and message) is what is checked
            raised = exc
        ok &= check(isinstance(raised, kind) and all(t in str(raised) for t in text),
                    f'refused: {label}', f'{type(raised).__name__}: {str(raised)[:70]}')
    ok &= check(not builds, 'the spin-flip block with n_alpha = n_beta is refused before any Fock '
                'build', f'{len(builds)} get_veff calls')
    mixed = [he2(2.0, '6-31g'), he2(2.0, '3-21g')]
    cold = [scf.UHF(m.mol) for m in mixed]
    labels = [m.mol.ao_labels() for m in mixed]
    for label, solutions in (('converged', mixed), ('not yet run, so the check comes first', cold)):
        try:
            SM.noci_levels(solutions)
            raised = None
        except Exception as exc:   # noqa: BLE001 -- the class and message are what is checked
            raised = exc
        ok &= check(labels[0] == labels[1] and isinstance(raised, ValueError)
                    and 'do not share one molecule' in str(raised),
                    f'refused: noci_levels over two bases with the same AO labels (He2+ in 6-31G '
                    f'and 3-21G, {label})', f'{type(raised).__name__}: {str(raised)[:70]}')
    return ok


def check_complex_instability():
    ok = True
    mf = uhf(gto.M(atom='O 0 0 0; O 0 0 1.21', basis='sto-3g', spin=2, verbose=0))
    for model in ('1d', 'gcm'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            rec = SM.soft_mode_open_shell(mf, 'spin_conserving', model)
        names = [f['flag'] for f in rec['numerical_flags']]
        unstable = [s for s in rec['soft'] if s['mass'] < 0 and s['stiffness'] > 0]
        ok &= check(mf.converged and len(unstable) == 1 and len(rec['soft']) > 1
                    and names.count('soft_mode_complex_instability') == 1
                    and any(issubclass(w.category, SM.SoftModeExcitedShort) for w in caught),
                    f"O2/STO-3G's default triplet UHF ({model}): its one soft mode of M < 0 and "
                    "mu > 0 raises 'soft_mode_complex_instability', its modes of M > 0 do not",
                    f"M {[round(s['mass'], 3) for s in rec['soft']]}, mu "
                    f"{[round(s['stiffness'], 4) for s in rec['soft']]}; level "
                    f"{unstable[0]['excitation'] * HARTREE_TO_EV:+.3f} eV; {names}"
                    if unstable else f'{names}')
    return ok


def run():
    warnings.simplefilter('ignore')
    ok = True
    print('\n-- 1. the blocks')
    ok &= check_matrices()
    print('\n-- 2. the reference')
    ok &= check_reference()
    print('\n-- 3. the kernels')
    ok &= check_kernels()
    print('\n-- 4. the paths')
    ok &= check_paths()
    print('\n-- 5. the closed-shell limit')
    ok &= check_closed_shell_limit()
    print('\n-- 6. the spin rotation')
    ok &= check_spin_rotation()
    print('\n-- 7. He2+ against FCI')
    ok &= check_he2()
    print('\n-- 8. H4 spin flip against FCI')
    ok &= check_h4()
    print('\n-- 9. refusals')
    ok &= check_refusals()
    print('\n-- 10. a complex instability of a ground-state determinant')
    ok &= check_complex_instability()
    print('\nALL PASSED' if ok else '\nFAILURES DETECTED')
    return ok


def test_soft_mode_open_shell():
    assert run()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
