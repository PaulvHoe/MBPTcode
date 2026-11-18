"""Geometries, recorded references and reference routines for the soft-mode triplet tests.

Not a test file: `test_soft_mode_triplet`, `test_soft_mode_triplet_slow`,
`test_soft_mode_triplet_full` and `test_soft_mode_variants_full` import it.

GEOMETRIES: MP2/cc-pVDZ minima (Angstrom), optimized with geomeTRIC 1.1 on pyscf 2.13.0
from idealized starting structures (all-trans planar polyenes). `ethylene_geometry` twists
one CH2 group rigidly about the C=C axis, or sets the C=C distance with both CH2 groups rigid.

EOM_CCSD: (triplet, IP, overlap, T1) recorded with pyscf 2.13.0 on a workstation (macOS arm64,
OpenBLAS 0.3.29), 17 November 2026, cc-pVDZ, all electrons, RHF conv_tol 1e-10, CCSD conv_tol
1e-8. triplet: the EOM-EE-CCSD triplet root (Ha) of eom_rccsd.EOMEETriplet(nroots=4) whose
singles overlap the soft direction d most, |<r1|d>| / |r1| = overlap; IP: the root of
ipccsd(nroots=3) with the largest HOMO weight (Ha); T1: the T1 diagnostic. For H2CO the matched
triplet is 3A1 (pi -> pi*), not the lowest triplet 3A2 (n -> pi*). EOM-CCSD from RHF misses the
soft triplet of O3 (3B2) by about -0.36 eV and twisted ethylene by up to -0.18 eV at 70 degrees
in STO-3G against FCI, while spin-flip EOM-CCSD stays within 0.07 eV (`spin_flip_gap`).

`eom_ccsd_references` and `spin_flip_gap` compute them again (cluster tier).

EOM_CCSD_ADA_EV and SPIN_FLIP_EV: the EOM-CCSD triplets of hexatriene, octatetraene and the C=C
stretch past 1.30 A, and the spin-flip EOM-CCSD gaps of O3 and of ethylene twisted by 45, 60 and
70 degrees, as `test_soft_mode_triplet_full` computed them on the ADA cluster (job 1238268, pyscf
2.12.1, 17 November 2026; cc-pVDZ, all electrons, the geometries of `geometry`, RHF conv_tol 1e-10,
CCSD conv_tol 1e-8) and printed them in its table, in eV to the meV. That run's claim 1 (every
reference well defined: converged and matched to d, and for the spin-flip rows d's leading pair,
the seeded ROHF on it, convergence and the singlet below) held on every row but O3 and the stretch
from 1.3145 to 1.42 A, which failed only because the fourth root of the EOM-CCSD triplet Davidson,
a guess left in the high continuum, did not converge; CCSD and the matched triplet root converged
everywhere, and claim 1 now reads only those (commit 18e0845).

QUEST_GEOMETRIES, CIPSI_IP_EV and G0W0_HF_IP_EV: BF's first ionization (1 2Sigma+,
(5 sigma)^-1) at the QUEST geometry (CC3/aug-cc-pVTZ, all electrons), from the ancillary files
BF.xyz and bf.json of arXiv:2402.13877 (Marie and Loos, J. Chem. Theory Comput. 20, 4751 (2024)):
CIPSI (their 'sCI', of FCI quality, frozen core, 0 meV uncertainty in aug-cc-pVDZ and
aug-cc-pVTZ) and G0W0@HF (QuAcK, eta = 1e-3 Ha, the quasiparticle equation not linearized).
`test_soft_mode_variants_full` reads them, and the ADA references above. The 'FCI' IPs that
docs/methods/parquet.md compares its Table III runs with (Marie and Loos, J. Chem. Phys. 163,
194115 (2025)) are not a second reference: all eight are these frozen-core CIPSI aug-cc-pVTZ
values rounded to 0.01 eV (BF's 11.15 is 11.149), so they are not listed.
"""
import numpy as np
import scipy.linalg
from pyscf import cc, gto, scf
from pyscf.cc import eom_rccsd, eom_uccsd

#: Angstrom; MP2/cc-pVDZ minima.
GEOMETRIES = {
    'H2O': 'O -0.00000000 -0.00000000 -0.01389461; H 0.74905697 -0.00000000 0.59345130; H -0.74905697 0.00000000 0.59345130',
    'N2': 'N -0.00000000 -0.00000000 -0.01565004; N 0.00000000 0.00000000 1.11365004',
    'H2CO': 'C 0.00000761 -0.00000000 0.00622679; O -0.00001594 0.00000000 1.22157130; H 0.94163150 0.00000000 -0.59181218; H -0.94162316 -0.00000000 -0.59180791',
    'O3': 'O 0.00000000 0.00000000 -0.00676747; O 1.09659820 0.00000000 0.66989374; O -1.09659820 -0.00000000 0.66989374',
    'ethylene': 'C -0.00640229 -0.00363006 -0.00000000; C 1.15821629 0.66863006 0.00000000; H -0.96653229 0.52045385 0.00000000; H -0.03247780 -1.09720422 -0.00000000; H 2.11834629 0.14454615 0.00000000; H 1.18429180 1.76220422 0.00000000',
    'butadiene': 'C -0.04081549 -0.00931772 0.00000000; C 1.14390517 0.64299308 -0.00000000; C 2.44144013 -0.03299327 -0.00000000; C 3.62616063 0.61931780 -0.00000000; H -0.98960996 0.53283601 0.00000000; H -0.08138014 -1.10339732 0.00000000; H 1.15701273 1.74034682 -0.00000000; H 2.42833267 -1.13034701 -0.00000000; H 4.57495518 0.07716420 0.00000000; H 3.66672509 1.71339741 0.00000000',
    'hexatriene': 'C -0.07423812 -0.00804342 0.00000000; C 1.11846160 0.63408518 0.00000000; C 2.40547876 -0.04791147 -0.00000000; C 3.60473735 0.59791142 -0.00000000; C 4.89175455 -0.08408516 -0.00000000; C 6.08445428 0.55804343 -0.00000000; H -1.01728394 0.54393089 0.00000000; H -0.12516037 -1.10165234 0.00000000; H 1.13878650 1.73139254 0.00000000; H 2.39314996 -1.14649490 -0.00000000; H 3.61706613 1.69649486 -0.00000000; H 4.87142968 -1.18139252 -0.00000000; H 7.02750010 0.00606913 -0.00000000; H 6.13537653 1.65165235 -0.00000000',
    'octatetraene': 'C -0.10560031 -0.00355794 0.00000000; C 1.09190911 0.63136125 -0.00000000; C 2.37353586 -0.05660096 -0.00000000; C 3.57836318 0.58537840 -0.00000000; C 4.85672394 -0.09537841 -0.00000000; C 6.06155125 0.54660097 -0.00000000; C 7.34317800 -0.14136124 -0.00000000; C 8.54068742 0.49355794 0.00000000; H -1.04493282 0.55470314 0.00000000; H -0.16348202 -1.09683357 0.00000000; H 1.11837546 1.72862645 -0.00000000; H 2.35719372 -1.15517075 0.00000000; H 3.59197931 1.68407349 -0.00000000; H 4.84310782 -1.19407350 0.00000000; H 6.07789339 1.64517077 -0.00000000; H 7.31671164 -1.23862644 -0.00000000; H 9.48001991 -0.06470318 0.00000000; H 8.59856915 1.58683357 0.00000000',
    'benzene': 'C 1.40458812 -0.00000005 -0.00000000; H 2.49880572 0.00000013 -0.00000000; C 0.70229409 1.21640902 0.00000000; H 1.24940283 2.16402930 0.00000000; C -0.70229407 1.21640905 -0.00000000; H -1.24940295 2.16402925 -0.00000000; C -1.40458811 0.00000000 0.00000000; H -2.49880571 0.00000016 0.00000000; C -0.70229405 -1.21640906 0.00000000; H -1.24940290 -2.16402927 -0.00000000; C 0.70229411 -1.21640914 0.00000000; H 1.24940291 -2.16402939 0.00000000',
}

#: (triplet Ha, IP Ha, overlap, T1); see the module docstring for how they were recorded.
EOM_CCSD = {
    'H2O': (0.2733285323898445, 0.43333608399706103, 0.9966318308671028, 0.0053773232035966515),
    'N2': (0.26314186902356385, 0.6123908674652037, 0.9968246702769942, 0.010673695976017958),
    'H2CO': (0.21718419776303288, 0.3804286287286475, 0.9956122781971486, 0.014449213242844419),
    'O3': (0.04221717574960994, 0.48121161124184636, 0.9917907950774053, 0.02541571383435266),
    'ethylene': (0.1623257429406195, 0.3818377012698538, 0.9949285007565182, 0.008785187032924135),
    'butadiene': (0.1184931402917003, 0.32610801117235033, 0.9754583330222716, 0.009357755716739585),
    'benzene': (0.1440916042251251, 0.3332795756843627, 0.9873606298805205, 0.008472438812633487),
    'ethylene@5': (0.16139160969285837, 0.3811596382358087, 0.9945822166949861, 0.008792882227565066),
    'ethylene@10': (0.15861329057086332, 0.3791779367537404, 0.9935797843816626, 0.008816072324375968),
    'ethylene@15': (0.1540604456068018, 0.37603168895787514, 0.9920214168039814, 0.008855074282551771),
    'ethylene@20': (0.1478411319382941, 0.3719049396136056, 0.9900461256965475, 0.008910444718261678),
    'ethylene@25': (0.14009153038614566, 0.3669869920274106, 0.9878004805567778, 0.008982989888024925),
    'ethylene@30': (0.13096467010962165, 0.36144869898592097, 0.9854101774072063, 0.009073947479527618),
    'ethylene@35': (0.12062025497457411, 0.3554347246119822, 0.9829622638664125, 0.00918493949015169),
    'ethylene@40': (0.10921680411660192, 0.3490649718512393, 0.9804980287767876, 0.009318254687849708),
    'ethylene@45': (0.09690686653720577, 0.34244032663169516, 0.9780147780333796, 0.009477135855442083),
    'ethylene@50': (0.08383510883883603, 0.3356494577311968, 0.9754704631803528, 0.009666247268738413),
    'ethylene@55': (0.070139244592803, 0.3287761834405897, 0.9727875153673294, 0.009892611981105689),
    'ethylene@60': (0.055953346701687355, 0.32190784388328986, 0.9698516853361651, 0.010167111912413847),
    'ethylene@65': (0.04141383331383936, 0.31514554417958984, 0.9665028700045831, 0.010507307352744385),
    'ethylene@70': (0.026670129741463135, 0.3086195310177633, 0.9625166394567117, 0.010942849843434977),
    'ethylene~1.2200': (0.21368886568622644, 0.40384125589680253, 0.9946962172450637, 0.0072302788717660185),
    'ethylene~1.2400': (0.20471880471484663, 0.3998962827947268, 0.9947616645904591, 0.007458309893032791),
    'ethylene~1.2600': (0.19603925016147175, 0.39611934461188214, 0.9948151697236008, 0.007695246772580133),
    'ethylene~1.2800': (0.1876428861187891, 0.39250377925467683, 0.9948577300398275, 0.00794051034593284),
    'ethylene~1.3000': (0.1795223629847692, 0.389043337063886, 0.9948901237928298, 0.008193570947450388),
}

#: eV. EOM-CCSD triplets (`eom_ccsd_references`: the root whose singles overlap d most) from the ADA
#: run of test_soft_mode_triplet_full (job 1238268, pyscf 2.12.1, 17 November 2026), to the meV as
#: printed; see the module docstring.
EOM_CCSD_ADA_EV = {
    'hexatriene': 2.587, 'octatetraene': 2.200,
    'ethylene~1.3145': 4.729, 'ethylene~1.3200': 4.671, 'ethylene~1.3245': 4.624,
    'ethylene~1.3295': 4.572, 'ethylene~1.3325': 4.541, 'ethylene~1.3365': 4.500,
    'ethylene~1.3400': 4.465, 'ethylene~1.3600': 4.265, 'ethylene~1.3800': 4.072,
    'ethylene~1.4000': 3.886, 'ethylene~1.4200': 3.706, 'ethylene~1.4400': 3.532,
    'ethylene~1.4600': 3.364,
}
#: eV. Spin-flip EOM-CCSD triplet - singlet gaps (`spin_flip_gap`, from the M_s = 1 ROHF seeded with
#: d's leading pair), the same run: the reference where EOM-CCSD from the RHF falls short.
SPIN_FLIP_EV = {'O3': 1.513, 'ethylene@45': 2.757, 'ethylene@60': 1.693, 'ethylene@70': 0.974}

#: Angstrom; the QUEST CC3/aug-cc-pVTZ geometry of BF (arXiv:2402.13877, BF.xyz).
QUEST_GEOMETRIES = {'BF': 'B 0.0000 0.0000 0.0000; F 0.0000 0.0000 1.2685'}
#: eV, keyed by (molecule, basis as pyscf spells it); see the module docstring. CIPSI from bf.json's
#: 'sCI' entries (frozen core), G0W0@HF from bf.json's 'GW' entries.
CIPSI_IP_EV = {('BF', 'aug-cc-pvdz'): 11.054, ('BF', 'aug-cc-pvtz'): 11.149}
G0W0_HF_IP_EV = {('BF', 'aug-cc-pvdz'): 11.117, ('BF', 'aug-cc-pvtz'): 11.325}


def ethylene_geometry(twist=0.0, rcc=None):
    """Ethylene (Angstrom string) with the second CH2 group turned by twist degrees about the C=C
    axis, and the C=C distance set to rcc (Angstrom) with both CH2 groups rigid."""
    atoms = [(a.split()[0], np.array([float(v) for v in a.split()[1:]]))
             for a in GEOMETRIES['ethylene'].split('; ')]
    c = [i for i, (s, _) in enumerate(atoms) if s == 'C']
    c0, c1 = atoms[c[0]][1], atoms[c[1]][1]
    axis = (c1 - c0) / np.linalg.norm(c1 - c0)
    shift = 0.0 if rcc is None else 0.5 * (rcc - np.linalg.norm(c1 - c0))
    t = np.radians(twist)
    out = []
    for s, x in atoms:
        if np.linalg.norm(x - c1) < np.linalg.norm(x - c0):
            if s == 'H' and twist:
                v = x - c1
                x = c1 + v * np.cos(t) + np.cross(axis, v) * np.sin(t) + axis * axis.dot(v) * (1 - np.cos(t))
            x = x + shift * axis
        else:
            x = x - shift * axis
        out.append(f'{s} {x[0]:.8f} {x[1]:.8f} {x[2]:.8f}')
    return '; '.join(out)


def geometry(label):
    """The Angstrom string of a label: a GEOMETRIES key, 'ethylene@T' (twist, degrees) or
    'ethylene~R' (C=C distance, Angstrom), as in EOM_CCSD."""
    if '@' in label:
        return ethylene_geometry(twist=float(label.split('@')[1]))
    if '~' in label:
        return ethylene_geometry(rcc=float(label.split('~')[1]))
    return GEOMETRIES[label]


def rhf(atom, basis='cc-pvdz'):
    """A tight RHF (conv_tol 1e-10, as the references were recorded with)."""
    mol = gto.M(atom=atom, basis=basis, unit='Angstrom', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf


def eom_ccsd_references(mf, direction, nroots=4):
    """(triplet, IP, overlap, HOMO weight, converged) by EOM-CCSD from the RHF: the triplet root
    whose singles overlap the soft direction most, its Davidson seeded with d itself so that the
    soft state's irreducible representation is searched whatever the diagonal favours, and the IP
    root with the largest HOMO weight (Ha). converged: CCSD and that matched triplet root; the
    other roots only widen the search, and one of them need not converge (O3/cc-pVDZ's fourth
    root, a guess left at 59 to 67 eV, not even in 200 Davidson cycles; ADA job 1240184)."""
    nocc = mf.mol.nelectron // 2
    nmo = mf.mo_coeff.shape[1]
    nvir = nmo - nocc
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-8
    mycc.verbose = 0
    mycc.kernel()
    eom = eom_rccsd.EOMEETriplet(mycc)
    seed = eom_rccsd.amplitudes_to_vector_triplet(
        direction.reshape(nocc, nvir),
        (np.zeros((nocc, nocc, nvir, nvir)), np.zeros((nocc, nocc, nvir, nvir))))
    guess = [seed] + list(eom.get_init_guess(nroots - 1, koopmans=True))
    et, vt = eom.kernel(nroots=nroots, guess=guess)
    et, vt = np.atleast_1d(et), np.atleast_2d(vt)
    ov = []
    for v in vt:
        r1 = eom_rccsd.vector_to_amplitudes_triplet(v, nmo, nocc)[0]
        ov.append(float(abs(np.sum(r1.ravel() * direction)) / np.linalg.norm(r1)))
    k = int(np.argmax(ov))
    eip, vip = mycc.ipccsd(nroots=3)
    eip, vip = np.atleast_1d(eip), np.atleast_2d(vip)
    hw = [float(abs(v[nocc - 1]) ** 2 / np.sum(np.abs(v[:nocc]) ** 2)) for v in vip]
    j = int(np.argmax(hw))
    converged = bool(mycc.converged and np.atleast_1d(eom.converged)[k])
    return float(et[k]), float(eip[j]), ov[k], hw[j], converged


def spin_flip_gap(mf, direction, nroots=4):
    """(gap, converged, match) by spin-flip EOM-CCSD from the high-spin (M_s = 1) ROHF triplet of
    the soft direction d:

    gap        vertical triplet - singlet gap (Ha): minus the lowest spin-flip root, the singlet
               below the reference (positive when that root lies below it)
    converged  the ROHF, the UCCSD and every spin-flip root converged
    match      the share of the ROHF's two singly occupied orbitals on span{u_1, v_1}, the leading
               pair of d_ia's singular-value decomposition (1: the same space)

    The ROHF starts from the RHF with one beta electron taken out of u_1 and one alpha electron
    put into v_1, so that it is the triplet d describes. From the default guess the M_s = 1
    reference of O3 opens another occupied orbital (match 0.48 in cc-pVDZ, 0.50 in STO-3G, where
    the seeded 3B2 gives 0.96 and 0.97); the twisted ethylenes land on d's triplet either way
    (0.97 to 0.99). The STO-3G comparison with FCI in the module docstring was made with a
    reference seeded this way. The gap means something only when the leading pair carries most
    of d (sigma_1^2 = 0.97 to 0.99 for O3 and the twisted ethylenes): for H2O/STO-3G with both
    bonds stretched to 1.6 times, d is split over two pairs (sigma_1^2 = 0.51), the seeded
    reference lies 9 eV above the ground state, and the gap comes out at 6.1 eV, where FCI's
    triplets lie at 2.2 to 3.1 eV."""
    nocc = mf.mol.nelectron // 2
    C = mf.mo_coeff
    u, _, vt = np.linalg.svd(direction.reshape(nocc, -1))
    u1, v1 = C[:, :nocc] @ u[:, 0], C[:, nocc:] @ vt[0]
    occ = C[:, :nocc] @ C[:, :nocc].T
    mol_t = mf.mol.copy()
    mol_t.spin = 2
    mol_t.build()
    rohf = scf.ROHF(mol_t)
    rohf.conv_tol = 1e-10
    rohf.kernel(dm0=(occ + np.outer(v1, v1), occ - np.outer(u1, u1)))
    S = mf.get_ovlp()
    dm_a, dm_b = rohf.make_rdm1()
    somo = scipy.linalg.eigh(S @ (dm_a - dm_b) @ S, S)[1][:, -2:]
    match = float(np.sum((np.array([u1, v1]) @ S @ somo) ** 2) / 2)
    mycc = cc.UCCSD(scf.addons.convert_to_uhf(rohf))
    mycc.conv_tol = 1e-8
    mycc.verbose = 0
    mycc.kernel()
    eom = eom_uccsd.EOMEESpinFlip(mycc)
    e_sf, _ = eom.kernel(nroots=nroots)
    converged = bool(rohf.converged and mycc.converged and np.all(eom.converged))
    return float(-np.min(np.atleast_1d(e_sf))), converged, match
