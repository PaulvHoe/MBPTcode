"""The spin-orbital layout of a mean field in the spin-orbital parquet driver.

* qsPA's final orbitals come in ascending order of eQP, not in the
  reference's layout: `run_parquet_so` labels each by its c^T S c weights
  (0 alpha, 1 beta to PARQUET_SO_SPIN_WEIGHT_TOL, -1 mixed) and gives no
  orbital index. The OH radical's (UHF) are pure alpha or beta; a closed
  shell's degenerate pairs are rotated by the eigensolver at will, so their
  labels follow their weights, whatever the LAPACK.
* An RHF's alpha and beta copy of each spatial orbital are one pair: its
  first spin orbital is solved and its root copied to the second, so the
  pair stays degenerate to the bit through evPA at s_1b = 1e8, where Newton
  between dense poles takes copies that differ in the last bit to other
  roots (meV apart from the second iteration on without pairs). The first
  iteration is Quack's (R_evParquet and R_Parquet_QP_graph, QUACK_FIRST) on
  the orbitals whose Z is at least Z_HELD in both codes: a satellite of
  Z = 3e-3 moves by 3 eV with a 1e-11 change of the two-body fixed point.
* qsPA starts from the mean field's own core Hamiltonian (mf.get_hcore()):
  an RHF in a field, an X2C RHF and the OH radical in a field (its pi hole
  pinned on the x axis, `test_parquet_so_vs_quack.pinned_uhf`) have their
  own Fock matrix diagonal in their orbitals with it and not with T + V, and
  their qsPA is Quack's G_qsParquet handed that Hc (QUACK_HCORE), the same
  run through the front door (`tests.pins.assert_same`).

Run: python tests/test_parquet_so_layout.py, or under pytest.
"""
import functools
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import gto, scf

from src.Base.numerics.constants import (HARTREE_TO_EV, PARQUET_SO_SPIN_WEIGHT_TOL,
                                         PARQUET_VERTEX_DAMPING)
from src.SingleReference.Parquet import spin_orbital
from tests.parquet_quack_newton import quack_newton
from tests.pins import assert_same, bitwise
from tests.test_parquet_restricted_vs_quack import rhf
from tests.test_parquet_so_vs_quack import pinned_uhf
from tests.test_parquet_spin_adaptation import both_layouts

OH = 'O 0 0 0; H 0 0 0.9697'
WATER = 'O 0 0 0; H 0.7571 0 0.5861; H -0.7571 0 0.5861'
#: au. The uniform field (along z, origin 0) of the core-Hamiltonian cases.
FIELD = 0.01
#: Ha. The reference's Fock matrix in its own orbitals with mf.get_hcore()
#: is diagonal to this (1.6e-8 at most measured, the SCF's convergence) ...
OWN_FOCK = 1e-7
#: ... and with T + V it is not (1.4e-2 at least measured).
OTHER_FOCK = 1e-3
#: Ha. qsPA eqp, diag(SigC) and err_1b against Quack's G_qsParquet: 1.2e-13 at
#: most at 1 to 4 threads and after 24 one-ulp changes of the core Hamiltonian
#: (the OH radical's pi hole pinned).
HCORE_EQP = 1e-10
#: Z below which an orbital's first-iteration root is not held against
#: Quack (the satellites 9, Z = -0.20, and 11, Z = 3e-3, of water).
Z_HELD = 0.1
#: Ha. The first iteration against Quack: 8e-13 at most at 1 to 4 threads and
#: after 24 one-ulp changes of the core Hamiltonian.
FIRST_EQP = 1e-10
#: Relative. Its Z: 4e-9 at most, as measured there.
FIRST_Z_REL = 1e-5
#: eV. Without pairs the copies part by more than this from the second
#: iteration on (0.15 eV measured at the second).
UNPAIRED_SPLIT_EV = 1e-3

T = True
#: The evPA of the pairing checks (water/6-31G): Newton at s_1b = 1e8, TDA,
#: s_2b = 1, damping 0.75, no DIIS.
EVPA_1E8 = dict(tda_eh=T, tda_pp=T, eta_1b=1e8, eta_2b=1.0, damping=PARQUET_VERTEX_DAMPING,
                conv_1b=1e-13)
#: The qsPA of the labels (OH, water): TDA, s_1b = 50, s_2b = 1, three
#: iterations, two-body cap 30.
QS_LABELS = dict(tda_eh=T, tda_pp=T, max_diis_1b=1, max_diis_2b=1, eta_1b=50.0, eta_2b=1.0,
                 max_it_1b=3, conv_1b=1e-12, max_it_2b=30, conv_2b=1e-9)
#: The OH radical's labels of that qsPA (no orbital mixed), as measured.
OH_LABELS = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1]
#: The qsPA of the core-Hamiltonian cases (two iterations).
QS_HCORE = dict(tda_eh=T, tda_pp=T, max_diis_1b=1, max_diis_2b=1, eta_1b=50.0, eta_2b=1.0,
                max_it_1b=2, conv_1b=1e-12, max_it_2b=20, conv_2b=1e-12)

# Quack's numbers (Ha), of the paper-era tree (git e3528a40), in copies of
# its routines that print their numbers in full precision and are otherwise
# unchanged, one thread. QUACK_HCORE: G_qsParquet handed each mean field's
# mf.get_hcore() (`mean_field`; the OH radical's UHF with its pi hole on the x
# axis) with QS_HCORE. QUACK_FIRST: R_evParquet with R_Parquet_QP_graph's
# Newton, its first one-body iteration on water/6-31G (EVPA_1E8, conv_2b
# 1e-12), per spatial orbital.
QUACK_HCORE = {
    'water_field/hcore_qs_cut2': {
        'n_it_1b': 2,
        'n_it_2b': [20, 20],
        'err_1b': [0.005187908896852889, 0.00198602829770117],
        'eqp': [-19.83046542087204, -19.830465420872002, -1.2940682206010752, -1.294068220601071,
                -0.6711089363468279, -0.6711089363468232, -0.49970818822173124,
                -0.4997081882217242, -0.42559839835360813, -0.42559839835360624,
                0.19938313939377503, 0.19938313939378322, 0.29093733143945694, 0.29093733143945805,
                1.0242706574252112, 1.0242706574252116, 1.091758253114266, 1.0917582531142667,
                1.1080691809627377, 1.1080691809627425, 1.1586821037333168, 1.158682103733324,
                1.2576593700678451, 1.2576593700678458, 1.6817510113715102, 1.6817510113715177],
        'sigc': [0.7415510957694699, 0.7415510957695027, 0.06162193058308122, 0.061621930583084496,
                 0.03944827665070769, 0.0394482766507081, 0.0666908039482219, 0.066690803948222,
                 0.0782523267064533, 0.07825232670645603, -0.016839590706682783,
                 -0.016839590706682374, -0.0211331489319636, -0.021133148931963875,
                 -0.03682295781766878, -0.036822957817667465, -0.09723498760562747,
                 -0.09723498760562917, -0.05440657743311973, -0.054406577433117116,
                 -0.055157638400159165, -0.055157638400155515, -0.11637763819265654,
                 -0.11637763819265602, -0.017369585385832907, -0.017369585385832373],
    },
    'water_x2c/hcore_qs_cut2': {
        'n_it_1b': 2,
        'n_it_2b': [20, 20],
        'err_1b': [0.005310211281396193, 0.0021597728643460434],
        'eqp': [-19.846708274969654, -19.846708274969618, -1.3006148637784254, -1.300614863778425,
                -0.6754692295862632, -0.6754692295862611, -0.4990682146342699, -0.4990682146342684,
                -0.4270514563462102, -0.427051456346209, 0.1854881325811218, 0.18548813258112212,
                0.27739130323821565, 0.2773913032382159, 1.0195965672837406, 1.0195965672837415,
                1.0841268729494153, 1.0841268729494196, 1.1031121848861467, 1.10311218488615,
                1.1526084077063843, 1.1526084077063845, 1.2475877572711431, 1.2475877572711498,
                1.6707051628199037, 1.6707051628199039],
        'sigc': [0.7491928839833294, 0.7491928839833664, 0.06255821169120725, 0.06255821169120312,
                 0.03989252965113281, 0.03989252965113509, 0.06817654146484266, 0.0681765414648446,
                 0.08042419266231239, 0.08042419266231428, -0.01702738717788007,
                 -0.01702738717788091, -0.021419571805860946, -0.02141957180586099,
                 -0.038019689145268896, -0.03801968914526846, -0.09996309857214382,
                 -0.09996309857214085, -0.05579393113278369, -0.05579393113278021,
                 -0.0572502270798087, -0.057250227079807425, -0.12170443010081665,
                 -0.12170443010081383, -0.019907404465601672, -0.019907404465606352],
    },
    'oh_field/hcore_qs_cut2': {
        'n_it_1b': 2,
        'n_it_2b': [20, 20],
        'err_1b': [0.007268505523221558, 0.00416243020408924],
        'eqp': [-19.858349455689552, -19.806802987830157, -1.262212028998319, -1.1698721631340763,
                -0.6152848278273442, -0.5727498262984511, -0.5533197009073255,
                -0.48196490486304194, -0.4356798054474396, -0.0021789376754192905,
                0.2161502647031512, 0.2280467315779717, 0.9859404509140058, 1.0153248645327049,
                1.0255289629710094, 1.0331357102515653, 1.0879989648566883, 1.1004715161401326,
                1.1065520854896647, 1.1272810368615878, 1.4681785383355002, 1.5149174985580927],
        'sigc': [0.7982092664375058, 0.810203667957981, 0.12539777263373195, 0.06324401102011351,
                 0.05973900716470731, 0.05102725549297845, 0.09269537648479528,
                 0.07836645107789374, 0.07255487621116953, -0.1362907258907791,
                 -0.014722382233550926, -0.015982109062273767, -0.26249256478239485,
                 -0.0506432173623907, -0.07885585438739302, -0.07832341075136025,
                 -0.0367659838109602, -0.07199573060228258, -0.0433603446916898,
                 -0.07431983027991511, -0.034519915904071885, -0.03226257047129264],
    },
}
QUACK_FIRST = {
    'water/evpa_newton_s1b1e8_first': {
        'eqp': [-20.057981517371154, -1.3042699934761801, -0.6746151450278324, -0.5024416875604799,
                -0.4329016819412278, 0.18846815911172682, 0.28210379368492533, 1.0325511203454707,
                1.1183058307605704, 1.1963201831046975, 1.1677775049246042, 2.5275767608972823,
                1.7158177454850982],
        'z': [0.7975468322165068, 0.13797793584694773, 0.9506476168489796, 0.9369165665827656,
              0.9301068372641813, 0.9833622818507273, 0.97589630197795, 0.9595835624424501,
              0.9590418962722976, -0.19849990563756578, 0.9341621160481398, 0.002884619959474609,
              0.43455678210184223],
    },
}['water/evpa_newton_s1b1e8_first']


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


@functools.lru_cache(maxsize=None)
def mean_field(name):
    """The mean fields of this file: 'oh' (UHF), the core-Hamiltonian cases
    'water_field' (RHF), 'water_x2c' (spin-free X2C RHF), 'oh_field' (UHF,
    its pi hole on the x axis: `pinned_uhf`, since its qsPA's err_1b turns
    with the hole)."""
    atom = OH if name.startswith('oh') else WATER
    mol = gto.M(atom=atom, basis='6-31g', spin=1 if name.startswith('oh') else 0, verbose=0)
    h = None
    if name.endswith('_field'):
        with mol.with_common_orig((0.0, 0.0, 0.0)):
            h = scf.hf.get_hcore(mol) + FIELD * mol.intor_symmetric('int1e_r')[2]
    if name == 'oh_field':
        return quiet(pinned_uhf, mol, h)
    if name == 'water_x2c':
        mf = scf.RHF(mol).x2c()
    else:
        mf = scf.UHF(mol) if name.startswith('oh') else scf.RHF(mol)
    if h is not None:
        mf.get_hcore = lambda *args, h=h: h
    mf.conv_tol = 1e-12
    quiet(mf.kernel)
    assert mf.converged, name
    return mf


def keep_core_hamiltonians(mf, monkeypatch):
    """The arrays mf.get_hcore() returns from here on (`monkeypatch`'s while
    the test runs), each kept as a copy as it comes back: what a driver
    received is held to that copy bitwise, a second call being another
    computation (X2C's a diagonalization and products)."""
    kept, real = [], mf.get_hcore

    def get_hcore(*args, **kwargs):
        h = real(*args, **kwargs)
        kept.append(np.array(h, dtype=float))
        return h
    monkeypatch.setattr(mf, 'get_hcore', get_hcore)
    return kept


def weights(c, s):
    """(alpha, beta) weights c^T S c of each spin-AO orbital (column of c)."""
    nao = s.shape[0]
    return (np.einsum('mk,mn,nk->k', c[:nao], s, c[:nao]),
            np.einsum('mk,mn,nk->k', c[nao:], s, c[nao:]))


def labels_of(c, s):
    w_alpha, w_beta = weights(c, s)
    out = np.full(c.shape[1], -1)
    out[np.abs(w_alpha - 1.0) <= PARQUET_SO_SPIN_WEIGHT_TOL] = 0
    out[np.abs(w_beta - 1.0) <= PARQUET_SO_SPIN_WEIGHT_TOL] = 1
    return out


def test_the_qspa_spin_labels_of_an_open_shell():
    mf = mean_field('oh')
    result = quiet(spin_orbital.run_parquet_so, mf, 'qs', **QS_LABELS)
    s = mf.mol.intor_symmetric('int1e_ovlp')
    assert result.orbital is None and result.reference == 'UHF'
    assert result.spin.tolist() == labels_of(result.mo_coeff, s).tolist() == OH_LABELS
    assert np.all(np.diff(result.eqp) >= 0.0)
    w_alpha, w_beta = weights(result.mo_coeff, s)
    assert np.allclose(w_alpha + w_beta, 1.0, atol=PARQUET_SO_SPIN_WEIGHT_TOL)


def test_the_qspa_spin_labels_of_a_closed_shell():
    """Water's degenerate pairs: each orbital's label is the one its weights
    give, a pure pair labelled one alpha and one beta, a rotated pair -1."""
    mf = rhf('water')
    result = quiet(spin_orbital.run_parquet_so, mf, 'qs', **QS_LABELS)
    s = mf.mol.intor_symmetric('int1e_ovlp')
    assert result.orbital is None and result.reference == 'RHF'
    assert result.spin.tolist() == labels_of(result.mo_coeff, s).tolist()
    for pair in result.spin.reshape(-1, 2):
        assert sorted(pair.tolist()) in ([0, 1], [-1, -1]), result.spin
    assert np.abs(result.eqp[0::2] - result.eqp[1::2]).max() * HARTREE_TO_EV < 1e-9


@functools.lru_cache(maxsize=None)
def evpa_1e8(pairs, iterations, conv_2b):
    """Water's evPA at s_1b = 1e8 on its interleaved spin orbitals, with or
    without the alpha/beta pairs, by Quack's Newton (`quack_newton`, handed in
    as a qp_solver function: the drivers' default is the house's root
    finder)."""
    eps, eri, nO, spin_pairs = both_layouts('water')[1]
    return quiet(spin_orbital.ev_parquet_so, eps, eri, nO, 'ev',
                 spin_pairs=spin_pairs if pairs else None, max_it_1b=iterations,
                 conv_2b=conv_2b, max_it_2b=2000, qp_solver=quack_newton, **EVPA_1E8)


@pytest.mark.parametrize('qp_solver', [quack_newton, 'linearized'], ids=['newton', 'linearized'])
def test_the_pairs_stay_degenerate_to_the_bit(qp_solver):
    """Every iteration of evPA at s_1b = 1e8, through ev_parquet_so with the
    pairs and through run_parquet_so on the RHF: each pair's eqp and Z equal
    to the bit, its second spin orbital solved by none of Newton's steps."""
    rhf_run = quiet(spin_orbital.run_parquet_so, rhf('water'), 'ev', qp_solver=qp_solver,
                    max_it_1b=3, conv_2b=1e-8, max_it_2b=2000, **EVPA_1E8)
    eps, eri, nO, pairs = both_layouts('water')[1]
    so_run = quiet(spin_orbital.ev_parquet_so, eps, eri, nO, 'ev', spin_pairs=pairs,
                   qp_solver=qp_solver, max_it_1b=3, conv_2b=1e-8, max_it_2b=2000, **EVPA_1E8)
    for result in (rhf_run, so_run):
        assert result.n_it_1b == 3
        for record in result.history:
            assert np.array_equal(record['eqp'][0::2], record['eqp'][1::2])
            assert np.array_equal(record['z'][0::2], record['z'][1::2])
            if qp_solver is quack_newton:
                assert not record['qp_n_iter'][1::2].any()


def test_without_pairs_the_copies_part():
    """The same evPA without the pairs: the copies differ in their last bits
    and Newton takes them to other roots from the second iteration on."""
    split = [np.abs(r['eqp'][0::2] - r['eqp'][1::2]).max() * HARTREE_TO_EV
             for r in evpa_1e8(False, 2, 1e-8).history]
    assert split[1] > UNPAIRED_SPLIT_EV, split


def test_the_first_iteration_is_quacks():
    result = evpa_1e8(True, 1, 1e-12)
    eqp, z = result.eqp[0::2], result.z[0::2]
    quack_eqp, quack_z = np.array(QUACK_FIRST['eqp']), np.array(QUACK_FIRST['z'])
    held = (z >= Z_HELD) & (quack_z >= Z_HELD)
    assert np.flatnonzero(held).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12], z
    assert np.abs(eqp[held] - quack_eqp[held]).max() <= FIRST_EQP
    assert (np.abs(z[held] - quack_z[held]) / quack_z[held]).max() <= FIRST_Z_REL


@pytest.mark.parametrize('case', sorted(QUACK_HCORE))
def test_qspa_starts_from_the_mean_fields_own_core_hamiltonian(case, monkeypatch):
    name = case.split('/')[0]
    mf = mean_field(name)
    kept = keep_core_hamiltonians(mf, monkeypatch)
    ref = spin_orbital.spin_orbital_reference(mf)
    assert len(kept) == 1 and bitwise(ref.hcore, kept[0])
    mol = mf.mol
    dm = mf.make_rdm1()
    t_plus_v = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    for hcore, small in ((ref.hcore, True), (t_plus_v, False)):
        fock = mf.get_fock(h1e=hcore, dm=dm)
        coeffs, energies = ((mf.mo_coeff,), (mf.mo_energy,))
        if fock.ndim == 3:
            coeffs, energies = mf.mo_coeff, mf.mo_energy
            fock = list(fock)
        else:
            fock = [fock]
        gap = max(np.abs(c.T @ f @ c - np.diag(e)).max()
                  for c, f, e in zip(coeffs, fock, energies))
        assert (gap < OWN_FOCK) if small else (gap > OTHER_FOCK), (case, small, gap)
    x = spin_orbital.orthogonalization_matrix(ref.ovlp)
    low = quiet(spin_orbital.qs_parquet_so, ref.eps, ref.mo_coeff, ref.ovlp, x, ref.hcore,
                ref.eri_ao, ref.nO, **QS_HCORE)
    quack = QUACK_HCORE[case]
    assert low.n_it_2b == quack['n_it_2b']
    assert np.abs(low.eqp - quack['eqp']).max() <= HCORE_EQP
    assert np.abs(np.diag(low.sigc) - quack['sigc']).max() <= HCORE_EQP
    err_1b = np.array([h['err_1b'] for h in low.history])
    assert np.abs(err_1b - quack['err_1b']).max() <= HCORE_EQP
    front = quiet(spin_orbital.run_parquet_so, mf, 'qs', **QS_HCORE)
    assert front.n_it_2b == low.n_it_2b, case
    assert_same(front.eqp, low.eqp, label=(case, 'front door eqp'))
    assert_same([h['err_1b'] for h in front.history], [h['err_1b'] for h in low.history],
                label=(case, 'front door err_1b'))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
