"""One occupied or one virtual orbital: the parquet drivers with an empty pp branch.

Spin orbitals. The H atom (UHF/6-31G and UHF/STO-3G: one occupied spin
orbital, so no 2h pair) and H2^- (UHF/STO-3G: one virtual spin orbital, no 2p
pair) run through `spin_orbital.ev_parquet_so` (FLEX, osPA, evPA) and
`qs_parquet_so` (qsPA) and are held against Quack's G_evParquet and
G_qsParquet, which run these (gfortran's maxval of the empty eigenvalue
change is -huge, `spin_orbital.EMPTY_MAXVAL`); the numbers are recorded in
QUACK_SO. The H atom's mean field is the proper UHF, pyscf.scf.uhf.UHF(mol)
(PySCF's scf.UHF returns its one-electron HF1e there, which drops the
two-electron terms: the beta 1s comes out degenerate with the occupied alpha
1s, a 0/0 in Quack's denominators); `test_the_h_atom_uhf_is_the_proper_one`
holds it to the UHF built by hand (alpha F = h + J[Pa] - K[Pa], beta F = h +
J[Pa]) of the review that found it. H/STO-3G has one eh root, the spin flip,
e_beta - e_alpha - J, zero in exact arithmetic: rounding of the mean-field
energies and J puts it at 2.2e-16 (two ulps of J, as Quack's recorded runs
had it), at exactly 0.0, or a few ulps below zero. Its full-BSE rows are held
as measured and flagged 'parquet_eh_zero_mode'; where the flip comes out at
0.0 or below they are skipped, since the full eh problem has no real
solution there and the run is NaN, as Quack's would be: at exactly 0.0 its
X + Y = Om^-1/2 Z^T (A - B)^1/2 is 0 times inf (its eigenvectors carry
Om^-1/2), and below zero A - B is not positive definite, so (A - B)^1/2 is
NaN ('parquet_eh_a_minus_b_not_positive'). In the eh TDA the flip forced to
exactly 0.0 (of either sign), where Quack's regularized 1/d is 0/0, and to
+-1e-16 and +-2.2e-16 gives the run as computed to the bit, in FLEX, osPA
(by the finder and linearized), evPA and qsPA, the pp problem in the TDA
and in full: every weight at the zero is 0, its limit, and so is every
slope in Z, the value Quack's formula gives a few ulps off zero. The
closed-shell analogue, the lowest triplet forced to 0.0 (a closed shell at
the edge of its triplet instability), runs in the restricted driver as
forced to 2.2e-16, to rounding: H2/6-31G, whose zero pole has no residue,
and LiH/STO-3G, whose pole has one, so that the slope's limit 2 s_1b would
have moved its linearized energies by mHa.

Closed shells. He/6-31G and H2/6-31G (nO = 1: no triplet hh pair) and
water/6-31G cut to its six lowest orbitals (nV = 1: no triplet ee pair) run
through the restricted driver, where Quack's R_evParquet ends the run (its
R_pp_triplet_screened_integral hands dgemm a zero leading dimension). They
are held against Quack's G_evParquet on the same closed shell over spin
orbitals (QUACK_G), and against the spin-orbital driver: at s_1b = 1e8,
damping 0 and linearized, where the two self-energies coincide, every number
of the two drivers is the same quantity; at the defaults (s_1b = 50) the two
self-energies are other partial fractions, but the spectra and EcRPA still
agree, and every run is finite.

Run: python tests/test_parquet_small_systems.py, or under pytest.
"""
import functools
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import scipy.linalg as la
from pyscf import ao2mo, gto, scf
from pyscf.scf import uhf

from src.Base.numerics.constants import PARQUET_VERTEX_DAMPING
from src.SingleReference.Parquet import (restricted, restricted_bse, spin_orbital,
                                         spin_orbital_two_body)

#: Ha. The mean fields of the Fortran runs.
SCF_CONV_TOL = 1e-12
#: Ha. H/STO-3G's spin flip, e_beta - e_alpha - J, where rounding leaves it
#: two ulps of J off zero (as in Quack's recorded runs; other rounding of the
#: mean field gives exactly 0.0): a zero mode a few ulps off.
ULPS_OFF = 2.220446049250313e-16
#: Ha. The values H/STO-3G's spin flip is forced to: exactly 0 of either
#: sign and a few ulps off on either side.
SPIN_FLIP_VALUES = (0.0, -0.0, 1e-16, -1e-16, ULPS_OFF, -ULPS_OFF)
#: Ha. The eh roots below this are the zero mode the tests force (H/STO-3G's
#: spin flip; its other roots lie far above).
ZERO_MODE_BELOW = 1e-12
#: Ha. A closed shell's lowest triplet forced to 0.0 against it forced to
#: ULPS_OFF, whose frequency-independent denominators then move by a few
#: ulps (eqp, eqp_lin and SigC): 1.1e-14 measured (H2's evPA eqp_lin; LiH
#: 3.3e-16 at most).
CLOSED_ZERO_MODE_GAP = 1e-12
#: Z there, absolute: 1.4e-13 (H2) and 1.1e-11 (LiH) measured in evPA, whose
#: roots at the iterated energies magnify a few ulps of Sigma's slope, 1e-14
#: at most in FLEX and osPA (bitwise, linearized).
CLOSED_ZERO_MODE_Z = 1e-9
#: The runs a zero mode is held through, at s_2b = 1 and the eh TDA: FLEX,
#: osPA by the default finder and linearized, evPA (the default finder) and
#: qsPA (the static kernel at D_p = D_q = 0), each with the pp problem in the
#: TDA and in full ('_pprpa').
ZERO_MODE_RUNS = {f'{name}{pp}': (scheme, dict(options, **pp_options))
                  for name, (scheme, options) in (
                      ('flex', ('flex', {})), ('os', ('os', {})),
                      ('os_linearized', ('os', {'qp_solver': 'linearized'})),
                      ('ev', ('ev', {})), ('qs', ('qs', {})))
                  for pp, pp_options in (('', {}), ('_pprpa', {'tda_pp': False}))}
#: The closed shells whose lowest triplet is forced: H2/6-31G (CLOSED_SHELLS'
#: geometry), whose zero pole has no residue (sigma_g -> sigma_u: H2's
#: inversion parity), and LiH/STO-3G, whose pole has one, so that the slope's
#: limit 2 s_1b in place of its value a few ulps off, 0, moved the linearized
#: energies by 1.5 to 2.5 mHa and the linearized Z by 0.17 to 0.18 (FLEX,
#: osPA, evPA). LiH skips qsPA, whose runs meet no bar from one forced value
#: to another: its SigC matrix rotates within LiH's degenerate pi pair by
#: 1e-2 between any two of them, -2.2e-16 against +2.2e-16 (neither an exact
#: zero) included, a freedom of the degenerate pair, not of the zero:
#: (geometry, basis, the runs of ZERO_MODE_RUNS it takes).
FORCED_TRIPLETS = {
    'h2': ('H 0 0 0; H 0 0 0.74', '6-31g', tuple(ZERO_MODE_RUNS)),
    'lih': ('Li 0 0 0; H 0 0 1.6', 'sto-3g',
            tuple(run for run in ZERO_MODE_RUNS if not run.startswith('qs'))),
}
#: Ha. Spin-orbital eqp and SigC against Quack: 1e-15 measured (two to four
#: spin orbitals); x1000 for another BLAS.
SMALL_EQP = 1e-12
#: Relative. Z there: 6e-16 measured.
SMALL_Z_REL = 1e-11
#: Ha. The restricted driver against G_evParquet and against the
#: spin-orbital driver (eqp, EcRPA, the spectra): 6e-13 at most measured
#: (water cut to six orbitals, full BSEs); x16 for another BLAS.
CLOSED_SHELL_EQP = 1e-11
#: Z there, absolute: 5.7e-13 measured.
CLOSED_SHELL_Z = 1e-10
#: Two-body iterations of the restricted and spin-orbital osPA compared at the
#: defaults (their converged loops stop at other counts: each driver tests the
#: change of its own vertices, the spin-adapted four or the spin-orbital two).
DEFAULTS_CAP = 3
#: Ha. The H atom's UHF energies by hand against pyscf.scf.uhf.UHF: 4e-15
#: measured (the hand-made SCF iterates fifty times).
UHF_BY_HAND = 1e-10
#: Fixed-point steps of the hand-made H-atom UHF (one electron: its alpha
#: Fock matrix depends on the alpha 1s alone).
UHF_BY_HAND_STEPS = 50
#: Ha. The proper UHF's beta 1s lies above the alpha 1s by more than this
#: (0.59 Ha at 6-31G; HF1e's are degenerate).
BETA_1S_GAP = 0.5

#: Spin-orbital systems: (geometry, basis, charge, spin, how the UHF is made).
OPEN_SHELLS = {
    'h_atom': ('H 0 0 0', '6-31g', 0, 1, uhf.UHF),
    'h_sto': ('H 0 0 0', 'sto-3g', 0, 1, uhf.UHF),
    'h2m': ('H 0 0 0; H 0 0 0.74', 'sto-3g', -1, 1, scf.UHF),
}
#: Closed shells: (geometry, basis, orbitals kept: None for all).
CLOSED_SHELLS = {
    'he': ('He 0 0 0', '6-31g', None),
    'h2': ('H 0 0 0; H 0 0 0.74', '6-31g', None),
    'water_cut6': ('O 0 0 0; H 0.7571 0 0.5861; H -0.7571 0 0.5861', '6-31g', 6),
}

T, F = True, False
_LIN = dict(max_diis_1b=1, max_diis_2b=1, eta_1b=50.0, eta_2b=1.0, qp_solver='linearized')
_FLEX = dict(_LIN, max_it_1b=1, conv_1b=1e-4, max_it_2b=1, conv_2b=1e-6)
_OS = dict(_LIN, max_it_1b=1, conv_1b=1e-4, max_it_2b=40, conv_2b=1e-9)
_EV = dict(_LIN, max_it_1b=3, conv_1b=1e-12, max_it_2b=40, conv_2b=1e-9)
_QS = dict(tda_eh=T, tda_pp=T, max_diis_1b=1, max_diis_2b=1, eta_1b=50.0, eta_2b=1.0,
           max_it_1b=3, conv_1b=1e-12, max_it_2b=40, conv_2b=1e-9)
#: The spin-orbital cases: (scheme, options, Quack's one-body verdict for evPA
#: and qsPA, the flags the run must raise).
SO_CASES = {}
for _mol in OPEN_SHELLS:
    SO_CASES.update({
        f'{_mol}/small_flex_tda': ('os', dict(_FLEX, tda_eh=T, tda_pp=T), None, ()),
        f'{_mol}/small_os_tda': ('os', dict(_OS, tda_eh=T, tda_pp=T), None, ()),
        f'{_mol}/small_ev_tda': ('ev', dict(_EV, tda_eh=T, tda_pp=T), _mol == 'h_sto', ()),
        f'{_mol}/small_qs_tda': ('qs', _QS, True, ()),
    })
for _mol in ('h_atom', 'h_sto'):
    SO_CASES.update({
        f'{_mol}/small_flex_ehtda_pprpa': ('os', dict(_FLEX, tda_eh=T, tda_pp=F), None, ()),
        f'{_mol}/small_os_ehtda_pprpa': ('os', dict(_OS, tda_eh=T, tda_pp=F), None, ()),
    })
SO_CASES.update({
    'h_sto/small_flex_full': ('os', dict(_FLEX, tda_eh=F, tda_pp=F), None,
                              ('parquet_eh_zero_mode',)),
    'h_sto/small_os_full': ('os', dict(_OS, tda_eh=F, tda_pp=F), None,
                            ('parquet_eh_zero_mode',)),
})

#: The closed-shell cases against G_evParquet: (scheme, options), s_1b = 1e8,
#: damping 0 (G_evParquet's alpha = 1), linearized.
_G_LIKE = dict(eta_1b=1e8, eta_2b=1.0, qp_solver='linearized', damping=0.0)
CLOSED_CASES = {}
for _mol in CLOSED_SHELLS:
    CLOSED_CASES.update({
        f'{_mol}/flex_tda': ('flex', dict(_G_LIKE, tda_eh=T, tda_pp=T)),
        f'{_mol}/flex_rpa': ('flex', dict(_G_LIKE, tda_eh=F, tda_pp=F)),
        f'{_mol}/os_tda_cap3_d0': ('os', dict(_G_LIKE, tda_eh=T, tda_pp=T, max_it_2b=3,
                                              conv_2b=1e-12)),
        f'{_mol}/os_rpa_cap3_d0': ('os', dict(_G_LIKE, tda_eh=F, tda_pp=F, max_it_2b=3,
                                              conv_2b=1e-12)),
    })

#: Where the numbers come from: Quack's G_evParquet ('os', 'ev') and
#: G_qsParquet ('qs') of the paper-era tree (git e3528a40), in copies that
#: print their numbers in full precision and are otherwise unchanged, one
#: thread: on each open shell's UHF as `open_shell` lays it out, and on the
#: interleaved spin-orbital problem of each closed shell (`closed_shell`,
#: `interleaved`), with the options of the case.
SOURCES = {**{case: (f"{'G_qsParquet' if SO_CASES[case][0] == 'qs' else 'G_evParquet'}, "
                     'git e3528a40, full-precision prints, one thread') for case in SO_CASES},
           **dict.fromkeys(CLOSED_CASES, 'G_evParquet on the interleaved spin orbitals, '
                                         'git e3528a40, full-precision prints, one thread')}

# Quack's spin-orbital numbers (Ha), as in test_parquet_so_vs_quack.QUACK.
QUACK_SO = {
    'h_atom/small_flex_tda': {
        'n_it_1b': 1,
        'n_it_2b': [1],
        'err_1b': [0.06448274748683125],
        'eqp': [-0.4982329107290701, 0.05932862050994798, 0.9161927617794764, 1.1725559538498673],
        'sigc': [0.0, -0.037281426227631934, 0.0, 0.23398666447727354],
        'z': [1.0, 0.9574324298725915, 1.0, 0.2755830022659014],
        'eqp_lin': [-0.4982329107290701, 0.05932862050994798, 0.9161927617794764,
                    1.1725559538498673],
    },
    'h_atom/small_os_tda': {
        'n_it_1b': 1,
        'n_it_2b': [36],
        'err_1b': [0.04758187074696041],
        'eqp': [-0.4982329107290701, 0.0712316213935244, 0.9161927617794764, 1.1556550771099965],
        'sigc': [0.0, -0.02436835704244831, 0.0, 0.12661658896931516],
        'z': [1.0, 0.9763253869441266, 1.0, 0.3757949186144284],
        'eqp_lin': [-0.4982329107290701, 0.0712316213935244, 0.9161927617794764,
                    1.1556550771099965],
    },
    'h_atom/small_ev_tda': {
        'n_it_1b': 3,
        'n_it_2b': [36, 28, 27],
        'err_1b': [0.04758187074696041, 0.03169766719730438, 0.012745411263904272],
        'eqp': [-0.4982329107290701, 0.07156278061306356, 0.9161927617794764, 1.2000981555712051],
        'sigc': [0.0, -0.024016595099878392, 0.0, 0.13619497717673584],
        'z': [1.0, 0.9768364875019526, 1.0, 0.6756853381512833],
        'eqp_lin': [-0.4982329107290701, 0.07156278061306356, 0.9161927617794764,
                    1.2000981555712051],
    },
    'h_atom/small_flex_ehtda_pprpa': {
        'n_it_1b': 1,
        'n_it_2b': [1],
        'err_1b': [0.06448274748683125],
        'eqp': [-0.4982329107290701, 0.059328620509947985, 0.9161927617794764, 1.1725559538498673],
        'sigc': [0.0, -0.03728142622763193, 0.0, 0.23398666447727356],
        'z': [1.0, 0.9574324298725915, 1.0, 0.2755830022659014],
        'eqp_lin': [-0.4982329107290701, 0.059328620509947985, 0.9161927617794764,
                    1.1725559538498673],
    },
    'h_atom/small_os_ehtda_pprpa': {
        'n_it_1b': 1,
        'n_it_2b': [36],
        'err_1b': [0.047581870746960186],
        'eqp': [-0.4982329107290701, 0.07123162139352439, 0.9161927617794764, 1.1556550771099963],
        'sigc': [0.0, -0.024368357042448316, 0.0, 0.12661658896931513],
        'z': [1.0, 0.9763253869441266, 1.0, 0.37579491861442826],
        'eqp_lin': [-0.4982329107290701, 0.07123162139352439, 0.9161927617794764,
                    1.1556550771099963],
    },
    'h2m/small_flex_tda': {
        'n_it_1b': 1,
        'n_it_2b': [1],
        'err_1b': [0.02417043330165025],
        'eqp': [-0.11813608583917178, 0.08515754151928424, 0.695313925215902, 1.3687949964047128],
        'sigc': [-0.10944061263551343, 0.0, 0.0246350359270093, 0.0],
        'z': [0.20178217949863161, 1.0, 0.9811405744754895, 1.0],
        'eqp_lin': [-0.11813608583917178, 0.08515754151928424, 0.695313925215902,
                    1.3687949964047128],
    },
    'h2m/small_os_tda': {
        'n_it_1b': 1,
        'n_it_2b': [40],
        'err_1b': [0.04191011894255493],
        'eqp': [-0.13796303943846733, 0.08515754151928424, 0.6859848417271046, 1.3687949964047128],
        'sigc': [-0.3256945962234869, 0.0, 0.014965388713454448, 0.0],
        'z': [0.12867919648810142, 1.0, 0.9917116151824323, 1.0],
        'eqp_lin': [-0.13796303943846733, 0.08515754151928424, 0.6859848417271046,
                    1.3687949964047128],
    },
    'h2m/small_ev_tda': {
        'n_it_1b': 3,
        'n_it_2b': [40, 40, 40],
        'err_1b': [0.04191011894255493, 0.039570104719572397, 0.04337041135699421],
        'eqp': [-0.22090355551503393, 0.08515754151928424, 0.6852352454477797, 1.3687949964047128],
        'sigc': [-0.6618909498986406, 0.0, 0.014195732675071963, 0.0],
        'z': [0.18862719763465668, 1.0, 0.992675324062238, 1.0],
        'eqp_lin': [-0.22090355551503393, 0.08515754151928424, 0.6852352454477797,
                    1.3687949964047128],
    },
    'h_sto/small_flex_tda': {
        'n_it_1b': 1,
        'n_it_2b': [1],
        'err_1b': [0.0],
        'eqp': [-0.4665818495572752, 0.30802409436262235],
        'sigc': [0.0, 0.0],
        'z': [1.0, 1.0],
        'eqp_lin': [-0.4665818495572752, 0.30802409436262235],
    },
    'h_sto/small_flex_full': {
        'n_it_1b': 1,
        'n_it_2b': [1],
        'err_1b': [0.0],
        'eqp': [-0.4665818495572752, 0.30802409436262235],
        'sigc': [0.0, 0.0],
        'z': [1.0, 1.0],
        'eqp_lin': [-0.4665818495572752, 0.30802409436262235],
    },
    'h_sto/small_os_tda': {
        'n_it_1b': 1,
        'n_it_2b': [1],
        'err_1b': [0.0],
        'eqp': [-0.4665818495572752, 0.30802409436262235],
        'sigc': [0.0, 0.0],
        'z': [1.0, 1.0],
        'eqp_lin': [-0.4665818495572752, 0.30802409436262235],
    },
    'h_sto/small_os_full': {
        'n_it_1b': 1,
        'n_it_2b': [1],
        'err_1b': [0.0],
        'eqp': [-0.4665818495572752, 0.30802409436262235],
        'sigc': [0.0, 0.0],
        'z': [1.0, 1.0],
        'eqp_lin': [-0.4665818495572752, 0.30802409436262235],
    },
    'h_sto/small_ev_tda': {
        'n_it_1b': 1,
        'n_it_2b': [1],
        'err_1b': [0.0],
        'eqp': [-0.4665818495572752, 0.30802409436262235],
        'sigc': [0.0, 0.0],
        'z': [1.0, 1.0],
        'eqp_lin': [-0.4665818495572752, 0.30802409436262235],
    },
    'h_sto/small_flex_ehtda_pprpa': {
        'n_it_1b': 1,
        'n_it_2b': [1],
        'err_1b': [0.0],
        'eqp': [-0.4665818495572752, 0.30802409436262235],
        'sigc': [0.0, 0.0],
        'z': [1.0, 1.0],
        'eqp_lin': [-0.4665818495572752, 0.30802409436262235],
    },
    'h_sto/small_os_ehtda_pprpa': {
        'n_it_1b': 1,
        'n_it_2b': [1],
        'err_1b': [0.0],
        'eqp': [-0.4665818495572752, 0.30802409436262235],
        'sigc': [0.0, 0.0],
        'z': [1.0, 1.0],
        'eqp_lin': [-0.4665818495572752, 0.30802409436262235],
    },
    'h_atom/small_qs_tda': {
        'n_it_1b': 1,
        'n_it_2b': [40],
        'err_1b': [1.3322676295501878e-15],
        'eqp': [-0.49823291072906994, 0.06936927648049204, 0.9161927617794756, 1.2360078605289777],
        'sigc': [-8.778143165047872e-96, -0.02676743614658541, -7.285992373922963e-66,
                 0.12904829978083032],
    },
    'h2m/small_qs_tda': {
        'n_it_1b': 1,
        'n_it_2b': [40],
        'err_1b': [8.326672684688674e-17],
        'eqp': [-0.42162663676741896, 0.08515754151928402, 0.6861043472035386, 1.3687949964047128],
        'sigc': [-0.3255737162715063, 0.0, 0.014960855289287008, 0.0],
    },
    'h_sto/small_qs_tda': {
        'n_it_1b': 1,
        'n_it_2b': [1],
        'err_1b': [0.0],
        'eqp': [-0.4665818495572752, 0.30802409436262235],
        'sigc': [0.0, 0.0],
    },
}

# G_evParquet on each closed shell over interleaved spin orbitals (Ha): the
# alpha spin orbitals' eQP, SigC and Z (the beta ones are the same), the eh
# and pp EcRPA and the two-body count.
QUACK_G = {
    'he/flex_tda': {
        'G_eqp_alpha': [-0.8594975736808708, 1.3699861981339694],
        'G_sigc_alpha': [0.05820043660439467, -0.031224340761775674],
        'G_z_alpha': [0.9386365145794484, 0.9567259512651834],
        'G_Ec_eh': 0.0,
        'G_Ec_pp': 2.220446049250313e-16,
        'G_n_it_2b': 1,
    },
    'he/flex_rpa': {
        'G_eqp_alpha': [-0.85784116049019, 1.370698427802682],
        'G_sigc_alpha': [0.060567589303711, -0.030706737854626895],
        'G_z_alpha': [0.929300122377221, 0.9496582667712984],
        'G_Ec_eh': -0.033672644943743,
        'G_Ec_pp': -0.008082420814979008,
        'G_n_it_2b': 1,
    },
    'he/os_tda_cap3_d0': {
        'G_eqp_alpha': [-0.8921690142617357, 1.387092996188063],
        'G_sigc_alpha': [0.02181427182918057, -0.01271534501883077],
        'G_z_alpha': [1.0065710445174705, 1.0040104334517275],
        'G_Ec_eh': -8.881784197001252e-16,
        'G_Ec_pp': -2.220446049250313e-16,
        'G_n_it_2b': 3,
    },
    'he/os_rpa_cap3_d0': {
        'G_eqp_alpha': [-0.8938888157373591, 1.3901420257222512],
        'G_sigc_alpha': [0.020120212463493534, -0.009687246449968701],
        'G_z_alpha': [1.0058448906333683, 1.0031033668693112],
        'G_Ec_eh': -0.04777868995270085,
        'G_Ec_pp': -0.20006453406696156,
        'G_n_it_2b': 3,
    },
    'h2/flex_tda': {
        'G_eqp_alpha': [-0.5973145394208923, 0.24157909012672238, 0.7439780217874034,
                        1.5153657897290524],
        'G_sigc_alpha': [-0.0016064546006608802, 0.003185964943212635, -0.038589474002837236,
                         0.20586434602199366],
        'G_z_alpha': [0.931926459906173, 0.9750184458541348, 0.7967165480386448,
                      0.5389664927302121],
        'G_Ec_eh': -8.881784197001252e-16,
        'G_Ec_pp': 2.220446049250313e-16,
        'G_n_it_2b': 1,
    },
    'h2/flex_rpa': {
        'G_eqp_alpha': [-0.6229001539786764, 0.25823122327378173, 0.7433029550933956,
                        1.5352409690306634],
        'G_sigc_alpha': [-0.029812815655616273, 0.020601170099168654, -0.04227430365999721,
                         0.2083001515326564],
        'G_z_alpha': [0.9084251692166461, 0.9590963833327921, 0.7432396631504267,
                      0.6280800224707125],
        'G_Ec_eh': -0.0642106626022505,
        'G_Ec_pp': -0.011403927750013088,
        'G_n_it_2b': 1,
    },
    'h2/os_tda_cap3_d0': {
        'G_eqp_alpha': [-0.5733013050550921, 0.2199572170944202, 0.7666926462408045,
                        1.479309127682962],
        'G_sigc_alpha': [0.022261112921727493, -0.01857155559598797, -0.010061154561718057,
                         0.24700225400857595],
        'G_z_alpha': [1.0114560262991104, 0.996981558659801, 0.7981437930911904,
                      0.303225259252014],
        'G_Ec_eh': -8.881784197001252e-16,
        'G_Ec_pp': 1.2212453270876722e-15,
        'G_n_it_2b': 3,
    },
    'h2/os_rpa_cap3_d0': {
        'G_eqp_alpha': [-0.5784459721920586, 0.2251751104822337, 0.7595700384911119,
                        1.5359750243659058],
        'G_sigc_alpha': [0.01751184084782537, -0.013438264064867391, -0.01683971571086403,
                         0.23479822827368457],
        'G_z_alpha': [0.991984214040926, 0.98953294806705, 0.8998284813491954, 0.5603245823353223],
        'G_Ec_eh': -0.05965513377506326,
        'G_Ec_pp': -0.26796671913580605,
        'G_n_it_2b': 3,
    },
    'water_cut6/flex_tda': {
        'G_eqp_alpha': [-20.436832414126766, -1.2948294521005814, -0.6895382167541069,
                        -0.5374235091603151, -0.46768388422448853, 0.20749379669625403],
        'G_sigc_alpha': [0.1808891023112449, 0.04640080609370584, 0.021455715978268054,
                         0.02450443680209495, 0.0364007223980402, 0.003782192552161752],
        'G_z_alpha': [0.6832987755993524, 1.3241530921750246, 0.9572854245692552,
                      0.9458602078057248, 0.9254497766306915, 0.9966795010278672],
        'G_Ec_eh': 0.0,
        'G_Ec_pp': -1.1368683772161603e-13,
        'G_n_it_2b': 1,
    },
    'water_cut6/flex_rpa': {
        'G_eqp_alpha': [-20.438118831250723, -1.298067457068848, -0.6902867372109506,
                        -0.5378849704896724, -0.46837445768350555, 0.20938750325641448],
        'G_sigc_alpha': [0.1782511217994973, 0.046028732498954325, 0.020665064643966023,
                         0.024080460239355797, 0.03562178502401319, 0.0056912522479503425],
        'G_z_alpha': [0.686194194851497, 1.2645094214021666, 0.9576899014747872, 0.943350340094757,
                      0.9263002101439242, 0.9950956483926574],
        'G_Ec_eh': -0.007002459909415393,
        'G_Ec_pp': -0.0013838208433014643,
        'G_n_it_2b': 1,
    },
    'water_cut6/os_tda_cap3_d0': {
        'G_eqp_alpha': [-20.434704262242615, -1.3197926192297726, -0.6918627931608002,
                        -0.542002691261156, -0.47510614496670467, 0.20340335965951686],
        'G_sigc_alpha': [0.18447812026705, 0.03210425217511142, 0.018791677653613222,
                         0.019198124007076683, 0.027599083992971854, -0.00032066495588006213],
        'G_z_alpha': [0.6815412788827292, 1.1362545852254053, 0.9692943923768279,
                      0.9687711975448917, 0.9516540359431626, 1.0004312762423437],
        'G_Ec_eh': -7.105427357601002e-15,
        'G_Ec_pp': -1.1368683772161603e-13,
        'G_n_it_2b': 3,
    },
    'water_cut6/os_rpa_cap3_d0': {
        'G_eqp_alpha': [-20.438230012493154, -1.3247790771633488, -0.6929706194125929,
                        -0.5427619896746093, -0.47614596998371034, 0.20444225818984288],
        'G_sigc_alpha': [0.17688560445578516, 0.0304171580581815, 0.01762163218183995,
                         0.018421148046497905, 0.026447196065648663, 0.0007185717118823666],
        'G_z_alpha': [0.6908629118696602, 1.0353414919759818, 0.9707864370723358,
                      0.9684136475353479, 0.9537855956686502, 0.9993369727672584],
        'G_Ec_eh': -0.0024108642531217583,
        'G_Ec_pp': -0.010992230550613158,
        'G_n_it_2b': 3,
    },
}


@functools.lru_cache(maxsize=None)
def open_shell(name):
    """(mean field, its spin-orbital reference) of an OPEN_SHELLS system."""
    atom, basis, charge, spin, kind = OPEN_SHELLS[name]
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = kind(mol)
    mf.conv_tol = SCF_CONV_TOL
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mf.kernel()
    assert mf.converged, name
    return mf, spin_orbital.spin_orbital_reference(mf)


@functools.lru_cache(maxsize=None)
def closed_shell(name):
    """(eps, eri, nO) of a CLOSED_SHELLS system: the RHF energies and
    <pq|rs>, cut to the orbitals kept."""
    atom, basis, cut = CLOSED_SHELLS[name]
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = SCF_CONV_TOL
    mf.kernel()
    assert mf.converged, name
    n = mf.mo_coeff.shape[1]
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape((n,) * 4).transpose(0, 2, 1, 3)
    eps = np.asarray(mf.mo_energy)
    if cut is not None:
        eps, eri = eps[:cut], eri[:cut, :cut, :cut, :cut]
    return np.ascontiguousarray(eps), np.ascontiguousarray(eri), mol.nelectron // 2


def interleaved(eps, eri):
    """A closed shell's (eps, eri) over interleaved spin orbitals (alpha 0,
    beta 0, alpha 1, ...) and its (alpha, beta) pairs."""
    n = len(eps)
    spin = np.tile([0, 1], n)
    orbital = np.repeat(np.arange(n), 2)
    same = spin[:, None] == spin[None, :]
    both_spins_kept = same[:, None, :, None] & same[None, :, None, :]
    eri_so = eri[np.ix_(orbital, orbital, orbital, orbital)] * both_spins_kept
    return np.repeat(eps, 2), np.ascontiguousarray(eri_so), np.arange(2 * n).reshape(n, 2)


def quiet(fn, *args, **kwargs):
    """fn(*args, **kwargs) with its warnings silenced (their flags are kept)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def gap(got, want, relative=False):
    """max |got - want| (relative to |want|), NaN where the shapes differ."""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    if got.shape != want.shape:
        return np.nan
    d = np.abs(got - want)
    return float((d / np.abs(want) if relative else d).max(initial=0.0))


def flag_names(result):
    return {entry['flag'] for entry in result.numerical_flags}


def spin_flip_not_above_zero_without_the_tda(case, options, result):
    """The skip reason of a case run without the eh TDA whose eh problem met
    its zero mode at or below zero, or None. H/STO-3G's spin flip, e_beta -
    e_alpha - J, its one eh root, is zero in exact arithmetic and rounds to
    ULPS_OFF (as in Quack's recorded runs), to exactly 0.0 or below zero.
    The full eh problem has no real solution at or below zero, and the run
    is NaN, as Quack's would be. At exactly 0.0 (of either sign, or an Om^2
    that underflows) the eigenvectors, which carry Om^-1/2 (X + Y = Om^-1/2
    Z^T (A - B)^1/2), are 0 times inf, and the run's eh spectrum (its last
    two-body iteration's) holds the 0.0. Below zero A - B is not positive
    definite, (A - B)^1/2 and the spectrum are NaN, and the run is flagged
    'parquet_eh_a_minus_b_not_positive', which a problem of one root owes to
    that root. Quack's numbers, of the flip a few ulps above zero, are not
    this run's."""
    if options.get('tda_eh', True):
        return None
    om = np.asarray(result.spectra.eh.om)
    if np.any(om == 0.0):
        where = ('at exactly 0.0, where the eigenvectors carry Om^-1/2 '
                 '(X + Y = Om^-1/2 Z^T (A - B)^1/2), 0 times inf')
    elif om.size == 1 and 'parquet_eh_a_minus_b_not_positive' in flag_names(result):
        where = 'below zero, where A - B is not positive definite and (A - B)^1/2 NaN'
    else:
        return None
    return (f'{case}: the spin flip, e_beta - e_alpha - J, came out {where} (the mean-field '
            f'energies and J rounded it there; Quack\'s recorded run had it at {ULPS_OFF:.1e}). '
            f'Without the eh TDA the run is NaN, as Quack\'s would be, and has no Quack numbers '
            f'to meet')


@pytest.mark.parametrize('case', sorted(SO_CASES))
def test_spin_orbital_small_systems_reproduce_quack(case):
    """The H atom, H/STO-3G and H2^-: eqp, SigC, Z, eQPlin, the one-body
    changes and the counts against G_evParquet and G_qsParquet. A case
    without the eh TDA whose eh problem has its root at exactly 0.0 or below
    zero (H/STO-3G's spin flip, where rounding can put it) is skipped, with
    the reason: its run is NaN (`spin_flip_not_above_zero_without_the_tda`)."""
    scheme, options, converged, flags = SO_CASES[case]
    mf, ref = open_shell(case.split('/')[0])
    if scheme == 'qs':
        x = spin_orbital.orthogonalization_matrix(ref.ovlp)
        result = quiet(spin_orbital.qs_parquet_so, ref.eps, ref.mo_coeff, ref.ovlp, x,
                       ref.hcore, ref.eri_ao, ref.nO, **options)
    else:
        eri = spin_orbital.spin_orbital_eri(ref.eri_ao, ref.mo_coeff)
        result = quiet(spin_orbital.ev_parquet_so, ref.eps, eri, ref.nO, scheme, **options)
    reason = spin_flip_not_above_zero_without_the_tda(case, options, result)
    if reason:
        pytest.skip(reason)
    quack = QUACK_SO[case]
    assert result.n_it_1b == quack['n_it_1b'] and result.n_it_2b == quack['n_it_2b'], case
    sigc = result.sigc if result.sigc.ndim == 1 else np.diag(result.sigc)
    for got, key in ((result.eqp, 'eqp'), (sigc, 'sigc'), (result.eqp_lin, 'eqp_lin'),
                     ([h['err_1b'] for h in result.history], 'err_1b')):
        if key in quack:
            assert gap(got, quack[key]) <= SMALL_EQP, (case, key, gap(got, quack[key]))
    if 'z' in quack:
        assert gap(result.z, quack['z'], relative=True) <= SMALL_Z_REL, case
    assert result.converged_1b is converged, (case, result.converged_1b)
    assert set(flags) <= flag_names(result), (case, flag_names(result))


def test_the_h_atom_uhf_is_the_proper_one():
    """pyscf.scf.uhf.UHF(mol) of the H atom is the UHF of the review's own SCF
    (alpha F = h + J[Pa] - K[Pa], beta F = h + J[Pa]), not PySCF's HF1e:
    the beta 1s lies far above the occupied alpha 1s."""
    mf, ref = open_shell('h_atom')
    mol = mf.mol
    s, h = mol.intor('int1e_ovlp'), mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    eri_ao = mol.intor('int2e', aosym='s1').transpose(0, 2, 1, 3)
    e_a, c_a = la.eigh(h, s)
    for _ in range(UHF_BY_HAND_STEPS):
        p_a = np.outer(c_a[:, 0], c_a[:, 0])
        j = np.einsum('mlns,ls->mn', eri_ao, p_a)
        k = -np.einsum('mlsn,ls->mn', eri_ao, p_a)
        e_a, c_a = la.eigh(h + j + k, s)
    p_a = np.outer(c_a[:, 0], c_a[:, 0])
    e_b, _ = la.eigh(h + np.einsum('mlns,ls->mn', eri_ao, p_a), s)
    by_hand = np.concatenate(([e_a[0]], np.sort(np.concatenate((e_a[1:], e_b)))))
    assert gap(ref.eps, by_hand) <= UHF_BY_HAND, (ref.eps, by_hand)
    assert ref.nO == 1 and ref.eps[1] - ref.eps[0] > BETA_1S_GAP, ref.eps


def spin_flip_at(value):
    """The spin-orbital driver's eh BSE solver with its roots below
    ZERO_MODE_BELOW (H/STO-3G's spin flip) put at `value`."""
    def solve(A, B, tda):
        solution = spin_orbital_two_body.solve_eh_bse_so(A, B, tda)
        om = np.where(np.abs(solution.om) < ZERO_MODE_BELOW, value, solution.om)
        return solution._replace(om=om)
    return solve


def lowest_triplet_at(value):
    """The closed-shell driver's eh BSE blocks and solver, with the lowest
    triplet root put at `value` (the blocks tell the solver which spin it
    solves): (blocks, solve)."""
    spins = []

    def blocks(eps, eri, nO, spin, tda):
        spins.append(spin)
        return restricted_bse.eh_bse_blocks(eps, eri, nO, spin, tda)

    def solve(A, B, tda):
        om, xpy, xmy, ec = restricted_bse.solve_eh_bse(A, B, tda)
        if spins.pop() == 'triplet':
            om = om.copy()
            om[0] = value
        return om, xpy, xmy, ec
    return blocks, solve


def assert_same_run(got, want, label, bar, z_bar):
    """Two parquet runs finite, with the same flags, and equal to `bar` in
    eqp, eqp_lin and SigC and to `z_bar` in Z (None: to the bit). qsPA's
    SigC matrix is held in the AO basis, S C SigC C^T S (`sigc_ao`), which no
    orbital sign changes: in the final orbitals each element carries the
    signs LAPACK gave two eigenvectors of the effective Fock matrix, and runs
    a few ulps apart can get opposite ones (ADA's OpenBLAS flipped one of
    H2's sigma_g orbitals between the triplet at 0.0 and at 2.2e-16, and with
    it the sign of their element, 4.7e-3)."""
    assert flag_names(got) == flag_names(want), (label, flag_names(got), flag_names(want))
    for key in ('eqp', 'eqp_lin', 'sigc', 'z'):
        a, b = getattr(got, key), getattr(want, key)
        if key == 'sigc' and np.ndim(b) == 2:
            a, b = got.sigc_ao, want.sigc_ao
        if b is None:
            assert a is None, (label, key)
            continue
        assert np.isfinite(a).all() and np.isfinite(b).all(), (label, key, a, b)
        limit = z_bar if key == 'z' else bar
        if limit is None:
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), (label, key, a, b)
        else:
            assert gap(a, b) <= limit, (label, key, gap(a, b))


@pytest.mark.parametrize('run', sorted(ZERO_MODE_RUNS))
def test_the_spin_flip_at_exactly_zero(run, monkeypatch):
    """H/STO-3G's spin flip is an exact zero that rounding puts at 2.2e-16 or
    at exactly 0.0, where Quack's regularized 1/d is 0/0 and made the whole
    run NaN. Forced to each of SPIN_FLIP_VALUES (exactly 0.0 of either sign,
    +-1e-16, +-ULPS_OFF), run_parquet_so (s_2b = 1, the eh TDA by default)
    gives the run as computed to the bit: finite, unflagged, the same eqp,
    eqp_lin, SigC and Z. Every weight at the zero is 0, its limit, as a few
    ulps off, where 1 - exp rounds to 0, and so is every slope in Z, the
    value of Quack's formula there (the flip's residues vanish as well:
    Sigma = 0 and Z = 1)."""
    scheme, options = ZERO_MODE_RUNS[run]
    mf, _ = open_shell('h_sto')
    computed = quiet(spin_orbital.run_parquet_so, mf, scheme, eta_2b=1.0, **options)
    assert abs(computed.spectra.eh.om[0]) < ZERO_MODE_BELOW, computed.spectra.eh.om
    for value in SPIN_FLIP_VALUES:
        with monkeypatch.context() as patch:
            patch.setattr(spin_orbital, 'solve_eh_bse_so', spin_flip_at(value))
            forced = quiet(spin_orbital.run_parquet_so, mf, scheme, eta_2b=1.0, **options)
        assert forced.spectra.eh.om.tobytes() == np.array([value]).tobytes(), (run, value)
        assert not flag_names(forced), (run, value, flag_names(forced))
        assert_same_run(forced, computed, (run, value), None, None)


@functools.lru_cache(maxsize=None)
def forced_triplet_rhf(name):
    """The RHF of a FORCED_TRIPLETS closed shell."""
    atom, basis, _ = FORCED_TRIPLETS[name]
    mf = scf.RHF(gto.M(atom=atom, basis=basis, verbose=0))
    mf.conv_tol = SCF_CONV_TOL
    mf.kernel()
    assert mf.converged
    return mf


@pytest.mark.parametrize('name, run', [(name, run) for name in sorted(FORCED_TRIPLETS)
                                       for run in sorted(FORCED_TRIPLETS[name][2])])
def test_a_closed_shells_triplet_at_exactly_zero(name, run, monkeypatch):
    """The closed-shell analogue: a closed shell has no zero mode of its own,
    but at the edge of its triplet instability its lowest triplet goes to
    zero. Forced to exactly 0.0, it runs through run_parquet (s_2b = 1)
    finite, with the flags of the run with it forced to ULPS_OFF and its eqp,
    eqp_lin, SigC and Z to rounding (the bosonic denominators move by a few
    ulps). H2's zero pole has no residue (sigma_g -> sigma_u: inversion
    parity); LiH's has one, and its derivative at the pole takes 0, the
    value of Quack's slope a few ulps off, so its Z and eqp_lin stay those of
    the triplet a few ulps off, where the slope's limit 2 s_1b times the
    residue would move them by up to 0.18 and 2.5 mHa
    (`test_parquet_flags.py::test_a_pole_hit_exactly_weighs_nothing`)."""
    scheme, options = ZERO_MODE_RUNS[run]
    results = {}
    for value in (0.0, ULPS_OFF):
        blocks, solve = lowest_triplet_at(value)
        with monkeypatch.context() as patch:
            patch.setattr(restricted, 'eh_bse_blocks', blocks)
            patch.setattr(restricted, 'solve_eh_bse', solve)
            results[value] = quiet(restricted.run_parquet, forced_triplet_rhf(name), scheme,
                                   eta_2b=1.0, **options)
        assert results[value].spectra.eh_triplet.om[0] == value, (name, run, value)
    assert_same_run(results[0.0], results[ULPS_OFF], (name, run), CLOSED_ZERO_MODE_GAP,
                    CLOSED_ZERO_MODE_Z)


def test_an_empty_change_is_gfortrans_maxval():
    """The eigenvalue change of an empty pp branch is gfortran's maxval of an
    empty array, -huge, which the max over the two branches passes over."""
    assert spin_orbital.maxval_abs_diff(np.zeros(0), np.zeros(0)) == spin_orbital.EMPTY_MAXVAL
    assert spin_orbital.EMPTY_MAXVAL == -np.finfo(float).max


@pytest.mark.parametrize('case', sorted(CLOSED_CASES))
def test_restricted_with_an_empty_triplet_branch_reproduces_g_evparquet(case):
    """He, H2 (nO = 1) and water cut to six orbitals (nV = 1) in the
    restricted driver against Quack's G_evParquet on the same closed shell,
    at s_1b = 1e8 where the two self-energies coincide."""
    scheme, options = CLOSED_CASES[case]
    eps, eri, nO = closed_shell(case.split('/')[0])
    result = quiet(restricted.ev_parquet, eps, eri, nO, scheme, **options)
    quack = QUACK_G[case]
    assert result.n_it_2b == [quack['G_n_it_2b']], (case, result.n_it_2b)
    assert gap(result.eqp, quack['G_eqp_alpha']) <= CLOSED_SHELL_EQP, case
    assert gap(result.sigc, quack['G_sigc_alpha']) <= CLOSED_SHELL_EQP, case
    assert gap(result.z, quack['G_z_alpha']) <= CLOSED_SHELL_Z, case
    assert abs(result.ec_eh - quack['G_Ec_eh']) <= CLOSED_SHELL_EQP, case
    assert abs(result.ec_pp - quack['G_Ec_pp']) <= CLOSED_SHELL_EQP, case


def spectra_gaps(r, g):
    """max |Omega_R - Omega_G| of the eh, ee and hh channels, the restricted
    singlets and three copies of the triplets against the spin-orbital ones."""
    out = []
    for channel in ('eh', 'ee', 'hh'):
        om_r = np.concatenate([getattr(r.spectra, f'{channel}_singlet').om]
                              + [getattr(r.spectra, f'{channel}_triplet').om] * 3)
        out.append(gap(np.sort(om_r), np.sort(getattr(g.spectra, channel).om)))
    return out


@pytest.mark.parametrize('case', sorted(CLOSED_CASES))
def test_restricted_with_an_empty_triplet_branch_is_the_spin_orbital_driver(case):
    """The same closed shells against the spin-orbital driver on their
    interleaved spin orbitals (pairs solved as one): eqp, Z, EcRPA and every
    spectrum, the alpha and beta copies the restricted orbital's."""
    scheme, options = CLOSED_CASES[case]
    eps, eri, nO = closed_shell(case.split('/')[0])
    r = quiet(restricted.ev_parquet, eps, eri, nO, scheme, **options)
    eps_so, eri_so, pairs = interleaved(eps, eri)
    g = quiet(spin_orbital.ev_parquet_so, eps_so, eri_so, 2 * nO, scheme, spin_pairs=pairs,
              **options)
    assert np.array_equal(g.eqp[0::2], g.eqp[1::2]), case
    assert gap(r.eqp, g.eqp[0::2]) <= CLOSED_SHELL_EQP, case
    assert gap(r.z, g.z[0::2]) <= CLOSED_SHELL_Z, case
    assert abs(r.ec_eh - g.ec_eh) <= CLOSED_SHELL_EQP, case
    assert abs(r.ec_pp - g.ec_pp) <= CLOSED_SHELL_EQP, case
    assert max(spectra_gaps(r, g)) <= CLOSED_SHELL_EQP, (case, spectra_gaps(r, g))
    assert r.n_it_2b == g.n_it_2b, case
    assert flag_names(r) == flag_names(g), (case, flag_names(r), flag_names(g))


@pytest.mark.parametrize('name', sorted(CLOSED_SHELLS))
@pytest.mark.parametrize('tda', [True, False])
def test_restricted_with_an_empty_triplet_branch_runs_at_the_defaults(name, tda):
    """At the driver's defaults (s_1b = 50, the house's root finder, damping
    0.75) the restricted osPA converges and is finite. Its first DEFAULTS_CAP two-body
    iterations are the spin-orbital driver's at the same damping (the
    spectra and EcRPA: the two-body problem does not see the self-energy);
    the converged loops stop at other counts, each driver measuring its own
    vertices' change."""
    eps, eri, nO = closed_shell(name)
    r = quiet(restricted.ev_parquet, eps, eri, nO, 'os', tda_eh=tda, tda_pp=tda, eta_2b=1.0,
              conv_2b=1e-8, max_it_2b=400)
    assert np.isfinite(r.eqp).all() and np.isfinite(r.z).all(), name
    assert r.converged, (name, r.n_it_2b)
    capped = dict(tda_eh=tda, tda_pp=tda, eta_2b=1.0, max_it_2b=DEFAULTS_CAP, conv_2b=1e-12,
                  damping=PARQUET_VERTEX_DAMPING)
    r = quiet(restricted.ev_parquet, eps, eri, nO, 'os', **capped)
    eps_so, eri_so, pairs = interleaved(eps, eri)
    g = quiet(spin_orbital.ev_parquet_so, eps_so, eri_so, 2 * nO, 'os', spin_pairs=pairs,
              **capped)
    assert np.isfinite(g.eqp).all(), name
    assert abs(r.ec_eh - g.ec_eh) <= CLOSED_SHELL_EQP, name
    assert abs(r.ec_pp - g.ec_pp) <= CLOSED_SHELL_EQP, name
    assert max(spectra_gaps(r, g)) <= CLOSED_SHELL_EQP, (name, spectra_gaps(r, g))


def test_every_case_has_its_quack_numbers():
    """No case runs without a reference, and none is recorded without a run."""
    assert set(SO_CASES) == set(QUACK_SO)
    assert set(CLOSED_CASES) == set(QUACK_G)
    assert set(SOURCES) == set(SO_CASES) | set(CLOSED_CASES)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
