"""The FCI vertex split (`fci_vertex_split`) on the pilot's larger frozen-core cases: the slow
twin of tests/test_fci_vertex_split.py, the same claim and bars on more molecules.

1. H2O/6-31G (245k determinants), Ne/cc-pVDZ (511k) and NH3/6-31G (1.0M), ncore = 1, the HOMO at
   the default depths: IP_exact, G0W0, G0W_TDHF, G0W_FCI and its Z, the lowest FCI-rung
   excitation, the W and Sigma sides, dHx_lin_scr and its IP, and the oracle density IPs against
   the pilot's full-precision numbers (tests/fci_vertex_split_references.py, PILOT and SOURCES),
   each class to the fast file's bar: those of the HF reference alone to PILOT_HF_BAR (1.8e-14 eV
   measured at worst, H2O), those of the FCI ground state to PILOT_FCI_BAR (5.2e-7 eV, NH3's
   G0W_FCI: both Davidsons stop on pyscf's residual tolerance, the square root of the energy
   tolerance, 3.2e-6 Ha in the pilot and 3.2e-7 Ha here, so most of the miss is the pilot's Psi0
   and part of it the module's own), the oracle IPs to PILOT_ORACLE_BAR
   (2.4e-6 eV, NH3, the pilot's own SCF reference) and Z to PILOT_Z_BAR (3.5e-8, NH3); no flag,
   the split adding up to minus the G0W0 error, the FCI rung settled at half depth and the hole's
   main pole within its residual bar.

LiF and BeO in 6-31G (the pilot's split_LiF.txt and split_BeO.txt) need ncore = 2 (ncore = 1 is
38M determinants) and take about 3.5 minutes each on one thread at the pilot's depth 16, HOMO
only; they are left to the cluster panel.

Wall times on a workstation (2 threads): H2O 16 s, Ne 43 s, NH3 108 s; the file runs in about
170 s alone.

Run: python tests/test_fci_vertex_split_slow.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from pyscf import gto

from src.SingleReference.GW.fci_vertex_split import fci_vertex_split
from tests.fci_vertex_split_references import GEOMETRIES, PILOT
from tests.reference_scf import rhf_conv_1e12
from tests.test_fci_vertex_split import FAST_PILOT, assert_pilot

#: The pilot cases the fast file leaves to this one.
SLOW_PILOT = tuple(key for key in PILOT if key not in FAST_PILOT)


@pytest.mark.parametrize('name,basis', SLOW_PILOT)
def test_the_larger_pilot_cases_are_reproduced(name, basis):
    mf = rhf_conv_1e12(gto.M(atom=GEOMETRIES[name], basis=basis, verbose=0))
    homo = mf.mol.nelectron // 2 - 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        rec = fci_vertex_split(mf, 1, orbitals=[homo])
    assert rec['numerical_flags'] == [] and not caught
    assert_pilot(rec['orbitals'][0], PILOT[(name, basis)])


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q', '-p', 'no:cacheprovider']))
