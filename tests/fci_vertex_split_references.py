"""Recorded references of the FCI vertex split tests (tests/test_fci_vertex_split.py and its slow
twin): the geometries and the pilot's full-precision numbers. Not a test file.

PILOT holds, per (molecule, basis), the numbers of the pilot that the module ports, an independent
code (its own RPA, TDHF, Lanczos, quasiparticle Newton and density terms), for the HOMO with
ncore = 1 and Lanczos depth 30, in Hartree (Z_ex dimensionless):

    IP_fci            the lowest cation state of a fix_spin_ FCI minus the FCI ground state
    IP_G0W0           G0W0@HF (direct RPA)
    IP_G0Wtdhf        G0W@HF with TDHF screening
    IP_G0Wex          G0W@HF with the FCI screening (Lanczos), Z_ex its pole strength
    minOm_ex          the lowest Ritz value of that screening, minus E0
    dP_ex, dSig_ex    IP_G0Wex - IP_G0W0 and IP_fci - IP_G0Wex
    dHx_lin_scr       the TDHF-screened (J - K/2) shift of the linearized-GW density, and
    IP_G0W0_dHx_lin   IP[G0W0] minus it
    dH_or, dHx_or     IP[G0W0] minus the oracle shifts <p|J[D_FCI - D_HF]|p> and
                      <p|(J - K/2)[D_FCI - D_HF]|p> (the alpha = 1 row of the pilot's dial, whose
                      Hartree-Fock is its own SCF in the active space)

SOURCES gives the provenance. The pilot's own log prints these to four decimals in eV (with
27.211386 eV/Ha); the values here come from rerunning it with every energy printed in Hartree at
full precision, nothing else changed.
"""

#: Angstrom.
GEOMETRIES = {
    'He': 'He 0 0 0',
    'H2': 'H 0 0 0; H 0 0 0.741',
    'H2@2.5': 'H 0 0 0; H 0 0 2.5',
    'LiH': 'Li 0 0 0; H 0 0 1.595',
    'LiH@3.0': 'Li 0 0 0; H 0 0 3.0',
    'HF': 'F 0 0 0; H 0 0 0.917',
    'H2O': 'O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
    'NH3': 'N 0 0 0.1128; H 0 0.9377 -0.2630; H 0.8121 -0.4689 -0.2630; '
           'H -0.8121 -0.4689 -0.2630',
    'Ne': 'Ne 0 0 0',
    'N2': 'N 0 0 0; N 0 0 1.0977',
    'N2@1.6': 'N 0 0 0; N 0 0 1.6',
    'MgO': 'Mg 0 0 0; O 0 0 1.749',
}

#: Ha (Z_ex dimensionless): the pilot's numbers, keys as in the module docstring.
PILOT = {
    ('HF', '6-31g'): dict(
        IP_fci=0.5629716743542517, IP_G0W0=0.5655601230913956, IP_G0Wtdhf=0.5380395349283141,
        IP_G0Wex=0.5402072585629019, Z_ex=0.932139185168586, minOm_ex=0.40254623210532614,
        dP_ex=-0.025352864528493724, dSig_ex=0.0227644157913498,
        dHx_lin_scr=-0.007116612540658408, IP_G0W0_dHx_lin=0.572676735632054,
        dH_or=0.5804413865482972, dHx_or=0.5722826928308946),
    ('LiH', 'cc-pvdz'): dict(
        IP_fci=0.2885588944486893, IP_G0W0=0.29275054772818304, IP_G0Wtdhf=0.26872824069534523,
        IP_G0Wex=0.26278865338241364, Z_ex=0.7542612938931479, minOm_ex=0.12812143081039418,
        dP_ex=-0.029961894345769402, dSig_ex=0.02577024106627568,
        dHx_lin_scr=0.005900951658418777, IP_G0W0_dHx_lin=0.2868495960697643,
        dH_or=0.29696751977075314, dHx_or=0.2862601070692833),
    ('H2O', '6-31g'): dict(
        IP_fci=0.4368448075320579, IP_G0W0=0.4431247351710092, IP_G0Wtdhf=0.4175370798194388,
        IP_G0Wex=0.4193122545661153, Z_ex=0.9211429745270607, minOm_ex=0.311967335813943,
        dP_ex=-0.023812480604893882, dSig_ex=0.017532552965942583,
        dHx_lin_scr=-0.003390210455891219, IP_G0W0_dHx_lin=0.4465149456269004,
        dH_or=0.45427602024836894, dHx_or=0.4455009343033666),
    ('Ne', 'cc-pvdz'): dict(
        IP_fci=0.7678772751554774, IP_G0W0=0.7669732689947891, IP_G0Wtdhf=0.7402747292094698,
        IP_G0Wex=0.7425712247668629, Z_ex=0.9467037589205903, minOm_ex=1.8559665982072318,
        dP_ex=-0.024402044227926245, dSig_ex=0.025306050388614554,
        dHx_lin_scr=-0.004451449150671608, IP_G0W0_dHx_lin=0.7714247181454608,
        dH_or=0.780377387949564, dHx_or=0.7717663641191672),
    ('NH3', '6-31g'): dict(
        IP_fci=0.3614174281227527, IP_G0W0=0.3705575922671024, IP_G0Wtdhf=0.34917656757645105,
        IP_G0Wex=0.350501694934969, Z_ex=0.9174077358323204, minOm_ex=0.2886056669267063,
        dP_ex=-0.020055897332133366, dSig_ex=0.01091573318778366,
        dHx_lin_scr=0.0006479291270000082, IP_G0W0_dHx_lin=0.3699096631401024,
        dH_or=0.37659033143021353, dHx_or=0.36789925852341643),
}

#: Where PILOT comes from.
SOURCES = {
    'scripts': 'exactw_alpha.py (sha256 eec39bc7b03b2bb2f3b09bd917dde18b71fbf140dd20539c484952b'
               '5d8e25843) with its helpers pilot.py (sha256 59d0e66130f539cfc99d518c426369d2acd'
               'daf94f397e56ee5c28f2c07831933), the soft-mode GW study\'s design and practitioner '
               'pilots of 2026-11-18',
    'run': 'python exactw_alpha.py NAME BASIS 1 30 (ncore = 1, Lanczos depth 30, the HOMO), from '
           'a copy whose only edits are HA = 1.0 in pilot.py and repr() in the two result prints',
    'settings': 'RHF conv_tol 1e-12, pyscf default guess; FCI ground state direct_spin1 '
                'conv_tol 1e-11; the cation fix_spin_(ss = 0.75) direct_spin1 conv_tol 1e-11; '
                'quasiparticle roots by Newton from e_p to 1e-10 Ha with a 1e-6 Ha central '
                'difference',
    'software': 'pyscf 2.13.0, numpy 2.1.3, scipy 1.15.3, Python 3.13.5, macOS arm64',
    'threads': 'OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 2',
    'date': '2026-11-20',
}
