"""Every wrong input of the parquet drivers is refused, before any work but
what a function handed in returns, which is checked where it returns.

The two drivers take one set of loop options, checked one way (`options`): a
ValueError that names the option, raised before an integral is built (front
doors) or a loop starts (lower levels). Held here for both drivers, each
scheme, both levels:

* each option's wrong values (a number that is a bool, a string, an array,
  a complex number, None, NaN, out of range, an integer too large for a
  float; a count that is not whole; a switch that is not a bool; a damping
  outside [0, 1); a vertex builder that is not callable) are refused, and
  its right ones (numpy's numbers too) reach the work;
* qp_solver: one of 'find_qp_root', 'linearized' (None only at a front
  door) or a function the five contract arguments bind to ('newton', no
  longer a name it takes, a class, a function of other arguments refused),
  and nothing at all for qsPA; what a qp_solver function returns, at the
  first one-body step (not three arrays of the orbitals' length, eqp or z
  not real, converged not bool); the closed-shell reg_pa and
  broadening_1b; the schemes' fixed iteration caps; nO; the scheme;
* the references: an unrestricted, open-shell, Kohn-Sham, density-fitted,
  frozen or orbital-less mean field, or one with an environment attached,
  where the driver cannot take it; complex orbitals (both drivers),
  orbitals of no RHF, UHF or GHF layout (spin-orbital driver), orbitals
  that are not spatial ones of the AO basis (closed-shell driver); PySCF's
  one-electron HF1e (what scf.HF, RHF, ROHF, UHF and GHF give for one
  electron); a reference without a virtual orbital (both drivers); for
  qsPA, before any two-electron integral, a complex core Hamiltonian (both
  drivers), one the spin-orbital driver cannot put on both spin blocks (of
  another shape, or a GHF's that depends on the spin; osPA does not read
  it), and a reference whose orbital space Quack's canonical
  orthogonalization would shrink (skipped under a PySCF without the switch
  that makes one, as 2.12.1; CI pins 2.13.1);
* the vertex hook's outputs, checked as each vertex comes back (the first
  at the first two-body iteration, after its BSEs and effective integrals,
  before any damping or DIIS): another shape (at the first vertex or a
  later one), a complex array, anything that is not an array of real
  numbers (a ragged list too, with the message naming vertex_builder and
  the vertex);
* the shared checks of `options` one by one.

"Before any work" is held by planting a sentinel (`Reached`) in the first
work each entry point does (the two-electron integrals of a front door, the
first two-body iteration or Fock build of a lower level): a refused value
raises ValueError and never reaches it, an accepted one does.

Run: python tests/test_parquet_refusals.py, or under pytest.
"""
import copy
import fractions
import functools
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import ao2mo, dft, gto, scf

from src.Base.environment_seam import attach_environment
from src.environments.environment import PointCharges
from src.SingleReference.Parquet import options, restricted, spin_orbital
from src.SingleReference.Parquet.restricted import static_reducible_vertex
from src.SingleReference.Parquet.spin_orbital import static_reducible_vertex_so
from tests.parquet_quack_newton import quack_newton
from tests.pins import assert_same

WATER = 'O 0 0 0; H 0.7571 0 0.5861; H -0.7571 0 0.5861'
NAN, INF = float('nan'), float('inf')

#: Values refused by every real option, whatever its range.
NOT_A_REAL = (None, 'x', '0.5', b'0.5', True, np.False_, np.array([0.5]), np.array(0.5),
              0.5 + 0j, 10 ** 400, [0.5])
#: Values refused by every count (iteration caps, DIIS lengths).
NOT_A_COUNT = (None, 'x', True, np.True_, 0, -1, 2.5, NAN, INF, 10 ** 400,
               fractions.Fraction(4, 2), np.array([3]))
#: Values refused by every switch.
NOT_A_SWITCH = (None, 'False', 'no', 0, 1, 1.0, np.array(True))

#: option: (refused values, accepted values); the iteration caps, which FLEX
#: and osPA fix, are held per scheme (`UNCAPPED`).
OPTIONS = {
    'tda_eh': (NOT_A_SWITCH, (True, np.False_)),
    'tda_pp': (NOT_A_SWITCH, (False, np.True_)),
    'eta_1b': (NOT_A_REAL + (-1.0, NAN, INF), (0.0, 1e8, np.float32(1.0), 3)),
    'eta_2b': (NOT_A_REAL + (-1.0, NAN, INF), (0.0, np.float64(2.0))),
    'conv_1b': (NOT_A_REAL + (0.0, 1.0, 2.0, -1e-3, NAN), (1e-6, np.float32(1e-3))),
    'conv_2b': (NOT_A_REAL + (0.0, 1.0, NAN, INF), (1e-8,)),
    'max_diis_1b': (NOT_A_COUNT, (1, 4)),
    'max_diis_2b': (NOT_A_COUNT, (1, np.int32(5))),
    'damping': (NOT_A_REAL[1:] + (1.0, -0.1, 1.5, NAN, INF), (None, 0, 0.0, 0.4, np.float32(0.5),
                                                              0.99)),
    'vertex_builder': (('static', 3, np.zeros(3)), (None, static_reducible_vertex)),
}
#: The schemes whose max_it_1b and max_it_2b are not fixed.
UNCAPPED = {'max_it_1b': ('ev', 'qs'), 'max_it_2b': ('os', 'ev', 'qs')}


class Reached(Exception):
    """The sentinel: the checks let the call through to the work."""


def reached(*args, **kwargs):
    raise Reached()


@functools.lru_cache(maxsize=None)
def water():
    """Water/6-31G: its RHF, MO integrals, spin-orbital integrals and the
    lower levels' AO pieces."""
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    ref = spin_orbital.spin_orbital_reference(mf)
    n = mf.mo_coeff.shape[1]
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape((n,) * 4).transpose(0, 2, 1, 3)
    c = np.asarray(mf.mo_coeff)
    s = mol.intor('int1e_ovlp')
    return dict(mf=mf, eps=np.asarray(mf.mo_energy), eri=np.ascontiguousarray(eri), nO=5,
                ref=ref, eri_so=spin_orbital.spin_orbital_eri(ref.eri_ao, ref.mo_coeff),
                qs=dict(eps_hf=np.asarray(mf.mo_energy), c_hf=c, s=s,
                        x=restricted.orthogonalization_matrix(s),
                        hc=mol.intor('int1e_kin') + mol.intor('int1e_nuc'),
                        eri_ao=np.ascontiguousarray(mol.intor('int2e').transpose(0, 2, 1, 3)),
                        nO=5),
                x_so=spin_orbital.orthogonalization_matrix(ref.ovlp))


def stop_two_electron_integrals(monkeypatch, mol):
    """Make every two-electron integral of `mol` the sentinel (the one-electron
    ones stay: the spin-orbital qsPA checks its core Hamiltonian first)."""
    real = mol.intor

    def intor(name, *args, **kwargs):
        if name.startswith('int2e'):
            raise Reached()
        return real(name, *args, **kwargs)
    monkeypatch.setattr(mol, 'intor', intor)
    monkeypatch.setattr(restricted, 'get_two_electron_integrals_chemist', reached)


@pytest.fixture
def sentinels(monkeypatch):
    """Water's fixtures, with the first work of every entry point planted."""
    w = water()
    stop_two_electron_integrals(monkeypatch, w['mf'].mol)
    monkeypatch.setattr(spin_orbital, 'spin_orbital_reference', reached)
    monkeypatch.setattr(restricted, 'two_body_spectra', reached)
    monkeypatch.setattr(restricted, 'hartree_exchange', reached)
    monkeypatch.setattr(spin_orbital, 'two_body_spectra_so', reached)
    monkeypatch.setattr(spin_orbital, 'ghf_fock', reached)
    return w


def entry_points(w, driver, scheme):
    """{level: call(**options)} of `driver` and `scheme` on water."""
    if driver == 'restricted':
        front = functools.partial(restricted.run_parquet, w['mf'], scheme)
        if scheme == 'qs':
            low = functools.partial(restricted.qs_parquet, **w['qs'])
        else:
            low = functools.partial(restricted.ev_parquet, w['eps'], w['eri'], w['nO'], scheme)
    else:
        front = functools.partial(spin_orbital.run_parquet_so, w['mf'], scheme)
        ref = w['ref']
        if scheme == 'qs':
            low = functools.partial(spin_orbital.qs_parquet_so, ref.eps, ref.mo_coeff, ref.ovlp,
                                    w['x_so'], ref.hcore, ref.eri_ao, ref.nO)
        else:
            low = functools.partial(spin_orbital.ev_parquet_so, ref.eps, w['eri_so'], ref.nO,
                                    scheme)
    return {'front': front, 'low': low}


def outcome(call, **kwargs):
    """'refused' (ValueError), 'reached' (the sentinel) or the other
    exception's name."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            call(**kwargs)
    except ValueError:
        return 'refused'
    except Reached:
        return 'reached'
    except Exception as err:          # noqa: BLE001 -- the outcome is the verdict
        return type(err).__name__
    return 'returned'


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
@pytest.mark.parametrize('scheme', ['flex', 'os', 'ev', 'qs'])
def test_every_loop_option(sentinels, driver, scheme):
    wrong = []
    for level, call in entry_points(sentinels, driver, scheme).items():
        for name, (refused, accepted) in OPTIONS.items():
            for value in refused:
                got = outcome(call, **{name: value})
                if got != 'refused':
                    wrong.append((level, name, repr(value), got))
            for value in accepted:
                got = outcome(call, **{name: value})
                if got != 'reached':
                    wrong.append((level, name, repr(value), got))
        for name, schemes in UNCAPPED.items():
            # None is the scheme's own cap (the default)
            not_a_cap = tuple(v for v in NOT_A_COUNT if v is not None)
            values = (not_a_cap, (2, 4.0, None)) if scheme in schemes else (
                not_a_cap + (2,), (1, None))
            for value in values[0]:
                if outcome(call, **{name: value}) != 'refused':
                    wrong.append((level, name, repr(value)))
            for value in values[1]:
                if outcome(call, **{name: value}) != 'reached':
                    wrong.append((level, name, repr(value)))
    assert wrong == [], wrong


def two_arguments(sigma, eps_hf):
    """Not a qp_solver function: two arguments where five are handed."""
    return None


class SolverClass:
    """Not a qp_solver function: a class, which a call constructs."""
    def __init__(self, sigma, eps_hf, eqp_lin, z_lin, orbitals):
        pass


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_qp_solver(sentinels, driver):
    """One of QP_SOLVERS for FLEX, osPA and evPA, or a function the five
    contract arguments bind to (None only at a front door); 'newton' is no
    longer a name it takes; qsPA takes none at its front door and has no such
    argument below; the old linearize switch is gone."""
    def keyword_only(*, sigma, eps_hf, eqp_lin, z_lin, orbitals):
        return None
    accepted = options.QP_SOLVERS + (quack_newton, functools.partial(quack_newton),
                                     lambda *args: None)
    refused = ('bogus', 'newton', 'Newton', True, 1, ['newton'], two_arguments, keyword_only,
               SolverClass)
    for scheme in ('flex', 'os', 'ev'):
        calls = entry_points(sentinels, driver, scheme)
        for level, call in calls.items():
            for value in accepted:
                assert outcome(call, qp_solver=value) == 'reached', (scheme, level, value)
            for value in refused:
                assert outcome(call, qp_solver=value) == 'refused', (scheme, level, value)
            want = 'reached' if level == 'front' else 'refused'
            assert outcome(call, qp_solver=None) == want, (scheme, level)
            assert outcome(call, linearize=True) == 'TypeError', (scheme, level)
    calls = entry_points(sentinels, driver, 'qs')
    for value in options.QP_SOLVERS + ('bogus', quack_newton):
        assert outcome(calls['front'], qp_solver=value) == 'refused', value
        assert outcome(calls['low'], qp_solver=value) == 'TypeError', value


def returning_solution(make):
    """A qp_solver function returning make(eqp_lin, z_lin) instead of a
    solution."""
    def solver(sigma, eps_hf, eqp_lin, z_lin, orbitals):
        return make(eqp_lin, z_lin)
    return solver


#: What a qp_solver function may not return, each refused at the first
#: one-body step with the function's label.
WRONG_SOLUTIONS = {
    'not three': lambda e, z: (e, z),
    'not a sequence': lambda e, z: None,
    'a short eqp': lambda e, z: (e[:-1], z, np.ones(len(z), dtype=bool)),
    'a 2-d z': lambda e, z: (e, z[None, :], np.ones(len(z), dtype=bool)),
    'a complex eqp': lambda e, z: (e + 0j, z, np.ones(len(z), dtype=bool)),
    'a string z': lambda e, z: (e, z.astype(str), np.ones(len(z), dtype=bool)),
    'an object eqp': lambda e, z: (e.astype(object), z, np.ones(len(z), dtype=bool)),
    'converged of 0 and 1': lambda e, z: (e, z, np.ones(len(z), dtype=int)),
    'converged as floats': lambda e, z: (e, z, np.ones(len(z))),
}


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_what_a_qp_solver_function_returns(driver):
    """A qp_solver function's (eqp, z, converged) is three arrays of the
    orbitals' length, eqp and z real (float or integer) and converged bool:
    anything else is refused at the first one-body step (H2's osPA), with
    the function's label; the arrays it was handed are accepted back."""
    for what, make in WRONG_SOLUTIONS.items():
        solver = returning_solution(make)
        with pytest.raises(ValueError, match='qp_solver .*returning_solution.<locals>.solver'):
            hook_run(driver, None, qp_solver=solver)
    kept = returning_solution(lambda e, z: (e, z, np.ones(len(z), dtype=bool)))
    result = hook_run(driver, None, qp_solver=kept)
    assert np.array_equal(result.eqp, result.eqp_lin) and result.qp_converged.all()


def test_reg_pa_and_broadening(sentinels):
    """The closed-shell qsPA's static self-energy: SRG (reg_pa True, eta_1b)
    or eta-broadened (reg_pa False, broadening_1b required, eta_1b not read);
    R_evParquet has no eta form."""
    calls = entry_points(sentinels, 'restricted', 'qs')
    for call in calls.values():
        for value in NOT_A_SWITCH:
            assert outcome(call, reg_pa=value) == 'refused', value
        assert outcome(call, reg_pa=np.True_) == 'reached'
        assert outcome(call, broadening_1b=0.05) == 'refused'
        for value in NOT_A_REAL + (0, 0.0, -1.0, NAN, INF):
            assert outcome(call, reg_pa=False, broadening_1b=value) == 'refused', value
        for value in (0.05, np.float32(0.1), 1):
            assert outcome(call, reg_pa=False, broadening_1b=value) == 'reached', value
        assert outcome(call, reg_pa=False, broadening_1b=0.05, eta_1b=None) == 'reached'
        for value in (-1.0, NAN, 'x'):
            assert outcome(call, reg_pa=False, broadening_1b=0.05, eta_1b=value) == 'refused'
    for scheme in ('flex', 'os', 'ev'):
        front = entry_points(sentinels, 'restricted', scheme)['front']
        assert outcome(front, reg_pa=False) == 'refused', scheme
        assert outcome(front, broadening_1b=0.05) == 'refused', scheme
        assert outcome(front, reg_pa='no') == 'refused', scheme


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_occupied_count_and_scheme(sentinels, driver):
    w = sentinels
    if driver == 'restricted':
        n = len(w['eps'])
        ev = functools.partial(restricted.ev_parquet, w['eps'], w['eri'])
        qs = functools.partial(restricted.qs_parquet,
                               **{k: v for k, v in w['qs'].items() if k != 'nO'})
        good = (5, 5.0, np.int64(5))
    else:
        ref = w['ref']
        n = len(ref.eps)
        ev = functools.partial(spin_orbital.ev_parquet_so, ref.eps, w['eri_so'])
        qs = functools.partial(spin_orbital.qs_parquet_so, ref.eps, ref.mo_coeff, ref.ovlp,
                               w['x_so'], ref.hcore, ref.eri_ao)
        good = (10, 10.0, np.int64(10))
    for value in (5.7, True, '5', 0, n, -1, None, 10 ** 400, fractions.Fraction(10, 2)):
        assert outcome(ev, nO=value) == 'refused', value
        assert outcome(qs, nO=value) == 'refused', value
    for value in good:
        assert outcome(ev, nO=value) == 'reached', value
        assert outcome(qs, nO=value) == 'reached', value
    front = (restricted.run_parquet if driver == 'restricted' else spin_orbital.run_parquet_so)
    assert outcome(front, mf=w['mf'], scheme='bogus') == 'refused'
    assert outcome(ev, nO=good[0], scheme='qs') == 'refused'
    assert outcome(front, mf=w['mf'], scheme='os', frozen=1) == 'NotImplementedError'


#: Frozen orbitals as a front door's keyword or the mean field's own
#: mf.frozen that mean none (Quack's nC = nR = 0): None, or a whole-number 0.
NO_FROZEN = (None, 0, 0.0, np.int64(0))
#: Those that freeze orbitals or are no count: one orbital or more as a list
#: or an array (np.arange(1) == 0 is array([True]), which an `in (None, 0)`
#: test takes for 0), a count, a bool, a string.
FROZEN = (1, np.int64(2), [0], [0, 1], np.arange(1), np.array([0, 1]), np.array(0), True,
          False, '0')


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_frozen_orbitals(sentinels, driver, monkeypatch):
    """Frozen orbitals are not ported: the frozen keyword and mf.frozen take
    None or a zero count, and anything else is refused with
    NotImplementedError before any integral (one orbital given as an array
    too, which was run all-electron, and two, which numpy's ambiguous truth
    value turned into a ValueError)."""
    front = entry_points(sentinels, driver, 'os')['front']
    mf = sentinels['mf']
    wrong = []
    for value in NO_FROZEN + FROZEN:
        want = 'reached' if any(value is v for v in NO_FROZEN) else 'NotImplementedError'
        monkeypatch.setattr(mf, 'frozen', None, raising=False)
        got = outcome(front, frozen=value)
        if got != want:
            wrong.append(('frozen=', repr(value), got))
        monkeypatch.setattr(mf, 'frozen', value)
        got = outcome(front)
        if got != want:
            wrong.append(('mf.frozen =', repr(value), got))
    assert wrong == [], wrong


def test_the_shapes_of_the_lower_levels(sentinels):
    w = sentinels
    assert outcome(restricted.ev_parquet, eps_hf=w['eps'], eri=w['eri'][:-1], nO=5) == 'refused'
    assert outcome(restricted.ev_parquet, eps_hf=w['eps'][None], eri=w['eri'], nO=5) == 'refused'
    qs = dict(w['qs'])
    for name in ('s', 'x', 'hc', 'eri_ao'):
        bad = dict(qs, **{name: qs[name][:-1]})
        assert outcome(restricted.qs_parquet, **bad) == 'refused', name
    assert outcome(restricted.qs_parquet, **qs, p_hf=np.zeros((2, 2))) == 'refused'
    ref = w['ref']
    assert outcome(spin_orbital.ev_parquet_so, eps_hf=ref.eps, eri=w['eri_so'][:-1],
                   nO=ref.nO) == 'refused'
    assert outcome(spin_orbital.qs_parquet_so, eps_hf=ref.eps, c_hf=ref.mo_coeff, s=ref.ovlp,
                   x=w['x_so'][:, :-1], hc=ref.hcore, eri_ao=ref.eri_ao,
                   nO=ref.nO) == 'refused'


def test_spin_pairs(sentinels):
    """The (alpha, beta) pairs of ev_parquet_so: integers in [0, nso), each
    once, both occupied or both virtual, with equal eps_hf."""
    ref = sentinels['ref']
    call = functools.partial(spin_orbital.ev_parquet_so, ref.eps, sentinels['eri_so'], ref.nO,
                             'os')
    good = np.arange(len(ref.eps)).reshape(-1, 2)
    assert outcome(call, spin_pairs=good) == 'reached'
    assert outcome(call, spin_pairs=np.zeros((0, 2), dtype=int)) == 'reached'
    for bad in (good.astype(float), good.reshape(-1), good[:, :1], np.array([[0, 26]]),
                np.array([[0, 1], [1, 2]]), np.array([[8, 10]]), np.array([[0, 2]])):
        assert outcome(call, spin_pairs=bad) == 'refused', bad


def one_electron(constructor):
    mol = gto.M(atom='H 0 0 0', basis='6-31g', spin=1, verbose=0)
    mf = constructor(mol)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mf.kernel()
    return mf


@pytest.mark.parametrize('constructor', [scf.HF, scf.RHF, scf.ROHF, scf.UHF, scf.GHF])
def test_pyscf_one_electron_mean_field(constructor):
    """PySCF's HF1e drops the two-electron terms: refused, naming the
    constructors that return it and the class that does not."""
    mf = one_electron(constructor)
    with pytest.raises(ValueError) as err:
        spin_orbital.run_parquet_so(mf, 'os')
    text = str(err.value)
    assert 'pyscf.scf.HF, RHF, ROHF, UHF and GHF' in text and 'pyscf.scf.uhf.UHF(mol)' in text


def test_the_references_each_driver_refuses(monkeypatch):
    """Refused before any two-electron integral: the closed-shell driver
    takes a closed-shell, four-index, gas-phase RHF; the spin-orbital one an
    integer-occupied Hartree-Fock reference with an occupied and a virtual
    spin orbital."""
    mol = gto.M(atom=WATER, basis='6-31g', verbose=0)
    oh = gto.M(atom='O 0 0 0; H 0 0 0.9697', basis='6-31g', spin=1, verbose=0)
    cases = []
    for name, mf in (('UHF', scf.UHF(oh)), ('ROHF', scf.ROHF(oh)), ('RKS', dft.RKS(mol)),
                     ('DF-RHF', scf.RHF(mol).density_fit()), ('GHF', scf.GHF(mol)),
                     ('RHF no orbitals', scf.RHF(mol)), ('RHF in point charges', scf.RHF(mol))):
        if name != 'RHF no orbitals':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mf.kernel()
        if name == 'RHF in point charges':
            attach_environment(mf, PointCharges([[0.0, 0.0, 5.0]], [0.5]))
        cases.append((name, mf))
    stop_two_electron_integrals(monkeypatch, mol)
    stop_two_electron_integrals(monkeypatch, oh)
    want_r = {'UHF': 'NotImplementedError', 'ROHF': 'NotImplementedError',
              'RKS': 'NotImplementedError', 'DF-RHF': 'NotImplementedError',
              'GHF': 'NotImplementedError', 'RHF no orbitals': 'refused',
              'RHF in point charges': 'NotImplementedError'}
    want_g = {'UHF': 'reached', 'ROHF': 'refused', 'RKS': 'NotImplementedError',
              'DF-RHF': 'NotImplementedError', 'GHF': 'reached', 'RHF no orbitals': 'refused',
              'RHF in point charges': 'NotImplementedError'}
    for name, mf in cases:
        assert outcome(restricted.run_parquet, mf=mf, scheme='os') == want_r[name], name
        assert outcome(spin_orbital.run_parquet_so, mf=mf, scheme='os') == want_g[name], name


def test_orbitals_and_core_hamiltonians_a_driver_cannot_read(monkeypatch):
    """Refused before any two-electron integral (water/STO-3G): complex
    orbitals by both drivers (their ports are real; the closed-shell one
    cast them to real numbers, or left them to the integral transformation,
    before), orbitals that are not spatial ones of the AO basis by the
    closed-shell driver (an RHF holding twice its AO rows, or one fewer),
    orbitals of no RHF, UHF or GHF layout by the spin-orbital one; by both
    drivers' qsPA a complex core Hamiltonian (spin-orbit terms), which the
    closed-shell qsPA had cast to real numbers; and, by the spin-orbital
    qsPA, one it cannot put on both spin blocks: of another shape than the
    AO basis's, a GHF's of the AO basis's shape instead of both spins'. osPA
    does not read the core Hamiltonian."""
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rhf = scf.RHF(mol).run(conv_tol=1e-10)
        ghf = scf.GHF(mol).run(conv_tol=1e-10)
    h = scf.hf.get_hcore(mol)
    stop_two_electron_integrals(monkeypatch, mol)

    def changed(mf, **attributes):
        new = copy.copy(mf)
        for name, value in attributes.items():
            setattr(new, name, value)
        return new
    for complex_orbitals in (changed(rhf, mo_coeff=rhf.mo_coeff * np.exp(0.3j)),
                             changed(rhf, mo_energy=rhf.mo_energy + 0j)):
        for front, scheme in ((spin_orbital.run_parquet_so, 'os'),
                              (restricted.run_parquet, 'os'), (restricted.run_parquet, 'qs')):
            with pytest.raises(NotImplementedError, match='complex orbitals'):
                front(complex_orbitals, scheme)
    twice, short = (changed(rhf, mo_coeff=np.vstack([rhf.mo_coeff] * 2)),
                    changed(rhf, mo_coeff=rhf.mo_coeff[:-1]))
    for mf in (twice, short):
        with pytest.raises(NotImplementedError, match='not spatial orbitals of the AO basis'):
            restricted.run_parquet(mf, 'os')
    with pytest.raises(ValueError, match='not those of an RHF, UHF or GHF'):
        spin_orbital.run_parquet_so(short, 'os')
    assert outcome(spin_orbital.run_parquet_so, mf=twice, scheme='os') == 'refused'
    hcores = {'complex': (rhf, h + 1e-3j * np.eye(len(h)), 'is complex'),
              'twice the AO basis': (rhf, np.kron(np.eye(2), h), 'has shape'),
              'an AO short': (rhf, h[:-1, :-1], 'has shape'),
              "the AO basis's, of a GHF": (ghf, h, 'has shape')}
    for name, (mf, hcore, says) in hcores.items():
        mf = changed(mf, get_hcore=lambda *args, hcore=hcore: hcore)
        with pytest.raises(ValueError, match=says):
            spin_orbital.run_parquet_so(mf, 'qs')
        assert outcome(spin_orbital.run_parquet_so, mf=mf, scheme='os') == 'reached', name
        if name == 'complex':
            with pytest.raises(ValueError, match='core Hamiltonian of the RHF is complex'):
                restricted.run_parquet(mf, 'qs')
            assert outcome(restricted.run_parquet, mf=mf, scheme='os') == 'reached', name


#: Short runs of every scheme, as the references test them (the quasiparticle
#: equation linearized: it is not what is held here).
SHORT_RUNS = {'flex': dict(eta_2b=1.0, qp_solver='linearized'),
              'os': dict(eta_2b=1.0, max_it_2b=4, qp_solver='linearized'),
              'ev': dict(eta_2b=1.0, max_it_2b=3, max_it_1b=2, qp_solver='linearized'),
              'qs': dict(eta_2b=1.0, max_it_2b=3, max_it_1b=2)}


@functools.lru_cache(maxsize=None)
def water_minimal():
    """Water's RHF in STO-3G (conv_tol 1e-10)."""
    mf = scf.RHF(gto.M(atom=WATER, basis='sto-3g', verbose=0))
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf


@pytest.mark.parametrize('front', [restricted.run_parquet, spin_orbital.run_parquet_so])
def test_a_closed_shell_rohf_is_its_rhf(front):
    """A closed-shell ROHF is the RHF it is: both front doors run every
    scheme on it and return the RHF's energies; the closed-shell qsPA takes
    its density, which PySCF's ROHF hands out per spin, as the RHF's. The
    ROHF holds the RHF's converged orbitals (water/STO-3G), so the two runs
    differ in the mean field's class alone."""
    mf = water_minimal()
    rohf = scf.ROHF(mf.mol)
    for name in ('mo_coeff', 'mo_energy', 'mo_occ'):
        setattr(rohf, name, np.array(getattr(mf, name)))
    rohf.e_tot, rohf.converged = mf.e_tot, mf.converged
    assert np.shape(rohf.make_rdm1()) == (2,) + np.shape(mf.make_rdm1())
    for scheme, opts in SHORT_RUNS.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            want = front(mf, scheme, **opts)
            got = front(rohf, scheme, **opts)
        assert_same(got.eqp, want.eqp, label=(front.__name__, scheme))


@pytest.mark.parametrize('atom', ['He 0 0 0', 'Ne 0 0 0'])
@pytest.mark.parametrize('scheme', ['os', 'qs'])
def test_a_reference_without_a_virtual_orbital(atom, scheme, monkeypatch):
    """Both front doors refuse it from its occupations, before any
    two-electron integral."""
    mol = gto.M(atom=atom, basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    stop_two_electron_integrals(monkeypatch, mol)
    with pytest.raises(ValueError, match='no virtual spin orbital'):
        spin_orbital.run_parquet_so(mf, scheme)
    with pytest.raises(ValueError, match='at least one occupied and one virtual orbital'):
        restricted.run_parquet(mf, scheme)


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_a_space_quacks_orthogonalization_would_shrink(driver, monkeypatch):
    """qsPA needs the RHF's orbitals to span the space Quack's canonical
    orthogonalization keeps: H2/STO-3G with a ghost 1s 1e-3 Angstrom from one
    H has an overlap eigenvalue of 5.3e-7, below
    PARQUET_ORTHOGONALIZATION_THRESHOLD; PySCF told to keep it (its own
    remove_overlap_zero_eigenvalue switch) gives an RHF with three orbitals,
    which the front doors refuse before any two-electron integral. A PySCF
    without that switch (2.12.1 has none; 2.13.0 has it) skips the test:
    CI pins pyscf 2.13.1 (tools/ci/pinned.txt)."""
    import pyscf
    import pyscf.scf.hf as pyscf_hf
    if not hasattr(pyscf_hf, 'remove_overlap_zero_eigenvalue'):
        pytest.skip(f'pyscf {pyscf.__version__} has no pyscf.scf.hf.'
                    f'remove_overlap_zero_eigenvalue, the switch this test sets; CI pins '
                    f'pyscf 2.13.1 (tools/ci/pinned.txt)')
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74; ghost-H 0 0 0.741', basis='sto-3g', verbose=0)
    with monkeypatch.context() as patch:
        patch.setattr(pyscf_hf, 'remove_overlap_zero_eigenvalue', False)
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mf.kernel()
    assert mf.converged and mf.mo_coeff.shape == (3, 3)
    assert restricted.orthogonalization_matrix(mf.get_ovlp()).shape == (3, 2)
    stop_two_electron_integrals(monkeypatch, mol)
    front = restricted.run_parquet if driver == 'restricted' else spin_orbital.run_parquet_so
    with pytest.raises(ValueError, match="canonical orthogonalization keeps 2 functions"):
        front(mf, 'qs')


def test_without_pyscfs_switch_the_orthogonalization_test_skips(monkeypatch):
    """Where pyscf.scf.hf has no remove_overlap_zero_eigenvalue (pyscf
    2.12.1; deleted here), test_a_space_quacks_orthogonalization_would_shrink
    skips before it builds anything, its reason naming CI's pyscf pin."""
    import pyscf.scf.hf as pyscf_hf
    monkeypatch.delattr(pyscf_hf, 'remove_overlap_zero_eigenvalue', raising=False)
    for driver in ('restricted', 'spin_orbital'):
        with pytest.raises(pytest.skip.Exception, match=r'CI pins pyscf 2\.13\.1'):
            test_a_space_quacks_orthogonalization_would_shrink(driver, monkeypatch)


def test_a_spin_dependent_core_hamiltonian(monkeypatch):
    """A GHF in a Zeeman field: its beta core block differs from its alpha
    one, which Quack's G_qsParquet cannot hold (one Hc on both blocks), so
    qsPA refuses it from the one-electron integrals alone; osPA does not read
    it. A spin-free GHF core Hamiltonian is taken by its alpha block."""
    mol = gto.M(atom=WATER, basis='sto-3g', verbose=0)
    ghf = scf.GHF(mol)
    h1 = scf.hf.get_hcore(mol)
    zeeman = np.block([[h1 + 0.01 * np.eye(len(h1)), np.zeros_like(h1)],
                       [np.zeros_like(h1), h1 - 0.01 * np.eye(len(h1))]])
    ghf.get_hcore = lambda *args: zeeman
    ghf.kernel()
    plain = scf.GHF(mol)
    plain.kernel()
    ref = spin_orbital.spin_orbital_reference(plain)
    assert np.array_equal(ref.hcore, np.asarray(plain.get_hcore())[:len(h1), :len(h1)])
    assert spin_orbital.spin_orbital_reference(ghf).hcore is None
    stop_two_electron_integrals(monkeypatch, mol)
    with pytest.raises(ValueError, match='spin-free core Hamiltonian'):
        spin_orbital.run_parquet_so(ghf, 'qs')
    assert outcome(spin_orbital.run_parquet_so, mf=ghf, scheme='os') == 'reached'


@functools.lru_cache(maxsize=None)
def h2():
    """H2/6-31G (norb 4): RHF, MO integrals and spin-orbital ones, for the
    hook's runs."""
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='6-31g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    ref = spin_orbital.spin_orbital_reference(mf)
    n = mf.mo_coeff.shape[1]
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape((n,) * 4).transpose(0, 2, 1, 3)
    return (np.asarray(mf.mo_energy), np.ascontiguousarray(eri), 1,
            (ref.eps, spin_orbital.spin_orbital_eri(ref.eri_ao, ref.mo_coeff), ref.nO))


def hook_run(driver, builder, **kwargs):
    """H2's osPA through `builder` (three two-body iterations; Quack's
    Newton unless `kwargs` names a qp_solver, cheap: the quasiparticle
    equation is not what is held)."""
    eps, eri, nO, so = h2()
    kwargs.setdefault('qp_solver', quack_newton)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if driver == 'restricted':
            return restricted.ev_parquet(eps, eri, nO, 'os', eta_2b=1.0, max_it_2b=3,
                                         vertex_builder=builder, **kwargs)
        return spin_orbital.ev_parquet_so(*so, 'os', eta_2b=1.0, max_it_2b=3,
                                          vertex_builder=builder, **kwargs)


def returning(driver, wrong, at=None):
    """A builder returning Quack's vertex, but `wrong(vertex)` for the vertex
    named `at` (every vertex when None)."""
    default = static_reducible_vertex if driver == 'restricted' else static_reducible_vertex_so

    def build(name, spectra, eta_2b):
        phi = default(name, spectra, eta_2b)
        return wrong(phi) if at in (None, name) else phi
    return build


WRONG_OUTPUTS = {
    'shape': lambda phi: phi[:-1],
    'complex': lambda phi: phi + 0.5j,
    'complex zero imaginary part': lambda phi: phi.astype(complex),
    'complex64': lambda phi: phi.astype(np.complex64),
    'dict': lambda phi: {'phi': phi},
    'object': lambda phi: object(),
    'string': lambda phi: 'phi',
    'ragged list': lambda phi: [[1.0, 2.0], [3.0]],
    'none': lambda phi: None,
}


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
@pytest.mark.parametrize('kind', sorted(WRONG_OUTPUTS))
@pytest.mark.parametrize('max_diis_2b', [1, 3])
def test_a_wrong_vertex_is_refused(driver, kind, max_diis_2b):
    """At the first vertex built and at the last one of an iteration (pp),
    with and without two-body DIIS."""
    for at in (None, 'pp_triplet' if driver == 'restricted' else 'pp'):
        with pytest.raises(ValueError) as err:
            hook_run(driver, returning(driver, WRONG_OUTPUTS[kind], at),
                     max_diis_2b=max_diis_2b)
        # the message names the hook and the vertex
        first = 'eh_singlet' if driver == 'restricted' else 'eh'
        assert 'vertex_builder returned' in str(err.value), err
        assert repr(at or first) in str(err.value), err


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_a_vertex_of_other_real_types_is_taken(driver):
    """Nested lists and Fortran-ordered or float32 arrays are real arrays of
    the right shape: taken (as float64)."""
    for convert in (lambda phi: phi.tolist(), np.asfortranarray,
                    lambda phi: phi.astype(np.float32)):
        result = hook_run(driver, returning(driver, convert))
        assert np.isfinite(result.eqp).all()


def test_the_shared_checks():
    """`options`' checks one by one: what each takes and what it refuses."""
    for value in (1, 1.5, np.float32(2.0), np.int64(3), fractions.Fraction(1, 2)):
        assert isinstance(options.checked_real('x', value), float)
    for value in NOT_A_REAL:
        with pytest.raises(ValueError, match='x = '):
            options.checked_real('x', value)
    assert options.checked_whole('n', 4.0, 'w') == 4 and options.checked_count('n', 2, 'c') == 2
    for value in (2.5, NAN, INF, True, '3', fractions.Fraction(4, 2), 10 ** 400):
        with pytest.raises(ValueError):
            options.checked_whole('n', value, 'w')
    for value in (0, -3):
        with pytest.raises(ValueError):
            options.checked_count('n', value, 'c')
    assert options.checked_switch('s', np.False_) is False
    for value in NOT_A_SWITCH:
        with pytest.raises(ValueError):
            options.checked_switch('s', value)
    assert options.checked_occupied(3.0, 10, 'occupied', 'n') == 3
    for value in (0, 10, 2.5, True):
        with pytest.raises(ValueError, match='between 1 and n - 1 = 9'):
            options.checked_occupied(value, 10, 'occupied', 'n')
    assert options.checked_damping(None, 0.75) == 0.75 and options.checked_damping(0, 0.75) == 0.0
    for value in (1.0, -0.1, NAN, True, '0.5'):
        with pytest.raises(ValueError, match='damping'):
            options.checked_damping(value, 0.75)
    assert options.checked_qp_solver('find_qp_root') == 'find_qp_root'
    assert options.checked_qp_solver(quack_newton) is quack_newton
    for value in (None, 'NEWTON', 'newton', 0, two_arguments, SolverClass):
        with pytest.raises(ValueError, match='qp_solver'):
            options.checked_qp_solver(value)
    assert options.qp_solver_label('linearized') == 'linearized'
    assert (options.qp_solver_label(quack_newton)
            == 'tests.parquet_quack_newton.quack_newton')
    assert (options.qp_solver_label(functools.partial(quack_newton))
            == 'partial(tests.parquet_quack_newton.quack_newton)')

    class Solver:
        def __call__(self, sigma, eps_hf, eqp_lin, z_lin, orbitals):
            return None
    assert options.qp_solver_label(Solver()) == f'{__name__}.{Solver.__qualname__}'
    assert options.checked_qp_solver(Solver()) is not None
    labels = [options.qp_solver_label(f) for f in (quack_newton, Solver(), two_arguments)]
    assert not set(labels) & set(options.QP_SOLVERS), labels
    good = (np.zeros(3), np.arange(3), np.ones(3, dtype=bool))
    eqp, z, ok = options.checked_qp_solution(quack_newton, good, 3)
    assert eqp.dtype == z.dtype == float and ok.dtype == bool and z.tolist() == [0.0, 1.0, 2.0]
    assert eqp is not good[0]
    for wrong in ((np.zeros(3), np.zeros(3)), (np.zeros(2), np.zeros(3), good[2]),
                  (np.zeros(3), np.zeros(3), np.ones(3))):
        with pytest.raises(ValueError, match='tests.parquet_quack_newton.quack_newton'):
            options.checked_qp_solution(quack_newton, wrong, 3)
    assert options.builder_name(static_reducible_vertex) == 'static_reducible_vertex'
    assert (options.builder_name(functools.partial(static_reducible_vertex_so))
            == 'partial(static_reducible_vertex_so)')

    class Keeper:
        def __call__(self, name, spectra, eta_2b):
            return None
    assert options.builder_name(Keeper()) == f'{__name__}.{Keeper.__qualname__}'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
