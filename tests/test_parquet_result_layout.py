"""What the two parquet drivers return: one layout, apart where it is by design.

`restricted.ParquetResult` and `spin_orbital.SpinOrbitalParquetResult` carry
the same fields, options record and history records for every scheme (FLEX,
osPA, evPA, qsPA through `run_parquet` and `run_parquet_so` on water/6-31G;
FLEX by the default root finder, osPA and evPA with the quasiparticle
equation linearized, whose records have the same keys, each driver writing
them in one statement for every route, in a fiftieth of the finder's time
or less over spin orbitals; evPA by the finder, its roots followed from the
second iteration on, on He/6-31G), except where the drivers differ by
design: the closed-shell (singlet, triplet) EcRPA parts and qsPA's reg_pa
and broadening_1b; the spin-orbital reference kind, spin labels and orbital
indices and the RHF's spin pairs.
Within a result the records agree with the fields: the last history entry
is the result, the channel energy is the sum of its spin parts, the counts
and verdicts are Quack's form (None where a scheme has no such loop), the
quasiparticle records are absent from qsPA, and 'linearized' leaves no
count or verdict. The history's diagnostics are in Quack's order and of
Quack's quantities: the per-vertex changes 'err' in the order of the
vertices, the eigenvalue changes 'err_eig' per channel (the pp one the larger
of its ee and hh branches). The defaults are held by value: s_1b = s_2b = 50
and tau = 1e-4 (the paper's Table III) in every entry point, and the
schemes' iteration caps.

Run: python tests/test_parquet_result_layout.py, or under pytest.
"""
import dataclasses
import functools
import inspect
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import scipy.linalg
from pyscf import gto, scf

from src.SingleReference.Parquet import restricted, spin_orbital
from src.SingleReference.Parquet.options import loop_options
from tests.parquet_quack_newton import quack_newton
from tests.pins import CROSS_PATH_RTOL
from tests.test_parquet_restricted_vs_quack import rhf
from tests.test_parquet_so_vs_quack import mean_field
from tests.test_parquet_spin_adaptation import both_layouts

#: Short runs of every scheme: (options). osPA and evPA solve the
#: quasiparticle equation linearized (by the default finder, the
#: spin-orbital roots took 4.5 and 16 s of these runs: workstation, 2
#: threads, load 2 to 3); test_the_quasiparticle_records holds each route's
#: own records.
SCHEMES = {
    'flex': dict(eta_2b=1.0),
    'os': dict(eta_2b=1.0, max_it_2b=4, conv_2b=1e-12, max_diis_2b=3, qp_solver='linearized'),
    'ev': dict(eta_2b=1.0, max_it_2b=3, conv_2b=1e-12, max_it_1b=2, max_diis_1b=2,
               qp_solver='linearized'),
    'qs': dict(eta_2b=1.0, max_it_2b=3, conv_2b=1e-12, max_it_1b=2, max_diis_1b=2),
}
#: The fields, options and history keys only one driver has, by design.
ONLY_RESTRICTED = {'fields': {'ec_eh_spin', 'ec_pp_spin'},
                   'options_qs': {'reg_pa', 'broadening_1b'},
                   'history': {'ec_eh_spin', 'ec_pp_spin'}}
ONLY_SPIN_ORBITAL = {'fields': {'reference', 'spin', 'orbital'},
                     'options_ev': {'spin_pairs'}}
#: The keys of every two-body iteration record and of every history entry.
TWO_BODY_KEYS = {'err', 'err_2b', 'err_eig', 'diis_rcond', 'seconds'}
HISTORY_KEYS = {'two_body', 'converged_2b', 'n_it_2b', 'om', 'ec_eh', 'ec_pp', 'eqp', 'sigc',
                'z', 'err_1b', 'diis_rcond', 'seconds'}
QP_KEYS = {'eqp_lin', 'qp_solver', 'qp_n_iter', 'qp_followed', 'qp_flipped'}
OPTION_KEYS = {'scheme', 'tda_eh', 'tda_pp', 'eta_1b', 'eta_2b', 'max_it_1b', 'conv_1b',
               'max_it_2b', 'conv_2b', 'max_diis_1b', 'max_diis_2b', 'damping', 'vertex_builder'}


@functools.lru_cache(maxsize=None)
def runs(scheme, qp_solver=None):
    """(restricted, spin-orbital) results of `scheme` on water's RHF."""
    mf = rhf('water')
    extra = {} if qp_solver is None else {'qp_solver': qp_solver}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return (restricted.run_parquet(mf, scheme, **SCHEMES[scheme], **extra),
                spin_orbital.run_parquet_so(mf, scheme, **SCHEMES[scheme], **extra))


def fields(result):
    return {f.name for f in dataclasses.fields(result)}


@pytest.mark.parametrize('scheme', sorted(SCHEMES))
def test_one_layout(scheme):
    r, g = runs(scheme)
    assert fields(r) - fields(g) == ONLY_RESTRICTED['fields']
    assert fields(g) - fields(r) == ONLY_SPIN_ORBITAL['fields']
    qp = set() if scheme == 'qs' else {'qp_solver'}
    assert set(r.options) == OPTION_KEYS | qp | (ONLY_RESTRICTED['options_qs']
                                                 if scheme == 'qs' else set())
    assert set(g.options) == OPTION_KEYS | qp | (ONLY_SPIN_ORBITAL['options_ev']
                                                 if scheme != 'qs' else set())
    for result, only in ((r, ONLY_RESTRICTED['history']), (g, set())):
        want = HISTORY_KEYS | only | (set() if scheme == 'qs' else QP_KEYS)
        for record in result.history:
            assert set(record) == want, (scheme, sorted(set(record) ^ want))
            for iteration in record['two_body']:
                assert set(iteration) == TWO_BODY_KEYS
        assert {'total', 'integrals', 'self_energy'} <= set(result.times), result.times
        assert result.options['scheme'] == scheme


@functools.lru_cache(maxsize=None)
def helium():
    """He/6-31G's RHF (conv_tol 1e-12)."""
    mf = scf.RHF(gto.M(atom='He 0 0 0', basis='6-31g', verbose=0))
    mf.conv_tol = 1e-12
    mf.kernel()
    return mf


def test_the_finders_records_have_the_layout():
    """evPA by the default root finder writes the history keys and options
    of the linearized runs above, in both drivers, at the first iteration
    and at those that follow the roots of the one before (He/6-31G, where
    the finder is cheap over spin orbitals too: water's evPA took 16 s)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r, g = (front(helium(), 'ev', eta_2b=1.0, max_it_1b=3, conv_1b=1e-12)
                for front in (restricted.run_parquet, spin_orbital.run_parquet_so))
    for result, only, options in ((r, ONLY_RESTRICTED['history'], set()),
                                  (g, set(), ONLY_SPIN_ORBITAL['options_ev'])):
        assert result.n_it_1b == 3 and result.options['qp_solver'] == 'find_qp_root'
        assert set(result.options) == OPTION_KEYS | {'qp_solver'} | options
        followed = [h['qp_followed'] for h in result.history]
        assert not followed[0].any() and followed[1].all() and followed[2].all()
        for record in result.history:
            assert set(record) == HISTORY_KEYS | QP_KEYS | only, sorted(record)
            for iteration in record['two_body']:
                assert set(iteration) == TWO_BODY_KEYS


@pytest.mark.parametrize('scheme', sorted(SCHEMES))
def test_the_records_are_the_result(scheme):
    for result in runs(scheme):
        last = result.history[-1]
        assert np.array_equal(last['eqp'], result.eqp)
        assert result.ec_eh == last['ec_eh'] and result.ec_pp == last['ec_pp']
        assert result.n_it_1b == len(result.history)
        assert result.n_it_2b == [h['n_it_2b'] for h in result.history]
        assert result.n_it_2b == [len(h['two_body']) for h in result.history]
        assert result.converged_2b == [h['converged_2b'] for h in result.history]
        assert isinstance(result.nO, int) and isinstance(result.converged, bool)
        assert result.eps_hf.shape == result.eqp.shape == result.z.shape
        for record in result.history:
            for iteration in record['two_body']:
                assert iteration['err_2b'] == max(iteration['err'])
                assert iteration['diis_rcond'] is None if result.options['max_diis_2b'] == 1 \
                    else isinstance(iteration['diis_rcond'], float)
            want = None if result.options['max_diis_1b'] == 1 else float
            assert (record['diis_rcond'] is None) if want is None else isinstance(
                record['diis_rcond'], float)
        if scheme in ('flex', 'os'):
            assert result.converged_1b is None
        if scheme == 'flex':
            assert result.converged_2b == [None]
        if scheme == 'qs':
            assert result.eqp_lin is None and result.qp_converged is None
            assert result.sigc.shape == (len(result.eqp),) * 2
            assert np.all(result.z == 1.0)
            assert result.mo_coeff is not None and result.sigc_ao is not None
            assert result.mo_coeff_two_body is not None
        else:
            assert result.sigc.shape == result.eqp.shape
            assert result.mo_coeff is None and result.sigc_ao is None
    r, _ = runs(scheme)
    for record, energy, parts in ((r, r.ec_eh, r.ec_eh_spin), (r, r.ec_pp, r.ec_pp_spin)):
        singlet, triplet = parts
        assert energy == singlet + 3.0 * triplet
    for record in r.history:
        assert record['ec_eh'] == record['ec_eh_spin'][0] + 3.0 * record['ec_eh_spin'][1]


@pytest.mark.parametrize('qp_solver', [quack_newton, 'linearized', 'find_qp_root'],
                         ids=['function', 'linearized', 'find_qp_root'])
def test_the_quasiparticle_records(qp_solver):
    """A qp_solver function's evaluations of the self-energy are counted
    per orbital (Quack's Newton: one per step), as the finder's are;
    'linearized' records no count and no verdict and returns eQPlin (over
    spin orbitals the first of each RHF pair's, for both); the spin-orbital
    copies of an RHF pair take no work of their own. The options and the
    history record a name as it is and a function by its label."""
    label = (qp_solver if isinstance(qp_solver, str)
             else 'tests.parquet_quack_newton.quack_newton')
    for result, step in zip(runs('flex', qp_solver), (1, 2)):
        record = result.history[0]
        assert record['qp_solver'] == result.options['qp_solver'] == label
        if qp_solver == 'linearized':
            assert record['qp_n_iter'] is None and result.qp_converged is None
            # an RHF pair takes its first spin orbital's eQPlin (the record keeps both's)
            assert np.array_equal(result.eqp[::step], result.eqp_lin[::step])
        else:
            assert record['qp_n_iter'].shape == result.eqp.shape
            assert result.qp_converged.dtype == bool and result.qp_converged.all()
    _, g = runs('flex', qp_solver)
    if qp_solver != 'linearized':
        assert not g.history[0]['qp_n_iter'][1::2].any()
    assert g.reference == 'RHF' and g.spin.tolist() == [0, 1] * (len(g.eqp) // 2)
    assert g.orbital.tolist() == np.repeat(np.arange(len(g.eqp) // 2), 2).tolist()


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_the_qspa_orbitals_and_self_energy(driver, monkeypatch):
    """qsPA's orbitals and static self-energy as the result hands them out:
    mo_coeff_two_body is the orbitals the last two-body loop's integrals were
    transformed with (those at the start of the last one-body iteration, a
    copy), mo_coeff the last diagonalization's (other orbitals); sigc_ao is
    the AO form of the returned sigc in the final orbitals, S C sigc C^T S
    (S over both spin blocks for spin orbitals), to CROSS_PATH_RTOL of its
    size; and the last history entry's 'om' is the eigenvalues of the
    result's spectra, channel by channel (copies)."""
    module, name = ((restricted, 'ao_to_mo_eri') if driver == 'restricted'
                    else (spin_orbital, 'spin_orbital_eri'))
    real, seen = getattr(module, name), []

    def kept(*args):
        seen.append(np.array(args[0] if driver == 'restricted' else args[1]))
        return real(*args)
    monkeypatch.setattr(module, name, kept)
    mf = rhf('water')
    front = restricted.run_parquet if driver == 'restricted' else spin_orbital.run_parquet_so
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = front(mf, 'qs', **SCHEMES['qs'])
    assert len(seen) == result.n_it_1b == SCHEMES['qs']['max_it_1b'], len(seen)
    assert np.array_equal(result.mo_coeff_two_body, seen[-1])
    assert np.abs(result.mo_coeff - result.mo_coeff_two_body).max() > 1e-6
    s = mf.mol.intor_symmetric('int1e_ovlp')
    if driver == 'spin_orbital':
        s = scipy.linalg.block_diag(s, s)
    sc = s @ result.mo_coeff
    want = sc @ result.sigc @ sc.T
    assert np.abs(result.sigc_ao - want).max() <= CROSS_PATH_RTOL * np.abs(want).max()
    channels = (restricted.CHANNEL_NAMES if driver == 'restricted' else ('eh', 'ee', 'hh'))
    om = result.history[-1]['om']
    assert set(om) == set(channels), sorted(om)
    for channel in channels:
        assert np.array_equal(om[channel], getattr(result.spectra, channel).om), channel


@functools.lru_cache(maxsize=None)
def ghf_water():
    """Water's GHF/6-31G (conv_tol 1e-12), the RHF's molecule."""
    mf = scf.GHF(rhf('water').mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mf


def test_the_labels_of_an_open_shell_and_a_ghf():
    """The spin-orbital driver's labels against the mean field it is handed:
    a UHF's (the OH radical) reference 'UHF', spin 0 alpha and 1 beta, and
    orbital the mean field's index, so that eps_hf[k] is
    mf.mo_energy[spin[k]][orbital[k]] (a copy), every (spin, orbital) once;
    a GHF's (water) reference 'GHF', spin -1 throughout and eps_hf[k]
    mf.mo_energy[orbital[k]], every orbital once."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        uhf = mean_field('oh')
        u = spin_orbital.run_parquet_so(uhf, 'flex', eta_2b=1.0, qp_solver='linearized')
        g = spin_orbital.run_parquet_so(ghf_water(), 'flex', eta_2b=1.0,
                                        qp_solver='linearized')
    assert u.reference == 'UHF' and set(u.spin.tolist()) == {0, 1}
    e = np.asarray(uhf.mo_energy)
    assert sorted(zip(u.spin.tolist(), u.orbital.tolist())) == sorted(
        (spin, k) for spin in (0, 1) for k in range(e.shape[1]))
    assert np.array_equal(u.eps_hf, e[u.spin, u.orbital])
    assert g.reference == 'GHF' and g.spin.tolist() == [-1] * len(g.eqp)
    assert sorted(g.orbital.tolist()) == list(range(len(g.eqp)))
    assert np.array_equal(g.eps_hf, np.asarray(ghf_water().mo_energy)[g.orbital])


def flex_from_zero(driver):
    """FLEX on water/6-31G at damping 0 through `driver`'s lower level: one
    two-body iteration from vanishing vertices, which it replaces whole (the
    quasiparticle equation linearized, which comes after it: the spin-orbital
    finder's roots took 8 to 10 s)."""
    (eps, eri, nO), (eps_so, eri_so, nO_so, pairs) = both_layouts('water')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if driver == 'restricted':
            return restricted.ev_parquet(eps, eri, nO, 'flex', eta_2b=1.0, damping=0.0,
                                         qp_solver='linearized')
        return spin_orbital.ev_parquet_so(eps_so, eri_so, nO_so, 'flex', eta_2b=1.0,
                                          damping=0.0, spin_pairs=pairs,
                                          qp_solver='linearized')


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_the_recorded_changes_are_quacks(driver):
    """From vanishing vertices and old eigenvalues (Quack's old_*_Om = 0):
    'err' is max |Phi| of each vertex in the vertices' order (closed shell eh
    singlet, eh triplet, pp singlet, pp triplet, Quack's err_eh_sing, ...;
    spin orbitals eh, pp), 'err_eig' max |Omega| of each channel, the pp one
    over its ee and hh branches."""
    result = flex_from_zero(driver)
    record = result.history[0]['two_body'][0]
    want = [float(np.abs(v).max()) for v in result.vertices]
    assert len(set(want)) == len(want), want            # the order is visible
    assert list(record['err']) == want, (record['err'], want)

    def top(*channels):
        return max(float(np.abs(c.om).max()) for c in channels)
    sp = result.spectra
    if driver == 'restricted':
        want = (top(sp.eh_singlet), top(sp.eh_triplet), top(sp.ee_singlet, sp.hh_singlet),
                top(sp.ee_triplet, sp.hh_triplet))
        assert top(sp.hh_singlet) > top(sp.ee_singlet)      # the hh branch decides
    else:
        want = (top(sp.eh), top(sp.ee, sp.hh))
        assert top(sp.hh) > top(sp.ee)
    assert tuple(float(x) for x in record['err_eig']) == want, (record['err_eig'], want)


def test_the_result_holds_its_own_arrays():
    """A result shares no memory with what it was handed: eps_hf is a copy of
    the mean field's mo_energy (both front doors, every scheme) and of the
    eps_hf a lower level takes, so a later change to either leaves the
    result as it was."""
    mf = rhf('water')
    for scheme in sorted(SCHEMES):
        for result in runs(scheme):
            assert not np.shares_memory(result.eps_hf, mf.mo_energy), scheme
    (eps, eri, nO), (eps_so, eri_so, nO_so, pairs) = both_layouts('water')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        low = (restricted.ev_parquet(eps, eri, nO, 'flex', eta_2b=1.0, qp_solver='linearized'),
               spin_orbital.ev_parquet_so(eps_so, eri_so, nO_so, 'flex', eta_2b=1.0,
                                          qp_solver='linearized', spin_pairs=pairs))
    for result, given in zip(low, (eps, eps_so)):
        assert not np.shares_memory(result.eps_hf, given)
        assert np.array_equal(result.eps_hf, given)


def test_the_defaults_by_value():
    """Every entry point's defaults are the paper's Table III settings,
    s_1b = s_2b = 50 Ha^-2 and tau = 1e-4, and the schemes' caps Quack's runs
    take: FLEX 1 and 1, osPA 1 and 200, evPA and qsPA 50 and 200 (one-body,
    two-body); no DIIS. The quasiparticle equation is the house's root
    finder's by default (qp_solver='find_qp_root'; None, which the front
    doors take, is the same)."""
    for fn in (restricted.run_parquet, restricted.ev_parquet, restricted.qs_parquet,
               spin_orbital.run_parquet_so, spin_orbital.ev_parquet_so,
               spin_orbital.qs_parquet_so):
        p = inspect.signature(fn).parameters
        assert (p['eta_1b'].default, p['eta_2b'].default) == (50.0, 50.0), fn.__name__
        assert (p['conv_1b'].default, p['conv_2b'].default) == (1e-4, 1e-4), fn.__name__
        assert (p['max_diis_1b'].default, p['max_diis_2b'].default) == (1, 1), fn.__name__
    for scheme, caps in (('flex', (1, 1)), ('os', (1, 200)), ('ev', (50, 200)),
                         ('qs', (50, 200))):
        o = loop_options(scheme, True, True, 50.0, 50.0, None, 1e-4, None, 1e-4, 1, 1, None,
                         None, default_damping=0.75,
                         default_builder=restricted.static_reducible_vertex)
        assert (o['max_it_1b'], o['max_it_2b']) == caps, scheme
    for fn, default in ((restricted.ev_parquet, 'find_qp_root'),
                        (spin_orbital.ev_parquet_so, 'find_qp_root'),
                        (restricted.run_parquet, None), (spin_orbital.run_parquet_so, None)):
        assert inspect.signature(fn).parameters['qp_solver'].default == default, fn.__name__
    for result in runs('flex'):
        assert result.options['qp_solver'] == 'find_qp_root'
        assert result.history[0]['qp_solver'] == 'find_qp_root'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
