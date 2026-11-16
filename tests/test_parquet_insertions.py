"""Per-insertion reducible vertices: the vertex hook's per_insertion protocol
in both parquet drivers, and `averaged_vertices.insertion_vertex_builder`.

A builder whose attribute per_insertion is True is called once per vertex
and role at every two-body iteration, builder(name, role, spectra, eta_2b,
eps=eps, nO=nO), and the loop carries one tensor per role: an eh vertex
crossed into the eh kernel ('phbar_eh') and into the pp kernel ('eh_pp', in
its direct and its swapped terms alike, which exchange turns into each
other), a pp vertex into the eh kernel ('pp_eh'); six closed-shell tensors
(`spectra.RestrictedInsertions`), three over spin orbitals. Held on
water/6-31G (the hook's refusals on H2/6-31G, the closed-shell memory on
water/cc-pVDZ):

* a per-insertion builder that returns the static vertex in every role gives
  the default run: to the bit without two-body DIIS (osPA under the TDA and
  with full BSEs, evPA, qsPA, both drivers: eqp, Z, SigC, every tensor, the
  spectra, the history, and the flags of these runs, in which no vertex goes
  NaN), and with DIIS 5 to rounding, the DIIS weighing each of an eh
  vertex's two tensors a half: where each two-body loop starts its history
  afresh (osPA in both drivers, the closed-shell qsPA) the same counts and
  eqp within DIIS_FRESH, where it carries over between one-body iterations
  (the closed-shell evPA) within DIIS_CARRIED, as Quack's DIIS and the
  port's agree (the package docstring); QuackDIIS's block weights (two
  copies of a block at a half each give the unweighted history) and the
  loops' weights, wrong weights refused;
* each role reaches the kernels it is inserted into and no other: the
  static vertex in every role gives the default's spectra to the bit, and
  a role's tensors scaled change the channels of its own kernel alone (the
  eh ones for 'phbar_eh' and 'pp_eh', the ee and hh ones for 'eh_pp'), both
  drivers, TDA and full;
* each role is its window's vertex (`averaged_vertex_builder`'s, 'gap' at
  the iteration's gap) or the default's static one, to the bit, and all
  roles windowed alike give the run of `averaged_vertex_builder` with that
  window, to the bit without DIIS and to rounding with it;
* `transverse_vertex_builder` (Kauch's prescription: the transverse
  insertion into the eh kernel at a window, the other roles static) at the
  iteration's gap converges in the closed-shell osPA, evPA and qsPA,
  unflagged, away from the static run and from all roles windowed alike,
  and the spin-orbital qsPA converges, unflagged, near the closed-shell one;
* the two drivers agree on the closed shell as the static vertices do
  (`test_parquet_spin_adaptation`) with each role windowed alone: SigC at
  every level, eqp and Z where 0 < Z < 1;
* the builder gets the iteration's one-body energies (read-only: eps_hf,
  then eOld in evPA, the current eQP in qsPA) and nO, once per vertex and
  role in the order of the stored tensors; 'gap' resolves to each
  iteration's own gap;
* the hook contract per role: a wrong output refused with the vertex and the
  role named (a wrong shape at each role, every kind of wrong output at the
  last); a NaN at an element no kernel reads in a role's tensor kept and
  flagged ('parquet_vertex_not_finite', naming the role), the run otherwise
  the default one; a builder whose per_insertion is not True or False
  refused before any work, a functools.partial taking its function's; the
  spin-orbital two-body loop refusing a state of the other protocol before
  any work; the options record, the result's insertions and vertices, the
  history's changes per tensor; wrong arguments of insertion_vertex_builder
  refused, and a 'gap' that is not positive;
* the memory: the per-insertion loop holds its six tensors (three over spin
  orbitals) and two-body DIIS vectors half as long again, and nothing more
  (traced peaks of `test_parquet_memory`'s runs).

Run: python tests/test_parquet_insertions.py, or under pytest.
"""
import functools
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.numerics.constants import PARQUET_VERTEX_DAMPING
from src.SingleReference.Parquet import averaged_vertices, restricted, spin_orbital
from src.SingleReference.Parquet.averaged_vertices import (ROLES, averaged_vertex_builder,
                                                           insertion_vertex_builder,
                                                           iteration_gap,
                                                           transverse_vertex_builder)
from src.SingleReference.Parquet.diis import QuackDIIS
from src.SingleReference.Parquet.options import builder_protocol
from src.SingleReference.Parquet.spectra import (EH_ROLES, PP_ROLES, RESTRICTED_INSERTIONS,
                                                 SPIN_ORBITAL_INSERTIONS,
                                                 RestrictedInsertions, RestrictedVertices,
                                                 SpinOrbitalInsertions, SpinOrbitalVertices,
                                                 insertion_weights)
from tests.parquet_quack_newton import quack_newton
from tests.test_parquet_memory import (CLOSED_SHELL_DIIS_PAIR, DIIS, OSPA, RUN_PEAK,
                                       SO_DIIS_PAIR, SO_RUN_PEAK, traced_peak, water_dz)
from tests.test_parquet_refusals import WRONG_OUTPUTS, entry_points, h2, outcome
from tests.test_parquet_refusals import sentinels  # noqa: F401 -- the fixture
from tests.test_parquet_spin_adaptation import (CROSS_DRIVER, CROSS_DRIVER_Z,
                                                assert_same_two_body, both_layouts, rhf)
from tests.test_parquet_vertex_hook import default_builder, quiet, run

#: Ha. eqp of the all-static per-insertion run against the default with
#: two-body DIIS 5 where each two-body loop starts its history afresh (osPA,
#: qsPA; s_2b = 1, conv_2b = 1e-10): 8.9e-16 and 4.6e-14 closed shell, the
#: same bits over spin orbitals, the counts equal (measured).
DIIS_FRESH = 1e-12
#: Ha. The same in the closed-shell evPA, whose history carries over between
#: one-body iterations and whose later two-body counts then differ:
#: 1.4e-11 measured; the bar the package docstring gives Quack against the
#: port there.
DIIS_CARRIED = 1e-9
#: Ha. How far a transverse run's principal IP lies from the static run's
#: and from all roles windowed alike for the window to count as having acted
#: (the exponential at the gap: 0.3 to 0.7 eV measured).
ACTED = 1e-3
#: Ha. The spin-orbital qsPA's principal level against the closed-shell one
#: with the transverse window (4.8e-4 measured); their two self-energy forms
#: part them already with the static vertex (7.7e-4).
SO_QS = 1e-3

#: Short runs without two-body DIIS, (scheme, options): s_2b = 1, the
#: quasiparticle equation linearized, fixed counts, each driver's damping.
STATIC = {
    'os': dict(eta_2b=1.0, max_it_2b=8, conv_2b=1e-12, qp_solver='linearized'),
    'ev': dict(eta_2b=1.0, max_it_2b=5, conv_2b=1e-12, max_it_1b=3, conv_1b=1e-12,
               qp_solver='linearized'),
    'qs': dict(eta_2b=1.0, max_it_2b=5, conv_2b=1e-12, max_it_1b=2, conv_1b=1e-12),
}
#: osPA with full eh and pp BSEs, where no vertex goes NaN in either driver
#: (at s_2b = 50 they do, and a builder other than the default's then adds
#: its own 'parquet_vertex_not_finite' flags).
FULL_BSE = dict(STATIC['os'], tda_eh=False, tda_pp=False)
#: Converged runs with two-body DIIS 5 to tight thresholds (s_2b = 1, damping
#: 0.75 in both drivers).
WITH_DIIS = {
    'os': dict(eta_2b=1.0, max_diis_2b=5, conv_2b=1e-10, max_it_2b=400,
               damping=PARQUET_VERTEX_DAMPING, qp_solver='linearized'),
    'ev': dict(eta_2b=1.0, max_diis_2b=5, conv_2b=1e-10, max_it_2b=400, max_it_1b=2,
               conv_1b=1e-7, damping=PARQUET_VERTEX_DAMPING, qp_solver='linearized'),
    'qs': dict(eta_2b=1.0, max_diis_2b=5, conv_2b=1e-10, max_it_2b=400, max_it_1b=2,
               conv_1b=1e-7, damping=PARQUET_VERTEX_DAMPING),
}
#: The converging runs of `transverse_vertex_builder`: Table III's TDA,
#: s_1b = s_2b = 50 and two-body DIIS 5 (as test_parquet_averaged_vertex's).
CONVERGING = {
    'os': dict(eta_1b=50.0, eta_2b=50.0, max_diis_2b=5, qp_solver='linearized'),
    'ev': dict(eta_1b=50.0, eta_2b=50.0, max_diis_2b=5, qp_solver='linearized'),
    'qs': dict(eta_1b=50.0, eta_2b=50.0, max_diis_2b=5, max_diis_1b=3),
}
#: The window of the transverse runs: the imaginary exponential at the gap.
GAP_WINDOW = ('exponential', 'imaginary', 'gap')
#: The fixed-count comparison of the two drivers (test_parquet_spin_adaptation's
#: 'os_tda_cap').
TWO_DRIVERS = dict(tda_eh=True, tda_pp=True, eta_1b=1e8, eta_2b=1.0, qp_solver='linearized',
                   max_it_2b=5, conv_2b=1e-12, damping=PARQUET_VERTEX_DAMPING)
#: Per role, an element of its tensor that no kernel and no effective integral
#: reads (in either driver): an eh vertex crossed into the eh kernel reads
#: [p, x, y, q] with x, y one occupied and one virtual, a pp vertex into the
#: eh kernel [p, x, q, y] likewise, an eh vertex into the pp kernel
#: [p, q, r, s] and [p, q, s, r] with (r, s) both occupied or both virtual;
#: nO is put in for 'nO'.
UNREAD = {'phbar_eh': (0, 0, 0, 0), 'eh_pp': (0, 0, 0, 'nO'), 'pp_eh': (0, 0, 0, 0)}
#: The kernels each role is inserted into: the eh channels' or the pp ones'
#: (ee and hh).
INTO = {'phbar_eh': 'eh', 'eh_pp': 'pp', 'pp_eh': 'eh'}


def insertions_of(driver):
    return RESTRICTED_INSERTIONS if driver == 'restricted' else SPIN_ORBITAL_INSERTIONS


class Static:
    """A per-insertion builder returning the driver's static vertex in every
    role, recording each call: (name, role, a copy of eps, nO, whether eps
    could be written)."""
    per_insertion = True

    def __init__(self, driver):
        self.default = default_builder(driver)
        self.calls = []

    def __call__(self, name, role, spectra, eta_2b, *, eps, nO):
        self.calls.append((name, role, np.array(eps), nO, eps.flags.writeable))
        return self.default(name, spectra, eta_2b)


def plain(x, skip=('seconds',)):
    """Every number of a result field as bytes, for an exact comparison, the
    dict keys `skip` left out."""
    if isinstance(x, np.ndarray):
        return (x.dtype.str, x.shape, x.tobytes())
    if isinstance(x, dict):
        return {k: plain(v, skip) for k, v in x.items() if k not in skip}
    if isinstance(x, (list, tuple)):
        return [plain(v, skip) for v in x]
    if isinstance(x, (float, np.floating)):
        return np.float64(x).tobytes()
    return x


def assert_bitwise_run(got, want, label):
    """Two runs give the same bits: eqp, Z, SigC, the vertices, the spectra,
    the history (but its seconds; and its changes per tensor, 'err', where
    one run has more tensors than the other) and the flags."""
    def stored(result):
        return len(result.vertices if result.insertions is None else result.insertions)
    skip = ('seconds',) if stored(got) == stored(want) else ('seconds', 'err')
    for field in ('eqp', 'z', 'sigc', 'vertices', 'spectra', 'history', 'n_it_2b', 'converged',
                  'ec_eh', 'ec_pp'):
        mine, theirs = plain(getattr(got, field), skip), plain(getattr(want, field), skip)
        assert mine == theirs, (label, field)
    assert ([e['flag'] for e in got.numerical_flags]
            == [e['flag'] for e in want.numerical_flags]), label


# ---------------------------------------------------------------------------
# the all-static per-insertion builder is the default run


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
@pytest.mark.parametrize('scheme, tda', [('os', True), ('os', False), ('ev', True),
                                         ('qs', True)])
def test_static_in_every_role_is_the_default_run_to_the_bit(driver, scheme, tda):
    builder = Static(driver)
    options = STATIC[scheme] if tda else FULL_BSE
    got = run(driver, scheme, vertex_builder=builder, **options)
    want = run(driver, scheme, **options)
    assert_bitwise_run(got, want, (driver, scheme, tda))
    assert np.isfinite(want.eqp).all(), (driver, scheme, tda)
    # every stored tensor is the default's vertex of its name, bitwise
    stored = insertions_of(driver)
    assert type(got.insertions).__name__ == ('RestrictedInsertions' if driver == 'restricted'
                                             else 'SpinOrbitalInsertions')
    for (name, _), tensor in zip(stored, got.insertions):
        assert tensor.tobytes() == getattr(want.vertices, name).tobytes(), (driver, name)
    assert want.insertions is None
    assert len(builder.calls) == len(stored) * sum(got.n_it_2b)
    # the change of each tensor is its vertex's change in the default run
    names = list(want.vertices._fields)
    for mine, theirs in zip(got.history, want.history):
        for iteration, reference in zip(mine['two_body'], theirs['two_body']):
            assert iteration['err'] == tuple(reference['err'][names.index(name)]
                                             for name, _ in stored), (driver, scheme)
            assert iteration['err_2b'] == max(iteration['err'])
    assert got.options == {**want.options, 'vertex_builder': got.options['vertex_builder']}
    assert got.options['vertex_builder'] == f'{Static.__module__}.{Static.__qualname__}'


@pytest.mark.parametrize('driver, scheme', [('restricted', 'os'), ('restricted', 'ev'),
                                            ('restricted', 'qs'), ('spin_orbital', 'os')])
def test_static_in_every_role_with_diis_is_the_default_run_to_rounding(driver, scheme):
    got = run(driver, scheme, vertex_builder=Static(driver), **WITH_DIIS[scheme])
    want = run(driver, scheme, **WITH_DIIS[scheme])
    gap = np.abs(got.eqp - want.eqp).max()
    # every two-body loop converged (the one-body loops are cut at two iterations)
    assert all(got.converged_2b) and all(want.converged_2b), (driver, scheme)
    if scheme == 'ev':
        assert gap <= DIIS_CARRIED, (driver, scheme, gap)
    else:
        assert got.n_it_2b == want.n_it_2b, (driver, scheme, got.n_it_2b, want.n_it_2b)
        assert gap <= DIIS_FRESH, (driver, scheme, gap)


def test_the_diis_weights():
    """QuackDIIS with block weights: two copies of a block at a half each and
    one at 1 give the history of the two blocks unweighted (to rounding: the
    halves of a sum summed in another order), and weights 2 throughout the
    same extrapolation (DIIS is scale invariant); the loops' weights are a
    half for each eh tensor and 1 for each pp one; wrong weights are
    refused."""
    rng = np.random.default_rng(3)
    a, b = (4, 3), (5,)
    plain_diis = QuackDIIS(4, [a, b])
    weighted = QuackDIIS(4, [a, a, b], weights=(0.5, 0.5, 1.0))
    scaled = QuackDIIS(4, [a, b], weights=(2.0, 2.0))
    for _ in range(6):
        e = (rng.standard_normal(a), rng.standard_normal(b))
        x = (rng.standard_normal(a), rng.standard_normal(b))
        want = plain_diis.push(e, x)
        got = weighted.push((e[0], e[0], e[1]), (x[0], x[0], x[1]))
        twice = scaled.push(e, x)
        for k in range(2):
            assert np.allclose(got[k], want[0], rtol=1e-10, atol=1e-12)
        assert np.allclose(got[2], want[1], rtol=1e-10, atol=1e-12)
        assert np.allclose(twice[0], want[0], rtol=1e-10, atol=1e-12)
    assert insertion_weights(RESTRICTED_INSERTIONS) == (0.5, 0.5, 0.5, 0.5, 1.0, 1.0)
    assert insertion_weights(SPIN_ORBITAL_INSERTIONS) == (0.5, 0.5, 1.0)
    for weights in ((1.0,), (1.0, 0.0), (1.0, -1.0), (1.0, np.inf), (1.0, np.nan)):
        with pytest.raises(ValueError, match='DIIS block weights'):
            QuackDIIS(4, [a, b], weights=weights)


# ---------------------------------------------------------------------------
# where each role goes


def test_the_insertion_sets_split_into_the_kernels():
    """The fields of the insertion sets follow RESTRICTED_INSERTIONS and
    SPIN_ORBITAL_INSERTIONS, and each set hands each kernel its role's
    tensors (no copy): the eh kernels the eh vertices in 'phbar_eh' and the
    pp ones in 'pp_eh', the pp kernels the eh vertices in 'eh_pp'."""
    for kind, order in ((RestrictedInsertions, RESTRICTED_INSERTIONS),
                        (SpinOrbitalInsertions, SPIN_ORBITAL_INSERTIONS)):
        assert kind._fields == tuple(f'{name}_{role}' for name, role in order)
        assert all(role in (EH_ROLES if name.startswith('eh') else PP_ROLES)
                   for name, role in order)
        tensors = kind(*(np.full((2,) * 4, float(k)) for k in range(len(order))))
        for vertices, role in ((tensors.eh_kernel(), 'phbar_eh'), (tensors.pp_kernel(), 'eh_pp')):
            for name, array in zip(vertices._fields, vertices):
                own = role if name.startswith('eh') else 'pp_eh'
                assert array is getattr(tensors, f'{name}_{own}'), (kind.__name__, role, name)
    assert ROLES == EH_ROLES + PP_ROLES


def capture(default, into):
    """A builder that is `default` and keeps the spectra it was handed last."""
    def build(name, spectra, eta_2b):
        into['spectra'] = spectra
        return default(name, spectra, eta_2b)
    return build


@functools.lru_cache(maxsize=None)
def captured(driver):
    """(spectra, eps, nO, eri) of the third two-body iteration of an osPA on
    water/6-31G (s_2b = 50), as `driver`'s builder is handed them."""
    (eps, eri, nO), (eps_so, eri_so, nO_so, pairs) = both_layouts('water')
    into = {}
    options = dict(eta_2b=50.0, max_it_2b=3, conv_2b=1e-12, qp_solver='linearized')
    if driver == 'restricted':
        quiet(restricted.ev_parquet, eps, eri, nO, 'os',
              vertex_builder=capture(default_builder(driver), into), **options)
        return into['spectra'], eps, nO, eri
    quiet(spin_orbital.ev_parquet_so, eps_so, eri_so, nO_so, 'os', spin_pairs=pairs,
          vertex_builder=capture(default_builder(driver), into), **options)
    return into['spectra'], eps_so, nO_so, eri_so


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
@pytest.mark.parametrize('tda', [True, False])
def test_each_role_reaches_its_own_kernels_alone(driver, tda):
    """One two-body iteration (`two_body_spectra`) from the static vertices
    of a captured iteration: with them in every role it gives the default's
    channels to the bit, and with one role's tensors scaled by 1.01 the
    channels of the kernels that role enters change (the eh ones for
    'phbar_eh' and 'pp_eh', ee and hh for 'eh_pp') and every other channel is
    the default's to the bit."""
    spectra, eps, nO, eri = captured(driver)
    default = default_builder(driver)
    if driver == 'restricted':
        kind, insertions, iterate = (RestrictedVertices, RestrictedInsertions,
                                     restricted.two_body_spectra)
    else:
        kind, insertions, iterate = (SpinOrbitalVertices, SpinOrbitalInsertions,
                                     spin_orbital.two_body_spectra_so)
    vertices = kind(*(default(name, spectra, 50.0) for name in kind._fields))

    def channels(given):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return iterate(eps, eri, nO, given, first=False, tda_eh=tda, tda_pp=tda)[0]
    want = channels(vertices)
    every = [getattr(vertices, name) for name, _ in insertions_of(driver)]
    assert plain(channels(insertions(*every))) == plain(want), (driver, tda)
    for role in ROLES:
        scaled = insertions(*(1.01 * t if r == role else t
                              for (_, r), t in zip(insertions_of(driver), every)))
        got = channels(scaled)
        for channel in want._fields:
            reached = channel.startswith('eh') == (INTO[role] == 'eh')
            same = plain(getattr(got, channel)) == plain(getattr(want, channel))
            assert same != reached, (driver, tda, role, channel)


# ---------------------------------------------------------------------------
# windows per role, transverse only


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_each_role_is_its_windows_vertex_to_the_bit(driver):
    """Each role of an insertion builder gives the vertex of its own window
    (`averaged_vertex_builder`'s, 'gap' at the iteration's gap) or, without
    one, the default's static vertex, to the bit."""
    spectra, eps, nO, _ = captured(driver)
    eta_2b = 50.0
    gap = iteration_gap(eps, nO)
    default = default_builder(driver)
    for windows, center in ((dict(phbar_eh=('box', 'imaginary', None), eh_pp=None,
                                  pp_eh=('exponential', 'imaginary', 'gap')), -0.4),
                            (dict(phbar_eh=None, eh_pp=('gaussian', 'real', 0.2), pp_eh=None),
                             0.0)):
        builder = insertion_vertex_builder(**windows, pp_center=center, driver=driver)
        for name, role in insertions_of(driver):
            got = builder(name, role, spectra, eta_2b, eps=eps, nO=nO)
            if windows[role] is None:
                want = default(name, spectra, eta_2b)
            else:
                window, axis, x_half = windows[role]
                want = averaged_vertex_builder(window, axis,
                                               x_half=gap if x_half == 'gap' else x_half,
                                               pp_center=center)(name, spectra, eta_2b)
            assert got.tobytes() == want.tobytes(), (driver, name, role)
        widths = builder.widths(eta_2b, eps, nO)
        assert all((widths[role] is None) == (windows[role] is None) for role in ROLES)
        if windows['pp_eh'] is not None:
            assert widths['pp_eh'][0] == gap
    with pytest.raises(ValueError, match='role'):
        builder(insertions_of(driver)[0][0], 'pp_eh', spectra, eta_2b, eps=eps, nO=nO)
    with pytest.raises(ValueError, match='driver='):
        builder('eh' if driver == 'restricted' else 'eh_singlet', 'phbar_eh', spectra, eta_2b,
                eps=eps, nO=nO)


@functools.lru_cache(maxsize=None)
def water_run(scheme, kind, driver='restricted'):
    """run_parquet (or run_parquet_so) on water/6-31G with CONVERGING[scheme]:
    the static vertex, the transverse window GAP_WINDOW, or GAP_WINDOW in
    every role."""
    if kind == 'static':
        builder = None
    elif kind == 'transverse':
        builder = transverse_vertex_builder(*GAP_WINDOW, driver=driver)
    else:
        builder = insertion_vertex_builder(*(GAP_WINDOW,) * len(ROLES), driver=driver)
    front = restricted.run_parquet if driver == 'restricted' else spin_orbital.run_parquet_so
    return quiet(front, rhf('water'), scheme, vertex_builder=builder, **CONVERGING[scheme])


def principal_level(result):
    """The highest occupied quasiparticle energy, Ha."""
    return float(np.max(result.eqp[:result.nO]))


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_all_roles_alike_are_the_averaged_vertex(driver):
    """The box at the SRG's width in every role: osPA without DIIS the run of
    averaged_vertex_builder('box') to the bit, with DIIS 5 to rounding."""
    alike = insertion_vertex_builder(*('box',) * len(ROLES), driver=driver)
    whole = averaged_vertex_builder('box', driver=driver)
    for options, bitwise in ((STATIC['os'], True), (WITH_DIIS['os'], False)):
        got = run(driver, 'os', vertex_builder=alike, **options)
        want = run(driver, 'os', vertex_builder=whole, **options)
        if bitwise:
            assert_bitwise_run(got, want, driver)
        else:
            assert got.n_it_2b == want.n_it_2b, driver
            assert np.abs(got.eqp - want.eqp).max() <= DIIS_FRESH, driver


@pytest.mark.parametrize('scheme', sorted(CONVERGING))
def test_the_transverse_window_converges_away_from_static_and_all_roles(scheme):
    transverse, static, alike = (water_run(scheme, kind)
                                 for kind in ('transverse', 'static', 'all'))
    for result in (transverse, alike):
        assert result.converged and np.isfinite(result.eqp).all(), scheme
        assert not result.numerical_flags, (scheme, result.numerical_flags)
    level = principal_level(transverse)
    assert abs(level - principal_level(static)) > ACTED, scheme
    assert abs(level - principal_level(alike)) > ACTED, scheme
    assert transverse.options['vertex_builder'] == (
        "insertion_vertex_builder(phbar_eh=('exponential', 'imaginary', 'gap'), eh_pp=None, "
        "pp_eh=None, pp_center=0.0, driver='restricted')")
    # the stored tensors: the transverse role windowed, the eh vertex in the pp kernel static
    tensors = transverse.insertions
    assert np.abs(tensors.eh_singlet_phbar_eh - tensors.eh_singlet_eh_pp).max() > 1e-4
    assert transverse.vertices.eh_singlet is tensors.eh_singlet_phbar_eh
    assert transverse.vertices.pp_triplet is tensors.pp_triplet_pp_eh


def test_the_spin_orbital_transverse_qspa_converges_beside_the_closed_shell_one():
    """The spin-orbital qsPA with the transverse window at the gap converges,
    unflagged, and its principal level lies within SO_QS of the closed-shell
    run's."""
    so, cs = (water_run('qs', 'transverse', driver)
              for driver in ('spin_orbital', 'restricted'))
    assert so.converged and np.isfinite(so.eqp).all() and not so.numerical_flags
    assert abs(principal_level(so) - principal_level(cs)) <= SO_QS


#: One role windowed at a time.
EACH_ROLE = {role: {role: GAP_WINDOW} for role in ROLES}


@pytest.mark.parametrize('case', sorted(EACH_ROLE))
def test_the_two_drivers_agree_with_each_role_windowed(case):
    """The spin-orbital driver is the closed-shell one spin-adapted with a
    per-insertion builder too: spectra, EcRPA and SigC within the bars of the
    static comparison, and eqp and Z wherever the quasiparticle equation is
    well conditioned (0 < Z < 1: with 'eh_pp' windowed the highest virtual
    level lies beside a pole of the self-energy, Z = 5.5, which magnifies the
    rounding of dSigma/dw into Z and eqp, 5.5e-10 and 5.7e-11 measured, SigC
    agreeing to 4e-14), the occupied levels among them; the alpha/beta
    copies equal to the bit, the window acting."""
    (eps, eri, nO), (eps_so, eri_so, nO_so, pairs) = both_layouts('water')
    r = quiet(restricted.ev_parquet, eps, eri, nO, 'os', **TWO_DRIVERS,
              vertex_builder=insertion_vertex_builder(**EACH_ROLE[case]))
    g = quiet(spin_orbital.ev_parquet_so, eps_so, eri_so, nO_so, 'os', spin_pairs=pairs,
              **TWO_DRIVERS, vertex_builder=insertion_vertex_builder(**EACH_ROLE[case],
                                                                     driver='spin_orbital'))
    assert r.n_it_2b == g.n_it_2b == [TWO_DRIVERS['max_it_2b']]
    assert_same_two_body(r, g, case)
    assert np.array_equal(g.eqp[0::2], g.eqp[1::2])
    assert np.abs(r.sigc - g.sigc[0::2]).max() <= CROSS_DRIVER, case
    conditioned = (r.z > 0.0) & (r.z < 1.0)
    assert conditioned[:nO].all(), (case, r.z)
    assert np.abs(r.eqp - g.eqp[0::2])[conditioned].max() <= CROSS_DRIVER, case
    assert np.abs(r.z - g.z[0::2])[conditioned].max() <= CROSS_DRIVER_Z, case
    static = quiet(restricted.ev_parquet, eps, eri, nO, 'os', **TWO_DRIVERS)
    assert np.abs(r.eqp - static.eqp).max() > ACTED, case


# ---------------------------------------------------------------------------
# what the builder is handed


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
@pytest.mark.parametrize('scheme', ['ev', 'qs'])
def test_the_builder_gets_each_iterations_energies_and_n_occupied(driver, scheme):
    """eps is the one-body energies of the iteration's BSE diagonals (eps_hf
    at the first one-body iteration, then eOld in evPA and the current eQP in
    qsPA: the previous record's eqp), read-only; nO the occupied count; the
    calls run over the stored tensors in their order, every iteration."""
    builder = Static(driver)
    result = run(driver, scheme, vertex_builder=builder, **STATIC[scheme])
    stored = insertions_of(driver)
    calls = iter(builder.calls)
    for k, record in enumerate(result.history):
        want = result.eps_hf if k == 0 else result.history[k - 1]['eqp']
        for _ in range(record['n_it_2b']):
            for name, role in stored:
                got = next(calls)
                assert got[:2] == (name, role)
                assert got[2].tobytes() == np.asarray(want).tobytes(), (driver, scheme, k)
                assert got[3] == result.nO and got[4] is False
    assert next(calls, None) is None


def test_gap_follows_the_iteration():
    """'gap' is each two-body iteration's own HOMO-LUMO gap: the windowed
    role is that of a window at that gap as a number, to the bit, and the
    gap moves from one evPA one-body iteration to the next."""
    transverse = transverse_vertex_builder('exponential', x_half='gap')
    seen = []

    class Checked:
        per_insertion = True

        def __call__(self, name, role, spectra, eta_2b, *, eps, nO):
            gap = iteration_gap(eps, nO)
            seen.append(gap)
            fixed = transverse_vertex_builder('exponential', x_half=gap)
            got = transverse(name, role, spectra, eta_2b, eps=eps, nO=nO)
            assert got.tobytes() == fixed(name, role, spectra, eta_2b, eps=eps,
                                          nO=nO).tobytes(), (name, role)
            return got
    result = run('restricted', 'ev', vertex_builder=Checked(), **STATIC['ev'])
    eps = result.eps_hf
    assert seen[0] == eps[result.nO] - eps[result.nO - 1]
    per_iteration = len(RESTRICTED_INSERTIONS) * result.n_it_2b[0]
    assert len(set(seen[:per_iteration])) == 1 and seen[per_iteration] != seen[0]
    assert seen[per_iteration] == iteration_gap(result.history[0]['eqp'], result.nO)


# ---------------------------------------------------------------------------
# the hook contract per role


def returning(driver, wrong, at):
    """A per-insertion builder returning the static vertex, but
    `wrong(vertex)` for the (name, role) `at`."""
    default = default_builder(driver)

    class Build:
        per_insertion = True

        def __call__(self, name, role, spectra, eta_2b, *, eps, nO):
            phi = default(name, spectra, eta_2b)
            return wrong(phi) if (name, role) == at else phi
    return Build()


def h2_run(driver, builder, **kwargs):
    """H2/6-31G's osPA through `builder`: three two-body iterations,
    linearized."""
    eps, eri, nO, so = h2()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if driver == 'restricted':
            return restricted.ev_parquet(eps, eri, nO, 'os', eta_2b=1.0, max_it_2b=3,
                                         qp_solver='linearized', vertex_builder=builder,
                                         **kwargs)
        return spin_orbital.ev_parquet_so(*so, 'os', eta_2b=1.0, max_it_2b=3,
                                          qp_solver='linearized', vertex_builder=builder,
                                          **kwargs)


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_a_wrong_tensor_is_refused_naming_its_role(driver):
    """'shape' at every (vertex, role), and every kind of wrong output at the
    last one, with and without two-body DIIS: a ValueError naming the hook,
    the vertex and the role."""
    stored = insertions_of(driver)
    cases = ([(at, 'shape') for at in stored]
             + [(stored[-1], kind) for kind in sorted(WRONG_OUTPUTS)])
    for (name, role), kind in cases:
        for max_diis_2b in (1, 3):
            with pytest.raises(ValueError) as err:
                h2_run(driver, returning(driver, WRONG_OUTPUTS[kind], (name, role)),
                       max_diis_2b=max_diis_2b)
            message = str(err.value)
            assert 'vertex_builder returned' in message, message
            assert repr(name) in message and repr(role) in message, message


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_a_nan_no_kernel_reads_is_kept_and_flagged_per_role(driver):
    """A NaN at UNREAD[role] of one tensor per role: kept, flagged naming the
    role, the run without DIIS the default one to the bit (the change passes
    over NaN, as gfortran's maxval)."""
    options = dict(eta_2b=1.0, max_it_2b=6, conv_2b=1e-12, qp_solver='linearized')
    want = run(driver, 'os', **options)
    nO = want.nO
    for at in insertions_of(driver)[::-1]:
        index = tuple(nO if i == 'nO' else i for i in UNREAD[at[1]])

        def nan_at(phi):
            phi = phi.copy()
            phi[index] = np.nan
            return phi
        got = run(driver, 'os', vertex_builder=returning(driver, nan_at, at), **options)
        assert got.eqp.tobytes() == want.eqp.tobytes() and got.z.tobytes() == want.z.tobytes()
        assert got.n_it_2b == want.n_it_2b, at
        flags = [e for e in got.numerical_flags if e['flag'] == 'parquet_vertex_not_finite']
        assert flags and all(f'(role {at[1]!r})' in e['message'] for e in flags), at
        tensor = getattr(got.insertions, f'{at[0]}_{at[1]}')
        others = np.delete(tensor.ravel(), np.ravel_multi_index(index, tensor.shape))
        assert np.isnan(tensor[index]) and np.isfinite(others).all(), at
        for record in got.history[0]['two_body']:
            assert np.isfinite(record['err']).all(), at


class Declares:
    """A builder declaring per_insertion = `value`."""

    def __init__(self, value):
        self.per_insertion = value

    def __call__(self, *args, **kwargs):
        raise AssertionError('not to be called')


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
@pytest.mark.parametrize('scheme', ['os', 'qs'])
def test_per_insertion_is_a_switch_refused_before_any_work(sentinels, driver,  # noqa: F811
                                                           scheme):
    """per_insertion True or False (numpy's bool too) is taken, from a
    functools.partial's function too; anything else is refused before any
    work, at every entry point."""
    wrong = []
    for level, call in entry_points(sentinels, driver, scheme).items():
        for value in ('yes', 1, 0, None, 1.0, np.array(True)):
            for builder in (Declares(value), functools.partial(Declares(value))):
                if outcome(call, vertex_builder=builder) != 'refused':
                    wrong.append((level, repr(value), type(builder).__name__))
        for value in (True, False, np.True_):
            for builder in (Declares(value), functools.partial(Declares(value))):
                if outcome(call, vertex_builder=builder) != 'reached':
                    wrong.append((level, repr(value), type(builder).__name__))
    assert wrong == [], wrong


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_a_partial_of_a_per_insertion_builder_is_one(driver):
    """A functools.partial of a per-insertion builder takes its function's
    protocol: the run of the builder itself, to the bit, the options naming
    it 'partial(...)'. `options.builder_protocol` reads a partial's own
    attribute first, else its function's, and a partial of such a partial
    is one partial (functools.partial flattens it)."""
    got = run(driver, 'os', vertex_builder=functools.partial(Static(driver)), **STATIC['os'])
    want = run(driver, 'os', vertex_builder=Static(driver), **STATIC['os'])
    assert_bitwise_run(got, want, driver)
    assert got.options['vertex_builder'] == f'partial({Static.__module__}.{Static.__qualname__})'
    assert builder_protocol(functools.partial(functools.partial(Static(driver)))) is True
    assert builder_protocol(functools.partial(default_builder(driver))) is False
    own = functools.partial(Static(driver))
    own.per_insertion = False
    assert builder_protocol(own) is False
    assert builder_protocol(functools.partial(own)) is False


def test_the_spin_orbital_two_body_loop_refuses_a_state_of_the_other_protocol():
    """two_body_loop_so with a per-insertion builder and a state made for the
    default one, or the other way round: a ValueError before any work (no
    builder call, the state not started)."""
    eps, eri, nO, _ = both_layouts('water')[1]
    nso = eps.shape[0]
    for per_insertion, builder in ((False, Static('spin_orbital')),
                                   (True, default_builder('spin_orbital'))):
        state = spin_orbital.TwoBodyState(nso, nO, 1, per_insertion)
        opts = dict(tda_eh=True, tda_pp=True, eta_2b=1.0, max_it_2b=2, conv_2b=1e-12,
                    alpha=1.0 - PARQUET_VERTEX_DAMPING, vertex_builder=builder)
        with pytest.raises(ValueError, match='per_insertion'):
            spin_orbital.two_body_loop_so(state, eps, eri, nO, opts)
        assert not state.started and getattr(builder, 'calls', []) == []


@pytest.mark.parametrize('max_diis_2b', [1, DIIS])
def test_the_per_insertion_state_holds_its_tensors_and_nothing_more(max_diis_2b):
    """Traced peaks of the memory test's osPA runs with the all-static
    per-insertion builder: the closed-shell loop carries six tensors, 2 N4
    above the default's bar (9.71 N4 measured, the default 7.71), and its
    two-body DIIS 12 N4 per history pair (69.71 N4 with five, the default
    47.71); the spin-orbital loop three, 1 nso^4 above (8.09 against 7.09),
    and 6 nso^4 per pair (38.09 against 27.09)."""
    eps, eri, nO = water_dz()
    peak = traced_peak(lambda: restricted.ev_parquet(
        eps, eri, nO, 'os', max_diis_2b=max_diis_2b, qp_solver=quack_newton,
        vertex_builder=Static('restricted'), **OSPA), eri.nbytes)
    pair = CLOSED_SHELL_DIIS_PAIR * len(RESTRICTED_INSERTIONS) / len(RestrictedVertices._fields)
    history = 0 if max_diis_2b == 1 else pair * max_diis_2b
    assert peak <= RUN_PEAK + 2 + history, peak
    eps, eri, nO, _ = both_layouts('water')[1]
    peak = traced_peak(lambda: spin_orbital.ev_parquet_so(
        eps, eri, nO, 'os', max_diis_2b=max_diis_2b, qp_solver='linearized',
        vertex_builder=Static('spin_orbital'), **OSPA), eri.nbytes)
    pair = SO_DIIS_PAIR * len(SPIN_ORBITAL_INSERTIONS) / len(SpinOrbitalVertices._fields)
    history = 0 if max_diis_2b == 1 else pair * max_diis_2b
    assert peak <= SO_RUN_PEAK + 1 + history, peak


#: Wrong arguments of the factories: (keywords, the text the error starts with).
REFUSED = [
    (dict(phbar_eh='sinc'), 'phbar_eh = '),
    (dict(eh_pp=('box', 'real')), 'eh_pp = '),
    (dict(eh_pp=('box', 'matsubara')), 'eh_pp = '),
    (dict(pp_eh=('box', 'imaginary', 0.0)), 'pp_eh = '),
    (dict(pp_eh=('box', 'imaginary', 'gapp')), 'pp_eh = '),
    (dict(phbar_eh=('box', 'imaginary', None, 'x')), 'phbar_eh = '),
    (dict(phbar_eh=3), 'phbar_eh = '),
    (dict(phbar_eh=()), 'phbar_eh = '),
    (dict(pp_center=0.3), 'pp_center = '),
    (dict(pp_eh='box', pp_center=float('nan')), 'pp_center = '),
    (dict(driver='so'), 'driver = '),
]


@pytest.mark.parametrize('keywords, start', REFUSED)
def test_wrong_arguments_are_refused(keywords, start):
    with pytest.raises(ValueError, match=f'^{start}'):
        insertion_vertex_builder(**keywords)


def test_a_gap_that_is_not_positive_is_refused():
    spectra, eps, nO, _ = captured('restricted')
    builder = transverse_vertex_builder('box', x_half='gap')
    crossed = np.array(eps)
    crossed[nO] = crossed[nO - 1] - 0.1
    for energies in (crossed, np.full_like(crossed, np.nan)):
        with pytest.raises(ValueError, match="'gap' takes a positive one"):
            builder('eh_singlet', 'phbar_eh', spectra, 50.0, eps=energies, nO=nO)
    # the static roles do not read it
    assert builder('eh_singlet', 'eh_pp', spectra, 50.0, eps=crossed, nO=nO).shape == (13,) * 4
    assert averaged_vertices.GAP == 'gap'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
