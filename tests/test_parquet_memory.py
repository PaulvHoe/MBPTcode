"""The parquet drivers hold the memory their module docstrings say.

Traced peaks (tracemalloc) of an osPA run, TDA, s_2b = 1, eight two-body
iterations, in units of one vertex (N4: norb^4 float64 numbers closed-shell,
nso^4 over spin orbitals), the integrals handed in (alive throughout, built
before tracing) counted as one more:

* the closed-shell driver on water/cc-pVDZ (norb 24): the two-body floor is
  7 N4 (the integrals, four vertices, the vertex being built, the six
  channels' effective integrals) plus blocks of at most 1/8 vertex; the BSEs
  and their integrals (`two_body_spectra`) peak at 6.26 N4 and the run at
  7.68 N4 (the self-energy built beside the vertices and spectra); two-body
  DIIS 5 adds its 8 x 5 N4 history and nothing more;
* the spin-orbital driver on water/6-31G (nso 26): 7.09 nso^4, and two-body
  DIIS 5 adds its 4 x 5 nso^4; its in-place damping (no DIIS) peaks at
  4.60 nso^4, one scratch block beside what is alive then, where a
  whole-vertex temporary would make it 5.49 (the run's peak, set by
  another stage, would not show it).

The bars: the BSE stage's sits 0.24 N4 above its measured peak and a full
N4 below an earlier version of that stage (7.46 N4), the runs' 0.2 N4 above
theirs, where one more vertex-sized temporary (+1 N4) fails them; measured with numpy 2.1 on
the workstation (tracemalloc counts numpy's temporaries, which another numpy
may arrange otherwise by a block or so).

Run: python tests/test_parquet_memory.py, or under pytest.
"""
import functools
import os
import sys
import tracemalloc
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from pyscf import ao2mo, gto, scf

from src.SingleReference.Parquet import restricted, spin_orbital
from tests.parquet_quack_newton import quack_newton
from tests.test_parquet_spin_adaptation import both_layouts

WATER = 'O 0 0 0; H 0.7571 0 0.5861; H -0.7571 0 0.5861'
#: The osPA traced: TDA, s_2b = 1, eight two-body iterations.
OSPA = dict(eta_2b=1.0, max_it_2b=8, conv_2b=1e-12)
#: Two-body DIIS length of the second run.
DIIS = 5
#: N4. The closed-shell bars without DIIS (measured 6.264 and 7.677).
TWO_BODY_SPECTRA = 6.5
RUN_PEAK = 7.9
#: N4 per DIIS pair: an error and an iterate of four vertices.
CLOSED_SHELL_DIIS_PAIR = 8
#: nso^4. The spin-orbital bar without DIIS (measured 7.093).
SO_RUN_PEAK = 7.3
#: nso^4 per DIIS pair: an error and an iterate of two vertices.
SO_DIIS_PAIR = 4
#: nso^4. The spin-orbital damping step's own bar (measured 4.601; 5.485
#: with a whole-vertex temporary).
SO_DAMPING_STEP = 4.8


@functools.lru_cache(maxsize=None)
def water_dz():
    mol = gto.M(atom=WATER, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    n = mf.mo_coeff.shape[1]
    eri = ao2mo.full(mol, mf.mo_coeff, compact=False).reshape((n,) * 4).transpose(0, 2, 1, 3)
    return np.asarray(mf.mo_energy), np.ascontiguousarray(eri), mol.nelectron // 2


def traced_peak(fn, units, watch=()):
    """The peak of fn() in `units` bytes, plus one for the integrals handed
    in; with `watch` ((module, attribute) calls), instead {attribute: the
    peak of its calls, each from its entry}, whose resets hide the run's
    own peak (so a run is traced once for each)."""
    stages = {}
    patches = []
    for module, name in watch:
        real = getattr(module, name)

        def wrapped(*args, _real=real, _name=name, **kwargs):
            tracemalloc.reset_peak()
            out = _real(*args, **kwargs)
            stages[_name] = max(stages.get(_name, 0), tracemalloc.get_traced_memory()[1])
            return out
        patches.append((module, name, real))
        setattr(module, name, wrapped)
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fn()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        for module, name, real in patches:
            setattr(module, name, real)
    if watch:
        return {k: (v - base) / units + 1.0 for k, v in stages.items()}
    return (peak - base) / units + 1.0


@pytest.mark.parametrize('max_diis_2b', [1, DIIS])
def test_the_closed_shell_two_body_loop(max_diis_2b):
    eps, eri, nO = water_dz()
    units = eri.nbytes

    def run():
        restricted.ev_parquet(eps, eri, nO, 'os', max_diis_2b=max_diis_2b,
                              qp_solver=quack_newton, **OSPA)
    peak = traced_peak(run, units)
    stages = traced_peak(run, units, watch=((restricted, 'two_body_spectra'),))
    history = 0 if max_diis_2b == 1 else CLOSED_SHELL_DIIS_PAIR * max_diis_2b
    assert stages['two_body_spectra'] <= TWO_BODY_SPECTRA + history, stages
    assert peak <= RUN_PEAK + history, peak


@pytest.mark.parametrize('max_diis_2b', [1, DIIS])
def test_the_spin_orbital_two_body_loop(max_diis_2b):
    eps, eri, nO, pairs = both_layouts('water')[1]

    def run():
        spin_orbital.ev_parquet_so(eps, eri, nO, 'os', max_diis_2b=max_diis_2b,
                                   qp_solver='linearized', **OSPA)
    peak = traced_peak(run, eri.nbytes)
    history = 0 if max_diis_2b == 1 else SO_DIIS_PAIR * max_diis_2b
    assert peak <= SO_RUN_PEAK + history, peak
    if max_diis_2b == 1:
        # the damping in place, blocked (with DIIS the damped vertex goes to its slot)
        stages = traced_peak(run, eri.nbytes, watch=((spin_orbital, '_damp_in_place'),))
        assert stages['_damp_in_place'] <= SO_DAMPING_STEP, stages


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
