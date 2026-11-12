"""`diis.QuackDIIS`, the DIIS of Quack's parquet loops, against Quack's own.

Quack's utils/DIIS_extrapolation (git e3528a40: prepend, the Gram matrix by
one dgemm, linear_solve's dsysvx, the weighted sum by dgemm) was run on a
deterministic history of seven steps of size-8 vectors through a size-3
history (`QUACK_*`, recorded from the Fortran; the Fortran does not run
here), once with one-hot iterates, whose extrapolation is the weight of each
step, and once with smooth ones. QuackDIIS on the same errors and iterates:

* returns Quack's weights and iterates to cond x eps (the systems' rcond is
  1.7e-3 to 0.14: WEIGHTS_REL) and its rcond, and holds the three newest
  pairs once more steps have come: a step that left the history weighs
  exactly zero, the weights sum to one;
* sums the weighted iterates newest first, each product rounded before it is
  added: the smooth extrapolation is that sum of its own one-hot weights, to
  the bit (the weights depend on the errors alone);
* a history of one returns the iterate itself, to the bit (Quack's
  max_diis 1, no DIIS);
* a NaN in an error makes the extrapolation NaN in every element and rcond
  NaN, flagged 'parquet_diis_not_finite' (Quack's weights are undefined
  there: dsysvx stops at the NaN pivot and linear_solve ignores its info); a
  NaN in an iterate beside finite errors enters the sum as it is;
* an exactly singular system (two equal errors) takes its least-squares
  weights, flagged 'parquet_diis_singular';
* the slots and their refusals: a wrong block, an output inside the
  history, an extrapolation without a slot taken.

Run: python tests/test_parquet_diis.py, or under pytest.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.numerical_flags import collecting
from src.SingleReference.Parquet.diis import QuackDIIS
from src.SingleReference.Parquet.events import (FLAG_DIIS_NOT_FINITE, FLAG_DIIS_SINGULAR,
                                                ParquetNotFinite)

#: The history Quack's numbers were recorded with: SIZE pairs of N numbers.
N, SIZE = 8, 3
#: Relative. QuackDIIS's weights and iterates against Quack's: 1.3e-14
#: measured (the step of rcond 1.7e-3); the Gram matrices are summed in
#: other orders (Quack's dgemm, the port's block dot products), so they
#: agree to cond x eps.
WEIGHTS_REL = 1e-12
#: Relative. dsysvx's condition estimate of the two Gram matrices: 1e-15
#: measured.
RCOND_REL = 1e-8
#: The least-squares (minimum-norm) weights of the singular system of two equal
#: errors.
SINGULAR_WEIGHTS = (0.5, 0.5)

#: Quack's DIIS_extrapolation was fed these errors (cos(0.9 (i + 1) (k + 1))
#: scaled by 0.6^k, step k) and SMOOTH_ITERATES (1 + sin(0.37 (i + 1) (k + 2)))
#: or the one-hot iterates of each step; its extrapolations QUACK_ONE_HOT,
#: QUACK_SMOOTH and condition estimates QUACK_RCOND (the same for both runs)
#: were recorded from Quack's routine itself (the paper-era tree, git
#: e3528a40, built as a shared library and called on these arrays), one
#: thread.
ERRORS = (
    (0.6216099682706644, -0.2272020946930871, -0.9040721420170612, -0.896758416334147,
     -0.2107957994307797, 0.6346928759426347, 0.9998586363834151, 0.6083513145322546),
    (-0.13632125681585225, -0.5380550498004881, 0.3808157255655808, 0.3650107887193527,
     -0.5466781571308061, -0.11659794387320087, 0.5996607513006028, -0.1558904137282535),
    (-0.325465971126142, 0.22848943533934848, -0.08767589534488543, -0.06995876632392052,
     0.21417143879156111, -0.31729409653040386, 0.3595420682076606, -0.3328098389690838),
    (-0.19369981792817573, 0.13140388393896696, -0.04197525979435231, -0.05612054894217126,
     0.1426284089807213, -0.19968590338145026, 0.21551161998048413, -0.18683781468916072),
    (-0.027319135606229047, -0.11808248194025413, 0.077101717964962, 0.08557704538843279,
     -0.11318028135611961, -0.03786118961190517, 0.12914224081940337, -0.016584094175711644),
    (0.049353718033299264, -0.01511109352596683, -0.06853552485056721, -0.07188692521732208,
     -0.022716713767143095, 0.0430506524316543, 0.07736459857325044, 0.05515486669755297),
    (0.04664940453910461, 0.04662962002113487, 0.04659665203971279, 0.04655050991578457,
     0.04649120669498521, 0.04641875914395024, 0.046333187745575356, 0.04623451669322524),
)
SMOOTH_ITERATES = (
    (1.674287911628145, 1.99588084453764, 1.7965654722360869, 1.180596267894233,
     0.4701638590915066, 0.036869069426683554, 0.10735232057176536, 0.6447464400119574),
    (1.8956986856800477, 1.7965654722360869, 0.8127053364570973, 0.036869069426683554,
     0.33076014272373866, 1.3679605105723838, 1.9964756147406004, 1.5182282059209768),
    (1.99588084453764, 1.180596267894233, 0.036869069426683554, 0.6447464400119574,
     1.898708095811627, 1.5182282059209768, 0.19526909073653287, 0.33583927647990486),
    (1.9612752029752998, 0.4701638590915066, 0.3307601427237373, 1.898708095811627,
     1.1738894853804336, 0.005447411796010604, 1.3742885011457568, 1.7882520673753164),
    (1.7965654722360869, 0.036869069426683554, 1.3679605105723838, 1.5182282059209768,
     0.005447411796010937, 1.6842898458495499, 1.1671746274635368, 0.11357855275775997),
    (1.5240443416872762, 0.10735232057176536, 1.9964756147406004, 0.19526909073653287,
     1.3742885011457584, 1.1671746274635368, 0.34094925376610175, 1.9554405263552677),
    (1.180596267894233, 0.6447464400119574, 1.5182282059209768, 0.33583927647990486,
     1.7882520673753164, 0.11357855275775997, 1.9554405263552677, 0.006960413510672225),
)
QUACK_ONE_HOT = (
    (0.9999999999999999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.2569205316468541, 0.7430794683531459, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.12521383000532882, 0.24689185792044163, 0.6278943120742295, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.016672701507367056, -1.4567509813605684, 2.4400782798532012, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, -1.3143420645664066, 2.200481533093554, 0.11386053147285238, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.09927668090743144, 0.24724659038270977, 0.6534767287098587, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.16712490671791083, 0.3415236835835763, 0.4913514096985129, 0.0),
)
QUACK_SMOOTH = (
    (1.6742879116281448, 1.9958808445376397, 1.7965654722360866, 1.1805962678942328,
     0.47016385909150654, 0.03686906942668355, 0.10735232057176534, 0.6447464400119572),
    (1.8388137118982912, 1.8477736836531924, 1.0654792056075812, 0.33071606931593334,
     0.36657581964649283, 1.0259757898426531, 1.511121053656287, 1.2938128062397987),
    (1.9308788023745633, 1.4347589535617635, 0.44875505307824753, 0.5617622758572023,
     1.3327210173394395, 1.2956416843371028, 0.6289639204322889, 0.6664409251161039),
    (1.9097700629699919, -0.5426445514473678, 0.7669215805203576, 3.6943760821446356,
     0.10395201946417157, -2.175580720433178, 3.1021997243292496, 3.8995537986109543),
    (1.8970376151178914, -0.5129225148648088, 0.8351297278857089, 3.5035210049047927,
     0.08819046101658089, -1.7917101285900925, 2.9003412114540246, 3.5065400771455835),
    (1.6348310907216834, 0.1259443023862143, 1.6757106948329854, 0.6914779917337258,
     1.0159522598859816, 1.179697179953185, 0.6478171510317081, 1.4834485181341575),
    (1.400835714992455, 0.35962217194959784, 1.6564475480775063, 0.4854378683799306,
     1.3489226436072554, 0.7359115435392082, 1.2723146550058895, 0.6902310656148077),
)
QUACK_RCOND = (0.04204962268104954, 0.10433587898446278, 0.14477025742279792, 0.010462749947392347,
               0.001731757304516054, 0.03155771089257297, 0.013723676259334105)


def history(iterates):
    """A size-SIZE QuackDIIS fed ERRORS with `iterates`: its extrapolation
    and rcond after each step."""
    diis = QuackDIIS(SIZE, [(N,)], label='test')
    out, rcond = [], []
    for error, iterate in zip(ERRORS, iterates):
        out.append(diis.push(np.array(error), np.array(iterate)))
        rcond.append(diis.rcond)
    return diis, out, rcond


def relative_gap(got, want):
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    return float(np.abs(got - want).max() / np.abs(want).max())


def test_weights_and_iterates_are_quacks():
    """One-hot iterates give each step's weight; both runs against Quack's,
    with Quack's rcond."""
    _, weights, rcond = history(np.eye(N)[:len(ERRORS)])
    for k, (got, want) in enumerate(zip(weights, QUACK_ONE_HOT)):
        assert relative_gap(got, want) <= WEIGHTS_REL, (k, got, want)
    for k, (got, want) in enumerate(zip(rcond, QUACK_RCOND)):
        assert abs(got - want) <= RCOND_REL * want, (k, got, want)
    _, out, _ = history(SMOOTH_ITERATES)
    for k, (got, want) in enumerate(zip(out, QUACK_SMOOTH)):
        assert relative_gap(got, want) <= WEIGHTS_REL, (k, got, want)


def test_the_history_keeps_the_newest_pairs():
    """Past SIZE steps the oldest pair leaves: its weight is exactly zero, the
    SIZE newest weigh, and the weights sum to one."""
    diis, weights, _ = history(np.eye(N)[:len(ERRORS)])
    assert len(diis) == SIZE and len(diis.rcond_history) == len(ERRORS)
    for k, w in enumerate(weights):
        kept = range(max(0, k - SIZE + 1), k + 1)
        assert all(w[j] == 0.0 for j in range(N) if j not in kept), (k, w)
        assert all(w[j] != 0.0 for j in kept), (k, w)
        assert abs(w.sum() - 1.0) <= WEIGHTS_REL, (k, w.sum())


def test_the_sum_runs_newest_first():
    """The smooth extrapolation is ((0 + w_1 x_1) + w_2 x_2) + w_3 x_3 over
    its pairs newest first (w the one-hot run's weights, the same errors), to
    the bit."""
    _, weights, _ = history(np.eye(N)[:len(ERRORS)])
    _, out, _ = history(SMOOTH_ITERATES)
    for k, (w, got) in enumerate(zip(weights, out)):
        want = np.zeros(N)
        for j in range(k, max(-1, k - SIZE), -1):
            want += w[j] * np.array(SMOOTH_ITERATES[j])
        assert np.array_equal(got, want), (k, got - want)


def test_a_history_of_one_is_no_diis():
    diis = QuackDIIS(1, [(N,)])
    for error, iterate in zip(ERRORS, SMOOTH_ITERATES):
        got = diis.push(np.array(error), np.array(iterate))
        assert np.asarray(got).tobytes() == np.array(iterate).tobytes()
    assert diis.rcond is None and diis.rcond_history == []


def flags_of(fn):
    """(fn(), the named flags it raised, the warning classes it warned)."""
    flags = []
    with collecting(flags), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fn()
    return result, [e['flag'] for e in flags], {w.category for w in caught}


def test_a_nan_error_makes_the_extrapolation_nan():
    diis = QuackDIIS(SIZE, [(N,)])
    diis.push(np.array(ERRORS[0]), np.array(SMOOTH_ITERATES[0]))
    bad = np.array(ERRORS[1])
    bad[3] = np.nan
    got, flags, classes = flags_of(lambda: diis.push(bad, np.array(SMOOTH_ITERATES[1])))
    assert np.isnan(got).all() and np.isnan(diis.rcond)
    assert flags == [FLAG_DIIS_NOT_FINITE] and ParquetNotFinite in classes


def test_a_nan_iterate_enters_the_sum_as_it_is():
    diis = QuackDIIS(SIZE, [(N,)])
    diis.push(np.array(ERRORS[0]), np.array(SMOOTH_ITERATES[0]))
    bad = np.array(SMOOTH_ITERATES[1])
    bad[3] = np.nan
    got, flags, _ = flags_of(lambda: diis.push(np.array(ERRORS[1]), bad))
    assert np.isnan(got[3]) and np.isfinite(np.delete(got, 3)).all() and flags == []


def test_an_exactly_singular_system():
    """Two equal errors make the bordered system exactly singular: dsysvx
    finds a zero pivot, and the least-squares weights (half each) are taken,
    flagged."""
    diis = QuackDIIS(SIZE, [(N,)])
    error = np.array(ERRORS[0])
    first, second = np.array(SMOOTH_ITERATES[0]), np.array(SMOOTH_ITERATES[1])
    diis.push(error, first)
    got, flags, classes = flags_of(lambda: diis.push(error.copy(), second))
    assert flags == [FLAG_DIIS_SINGULAR] and ParquetNotFinite in classes
    want = SINGULAR_WEIGHTS[0] * second + SINGULAR_WEIGHTS[1] * first
    assert relative_gap(got, want) <= WEIGHTS_REL, (got, want)


def test_the_slots():
    """A slot asked for twice before its extrapolation is the same; the
    extrapolation is written into the arrays handed in; reset forgets the
    history and keeps the slots (the next steps are a new history's, to the
    bit)."""
    diis = QuackDIIS(SIZE, [(2, 4), (N,)])
    errors, iterates = diis.next_slot()
    assert diis.next_slot()[0] is errors
    for block, error in zip(errors, (np.array(ERRORS[0]).reshape(2, 4), np.array(ERRORS[1]))):
        block[...] = error
    for block, iterate in zip(iterates, (np.array(SMOOTH_ITERATES[0]).reshape(2, 4),
                                         np.array(SMOOTH_ITERATES[1]))):
        block[...] = iterate
    out = (np.empty((2, 4)), np.empty(N))
    got = diis.extrapolate(out=out)
    assert got[0] is out[0] and got[1] is out[1] and len(diis) == 1
    used, fresh = QuackDIIS(SIZE, [(N,)]), QuackDIIS(SIZE, [(N,)])
    for error, iterate in zip(ERRORS[4:], SMOOTH_ITERATES[4:]):
        used.push(np.array(error), np.array(iterate))
    used.reset()
    assert len(used) == 0
    for error, iterate in zip(ERRORS[:4], SMOOTH_ITERATES[:4]):
        a = fresh.push(np.array(error), np.array(iterate))
        b = used.push(np.array(error), np.array(iterate))
        assert a.tobytes() == b.tobytes()


def test_the_refusals():
    with pytest.raises(ValueError, match='at least one pair'):
        QuackDIIS(0, [(N,)])
    with pytest.raises(ValueError, match='at least one block'):
        QuackDIIS(SIZE, [])
    diis = QuackDIIS(SIZE, [(N,)])
    with pytest.raises(RuntimeError, match='next_slot'):
        diis.extrapolate()
    with pytest.raises(ValueError, match='a block of shape'):
        diis.push(np.zeros(N + 1), np.zeros(N + 1))
    assert len(diis) == 0                  # refused before the history changed
    errors, iterates = diis.next_slot()
    with pytest.raises(ValueError, match='lies in the history'):
        diis.extrapolate(out=(iterates[0],))
    with pytest.raises(ValueError, match='C-contiguous float64'):
        diis.extrapolate(out=(np.empty(N, dtype=np.float32),))
    with pytest.raises(ValueError, match='output blocks'):
        diis.extrapolate(out=(np.empty(N), np.empty(N)))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
