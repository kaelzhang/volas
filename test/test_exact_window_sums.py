"""Running window sums are exact: each is the correctly rounded sum of its window.

A sliding sum that adds the entering row and subtracts the leaving one keeps, with a
plain `f64` accumulator, the rounding of every row it ever held: after a spike enters
and leaves, every later window is wrong; after a long history a flat window's deviation
is not zero. volas's running sums carry their rounding error (compensated summation),
detect when the compensation itself would round and then re-sum the window exactly, and
are recomputed from the window's own rows at every row whose absolute position is a
multiple of the window length. So a window's sum is `math.fsum` of its rows, and every
path (full, resumed, sliced, bounded) agrees bit for bit.
"""

import math
from fractions import Fraction

import numpy as np
import pytest

import volas
from volas import DataFrame

from .test_window_recompute import ACCUMULATING, _directive

# OHLCV in runs of huge and tiny values (see `test_window_recompute.WIDE`), long enough
# for the longest accumulating lookback to settle twice.
N = 1200
_rng = np.random.default_rng(29)
_c = 1000.0 + np.pi * np.cumsum(_rng.standard_normal(N))
_RUNS = np.where((np.arange(N) // 13) % 2 == 0, 1e12, 1e-8)
WIDE = {
    "open": (_c + _rng.uniform(-0.5, 0.5, N) * np.e) * _RUNS,
    "high": (_c + 1.0 + _rng.uniform(0, 1, N) * np.e) * _RUNS,
    "low": (_c - 1.0 - _rng.uniform(0, 1, N) * np.e) * _RUNS,
    "close": _c * _RUNS,
    "volume": _rng.uniform(100, 900, N) * np.sqrt(2.0) * _RUNS,
    "periods": np.full(N, 5.0),
}


def _bits(got, want, ctx):
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    same = (got == want) | (np.isnan(got) & np.isnan(want))
    if not same.all():
        bad = int(np.flatnonzero(~same)[0])
        raise AssertionError(f"{ctx}: row {bad}: {got[bad]!r} vs {want[bad]!r}")


def test_a_spike_leaves_no_trace():
    df = DataFrame({"close": np.array([1e20, 1.0, 2.0, 3.0, 4.0, 5.0])})
    ma = np.asarray(df["ma:2"])
    assert np.isnan(ma[0]) and ma[1:].tolist() == [5e19, 1.5, 2.5, 3.5, 4.5]
    assert np.asarray(df["sum:2"])[1:].tolist() == [1e20, 3.0, 5.0, 7.0, 9.0]
    # the same, appended one row at a time
    live = DataFrame({"close": np.array([1e20, 1.0])})
    live["ma:2"]
    for x in (2.0, 3.0, 4.0, 5.0):
        live.append(DataFrame({"close": np.array([x])}))
        live.fulfill()
    assert np.asarray(live["ma:2"])[1:].tolist() == [5e19, 1.5, 2.5, 3.5, 4.5]


def test_a_window_sum_is_its_correctly_rounded_exact_sum():
    rng = np.random.default_rng(7)
    c = 1e6 + np.pi * np.cumsum(rng.standard_normal(100_000))
    df = DataFrame({"close": c})
    for w in (20, 97):
        got_sum = np.asarray(df[f"sum:{w}"])[w - 1:]
        got_ma = np.asarray(df[f"ma:{w}"])[w - 1:]
        ref = np.array([math.fsum(c[i - w + 1:i + 1]) for i in range(w - 1, len(c))])
        assert (got_sum == ref).all(), w
        # the mean against the exact (rational) one, on every 97th window
        for k in range(0, len(ref), 97):
            exact = float(sum(Fraction(x) for x in c[k:k + w]) / w)
            assert abs(got_ma[k] - exact) <= np.spacing(abs(exact)), (w, k)


def test_a_flat_window_is_flat_after_a_long_history():
    rng = np.random.default_rng(3)
    walk = 1000.0 + np.pi * np.cumsum(rng.standard_normal(100_000))
    flat = 1234.5678
    close = np.concatenate([walk, np.full(60, flat)])
    first = len(walk) + 19  # the first all-flat 20-row window
    df = DataFrame({"close": close})
    for frame in (df, _appended(walk, np.full(60, flat))):
        assert (np.asarray(frame["stddev:20"])[first:] == 0.0).all()
        assert (np.asarray(frame["var:20"])[first:] == 0.0).all()
        assert (np.asarray(frame["ma:20"])[first:] == flat).all()
        assert (np.asarray(frame["boll.upper"])[first:] == flat).all()
        assert not np.asarray(frame["close > ma:20"])[first:].any()


def _appended(head, tail):
    """A frame over `head` with every directive of the test then fed `tail` live."""
    live = DataFrame({"close": head})
    for d in ("stddev:20", "var:20", "ma:20", "boll.upper", "close > ma:20"):
        live[d]
    for x in tail:
        live.append(DataFrame({"close": np.array([x])}))
        live.fulfill()
    return live


def _settled(directive, row):
    """The first row whose value no longer sees `row` through any running sum: the bad
    row has left every window, and an anchor has passed in each."""
    return row + 4 * (volas.directive_lookback(directive) + 1) + 30


@pytest.mark.parametrize("kind", ["spike", "nan"])
@pytest.mark.parametrize("key", sorted(ACCUMULATING))
def test_a_bad_row_stops_mattering_once_an_anchor_has_passed(key, kind):
    directive = _directive(key)
    bad = {k: v.copy() for k, v in WIDE.items()}
    for col in ("open", "high", "low", "close", "volume"):
        bad[col][150] = bad[col][150] * 1e15 if kind == "spike" else np.nan
    clean = np.asarray(DataFrame(WIDE).exec(directive), dtype=float)
    dirty = np.asarray(DataFrame(bad).exec(directive), dtype=float)
    start = _settled(directive, 150)
    assert start < N, directive
    _bits(dirty[start:], clean[start:], f"{directive} {kind}")


@pytest.mark.parametrize("key", sorted(ACCUMULATING))
def test_a_slice_keeps_its_rows_absolute_positions(key):
    """A contiguous slice is the same series from its first row on: computed fresh over
    the slice, a value equals the full frame's once the slice holds its sums' rows."""
    directive = _directive(key)
    df = DataFrame(WIDE)
    sub = df.iloc[123:]
    full = np.asarray(df.exec(directive), dtype=float)
    part = np.asarray(sub.exec(directive), dtype=float)
    skip = _settled(directive, 0)
    _bits(part[skip:], full[123 + skip:], directive)


@pytest.mark.parametrize("directive", ["minindex:5", "maxindex:5", "minmaxindex.max:5", "close + minindex:3"])
def test_positions_are_the_series_own_on_a_slice_and_a_bounded_frame(directive):
    """The index family emits absolute positions: computed fresh on a slice, carried
    across one, or refreshed on a bounded frame after its head was dropped."""
    df = DataFrame(WIDE)
    full = np.asarray(df.exec(directive), dtype=float)
    _bits(np.asarray(df.iloc[300:].exec(directive), dtype=float)[5:], full[305:], f"{directive} fresh")
    live = DataFrame({k: v[:40] for k, v in WIDE.items()}, window=30, max_lookback=[directive])
    live[directive]
    for i in range(40, 200):
        live.append(DataFrame({k: v[i:i + 1] for k, v in WIDE.items()}))
        live.fulfill()
    _bits(live[directive], full[170:200], f"{directive} bounded")


# --- multi-magnitude windows: every sum is the correctly rounded exact sum -------------
#
# Neumaier's single compensation word is not enough when two outliers of different
# magnitudes share a window (`1e35`, `1e20`, then ordinary values): the small values'
# bits round away inside the compensation and, once the outliers have left, the window
# reads 0. The sums detect that rounding and re-sum the window exactly, so every window
# sum is `math.fsum` of the window, on every path.

TWO_OUTLIERS = [1.1, 1e35, 1e20, 1.3, 2.7, 3.1, 4.9, 5.3, 6.1, 7.7, 8.3, 9.9, 10.1]


def _live(head, tail, directives, **kw):
    """A frame over `head` computing `directives`, then fed `tail` one row at a time."""
    frame = DataFrame({"close": np.asarray(head, dtype=float)}, **kw)
    for d in directives:
        frame[d]
    for x in tail:
        frame.append(DataFrame({"close": np.array([x], dtype=float)}))
        frame.fulfill()
    return frame


def test_two_outliers_of_different_magnitudes_leave_no_trace():
    directives = ["sum:4", "ma:4", "stddev:4", "close > ma:4"]
    full = DataFrame({"close": np.array(TWO_OUTLIERS)})
    live = _live(TWO_OUTLIERS[:3], TWO_OUTLIERS[3:], directives)
    bounded = _live(TWO_OUTLIERS[:3], TWO_OUTLIERS[3:], directives, window=4, max_lookback=directives)
    for row in (6, 7):
        window = TWO_OUTLIERS[row - 3:row + 1]
        alone = DataFrame({"close": np.array(window)})  # the window computed in isolation
        for d in directives:
            want = np.asarray(alone[d], dtype=float)[3]
            _bits(np.asarray(full[d], dtype=float)[row:row + 1], [want], f"{d} full @{row}")
            _bits(np.asarray(live[d], dtype=float)[row:row + 1], [want], f"{d} live @{row}")
        assert np.asarray(full["sum:4"])[row] == math.fsum(window)
        assert np.asarray(full["ma:4"])[row] == math.fsum(window) / 4
        assert np.asarray(full["close > ma:4"])[row] == (window[-1] > math.fsum(window) / 4)
    for d in directives:
        b = np.asarray(bounded[d], dtype=float)
        _bits(b, np.asarray(full[d], dtype=float)[-len(b):], f"{d} bounded")


def _multi_magnitude(n, seed):
    """Ordinary prices mixed with outliers anywhere in `±1e±300`, missing and infinite
    rows."""
    rng = np.random.default_rng(seed)
    outlier = rng.uniform(1, 10, n) * 10.0 ** rng.integers(-300, 301, n) * rng.choice([-1.0, 1.0], n)
    x = np.where(rng.random(n) < 0.35, outlier, 100.0 + rng.standard_normal(n))
    x[rng.random(n) < 0.03] = np.nan
    x[rng.random(n) < 0.02] = np.inf
    x[rng.random(n) < 0.02] = -np.inf
    return x


def _ieee_or_fsum(values):
    """A window's sum: NaN when it holds a NaN or both infinities, the infinity when it
    holds one, else the correctly rounded exact sum (`math.fsum`)."""
    if any(math.isnan(v) for v in values):
        return math.nan
    infs = {v for v in values if math.isinf(v)}
    if infs:
        return math.nan if len(infs) == 2 else infs.pop()
    return math.fsum(values)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("w", [1, 2, 3, 5, 8])
def test_a_window_sum_is_fsum_on_every_path(w, seed):
    x = _multi_magnitude(260, seed)
    want = np.array([math.nan] * (w - 1) + [_ieee_or_fsum(x[i - w + 1:i + 1]) for i in range(w - 1, len(x))])
    d = f"sum:{w}"
    full = np.asarray(DataFrame({"close": x}).exec(d), dtype=float)
    _bits(full, want, f"{d} full")
    live = _live(x[:w + 2], x[w + 2:], [d])
    _bits(live[d], want, f"{d} live")
    bounded = _live(x[:w + 2], x[w + 2:], [d], window=w + 3, max_lookback=[d])
    b = np.asarray(bounded[d], dtype=float)
    _bits(b, want[-len(b):], f"{d} bounded")
    # the mean of each clean window is its exact mean to within an ulp, on every path
    m = f"ma:{w}"
    ma = np.asarray(DataFrame({"close": x}).exec(m), dtype=float)
    _bits(_live(x[:w + 2], x[w + 2:], [m])[m], ma, f"{m} live")
    for i in range(w - 1, len(x)):
        win = x[i - w + 1:i + 1]
        if np.isfinite(win).all():
            exact = float(sum(Fraction(v) for v in win) / w)
            assert abs(ma[i] - exact) <= np.spacing(abs(exact)), (m, i)


def _present_sum(values):
    present = [v for v in values if not math.isnan(v)]
    return _ieee_or_fsum(present) if present else math.nan


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_pandas_window_sums_are_fsum(seed):
    x = _multi_magnitude(260, seed)
    s = DataFrame({"close": x})["close"]
    for w in (1, 2, 4, 7):
        want = [_present_sum(x[max(0, i - w + 1):i + 1]) for i in range(len(x))]
        _bits(s.rolling(w, min_periods=1).sum(), want, f"rolling({w}).sum")
        got = np.asarray(s.rolling(w, min_periods=1).mean(), dtype=float)
        for i in range(len(x)):
            win = [v for v in x[max(0, i - w + 1):i + 1] if not math.isnan(v)]
            if win and all(math.isfinite(v) for v in win):
                exact = float(sum(Fraction(v) for v in win) / len(win))
                assert abs(got[i] - exact) <= np.spacing(abs(exact)), (w, i)
    _bits(s.expanding(1).sum(), [_present_sum(x[:i + 1]) for i in range(len(x))], "expanding.sum")
    # the outliers of the issue: the rolling sum forgets them
    t = DataFrame({"close": np.array(TWO_OUTLIERS)})["close"]
    _bits(t.rolling(4).sum(), [math.nan] * 3 + [math.fsum(TWO_OUTLIERS[i - 3:i + 1]) for i in range(3, 13)], "issue")


def test_an_expanding_sum_holds_every_bit():
    # a long prefix whose exact sum needs more than two words: 1e30, then values whose
    # low bits a compensated pair cannot keep beside it
    x = np.array([1e30, 1e-30] + [1.0 + k * 2.0 ** -40 for k in range(50)] + [-1e30])
    s = DataFrame({"close": x})["close"]
    _bits(s.expanding(1).sum(), [math.fsum(x[:i + 1]) for i in range(len(x))], "expanding")
    want = float(sum(Fraction(v) for v in x) / len(x))
    assert abs(np.asarray(s.expanding(1).mean())[-1] - want) <= np.spacing(want)
