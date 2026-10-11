"""Running window sums are exact: compensated, and re-anchored at absolute positions.

A sliding sum that adds the entering row and subtracts the leaving one keeps, with a
plain `f64` accumulator, the rounding of every row it ever held: after a spike enters
and leaves, every later window is wrong; after a long history a flat window's deviation
is not zero. volas's running sums carry their rounding error (compensated summation)
and are recomputed from the window's own rows at every row whose absolute position is
a multiple of the window length, so a value depends only on the rows near it: once a
bad row has left the window and an anchor has passed, the values are the clean data's,
bit for bit — and every path (full, resumed, sliced, bounded) agrees.
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


def test_a_window_sum_is_its_exact_sum_within_an_ulp():
    rng = np.random.default_rng(7)
    c = 1e6 + np.pi * np.cumsum(rng.standard_normal(100_000))
    df = DataFrame({"close": c})
    for w in (20, 97):
        got_sum = np.asarray(df[f"sum:{w}"])[w - 1:]
        got_ma = np.asarray(df[f"ma:{w}"])[w - 1:]
        ref = np.array([math.fsum(c[i - w + 1:i + 1]) for i in range(w - 1, len(c))])
        assert (np.abs(got_sum - ref) <= np.spacing(np.abs(ref))).all(), w
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
