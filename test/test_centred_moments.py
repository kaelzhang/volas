"""Variances, covariances and correlations come from centred moments of exact sums.

The one-pass raw-moment forms — `Σx²/n − (Σx/n)²`, `(Σxy − ΣxΣy/n) / sqrt(…)` —
cancel catastrophically: at a price level of `12345.67` a window's variance of `1e-12`
is lost in rounding, a flat window's correlation comes out as `-1`, and a variance can
go negative. volas forms `n·Σab − Σa·Σb` from the window's exact sums (squares and
products summed exactly, as their rounded values plus their rounding errors), in
double-double arithmetic and, near cancellation, exactly — so a flat window is exactly
flat, a correlation stays in `[-1, 1]`, and a zero variance is tested exactly.
"""

from fractions import Fraction

import numpy as np
import pytest

from volas import DataFrame

LEVELS = (0.1, 100.0, 1234.5, 12345.67, 98765.4321)


def _bits(got, want, ctx):
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    same = (got == want) | (np.isnan(got) & np.isnan(want))
    if not same.all():
        bad = int(np.flatnonzero(~same)[0])
        raise AssertionError(f"{ctx}: row {bad}: {got[bad]!r} vs {want[bad]!r}")


def _paths(cols, directive, seed=30):
    """The directive over `cols` computed in full, appended one row at a time after
    `seed` rows, and on a bounded frame fed the same rows: (full, live, bounded)."""
    n = len(next(iter(cols.values())))
    full = np.asarray(DataFrame(cols).exec(directive), dtype=float)
    live = DataFrame({k: v[:seed] for k, v in cols.items()})
    bounded = DataFrame({k: v[:seed] for k, v in cols.items()}, window=25, max_lookback=[directive])
    for frame in (live, bounded):
        frame[directive]
    for i in range(seed, n):
        for frame in (live, bounded):
            frame.append(DataFrame({k: v[i:i + 1] for k, v in cols.items()}))
            frame.fulfill()
    return full, np.asarray(live[directive], dtype=float), np.asarray(bounded[directive], dtype=float)


def _check_paths(cols, directive):
    full, live, bounded = _paths(cols, directive)
    _bits(live, full, f"{directive} live")
    _bits(bounded, full[-len(bounded):], f"{directive} bounded")
    return full


@pytest.mark.parametrize("history", [0, 500])
@pytest.mark.parametrize("level", LEVELS)
def test_a_flat_window_has_zero_correlation(level, history):
    rng = np.random.default_rng(0)
    hist = level + np.cumsum(rng.normal(0, level * 1e-3, history))
    close = np.concatenate([hist, np.full(60, level)])
    cols = {"close": close, "open": np.concatenate([hist * 0.999, np.full(60, level)])}
    full = _check_paths(cols, "correl:20@close,open")
    assert (full[-40:] == 0.0).all(), (level, full[-40:])
    # one flat series against a moving one: zero variance on one side is enough
    cols["open"] = np.concatenate([hist * 0.999, level + np.arange(60.0)])
    assert (_check_paths(cols, "correl:20@close,open")[-40:] == 0.0).all()


def _exact_var(window):
    w = [Fraction(v) for v in window]
    m = sum(w) / len(w)
    return sum((v - m) ** 2 for v in w) / len(w)


@pytest.mark.parametrize("level,sigma", [(12345.67, 1e-6), (12345.67, 1e-4), (100.0, 1e-9), (0.1, 1e-12)])
def test_a_near_flat_window_keeps_its_variance(level, sigma):
    rng = np.random.default_rng(1)
    close = level + rng.normal(0, sigma, 120)
    cols = {"close": close}
    var = _check_paths(cols, "var:20")
    std = _check_paths(cols, "stddev:20")
    for i in range(19, len(close)):
        exact = _exact_var(close[i - 19:i + 1])
        assert abs(Fraction(var[i]) - exact) <= exact * Fraction(1, 2**50), (i, var[i], float(exact))
        assert abs(std[i] - float(exact) ** 0.5) <= 1e-14 * float(exact) ** 0.5, i
    up = _check_paths(cols, "boll.upper")
    ma = np.asarray(DataFrame(cols)["ma:20"], dtype=float)
    assert (up[19:] >= ma[19:]).all()


def test_a_flat_window_has_zero_variance_slope_and_beta():
    for level in LEVELS:
        close = np.concatenate([level * (1.0 + np.arange(30) * 1e-3), np.full(40, level)])
        cols = {"close": close, "open": close[::-1].copy()}
        for d in ("var:20", "stddev:20", "linearreg_slope:20"):
            assert (np.abs(_check_paths(cols, d)[-20:]) == 0.0).all(), (level, d)
    # constant returns (a doubling series): beta's x variance is exactly zero
    x = 2.0 ** np.arange(40)
    cols = {"close": x, "open": x * np.linspace(1, 2, 40)}
    assert (_check_paths(cols, "beta:5@close,open")[6:] == 0.0).all()


def test_a_near_flat_slope_is_exact():
    close = 12345.67 + np.arange(80) * 1e-9
    slope = _check_paths({"close": close}, "linearreg_slope:20")
    ages = [Fraction(k) for k in range(20)]
    sx = sum(ages)
    sxx = sum(a * a for a in ages)
    for i in range(19, len(close)):
        w = [Fraction(v) for v in close[i - 19:i + 1]][::-1]  # age 0 = newest
        sy = sum(w)
        sxy = sum(a * b for a, b in zip(ages, w))
        exact = -(20 * sxy - sx * sy) / (20 * sxx - sx * sx)  # per bar forward in time
        assert abs(Fraction(slope[i]) - exact) <= abs(exact) * Fraction(1, 2**48), (i, slope[i], float(exact))


@pytest.mark.parametrize("level", LEVELS)
def test_a_perfect_correlation_stays_within_one(level):
    rng = np.random.default_rng(2)
    x = level + np.cumsum(rng.normal(0, level * 1e-7, 120))
    cols = {"close": x, "open": 3.0 * x + 1.0, "high": -x}
    pos = _check_paths(cols, "correl:2@close,open")
    neg = _check_paths(cols, "correl:20@close,high")
    assert (np.abs(pos[1:]) <= 1.0).all() and (np.abs(pos[1:] - 1.0) <= 1e-12).all()
    assert (np.abs(neg[19:]) <= 1.0).all() and (np.abs(neg[19:] + 1.0) <= 1e-12).all()


@pytest.mark.parametrize("level,sigma", [(12345.67, 1e-6), (100.0, 1e-9)])
def test_the_pandas_window_variance_is_exact(level, sigma):
    rng = np.random.default_rng(3)
    x = level + rng.normal(0, sigma, 80)
    x[[10, 40]] = np.nan
    x[60:75] = level  # a constant run
    s = DataFrame({"close": x})["close"]
    var = np.asarray(s.rolling(7, min_periods=2).var(), dtype=float)
    std = np.asarray(s.rolling(7, min_periods=2).std(), dtype=float)
    for i in range(len(x)):
        w = [Fraction(v) for v in x[max(0, i - 6):i + 1] if not np.isnan(v)]
        if len(w) < 2:
            continue
        m = sum(w) / len(w)
        exact = sum((v - m) ** 2 for v in w) / (len(w) - 1)
        assert abs(Fraction(var[i]) - exact) <= exact * Fraction(1, 2**50), (i, var[i], float(exact))
        assert abs(std[i] - float(exact) ** 0.5) <= 1e-15 * float(exact) ** 0.5, i
    assert (var[66:75] == 0.0).all() and (std[66:75] == 0.0).all()
