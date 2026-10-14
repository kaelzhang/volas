"""Variances, covariances and correlations come from centred moments of exact sums.

The one-pass raw-moment forms — `Σx²/n − (Σx/n)²`, `(Σxy − ΣxΣy/n) / sqrt(…)` —
cancel catastrophically: at a price level of `12345.67` a window's variance of `1e-12`
is lost in rounding, a flat window's correlation comes out as `-1`, and a variance can
go negative. volas shifts the data by a value inside the window (the value at its
anchor row) before squaring, so the deviations carry no cancellation: a flat window is
exactly flat, a variance is within the documented `(36n + 2)·2⁻⁵³` relative bound
(docs/indicator-inputs.md "Centred moments"), a correlation stays in `[-1, 1]`, and a
zero variance is tested exactly.
"""

from fractions import Fraction

import numpy as np
import pytest

from volas import DataFrame

from test.exact import assert_exact

U = Fraction(1, 2**53)


def _var_bound(n):
    """The documented relative bound of a shifted variance over `n` rows
    (docs/indicator-inputs.md "Centred moments")."""
    return (36 * n + 2) * U

LEVELS = (0.1, 100.0, 1234.5, 12345.67, 98765.4321)


def _bits(got, want, ctx):
    assert_exact(got, want, ctx)


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
        assert abs(Fraction(var[i]) - exact) <= exact * _var_bound(20), (i, var[i], float(exact))
        assert abs(Fraction(std[i]) - Fraction(float(exact) ** 0.5)) <= Fraction(float(exact) ** 0.5) * (_var_bound(20) / 2 + 2 * U), i
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
        span = Fraction(max(close[i - 19:i + 1])) - Fraction(min(close[i - 19:i + 1]))
        assert abs(Fraction(slope[i]) - exact) <= 90 * U * span / 21 + U * abs(exact), (i, slope[i], float(exact))


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
        assert abs(Fraction(var[i]) - exact) <= exact * _var_bound(len(w)), (i, var[i], float(exact))
        assert abs(Fraction(std[i]) - Fraction(float(exact) ** 0.5)) <= Fraction(float(exact) ** 0.5) * (_var_bound(len(w)) / 2 + 2 * U), i
    assert (var[66:75] == 0.0).all() and (std[66:75] == 0.0).all()


def test_mfi_does_not_depend_on_the_volume_unit():
    """MFI is a ratio of money flows: only an exactly zero total flow reads 0, never a
    total below an absolute threshold, so the volume's unit cannot change it."""
    rng = np.random.default_rng(4)
    close = 0.05 + np.abs(np.cumsum(rng.normal(0, 1e-3, 200)))
    base = {"high": close * 1.001, "low": close * 0.999, "close": close}
    volume = rng.uniform(1, 10, 200)
    small = _check_paths({**base, "volume": volume * 1e-3}, "mfi:14")
    large = _check_paths({**base, "volume": volume * 1e6}, "mfi:14")
    assert np.isfinite(small[14:]).all() and (small[14:] > 0.0).any()
    assert np.allclose(small, large, rtol=1e-12, atol=0.0, equal_nan=True)
    # a window with no flow at all (flat prices) is exactly 0
    flat = np.full(40, 7.0)
    zero = _check_paths({"high": flat, "low": flat, "close": flat, "volume": np.full(40, 1e6)}, "mfi:14")
    assert (zero[14:] == 0.0).all()


def _exact_present_var(window, ddof=1):
    w = [Fraction(float(v)) for v in window if not np.isnan(v)]
    m = sum(w) / len(w)
    return sum((v - m) ** 2 for v in w) / (len(w) - ddof)


def _check_present_var(x, w, min_periods=2):
    var = np.asarray(DataFrame({"close": x})["close"].rolling(w, min_periods=min_periods).var(), dtype=float)
    for i in range(len(x)):
        win = x[max(0, i - w + 1):i + 1]
        present = win[~np.isnan(win)]
        if len(present) < max(min_periods, 2):
            assert np.isnan(var[i]), i
            continue
        exact = _exact_present_var(win)
        assert abs(Fraction(var[i]) - exact) <= exact * _var_bound(len(present)), (i, var[i], float(exact))


def test_the_window_shift_stays_inside_the_window_across_missing_rows():
    """An anchor segment whose rows are all missing shifts by the nearest present value
    before the anchor inside the window, never by a value that has left it."""
    x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1e9, 1.0 + 1e-7, 1.0 + 2e-7, 1.0 + 3e-7, 1.0 + 4e-7,
                  np.nan, np.nan, 2.0, 3.0, 4.0, 5.0])
    _check_present_var(x, 5)


@pytest.mark.parametrize("seed", range(6))
def test_the_window_variance_is_within_its_bound_with_missing_runs(seed):
    rng = np.random.default_rng(seed)
    x = 12345.67 + rng.normal(0, 1e-6, 300)
    x[rng.random(300) < 0.08] = 1e9  # outliers that leave
    for start in rng.integers(0, 290, 25):
        x[start:start + rng.integers(1, 9)] = np.nan  # missing runs, some covering anchors
    for w in (3, 5, 8):
        _check_present_var(x, w)


@pytest.mark.parametrize("method", ["var", "std", "sem"])
def test_the_window_variance_depends_on_the_window_values_alone(method):
    """The window API is a snapshot: a slice, a bounded frame and a fresh series of the
    same rows give the full computation's bits for every window they hold whole."""
    rng = np.random.default_rng(5)
    x = 12345.67 + rng.normal(0, 1e-6, 400)
    x[rng.random(400) < 0.05] = np.nan
    x[[50, 51, 230]] = [1e9, -1e9, 1e12]
    w = 7
    full = np.asarray(getattr(DataFrame({"close": x})["close"].rolling(w, min_periods=2), method)(), dtype=float)
    for start in (1, 3, 13, 100):
        part = DataFrame({"close": x}).iloc[start:]["close"]
        got = np.asarray(getattr(part.rolling(w, min_periods=2), method)(), dtype=float)
        _bits_eq(got[w - 1:], full[start + w - 1:], f"{method} slice {start}")
        fresh = np.asarray(getattr(DataFrame({"close": x[start:]})["close"].rolling(w, min_periods=2), method)(), dtype=float)
        _bits_eq(fresh[w - 1:], full[start + w - 1:], f"{method} fresh {start}")
    bounded = DataFrame({"close": x[:20]}, window=30, max_lookback=0)
    for i in range(20, len(x)):
        bounded.append(DataFrame({"close": x[i:i + 1]}))
    got = np.asarray(getattr(bounded["close"].rolling(w, min_periods=2), method)(), dtype=float)
    _bits_eq(got[w - 1:], full[len(x) - len(got) + w - 1:], f"{method} bounded")


def _bits_eq(got, want, ctx):
    assert_exact(got, want, ctx)
