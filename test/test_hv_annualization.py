"""`hv` annualization: the sample standard deviation of one-bar log returns over
`period` bars, times sqrt(bars per year), where bars per year is chosen by the
time frame's unit:

- sub-day bars: trading_days x session_minutes / bar_minutes (exact for second bars);
- k-day bars: trading_days / k;
- weekly bars: 52; monthly bars: 12; yearly bars: 1.

Every value is checked against an independent NumPy computation, per unit and per
market session; an appended (resumed) column equals a full recompute bit for bit;
and a time frame outside the `TimeFrame` labels is rejected, never a panic.
"""

import math

import numpy as np
import pytest

import volas
from volas import DataFrame

N = 240
PERIOD = 10
_rng = np.random.default_rng(7)
CLOSE = 100.0 * np.exp(np.cumsum(_rng.normal(0.0, 0.01, N)))


def _frame(close=CLOSE):
    return DataFrame({"close": close})


def _reference(close, period, bars_per_year):
    """NumPy: ddof=1 rolling std of ln(c_t / c_{t-1}) times sqrt(bars_per_year)."""
    returns = np.full(len(close), np.nan)
    returns[1:] = np.log(close[1:] / close[:-1])
    out = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        out[i] = np.std(returns[i - period + 1:i + 1], ddof=1)
    return out * math.sqrt(bars_per_year)


# (directive, bars per year, what it models)
CASES = [
    ("hv:10", 252, "daily bars, the default 252 trading days"),
    ("hv:10,1d,252", 252, "daily bars, 252 trading days"),
    ("hv:10,1d,365", 365, "crypto daily bars, 365 calendar days"),
    ("hv:10,3d", 252 / 3, "3-day bars"),
    ("hv:10,1w", 52, "weekly bars"),
    ("hv:10,1W", 52, "weekly bars, upper-case alias"),
    ("hv:10,1M", 12, "monthly bars"),
    ("hv:10,1y", 1, "yearly bars"),
    ("hv:10,15m,252,390", 252 * 390 / 15, "US equities, 6.5 h session, 15-minute bars"),
    ("hv:10,1m,252,390", 252 * 390, "US equities, 1-minute bars"),
    ("hv:10,4h,252,390", 252 * 390 / 240, "US equities, 4-hour bars (fractional bars per day)"),
    ("hv:10,1s,252,390", 252 * 390 * 60, "US equities, 1-second bars (exact)"),
    ("hv:10,15m,252,240", 252 * 240 / 15, "CN A-shares, 4 h session"),
    ("hv:10,5m,247,330", 247 * 330 / 5, "HK equities, 5.5 h session"),
    ("hv:10,1h,252,1380", 252 * 1380 / 60, "CME E-mini, 23 h session"),
    ("hv:10,15m,365", 365 * 1440 / 15, "crypto 15-minute bars, the default 24 h session"),
    ("hv:10,1h,365,1440", 365 * 24, "crypto hourly bars"),
]


@pytest.mark.parametrize("directive,bars_per_year,what", CASES, ids=[c[0] for c in CASES])
def test_hv_matches_numpy_per_unit_and_market(directive, bars_per_year, what):
    got = np.asarray(_frame()[directive], dtype=float)
    want = _reference(CLOSE, PERIOD, bars_per_year)
    assert np.array_equal(np.isnan(got), np.isnan(want)), what
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0, err_msg=what)


def test_hv_defaults_are_daily_252_and_a_24h_session():
    df = _frame()
    full = np.asarray(df["hv:10,1d,252,1440"], dtype=float)
    np.testing.assert_array_equal(np.asarray(df["hv:10"], dtype=float), full)
    assert volas.directive_stringify("hv:10,1d,252,1440") == "hv:10"
    assert volas.directive_stringify("hv:10,15m,252,390") == "hv:10,15m,,390"


def test_hv_session_minutes_ignored_for_daily_and_longer_bars():
    df = _frame()
    for tf in ("1d", "3d", "1w", "1M", "1y"):
        np.testing.assert_array_equal(
            np.asarray(df[f"hv:10,{tf},252,390"], dtype=float),
            np.asarray(df[f"hv:10,{tf}"], dtype=float),
        )


@pytest.mark.parametrize("directive", [
    "hv:10", "hv:10,15m,252,390", "hv:10,1s,252,240", "hv:10,1w", "hv:10,1M",
])
def test_hv_resume_equals_full_recompute(directive):
    seed = 30
    live = _frame(CLOSE[:seed])
    live[directive]
    for i in range(seed, N):
        live.append(DataFrame({"close": CLOSE[i:i + 1]}, index=[i]))
        live.fulfill()
        if i % 11 == 0 or i == N - 1:
            got = np.asarray(live[directive], dtype=float)
            want = np.asarray(_frame(CLOSE[:i + 1])[directive], dtype=float)
            same = (got == want) | (np.isnan(got) & np.isnan(want))
            assert same.all(), f"{directive} @{i}"


@pytest.mark.parametrize("directive", [
    "hv:10,30s",          # not a TimeFrame label (used to round down to 1 minute)
    "hv:10,90s",          # not a TimeFrame label (used to round down to 1 minute)
    "hv:10,10m",          # not a TimeFrame label
    "hv:10,0m",
    "hv:10,5x",
    "hv:10,xm",
    "hv:10,1é",      # a multibyte last character (used to panic)
    "hv:10,15m,252,0",    # a session of zero minutes
    "hv:10,15m,252,1441", # longer than a day
    "hv:10,1d,0",         # zero trading days
])
def test_hv_rejects_an_invalid_configuration(directive):
    with pytest.raises(volas.DirectiveValueError):
        _frame()[directive]
    with pytest.raises(volas.DirectiveValueError):
        volas.directive_lookback(directive)
    with pytest.raises(volas.DirectiveValueError):
        volas.directive_stringify(directive)
