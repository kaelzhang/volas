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

from test.exact import assert_exact, same_exact

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
    assert same_exact(np.isnan(got), np.isnan(want)), what
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0, err_msg=what)


def test_hv_defaults_are_daily_252_and_a_24h_session():
    df = _frame()
    full = np.asarray(df["hv:10,1d,252,1440"], dtype=float)
    assert_exact(np.asarray(df["hv:10"], dtype=float), full)
    assert volas.directive_stringify("hv:10,1d,252,1440") == "hv:10"
    assert volas.directive_stringify("hv:10,15m,252,390") == "hv:10,15m,,390"


@pytest.mark.parametrize("directive,canonical", [
    ("hv:10,4h,252,240", "hv:10,4h,,240"),  # a bar exactly as long as the session
    ("hv:10,12h,252,720", "hv:10,12h,,720"),
    ("hv:10,1d,252,1440", "hv:10"),         # default arguments stay legal on any frame
    ("hv:10,1w,252,1440", "hv:10,1w"),
    ("hv:10,1M,252", "hv:10,1M"),
    ("hv:10,1d,365", "hv:10,,365"),         # trading_days applies to day bars
    ("hv:10,3d,250", "hv:10,3d,250"),
])
def test_hv_applicable_arguments_execute(directive, canonical):
    assert len(_frame()[directive]) == N
    volas.directive_lookback(directive)
    assert volas.directive_stringify(directive) == canonical


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
            same = same_exact(got, want)
            assert same, f"{directive} @{i}"


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


@pytest.mark.parametrize("directive,message", [
    # a sub-day bar longer than the trading session
    ("hv:10,8h,252,240", "longer than the 240-minute session"),
    ("hv:10,12h,252,390", "longer than the 390-minute session"),
    ("hv:,8h,252,240", "longer than the 240-minute session"),  # form without the period
    # an explicit non-default argument the frame does not use
    ("hv:10,1d,252,390", "session_minutes applies to sub-day time frames only"),
    ("hv:10,3d,252,240", "session_minutes applies to sub-day time frames only"),
    ("hv:10,1w,252,390", "session_minutes applies to sub-day time frames only"),
    ("hv:10,1w,365", "trading_days applies to sub-day and day time frames only"),
    ("hv:10,1M,250", "trading_days applies to sub-day and day time frames only"),
    ("hv:10,1y,365,1440", "trading_days applies to sub-day and day time frames only"),
])
def test_hv_rejects_an_inapplicable_argument(directive, message):
    entries = [volas.directive_stringify]
    if not directive.startswith("hv:,"):  # the bound entries need the period
        entries += [lambda d: _frame()[d], volas.directive_lookback]
    for entry in entries:
        with pytest.raises(volas.DirectiveValueError, match=message):
            entry(directive)
