"""A missing row does not exist for a recursive indicator.

A row where any input of a recursive command (EMA / Wilder / cumulative / Hilbert /
SAR families, and a TA-Lib MA-type argument that selects a recursive average) is NaN
outputs NaN, and the recursion continues from the last complete row as if the row
were not there. So the output equals the command run on the series with those rows
deleted, NaN re-inserted at the deleted positions — bit-exact, since both run the
same kernel over the same rows. An infinity is a value, not a missing row.

Every incremental path (the plain append resume, the tf forming row, the period
rollover) must agree with a from-scratch full recompute, row for row.
"""

import re
from pathlib import Path

import numpy as np
import pytest

import volas
from volas import DataFrame

from test.test_window_locality import RECURSIVE, SUBCOMMANDS

ROOT = Path(__file__).resolve().parent.parent
OHLCV = ("open", "high", "low", "close", "volume")

_C = ("close",)
_HL = ("high", "low")
_HLC = ("high", "low", "close")
_CV = ("close", "volume")
_HLCV = ("high", "low", "close", "volume")

# The input columns each recursive command reads (its default `@` series).
INPUTS = {
    **{c: _C for c in (
        "ema", "smma", "macdfix", "dema", "tema", "t3", "kama", "macd", "rsi", "cmo",
        "trix", "stochrsi", "tsi", "crsi", "keltner", "ht_dcperiod", "ht_dcphase",
        "ht_phasor", "ht_sine", "ht_trendline", "ht_trendmode", "mama",
    )},
    **{c: _HL for c in ("sar", "sarext", "plus_dm", "minus_dm", "mass_index")},
    **{c: _HLC for c in (
        "kdj", "stoch_momentum", "ttm_squeeze", "wad", "supertrend", "atr", "plus_di",
        "minus_di", "dx", "adx", "adxr", "natr", "kcw",
    )},
    **{c: _CV for c in ("pvt", "nvi", "pvi", "efi", "obv")},
    **{c: _HLCV for c in ("ad", "adosc")},
    "asi": ("open", "high", "low", "close"),
}
SUB_INPUTS = {"keltner.upper": _HLC, "keltner.lower": _HLC}

# A TA-Lib MA-type argument that selects a recursive average makes the command recursive.
MATYPE = {
    "ma:5,1": _C,
    "ma:5,7": _C,
    "apo:5,10,1": _C,
    "ppo:5,10,3": _C,
    "macdext:5,1,10,6": _C,
    "macdext.signal:5,0,10,0,4,1": _C,
    "mavp:2,8,8@close,periods": ("close", "periods"),
    "stoch.k:5,3,1": _HLC,
    "stoch.d:5,3,0,3,4": _HLC,
    "stochf.d:5,3,1": _HLC,
}
# Explicit operands: the inputs are what the operands read.
OPERANDS = {
    "ema:5@open": ("open",),
    "obv@open,volume": ("open", "volume"),
    "rsi:5@(close + open)": ("close", "open"),
    "ema:5@(rsi:5)": _C,
    "atr:5@high,low,open": ("high", "low", "open"),
}


def _sub_key(key):
    return key.split(":", 1)[0]


CASES = {
    **{d: INPUTS[c] for c, d in RECURSIVE.items()},
    **{d: SUB_INPUTS.get(k, INPUTS[k.split(".", 1)[0]])
       for k, d in SUBCOMMANDS.items() if k.split(".", 1)[0] in RECURSIVE},
    **MATYPE,
    **OPERANDS,
}

N = 260
_rng = np.random.default_rng(7)
_c = 100.0 + np.cumsum(_rng.standard_normal(N))
BASE = dict(
    open=_c + _rng.uniform(-0.5, 0.5, N), high=_c + 1.0 + _rng.uniform(0, 1, N),
    low=_c - 1.0 - _rng.uniform(0, 1, N), close=_c.copy(),
    volume=_rng.integers(100, 900, N).astype(float), periods=np.full(N, 5.0),
)

# name -> {column: rows set to NaN}
PATTERNS = {
    "row": {c: [100] for c in OHLCV},
    "run": {c: list(range(120, 125)) for c in OHLCV},
    "close_only": {"close": [90, 91, 160]},
    "high_only": {"high": [130]},
    "volume_only": {"volume": [140, 200]},
    "open_only": {"open": [110]},
    "lead_and_interior": {
        **{c: [0, 1, 2, 100, 150, 151, 152] for c in OHLCV},
        "close": [0, 1, 2, 100, 150, 151, 152, 230],
    },
    "last_rows": {c: [N - 2, N - 1] for c in OHLCV},
}


def _mutated(pattern):
    data = {k: v.copy() for k, v in BASE.items()}
    for col, rows in PATTERNS[pattern].items():
        data[col][rows] = np.nan
    return data


def _bit_exact(got, want, ctx):
    """Equal where both are numbers, NaN in the same cells — no tolerance."""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"{ctx}: shape {got.shape} != {want.shape}"
    nan = np.isnan(got) & np.isnan(want)
    a, b = np.where(nan, 0.0, got), np.where(nan, 0.0, want)
    if not np.array_equal(a, b):
        bad = int(np.flatnonzero(a != b)[0])
        raise AssertionError(f"{ctx}: row {bad}: got {got[bad]!r}, want {want[bad]!r}")


def _deleted(directive, data, inputs):
    """The directive on the series with every row missing an input deleted, NaN
    re-inserted at the deleted positions."""
    missing = np.zeros(N, dtype=bool)
    for col in inputs:
        missing |= np.isnan(data[col])
    kept = ~missing
    out = np.full(N, np.nan)
    out[kept] = np.asarray(DataFrame({k: v[kept] for k, v in data.items()}).exec(directive), float)
    return out


@pytest.mark.parametrize("pattern", sorted(PATTERNS))
@pytest.mark.parametrize("directive", sorted(CASES))
def test_missing_rows_are_deleted_rows(directive, pattern):
    data = _mutated(pattern)
    got = np.asarray(DataFrame(data).exec(directive), float)
    _bit_exact(got, _deleted(directive, data, CASES[directive]), f"{directive} [{pattern}]")


def test_recursion_continues_from_the_last_complete_row():
    close = np.arange(10.0, 30.0)
    close[8] = np.nan
    got = np.asarray(DataFrame({"close": close}).exec("ema:3"), float)
    assert np.isnan(got[8])
    # the row-7 state (16.0) stepped over close[9] = 19 with k = 0.5
    assert got[9] == 17.5
    assert np.isfinite(got[9:]).all()


def test_an_infinity_is_a_value_not_a_missing_row():
    close = np.arange(10.0, 30.0)
    close[8] = np.inf
    got = np.asarray(DataFrame({"close": close}).exec("ema:3"), float)
    assert got[8] == np.inf
    # it is kept, not skipped: an RSI over it is NaN from there on, on every path
    close = BASE["close"].copy()
    close[100] = np.inf
    got = np.asarray(DataFrame({"close": close}).exec("stochrsi.d"), float)
    assert np.isfinite(got[50:100]).all() and np.isnan(got[101:]).all()


def test_an_integer_column_na_is_a_missing_row():
    close = BASE["close"][:40].copy()
    volume = [int(v) for v in BASE["volume"][:40]]
    got = DataFrame({"close": close, "volume": volume[:20] + [None] + volume[21:]}).exec("obv")
    kept = [i for i in range(40) if i != 20]
    want = np.full(40, np.nan)
    want[kept] = np.asarray(
        DataFrame({"close": close[kept], "volume": [volume[i] for i in kept]}).exec("obv"), float
    )
    _bit_exact(np.asarray(got, float), want, "obv over an int volume with NA")


def test_a_required_operand_left_out_is_still_an_error():
    with pytest.raises(ValueError, match="series argument #1 is required"):
        DataFrame(BASE).exec("mavp:2,8,8@close,")


def test_every_recursive_command_is_covered():
    """The Rust classification of recursive commands is exactly the one exercised here."""
    src = "".join(p.read_text() for p in (ROOT / "crates/volas-directive/src").rglob("*.rs"))
    body = re.search(r"const RECURSIVE: &\[&str\] = &\[(.*?)\];", src, re.S).group(1)
    assert set(re.findall(r'"([a-z0-9_]+)"', body)) == set(RECURSIVE)


@pytest.mark.parametrize("directive", sorted(CASES))
def test_inputs_are_what_the_command_reads(directive):
    """`INPUTS` is right: a NaN in an input column is a missing row, elsewhere it is not."""
    clean = np.asarray(DataFrame(BASE).exec(directive), float)
    row = 150
    for col in OHLCV:
        data = {k: v.copy() for k, v in BASE.items()}
        data[col][row] = np.nan
        got = np.asarray(DataFrame(data).exec(directive), float)
        if col in CASES[directive]:
            assert np.isnan(got[row]), f"{directive}: NaN {col} must be a missing row"
        else:
            _bit_exact(got, clean, f"{directive}: {col} is not an input")


# --- every incremental path agrees with a full recompute --------------------------

# `mavp` reads a `periods` column the OHLCV fold does not carry; it has no resume.
_LIVE = sorted(set(CASES) - set(OPERANDS) - {"mavp:2,8,8@close,periods"}) + [
    "ema:5@open", "obv@open,volume",
]


# Live-path divergences that clean data shows too, independent of missing rows: a
# sliding-sum resume drifts by a few ULPs, and a tf-folded column fills its last
# warm-up row. Those are compared within 1e-9 / past the warm-up; the rest bit-exact.
ULP_DRIFT = {"ht_trendline", "stochrsi.d"}
FOLD_WARMUP = {
    "adosc", "ht_dcperiod", "ht_dcphase", "ht_phasor", "ht_phasor.quadrature", "ht_sine",
    "ht_sine.leadsine", "ht_trendline", "ht_trendmode", "mama", "mama.fama",
}


def _live_matches(live, ref, directive, ctx, fold=False):
    live, ref = np.asarray(live, dtype=float), np.asarray(ref, dtype=float)
    if fold and directive in FOLD_WARMUP:
        warm = volas.directive_lookback(directive)
        live, ref = live[warm:], ref[warm:]
    if directive in ULP_DRIFT:
        np.testing.assert_array_equal(np.isnan(live), np.isnan(ref), err_msg=ctx)
        np.testing.assert_allclose(live, ref, rtol=1e-9, atol=1e-9, err_msg=ctx)
    else:
        _bit_exact(live, ref, ctx)


def _plain_frame(data, rows):
    return DataFrame({k: v[rows] for k, v in data.items()})


@pytest.mark.parametrize("directive", _LIVE)
def test_plain_append_matches_a_full_recompute(directive):
    data = {k: v.copy() for k, v in BASE.items()}
    # missing rows around the cache seed, back to back, and at the newest row
    for col, rows in {
        "close": [99, 120, 121, 170, 171, 172, 205, 230, 259],
        "high": [100, 182, 240],
        "low": [190],
        "volume": [150, 211],
        "open": [215],
    }.items():
        data[col][rows] = np.nan
    seed = 100
    live = _plain_frame(data, slice(0, seed))
    live[directive]
    i = seed
    steps = [1, 1, 3, 1, 2]  # single-bar (scalar resume) and multi-bar (Vec resume)
    k = 0
    while i < N:
        j = min(N, i + steps[k % len(steps)])
        live.append(_plain_frame(data, slice(i, j)))
        live.fulfill()
        ref = _plain_frame(data, slice(0, j))
        _live_matches(live[directive], ref.exec(directive), directive, f"{directive} @{j}")
        i, k = j, k + 1


def _fine(n_bars, missing_periods, missing_fine):
    """`n_bars` 1-minute bars; every fine bar of a 15m period in `missing_periods`, and
    each fine bar in `missing_fine`, has all its OHLCV missing."""
    n = 15 * n_bars
    x = np.arange(n, dtype=float)
    close = 100.0 + 0.01 * x + 2.0 * np.sin(0.05 * x) + np.sin(0.7 * x)
    cols = dict(open=close - 0.1, high=close + 0.5 + 0.1 * np.cos(x), low=close - 0.6,
                close=close, volume=100.0 + (x % 7))
    for p in missing_periods:
        for c in OHLCV:
            cols[c][15 * p:15 * (p + 1)] = np.nan
    for i in missing_fine:
        for c in OHLCV:
            cols[c][i] = np.nan
    ts = (np.datetime64("2026-01-05") + x.astype("timedelta64[m]")).astype("datetime64[ns]")
    return cols, ts


def _fold_ref(live):
    ohlcv = {c: np.asarray(live[c], dtype=float) for c in OHLCV}
    return DataFrame(ohlcv, index=np.asarray(live.index)), ohlcv


# Periods 70, 71 and 85 fold to missing bars: the forming row is missing while each
# forms, and the rollover after each closes a missing bar. Fine bar 15*90+14 (the last
# of period 90) is missing alone.
_MISSING_PERIODS = (70, 71, 85)
_MISSING_FINE = (15 * 90 + 14, 15 * 95 + 3)


@pytest.mark.parametrize("directive", _LIVE)
def test_tf_fold_matches_a_full_recompute(directive):
    n_bars = 110
    cols, ts = _fine(n_bars, _MISSING_PERIODS, _MISSING_FINE)
    live = volas.DataFrame({k: v[:1] for k, v in cols.items()}, index=ts[:1], time_frame="15m")
    live[directive]
    checks = {15 * p + o for p in (60, 70, 71, 72, 85, 86, 90, 91, 95, 109) for o in (0, 7, 14)}
    seen_missing_forming = False
    for i in range(1, len(ts)):
        live.append(volas.DataFrame({k: v[i:i + 1] for k, v in cols.items()}, index=ts[i:i + 1]))
        live.fulfill()
        if i in checks:
            ref, ohlcv = _fold_ref(live)
            seen_missing_forming |= bool(np.isnan(ohlcv["close"][-1]))
            _live_matches(live[directive], ref[directive], directive, f"{directive} i={i}", fold=True)
    assert seen_missing_forming, "the scenario must fold a missing forming row"
    assert np.isnan(np.asarray(live["close"], float)).sum() >= len(_MISSING_PERIODS)


@pytest.mark.parametrize("directive", _LIVE)
def test_constructed_tf_frame_rolls_over_a_missing_closed_bar(directive):
    """Built from closed bars whose last one is missing, then rolled over."""
    n_closed, n_fine = 90, 40
    x = np.arange(n_closed + n_fine, dtype=float)
    close = 100.0 + 0.2 * x + 3.0 * np.sin(1.1 * x)
    cols = dict(open=close - 0.3, high=close + 1.0, low=close - 1.0, close=close, volume=1000.0 + x)
    for c in OHLCV:
        cols[c][[40, 60, 61, n_closed - 1]] = np.nan
    minutes = np.r_[np.arange(n_closed) * 5, 5 * n_closed + np.arange(n_fine)]
    ts = (np.datetime64("2026-01-01") + minutes.astype("timedelta64[m]")).astype("datetime64[ns]")
    for c in OHLCV:  # the fine bars of the second folded period are all missing
        cols[c][n_closed + 5:n_closed + 10] = np.nan
    live = volas.DataFrame({k: v[:n_closed] for k, v in cols.items()}, index=ts[:n_closed], time_frame="5m")
    live[directive]
    for i in range(n_closed, n_closed + n_fine):
        live.append(volas.DataFrame({k: v[i:i + 1] for k, v in cols.items()}, index=ts[i:i + 1]))
        live.fulfill()
        ref, _ = _fold_ref(live)
        _live_matches(live[directive], ref[directive], directive, f"{directive} i={i}", fold=True)
