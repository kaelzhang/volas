"""R1 — the live tf-fold's anchor-preserving forming refresh must stay bit-exact
versus a from-scratch full recompute of the same indicator over the same folded
15m OHLCV, across warm-up, many rollovers, the volatile forming row, and window
compaction. (atr:14 is Wilder-recursive; ema:20 is single-state; rsi:14 is
two-state — all carry state and so exercise the anchor/finalize lifecycle.)
"""

import numpy as np
import pandas as pd
import pytest

import volas

from test.exact import assert_exact


def _load(n):
    csv = pd.read_csv("test/data/sample_1m_20k.csv")
    cols = {c: csv[c].to_numpy(float)[:n] for c in ("open", "high", "low", "close", "volume")}
    ts = (csv["open_time"].to_numpy()[:n] * 1_000_000).astype("datetime64[ns]")
    return cols, ts


def _bit_exact(live, ref, ctx):
    assert_exact(live, ref, ctx)


@pytest.mark.parametrize("directive", ["atr:14", "ema:20", "rsi:14", "natr:14", "cmo:14"])
def test_forming_resume_bit_exact_vs_full_recompute(directive):
    n = 3000
    cols, ts = _load(n)
    live = volas.DataFrame({k: v[:1] for k, v in cols.items()}, index=ts[:1], time_frame="15m")
    live[directive]
    checkpoints = {300, 900, 1800, n - 1}
    for i in range(1, n):
        bar = volas.DataFrame({k: v[i : i + 1] for k, v in cols.items()}, index=ts[i : i + 1])
        live.append(bar)
        live.fulfill()
        if i in checkpoints:
            ohlcv = {c: np.asarray(live[c]) for c in ("open", "high", "low", "close", "volume")}
            ref = volas.DataFrame(ohlcv, index=np.asarray(live.index))
            _bit_exact(np.asarray(live[directive]), np.asarray(ref[directive]), f"{directive} i={i}")


# A plain (non-tf) append advances the carried state every bar, so the scalar step
# must write back the state it advanced to — which is not the row's value for the
# two-state Wilder pair (rsi / cmo carry the average gain and loss) or for natr (it
# carries the ATR it scales).
@pytest.mark.parametrize(
    "directive", ["atr:14", "ema:20", "smma:14", "rsi:14", "natr:14", "cmo:14"]
)
def test_plain_append_scalar_resume_bit_exact_vs_full_recompute(directive):
    n, seed = 600, 100
    cols, ts = _load(n)
    live = volas.DataFrame({k: v[:seed] for k, v in cols.items()}, index=ts[:seed])
    live[directive]
    for i in range(seed, n):
        live.append(volas.DataFrame({k: v[i : i + 1] for k, v in cols.items()}, index=ts[i : i + 1]))
        live.fulfill()
    ref = volas.DataFrame(cols, index=ts)
    _bit_exact(np.asarray(live[directive]), np.asarray(ref[directive]), directive)


# Every recursive indicator with a zero-alloc scalar `*_resume_one` twin: the live
# forming-row resume must stay bit-exact versus a from-scratch full recompute, exactly
# like atr above. (The index family and stochrsi `.d` have no scalar twin and keep the
# Vec resume; the HT family + a few length-dependent-warm-up indicators are excluded
# from the scalar path because their resume already diverges in warm-up on either path.)
_SCALAR_TWINS = [
    "obv", "ad", "dema:20", "tema:20", "t3:20", "kama:20", "macd", "macdfix", "sar",
    "sarext", "plus_dm:14", "minus_dm:14", "plus_di:14", "minus_di:14", "dx:14", "adx:14", "adxr:14", "pvt", "nvi",
    "pvi", "efi:13", "tsi", "mass_index", "wad", "keltner:20", "supertrend", "trix:30",
    # multi-output sub-lines (exercise the sub-selector dispatch arms + their twins):
    "macd.signal", "macd.histogram", "macdfix.signal", "keltner.upper", "keltner.lower",
    "supertrend.direction",
]


@pytest.mark.parametrize("directive", _SCALAR_TWINS)
def test_scalar_twin_forming_bit_exact(directive):
    n = 1800
    cols, ts = _load(n)
    live = volas.DataFrame({k: v[:1] for k, v in cols.items()}, index=ts[:1], time_frame="15m")
    live[directive]
    checkpoints = {600, n - 1}
    for i in range(1, n):
        live.append(volas.DataFrame({k: v[i : i + 1] for k, v in cols.items()}, index=ts[i : i + 1]))
        live.fulfill()
        if i in checkpoints:
            ohlcv = {c: np.asarray(live[c]) for c in ("open", "high", "low", "close", "volume")}
            ref = volas.DataFrame(ohlcv, index=np.asarray(live.index))
            _bit_exact(np.asarray(live[directive]), np.asarray(ref[directive]), f"{directive} i={i}")


def test_forming_resume_windowed_matches_unbounded():
    """A bounded window (forcing periodic compaction) yields the same visible atr
    tail as the unbounded live frame — the anchor survives the head-dropping slice."""
    n = 2500
    cols, ts = _load(n)
    unb = volas.DataFrame({k: v[:1] for k, v in cols.items()}, index=ts[:1], time_frame="15m")
    win = volas.DataFrame(
        {k: v[:1] for k, v in cols.items()},
        index=ts[:1],
        time_frame="15m",
        window=20,
        max_lookback=["atr:14"],
    )
    unb["atr:14"]
    win["atr:14"]
    for i in range(1, n):
        bar = volas.DataFrame({k: v[i : i + 1] for k, v in cols.items()}, index=ts[i : i + 1])
        unb.append(bar)
        unb.fulfill()
        win.append(bar)
        win.fulfill()
    w = np.asarray(win["atr:14"])
    full = np.asarray(unb["atr:14"])
    _bit_exact(w, full[len(full) - len(w) :], "windowed vs unbounded tail")


# A column cached while the last row is already a FORMING bar (the frame was grown
# period by period, so the newest row is open) must anchor its carried state at the
# last CLOSED bar — the first fold into that row resumes it from the anchor.
_CACHED_ON_FORMING = [
    "ema:14", "smma:14", "atr:14", "natr:14", "rsi:14", "cmo:14", "obv", "adx:14",
    "macd", "rsi:14@close", "cmo:14@close",
]


@pytest.mark.parametrize("directive", _CACHED_ON_FORMING)
def test_cached_while_forming_then_folded_bit_exact(directive):
    n_closed, n_fine = 40, 24
    x = np.arange(n_closed + n_fine, dtype=float)
    close = 100.0 + 0.2 * x + 3.0 * np.sin(1.1 * x)
    cols = dict(open=close - 0.3, high=close + 1.0, low=close - 1.0, close=close, volume=1000.0 + x)
    minutes = np.r_[np.arange(n_closed) * 5, 196 + np.arange(n_fine)]
    ts = (np.datetime64("2026-01-01") + minutes.astype("timedelta64[m]")).astype("datetime64[ns]")
    live = volas.DataFrame({k: v[:1] for k, v in cols.items()}, index=ts[:1], time_frame="5m")
    live.append(volas.DataFrame({k: v[1:n_closed] for k, v in cols.items()}, index=ts[1:n_closed]))
    live[directive]  # cached while row n_closed-1 is the open forming bar
    for i in range(n_closed, n_closed + n_fine):
        live.append(volas.DataFrame({k: v[i : i + 1] for k, v in cols.items()}, index=ts[i : i + 1]))
        live.fulfill()
        ohlcv = {c: np.asarray(live[c]) for c in ("open", "high", "low", "close", "volume")}
        ref = volas.DataFrame(ohlcv, index=np.asarray(live.index))
        _bit_exact(np.asarray(live[directive]), np.asarray(ref[directive]), f"{directive} i={i}")



# A tf frame built straight from closed bars has no forming row: its last row is
# closed and a column cached on it carries state THROUGH that row. The first rollover
# must not re-finalize that row as if it were a forming bar.
@pytest.mark.parametrize("directive", _CACHED_ON_FORMING)
def test_cached_on_constructed_tf_frame_then_rolled_over_bit_exact(directive):
    n_closed, n_fine = 40, 24
    x = np.arange(n_closed + n_fine, dtype=float)
    close = 100.0 + 0.2 * x + 3.0 * np.sin(1.1 * x)
    cols = dict(open=close - 0.3, high=close + 1.0, low=close - 1.0, close=close, volume=1000.0 + x)
    minutes = np.r_[np.arange(n_closed) * 5, 200 + np.arange(n_fine)]
    ts = (np.datetime64("2026-01-01") + minutes.astype("timedelta64[m]")).astype("datetime64[ns]")
    live = volas.DataFrame({k: v[:n_closed] for k, v in cols.items()}, index=ts[:n_closed], time_frame="5m")
    live[directive]  # cached while every row is closed
    for i in range(n_closed, n_closed + n_fine):
        live.append(volas.DataFrame({k: v[i : i + 1] for k, v in cols.items()}, index=ts[i : i + 1]))
        live.fulfill()
        ohlcv = {c: np.asarray(live[c]) for c in ("open", "high", "low", "close", "volume")}
        ref = volas.DataFrame(ohlcv, index=np.asarray(live.index))
        _bit_exact(np.asarray(live[directive]), np.asarray(ref[directive]), f"{directive} i={i}")
