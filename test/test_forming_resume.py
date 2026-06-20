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


def _load(n):
    csv = pd.read_csv("test/data/btcusdt_1m_20k.csv")
    cols = {c: csv[c].to_numpy(float)[:n] for c in ("open", "high", "low", "close", "volume")}
    ts = (csv["open_time"].to_numpy()[:n] * 1_000_000).astype("datetime64[ns]")
    return cols, ts


def _bit_exact(live, ref, ctx):
    """Equal where both finite, NaN in the same cells — same kernel ⇒ no tolerance."""
    assert live.shape == ref.shape, f"{ctx}: shape {live.shape} != {ref.shape}"
    both_nan = np.isnan(live) & np.isnan(ref)
    a = np.where(both_nan, 0.0, live)
    b = np.where(both_nan, 0.0, ref)
    if not np.array_equal(a, b):
        bad = np.argmax(np.abs(a - b))
        raise AssertionError(
            f"{ctx}: diverged at row {bad}: live={live[bad]!r} ref={ref[bad]!r} "
            f"max|Δ|={np.nanmax(np.abs(a - b))!r}"
        )


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
