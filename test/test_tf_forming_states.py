"""The tf-aware frame's forming period: the open period's fine bars are kept in a
small frame whose cumulation REPLACES the frame's last row on every append.

- The constructor's last seed row is the first entry of that small frame, so a finer
  bar in its period folds into it (never a duplicate period row).
- A bar with the same timestamp as one already in the small frame replaces it — a
  re-sent fine bar, or a snapshot of the frame's own time frame stamped at the
  period start (an exchange kline push re-sent while it forms).
- A recursive cached column carries two states on a tf frame — after the last row
  and before it — so the forming refresh recomputes the last row from the "before"
  state and a rollover continues without re-stepping the closed bar. Every path must
  stay bit-exact versus a from-scratch full recompute over the frame's OHLCV.
"""

import numpy as np
import pandas as pd
import pytest

import volas

from .test_window_locality import RECURSIVE, SUBCOMMANDS

OHLCV = ("open", "high", "low", "close", "volume")
T0 = np.datetime64("2026-01-01T00:00")


def _ts(minutes):
    return (T0 + np.asarray(minutes).astype("timedelta64[m]")).astype("datetime64[ns]")


def _bars(minutes, seed=0):
    """Deterministic OHLCV bars at the given minute offsets."""
    x = np.asarray(minutes, dtype=float) + seed
    close = 100.0 + 0.2 * x + 3.0 * np.sin(1.1 * x)
    return dict(
        open=close - 0.3,
        high=close + 1.0 + 0.5 * np.cos(0.7 * x) ** 2,
        low=close - 1.0 - 0.5 * np.sin(0.3 * x) ** 2,
        close=close,
        volume=1000.0 + 37.0 * np.abs(np.sin(x)),
    )


def _frame(cols, minutes, **kw):
    return volas.DataFrame(cols, index=_ts(minutes), **kw)


def _bar(cols, i, minute):
    return volas.DataFrame({k: v[i : i + 1] for k, v in cols.items()}, index=_ts([minute]))


def _bit_exact(live, ref, ctx):
    live = np.asarray(live, dtype=float)
    ref = np.asarray(ref, dtype=float)
    assert live.shape == ref.shape, f"{ctx}: shape {live.shape} != {ref.shape}"
    both_nan = np.isnan(live) & np.isnan(ref)
    a = np.where(both_nan, 0.0, live)
    b = np.where(both_nan, 0.0, ref)
    if not np.array_equal(a, b):
        bad = int(np.argmax((a != b) & ~(np.isnan(a) & np.isnan(b))))
        raise AssertionError(f"{ctx}: diverged at row {bad}: live={live[bad]!r} ref={ref[bad]!r}")


def _reference(live):
    ohlcv = {c: np.asarray(live[c]) for c in OHLCV}
    return volas.DataFrame(ohlcv, index=np.asarray(live.index))


def _check(live, directives, ctx):
    ref = _reference(live)
    for d in directives:
        _bit_exact(live[d], ref[d], f"{d} {ctx}")


# --- the constructor's last seed row opens the forming period ----------------


def test_constructed_seed_last_row_folds_finer_bars():
    seed = {
        "open": [10.0, 20.0], "high": [12.0, 22.0], "low": [9.0, 19.0],
        "close": [11.0, 21.0], "volume": [100.0, 200.0],
    }
    df = _frame(seed, [16 * 60 + 45, 17 * 60], time_frame="15m")
    df.append(volas.DataFrame(
        {"open": [21.5], "high": [23.0], "low": [20.5], "close": [22.5], "volume": [7.0]},
        index=_ts([17 * 60 + 9]),
    ))
    df.append(volas.DataFrame(
        {"open": [22.5], "high": [22.8], "low": [18.0], "close": [19.5], "volume": [5.0]},
        index=_ts([17 * 60 + 10]),
    ))
    assert len(df) == 2
    assert list(np.asarray(df.index)) == list(_ts([16 * 60 + 45, 17 * 60]))
    last = df.iloc[-1].to_dict()
    assert last == {"open": 20.0, "high": 23.0, "low": 18.0, "close": 19.5, "volume": 212.0}
    # A re-sent 17:10 bar replaces the earlier 17:10, it is not added on top.
    df.append(volas.DataFrame(
        {"open": [22.5], "high": [22.9], "low": [21.0], "close": [21.7], "volume": [6.0]},
        index=_ts([17 * 60 + 10]),
    ))
    assert len(df) == 2
    last = df.iloc[-1].to_dict()
    assert last == {"open": 20.0, "high": 23.0, "low": 19.0, "close": 21.7, "volume": 213.0}
    # The first row is untouched.
    assert df.iloc[0].to_dict() == {
        "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "volume": 100.0,
    }


def test_same_time_frame_snapshot_replaces_the_last_row_whole():
    seed = {k: [1.0, 2.0] for k in OHLCV}
    df = _frame(seed, [16 * 60 + 30, 16 * 60 + 45], time_frame="15m")
    snaps = [
        {"open": 5.0, "high": 6.0, "low": 4.0, "close": 5.5, "volume": 10.0},
        {"open": 5.0, "high": 7.0, "low": 3.5, "close": 6.5, "volume": 30.0},
        {"open": 5.0, "high": 7.2, "low": 3.0, "close": 4.5, "volume": 45.0},
    ]
    for s in snaps:
        df.append(volas.DataFrame({k: [v] for k, v in s.items()}, index=_ts([17 * 60])))
    assert len(df) == 3
    assert df.iloc[-1].to_dict() == snaps[-1]
    # A snapshot stamped at the seed's own last period start replaces that row too.
    df2 = _frame(seed, [16 * 60 + 30, 16 * 60 + 45], time_frame="15m")
    df2.append(volas.DataFrame({k: [v] for k, v in snaps[1].items()}, index=_ts([16 * 60 + 45])))
    assert len(df2) == 2
    assert df2.iloc[-1].to_dict() == snaps[1]


# --- the reporter scenario ----------------------------------------------------

_REPORTER = ["rsi:14", "natr:14", "cmo:14", "rsi:14@close", "cmo:14@close"]


def test_reporter_seeded_5m_frame_folds_1m_bars_bit_exact():
    n_seed, n_fine = 40, 24
    seed_minutes = np.arange(n_seed) * 5  # 0, 5, ..., 195 (period starts)
    fine_minutes = 196 + np.arange(n_fine)  # 196, 197, ...
    seed = _bars(seed_minutes)
    fine = _bars(fine_minutes, seed=0.37)
    live = _frame(seed, seed_minutes, time_frame="5m")
    for d in _REPORTER:
        live[d]
    for i, m in enumerate(fine_minutes):
        live.append(_bar(fine, i, m))
        live.fulfill()
        want_rows = n_seed + (m - 195) // 5
        assert len(live) == want_rows, f"minute {m}"
        _check(live, _REPORTER, f"minute={m}")


# --- every recursive class, across the ways a forming row is produced ---------

_RECURSIVE_SUBS = sorted(k for k in SUBCOMMANDS if k.split(".", 1)[0] in RECURSIVE)
_ALL_RECURSIVE = sorted(set(RECURSIVE.values()) | {SUBCOMMANDS[k] for k in _RECURSIVE_SUBS})
# The recursions a bounded window keeps exactly: every one carrying a state (the
# others — a command that recomputes — cannot be declared on a windowed frame).
_UNKEPT = {"crsi", "kcw", "stoch_momentum", "stoch_momentum.signal"}
_WINDOWED = sorted(set(_ALL_RECURSIVE) - _UNKEPT)


def _closed_15m(minutes_1m, fine):
    """Aggregate 1m bars into 15m bars labelled at the period start (pandas oracle)."""
    pdf = pd.DataFrame(fine, index=pd.DatetimeIndex(_ts(minutes_1m)))
    agg = pdf.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return {c: agg[c].to_numpy(float) for c in OHLCV}, agg.index.to_numpy().astype("datetime64[ns]")


_N_1M = 15 * 70 + 7  # 70 whole 15m periods + 7 minutes of an open one
_HIST = _bars(np.arange(_N_1M + 15 * 6))


def _hist(lo, hi):
    return {k: v[lo:hi] for k, v in _HIST.items()}


def _feed(live, lo, hi, directives, cache_at=None, every=15):
    for i in range(lo, hi):
        live.append(volas.DataFrame(_hist(i, i + 1), index=_ts([i])))
        if cache_at is not None and i == cache_at:
            for d in directives:
                live[d]
        live.fulfill()
        if (i - lo) % every == every - 1 or i == hi - 1:
            _check(live, directives, f"1m={i}")


@pytest.mark.parametrize("directive", _ALL_RECURSIVE)
def test_constructed_seed_then_rollovers_bit_exact(directive):
    seed_cols, seed_idx = _closed_15m(np.arange(_N_1M), _hist(0, _N_1M))
    live = volas.DataFrame(seed_cols, index=seed_idx, time_frame="15m")
    live[directive]  # cached before the first append (the last row is forming)
    _feed(live, _N_1M, _N_1M + 15 * 5, [directive])


@pytest.mark.parametrize("directive", _ALL_RECURSIVE)
def test_cached_mid_period_bit_exact(directive):
    seed_cols, seed_idx = _closed_15m(np.arange(_N_1M), _hist(0, _N_1M))
    live = volas.DataFrame(seed_cols, index=seed_idx, time_frame="15m")
    _feed(live, _N_1M, _N_1M + 15 * 5, [directive], cache_at=_N_1M + 3)


@pytest.mark.parametrize("directive", _ALL_RECURSIVE)
def test_cumulate_built_frame_bit_exact(directive):
    hist = volas.DataFrame(_hist(0, _N_1M), index=_ts(np.arange(_N_1M)))
    live = hist.cumulate("15m")
    live[directive]
    _feed(live, _N_1M, _N_1M + 15 * 5, [directive])


@pytest.mark.parametrize("directive", _ALL_RECURSIVE)
def test_same_time_frame_snapshots_bit_exact(directive):
    # A 15m frame fed 15m snapshots stamped at the period start, re-sent while the
    # period forms (each snapshot the running aggregate so far), then the next period.
    seed_cols, seed_idx = _closed_15m(np.arange(15 * 70), _hist(0, 15 * 70))
    live = volas.DataFrame(seed_cols, index=seed_idx, time_frame="15m")
    live[directive]
    for p in range(70, 75):
        start = 15 * p
        for k in (1, 4, 9, 15):
            snap_cols, snap_idx = _closed_15m(np.arange(start, start + k), _hist(start, start + k))
            live.append(volas.DataFrame(snap_cols, index=snap_idx))
            live.fulfill()
            assert len(live) == p + 1
            _check(live, [directive], f"period={p} k={k}")


@pytest.mark.parametrize("directive", _WINDOWED)
def test_windowed_tf_frame_matches_unbounded(directive):
    # Both frames start from the same short seed (inside the window's retention), then
    # take 5m bars into a 15m frame until the windowed one has compacted many times.
    n_seed, n_periods = 3, 240  # the seed fits inside every window's retention
    seed_cols, seed_idx = _closed_15m(np.arange(15 * n_seed), _hist(0, 15 * n_seed))
    unb = volas.DataFrame(seed_cols, index=seed_idx, time_frame="15m")
    win = volas.DataFrame(
        seed_cols, index=seed_idx, time_frame="15m", window=4, max_lookback=[directive]
    )
    unb[directive]
    win[directive]
    bars = _bars(np.arange(3 * n_periods), seed=0.61)
    for j in range(3 * n_periods):
        minute = 15 * (n_seed - 1) + 5 * j
        bar = volas.DataFrame({k: v[j : j + 1] for k, v in bars.items()}, index=_ts([minute]))
        unb.append(bar)
        unb.fulfill()
        win.append(bar)
        win.fulfill()
    w = np.asarray(win[directive], dtype=float)
    full = np.asarray(unb[directive], dtype=float)
    _bit_exact(w, full[len(full) - len(w):], f"{directive} windowed vs unbounded tail")
    _check(unb, [directive], "unbounded end")


# --- edge paths ------------------------------------------------------------------


def test_replaced_bar_with_a_non_foldable_column_recomputes_exactly():
    # A bool column cannot be replaced in place, so the re-cumulated forming row is
    # written positionally and the cached columns recompute from scratch — still exact.
    cols = _bars(np.arange(30) * 15)
    cols["flag"] = np.arange(30) % 2 == 0
    live = _frame(cols, np.arange(30) * 15, time_frame="15m")
    live["ema:5"]
    live["rsi:5"]
    for k, close in enumerate([131.0, 129.5, 133.25]):
        bar = {c: [v] for c, v in zip(OHLCV, (130.0, 134.0, 128.0, close, 9.0))}
        bar["flag"] = [k % 2 == 1]
        live.append(volas.DataFrame(bar, index=_ts([30 * 15])))  # re-sent each time
        live.fulfill()
        assert len(live) == 31
        assert live.iloc[-1]["close"] == close
        assert bool(live.iloc[-1]["flag"]) is (k % 2 == 1)
        _check(live, ["ema:5", "rsi:5"], f"resend {k}")


def test_cached_column_carried_into_a_time_frame_frame():
    minutes = np.arange(40) * 5
    plain = _frame(_bars(minutes), minutes)
    plain["rsi:14"]
    plain["ema:9"]
    live = volas.DataFrame(plain, time_frame="5m")
    assert live.is_computed("rsi:14")
    fine = _bars(196 + np.arange(12), seed=0.5)
    for i, m in enumerate(196 + np.arange(12)):
        live.append(_bar(fine, i, m))
        live.fulfill()
        _check(live, ["rsi:14", "ema:9"], f"minute={m}")


def test_empty_time_frame_frame_with_a_cached_column():
    empty = {c: np.array([], dtype=float) for c in OHLCV}
    live = volas.DataFrame(empty, index=_ts([]), time_frame="15m")
    assert len(live) == 0
    directives = ["ema:3", "ma:3", "obv", "ad", "pvt", "avgprice"]
    for d in directives:
        live[d]
    bars = _bars(np.arange(40) * 5)
    for i in range(40):
        live.append(_bar(bars, i, 5 * i))
        live.fulfill()
        _check(live, directives, f"bar={i}")
    assert len(live) == 14


@pytest.mark.parametrize("directive", sorted(_UNKEPT & set(_ALL_RECURSIVE)))
def test_a_windowed_tf_frame_refuses_a_recursion_it_cannot_keep(directive):
    seed_cols, seed_idx = _closed_15m(np.arange(45), _hist(0, 45))
    with pytest.raises(ValueError, match=r"cannot keep .* exact on a windowed frame"):
        volas.DataFrame(seed_cols, index=seed_idx, time_frame="15m", window=4, max_lookback=[directive])
