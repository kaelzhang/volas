"""Windowed (bounded rolling-window) DataFrame tests.

A windowed frame (``DataFrame(data, window=M, max_lookback=L)``, where ``L`` is an
int or a list of indicator directives to derive it from) keeps only the last ``M``
rows visible while physically retaining ``M + L`` rows of margin so cached indicators
stay bit-exact across the
periodic front-drop. Every row-facing surface presents the logical ``M`` view; the
hidden margin is never observable, and memory stays bounded under unbounded appends.
"""

import numpy as np
import pytest

from volas import DataFrame


def _ts(n, start=0):
    base = np.datetime64('2020-01-01T00:00:00')
    return (base + (start + np.arange(n)) * np.timedelta64(1, 's')).astype('datetime64[ns]')


def _ohlcv(n, seed=0):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n))
    high = close + np.abs(rng.standard_normal(n))
    low = close - np.abs(rng.standard_normal(n))
    openp = close + rng.standard_normal(n) * 0.1
    vol = rng.integers(1, 1000, n).astype(float)
    return dict(open=openp, high=high, low=low, close=close, volume=vol)


def _full(data, n):
    return DataFrame({k: v[:n] for k, v in data.items()}, index=_ts(n))


def _incremental(data, n0, n, directive):
    """An *unbounded* frame built by the same append+refresh sequence (no window,
    so no compaction) — the reference for "compaction adds zero error"."""
    uf = DataFrame({k: v[:n0] for k, v in data.items()}, index=_ts(n0))
    _feed(uf, data, n0, n, refresh=directive)
    return uf


def _windowed(data, n0, **kw):
    return DataFrame({k: v[:n0] for k, v in data.items()}, index=_ts(n0), **kw)


def _feed(wf, data, n0, n, refresh=None):
    """Append rows ``[n0, n)`` one at a time (the live single-bar path)."""
    for k in range(n0, n):
        one = DataFrame({c: [data[c][k]] for c in data}, index=np.array([_ts(1, start=k)[0]]))
        wf.append(one)
        if refresh is not None:
            _ = wf[refresh]
        else:
            wf.fulfill()


# --------------------------------------------------------------------------- #
# Construction validation                                                      #
# --------------------------------------------------------------------------- #

def test_window_zero_rejected():
    with pytest.raises(ValueError, match="window must be a positive"):
        DataFrame({'a': [1.0, 2.0]}, window=0, max_lookback=1)


def test_max_lookback_without_window_rejected():
    with pytest.raises(ValueError, match="only applies to a windowed frame"):
        DataFrame({'a': [1.0, 2.0]}, max_lookback=3)


def test_max_lookback_list_without_window_rejected():
    with pytest.raises(ValueError, match="only applies to a windowed frame"):
        DataFrame({'a': [1.0, 2.0]}, max_lookback=['ma:3'])


def test_window_without_lookback_rejected():
    with pytest.raises(ValueError, match="needs its lookback bound"):
        DataFrame({'a': [1.0, 2.0]}, window=5)


def test_max_lookback_int_and_list_equivalent():
    # max_lookback=N and max_lookback=[directives whose largest lookback is N] size the
    # retained margin (window + N) identically — the list form just derives the int.
    data = _ohlcv(100)
    by_int = _windowed(data, 100, window=30, max_lookback=14)
    by_list = _windowed(data, 100, window=30, max_lookback=['atr:14'])
    assert by_int._physical_height == by_list._physical_height


# --------------------------------------------------------------------------- #
# Logical-M presentation                                                       #
# --------------------------------------------------------------------------- #

def test_initial_data_bounded_to_capacity():
    # 100 rows, window=5, max_lookback=3 -> capacity 8: only the last 8 retained.
    data = _ohlcv(100)
    wf = _windowed(data, 100, window=5, max_lookback=3)
    assert len(wf) == 5
    assert wf._physical_height == 8


def test_len_shape_index_are_logical():
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    assert len(wf) == 5
    assert wf.shape == (5, 5)
    assert len(wf.index) == 5
    # the visible labels are the last five timestamps
    assert np.array_equal(wf.index, _ts(5, start=55))


def test_column_access_is_logical():
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    full = _full(data, 60)
    assert np.array_equal(wf['close'].to_numpy(), full['close'].to_numpy()[-5:])
    # list projection is also logical-M
    proj = wf[['close', 'high']]
    assert len(proj) == 5
    assert np.array_equal(proj['high'].to_numpy(), full['high'].to_numpy()[-5:])


def test_to_numpy_only_window():
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    full = _full(data, 60)
    arr = wf.to_numpy()
    assert arr.shape == (5, 5)
    assert np.allclose(arr, full.to_numpy()[-5:])


def test_to_csv_only_window():
    # The user-emphasized invariant: to_csv carries exactly M rows, never the margin.
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    csv = wf.to_csv()
    body = csv.strip().splitlines()
    assert len(body) - 1 == 5  # minus the header
    full = _full(data, 60)
    assert wf.to_csv() == full.iloc[-5:].to_csv() == _expected_tail_csv(full)


def _expected_tail_csv(full):
    return full.iloc[-5:].to_csv()


def test_to_pandas_only_window():
    pytest.importorskip("pandas")
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    pdf = wf.to_pandas()
    assert len(pdf) == 5
    full = _full(data, 60)
    assert np.allclose(pdf['close'].to_numpy(), full['close'].to_numpy()[-5:])


def test_repr_and_to_string_only_window():
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    # 5 data rows + header line -> the repr body has the window rows only.
    assert repr(wf) == wf.to_string()
    # The margin timestamps must not appear in the rendering.
    margin_ts = str(_ts(1, start=40)[0])[:19]
    assert margin_ts not in wf.to_string()


def test_head_tail_within_window():
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    full = _full(data, 60)
    assert np.array_equal(wf.head(2)['close'].to_numpy(), full['close'].to_numpy()[-5:-3])
    assert np.array_equal(wf.tail(2)['close'].to_numpy(), full['close'].to_numpy()[-2:])


def test_iloc_iat_loc_at_are_logical():
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    full = _full(data, 60)
    # iloc positional -> last-M
    assert wf.iloc[0]['close'] == full.iloc[-5]['close']
    assert wf.iloc[-1]['close'] == full.iloc[-1]['close']
    assert len(wf.iloc[1:3]) == 2
    # iat scalar
    j = wf.columns.index('close')
    assert wf.iat[0, j] == full.iat[len(full) - 5, j]
    # loc / at by the visible label
    label = _ts(1, start=55)[0]
    assert wf.loc[label]['close'] == full.loc[label]['close']
    assert wf.at[label, 'close'] == full.at[label, 'close']
    # a margin label is not addressable
    with pytest.raises(KeyError):
        _ = wf.loc[_ts(1, start=40)[0]]


def test_reductions_over_window_only():
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    full = _full(data, 60)
    tail = full.iloc[-5:]
    assert np.allclose(wf.mean().to_numpy(), tail.mean().to_numpy())
    assert np.allclose(wf.sum().to_numpy(), tail.sum().to_numpy())
    assert np.allclose(wf.std().to_numpy(), tail.std().to_numpy())
    assert np.allclose(wf.min().to_numpy(), tail.min().to_numpy())
    assert np.allclose(wf.max().to_numpy(), tail.max().to_numpy())


# --------------------------------------------------------------------------- #
# fill_into — zero-allocation feature export                                   #
# --------------------------------------------------------------------------- #

def test_fill_into_matches_to_numpy():
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    cols = ['open', 'high', 'low', 'close', 'volume']
    for dt in (np.float32, np.float64):
        out = np.empty((5, len(cols)), dtype=dt)
        wf.fill_into(out, columns=cols)
        ref = wf[cols].to_numpy(dtype=dt.__name__)
        assert np.array_equal(out, ref)


def test_fill_into_default_columns():
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    out = np.empty((5, 5), dtype=np.float32)
    wf.fill_into(out)
    assert np.allclose(out, wf.to_numpy(dtype='float32'))


def test_fill_into_reuses_buffer_across_appends():
    # The live loop: one preallocated buffer, refilled each round (zero per-bar alloc).
    data = _ohlcv(200, seed=2)
    cols = ['close', 'high']
    wf = _windowed(data, 30, window=30, max_lookback=['atr:14'])
    out = np.empty((30, len(cols)), dtype=np.float32)
    for k in range(30, 200):
        one = DataFrame({c: [data[c][k]] for c in data}, index=np.array([_ts(1, start=k)[0]]))
        wf.append(one)
        _ = wf['atr:14']
        wf.fill_into(out, columns=cols)
    full = _full(data, 200)
    assert np.allclose(out[:, 0], full['close'].to_numpy()[-30:].astype(np.float32))


def test_fill_into_shape_mismatch_raises():
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    with pytest.raises(ValueError, match="does not match"):
        wf.fill_into(np.empty((4, 5), dtype=np.float32))


def test_fill_into_string_column_rejected():
    df = DataFrame({'a': [1.0, 2.0], 's': ['x', 'y']}, window=2, max_lookback=0)
    with pytest.raises(ValueError, match="string column"):
        df.fill_into(np.empty((2, 2), dtype=np.float32))


def test_fill_into_bad_dtype_rejected():
    data = _ohlcv(20)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    with pytest.raises(TypeError, match="float32 or float64"):
        wf.fill_into(np.empty((5, 5), dtype=np.int64))


def test_fill_into_unknown_column_rejected():
    data = _ohlcv(20)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    with pytest.raises(KeyError, match="not found"):
        wf.fill_into(np.empty((5, 1), dtype=np.float32), columns=['nope'])


def test_fill_into_unbounded_frame():
    # fill_into works on a plain (unbounded) frame too — logical view is the whole frame.
    data = _ohlcv(8)
    df = _full(data, 8)
    out = np.empty((8, 1), dtype=np.float64)
    df.fill_into(out, columns=['close'])
    assert np.array_equal(out[:, 0], data['close'])


# --------------------------------------------------------------------------- #
# Bit-exactness across compactions                                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("directive", ['atr:14', 'rsi:14', 'ema:20', 'ma:10', 'macd.macd'])
def test_indicator_bit_exact_across_compactions(directive):
    # The windowing invariant: the periodic compaction (front-drop via slice) adds
    # ZERO error vs the SAME incremental computation without a window. (Comparing to
    # a one-shot batch recompute would fold in the append-resume's own ~1e-14 FP
    # path difference, which windowing does not introduce.)
    data = _ohlcv(300, seed=7)
    wf = _windowed(data, 30, window=30, max_lookback=[directive])
    _feed(wf, data, 30, 300, refresh=directive)
    uf = _incremental(data, 30, 300, directive)
    got = wf[directive].to_numpy()
    want = uf[directive].to_numpy()[-30:]
    assert np.array_equal(np.isnan(got), np.isnan(want))
    m = ~np.isnan(got)
    assert np.array_equal(got[m], want[m]), f"{directive}: {got[m][-1]!r} vs {want[m][-1]!r}"


def test_margin_keeps_indicator_history():
    # The visible window's earliest indicator value must reflect the *margin* history,
    # not restart from the window's own first row (that is the whole point of margin).
    data = _ohlcv(300, seed=3)
    wf = _windowed(data, 30, window=30, max_lookback=['ma:10'])
    _feed(wf, data, 30, 300, refresh='ma:10')
    got = wf['ma:10'].to_numpy()
    # ma:10 needs 9 rows of history; with a 10-row margin every visible row is valid.
    assert not np.isnan(got).any()


def test_tf_fold_fast_path_matches_batch_aggregate():
    # The in-place incremental tf-fold (numeric columns) must be bit-exact with the
    # batch-aggregate path. We build the SAME tf frame two ways: all-numeric (the
    # fast path) vs. with an extra string column that opts the bar out of the fast
    # path into the batch aggregate. The shared OHLCV must match exactly.
    def _mins(k):
        base = np.datetime64('2020-01-01T00:00:00')
        return (base + np.arange(k) * np.timedelta64(1, 'm')).astype('datetime64[ns]')

    n = 400
    data = _ohlcv(n, seed=9)
    ts = _mins(n)

    def build(with_str):
        seed = {k: v[:1] for k, v in data.items()}
        if with_str:
            seed['sym'] = ['X']
        f = DataFrame(seed, index=ts[:1], time_frame='15m')
        for k in range(1, n):
            bar = {c: [data[c][k]] for c in data}
            if with_str:
                bar['sym'] = ['X']
            f.append(DataFrame(bar, index=np.array([ts[k]])))
        return f

    fast = build(False)   # all numeric -> in-place fast fold
    slow = build(True)    # a str column -> batch-aggregate fallback
    cols = ['open', 'high', 'low', 'close', 'volume']
    assert np.array_equal(fast[cols].to_numpy(), slow[cols].to_numpy())
    assert np.array_equal(np.asarray(fast.index), np.asarray(slow.index))


# --------------------------------------------------------------------------- #
# Memory bound                                                                 #
# --------------------------------------------------------------------------- #

def test_memory_bounded_under_unbounded_appends():
    data = _ohlcv(2000, seed=1)
    cap = 30 + 14  # window + atr lookback
    wf = _windowed(data, 30, window=30, max_lookback=['atr:14'])
    peak = wf._physical_height
    for k in range(30, 2000):
        one = DataFrame({c: [data[c][k]] for c in data}, index=np.array([_ts(1, start=k)[0]]))
        wf.append(one)
        _ = wf['atr:14']
        peak = max(peak, wf._physical_height)
    assert len(wf) == 30
    assert peak <= 2 * cap, f"physical height {peak} exceeded 2*capacity {2 * cap}"


def test_ready_property():
    data = _ohlcv(100)
    # window + max_lookback = capacity = 8; not ready until 8 rows accumulate.
    wf = _windowed(data, 3, window=5, max_lookback=3)
    assert not wf.ready
    _feed(wf, data, 3, 8)
    assert wf.ready
    # unbounded frames are always "ready"
    assert _full(data, 10).ready


# --------------------------------------------------------------------------- #
# In-place assignment maps to the logical window                              #
# --------------------------------------------------------------------------- #

def test_iloc_setitem_maps_to_window():
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    j = wf.columns.index('close')
    wf.iloc[0, j] = -1.0
    assert wf.iloc[0]['close'] == -1.0
    assert wf['close'].to_numpy()[0] == -1.0


def test_iat_setitem_maps_to_window():
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    j = wf.columns.index('close')
    wf.iat[-1, j] = -2.0
    assert wf.iat[-1, j] == -2.0


def test_new_column_assignment_is_logical():
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    wf['signal'] = np.arange(5, dtype=float)
    assert np.array_equal(wf['signal'].to_numpy(), np.arange(5, dtype=float))
    # a scalar broadcasts across the visible window
    wf['flag'] = 1.0
    assert np.array_equal(wf['flag'].to_numpy(), np.ones(5))


def test_boolean_mask_assignment_is_logical():
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    wf[wf['close'] > wf['close'].to_numpy()[2]] = 0.0
    # rows where the mask was true are zeroed; the frame still shows M rows.
    assert len(wf) == 5


def test_cell_mask_assignment_is_logical():
    # df[bool_frame] = v on a windowed frame writes only the visible cells in place
    # (the margin is preserved, not rebuilt).
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    before_phys = wf._physical_height
    wf[wf == wf] = -1.0          # an all-True cell mask -> every visible cell set
    assert wf._physical_height == before_phys
    assert len(wf) == 5
    assert np.all(wf.to_numpy() == -1.0)


def test_new_column_wrong_length_rejected():
    data = _ohlcv(60)
    wf = _windowed(data, 10, window=5, max_lookback=3)
    _feed(wf, data, 10, 60)
    with pytest.raises(ValueError, match="window length"):
        wf['signal'] = np.arange(4, dtype=float)   # 4 != window 5


# --------------------------------------------------------------------------- #
# tf + windowed composition                                                   #
# --------------------------------------------------------------------------- #

def test_tf_windowed_compose():
    # A windowed, tf-aware frame: fine (1-minute) bars fold into forming 15-minute
    # periods and the visible window stays bounded.
    def _min(n, start=0):
        base = np.datetime64('2020-01-01T00:00:00')
        return (base + (start + np.arange(n)) * np.timedelta64(1, 'm')).astype('datetime64[ns]')

    n = 1500
    data = _ohlcv(n, seed=5)
    # seed the frame with 15-minute-spaced final bars, then fold 1-minute bars.
    seed_n = 40
    seed_idx = (np.datetime64('2020-01-01T00:00:00')
                + np.arange(seed_n) * np.timedelta64(15, 'm')).astype('datetime64[ns]')
    wf = DataFrame(
        {k: v[:seed_n] for k, v in data.items()},
        index=seed_idx,
        time_frame='15m',
        window=10,
        max_lookback=5,
    )
    start_min = seed_n * 15
    for k in range(seed_n, n):
        ts = _min(1, start=start_min + (k - seed_n))[0]
        one = DataFrame({c: [data[c][k]] for c in data}, index=np.array([ts]))
        wf.append(one)
    assert len(wf) <= 10
    assert wf._physical_height <= 2 * (10 + 5)


# --------------------------------------------------------------------------- #
# Unbounded frames are unaffected (regression)                                #
# --------------------------------------------------------------------------- #

def test_unbounded_frame_unchanged():
    data = _ohlcv(50)
    df = _full(data, 50)
    assert len(df) == 50
    assert df._physical_height == 50
    assert df.shape == (50, 5)
    # window is None -> the full frame is the logical view
    assert np.array_equal(df['close'].to_numpy(), data['close'])
