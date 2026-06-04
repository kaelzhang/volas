"""volas fulfill tests — incremental indicator refresh after append.

Ported / adapted from stock-pandas's ``test_fulfill.py``. ``df[directive]``
auto-caches the indicator as a real column; after ``append`` the new rows are
stale (NaN); ``fulfill()`` recomputes only the tail (O(lookback)) in place, not
the whole column (O(n)).
"""

import time
from pathlib import Path

import numpy as np

import volas

TENCENT = str((Path(__file__).parent / 'data' / 'tencent.csv').resolve())


def test_directive_auto_caches_as_column():
    df = volas.read_csv(TENCENT).iloc[:40]
    assert 'ma:20' not in df.columns
    df['ma:20']
    assert 'ma:20' in df.columns  # materialized into the frame


def test_getitem_auto_refreshes_after_append():
    # df[directive] always returns fresh values (it refreshes its stale tail
    # incrementally on access).
    full = volas.read_csv(TENCENT)
    df = volas.read_csv(TENCENT).iloc[:40]
    df['ma:20']                                  # cache
    df = df.append(full.iloc[40:41])
    expected = volas.read_csv(TENCENT).iloc[:41]['ma:20'].to_numpy()[-1]
    # The O(n) sliding sma is path-dependent: the incremental tail-refresh and a
    # full recompute accumulate the running sum from different offsets, so they
    # agree to float tolerance, not bit-for-bit (same contract as
    # test_fulfill_matches_full_recompute; EWMA indicators were always like this).
    assert np.isclose(df['ma:20'].to_numpy()[-1], expected, rtol=1e-9)


def test_fulfill_refreshes_bulk_read():
    # to_numpy / iloc read the cached columns directly (no auto-refresh), so the
    # appended rows are stale until fulfill() batch-refreshes them.
    full = volas.read_csv(TENCENT)
    df = volas.read_csv(TENCENT).iloc[:40]
    df['ma:20']                                  # cache
    df = df.append(full.iloc[40:41])
    j = df.columns.index('ma:20')
    assert np.isnan(df.to_numpy()[-1, j])        # stale before fulfill
    df.fulfill()                                 # batch tail recompute (in place)
    expected = volas.read_csv(TENCENT).iloc[:41]['ma:20'].to_numpy()[-1]
    # fresh after fulfill, to float tolerance (sliding sma is path-dependent)
    assert np.isclose(df.to_numpy()[-1, j], expected, rtol=1e-9)


def test_fulfill_matches_full_recompute():
    full = volas.read_csv(TENCENT)
    df = volas.read_csv(TENCENT).iloc[:60]
    df['ma:20']
    df = df.append(full.iloc[60:])               # append the remaining bars
    df.fulfill()
    expected = volas.read_csv(TENCENT)['ma:20'].to_numpy()
    np.testing.assert_allclose(df['ma:20'].to_numpy(), expected, equal_nan=True)


def test_fulfill_no_computed_columns_is_noop():
    df = volas.read_csv(TENCENT)
    df.fulfill()  # nothing cached -> harmless
    assert 'ma:20' not in df.columns


def test_fulfill_is_incremental_not_full_recompute():
    # fulfill (O(lookback)) must be dramatically cheaper than a full O(n) recompute.
    n = 100_000
    base = {c: np.arange(n, dtype=float) for c in ['open', 'high', 'low', 'close', 'volume']}
    df = volas.DataFrame(base)
    df['ma:20']
    one = volas.DataFrame({c: [1.0] for c in ['open', 'high', 'low', 'close', 'volume']})
    df = df.append(one)

    t0 = time.perf_counter()
    df.fulfill()
    fulfill_us = (time.perf_counter() - t0) * 1e6

    t0 = time.perf_counter()
    df.exec('ma:30')  # a full O(n) recompute of a fresh directive
    full_us = (time.perf_counter() - t0) * 1e6

    assert fulfill_us * 10 < full_us, (fulfill_us, full_us)
