"""volas fulfill tests — incremental indicator refresh after append.

Ported / adapted from stock-pandas's ``test_fulfill.py``. ``df[directive]``
auto-caches the indicator as a real column; after ``append`` the new rows are
stale (NaN); ``fulfill()`` recomputes only the tail (O(lookback)) in place, not
the whole column (O(n)).
"""

import time
import pytest
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
    # to_numpy / iloc read the cached columns directly (no auto-refresh). After an
    # append the cached columns are stale, so a bulk read RAISES (EX-1) instead of
    # returning silent NaN; fulfill() makes it fresh.
    full = volas.read_csv(TENCENT)
    df = volas.read_csv(TENCENT).iloc[:40]
    df['ma:20']                                  # cache
    df = df.append(full.iloc[40:41])
    j = df.columns.index('ma:20')
    with pytest.raises(ValueError):              # stale bulk read is loud, not silent
        df.to_numpy()
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


def test_fulfill_single_bar_resume_ema_smma_explicit_series():
    # The single-bar resume fast path: ema / smma compute one value with no Vec and
    # update state in place; an EXPLICIT @-series directive (ema:14@high) declines the
    # scalar path and resumes via the general route. All bit-identical to the unbounded
    # full recompute, bar by bar.
    n = 80
    s = (100 + 5 * np.sin(np.arange(n) * 0.2))
    data = {'open': s, 'high': s + 1.0, 'low': s - 1.0, 'close': s, 'volume': np.full(n, 1e3)}
    directives = ('ema:14', 'smma:20', 'ema:14@high')
    df = volas.DataFrame({k: v[:40].tolist() for k, v in data.items()})
    for d in directives:
        df[d]
    df.fulfill()
    for i in range(40, n):  # one bar at a time -> the height == valid_rows + 1 fast path
        df = df.append(volas.DataFrame({k: [v[i]] for k, v in data.items()}))
        df.fulfill()
    ref = volas.DataFrame({k: v.tolist() for k, v in data.items()})
    for d in directives:
        exp = np.asarray(ref[d].to_numpy(), float)
        got = np.asarray(df[d].to_numpy(), float)
        m = ~np.isnan(exp)
        np.testing.assert_array_equal(got[m], exp[m])  # strict bit-exactness


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


# The probe path executes on the LIVE frame (incl. the stale cached column) for
# default-series directives — safe ONLY because `execute_refresh` dispatches a
# bare NAME node as a command. Resolving the name as a column would return the
# directive's OWN stale cache (a self-referential no-op that "verifies" against
# itself and splices the stale tail back). `macd` is an all-defaults command whose
# canonical form is the bare name, so `macd:12,26` exercises exactly that node shape.
@pytest.mark.parametrize('directive', ['macd', 'macd:12,26', 'boll', 'bbw', 'obv'])
def test_fulfill_bare_canonical_directive_not_self_referential(directive):
    full = volas.read_csv(TENCENT)
    expected = full[directive].to_numpy()
    df = volas.read_csv(TENCENT)
    head = df.iloc[:-1]
    _ = head[directive]                      # cache over n-1 bars
    head.append(df.iloc[-1])
    head.fulfill()
    got = head[directive].to_numpy()
    # the appended row must be REFRESHED (not the stale NaN placeholder) ...
    assert not np.isnan(got[-1])
    # ... and the whole column must match a batch compute (probe tolerance 1e-9).
    np.testing.assert_allclose(got, expected, rtol=1e-9, equal_nan=True)
