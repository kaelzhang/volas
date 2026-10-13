"""A cell write costs O(cells written), never O(rows).

A single-cell write (``iat`` / ``at`` / ``iloc`` / ``loc``) lands in place on the
column's buffer: no whole-column copy, no whole-index copy, no frame-wide pass —
on a plain frame, a time-frame frame (closed rows and the forming row) and a
bounded window alike. A live loop that corrects a cell per bar must not slow down
as the frame grows.
"""

import time

import numpy as np
import pytest

from scripts.perf_costs import timing_rounds

import volas

COLS = ['open', 'high', 'low', 'close', 'volume']


def _frame(kind, n):
    t = np.datetime64('2024-01-01', 'ns') + np.arange(n) * np.timedelta64(1, 'm')
    data = {c: np.arange(n, dtype=float) + 1.0 for c in COLS}
    df = volas.DataFrame({'t': t, **data}).set_index('t')
    if kind == 'tf':
        return volas.DataFrame(df, time_frame='1m')
    if kind == 'windowed':
        return volas.DataFrame(df, time_frame='1m', window=n - 10, max_lookback=10)
    return df


def _writer(df, how, row):
    i = len(df) - 1 if row == 'last' else len(df) // 2
    label = df.index[i]
    j = COLS.index('volume')
    return {
        'iat': lambda x: df.iat.__setitem__((i, j), x),
        'at': lambda x: df.at.__setitem__((label, 'volume'), x),
        'iloc': lambda x: df.iloc.__setitem__((i, j), x),
        'loc': lambda x: df.loc.__setitem__((label, 'volume'), x),
    }[how]


def _per_write_us(kind, how, row, n):
    write = _writer(_frame(kind, n), how, row)
    for k in range(50):                          # warm: the first write may own the buffer
        write(float(k))
    best = float('inf')
    for _ in range(timing_rounds(3)):                           # best of 3 damps scheduler noise
        t = time.perf_counter()
        for k in range(500):
            write(float(k) + 0.5)
        best = min(best, (time.perf_counter() - t) / 500 * 1e6)
    return best


def _hole_fill_us(n):
    # an int column with a hole elsewhere: each write opens a hole and fills it again,
    # so its validity bitmap stays materialized and keeps its missing count
    df = volas.DataFrame({'q': np.arange(n)})
    df.iat[0, 0] = None
    i = n // 2
    best = float('inf')
    for _ in range(timing_rounds(3)):
        t = time.perf_counter()
        for k in range(300):
            df.iat[i, 0] = None
            df.iat[i, 0] = k
        best = min(best, (time.perf_counter() - t) / 600 * 1e6)
    assert df.iat[0, 0] is volas.NA and df.iat[i, 0] == 299
    return best


def test_int_hole_fill_does_not_scale_with_rows(cost_guard):
    small, large = cost_guard.pair(lambda: _hole_fill_us(2_000), lambda: _hole_fill_us(10_000_000))
    cost_guard(large, small * 5)     # O(1) missing count, no popcount


@pytest.mark.parametrize('row', ['middle', 'last'])
@pytest.mark.parametrize('how', ['iat', 'at', 'iloc', 'loc'])
@pytest.mark.parametrize('kind', ['plain', 'tf', 'windowed'])
def test_cell_write_does_not_scale_with_rows(kind, how, row, cost_guard):
    small, large = cost_guard.pair(lambda: _per_write_us(kind, how, row, 2_000),
                                  lambda: _per_write_us(kind, how, row, 200_000))
    cost_guard(large, small * 5)     # O(cells written), not ~100x


@pytest.mark.parametrize('how', ['iat', 'at', 'iloc', 'loc'])
@pytest.mark.parametrize('kind', ['tf', 'windowed'])
def test_a_closed_row_tf_write_costs_about_a_plain_one(kind, how, cost_guard):
    # a write that reaches neither the forming row nor a cached column cannot move the
    # open period: it costs what the same write costs on a plain frame
    plain, tf = cost_guard.pair(lambda: _per_write_us('plain', how, 'middle', 200_000),
                               lambda: _per_write_us(kind, how, 'middle', 200_000))
    cost_guard(tf, plain * 2)
