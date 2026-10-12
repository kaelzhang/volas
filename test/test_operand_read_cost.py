"""What reading a command's inputs costs on the live path.

A live refresh continues a cached column over the new row from a trailing window of
the frame: a non-float64 input (int64 volume, float32 prices) is converted over that
window only, and an `@` expression operand is evaluated over it alone — so one appended
row costs the same at a million rows as at ten thousand. A composite's state capture
converts its input once, not once per row. Each test compares the cost at two heights:
a cost that grows with the frame's height (a whole-column conversion or evaluation per
row) shows as a ratio near the heights' ratio, far past the bound.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import volas

SMALL, LARGE = 10_000, 1_000_000


def _data(n, kind):
    rng = np.random.default_rng(1)
    m = n + 32
    c = 100 + np.cumsum(rng.standard_normal(m))
    o = c + 0.3 * rng.standard_normal(m)
    data = {'open': o, 'high': np.maximum(o, c) + 0.5, 'low': np.minimum(o, c) - 0.5, 'close': c,
            'volume': rng.uniform(10, 1000, m)}
    if kind == 'int64':
        data['volume'] = rng.integers(10, 1000, m).astype(np.int64)
    elif kind == 'float32':
        data = {k: v.astype(np.float32) for k, v in data.items()}
    return data


def _append_cost(directive, kind, n, k=24):
    """The cheapest of `k` single-row append + refresh rounds of a cached `directive`
    at height `n`, and the refreshed column's last value against a full recompute."""
    data = _data(n, kind)
    df = volas.DataFrame({c: v[:n] for c, v in data.items()})
    df[directive]
    bars = [volas.DataFrame({c: v[i:i + 1] for c, v in data.items()}) for i in range(n, n + k + 1)]
    df.append(bars[0])
    df.fulfill()
    best = float('inf')
    for bar in bars[1:]:
        t0 = time.perf_counter()
        df.append(bar)
        df.fulfill()
        best = min(best, time.perf_counter() - t0)
    return best, df


CASES = [
    ('obv', 'int64'), ('mfi:14', 'int64'), ('vwma:20', 'int64'),
    ('ema:20', 'float32'), ('atr:14', 'float32'), ('ma:20', 'float32'),
    ('ema:14@(close+open)', 'float64'), ('rsi:14@(ma:5)', 'float64'), ('ma:5@(close*volume)', 'float64'),
]


@pytest.mark.parametrize('directive, kind', CASES, ids=[f'{d}-{k}' for d, k in CASES])
def test_a_live_refresh_costs_the_same_at_a_million_rows(directive, kind):
    small, _ = _append_cost(directive, kind, SMALL)
    large, df = _append_cost(directive, kind, LARGE)
    # a whole-column conversion / evaluation per row would be ~100x here
    assert large < 8 * small + 20e-6, (directive, kind, small, large)
    full = df.exec(directive, dtype='float64')[-3:]
    got = np.asarray(df[directive], dtype=float)[-3:]
    assert ((got == full) | (np.isnan(got) & np.isnan(full))).all()


def test_a_composite_captures_its_state_from_one_conversion():
    """A composite's state is stepped over every row once; its float32 input is
    converted once, not once per row (which made the capture quadratic)."""
    def first(n):
        data = _data(n, 'float32')
        best = float('inf')
        for _ in range(3):
            df = volas.DataFrame({c: v[:n] for c, v in data.items()})
            t0 = time.perf_counter()
            df['apo:12,26,0']
            best = min(best, time.perf_counter() - t0)
        return best
    # linear: ~5x; quadratic: ~25x
    assert first(100_000) < 12 * first(20_000)
