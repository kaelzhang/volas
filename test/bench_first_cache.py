"""Builders for the first-materialization benchmark rows (``test_benchmark.test_first_cache``):
the first ``df[directive]`` on a fresh frame — the column's computation together with
the states a live refresh continues from — at heights where a second pass over the
rows, or a copy of them, shows.

Each round builds a fresh frame (the setup, not timed) and times the one read that
caches the column. volas only; not a report section — the perf gate reads these rows.
"""

import numpy as np

from volas import DataFrame as VolasDataFrame

FIRST_CACHE_INDICATORS = ['kcw', 'ma:20,1', 'ma:20,3', 'ma:20,6', 'ma:20,8', 'ema:20', 'atr:14', 'ma:20']
FIRST_CACHE_LENGTHS = (100_000, 1_000_000)
FIRST_CACHE_ROUNDS = 20
# the rows' ids: ``<directive>@n=<rows>``
FIRST_CACHE_IDS = [f'{d}@n={n}' for d in FIRST_CACHE_INDICATORS for n in FIRST_CACHE_LENGTHS]


def _ohlcv(n, seed=20261013):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    open_ = close + rng.normal(0.0, 0.2, n)
    return {
        'open': open_,
        'high': np.maximum(open_, close) + rng.random(n),
        'low': np.minimum(open_, close) - rng.random(n),
        'close': close,
        'volume': np.abs(rng.normal(1.0e6, 2.0e5, n)) + 1.0,
    }


_DATA = {}


def build(indicator):
    """``(run, setup)`` for one first materialization of the row ``indicator``
    (``<directive>@n=<rows>``)."""
    directive, rows = indicator.split('@n=')
    n = int(rows)
    data = _DATA.get(n)
    if data is None:
        data = _DATA[n] = _ohlcv(n)

    def setup():
        return (VolasDataFrame(data),), {}

    def run(df):
        _ = df[directive]

    return run, setup
