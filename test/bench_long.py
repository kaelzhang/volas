"""The long history of the derived-frame benchmark rows (``test_benchmark.test_api``'s
``tails`` and ``middle``): a million bars carrying cached indicators (volas), or the same
columns as plain data (pandas / polars). ``tails`` selects its last 100, 1 000 and 5 000
rows — what a live system looks at every bar — and ``middle`` the second quarter of it.
"""

import numpy as np
import pandas as pd

from volas import DataFrame as VolasDataFrame

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

LONG_N = 1_000_000
CACHED = ['ma:20', 'ema:20', 'macd', 'sar', 'boll', 'rsi:14', 'atr:14']
TAILS = (100, 1_000, 5_000)
OPERATIONS = ('tails', 'middle')


def _columns():
    rng = np.random.default_rng(7)
    c = 100 + np.cumsum(rng.normal(0, 1, LONG_N))
    return {'open': c + rng.normal(0, .3, LONG_N), 'high': c + 1 + rng.random(LONG_N),
            'low': c - 1 - rng.random(LONG_N), 'close': c,
            'volume': rng.integers(1, 1000, LONG_N).astype(float)}


def frames():
    """candidate -> the long frame: volas with its indicators cached, the others with
    their values as plain columns."""
    cols = _columns()
    vdf = VolasDataFrame(cols)
    plain = dict(cols)
    for d in CACHED:
        plain[d] = vdf[d].to_numpy()
    out = {'volas': vdf, 'pandas': pd.DataFrame(plain)}
    if pl is not None:
        out['polars'] = pl.DataFrame(plain)
    return out


def ops():
    """candidate -> {``tails``, ``middle``} over its long frame."""
    long = frames()
    mid = slice(LONG_N // 4, LONG_N // 2)
    out = {c: {'tails': (lambda f=f: [f.iloc[-k:] for k in TAILS]), 'middle': (lambda f=f: f.iloc[mid])}
           for c, f in long.items() if c != 'polars'}
    if 'polars' in long:
        f = long['polars']
        out['polars'] = {'tails': lambda: [f[-k:] for k in TAILS], 'middle': lambda: f[mid]}
    return out
