"""volas.rolling_calc tests.

Ported from stock-pandas's ``test_rolling_calc.py`` and adapted to volas's native
API (decision B): ``rolling_calc`` is a standalone function over array-like input
(not a DataFrame method). Verified against the equivalent ``hhv`` directive.
"""

from pathlib import Path

import numpy as np

import volas

TENCENT = str((Path(__file__).parent / 'data' / 'tencent.csv').resolve())


def test_rolling_calc_backward():
    stock = volas.read_csv(TENCENT)
    hhv = volas.rolling_calc(stock['open'], 5, max)
    expected = stock['hhv:5@open'].to_numpy()
    assert np.array_equal(hhv, expected, equal_nan=True)


def test_rolling_calc_forward():
    stock = volas.read_csv(TENCENT)
    hhv = volas.rolling_calc(stock['open'], 5, max, forward=True)
    expected = stock['hhv:5@open'].to_numpy()
    start = 4
    assert np.array_equal(hhv[:-start], expected[start:], equal_nan=True)


def test_rolling_calc_custom_reducer():
    stock = volas.read_csv(TENCENT)
    # an arbitrary reducer (window range) — the point of rolling_calc
    rng = volas.rolling_calc(stock['open'], 3, lambda w: w.max() - w.min())
    o = stock['open'].to_numpy()
    assert np.isnan(rng[0]) and np.isnan(rng[1])
    assert rng[2] == o[:3].max() - o[:3].min()


def test_rolling_calc_window_larger_than_data():
    out = volas.rolling_calc(np.array([1.0, 2.0]), 5, max)
    assert np.isnan(out).all()
