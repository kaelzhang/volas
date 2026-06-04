"""Parity against TA-Lib — the canonical oracle for TA-Lib-aligned indicators.

As indicators are aligned to (or added from) TA-Lib per the alignment proposal,
their 1:1 parity is asserted here against the ``talib`` package (a thin wrapper over
the TA-Lib C library). Skipped automatically where ``talib`` is not installed.
"""

from pathlib import Path

import numpy as np
import pytest

import volas

talib = pytest.importorskip('talib')

TENCENT = str((Path(__file__).parent / 'data' / 'tencent.csv').resolve())


@pytest.fixture(scope='module')
def ohlc():
    df = volas.read_csv(TENCENT)
    return (
        df,
        df['high'].to_numpy(),
        df['low'].to_numpy(),
        df['close'].to_numpy(),
    )


def _parity(got, want):
    np.testing.assert_allclose(
        np.asarray(got, dtype=float),
        np.asarray(want, dtype=float),
        rtol=1e-9, atol=1e-9, equal_nan=True,
    )


def test_tr_matches_talib(ohlc):
    df, h, l, c = ohlc
    _parity(df.exec('tr'), talib.TRANGE(h, l, c))


def test_atr_matches_talib(ohlc):
    df, h, l, c = ohlc
    for p in (14, 20):
        _parity(df.exec(f'atr:{p}'), talib.ATR(h, l, c, p))
