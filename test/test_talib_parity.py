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


def _parity(got, want, mask_want=False):
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    if mask_want:
        # volas may emit a value where TA-Lib emits NaN (e.g. the MACD line, which
        # volas starts at its natural first-valid row rather than delaying it to the
        # signal's start, per best practice). Compare only where TA-Lib emits.
        m = ~np.isnan(want)
        np.testing.assert_allclose(got[m], want[m], rtol=1e-9, atol=1e-9)
    else:
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9, equal_nan=True)


def test_tr_matches_talib(ohlc):
    df, h, l, c = ohlc
    _parity(df.exec('tr'), talib.TRANGE(h, l, c))


def test_atr_matches_talib(ohlc):
    df, h, l, c = ohlc
    for p in (14, 20):
        _parity(df.exec(f'atr:{p}'), talib.ATR(h, l, c, p))


def test_ema_matches_talib(ohlc):
    df, h, l, c = ohlc
    for p in (12, 20):
        _parity(df.exec(f'ema:{p}'), talib.EMA(c, p))


def test_rsi_matches_talib(ohlc):
    df, h, l, c = ohlc
    for p in (6, 14):
        _parity(df.exec(f'rsi:{p}'), talib.RSI(c, p))


def test_macd_is_clean_ema_difference(ohlc):
    # volas MACD line = EMA(fast) - EMA(slow), the textbook definition, and matches
    # the EMAs TA-Lib produces *standalone*. It intentionally diverges from
    # talib.MACD, which is internally inconsistent with its own EMA (a documented
    # quirk: talib.MACD != talib.EMA(12) - talib.EMA(26)); the clean, self-consistent
    # difference is the better practice (owner: best practice may differ from TA-Lib).
    df, h, l, c = ohlc
    e12, e26 = talib.EMA(c, 12), talib.EMA(c, 26)
    _parity(df.exec('macd'), e12 - e26, mask_want=True)
    # macd.signal = EMA(9) of the (clean) line; macd.histogram = line - signal.
    # Both follow from the verified line + ema_seeded, so are not re-checked against
    # talib's quirk-based MACD signal / hist.


def test_price_transform_matches_talib(ohlc):
    df, h, l, c = ohlc
    o = df['open'].to_numpy()
    _parity(df.exec('avgprice'), talib.AVGPRICE(o, h, l, c))
    _parity(df.exec('medprice'), talib.MEDPRICE(h, l))
    _parity(df.exec('typprice'), talib.TYPPRICE(h, l, c))
    _parity(df.exec('wclprice'), talib.WCLPRICE(h, l, c))


def test_momentum_roc_matches_talib(ohlc):
    df, h, l, c = ohlc
    for p in (10, 14):  # 10 is the shared TA-Lib default
        _parity(df.exec(f'mom:{p}'), talib.MOM(c, p))
        _parity(df.exec(f'roc:{p}'), talib.ROC(c, p))
        _parity(df.exec(f'rocp:{p}'), talib.ROCP(c, p))
        _parity(df.exec(f'rocr:{p}'), talib.ROCR(c, p))
        _parity(df.exec(f'rocr100:{p}'), talib.ROCR100(c, p))
    # Defaults (period 10) resolve identically to the explicit form.
    _parity(df.exec('roc'), talib.ROC(c, 10))
