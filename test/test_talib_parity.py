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


def test_overlap_ma_variants_match_talib(ohlc):
    df, h, l, c = ohlc
    for p in (10, 30):  # 30 is the shared TA-Lib default
        _parity(df.exec(f'wma:{p}'), talib.WMA(c, p))
        _parity(df.exec(f'dema:{p}'), talib.DEMA(c, p))
        _parity(df.exec(f'tema:{p}'), talib.TEMA(c, p))
    for p in (9, 10, 30):  # exercise both odd and even periods (double-SMA split)
        _parity(df.exec(f'trima:{p}'), talib.TRIMA(c, p))
    for p in (5, 10):
        _parity(df.exec(f't3:{p}'), talib.T3(c, p))  # vfactor defaults to 0.7
        _parity(df.exec(f't3:{p},0.5'), talib.T3(c, p, vfactor=0.5))
    for p in (10, 30):  # 30 is the TA-Lib default
        _parity(df.exec(f'kama:{p}'), talib.KAMA(c, p))
    _parity(df.exec('wma'), talib.WMA(c, 30))  # default resolves to 30
    _parity(df.exec('trima'), talib.TRIMA(c, 30))
    _parity(df.exec('t3'), talib.T3(c, 5))
    _parity(df.exec('kama'), talib.KAMA(c, 30))


def test_linear_regression_matches_talib(ohlc):
    df, h, l, c = ohlc
    for p in (9, 14):  # 14 is the shared TA-Lib default
        _parity(df.exec(f'linearreg:{p}'), talib.LINEARREG(c, p))
        _parity(df.exec(f'linearreg_slope:{p}'), talib.LINEARREG_SLOPE(c, p))
        _parity(df.exec(f'linearreg_intercept:{p}'), talib.LINEARREG_INTERCEPT(c, p))
        _parity(df.exec(f'linearreg_angle:{p}'), talib.LINEARREG_ANGLE(c, p))
        _parity(df.exec(f'tsf:{p}'), talib.TSF(c, p))
    _parity(df.exec('tsf'), talib.TSF(c, 14))  # default resolves to 14


def test_volume_matches_talib(ohlc):
    df, h, l, c = ohlc
    v = np.asarray(df['volume'].to_numpy(), dtype=float)  # talib requires double
    _parity(df.exec('obv'), talib.OBV(c, v))
    _parity(df.exec('ad'), talib.AD(h, l, c, v))
    for fast, slow in ((3, 10), (5, 20)):
        _parity(
            df.exec(f'adosc:{fast},{slow}'),
            talib.ADOSC(h, l, c, v, fastperiod=fast, slowperiod=slow),
        )
    _parity(df.exec('adosc'), talib.ADOSC(h, l, c, v))  # defaults 3, 10


def test_variance_stddev_matches_talib(ohlc):
    df, h, l, c = ohlc
    for p in (5, 20):  # 5 is the shared TA-Lib default
        _parity(df.exec(f'var:{p}'), talib.VAR(c, p))
        _parity(df.exec(f'stddev:{p}'), talib.STDDEV(c, p))  # nbdev defaults to 1
        _parity(df.exec(f'stddev:{p},2'), talib.STDDEV(c, p, nbdev=2.0))
    _parity(df.exec('var'), talib.VAR(c, 5))  # default resolves to 5


def test_math_operators_match_talib(ohlc):
    df, h, l, c = ohlc

    def idx_parity(got, want, start):
        # MAXINDEX/MININDEX/MINMAXINDEX emit integer indices; TA-Lib fills their
        # warm-up with 0, whereas volas uses NaN — its uniform warm-up convention,
        # and 0 is a valid index, so emitting it would be ambiguous. Assert the NaN
        # warm-up, then compare the valid region (the index values themselves match).
        got = np.asarray(got, dtype=float)
        assert np.all(np.isnan(got[:start])), 'index outputs warm up with NaN'
        np.testing.assert_allclose(got[start:], np.asarray(want, dtype=float)[start:],
                                   rtol=1e-9, atol=1e-9)

    for p in (10, 30):  # 30 is the shared TA-Lib default
        _parity(df.exec(f'sum:{p}'), talib.SUM(c, p))
        idx_parity(df.exec(f'maxindex:{p}'), talib.MAXINDEX(c, p), p - 1)
        idx_parity(df.exec(f'minindex:{p}'), talib.MININDEX(c, p), p - 1)
        mn, mx = talib.MINMAX(c, p)  # value outputs: NaN warm-up, matches volas
        _parity(df.exec(f'minmax.min:{p}'), mn)
        _parity(df.exec(f'minmax.max:{p}'), mx)
        mni, mxi = talib.MINMAXINDEX(c, p)
        idx_parity(df.exec(f'minmaxindex.min:{p}'), mni, p - 1)
        idx_parity(df.exec(f'minmaxindex.max:{p}'), mxi, p - 1)
    _parity(df.exec('sum'), talib.SUM(c, 30))  # default resolves to 30


def test_aroon_matches_talib(ohlc):
    df, h, l, c = ohlc
    for p in (14, 25):  # 14 is the TA-Lib default
        down, up = talib.AROON(h, l, p)  # talib returns (down, up)
        _parity(df.exec(f'aroon.up:{p}'), up)
        _parity(df.exec(f'aroon.down:{p}'), down)
        _parity(df.exec(f'aroonosc:{p}'), talib.AROONOSC(h, l, p))
    # default resolves to 14, and the .u/.d abbreviations match the full names
    _parity(df.exec('aroon.u'), talib.AROON(h, l, 14)[1])
    _parity(df.exec('aroon.d'), talib.AROON(h, l, 14)[0])


def test_ma_matype_apo_ppo_match_talib(ohlc):
    df, h, l, c = ohlc
    # MA across every supported type: 0 SMA, 1 EMA, 2 WMA, 3 DEMA, 4 TEMA, 5 TRIMA,
    # 6 KAMA, 8 T3 (period 10 keeps the heaviest warm-ups within the dataset).
    for mt in (0, 1, 2, 3, 4, 5, 6, 8):
        _parity(df.exec(f'ma:10,{mt}'), talib.MA(c, 10, mt))
    _parity(df.exec('ma:20'), talib.MA(c, 20, 0))  # default matype 0 = SMA
    with pytest.raises(Exception):
        df.exec('ma:10,7')  # MAMA (matype 7) is not yet implemented
    # Price oscillators across MA types
    for mt in (0, 1, 5):
        _parity(df.exec(f'apo:12,26,{mt}'), talib.APO(c, 12, 26, mt))
        _parity(df.exec(f'ppo:12,26,{mt}'), talib.PPO(c, 12, 26, mt))
    _parity(df.exec('apo'), talib.APO(c, 12, 26, 0))  # defaults 12/26/SMA
    _parity(df.exec('ppo'), talib.PPO(c, 12, 26, 0))


def test_accbands_matches_talib(ohlc):
    df, h, l, c = ohlc
    for p in (10, 20):  # 20 is the TA-Lib default
        up, mid, low = talib.ACCBANDS(h, l, c, p)  # talib returns (upper, middle, lower)
        _parity(df.exec(f'accbands.upper:{p}'), up)
        _parity(df.exec(f'accbands:{p}'), mid)  # bare = middle band
        _parity(df.exec(f'accbands.lower:{p}'), low)
    up, mid, low = talib.ACCBANDS(h, l, c, 20)
    _parity(df.exec('accbands.u'), up)  # abbreviations
    _parity(df.exec('accbands.m'), mid)
    _parity(df.exec('accbands.l'), low)
    # the .m abbreviation also resolves the Bollinger middle band
    np.testing.assert_allclose(np.asarray(df.exec('boll.m'), dtype=float),
                               np.asarray(df.exec('boll'), dtype=float), equal_nan=True)


def test_cci_trix_match_talib(ohlc):
    df, h, l, c = ohlc
    v = np.asarray(df['volume'].to_numpy(), dtype=float)
    for p in (14, 20):
        _parity(df.exec(f'cci:{p}'), talib.CCI(h, l, c, p))
        _parity(df.exec(f'mfi:{p}'), talib.MFI(h, l, c, v, p))
    for p in (15, 30):  # 30 is the TA-Lib default
        _parity(df.exec(f'trix:{p}'), talib.TRIX(c, p))
    _parity(df.exec('cci'), talib.CCI(h, l, c, 14))  # default resolves to 14
    _parity(df.exec('trix'), talib.TRIX(c, 30))


def test_bop_cmo_natr_match_talib(ohlc):
    df, h, l, c = ohlc
    o = df['open'].to_numpy()
    _parity(df.exec('bop'), talib.BOP(o, h, l, c))
    for p in (9, 14):  # 14 is the shared TA-Lib default
        _parity(df.exec(f'cmo:{p}'), talib.CMO(c, p))
        _parity(df.exec(f'natr:{p}'), talib.NATR(h, l, c, p))
    _parity(df.exec('cmo'), talib.CMO(c, 14))  # default resolves to 14


def test_range_based_match_talib(ohlc):
    df, h, l, c = ohlc
    for p in (7, 14):  # 14 is the shared TA-Lib default
        _parity(df.exec(f'midpoint:{p}'), talib.MIDPOINT(c, p))
        _parity(df.exec(f'midprice:{p}'), talib.MIDPRICE(h, l, p))
        _parity(df.exec(f'willr:{p}'), talib.WILLR(h, l, c, p))
    _parity(df.exec('willr'), talib.WILLR(h, l, c, 14))  # default resolves to 14


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
