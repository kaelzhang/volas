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


def test_macdfix_matches_macd_with_fixed_periods(ohlc):
    # MACDFIX is MACD with fast/slow fixed at 12/26; it reuses the verified macd
    # line/signal/histogram, so it equals macd:12,26 exactly. Like macd, the line is
    # the clean EMA(12)-EMA(26) difference (diverging from talib.MACDFIX's internally
    # inconsistent EMA by the same documented quirk).
    df, h, l, c = ohlc
    np.testing.assert_array_equal(df.exec('macdfix'), df.exec('macd:12,26'))
    np.testing.assert_array_equal(df.exec('macdfix.signal:9'), df.exec('macd.signal:12,26,9'))
    np.testing.assert_array_equal(df.exec('macdfix.histogram'), df.exec('macd.histogram:12,26,9'))
    _parity(df.exec('macdfix'), talib.EMA(c, 12) - talib.EMA(c, 26), mask_want=True)


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
        for vf in (0.0, 0.3, 0.5, 0.9, 1.0):  # cover the vfactor range incl. boundaries
            _parity(df.exec(f't3:{p},{vf}'), talib.T3(c, p, vfactor=vf))
    for p in (10, 30):  # 30 is the TA-Lib default
        _parity(df.exec(f'kama:{p}'), talib.KAMA(c, p))
    # mavp: per-row period from a (required) second series; use (high-low) as the periods.
    periods = h - l
    _parity(df.exec('mavp:2,30@close,(high-low)'), talib.MAVP(c, periods, 2, 30))
    _parity(df.exec('mavp:5,20,1@close,(high-low)'),
            talib.MAVP(c, periods, minperiod=5, maxperiod=20, matype=1))
    _parity(df.exec('sar'), talib.SAR(h, l))  # defaults 0.02 / 0.2
    _parity(df.exec('sar:0.01,0.1'), talib.SAR(h, l, acceleration=0.01, maximum=0.1))
    # sarext: signed SAR (negative while short). Defaults, then a forced-short start + offset.
    _parity(df.exec('sarext'), talib.SAREXT(h, l))
    _parity(df.exec('sarext:-0.5,0.1,0.02,0.03,0.25,0.02,0.04,0.3'),
            talib.SAREXT(h, l, startvalue=-0.5, offsetonreverse=0.1,
                         accelerationinitlong=0.02, accelerationlong=0.03, accelerationmaxlong=0.25,
                         accelerationinitshort=0.02, accelerationshort=0.04, accelerationmaxshort=0.3))
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
        for nb in (0.5, 1.5, 2.0, 3.0):  # the standard-deviation multiplier
            _parity(df.exec(f'stddev:{p},{nb}'), talib.STDDEV(c, p, nbdev=nb))
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


def test_llv_hhv_match_talib(ohlc):
    # llv / hhv = rolling lowest-low / highest-high (TA-Lib MIN over low, MAX over high).
    df, h, l, c = ohlc
    for p in (10, 20):
        _parity(df.exec(f'llv:{p}'), talib.MIN(l, p))
        _parity(df.exec(f'hhv:{p}'), talib.MAX(h, p))


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
    # matype 7 = MAMA: TA_MA ignores the period and returns the MAMA line (0.5/0.05).
    _parity(df.exec('ma:10,7'), talib.MA(c, 10, 7))
    # Price oscillators across MA types
    for mt in (0, 1, 5):
        _parity(df.exec(f'apo:12,26,{mt}'), talib.APO(c, 12, 26, mt))
        _parity(df.exec(f'ppo:12,26,{mt}'), talib.PPO(c, 12, 26, mt))
    _parity(df.exec('apo'), talib.APO(c, 12, 26, 0))  # defaults 12/26/SMA
    _parity(df.exec('ppo'), talib.PPO(c, 12, 26, 0))


def test_stochastic_family_matches_talib(ohlc):
    # The k line (slowk/fastk) is emitted at its natural first-valid row; TA-Lib delays
    # it to align with the d line (so STOCH's two outputs share a start). Best practice,
    # same as the MACD line — so the k line is compared where TA-Lib emits (mask_want),
    # the d line exactly. TA-Lib computes k internally at its natural start regardless,
    # so the d values match outright.
    df, h, l, c = ohlc
    sk, sd = talib.STOCH(h, l, c)  # defaults 5,3,SMA,3,SMA
    _parity(df.exec('stoch.k'), sk, mask_want=True)
    _parity(df.exec('stoch.d'), sd)
    sk, sd = talib.STOCH(h, l, c, 9, 3, 1, 3, 1)  # EMA smoothing (matype 1)
    _parity(df.exec('stoch.k:9,3,1,3,1'), sk, mask_want=True)
    _parity(df.exec('stoch.d:9,3,1,3,1'), sd)
    fk, fd = talib.STOCHF(h, l, c)  # defaults 5,3,SMA
    _parity(df.exec('stochf.k'), fk, mask_want=True)
    _parity(df.exec('stochf.d'), fd)
    fk, fd = talib.STOCHF(h, l, c, 9, 4, 1)
    _parity(df.exec('stochf.k:9,4,1'), fk, mask_want=True)
    _parity(df.exec('stochf.d:9,4,1'), fd)
    with pytest.raises(Exception):
        df.exec('stoch')  # multi-output: requires a sub-command
    # StochRSI = stochastic of RSI; same k-line natural-start divergence.
    fk, fd = talib.STOCHRSI(c)  # defaults rsi 14, fastk 5, fastd 3, SMA
    _parity(df.exec('stochrsi.k'), fk, mask_want=True)
    _parity(df.exec('stochrsi.d'), fd)
    fk, fd = talib.STOCHRSI(c, 10, 5, 4, 1)
    _parity(df.exec('stochrsi.k:10,5,4,1'), fk, mask_want=True)
    _parity(df.exec('stochrsi.d:10,5,4,1'), fd)
    # The slow / fast-d line can be smoothed by ANY TA-Lib MA type. WMA/TRIMA/KAMA
    # (and DEMA/TEMA/T3) previously mishandled the %K line's leading-NaN warm-up
    # (all-NaN, or seeded too early with garbage); verify every smoothing type.
    for mt in (0, 1, 2, 3, 4, 5, 6, 8):  # SMA EMA WMA DEMA TEMA TRIMA KAMA T3
        _parity(df.exec(f'stochf.d:5,3,{mt}'), talib.STOCHF(h, l, c, 5, 3, mt)[1])
        _parity(df.exec(f'stoch.d:5,3,{mt},3,{mt}'), talib.STOCH(h, l, c, 5, 3, mt, 3, mt)[1])


def test_correl_beta_match_talib(ohlc):
    df, h, l, c = ohlc
    v = np.asarray(df['volume'].to_numpy(), dtype=float)
    for p in (14, 30):  # 30 is CORREL's default
        _parity(df.exec(f'correl:{p}@high,low'), talib.CORREL(h, l, p))
    for p in (5, 10):  # 5 is BETA's default
        _parity(df.exec(f'beta:{p}@high,low'), talib.BETA(h, l, p))
    # The first series defaults to close via an empty leading operand (@,<second>).
    _parity(df.exec('correl:30@,volume'), talib.CORREL(c, v, 30))
    _parity(df.exec('beta@,volume'), talib.BETA(c, v, 5))  # default period 5
    # The second series is required.
    with pytest.raises(Exception):
        df.exec('correl:30')
    with pytest.raises(Exception):
        df.exec('beta')


def test_directional_family_matches_talib(ohlc):
    df, h, l, c = ohlc
    for p in (14, 20):  # 14 is the shared TA-Lib default
        _parity(df.exec(f'plus_dm:{p}'), talib.PLUS_DM(h, l, p))
        _parity(df.exec(f'minus_dm:{p}'), talib.MINUS_DM(h, l, p))
        _parity(df.exec(f'plus_di:{p}'), talib.PLUS_DI(h, l, c, p))
        _parity(df.exec(f'minus_di:{p}'), talib.MINUS_DI(h, l, c, p))
        _parity(df.exec(f'dx:{p}'), talib.DX(h, l, c, p))
        _parity(df.exec(f'adx:{p}'), talib.ADX(h, l, c, p))
        _parity(df.exec(f'adxr:{p}'), talib.ADXR(h, l, c, p))
    _parity(df.exec('adx'), talib.ADX(h, l, c, 14))  # default resolves to 14


# All 61 TA-Lib candlestick patterns: (volas name, talib function, lookback).
_CANDLE_PATTERNS = [
        ('doji', talib.CDLDOJI, 10), ('marubozu', talib.CDLMARUBOZU, 10),
        ('closingmarubozu', talib.CDLCLOSINGMARUBOZU, 10), ('longline', talib.CDLLONGLINE, 10),
        ('shortline', talib.CDLSHORTLINE, 10), ('highwave', talib.CDLHIGHWAVE, 10),
        ('spinningtop', talib.CDLSPINNINGTOP, 10), ('dragonflydoji', talib.CDLDRAGONFLYDOJI, 10),
        ('gravestonedoji', talib.CDLGRAVESTONEDOJI, 10),
        ('longleggeddoji', talib.CDLLONGLEGGEDDOJI, 10),
        ('rickshawman', talib.CDLRICKSHAWMAN, 10), ('belthold', talib.CDLBELTHOLD, 10),
        ('hammer', talib.CDLHAMMER, 11), ('hangingman', talib.CDLHANGINGMAN, 11),
        ('invertedhammer', talib.CDLINVERTEDHAMMER, 11),
        ('shootingstar', talib.CDLSHOOTINGSTAR, 11), ('takuri', talib.CDLTAKURI, 10),
        # two-bar
        ('engulfing', talib.CDLENGULFING, 2), ('harami', talib.CDLHARAMI, 11),
        ('haramicross', talib.CDLHARAMICROSS, 11), ('piercing', talib.CDLPIERCING, 11),
        ('darkcloudcover', talib.CDLDARKCLOUDCOVER, 11), ('dojistar', talib.CDLDOJISTAR, 11),
        ('homingpigeon', talib.CDLHOMINGPIGEON, 11), ('matchinglow', talib.CDLMATCHINGLOW, 6),
        ('inneck', talib.CDLINNECK, 11), ('onneck', talib.CDLONNECK, 11),
        ('thrusting', talib.CDLTHRUSTING, 11), ('kicking', talib.CDLKICKING, 11),
        ('kickingbylength', talib.CDLKICKINGBYLENGTH, 11),
        ('separatinglines', talib.CDLSEPARATINGLINES, 11),
        ('counterattack', talib.CDLCOUNTERATTACK, 11),
        # three-bar
        ('morningstar', talib.CDLMORNINGSTAR, 12), ('eveningstar', talib.CDLEVENINGSTAR, 12),
        ('3inside', talib.CDL3INSIDE, 12), ('3outside', talib.CDL3OUTSIDE, 3),
        ('3whitesoldiers', talib.CDL3WHITESOLDIERS, 12),
        ('3blackcrows', talib.CDL3BLACKCROWS, 13),
        ('morningdojistar', talib.CDLMORNINGDOJISTAR, 12),
        ('eveningdojistar', talib.CDLEVENINGDOJISTAR, 12),
        ('abandonedbaby', talib.CDLABANDONEDBABY, 12), ('2crows', talib.CDL2CROWS, 12),
        ('upsidegap2crows', talib.CDLUPSIDEGAP2CROWS, 12),
        ('advanceblock', talib.CDLADVANCEBLOCK, 12),
        ('stalledpattern', talib.CDLSTALLEDPATTERN, 12),
        ('identical3crows', talib.CDLIDENTICAL3CROWS, 12),
        ('sticksandwich', talib.CDLSTICKSANDWICH, 7), ('tristar', talib.CDLTRISTAR, 12),
        ('unique3river', talib.CDLUNIQUE3RIVER, 12),
        ('gapsidesidewhite', talib.CDLGAPSIDESIDEWHITE, 7),
        ('tasukigap', talib.CDLTASUKIGAP, 7),
        ('3starsinsouth', talib.CDL3STARSINSOUTH, 12),
        # four / five-bar
        ('3linestrike', talib.CDL3LINESTRIKE, 8), ('breakaway', talib.CDLBREAKAWAY, 14),
        ('ladderbottom', talib.CDLLADDERBOTTOM, 14),
        ('concealbabyswall', talib.CDLCONCEALBABYSWALL, 13),
        ('mathold', talib.CDLMATHOLD, 14),
        ('risefall3methods', talib.CDLRISEFALL3METHODS, 14),
        ('xsidegap3methods', talib.CDLXSIDEGAP3METHODS, 2),
        ('hikkake', talib.CDLHIKKAKE, 5), ('hikkakemod', talib.CDLHIKKAKEMOD, 10),
]
assert len(_CANDLE_PATTERNS) == 61, 'all 61 TA-Lib candlestick patterns covered'


def _candle_pat(got, want, lb):
    # f64 -100/0/100, warm-up NaN (TA-Lib fills its int output's warm-up with 0).
    # Values are exact, so compare the valid region.
    got = np.asarray(got, dtype=float)
    assert np.all(np.isnan(got[:lb])), 'pattern warm-up is NaN'
    np.testing.assert_array_equal(got[lb:], np.asarray(want, dtype=float)[lb:])


def test_candlestick_patterns_match_talib(ohlc):
    df, h, l, c = ohlc
    o = df['open'].to_numpy()
    for name, fn, lb in _CANDLE_PATTERNS:
        want = fn(o, h, l, c)
        _candle_pat(df.exec(f'style.{name}'), want, lb)
        _candle_pat(df.exec(f'cdl.{name}'), want, lb)  # the cdl alias matches
    # the penetration ratio is an optional arg (default 0.5)
    _candle_pat(df.exec('cdl.darkcloudcover:0.6'),
                talib.CDLDARKCLOUDCOVER(o, h, l, c, penetration=0.6), 11)


def _synthetic_candle_ohlc():
    """A long deterministic OHLC series that triggers every candlestick pattern.

    The tencent set fires only ~38 of the 61 patterns, leaving most pattern-geometry
    branches unexercised. A pseudo-random walk (deterministic LCG) fires 55; six rare
    patterns (3whitesoldiers, identical3crows, kicking, kickingbylength, breakaway,
    concealbabyswall) need exact geometry, appended as hand-built motifs (each preceded
    by flat context so its candle-settings averages are clean). Every value is still
    checked 1:1 against TA-Lib, so this doubles as a fuzz parity test."""
    def lcg(seed, count):
        out = []
        x = seed
        for _ in range(count):
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            out.append(x / 0x7FFFFFFF)
        return out

    n = 20000
    r = lcg(123456789, n * 5)
    o, h, l, c = [], [], [], []
    price = 100.0
    for i in range(n):
        g, bd, us, ls, dr = r[5 * i: 5 * i + 5]
        price += (g - 0.5) * (8 if dr > 0.8 else 2)
        op = price
        cl = op + (bd - 0.5) * (6 if us > 0.7 else 1.5)
        hi = max(op, cl) + us * 3.0 * (0.2 if ls < 0.3 else 1.0)
        lo = min(op, cl) - ls * 3.0 * (0.2 if us < 0.3 else 1.0)
        o.append(op); h.append(hi); l.append(lo); c.append(cl); price = cl

    # [open, high, low, close] motifs for the six patterns a random walk won't produce.
    motifs = [
        [[100, 110.05, 99.9, 110], [105, 115.05, 104.9, 115], [110, 120.05, 109.9, 120]],
        [[120, 120.05, 109.95, 110], [110, 110.05, 99.95, 100], [100, 100.05, 89.95, 90]],
        [[110, 110, 100, 100], [115, 130, 115, 130]],  # kicking: 2nd marubozu longer
        [[115, 115, 100, 100], [120, 130, 120, 130]],  # kicking: 1st marubozu longer
        [[120, 120.1, 109.9, 110], [108, 108.1, 103.9, 104], [105, 105.1, 101.9, 102],
         [103, 103.1, 99.9, 100], [101, 109.1, 100.9, 109]],
        [[110, 110, 100, 100], [99, 99, 90, 90], [88, 92, 84, 85], [95, 95.5, 81, 82]],
    ]
    for m in motifs:
        for k in range(16):  # flat context isolates the motif's candle-settings averages
            base = 100.0 + (k % 2) * 0.2
            bo, bc = base, base + (0.3 if k % 2 == 0 else -0.3)
            o.append(bo); h.append(max(bo, bc) + 0.2); l.append(min(bo, bc) - 0.2); c.append(bc)
        for b in m:
            o.append(b[0]); h.append(b[1]); l.append(b[2]); c.append(b[3])
    return tuple(np.array(x, dtype=float) for x in (o, h, l, c))


@pytest.fixture(scope='module')
def synth_candle_ohlc():
    return _synthetic_candle_ohlc()


def test_candlestick_patterns_match_talib_synthetic(synth_candle_ohlc):
    # Exercises every pattern's signal branch (tencent fires only ~38/61) while still
    # asserting exact parity with TA-Lib on a long deterministic series.
    o, h, l, c = synth_candle_ohlc
    df = volas.DataFrame({'open': o, 'high': h, 'low': l, 'close': c})
    for name, fn, lb in _CANDLE_PATTERNS:
        got = np.asarray(df.exec(f'style.{name}'), dtype=float)
        np.testing.assert_array_equal(got[lb:], np.asarray(fn(o, h, l, c), dtype=float)[lb:])
        assert np.any(got[lb:] != 0), f'{name} never fired on the synthetic series'


def test_math_transform_series_methods_match_talib(ohlc):
    # TA-Lib "Math Transform" group, implemented as element-wise Series methods (not
    # directives). Parity holds on the raw close series — out-of-domain inputs (acos of
    # a price, exp overflow) yield NaN/inf identically in both.
    df, _, _, c = ohlc
    close = df['close']
    pairs = [
        ('acos', talib.ACOS), ('asin', talib.ASIN), ('atan', talib.ATAN),
        ('ceil', talib.CEIL), ('cos', talib.COS), ('cosh', talib.COSH),
        ('exp', talib.EXP), ('floor', talib.FLOOR), ('ln', talib.LN),
        ('log10', talib.LOG10), ('sin', talib.SIN), ('sinh', talib.SINH),
        ('sqrt', talib.SQRT), ('tan', talib.TAN), ('tanh', talib.TANH),
    ]
    for name, fn in pairs:
        _parity(getattr(close, name)(), fn(c))


def test_macdext_matches_talib(ohlc):
    df, h, l, c = ohlc
    macd, signal, hist = talib.MACDEXT(c)  # all matypes default to SMA
    _parity(df.exec('macdext'), macd, mask_want=True)  # line at natural start
    _parity(df.exec('macdext.signal'), signal)
    _parity(df.exec('macdext.histogram'), hist)
    # With a seeded MA (EMA fast), volas computes the clean standalone MA difference
    # (best practice, like macd); talib.MACDEXT instead uses its own seeded/aligned MAs
    # so its line != standalone MA(12,1)-MA(26,0). Compare against the clean diff.
    clean = talib.MA(c, 12, 1) - talib.MA(c, 26, 0)
    _parity(df.exec('macdext:12,1,26,0'), clean, mask_want=True)
    # The signal line can use any MA type. With an SMA line (no EMA-seeding quirk) the
    # signal/histogram match talib.MACDEXT exactly across every smoothing type — and
    # this exercises the leading-NaN-skip fix, since the macd line warms up with NaN.
    for sig_mt in (0, 1, 2, 3, 4, 5, 6, 8):  # SMA EMA WMA DEMA TEMA TRIMA KAMA T3
        _m, sig, hist = talib.MACDEXT(c, 12, 0, 26, 0, 9, sig_mt)
        _parity(df.exec(f'macdext.signal:12,0,26,0,9,{sig_mt}'), sig)
        _parity(df.exec(f'macdext.histogram:12,0,26,0,9,{sig_mt}'), hist)


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


def test_bollinger_bands_match_talib(ohlc):
    # boll == TA-Lib BBANDS: an SMA middle band ± nbdev * population (ddof 0) stddev.
    # The bare directive is the middle band; upper/lower take (period, nbdev). BBANDS
    # uses one nbdev for both sides with matype 0 (SMA).
    df, h, l, c = ohlc
    for p, nb in [(20, 2.0), (5, 2.0), (20, 1.0), (10, 3.0)]:
        up, mid, low = talib.BBANDS(c, timeperiod=p, nbdevup=nb, nbdevdn=nb, matype=0)
        _parity(df.exec(f'boll:{p}'), mid)
        _parity(df.exec(f'boll.upper:{p},{nb}'), up)
        _parity(df.exec(f'boll.lower:{p},{nb}'), low)
    # default form (period 20, nbdev 2) + abbreviations
    up, mid, low = talib.BBANDS(c, 20, 2.0, 2.0, 0)
    _parity(df.exec('boll.upper'), up)
    _parity(df.exec('boll.l:20,2'), low)
    # band-width = (upper - lower) / middle, the same BBANDS-derived quantity.
    _parity(df.exec('bbw'), (up - low) / mid)


def test_cci_trix_match_talib(ohlc):
    df, h, l, c = ohlc
    v = np.asarray(df['volume'].to_numpy(), dtype=float)
    for p in (14, 20):
        _parity(df.exec(f'cci:{p}'), talib.CCI(h, l, c, p))
        _parity(df.exec(f'mfi:{p}'), talib.MFI(h, l, c, v, p))
    for p in (15, 30):  # 30 is the TA-Lib default
        _parity(df.exec(f'trix:{p}'), talib.TRIX(c, p))
    _parity(df.exec('ultosc'), talib.ULTOSC(h, l, c))  # defaults 7/14/28
    # A non-default triplet and the upper boundary. (The degenerate (1,1,1) is omitted:
    # its values agree, but TA-Lib emits at row 0 from a high-low TR while volas's TR is
    # NaN with no prior close — its documented convention, see test_tr — so volas warms
    # up one row later there.)
    for t in ((5, 10, 20), (30, 30, 30)):
        _parity(df.exec(f'ultosc:{t[0]},{t[1]},{t[2]}'), talib.ULTOSC(h, l, c, *t))
    _parity(df.exec('cci'), talib.CCI(h, l, c, 14))  # default resolves to 14
    _parity(df.exec('trix'), talib.TRIX(c, 30))
    o = df['open'].to_numpy()
    for p in (14, 20):
        _parity(df.exec(f'imi:{p}'), talib.IMI(o, c, p))


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


# --- Hilbert-transform suite ----------------------------------------------------
#
# volas implements the canonical TA-Lib 0.6.4 Hilbert-transform algorithms (verified
# against the official TA-Lib source). The functions that derive purely from the
# shared Hilbert core — HT_DCPERIOD, HT_PHASOR, MAMA/FAMA — are bit-exact against the
# installed oracle on any series.
#
# The four functions that additionally read the smoothed-price DFT buffer
# (HT_DCPHASE, HT_SINE) or the raw-price iTrend window (HT_TRENDLINE, HT_TRENDMODE)
# expose a quirk of the installed `ta_lib` build: its warm-up uses a different buffer
# initial condition than canonical TA-Lib, producing a transient that decays over
# ~90 bars before both converge bit-exactly. The 100-row tencent set is too short for
# that, so these four are verified on the 1999-row series over the converged region.
# (See tasks/04/designs impl-log for the full investigation.)

TENCENT_FULL = str((Path(__file__).parent / 'data' / 'tencent_full.csv').resolve())

# First bar past the installed build's ~90-bar Hilbert warm-up transient.
_HT_CONVERGED = 300


@pytest.fixture(scope='module')
def close_full():
    df = volas.read_csv(TENCENT_FULL)
    return df, df['close'].to_numpy()


def _converged(got, want, atol=1e-6):
    """Assert parity over the converged region (past the warm-up transient)."""
    got = np.asarray(got, dtype=float)[_HT_CONVERGED:]
    want = np.asarray(want, dtype=float)[_HT_CONVERGED:]
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=atol)


def test_ht_dcperiod_matches_talib(ohlc):
    df, h, l, c = ohlc
    _parity(df.exec('ht_dcperiod'), talib.HT_DCPERIOD(c))


def test_ht_phasor_matches_talib(ohlc):
    df, h, l, c = ohlc
    inphase, quadrature = talib.HT_PHASOR(c)
    _parity(df.exec('ht_phasor'), inphase)  # primary line = in-phase
    _parity(df.exec('ht_phasor.quadrature'), quadrature)


def test_mama_matches_talib(ohlc):
    df, h, l, c = ohlc
    mama, fama = talib.MAMA(c, 0.5, 0.05)
    _parity(df.exec('mama'), mama)  # primary line = mama
    _parity(df.exec('mama.fama'), fama)
    # matype 7 (MAMA) via the generic MA dispatch returns the mama line (period ignored).
    _parity(df.exec('ma:30,7'), mama)


def test_ht_dcphase_matches_talib(close_full):
    df, c = close_full
    # DCPhase is degrees; at convergence the values coincide so there is no 360°
    # wrap ambiguity — a plain tolerance suffices.
    _converged(df.exec('ht_dcphase'), talib.HT_DCPHASE(c), atol=1e-4)


def test_ht_sine_matches_talib(close_full):
    df, c = close_full
    sine, leadsine = talib.HT_SINE(c)
    _converged(df.exec('ht_sine'), sine)  # primary line = sine
    _converged(df.exec('ht_sine.leadsine'), leadsine)


def test_ht_trendline_matches_talib(close_full):
    df, c = close_full
    _converged(df.exec('ht_trendline'), talib.HT_TRENDLINE(c))


def test_ht_trendmode_matches_talib(close_full):
    df, c = close_full
    # Integer 0/1 output: exact match over the converged region.
    got = np.asarray(df.exec('ht_trendmode'), dtype=float)[_HT_CONVERGED:]
    want = np.asarray(talib.HT_TRENDMODE(c), dtype=float)[_HT_CONVERGED:]
    np.testing.assert_array_equal(got, want)
