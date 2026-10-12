"""The idiomatic pandas reference of each core benchmark indicator
(``test_benchmark``'s ``pandas`` candidate): one function per indicator over the
benchmark history frame."""

import pandas as pd


def _pd_ma(csv):
    return csv['close'].rolling(20).mean()


def _pd_ema(csv):
    return csv['close'].ewm(span=12, adjust=True, min_periods=12).mean()


def _pd_macd(csv):
    c = csv['close']
    return (c.ewm(span=12, adjust=True, min_periods=12).mean()
            - c.ewm(span=26, adjust=True, min_periods=26).mean())


def _pd_macd_signal(csv):
    return _pd_macd(csv).ewm(span=9, adjust=True, min_periods=9).mean()


def _pd_boll_upper(csv):
    c = csv['close']
    return c.rolling(20).mean() + 2.0 * c.rolling(20).std(ddof=0)


def _pd_willr(csv):
    h, lo, c = csv['high'], csv['low'], csv['close']
    hh, ll = h.rolling(14).max(), lo.rolling(14).min()
    return -100.0 * (hh - c) / (hh - ll)


def _pd_rsi(csv):
    c = csv['close']
    delta = c.diff()
    ag = delta.clip(lower=0.0).ewm(alpha=1.0 / 14, adjust=True, min_periods=14).mean()
    al = (-delta).clip(lower=0.0).ewm(alpha=1.0 / 14, adjust=True, min_periods=14).mean()
    return 100.0 - 100.0 / (1.0 + ag / al)


def _pd_atr(csv):
    h, lo, c = csv['high'], csv['low'], csv['close']
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def _pd_llv(csv):
    return csv['low'].rolling(10).min()


def _pd_hhv(csv):
    return csv['high'].rolling(10).max()


PANDAS_CALC = {
    'ma:20': _pd_ma, 'ema:12': _pd_ema, 'macd': _pd_macd,
    'macd.signal': _pd_macd_signal, 'boll.upper': _pd_boll_upper, 'willr:14': _pd_willr,
    'rsi:14': _pd_rsi, 'atr:14': _pd_atr, 'llv:10': _pd_llv, 'hhv:10': _pd_hhv,
}
