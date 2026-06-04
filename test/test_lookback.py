"""volas directive-lookback tests.

Ported 1:1 from stock-pandas's ``test_basic.py::test_lookback`` — the minimum
number of prior rows a directive needs before it yields a valid value. volas
exposes the same staticmethod ``DataFrame.directive_lookback(directive)``.
"""

import pytest

from volas import DataFrame

# (directive, expected_lookback) — identical to the stock-pandas suite.
CASES = [
    # Trend-following: ma, ema (lookback = period - 1)
    ('ma:5', 4), ('ema:5', 4),
    # MACD variants
    ('macd', 25), ('macd.signal', 33), ('macd.histogram', 33),
    # BBI (lookback = max of all periods)
    ('bbi', 24),
    # TR & ATR
    ('tr', 1), ('atr', 14),
    # LLV, HHV, Donchian (lookback = period - 1)
    ('llv:5', 4), ('hhv:5', 4), ('donchian:5', 4),
    # RSV & KDJ
    ('rsv:9', 8), ('kdj.k', 27), ('kdj.d', 27), ('kdj.j', 27),
    # RSI (lookback = period, due to diff + SMMA warmup)
    ('rsi', 14),
    # Bollinger Bands (lookback = period - 1)
    ('boll', 19), ('boll.upper', 19), ('boll.lower', 19), ('bbw', 19),
    # Historical Volatility (lookback = period, due to log return)
    ('hv:20', 20),
    # Tools
    ('increase:1@close', 0), ('style:bullish', 0),
    ('repeat:2@(style:bullish)', 1), ('change:2@close', 1),
    # Compound directives: lookback = base_lb + series_lb
    ('repeat:5@(close > boll.upper)', 23),
    ('repeat:3@(ma:10 > ma:20)', 21),
    # --- Additional cases with varying parameters ---
    ('ma:10', 9), ('ma:20', 19), ('ema:12', 11), ('ema:26', 25),
    ('macd:5,10', 9), ('macd.signal:5,10,3', 11), ('macd.histogram:8,17,5', 20),
    ('bbi:5,10,15,20', 20),
    ('atr:7', 7), ('rsi:7', 7),
    ('boll:10', 9), ('boll.upper:30,2.5', 29),
    ('kdj.k:5,3,50', 15), ('kdj.d:14,5,5,50', 42),
    ('hv:10', 10), ('hv:30', 30),
]


@pytest.mark.parametrize('directive,expected', CASES)
def test_directive_lookback(directive, expected):
    assert DataFrame.directive_lookback(directive) == expected
