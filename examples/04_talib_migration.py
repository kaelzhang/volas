"""For TA-Lib users: the same indicators, addressed as cached frame columns.

    python examples/04_talib_migration.py

TA-Lib is a function surface over NumPy arrays. volas is an OHLCV-native frame
that owns the indicator cache, so the same indicators are requested as columns
and refresh incrementally when you append a bar.

    TA-Lib                                          volas
    ------------------------------------------      --------------------------------
    talib.RSI(close, timeperiod=14)                 df["rsi:14"]
    talib.ATR(high, low, close, timeperiod=14)      df["atr:14"]
    macd, signal, hist = talib.MACD(close,          df[["macd", "macd.signal",
        fastperiod=12, slowperiod=26,                     "macd.histogram"]]
        signalperiod=9)
"""

import numpy as np

from volas import DataFrame

n = 64
close = 100.0 + 5.0 * np.sin(np.arange(n) / 4.0) + np.arange(n) * 0.1
df = DataFrame(
    {
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 1_000.0),
    }
)

rsi = df["rsi:14"]
atr = df["atr:14"]
macd = df[["macd", "macd.signal", "macd.histogram"]]

print("rsi:14 tail:", rsi.tail(3).to_numpy())
print("atr:14 tail:", atr.tail(3).to_numpy())
print("macd lines tail:\n", macd.tail(3).to_numpy())

print("OK: RSI / ATR / MACD computed via volas directives (TA-Lib parity surface).")
