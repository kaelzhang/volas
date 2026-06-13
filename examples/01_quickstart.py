"""volas in five lines: cache a few indicator directives, read them back.

    python examples/01_quickstart.py

A directive such as ``rsi:14`` or ``macd.signal`` is requested like a column.
volas computes it once, caches it on the frame, and hands you a Series.
"""

import numpy as np

from volas import DataFrame

# A small synthetic OHLCV frame (replace with volas.read_csv("btc_1m.csv")).
n = 64
t = np.arange(n, dtype=float)
close = 100.0 + 5.0 * np.sin(t / 5.0) + t * 0.1
df = DataFrame(
    {
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": 1_000.0 + 10.0 * t,
    }
)

# Single-output directives.
df["ma:5"]
df["rsi:14"]
df["atr:14"]

# Multi-output directives expose each line as a sub-command.
lines = df[["macd", "macd.signal", "macd.histogram"]]

print(lines.tail(3))
print(f"OK: cached {len(['ma:5', 'rsi:14', 'atr:14', 'macd'])} indicator directives on a {n}-bar frame.")
