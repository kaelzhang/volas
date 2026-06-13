"""The headline use case: append one live bar, refresh only the stale tail.

    python examples/03_live_ohlcv_append.py

In a live loop a new bar arrives and the cached indicator columns must update.
volas does NOT recompute the full series: appending marks the cache stale, and
the next read refreshes only the affected tail in O(lookback), not O(n).
"""

import numpy as np

from volas import DataFrame

# Seed history.
n = 50
close = 100.0 + np.cumsum(np.full(n, 0.3))
bars = DataFrame(
    {
        "open": close - 0.2,
        "high": close + 0.4,
        "low": close - 0.4,
        "close": close,
        "volume": np.full(n, 1_000.0),
    }
)

# Cache indicator columns up front.
bars["ma:3"]
bars["rsi:14"]

# A new OHLCV bar arrives (one-row frame). append() then fulfill() applies it.
new_bar = DataFrame(
    {
        "open": [115.0],
        "high": [116.0],
        "low": [114.5],
        "close": [115.5],
        "volume": [1_500.0],
    }
)
bars.append(new_bar)
bars.fulfill()

# The cached directives refreshed only their tail — values are still correct.
print(bars[["close", "ma:3", "rsi:14"]].tail(3))

print(f"OK: appended 1 live bar (now {len(bars)} bars); cached indicators refreshed incrementally.")
