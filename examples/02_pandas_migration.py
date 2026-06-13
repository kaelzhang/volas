"""For pandas users: the indexing you already know, on an OHLCV frame.

    python examples/02_pandas_migration.py

volas is a pandas-SHAPED API, not a general-purpose pandas replacement. The
indexing surface (`.loc` / `.iloc` / `.at`, `to_numpy`) is familiar; the value
add is OHLCV indicator directives and an incremental cache. For plain dataframe
analysis (joins, pivots, arbitrary reshaping) keep using pandas or polars.
"""

import numpy as np

from volas import DataFrame

n = 32
close = 100.0 + np.arange(n, dtype=float)
df = DataFrame(
    {
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 1_000.0),
    }
)

# Same indexing idioms as pandas:
df["rsi:14"]                      # request a directive like a column
row = df.iloc[-1]                 # positional row access
last_close = df.at[n - 1, "close"]  # scalar by label
recent = df.iloc[-5:][["close", "rsi:14"]]  # slice + column selection

print("last row:", row)
print("last close (.at):", last_close)
print("recent close + rsi:\n", recent.to_numpy())

print("OK: pandas-style .iloc / .at / slicing works on an OHLCV frame.")
