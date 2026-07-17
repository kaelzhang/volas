# pandas migration for OHLCV workflows

volas is pandas-shaped, not a general-purpose pandas clone. Use it when your
pipeline is mostly OHLCV data plus technical indicators, especially when new
bars arrive and indicator columns must refresh quickly.

## What usually transfers

These pandas-style patterns are the intended surface:

```py
from volas import read_csv

df = read_csv("bars.csv")

close = df["close"]
last = df.iloc[-1]
last_close = df.at[df.index[-1], "close"]
recent = df.iloc[-100:][["open", "high", "low", "close"]]
features = df[["close", "rsi:14", "macd.signal"]].to_numpy()
```

For live data, request indicator directives as columns:

```py
df["rsi:14"]
df.append(new_bar)
df["rsi:14"]       # refresh only the stale tail
```

If you later read several cached indicator columns together, call `df.fulfill()`
after `append` and before the bulk read:

```py
df.fulfill()
features = df[["close", "rsi:14", "macd.signal"]].to_numpy()
```

## What to change

Replace pandas-only indicator code with directives:

```py
# pandas-style pipeline
pdf["rsi_14"] = compute_rsi(pdf["close"], 14)

# volas
df["rsi:14"]
```

Use explicit export boundaries when you need another library:

```py
arr = df[["close", "rsi:14", "atr:14"]].to_numpy()
pdf = df.to_pandas()
```

## What not to migrate

Keep pandas or polars for general dataframe analytics: joins, pivots,
group-bys, arbitrary reshaping, mixed `object` columns, and non-OHLCV tables.
Those are not the target of volas.

Read [PANDAS-DIFFERENCES.md](../PANDAS-DIFFERENCES.md) before depending on edge
cases. The type system is deliberately stricter: volas preserves dtype and raises
on lossy implicit conversions instead of silently widening to `object` or
`float64`.

See also:

- [pandas migration example](../examples/02_pandas_migration.py)
- [When not to use volas](when-not-to-use.md)
