# TA-Lib migration

TA-Lib is a function library over arrays. volas is an OHLCV-native `DataFrame`
that owns cached indicator columns. You ask for an indicator as a directive,
then the frame keeps that column fresh after `append`.

```py
from volas import DataFrame

df = DataFrame({
    "open": open_,
    "high": high,
    "low": low,
    "close": close,
    "volume": volume,
})
```

## Common mappings

| TA-Lib call | volas directive |
| --- | --- |
| `talib.RSI(close, timeperiod=14)` | `df["rsi:14"]` |
| `talib.ATR(high, low, close, timeperiod=14)` | `df["atr:14"]` |
| `talib.MACD(close, 12, 26, 9)[0]` | `df["macd:12,26"]` |
| `talib.MACD(close, 12, 26, 9)[1]` | `df["macd.signal:12,26,9"]` |
| `talib.MACD(close, 12, 26, 9)[2]` | `df["macd.histogram:12,26,9"]` |
| `talib.BBANDS(close, 20, 2, 2)[0]` | `df["boll.upper:20,2"]` |
| `talib.BBANDS(close, 20, 2, 2)[1]` | `df["boll:20"]` |
| `talib.BBANDS(close, 20, 2, 2)[2]` | `df["boll.lower:20,2"]` |

Default arguments can be omitted:

```py
df["rsi:14"]
df["atr:14"]
df[["macd", "macd.signal", "macd.histogram"]]
df[["boll.upper", "boll", "boll.lower"]]
```

Use `@` to choose a different input column:

```py
df["rsi:14@settle"]
df["boll.upper:20,2@typical_price"]
```

## Live append

The main difference is after new data arrives:

```py
df["rsi:14"]       # materialize and cache the indicator column
df.append(new_bar)
df["rsi:14"]       # refresh only the stale tail
```

If a frame has several cached indicator columns and you plan to export or slice
them together after `append`, call `df.fulfill()` first. That refreshes every
stale cached directive before a bulk read.

TA-Lib itself has no frame-owned indicator cache, so a live loop usually
recomputes from arrays unless you build caching around it. volas makes the cache
part of the frame.

See also:

- [TA-Lib example script](../examples/04_talib_migration.py)
- [Directive cheat sheet](directive-cheatsheet.md)
- [Benchmark FAQ](benchmark-faq.md)
