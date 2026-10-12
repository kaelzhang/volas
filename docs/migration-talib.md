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

## Missing rows

A NaN in the middle of the input is where the two libraries part ways for a
recursive indicator (EMA, MACD, RSI, ATR, ADX, OBV, SAR, the Hilbert transform, …).
TA-Lib carries the NaN into every later output. volas skips the row: it outputs NaN
there and continues the recursion from the last complete row, exactly as if the row
had been deleted. A finite-window indicator differs too: TA-Lib's MAX / MIN / WILLR /
AROON step over a NaN inside their window, and its candlestick patterns carry one bad
candle into every later decision; volas is NaN exactly while the window holds the bad
row. See [Missing and non-finite inputs](indicator-inputs.md#missing-and-non-finite-inputs)
for the full rule.

## Sliding sums

TA-Lib's moving averages, rolling sums, variances and regressions slide a plain
running sum, which keeps the rounding of every row it ever held: after a spike leaves
the window every later value stays off, and a long history drifts. volas keeps each
window's sum exact (see [Exact window sums](indicator-inputs.md#exact-window-sums)),
so on such data its values are the exact ones and TA-Lib's are not; on ordinary data the
two agree to the last few bits. Variances, correlations, betas and regression slopes are
taken from centred moments of those exact sums rather than TA-Lib's raw moments (see
[centred moments](indicator-inputs.md#centred-moments)):
where TA-Lib's cancel — a window whose mean dwarfs its spread — volas's values are the
exact ones, and a correlation never leaves `[-1, 1]`.

The candlestick patterns' body / range averages slide the same exact sum. A pattern is a
threshold test (`body <= 0.1 · average range` for a doji), so on a bar whose two sides
are within a rounding error of each other TA-Lib's drifted running total and volas's exact
sum can decide differently. On prices quantised to a 0.1 tick this shows after about ten
thousand bars (a few bars per pattern in twenty thousand); checked against rational
arithmetic on the same float inputs, volas's decision is the exact one.

## Deliberate divergences

TA-Lib is the oracle for every TA-Lib-compatible directive; where volas departs from
it, the departure is deliberate and listed here.

- **Zero tests are exact.** TA-Lib's `TA_IS_ZERO` treats any `|x| < 1e-14` as zero, a
  threshold in price units (MFI's is `total < 1.0`, in volume units). volas tests exact
  zero in RSI and CMO (`avgGain + avgLoss`), PPO (the slow average), NATR (the close),
  CORREL / BETA (the variances) and MFI (the money-flow total). Both give the same result
  on a flat series; on a tiny-priced series (FX-scale returns, prices scaled by `1e-15`)
  TA-Lib returns 0 where volas returns the value the same series has at any other scale.
- **ULTOSC weights by position.** `ultosc:<short>,<medium>,<long>` gives 4/2/1 to the
  arguments in the order written; TA-Lib sorts the periods ascending first. The two
  agree whenever the arguments are in order.
- **±DI at period 1** keeps the ×100 of every other period (TA-Lib's period-1 branch
  drops it).
- **MACD / MACDEXT lines** are the clean fast-minus-slow difference from each average's
  own start; TA-Lib seeds them from an offset aligned to the signal line.
- **Warm-up rows are NaN**, also for TA-Lib's integer outputs (HT_TRENDMODE, the
  candlestick patterns) where TA-Lib writes 0.

Where TA-Lib's arguments are out of order, volas follows it: a `slow` period shorter
than `fast` swaps the two averages in `macd`, `apo`, `ppo` and `macdext`.

See also:

- [TA-Lib example script](https://github.com/kaelzhang/volas/blob/main/examples/04_talib_migration.py)
- [Directive cheat sheet](directive-cheatsheet.md)
- [Benchmark FAQ](benchmark-faq.md)
