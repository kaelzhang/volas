# volas

[![ci](https://github.com/kaelzhang/volas/actions/workflows/ci.yml/badge.svg)](https://github.com/kaelzhang/volas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kaelzhang/volas/branch/main/graph/badge.svg)](https://codecov.io/gh/kaelzhang/volas)
[![PyPI version](https://img.shields.io/pypi/v/volas.svg)](https://pypi.org/project/volas/)
[![Python versions](https://img.shields.io/pypi/pyversions/volas.svg)](https://pypi.org/project/volas/)

> High-performance, Rust-backed columnar kernel for stock / candlestick (OHLCV) time-series data.

**Status:** early development — version `0.0.0`. APIs are not yet stable and may change at any time.

`volas` is a focused, pandas-independent `DataFrame` / `Series` purpose-built for
financial candlestick (OHLCV) time series and quantitative-trading workflows. It
gives you **stock-indicator "directives"** (`df['macd.signal']`, `df['ma:5 > ma:20']`)
and a small, pandas-compatible indexing surface (`.loc` / `.iloc` / `.at`), with
the storage and compute core implemented in **Rust** for low, predictable latency
on the incremental ("append one bar") hot path of live trading.

It is a **drop-in replacement** for the slice of two libraries that candlestick /
quant code actually uses:

- **pandas** — the same `DataFrame` / `Series` construction, `.loc` / `.iloc` /
  `.at` / `.iat` indexing, `read_csv`, `to_numpy`, and resampling. (For what is
  intentionally *not* covered, see [Index limitations](#index-limitations-vs-pandas)
  and [non-goals](#design-notes--non-goals).)
- **[stock-pandas](https://github.com/kaelzhang/stock-pandas)** — the same
  indicator-directive syntax (`df['macd.signal']`, `df['ma:5 > ma:20']`).

…all with **no pandas dependency** and a Rust kernel.

- [Why volas](#why-volas)
- [Installation](#installation)
- [Quick start](#quick-start)
- [API at a glance](#api-at-a-glance)
- [Creating a DataFrame](#creating-a-dataframe)
- [Indicator directives](#indicator-directives)
- [Indexing & selection](#indexing--selection)
- [Reading CSV](#reading-csv)
- [Resampling (cumulation)](#resampling-cumulation)
- [Incremental refresh (`fulfill`)](#incremental-refresh-fulfill)
- [Aliases](#aliases)
- [Rolling over a custom function](#rolling-over-a-custom-function)
- [Error handling](#error-handling)
- [Design notes & non-goals](#design-notes--non-goals)
- [Development](#development)

## Why volas

- **Live-trading first** — optimized for the incremental hot path: append a new
  bar, refresh indicators on only the affected tail (`O(lookback)`, not `O(n)`),
  read the result, with minimal per-bar latency.
- **Indicator directives** — compute moving averages, MACD, Bollinger Bands, KDJ,
  RSI, ATR and more by indexing with a string: `df['ma:20']`, `df['boll.upper']`.
- **Small, regular data model** — a row index (range / datetime / integer /
  **string**) plus 2-D columns of numbers, booleans, integers and strings; no
  multi-level indexes, no general-purpose reshaping.
- **Rust core, pandas-free** — storage, the directive parser and the indicator
  kernels live in a compiled Rust extension; pandas is *not* a runtime dependency.
- **First-class NumPy interop** — `to_numpy()` exports columns / frames for
  NumPy and `torch.Tensor` pipelines.

## Installation

```sh
pip install volas
```

Requires Python >= 3.11. Wheels are published for Linux (x86_64 / aarch64),
macOS (x86_64 / arm64) and Windows (x86_64). For a local build from source, see
[Development](#development).

## Quick start

```py
from volas import DataFrame

df = DataFrame({
    'open':   [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    'high':   [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
    'low':    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    'close':  [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    'volume': [100, 200, 300, 400, 500, 600],
})

# A plain column -> Series
df['close']                      # Series([3, 4, 5, 6, 7, 8], name='close')

# An indicator directive -> Series (2-period SMA of `close`)
df['ma:2']                       # Series([nan, 3.5, 4.5, 5.5, 6.5, 7.5])

# A boolean directive -> bool Series, usable as a row mask
bullish = df['close > open']
df[bullish]                      # DataFrame of the rows where close > open

# Several directives at once -> DataFrame
df[['ma:2', 'ma:3', 'close > open']]

# Zero-copy-ish export to NumPy
df['close'].to_numpy()           # 1-D ndarray
df.to_numpy()                    # 2-D ndarray (rows x columns)
```

## API at a glance

A compact reference of the entire public surface (handy for tooling and agents).

```py
import volas
from volas import (
    DataFrame, Series, TimeFrame, Cumulator, read_csv, rolling_calc,
    DirectiveError, DirectiveSyntaxError, DirectiveValueError,
)

# --- construction ---------------------------------------------------------
DataFrame(data: dict[str, list | np.ndarray], date_col: str | None = None)
read_csv(path, sep=None, delimiter=None, header=True,
         parse_dates=None, index_col=None, na_values=None, keep_default_na=True)

# --- DataFrame ------------------------------------------------------------
df.columns                       # list[str]
df.shape                         # (n_rows, n_cols)
len(df)                          # n_rows
df.index                         # NumPy array of row labels
df[col] / df[directive]          # -> Series
df[[col_or_directive, ...]]      # -> DataFrame
df[bool_mask]                    # -> DataFrame (filter rows; mask = Series | ndarray)
df.get_column(name)              # -> Series (plain column, no directive parsing)
df.exec(directive, create_column=False)   # -> np.ndarray (compute without caching)
df.fulfill()                     # batch-refresh cached directive columns in place
df.append(other)                 # -> DataFrame (other: DataFrame | Row)
df.drop([label, ...])            # -> DataFrame (drop rows by index label)
df.rename({old: new})            # -> DataFrame
df.astype({col: dtype})          # -> DataFrame ('float'|'int'|'bool'|'str'|'datetime')
df.set_index(col)                # -> DataFrame (move a column into the row index)
df.alias(new_name, src_name)     # add a column alias (in place)
df.cumulate(time_frame, cumulators=None)  # -> DataFrame (resample OHLCV)
df.copy() / df.to_numpy()
df.iloc[...] / df.loc[...] / df.at[label, col] / df.iat[i, j]

# --- Series ---------------------------------------------------------------
s.name / s.dtype / len(s) / s.to_numpy()
s.iloc[...] / s.loc[...]
s + s, s - 1, -s, ...            # elementwise arithmetic

# --- resampling -----------------------------------------------------------
TimeFrame.m5 / TimeFrame.H1 / ...   # or the labels '5m', '1h', '1d', ...
TimeFrame.m5.minutes                # 5
Cumulator(time_frame, cumulators=None)
cum.append(df) ; cum.frame ; cum.last

# --- misc -----------------------------------------------------------------
rolling_calc(values, window, apply, forward=False, fill=nan)  # -> np.ndarray
```

## Creating a DataFrame

A `DataFrame` is built from a dict of equal-length columns (Python lists or NumPy
arrays). Columns may be float, int, bool or string.

```py
from volas import DataFrame
import numpy as np

df = DataFrame({'close': np.array([1.0, 2.0, 3.0]), 'volume': [10, 20, 30]})
df.shape        # (3, 2)
df.columns      # ['close', 'volume']

# Pass date_col to parse a string/datetime column and set it as the row index:
df = DataFrame(data, date_col='time_key')   # -> DatetimeIndex
```

## Indicator directives

The headline feature: index with a **directive** string and volas parses and
computes it against the frame, returning a `Series` (or a `DataFrame` for a list
of directives). The directive grammar mirrors stock-pandas.

```
command : args . sub @ series  op  command ...
   |       |    |     |
   |       |    |     └── operand column / sub-expression  (e.g. @open, @(boll))
   |       |    └── sub-command                            (e.g. macd.signal)
   |       └── comma-separated arguments                   (e.g. ma:20, kdj.k:9,3)
   └── indicator name                                      (e.g. ma, macd, boll)
```

```py
df['ma:20']                 # 20-period SMA of `close`
df['ma:20@open']            # ... of `open` instead
df['macd.signal']           # MACD signal line (default 12,26,9)
df['boll.upper:20,2']       # upper Bollinger band, period 20, width 2
df['kdj.j']                 # KDJ %J line
df['ma:2@(boll.upper)']     # directives compose: SMA of a sub-directive

# operators yield boolean / numeric Series
df['close > open']          # comparison -> bool
df['ma:5 // ma:10']         # `//` cross-up, `\\` cross-down, `><` either way
df['ma:5 > ma:20']
df['(close > boll.upper) & (volume > ma:20@volume)']
```

`df[directive]` **caches** the result as a real column (so repeated reads are
free), then auto-refreshes its stale tail on access after an `append`. Use
`df.exec(directive)` to compute a directive as a NumPy array **without** caching
it:

```py
df.exec('ma:5')                       # -> np.ndarray, nothing added to df.columns
df.exec('ma:5', create_column=True)   # also materialize it as a column
```

### Supported indicators

| Directive | Indicator | Example |
| --- | --- | --- |
| `ma` | Simple moving average | `ma:20`, `ma:10@open` |
| `ema` | Exponential moving average | `ema:12` |
| `smma` | Smoothed moving average | `smma:7` |
| `macd` | MACD (`.signal`/`.dea`, `.histogram`, `.dif`) | `macd`, `macd.signal` |
| `boll` | Bollinger Bands (`.upper`/`.u`, `.lower`/`.l`) | `boll`, `boll.upper:20,2` |
| `bbw` | Bollinger Band width | `bbw` |
| `rsv` | Raw stochastic value | `rsv:9` |
| `kdj` | KDJ stochastic (`.k`/`.d`/`.j`) | `kdj.j`, `kdj.k:9,3` |
| `rsi` | Relative Strength Index | `rsi:14` |
| `bbi` | Bull and Bear Index | `bbi` |
| `tr` / `atr` | (Average) True Range | `tr`, `atr:14` |
| `llv` / `hhv` | Lowest-low / highest-high value | `llv:10`, `hhv:10@high` |
| `donchian` | Donchian channel (`.upper`/`.lower`) | `donchian:20` |
| `hv` | Historical volatility | `hv:20,1d,252` |
| `change` | Percentage change over N bars | `change:2` |
| `increase` | Monotonic increase/decrease over N bars | `increase:3@close` |
| `style` | Candle style (`bullish` / `bearish`) | `style:bullish` |
| `repeat` | A boolean condition holding N bars in a row | `repeat:2@(style:bullish)` |

Operators: comparison `< <= == != >= >`, cross `// \\ ><`, arithmetic
`+ - * /`, logical `& | ^`, unary `~` (not) and `-` (negate).

## Indexing & selection

A pandas-compatible subset for label and positional access. The row index may be
a range, a `DatetimeIndex`, an integer index, or a **string index**.

```py
df.iloc[2]          # a Row by position (row.name is its index label)
df.iloc[10:]        # a DataFrame slice by position
df.loc[label]       # a Row by index label
df.loc[lo:hi]       # inclusive label slice (lexicographic for string indexes)
df.at[label, col]   # a scalar by label + column
df.iat[i, j]        # a scalar by position
df.index            # the row labels, as a NumPy array
```

String (symbol) index — `set_index` on a string column, then look up by symbol:

```py
df = DataFrame({'sym': ['aa', 'bb', 'cc'], 'px': [1.0, 2.0, 3.0]}).set_index('sym')
df.loc['bb']           # the row keyed 'bb'
df.loc['aa':'bb']      # inclusive, lexicographic slice
df.at['cc', 'px']      # 3.0
df.drop(['bb'])        # drop by string label
```

### Index limitations (vs pandas)

The index is deliberately simple — a **single level** of one homogeneous label
type. Relative to pandas, volas does **not** support:

- **`MultiIndex`** (hierarchical / multi-level indexes), on rows *or* columns —
  columns are a flat list of unique string names.
- **Arbitrary label dtypes** — an index is exactly one of range, datetime
  (`datetime64[ns]`), integer, or string. There is no float, categorical,
  interval, period, timedelta, or mixed-type `object` index.
- **Index algebra** — reindexing, index set operations (union / intersection),
  and automatic alignment-on-index when combining frames.
- **Duplicate-label** lookups (label access assumes unique labels).

If your workflow needs any of these, keep using pandas; volas targets the
single-level, OHLCV-shaped index that candlestick data uses.

## Reading CSV

A fast, pandas-subset CSV reader that infers per-column dtypes.

```py
import volas

df = volas.read_csv('klines.csv')                       # RangeIndex
df = volas.read_csv('klines.csv',
                    parse_dates=['time_key'],           # parse to datetime
                    index_col='time_key')               # -> DatetimeIndex
df = volas.read_csv('data.tsv', sep='\t', header=False, # no header -> '0'..'n-1'
                    na_values=['NA', 'null'])
```

## Resampling (cumulation)

Aggregate fine bars to a coarser time frame. Defaults to OHLCV semantics
(`open`=first, `high`=max, `low`=min, `close`=last, `volume`=sum); requires a
`DatetimeIndex`.

```py
from volas import TimeFrame

coarse = fine.cumulate('5m')                  # or TimeFrame.m5
coarse = fine.cumulate('1h', cumulators={'volume': 'last'})  # override an aggregator
```

Time frames: `'1s' '1m' '3m' '5m' '15m' '30m' '1h' '2h' '4h' '6h' '8h' '12h'
'1d' '3d' '1w' '1M' '1y'` (or the `TimeFrame.s1 / m1 / m5 / H1 / D1 / W1 / M1 /
Y1` constants).

For **live** streaming, feed bars to a `Cumulator` and read the running result:

```py
from volas import Cumulator

cum = Cumulator('5m')
for bar in stream:               # each `bar` is a 1-row DataFrame
    cum.append(bar)
    cum.frame                    # closed periods + the open period as a live last row
    cum.last                     # just the current (still-open) period, aggregated
```

Re-sending a bar with a timestamp already seen **updates** that period (it does
not double-count), which matches exchange data that revises the latest bar.

## Incremental refresh (`fulfill`)

`df[directive]` materializes the indicator as a column. After you `append` new
bars, those new rows are stale; reading `df[directive]` again refreshes only the
affected tail incrementally. For bulk reads (`to_numpy()`, `.iloc`) call
`fulfill()` once to batch-refresh every cached directive column in place —
`O(lookback)`, not an `O(n)` recompute.

```py
df['ma:20']                  # cache the 20-SMA as a column
df = df.append(new_bar)      # the new row's ma:20 is stale (NaN)
df.fulfill()                 # recompute only the tail of every cached column
df.to_numpy()                # now fresh
```

## Aliases

Map an alternative name onto an existing column; it resolves everywhere a column
is looked up, **including inside directives**, and survives `drop` / `copy` /
slicing.

```py
df.alias('Open', 'open')
df['Open']                   # same data as df['open']
df['ma:5@Open']              # alias resolves inside directives too
```

## Rolling over a custom function

`rolling_calc` applies an arbitrary Python callable over a trailing (or forward)
window of an array — the escape hatch for indicators not expressible as a
directive.

```py
import volas

highs = volas.rolling_calc(df['high'], 5, max)               # == df['hhv:5@high']
span  = volas.rolling_calc(df['close'], 10, lambda w: w.max() - w.min())
fwd   = volas.rolling_calc(df['close'], 5, max, forward=True)  # look-ahead window
```

## Error handling

Directive problems raise typed exceptions. Both subclass `DirectiveError` and the
built-in `ValueError`, so existing `except ValueError` handling keeps working.

```py
from volas import DirectiveSyntaxError, DirectiveValueError

try:
    df['ma:2,3']                 # too many arguments
except DirectiveValueError as e:
    ...                          # unknown command/sub-command, bad arg, bad value

try:
    df['a >']                    # malformed expression
except DirectiveSyntaxError as e:
    ...                          # message carries the line / column of the error
```

## Design notes & non-goals

- **Not a general-purpose DataFrame.** volas models exactly what OHLCV
  quant workflows need; it deliberately omits multi-level indexes, heterogeneous
  per-cell storage, joins and general reshaping.
- **pandas-independent at runtime.** pandas / stock-pandas are used only as test
  oracles (1:1 parity tests and a 3-way benchmark), never imported at runtime.
- **External API cleanliness first.** The Python surface is kept clean and
  pandas-shaped; internal layering is secondary to per-bar latency.

## Development

Requires Python >= 3.11 and a Rust toolchain.

```sh
make install        # Rust toolchain + maturin + Python dev deps
make build          # build the Rust extension, install the package in-place
make test           # run the Python test suite
make coverage       # true cargo-test ∪ pytest line coverage (see scripts/coverage.sh)
make benchmark      # 3-way performance benchmark: pandas vs stock-pandas vs volas
make build-pkg      # build a release wheel + sdist into dist/
```

## License

[MIT](LICENSE)
