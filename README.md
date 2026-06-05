[![ci](https://github.com/kaelzhang/volas/actions/workflows/ci.yml/badge.svg)](https://github.com/kaelzhang/volas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kaelzhang/volas/branch/main/graph/badge.svg)](https://codecov.io/gh/kaelzhang/volas)
[![PyPI version](https://img.shields.io/pypi/v/volas.svg)](https://pypi.org/project/volas/)
[![Python versions](https://img.shields.io/pypi/pyversions/volas.svg)](https://pypi.org/project/volas/)

# volas

> High-performance, Rust-backed columnar kernel for stock / candlestick (OHLCV) time-series data.

**volas** is a Rust-powered, **pandas-compatible** `DataFrame` for candlestick
(OHLCV) data, with trading-indicator directives built in. Know pandas? You
already know to use volas.

The difference is speed that **volas** beats every solution in terms of indicator calculating.

## Why volas

- **Drop-in for pandas.** The same `.loc` / `.iloc` / `.at`, `read_csv`,
  `to_numpy` and resampling — change the import, keep your code. (See
  [what's not covered](#index-limitations-vs-pandas))
- **Fastest in the field.** Quicker than pandas, polars, TA-Lib and DuckDB on
  nearly every indicator — and faster than pandas even off the trading desk.
  ([benchmark](benchmark-report.html))
  - Beats TA-Lib on 112 / 144 covered indicators for first calculation.
  - 10-110x faster than pandas on incremental indicator calculation, 3-4x faster even than TA-Lib
- **Built for the live tick.** A new bar touches only the affected tail
  (`O(lookback)`, not `O(n)`); indicators refresh in microseconds, never a full
  recompute.
- **Rust inside, NumPy / Torch out.** Compiled kernels, zero pandas at runtime;
  `to_numpy()` feeds NumPy and `torch.Tensor` pipelines.

## Table of Content
- [Installation](#installation)
- [Quick start](#quick-start)
- [API at a glance](#api-at-a-glance)
- [Usage](#usage)
- [Syntax of directive](#syntax-of-directive)
- [Built-in indicators](#built-in-indicators)
- [Indexing & selection](#indexing--selection)
- [Writing & assignment](#writing--assignment)
- [Reading CSV](#reading-csv)
- [Timezones](#timezones)
- [pandas interop](#pandas-interop)
- [Cumulation and DatetimeIndex](#cumulation-and-datetimeindex)
- [Error handling](#error-handling)
- [Design notes & non-goals](#design-notes--non-goals)
- [Development](#development)

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
    DataFrame, Series, Row, Timestamp, TimeFrame, Cumulator,
    read_csv, from_pandas, rolling_calc,
    DirectiveError, DirectiveSyntaxError, DirectiveValueError,
)

# --- construction ---------------------------------------------------------
DataFrame(data: dict[str, list | np.ndarray],
          date_col=None, tz=None, date_unit=None)   # tz / date_unit: see Timezones
read_csv(path, sep=None, delimiter=None, header=True,
         parse_dates=None, index_col=None, na_values=None, keep_default_na=True,
         tz=None, date_unit=None)
from_pandas(pdf)                 # lazy bridge from a pandas.DataFrame
Timestamp(value, tz=None)        # typed, cross-tz datetime label -> UTC instant

# --- DataFrame ------------------------------------------------------------
df.columns / df.shape / len(df) / df.dtypes      # metadata (dtypes -> dict)
df.index / df.tz                 # row labels (ndarray); DatetimeIndex tz or None
col in df ; for col in df        # membership / iterate column names
df[col] / df[directive]          # -> Series
df[[col_or_directive, ...]]      # -> DataFrame
df[bool_mask]                    # -> DataFrame (filter rows; mask = Series | ndarray)
df[col] = scalar | array | Series          # add / replace a column (positional)
df.loc[mask, col] = value ; df.iloc[i, j] = value ; df.at[label, col] = value
df.get_column(name)              # -> Series (plain column, no directive parsing)
df.exec(directive, create_column=False)   # -> np.ndarray (compute without caching)
df.fulfill()                     # batch-refresh cached directive columns in place
df.append(other)                 # -> DataFrame (other: DataFrame | Row)
df.head(n=5) / df.tail(n=5)
df.drop([label, ...], axis=0)    # drop rows by index label (axis=1 -> columns)
df.dropna(how='any') / df.sort_index(ascending=True) / df.reset_index(drop=False)
df.rename({old: new})            # -> DataFrame
df.astype({col: dtype})          # -> DataFrame ('float'|'int'|'bool'|'str'|'datetime')
df.set_index(col)                # -> DataFrame (move a column into the row index)
df.tz_localize(tz) / df.tz_convert(tz)     # attach / change the DatetimeIndex tz
df.alias(new_name, src_name)     # add a column alias (in place)
df.cumulate(time_frame, cumulators=None)  # -> DataFrame (resample OHLCV)
df.copy() / df.to_numpy(dtype=None)
df.iloc[...] / df.loc[...] / df.at[label, col] / df.iat[i, j]
df.to_pandas() / df.to_csv(path=None, sep=',', index=True, header=True, ...)

# --- Series ---------------------------------------------------------------
s.name / s.dtype / len(s) / s.tz / s.to_numpy(dtype=None) / s.to_list()
s.iloc[...] / s.loc[...]
s + s, s - 1, -s, ...            # elementwise arithmetic
s > 0, s == t, s != t, ...       # comparison -> bool Series
s & t, s | t, ~s, s ^ t          # logical -> bool Series
s.sum() / s.mean() / s.min() / s.max() / s.std() / s.var() / s.median()   # NaN-skipping
s.shift(n=1) / s.diff(n=1) / s.fillna(v) / s.isna() / s.notna() / s.dropna()

# --- resampling -----------------------------------------------------------
TimeFrame.m5 / TimeFrame.H1 / ...   # or the labels '5m', '1h', '1d', ...
TimeFrame.m5.minutes                # 5
Cumulator(time_frame, cumulators=None)
cum.append(df) ; cum.frame ; cum.last

# --- misc -----------------------------------------------------------------
rolling_calc(values, window, apply, forward=False, fill=nan)  # -> np.ndarray
```

## Usage

```py
from volas import DataFrame

# or
import volas
```

### DataFrame

`DataFrame` has a **pandas-compatible API**, so if you are familiar with
`pandas.DataFrame`, you are already ready to use volas. Unlike pandas, volas is
backed by a Rust kernel and has no pandas runtime dependency.

```py
import volas
stock = volas.read_csv('stock.csv')
```

As we know, we could use `[]`, which is called **pandas indexing** (a.k.a.
`__getitem__` in python) to select out lower-dimensional slices. In addition to
indexing with `colname` (the column name of the `DataFrame`), we could also do
indexing by `directive`s.

```py
stock[directive]                  # Gets a volas.Series

stock[[directive0, directive1]]   # Gets a volas.DataFrame
```

We have an example to show the most basic indexing using `[directive]`

```py
stock = DataFrame({
    'open' : ...,
    'high' : ...,
    'low'  : ...,
    'close': [5, 6, 7, 8, 9]
})

stock['ma:2']

# 0    NaN
# 1    5.5
# 2    6.5
# 3    7.5
# 4    8.5
# Name: ma:2, dtype: float64
```

Which gets the 2-period simple moving average on column `"close"`.

#### Parameters

- **date_col** `Optional[str] = None` If set, the column named `date_col` is
  parsed and set as the [`DatetimeIndex`](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html)
  of the data frame.
- **tz** `Optional[str] = None` The timezone of the `DatetimeIndex` — a fixed
  offset such as `'+08:00'`, or a named IANA zone such as `'America/New_York'`.
  See [Timezones](#timezones).
- **date_unit** `Optional[str] = None` The unit of integer epoch timestamps
  (`'s'`, `'ms'`, `'us'`, `'ns'`). See [Timezones](#timezones).

### stock.exec(directive: str, create_column: bool = False) -> np.ndarray

Executes the given directive and returns a numpy ndarray according to the
directive.

```py
stock['ma:5']  # returns a Series

stock.exec('ma:5', create_column=True)  # returns a numpy ndarray
```

```py
# This will only calculate without creating a new column in the dataframe
stock.exec('ma:20')
```

The difference between `stock[directive]` and `stock.exec(directive)` is that
- the former will create a new column for the result of `directive` as a cache
  for later use, while `stock.exec(directive)` does not unless we pass the
  parameter `create_column` as `True`
- the former one accepts other pandas indexing targets, while
  `stock.exec(directive)` only accepts a valid **volas** directive string
- the former one returns a `volas.Series` or `volas.DataFrame` object while the
  latter one returns an [`np.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)

### stock.alias(as_name: str, src_name: str) -> None

Defines a column alias.

- **as_name** `str` the alias name
- **src_name** `str` the name of an existing column

```py
# Some plot libraries such as `mplfinance` require a column named capitalized
# `Open`, but it is ok, we could create an alias.
stock.alias('Open', 'open')
```

The alias resolves everywhere a column is looked up, **including inside
directives**, and survives `drop` / `copy` / slicing.

```py
stock['Open']        # same data as stock['open']
stock['ma:5@Open']   # the alias resolves inside directives too
```

### stock.get_column(key: str) -> Series

Directly gets the column value by `key`, returning a volas `Series`.

If the given `key` is an alias name, it returns the value of the corresponding
original column. If the column is not found, a `KeyError` is raised.

```py
stock = DataFrame({
    'open' : ...,
    'high' : ...,
    'low'  : ...,
    'close': [5, 6, 7, 8, 9]
})

stock.get_column('close')
# 0    5
# 1    6
# 2    7
# 3    8
# 4    9
# Name: close, dtype: float64
```

### stock.append(other) -> DataFrame

Appends rows of `other` (a `DataFrame` or a `Row`) to the end of the caller,
returning a new object, and applies the `DatetimeIndex` to the newly-appended
row(s) if possible.

By default, appending new rows does not update the indicator columns of the new
rows; they stay stale until they are read again or until `stock.fulfill()` is
called (see below).

### volas.rolling_calc(values, window, apply, forward=False, fill=nan) -> np.ndarray

Applies a 1-D function along `values` (a column / `Series` / array) over a
trailing (or forward) window — the escape hatch for indicators not expressible
as a directive.

- **values** the array / `Series` / column to roll over
- **window** `int` the size of the rolling window
- **apply** `Callable[[np.ndarray], Any]` the 1-D function to apply
- **forward?** `bool = False` whether to look forward (instead of backward) to
  form each rolling window
- **fill?** `Any = np.nan` the value used where there are not enough items to
  form a full window

```py
volas.rolling_calc(stock['high'], 5, max)

# Whose return value equals to
stock['hhv:5@high'].to_numpy()
```

### stock.cumulate(time_frame, cumulators=None) -> DataFrame

Cumulate (resample) the data frame to a coarser `time_frame`, returning a new
`DataFrame`. Requires a `DatetimeIndex`.

```py
# from 1-minute klines to 5-minute klines
five_minute = one_minute.cumulate('5m')
```

See [Cumulation and DatetimeIndex](#cumulation-and-datetimeindex) for details.

### volas.Cumulator(time_frame, cumulators=None)

For **live** streaming, feed bars to a `Cumulator` and read the running result,
instead of re-cumulating the whole frame each time.

```py
from volas import Cumulator

cum = Cumulator('5m')
for bar in stream:        # each `bar` is a 1-row DataFrame
    cum.append(bar)
    cum.frame             # closed periods + the open period as a live last row
    cum.last              # just the current (still-open) period, aggregated
```

See [Cumulation and DatetimeIndex](#cumulation-and-datetimeindex) for details.

### stock.fulfill() -> None

Fulfill all indicator columns. By default, adding new rows to a `DataFrame` will
not update the indicators of the new rows.

Indicators are only updated when accessing the indicator column or calling
`stock.fulfill()`. Accessing `stock[directive]` refreshes only the affected tail
incrementally (`O(lookback)`, not an `O(n)` recompute); for bulk reads
(`to_numpy()`, `.iloc`) call `fulfill()` once to batch-refresh every cached
directive column in place.

```py
stock['ma:20']                 # cache the 20-period SMA as a column
stock = stock.append(new_bar)  # the new row's ma:20 is stale (NaN)
stock.fulfill()                # recompute only the tail of every cached column
stock.to_numpy()               # now fresh
```

### DataFrame.directive_stringify(directive: str) -> str

The staticmethod to get the canonical full name of the `directive`, which is also
the actual column name of the data frame. The command name is lowercased and the
default arguments / series are omitted to save space.

```py
DataFrame.directive_stringify('kdj.j')
# 'kdj.j'

DataFrame.directive_stringify('kdj.j:9,3,2,100@high,close,close')
# 'kdj.j:,,2,100@,close'

# command names are case-insensitive and canonicalize to lowercase
DataFrame.directive_stringify('MACD:12,26')
# 'macd'
```

### DataFrame.directive_lookback(directive: str) -> int

The staticmethod to get the lookback period of a `directive`, which is the
minimum number of prior data points required before the indicator produces a
valid result.

```py
DataFrame.directive_lookback('ma:20')
# 19

DataFrame.directive_lookback('boll')
# 19 (default period 20)

# Compound directive: lookback accumulates across nested expressions.
# repeat:5 needs 4 extra points, boll.upper (period 20) needs 19 -> 23
DataFrame.directive_lookback('repeat:5@(close > boll.upper)')
# 23
```

## Syntax of `directive`

```
command . sub : args @ series  op  command ...
   |      |     |      |
   |      |     |      └── operand column / sub-expression  (e.g. @open, @(boll))
   |      |     └── comma-separated arguments               (e.g. ma:20, kdj.k:9,3)
   |      └── sub-command                                   (e.g. macd.signal)
   └── indicator name                                       (e.g. ma, macd, boll)
```

#### `directive` Example

Here lists several use cases of column names

```py
# The middle band of bollinger bands
#   which is actually a 20-period (default) moving average
stock['boll']

# kdj j less than 0
# This returns a series of bool type
stock['kdj.j < 0']

# kdj %K cross up kdj %D
stock['kdj.k // kdj.d']

# 5-period simple moving average
stock['ma:5']

# 10-period simple moving average on (@) open prices
stock['ma:10@open']

# A DataFrame of 5-period, 10-period and 30-period ma
stock[[
    'ma:5',
    'ma:10',
    'ma:30'
]]

# Which means we use the default values of the first and the second parameters,
# and specify the third parameter (for macd.signal)
stock['macd.signal:,,10']

# We must wrap a parameter which is a nested command or directive
stock['increase:3@(ma:20@close)']

# volas has a powerful directive parser,
# so we could even write directives like this:
stock['''
repeat
    :   5
    @   (
            close > boll.upper
        )
''']
```

#### Operators

```
left operator right
```

- `//` — whether `left` **crosses up** through `right` (from below to above),
  which we call a "gold cross": `stock['macd // macd.signal']`.
- `\\` — whether `left` **crosses down** through `right`, a "dead cross". In a
  Python string the backslash must be escaped, so we write `'macd \\ macd.signal'`.
- `><` — whether `left` crosses `right`, either up or down.
- `<` `<=` `==` `!=` `>=` `>` — for the same record, the value comparison between
  `left` and `right`, returning a `bool` series.
- arithmetic `+ - * /`, logical `& | ^`, and unary `~` (not) / `-` (negate).

`stock[directive]` **caches** the result as a real column (so repeated reads are
free), then auto-refreshes its stale tail on access after an `append`. Use
`stock.exec(directive)` to compute a directive as a NumPy array **without**
caching it (see [Usage](#usage)).

## Built-in indicators

volas implements every [TA-Lib](https://ta-lib.org) 0.6.4 function (all 10 groups,
including all 61 candlestick patterns), each verified 1:1 against the `talib`
package, plus a handful of extra OHLCV indicators. Directive names are lowercase
and case-insensitive; multi-output indicators expose each line as a sub-command
(`macd.signal`, `boll.upper`). Names mirror TA-Lib (e.g. `ht_dcperiod`); where a
common alternative name exists, both spellings are accepted.

**Overlap studies (trend / moving averages)**

| Directive | Indicator | Example |
| --- | --- | --- |
| `ma` | Moving average — optional MA type `0..8` (SMA/EMA/WMA/DEMA/TEMA/TRIMA/KAMA/MAMA/T3) | `ma:20`, `ma:10,1` |
| `ema` / `smma` | Exponential / smoothed (Wilder) MA | `ema:12`, `smma:7` |
| `wma` `dema` `tema` `trima` `kama` `t3` | Weighted / (triple-)double-exp / triangular / adaptive / T3 MA | `dema:30`, `t3:5,0.7` |
| `mama` | MESA adaptive MA (`.fama`) | `mama`, `mama.fama:0.5,0.05` |
| `mavp` | MA with a variable per-row period | `mavp:2,30@,periods` |
| `midpoint` / `midprice` | Midpoint of value / of high-low | `midpoint:14`, `midprice:14` |
| `bbi` | Bull and Bear Index | `bbi` |
| `boll` / `bbw` | Bollinger Bands (`.upper`/`.lower`) / band width | `boll.upper:20,2`, `bbw` |
| `accbands` | Acceleration Bands (`.upper`/`.lower`) | `accbands:20` |
| `sar` / `sarext` | Parabolic SAR / extended SAR | `sar:0.02,0.2`, `sarext` |
| `ht_trendline` | Hilbert Transform — instantaneous trendline | `ht_trendline` |

**Momentum**

| Directive | Indicator | Example |
| --- | --- | --- |
| `macd` / `macdext` / `macdfix` | MACD (`.signal`/`.dea`, `.histogram`) / per-line MA types / fixed 12-26 | `macd.signal`, `macdfix` |
| `rsi` `cmo` `cci` `mfi` `bop` `willr` | RSI / Chande momentum / CCI / Money Flow / Balance of Power / Williams %R | `rsi:14`, `cci:14` |
| `mom` `roc` `rocp` `rocr` `rocr100` | Momentum / rate-of-change family | `roc:10` |
| `apo` / `ppo` | Absolute / percentage price oscillator | `ppo:12,26,0` |
| `stoch` / `stochf` / `stochrsi` | Stochastic (slow/fast) / Stochastic RSI (`.k`/`.d`) | `stoch.k`, `stochrsi.d` |
| `trix` `ultosc` `imi` | TRIX / Ultimate Oscillator / Intraday Momentum Index | `trix:30`, `ultosc` |
| `aroon` / `aroonosc` | Aroon (`.up`/`.down`) / Aroon oscillator | `aroon.up:14` |
| `plus_di` `minus_di` `plus_dm` `minus_dm` `dx` `adx` `adxr` | Directional movement system | `adx:14`, `plus_di:14` |

**Volume · Volatility · Price transform**

| Directive | Indicator | Example |
| --- | --- | --- |
| `obv` `ad` `adosc` | On-Balance Volume / Chaikin A/D line / A/D oscillator | `adosc:3,10` |
| `tr` `atr` `natr` | (Normalized) (Average) True Range | `atr:14`, `natr:14` |
| `avgprice` `medprice` `typprice` `wclprice` | Average / median / typical / weighted-close price | `typprice` |

**Cycle (Hilbert Transform)**

| Directive | Indicator | Example |
| --- | --- | --- |
| `ht_dcperiod` / `ht_dcphase` | Dominant cycle period / phase | `ht_dcperiod` |
| `ht_phasor` / `ht_sine` | Phasor (`.quadrature`) / sine wave (`.leadsine`) | `ht_sine.leadsine` |
| `ht_trendmode` | Trend (1) vs cycle (0) mode | `ht_trendmode` |

**Statistic functions**

| Directive | Indicator | Example |
| --- | --- | --- |
| `linearreg` (`_slope`/`_intercept`/`_angle`) / `tsf` | Linear regression / time-series forecast | `linearreg:14`, `tsf:14` |
| `var` `stddev` `correl` `beta` | Variance / std-dev / Pearson correlation / beta | `correl:30@high,low` |
| `sum` `maxindex` `minindex` `minmax` `minmaxindex` | Rolling sum / arg-extrema / extrema (`.min`/`.max`) | `sum:30`, `minmax.max:30` |

`PySeries` also exposes the 15 Math Transform functions (`acos`…`tanh`) as methods.

**Pattern recognition** — all 61 TA-Lib candlesticks via `style.<name>` (alias
`cdl.<name>`), output `-100`/`0`/`+100` (some `±80`/`±200`). Patterns taking a
penetration ratio accept it as an arg (e.g. `style.morningstar:0.3`):

`2crows` `3blackcrows` `3inside` `3linestrike` `3outside` `3starsinsouth`
`3whitesoldiers` `abandonedbaby` `advanceblock` `belthold` `breakaway`
`closingmarubozu` `concealbabyswall` `counterattack` `darkcloudcover` `doji`
`dojistar` `dragonflydoji` `engulfing` `eveningdojistar` `eveningstar`
`gapsidesidewhite` `gravestonedoji` `hammer` `hangingman` `harami` `haramicross`
`highwave` `hikkake` `hikkakemod` `homingpigeon` `identical3crows` `inneck`
`invertedhammer` `kicking` `kickingbylength` `ladderbottom` `longleggeddoji`
`longline` `marubozu` `matchinglow` `mathold` `morningdojistar` `morningstar`
`onneck` `piercing` `rickshawman` `risefall3methods` `separatinglines`
`shootingstar` `shortline` `spinningtop` `stalledpattern` `sticksandwich` `takuri`
`tasukigap` `thrusting` `tristar` `unique3river` `upsidegap2crows` `xsidegap3methods`

**Extras beyond TA-Lib**

| Directive | Indicator | Example |
| --- | --- | --- |
| `rsv` / `kdj` | Raw stochastic value / KDJ (`.k`/`.d`/`.j`) | `rsv:9`, `kdj.j` |
| `llv` / `hhv` | Lowest-low / highest-high value | `llv:10`, `hhv:10@high` |
| `donchian` | Donchian channel (`.upper`/`.lower`) | `donchian:20` |
| `hv` | Historical volatility | `hv:20,1d,252` |
| `change` | Percentage change over N bars | `change:2` |
| `increase` | Monotonic increase/decrease over N bars | `increase:3@close` |
| `style` | Candle color (`bullish` / `bearish`) | `style:bullish` |
| `repeat` | A boolean condition holding N bars in a row | `repeat:2@(style:bullish)` |

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

## Writing & assignment

Assign a whole column, or write into a positional / label / boolean selection
(copy-on-write under the hood). Series assignment is **positional** (by row order,
not index-aligned).

```py
df['signal'] = 0.0                      # add / replace a column (scalar | array | Series)
df.iat[3, 0] = 99.0                     # one cell by position
df.at[label, 'close'] = 99.0            # one cell by label + column
df.iloc[10:20, 0] = 0.0                 # a column slice
df.loc[df['close'] > df['open'], 'signal'] = 1.0   # masked column assignment
```

Writing a fractional value into an integer column widens it to float (pandas
semantics). Writing into a cached directive column drops its cached status, so a
later `fulfill()` can never silently overwrite your edit.

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

## Timezones

Storage is always **UTC epoch-ns** — the universal axis on which crypto, US, HK
and A-share frames coexist and align on the absolute instant. A `DatetimeIndex`
additionally carries a **per-frame timezone** that governs how those instants
render, how bare-string labels match, and how `cumulate` aligns day+ buckets. A
tz is either a **fixed offset** (cheap; crypto / A-share / HK) or a **named IANA
zone** (DST-aware via `chrono-tz`; US / EU). The default is UTC.

```py
# ingest naive local strings in a zone (stored UTC, index tagged):
df = volas.DataFrame({'t': ['2021-01-04 09:30:00'], 'close': [100.0]},
                     date_col='t', tz='America/New_York')   # or '+08:00'
df.tz                                  # 'America/New_York'

# offset-aware strings and epoch integers are absolute:
volas.DataFrame({'t': ['2021-01-04T09:30:00+08:00'], 'c': [1.0]}, date_col='t')
volas.read_csv('klines.csv', index_col='ts', date_unit='ms')   # epoch milliseconds

# attach / change a tz after the fact:
df.tz_localize('America/New_York')     # reinterpret the wall-clock (instant moves)
df.tz_convert('+08:00')                # keep the instant, change display

# Timestamp is a typed, cross-tz label that matches on the absolute instant:
ts = volas.Timestamp('2021-01-04 22:30:00', tz='+08:00')   # == 09:30 New York
df.at[ts, 'close']                     # matches the right row regardless of df.tz
```

`cumulate` to a daily (or coarser) bar aligns buckets to the frame's local
trading day — DST-aware for a named zone — while the raw `.index` numpy export
stays UTC (matching pandas `.values`).

## pandas interop

pandas is **not** a runtime dependency; these bridges import it lazily, only when
called, so `import volas` stays pandas-free.

```py
df = volas.from_pandas(pandas_df)      # numeric/bool native; a DatetimeIndex round-trips
pdf = df.to_pandas()                   # -> pandas.DataFrame
df.to_csv('out.csv', index=True)       # subset of pandas to_csv; returns a str if path=None
```

## Cumulation and DatetimeIndex

Suppose we have a csv file containing kline data of a stock in the 1-minute time
frame:

```py
csv = volas.read_csv(csv_path)

print(csv)
```

```
                   date   open   high    low  close    volume
0   2020-01-01 00:00:00  329.4  331.6  327.6  328.8  14202519
1   2020-01-01 00:01:00  330.0  332.0  328.0  331.0  13953191
2   2020-01-01 00:02:00  332.8  332.8  328.4  331.0  10339120
3   2020-01-01 00:03:00  332.0  334.2  330.2  331.0   9904468
4   2020-01-01 00:04:00  329.6  330.2  324.9  324.9  13947162
5   2020-01-01 00:04:00  329.6  330.2  324.8  324.8  13947163    <- an update of
                                                                    2020-01-01 00:04:00
...
19  2020-01-01 00:19:00  327.0  327.2  322.0  323.0  15086985
```

> Note that duplicated records of the same timestamp are not cumulated. All
> records except the latest one are discarded.

Read the same csv, but parse the `date` column into a `DatetimeIndex`:

```py
stock = volas.read_csv(
    csv_path,
    parse_dates=['date'],
    index_col='date'
)

print(stock)
```

```
                      open   high    low  close    volume
2020-01-01 00:00:00  329.4  331.6  327.6  328.8  14202519
2020-01-01 00:01:00  330.0  332.0  328.0  331.0  13953191
...
2020-01-01 00:19:00  327.0  327.2  322.0  323.0  15086985
```

You must have figured it out that the data frame now has a
[`DatetimeIndex`](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html).

But it will not become a 5-minute kline unless we cumulate it:

```py
stock_5m = stock.cumulate('5m')

print(stock_5m)
```

Now we get a 5-minute kline:

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  334.2  324.8  324.8  62346461.0
2020-01-01 00:05:00  325.0  327.8  316.2  322.0  82176419.0
2020-01-01 00:10:00  323.0  327.8  314.6  327.6  74409815.0
2020-01-01 00:15:00  330.0  335.2  322.0  323.0  82452902.0
```

`cumulate` defaults to OHLCV semantics (`open`=first, `high`=max, `low`=min,
`close`=last, `volume`=sum); pass `cumulators=` to override an aggregator:

```py
stock.cumulate('1h', cumulators={'volume': 'last'})
```

Time frames: `'1s' '1m' '3m' '5m' '15m' '30m' '1h' '2h' '4h' '6h' '8h' '12h'
'1d' '3d' '1w' '1M' '1y'` (or the `TimeFrame.s1 / m1 / m5 / H1 / D1 / W1 / M1 /
Y1` constants).

For **live** streaming, feed bars to a `Cumulator` and read the running result,
instead of re-cumulating the whole frame each time:

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
- **pandas-independent at runtime.** pandas and TA-Lib are used only as test
  oracles (1:1 parity tests and the benchmark), never imported at runtime.
- **External API cleanliness first.** The Python surface is kept clean and
  pandas-shaped; internal layering is secondary to per-bar latency.

## Development

Requires Python >= 3.11 and a Rust toolchain.

```sh
make install        # Rust toolchain + maturin + Python dev deps
make build          # build the Rust extension, install the package in-place
make test           # run the Python test suite
make coverage       # true cargo-test ∪ pytest line coverage (see scripts/coverage.sh)
make benchmark      # multi-library benchmark: pandas / polars / TA-Lib / DuckDB / volas
make build-pkg      # build a release wheel + sdist into dist/
```

### Dependency groups

- **`dev`** (`pip install -e .[dev]`) — everything the test suite needs; this is all
  CI installs. It includes pandas because the *parity tests* use it as an oracle
  (test-time only — volas has no pandas runtime dependency).
- **`benchmark`** (`pip install -e .[benchmark]`) — the extra comparison libraries
  (polars, TA-Lib, DuckDB) used *only* by the benchmark. `make benchmark` installs
  `.[dev,benchmark]`; a library that is only needed to benchmark, never to test,
  belongs here so CI test runs stay lean.

### Benchmark & web report

`make benchmark` times every candidate on two workloads — **batch** indicator
computation and the incremental **append-one-bar** path — and prints a table. Add
`WEB_REPORT=1` to also (re)generate a self-contained HTML report:

```sh
make benchmark WEB_REPORT=1     # writes ./benchmark-report.html (overwrites)
```

[`benchmark-report.html`](benchmark-report.html) (committed) has, per indicator, a
bar chart of median time (shorter = faster) and a table of `Mean / Median / OPS /
rounds / Perf`, where `Perf` shows the slowest candidate as `1.00×` and each faster
candidate as `{slowest ÷ this}×`. A comparison library that is not installed is
simply omitted; DuckDB only appears for the non-recursive indicators a SQL window
can express. TA-Lib needs the TA-Lib C library on the system.

## License

[MIT](LICENSE)
