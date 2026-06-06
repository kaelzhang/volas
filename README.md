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
- **Fastest in the field.** Quicker than pandas, polars and TA-Lib on
  nearly every indicator — and faster than pandas even off the trading desk.
  ([benchmark](benchmark-report.html))
  - Beats TA-Lib on **129 / 158** covered indicators in batch computation.
  - Refreshes indicators incrementally on each new bar — up to **~2.7×** faster
    than TA-Lib, and 10-140x faster than pandas.
- **Built for the live tick.** A new bar touches only the affected tail
  (`O(lookback)`, not `O(n)`); indicators refresh in microseconds, never a full
  recompute.
- **Rust inside, NumPy / Torch out.** Compiled kernels, zero pandas at runtime;
  `to_numpy()` feeds NumPy and `torch.Tensor` pipelines.

## Table of Content
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Cumulation and DatetimeIndex](#cumulation-and-datetimeindex)
- [TimeFrame](#timeframe)
- [Syntax of directive](#syntax-of-directive)
- [Built-in indicators](#built-in-indicators)
- [Indexing & selection](#indexing--selection)
- [Writing & assignment](#writing--assignment)
- [Reading CSV](#reading-csv)
- [Timezones](#timezones)
- [pandas interop](#pandas-interop)
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

## Usage

```py
from volas import (
    DataFrame, Series, read_csv, to_datetime, TimeFrame, Timestamp,
)
```

The sub-sections below follow volas's public surface in order: the `DataFrame`
class, then its instance methods, its static methods, the other classes, and the
top-level package functions — closing with the rest of the **pandas-compatible**
API that behaves exactly as it does in pandas. (A top-level name imported from
`volas`, such as `read_csv`, is written without a `volas.`
prefix.)

### DataFrame(data, time_frame=None, cumulators=None)

`DataFrame` has a **pandas-compatible API**, so if you are familiar with
`pandas.DataFrame`, you are already ready to use volas. Unlike pandas, volas is
backed by a Rust kernel and has no pandas runtime dependency.

```py
df = read_csv('stock.csv')
```

We can use `[]`, which is called **pandas indexing** (a.k.a.
`__getitem__` in python) to select out lower-dimensional slices. In addition to
indexing with `colname` (the column name of the `DataFrame`), we could also do
indexing by `directive`s.

```py
df[directive]                  # Gets a Series

df[[directive0, directive1]]   # Gets a DataFrame
```

We have an example to show the most basic indexing using `[directive]`

```py
df = DataFrame({
    'open' : ...,
    'high' : ...,
    'low'  : ...,
    'close': [5, 6, 7, 8, 9]
})

df['ma:2']

# 0    NaN
# 1    5.5
# 2    6.5
# 3    7.5
# 4    8.5
# Name: ma:2, dtype: float64
```

Which gets the 2-period simple moving average on column `"close"`.

#### Parameters

- **data** `dict[str, list | np.ndarray] | DataFrame` the column data — a dict
  mapping each column name to an equal-length list or NumPy array (float, int,
  bool, `datetime64` or string) — **or another volas `DataFrame`, which is then
  copied** (like `pandas.DataFrame(df)`). There is no `date_col` / `tz` /
  `date_unit` shortcut: attach a
  [`DatetimeIndex`](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html)
  the pandas way — parse a column with `to_datetime`, promote it with
  `set_index`, then tag a zone with `tz_localize` / `tz_convert`. See
  [Timezones](#timezones).
- **time_frame** `Optional[str | TimeFrame] = None` If set, makes this a
  **tf-aware** (cumulating) DataFrame at this bar interval: the given rows are
  taken as already-final bars at that frame, and later `append`s fold finer
  bars into the forming bar. Requires a `DatetimeIndex`. See
  [Cumulation and DatetimeIndex](#cumulation-and-datetimeindex).
- **cumulators** `Optional[dict[str, str]] = None` Per-column aggregator
  overrides used when folding (e.g. `{'amount': 'sum'}`); defaults to OHLCV
  semantics (`open`=first, `high`=max, `low`=min, `close`=last, `volume`=sum;
  any other column `last`). Only meaningful together with `time_frame`.

### df.exec(directive: str, create_column: bool = False) -> np.ndarray

Executes the given directive and returns a numpy ndarray according to the
directive.

```py
df['ma:5']  # returns a Series

df.exec('ma:5', create_column=True)  # returns a numpy ndarray
```

```py
# This will only calculate without creating a new column in the dataframe
df.exec('ma:20')
```

The difference between `df[directive]` and `df.exec(directive)` is that
- the former will create a new column for the result of `directive` as a cache
  for later use, while `df.exec(directive)` does not unless we pass the
  parameter `create_column` as `True`
- the former one accepts other pandas indexing targets, while
  `df.exec(directive)` only accepts a valid **volas** directive string
- the former one returns a `Series` or `DataFrame` object while the latter one
  returns an [`np.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)

### df.get_column(key: str) -> Series

Directly gets the column value by `key`, returning a `Series`.

If the given `key` is an alias name, it returns the value of the corresponding
original column. If the column is not found, a `KeyError` is raised.

```py
df = DataFrame({
    'open' : ...,
    'high' : ...,
    'low'  : ...,
    'close': [5, 6, 7, 8, 9]
})

df.get_column('close')
# 0    5
# 1    6
# 2    7
# 3    8
# 4    9
# Name: close, dtype: float64
```

### df.append(other: DataFrame | Row) -> DataFrame

Appends rows of `other` (a `DataFrame` or a `Row`) to the end of the caller,
returning a new object, and applies the `DatetimeIndex` to the newly-appended
row(s) if possible.

If the caller is a **tf-aware** DataFrame (one built with a `time_frame`, or
the result of `cumulate`), `append` instead **folds** each finer bar into the
forming bar rather than adding a row — see
[Live cumulation](#live-cumulation--a-tf-aware-dataframe).

By default, appending new rows does not update the indicator columns of the new
rows; they stay stale until they are read again or until `df.fulfill()` is
called (see below).

### df.cumulate(time_frame: TimeFrame | str, cumulators: dict | None = None) -> DataFrame

Cumulate (resample) the data frame to a coarser `time_frame`, returning a new
`DataFrame`. Requires a `DatetimeIndex`.

- **time_frame** `TimeFrame | str` the target bar interval, e.g. `TimeFrame.m5`
  or `'5m'`. See [TimeFrame](#timeframe).
- **cumulators?** `dict[str, str] | None = None` per-column aggregator overrides
  (e.g. `{'amount': 'sum'}`); defaults to OHLCV semantics (`open`=first,
  `high`=max, `low`=min, `close`=last, `volume`=sum; any other column `last`).

```py
# from 1-minute klines to 5-minute klines
five_minute = one_minute.cumulate('5m')
```

See [Cumulation and DatetimeIndex](#cumulation-and-datetimeindex) for details.

### df.fulfill() -> None

Fulfill all indicator columns. By default, adding new rows to a `DataFrame` will
not update the indicators of the new rows.

Indicators are only updated when accessing the indicator column or calling
`df.fulfill()`. Accessing `df[directive]` refreshes only the affected tail
incrementally (`O(lookback)`, not an `O(n)` recompute); for bulk reads
(`to_numpy()`, `.iloc`) call `fulfill()` once to batch-refresh every cached
directive column in place.

```py
df['ma:20']              # cache the 20-period SMA as a column
df = df.append(new_bar)  # the new row's ma:20 is stale (NaN)
df.fulfill()             # recompute only the tail of every cached column
df.to_numpy()            # now fresh
```

### df.alias(as_name: str, src_name: str) -> None

Defines a column alias.

- **as_name** `str` the alias name
- **src_name** `str` the name of an existing column

```py
# Some plot libraries such as `mplfinance` require a column named capitalized
# `Open`, but it is ok, we could create an alias.
df.alias('Open', 'open')
```

The alias resolves everywhere a column is looked up, **including inside
directives**, and survives `drop` / `copy` / slicing.

```py
df['Open']        # same data as df['open']
df['ma:5@Open']   # the alias resolves inside directives too
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

### Series

`df[col]` and `df[directive]` return a `Series` — a named 1-D column whose API is
pandas-compatible: arithmetic / comparison / logical operators, `.sum()` /
`.mean()` / `.std()` / …, `.shift()` / `.diff()` / `.fillna()`, `.iloc` /
`.loc`, `.to_numpy()` / `.to_list()`. See
[the rest of the pandas-compatible API](#the-rest-of-the-pandas-compatible-api)
for the full list. There is no public `Series` constructor — a `Series` is
always obtained by indexing a `DataFrame`.

```py
s = df['close']
s.name                 # 'close'
(s - s.shift(1)).mean()
df['ma:5 > ma:20']     # a directive likewise returns a Series (here a bool one)
```

Beyond pandas, a `Series` also exposes the 15 TA-Lib **Math Transform** functions
as methods — `acos` `asin` `atan` `ceil` `cos` `cosh` `exp` `floor` `ln`
`log10` `sin` `sinh` `sqrt` `tan` `tanh`:

```py
df['close'].ln()
df['high'].sqrt()
```

### Row

`df.iloc[i]` and `df.loc[label]` return a `Row` — a single record whose `.name`
is its index label. A `Row` has **no public constructor** (`Row(...)` raises
`TypeError: No constructor defined for Row`); you only obtain one by indexing a
frame, and you may pass it to `df.append`.

```py
row = df.iloc[-1]      # the latest bar
row.name               # its index label (e.g. a Timestamp for a DatetimeIndex)
row.to_dict()          # {column: value}
row.to_numpy()         # the numeric cells as a 1-D ndarray
```

### Live cumulation — a tf-aware DataFrame

For **live** streaming, give a DataFrame a `time_frame` and `append` finer bars
into it, instead of re-cumulating the whole frame each tick. `df.cumulate(tf)`
returns such a frame (the forming period kept live), or build one directly with
`DataFrame(data, time_frame=..., cumulators=...)` (the given rows are taken as
already-final bars at that frame; requires a DatetimeIndex).

On a tf-aware frame:

- **df.append(bar)** folds the bar in: one in the open period **updates the
  forming last row** (`df.iloc[-1]`); one in a new period rolls over into a fresh
  row; a re-sent forming bar (same timestamp) updates rather than double-counts.
- **df.iloc[-1]** is the current (still-open) period — the live bar.
- **df[directive]** / **df.exec(directive)** computes indicators over the
  cumulated frame including the forming row — lazily, on read: an `append` only
  marks them stale, and the next read recomputes just the tail.
- **df.cumulate(target)** must be a whole multiple of the source frame (e.g.
  `5m→15m`, not `5m→7m`; a week or 3-day bar does not nest into a month/year);
  the same frame is a `copy()`.

```py
df = history.cumulate('5m')   # a tf-aware 5m frame (history is finer, e.g. 1m)
for bar in stream:            # each `bar` is a finer DataFrame
    df.append(bar)            # folds into the forming 5m bar
    df.iloc[-1]               # the live, still-forming bar
    df['macd']               # indicators over the cumulated frame
```

See [Cumulation and DatetimeIndex](#cumulation-and-datetimeindex) for details.

### read_csv(path, ...) -> DataFrame

A top-level function that reads a CSV file into a `DataFrame`, inferring
per-column dtypes. Its full parameter list (`sep`, `parse_dates`, `index_col`,
`na_values`, `tz`, `date_unit`, …) is documented under [Reading CSV](#reading-csv).

### from_pandas(pdf) -> DataFrame

A top-level function that bridges a `pandas.DataFrame` (`pdf`) into volas (and
`df.to_pandas()` bridges back). See [pandas interop](#pandas-interop).

### to_datetime(obj, unit='ns') -> Series

A top-level function that converts epoch numbers or datetime strings to a
datetime `Series`, mirroring `pandas.to_datetime`. `obj` may be a `Series`, a 1-D
NumPy array, or a list.

- **obj** the values to convert — numeric epochs, datetime strings, or an
  already-datetime `Series` (returned unchanged).
- **unit?** `str = 'ns'` the epoch unit for **numeric** input (`'s'` / `'ms'` /
  `'us'` / `'ns'`); sub-unit fractions are preserved, like `pd.to_datetime`.

Naive strings parse as UTC and offset-aware strings (`…+08:00`) are absolute. To
*display* the resulting index in a zone, make it the index and tag the zone with
`tz_localize` / `tz_convert` (see [Timezones](#timezones)).

```py
from volas import to_datetime

# parse an epoch-seconds column to datetime, then make it the index
df['time'] = to_datetime(df['time'], unit='s')
df = df.set_index('time')                       # -> DatetimeIndex
df = df.tz_localize('America/New_York')         # tag the display zone (see Timezones)
```

For an in-place, **truncating** cast (the NumPy / pandas `astype` idiom), use
`df.astype({'time': 'datetime64[s]'})` instead.

### The rest of the pandas-compatible API

Everything below behaves like its `pandas` counterpart — if you know it from
pandas, it works the same in volas.

```py
# --- DataFrame: metadata --------------------------------------------------
df.columns / df.shape / len(df) / df.dtypes      # dtypes -> dict
df.index                          # row labels, as a NumPy array
col in df ; for col in df         # membership / iterate column names
df.tz / df.tz_localize(tz) / df.tz_convert(tz)   # DatetimeIndex tz; see Timezones

# --- DataFrame: selection -------------------------------------------------
df[col]                           # -> Series
df[[col, ...]]                    # -> DataFrame
df[bool_mask]                     # -> DataFrame (filter rows; mask = Series | ndarray)
df.iloc[...] / df.loc[...] / df.at[label, col] / df.iat[i, j]
df.head(n=5) / df.tail(n=5)

# --- DataFrame: reshaping & dtypes ----------------------------------------
df.drop([label, ...], axis=0)     # drop rows by label (axis=1 -> columns)
df.dropna(how='any') / df.sort_index(ascending=True) / df.reset_index(drop=False)
df.rename({old: new}) / df.astype({col: dtype}) / df.set_index(col)
df.astype({col: 'datetime64[s]'})  # numeric epoch -> datetime (unit s|ms|us|ns; truncating)
df.copy() / df.to_numpy(dtype=None) / df.equals(other) / df.to_csv(path=None, ...)

# --- DataFrame: writing ---------------------------------------------------
df[col] = scalar | array | Series          # add / replace a column (positional)
df.loc[mask, col] = value ; df.iloc[i, j] = value ; df.at[label, col] = value

# --- Series ---------------------------------------------------------------
s.name / s.dtype / len(s) / s.tz / s.index
s.to_numpy(dtype=None) / s.to_list()
s.iloc[...] / s.loc[...]
s + s, s - 1, -s, ...             # elementwise arithmetic
s > 0, s == t, s != t, ...        # comparison -> bool Series
s & t, s | t, ~s, s ^ t           # logical -> bool Series
s.sum() / s.mean() / s.min() / s.max() / s.std() / s.var() / s.median()   # NaN-skipping
s.shift(n=1) / s.diff(n=1) / s.fillna(v) / s.isna() / s.notna() / s.dropna() / s.equals(t)
```

The pandas-shaped indexing and writing details have their own sections —
[Indexing & selection](#indexing--selection) and
[Writing & assignment](#writing--assignment).

## Cumulation and DatetimeIndex

Suppose we have a csv file containing kline data of a stock in the 1-minute time
frame:

```py
csv = read_csv(csv_path)

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
df = read_csv(
    csv_path,
    parse_dates=['date'],
    index_col='date'
)

print(df)
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
df_5m = df.cumulate('5m')

print(df_5m)
```

Now we get a 5-minute kline:

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  334.2  324.8  324.8  62346461.0
2020-01-01 00:05:00  325.0  327.8  316.2  322.0  82176419.0
2020-01-01 00:10:00  323.0  327.8  314.6  327.6  74409815.0
2020-01-01 00:15:00  330.0  335.2  322.0  323.0  82452902.0
```

`cumulate` defaults to OHLCV semantics — `open`=first, `high`=max, `low`=min,
`close`=last, `volume`=sum — and **any other column falls back to `last`**. Pass
`cumulators=` to override a column's aggregator; the common case is a non-OHLCV
column that should be summed, such as a turnover (`amount`) column that would
otherwise default to `last`:

```py
df.cumulate('1h', cumulators={'amount': 'sum'})
```

The supported aggregators are `first`, `max`, `min`, `last` and `sum`.

The `time_frame` may be a string label or a `TimeFrame` constant — see
[TimeFrame](#timeframe) for the full list.

For **live** streaming you do not re-cumulate the whole history on every tick —
you keep the current 5-minute bar *forming* and update it as each finer bar
arrives. A **tf-aware DataFrame** does exactly that: it stays an ordinary
DataFrame (read columns, run directives, slice it), except `append` **folds**
each finer bar into the bar currently forming instead of adding a row. You make
one with `df.cumulate('5m')` or `DataFrame(data, time_frame='5m')`, and the live
loop is then just:

| step                           | call                      |
| ------------------------------ | ------------------------- |
| make a `5m` frame              | `cum = df.cumulate('5m')` |
| feed it the next finer bar     | `cum.append(bar)`         |
| read the current forming bar   | `cum.iloc[-1]`            |
| read an indicator over it      | `cum['macd']`             |

#### Watch the forming bar grow

Build the 5-minute frame from the 1-minute `df` above one bar at a time. Seed it
with the `00:00` bar, then fold in `00:01`. Both fall in the same `00:00`–`00:05`
window, so the frame still holds **one** row — the forming bar — now updated
(`high` rose to `332.0`, `close` to `331.0`, `volume` summed):

```py
cum = df.iloc[0:1].cumulate('5m')   # seed the 5m frame with the 00:00 bar
cum.append(df.iloc[1:2])            # fold in 00:01 (same 5m window)

print(cum)
```

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  332.0  327.6  331.0  28155710.0
```

Fold in `00:02`, `00:03` and `00:04` and the window fills up. That single forming
row is now the **finished** first 5-minute bar — identical to the first row of
the one-shot `df.cumulate('5m')` printed earlier:

```py
for i in range(2, 5):
    cum.append(df.iloc[i:i + 1])

print(cum)
```

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  334.2  324.8  324.8  62346461.0
```

Now fold in `00:05`. It opens the **next** window, so the `00:00` bar is finalized
and a fresh forming bar starts; the frame grows to two rows and `cum.iloc[-1]` is
the new, still-forming `00:05` bar:

```py
cum.append(df.iloc[5:6])

print(cum)
```

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  334.2  324.8  324.8  62346461.0   <- finalized
2020-01-01 00:05:00  325.0  327.8  324.8  327.6  10448427.0   <- still forming
```

Two properties make this safe for a live feed:

- **Indicators are lazy, and fresh on read.** `append` does not recompute
  anything — it only flags the dependent directive columns as stale (their
  valid-row cursor now lags the frame height). The recompute happens when you
  **read** `cum['ema:9']` (or any directive): only the stale tail is refreshed —
  `O(lookback)`, not the whole column — over the frame *including* the forming
  row, bit-identical to a one-shot cumulate-then-compute. (A bulk read such as
  `to_numpy()` does not auto-refresh; call `cum.fulfill()` first, or just read
  the directive.)
- **Re-sent bars do not double-count.** Folding a bar whose timestamp you have
  already seen **updates** that period instead of adding to it — the same dedup
  rule shown at the top of this section — matching exchanges that revise their
  most recent bar.

See [Live cumulation](#live-cumulation--a-tf-aware-dataframe) for the API summary.

## TimeFrame

A `TimeFrame` names a bar interval. It is accepted anywhere volas resamples —
`df.cumulate`, the `time_frame` DataFrame argument, and the `hv` indicator —
either as a `TimeFrame` constant or as its equivalent **string label**. There is no `TimeFrame(...)`
constructor — use one of the constants below or a label string.

```py
TimeFrame.m5            # the 5-minute frame
'5m'                    # the equivalent label string, accepted everywhere too

df.cumulate(TimeFrame.m5)     # identical to df.cumulate('5m')
```

Supported frames (constant ⇄ label):

| Constant | Label | Constant | Label | Constant | Label |
| --- | --- | --- | --- | --- | --- |
| `TimeFrame.s1` | `'1s'` | `TimeFrame.m30` | `'30m'` | `TimeFrame.H12` | `'12h'` |
| `TimeFrame.m1` | `'1m'` | `TimeFrame.H1` | `'1h'` | `TimeFrame.D1` | `'1d'` |
| `TimeFrame.m3` | `'3m'` | `TimeFrame.H2` | `'2h'` | `TimeFrame.D3` | `'3d'` |
| `TimeFrame.m5` | `'5m'` | `TimeFrame.H4` | `'4h'` | `TimeFrame.W1` | `'1w'` |
| `TimeFrame.m15` | `'15m'` | `TimeFrame.H6` | `'6h'` | `TimeFrame.M1` | `'1M'` |
| | | `TimeFrame.H8` | `'8h'` | `TimeFrame.Y1` | `'1y'` |

`tf.unify(ts)` snaps a timestamp to the start of its bar (used internally by
cumulation).

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
df['boll']

# kdj j less than 0
# This returns a series of bool type
df['kdj.j < 0']

# kdj %K cross up kdj %D
df['kdj.k // kdj.d']

# 5-period simple moving average
df['ma:5']

# 10-period simple moving average on (@) open prices
df['ma:10@open']

# A DataFrame of 5-period, 10-period and 30-period ma
df[[
    'ma:5',
    'ma:10',
    'ma:30'
]]

# Which means we use the default values of the first and the second parameters,
# and specify the third parameter (for macd.signal)
df['macd.signal:,,10']

# We must wrap a parameter which is a nested command or directive
df['increase:3@(ma:20@close)']

# volas has a powerful directive parser,
# so we could even write directives like this:
df['''
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
  which we call a "gold cross": `df['macd // macd.signal']`.
- `\\` — whether `left` **crosses down** through `right`, a "dead cross". In a
  Python string the backslash must be escaped, so we write `'macd \\ macd.signal'`.
- `><` — whether `left` crosses `right`, either up or down.
- `<` `<=` `==` `!=` `>=` `>` — for the same record, the value comparison between
  `left` and `right`, returning a `bool` series.
- arithmetic `+ - * /`, logical `& | ^`, and unary `~` (not) / `-` (negate).

`df[directive]` **caches** the result as a real column (so repeated reads are
free), then auto-refreshes its stale tail on access after an `append`. Use
`df.exec(directive)` to compute a directive as a NumPy array **without**
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
from volas import read_csv

df = read_csv('klines.csv')                       # RangeIndex
df = read_csv('klines.csv',
             parse_dates=['time_key'],            # parse to datetime
             index_col='time_key')                # -> DatetimeIndex
df = read_csv('data.tsv', sep='\t', header=False, # no header -> '0'..'n-1'
             na_values=['NA', 'null'])
```

## Timezones

Storage is always **UTC epoch-nanoseconds** — the universal axis on which crypto,
US, HK and A-share frames coexist and align on the absolute instant. A
`DatetimeIndex` additionally carries a **per-frame timezone** that governs how
those instants render, how bare-string labels match, and how `cumulate` aligns
day-and-coarser buckets. A timezone is either a **fixed offset** (`'+08:00'`,
cheap; crypto / A-share / HK) or a **named IANA zone** (`'America/New_York'`,
DST-aware via `chrono-tz`; US / EU). The default is UTC.

Here is the whole picture. There is **no constructor shortcut** — you build a
`DatetimeIndex` the pandas way: parse a column with `to_datetime`, promote it with
`set_index`, then tag the display zone with `tz_localize` (reinterpret a naive
wall-clock *as* that zone — the instant moves) or `tz_convert` (keep the instant,
restate the zone). A US exchange opens at 09:30 local on 2021-01-04, held as a
naive local string:

```py
from volas import DataFrame, to_datetime, Timestamp

# Parse the naive 't' strings to UTC instants and make them the index, then read
# the wall-clock *as New York local time* with tz_localize. The instant is stored
# UTC (14:30Z), but the index renders and matches in New York.
df = DataFrame({'t': ['2021-01-04 09:30:00'], 'close': [100.0]})
df['t'] = to_datetime(df['t'])
df = df.set_index('t').tz_localize('America/New_York')
df.tz       # 'America/New_York'
df.index    # ['2021-01-04T14:30:00.000000000']  (raw .index is UTC, matching pandas .values)

# The tz is what lets a bare local string match the right row — it is parsed in df.tz:
df.at['2021-01-04 09:30:00', 'close']   # 100.0

# A Timestamp is a typed, cross-tz label. The SAME instant in Shanghai is
# 22:30+08:00, and it still matches, regardless of df.tz:
ts = Timestamp('2021-01-04 22:30:00', tz='+08:00')   # == 09:30 New York
df.at[ts, 'close']                       # 100.0
ts.value                                 # its UTC epoch-nanoseconds (int)
ts.tz                                    # '+08:00'

# Integer epochs: to_datetime(unit=...) reads the unit. An epoch is *absolute*, so
# tag the zone with tz_convert (display only). 1609770600000 ms == 14:30Z:
e = DataFrame({'t': [1609770600000], 'close': [100.0]})
e['t'] = to_datetime(e['t'], unit='ms')
e.set_index('t').tz_convert('America/New_York').index
# ['2021-01-04T14:30:00.000000000']

# An offset-aware string is already absolute too — to_datetime resolves the offset:
o = DataFrame({'t': ['2021-01-04T09:30:00+08:00'], 'close': [1.0]})
o['t'] = to_datetime(o['t'])
o.set_index('t').index
# ['2021-01-04T01:30:00.000000000']  (09:30+08:00 == 01:30Z)
```

Once a frame carries a tz, you can re-interpret or re-display it:

```py
df.tz_localize('America/New_York')   # reinterpret the naive wall-clock (the instant moves)
df.tz_convert('+08:00')              # keep the instant, change only how it displays
```

`cumulate` to a daily (or coarser) bar aligns buckets to the frame's local
trading day — DST-aware for a named zone — while the raw `.index` numpy export
stays UTC (matching pandas `.values`).

## pandas interop

pandas is **not** a runtime dependency; these bridges import it lazily, only when
called, so `import volas` stays pandas-free.

```py
from volas import from_pandas

df = from_pandas(pandas_df)        # numeric/bool/datetime native; a (tz-aware) DatetimeIndex round-trips
pdf = df.to_pandas()               # -> pandas.DataFrame
df.to_csv('out.csv', index=True)   # subset of pandas to_csv; returns a str if path=None
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
