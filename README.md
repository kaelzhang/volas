[![ci](https://github.com/kaelzhang/volas/actions/workflows/ci.yml/badge.svg)](https://github.com/kaelzhang/volas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kaelzhang/volas/branch/main/graph/badge.svg)](https://codecov.io/gh/kaelzhang/volas)
[![PyPI version](https://img.shields.io/pypi/v/volas.svg)](https://pypi.org/project/volas/)
[![Python versions](https://img.shields.io/pypi/pyversions/volas.svg)](https://pypi.org/project/volas/)

# [volas](https://github.com/kaelzhang/volas)

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
  - Beats TA-Lib on **153 / 158** covered indicators in batch computation.
  - Refreshes indicators incrementally on each new bar — up to **~5×** faster
    than TA-Lib, and up to **~200x** faster than pandas.
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
- [Indexing & selection](#indexing--selection)
- [Writing & assignment](#writing--assignment)
- [Timezones](#timezones)
- [pandas interop](#pandas-interop)
- [Error handling](#error-handling)
- [Built-in Indicators](#built-in-indicators)
- [License](#license)
- [For Developers](#for-developers)

## Installation

```sh
pip install volas
```

Requires Python >= 3.11. Wheels are published for Linux (x86_64 / aarch64),
macOS (x86_64 / arm64) and Windows (x86_64). For a local build from source, see
[For Developers](#for-developers).

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

### DataFrame(data, columns=None, time_frame=None, cumulators=None)

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
  copied** (like `pandas.DataFrame(df)`). To attach a
  [`DatetimeIndex`](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html),
  parse a column with `to_datetime`, promote it with `set_index`, then tag a zone
  with `tz_localize` / `tz_convert`. See [Timezones](#timezones).
- **columns** `Optional[list[str]] = None` Select and order the columns to keep —
  the same projection as `df[[...]]`. A name not present raises `KeyError`; an empty
  list or a duplicate name is rejected, and an absent column is never NaN-filled.
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

### read_csv(path, sep=',', header=True, parse_dates=None, index_col=None, na_values=None, keep_default_na=True, tz=None, date_unit=None) -> DataFrame

A top-level function that reads a CSV file into a `DataFrame`, inferring per-column
dtypes — a fast, pandas-subset CSV reader.

- **path** `str` the CSV file path.
- **sep?** `str = ','` the field delimiter (a single character); `delimiter` is an
  accepted alias.
- **header?** `bool = True` `True` (or omitted) treats the first row as the header;
  `False` / `None` means no header (columns are named `'0'`…`'n-1'`).
- **parse_dates?** `list[str] | None = None` column names to parse into datetime
  columns.
- **index_col?** `str | int | None = None` a column name or integer position to move
  into the row index; applied *after* `parse_dates`, so naming a parsed date column
  yields a `DatetimeIndex`.
- **na_values?** `str | list[str] | None = None` extra missing-value tokens.
- **keep_default_na?** `bool = True` also treat the default NA tokens as missing.
- **tz?** `str | None = None` the timezone for the `index_col` datetime: a *naive*
  date string is read in `tz` (stored UTC, the index tagged). Accepts a fixed offset
  (`'+08:00'`) or an IANA name (`'America/New_York'`); pass the date column via
  `index_col` and do *not* also list it in `parse_dates`. See [Timezones](#timezones).
- **date_unit?** `str | None = None` read `index_col` as an epoch integer in this unit
  (`'s'` / `'ms'` / `'us'` / `'ns'`, absolute UTC); `tz` then only sets the display zone.

```py
from volas import read_csv

df = read_csv('klines.csv')                        # RangeIndex
df = read_csv('klines.csv',
              parse_dates=['time_key'],            # parse to datetime
              index_col='time_key')                # -> DatetimeIndex
df = read_csv('data.tsv', sep='\t', header=False,  # no header -> '0'..'n-1'
              na_values=['NA', 'null'])
```

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

### directive_stringify(directive: str) -> str

Get the canonical full name of a `directive` — the actual column name volas caches
it under. The command name is lowercased and default arguments / series are dropped
to save space.

```py
from volas import directive_stringify

directive_stringify('kdj.j')
# 'kdj.j'

directive_stringify('kdj.j:9,3,2,100@high,close,close')
# 'kdj.j:,,2,100@,close'

# command names are case-insensitive and canonicalize to lowercase
directive_stringify('MACD:12,26')
# 'macd'
```

### directive_lookback(directive: str) -> int

Get the lookback period of a `directive` — the minimum number of prior data points
required before the indicator produces a valid result.

```py
from volas import directive_lookback

directive_lookback('ma:20')
# 19

directive_lookback('boll')
# 19 (default period 20)

# Compound directive: lookback accumulates across nested expressions.
# repeat:5 needs 4 extra points, boll.upper (period 20) needs 19 -> 23
directive_lookback('repeat:5@(close > boll.upper)')
# 23
```

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

## Timezones

Storage is always **UTC epoch-nanoseconds** — the universal axis on which crypto,
US, HK and A-share frames coexist and align on the absolute instant. A
`DatetimeIndex` additionally carries a **per-frame timezone** that governs how
those instants render, how bare-string labels match, and how `cumulate` aligns
day-and-coarser buckets. A timezone is either a **fixed offset** (`'+08:00'`,
cheap; crypto / A-share / HK) or a **named IANA zone** (`'America/New_York'`,
DST-aware via `chrono-tz`; US / EU). The default is UTC.

Here is the whole picture. Build a `DatetimeIndex` by parsing a column with
`to_datetime`, promoting it with `set_index`, then tagging the display zone with
`tz_localize` (reinterpret a naive wall-clock *as* that zone — the instant moves)
or `tz_convert` (keep the instant, restate the zone). A US exchange opens at 09:30
local on 2021-01-04, held as a naive local string:

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

# Built-in Indicators

Volas supports indicators in two groups. The first group is native to Volas or
inherits stock-pandas directive names; TA-Lib either has no equivalent or no
first-class function with the same directive name and OHLCV defaults. The second
group follows TA-Lib's function surface: directive names are lowercase,
arguments are positional, and multi-output indicators expose each line as a
sub-command such as `macd.signal`, `boll.upper`, or `ht_sine.leadsine`.

## Volas-exclusive indicators

These directives are implemented by Volas itself. Many of them follow the
stock-pandas directive vocabulary, with the examples adapted to `volas.DataFrame`.

### `smma`, Smoothed Moving Average

```
smma:<period>@<on>
```

Gets the `period`-period smoothed moving average on column or directive `on`.
`SMA` is often confused between simple moving average and smoothed moving
average, so Volas uses `ma` for simple moving average and `smma` for smoothed
moving average.

- **period** `int` (required)
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
# Equivalent to df['smma:5@close']
df['smma:5']

df['smma:10@open']
```

### `bbi`, Bull and Bear Index (多空指标)

```
bbi:<a>,<b>,<c>,<d>@<on>
```

Calculates BBI (Bull and Bear Index), which is the average of `ma:3`, `ma:6`,
`ma:12`, and `ma:24` by default.

- **a?** `int=3`
- **b?** `int=6`
- **c?** `int=12`
- **d?** `int=24`
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
# Uses default parameters
df['bbi']

# Custom parameters
df['bbi:5,10,20,30@close']
```

### `bbw`, Bollinger Band Width

```
bbw:<period>@<on>
```

Gets Bollinger Band Width for a series.

- **period?** `int=20`
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
# Bollinger band width
df['bbw']

# Equivalent definition
(df['boll.upper'] - df['boll.lower']) / df['boll']

# Or as a directive expression
df['(boll.upper - boll.lower) / boll']
```

### `rsv`, Raw Stochastic Value (未成熟随机值)

```
rsv:<period>@<high>,<low>,<close>
```

Calculates the raw stochastic value, which is often used to calculate KDJ.

- **period** `int` (required)
- **high?** `str='high'` The column name for high prices.
- **low?** `str='low'` The column name for low prices.
- **close?** `str='close'` The column name for close prices.

```py
# Uses default columns: high, low, close
df['rsv:9']

# Specify custom columns
df['rsv:9@high,low,close']
```

### `kdj`, A Variety of Stochastic Oscillator (随机指标)

KDJ is a variety of the [Stochastic Oscillator](https://en.wikipedia.org/wiki/Stochastic_oscillator)
indicator created by [Dr. George Lane](https://en.wikipedia.org/wiki/George_Lane_(technical_analyst)),
which follows the formula:

```
RSV = rsv(period_rsv)
%K = ewma(RSV, period_k, init_value)
%D = ewma(%K, period_d, init_value)
%J = 3 * %K - 2 * %D
```

The EWMA here is seeded by `init_value`. Trading software from different vendors
usually uses one of `0.0`, `50.0`, or `100.0` as the initial value; Volas defaults
to `50.0`.

```
kdj.k:<period_rsv>,<period_k>,<init_value>@<high>,<low>,<close>
kdj.d:<period_rsv>,<period_k>,<period_d>,<init_value>@<high>,<low>,<close>
kdj.j:<period_rsv>,<period_k>,<period_d>,<init_value>@<high>,<low>,<close>
```

- **period_rsv?** `int=9` The period for calculating RSV.
- **period_k?** `int=3` The period for smoothing RSV into %K.
- **period_d?** `int=3` The period for smoothing %K into %D.
- **init_value?** `float=50.0` The initial value for smoothing.
- **high?** `str='high'` The column name for high prices.
- **low?** `str='low'` The column name for low prices.
- **close?** `str='close'` The column name for close prices.

```py
# The %D series of KDJ
df['kdj.d']

# Equivalent to default parameters and columns
df['kdj.d:9,3,3,50@high,low,close']

# KDJ lines with custom periods
df[['kdj.k:9,9,50', 'kdj.d:9,9,9,50', 'kdj.j:9,9,9,50']]
```

### `llv`, Lowest of Low Values

```
llv:<period>@<on>
```

Gets the lowest value in N periods. By default, it reads the `low` column.

- **period** `int` (required)
- **on?** `str='low'` Which column or directive the calculation is based on.

```py
# The 10-period lowest low prices
df['llv:10']

# The 10-period lowest close prices
df['llv:10@close']
```

### `hhv`, Highest of High Values

```
hhv:<period>@<on>
```

Gets the highest value in N periods. By default, it reads the `high` column.

- **period** `int` (required)
- **on?** `str='high'` Which column or directive the calculation is based on.

```py
# The 10-period highest high prices
df['hhv:10']

# The 10-period highest close prices
df['hhv:10@close']
```

### `donchian`, Donchian Channels

```
donchian:<period>@<high>,<low>
donchian.upper:<period>@<high>
donchian.lower:<period>@<low>
```

Gets Donchian channels, the historical view of price volatility by charting a
security's highest and lowest prices over a set period.

- **period** `int` (required)
- **high?** `str='high'` The column to calculate highest high values.
- **low?** `str='low'` The column to calculate lowest low values.

```py
# Donchian middle channel with default columns
df['donchian:20']

# Donchian upper and lower channels
df['donchian.upper:20']
df['donchian.lower:20']

# Short aliases
df['donchian.u:20']
df['donchian.l:20']
```

### `hv`, Historical Volatility

```
hv:<period>,<time_frame>,<trading_days>@<on>
```

Gets historical volatility, the statistical measure of the dispersion of returns
for a security or index over a period of time.

- **period** `int` (required)
- **time_frame?** `str='1d'` Time frame such as `1m`, `15m`, `1h`, or `1d`.
- **trading_days?** `int=252` Trading days in a year; crypto workflows often use
  `365`.
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
# 10-period historical volatility for 15-minute data based on 365 yearly days
df['hv:10,15m,365']

# Uses default time_frame and trading_days
df['hv:10']
```

### `psy`, Psychological Line (心理线)

```
psy:<period>@<on>
```

The percentage of rising days (close above the previous close) over the last
`period` bars.

- **period?** `int=12`
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
df['psy']
df['psy:6']
```

### `dpo`, Detrended Price Oscillator

```
dpo:<period>@<on>
```

The price `period/2 + 1` bars ago minus the `period`-bar SMA, removing the trend
to expose shorter cycles.

- **period?** `int=20`
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
df['dpo']
df['dpo:10']
```

### `tsi`, True Strength Index

```
tsi:<long>,<short>@<on>
```

A double-EMA-smoothed momentum oscillator: `100 * EMA_short(EMA_long(Δclose)) /
EMA_short(EMA_long(|Δclose|))`.

- **long?** `int=25`
- **short?** `int=13`
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
df['tsi']
df['tsi:25,13']
```

### `kst`, Know Sure Thing

```
kst@<on>
```

Pring's momentum oscillator: a weighted sum of four SMA-smoothed rate-of-change
terms (ROC 10/15/20/30, smoothed by SMA 10/10/10/15, weighted 1/2/3/4).

- **on?** `str='close'` Which column or directive the calculation is based on.

```py
df['kst']
```

### `crsi`, Connors RSI

```
crsi:<rsi>,<streak>,<rank>@<on>
```

Connors' composite: the average of `rsi:<rsi>`, the RSI of the consecutive up /
down streak length, and the percent-rank of the 1-bar return over the last `rank`
bars.

- **rsi?** `int=3`
- **streak?** `int=2`
- **rank?** `int=100`
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
df['crsi']
df['crsi:3,2,100']
```

### `chop`, Choppiness Index

```
chop:<period>@<high>,<low>,<close>
```

How choppy versus trending the market is over `period` bars:
`100 * log10(sum(TR) / (HHV − LLV)) / log10(period)`. Higher is choppier.

- **period?** `int=14`
- **high? / low? / close?** `str` the input columns; default to the like-named frame columns.

```py
df['chop']
df['chop:14']
```

### `cmf`, Chaikin Money Flow

```
cmf:<period>@<high>,<low>,<close>,<volume>
```

The `period`-bar sum of money-flow volume divided by the sum of volume — positive
is buying pressure, negative is selling pressure.

- **period?** `int=20`
- **high? / low? / close? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['cmf']
df['cmf:20']
```

### `emv`, Ease of Movement

```
emv:<period>@<high>,<low>,<volume>
```

The `period`-bar SMA of price displacement per unit of volume (StockCharts' 1e8
volume scale) — how easily price moves.

- **period?** `int=14`
- **high? / low? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['emv']
df['emv:14']
```

### `efi`, Elder Force Index

```
efi:<period>@<close>,<volume>
```

`EMA_period(Δclose * volume)` — the force of a move, combining its direction, size,
and volume.

- **period?** `int=13`
- **close? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['efi']
df['efi:13']
```

### `pvt`, Price Volume Trend

```
pvt@<close>,<volume>
```

A cumulative volume line weighted by each bar's return:
`PVT += (Δclose / prev close) * volume`.

- **close? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['pvt']
```

### `nvi`, Negative Volume Index

```
nvi@<close>,<volume>
```

A cumulative line (base 1000) that compounds the return only on bars where volume
fell — tracking the "smart money" days.

- **close? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['nvi']
```

### `pvi`, Positive Volume Index

```
pvi@<close>,<volume>
```

A cumulative line (base 1000) that compounds the return only on bars where volume
rose — tracking the "crowd" days.

- **close? / volume?** `str` the input columns; default to the like-named frame columns.

```py
df['pvi']
```

### `mass_index`, Mass Index

```
mass_index:<period>@<high>,<low>
```

The `period`-bar sum of the 9-EMA / double-9-EMA ratio of the high−low range; a
range "bulge" can flag a coming reversal.

- **period?** `int=25`
- **high? / low?** `str` the input columns; default to the like-named frame columns.

```py
df['mass_index']
df['mass_index:25']
```

### `bias`, Bias Ratio (乖离率)

```
bias:<period>@<on>
```

The percentage deviation of the series from its `period`-bar SMA,
`(close − SMA) / SMA × 100`. This is the China-market name for `ppo:1,<period>,0`; the
classic triple is `bias:6`, `bias:12`, `bias:24`.

- **period?** `int=6`
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
df['bias']
df['bias:24']
```

### `dma`, Difference of Moving Average (平行线差)

```
dma:<fast>,<slow>@<on>
dma.ama:<fast>,<slow>,<m>@<on>
```

The DDD line is the difference of two SMAs, `SMA_fast − SMA_slow` — the China-market name for
`apo:<fast>,<slow>,0`. The AMA signal line is the `m`-bar SMA of the DDD line. `dma.ddd` is an
alias of the main `dma` line.

- **fast?** `int=10`
- **slow?** `int=50`
- **m?** `int=10` The AMA signal period (only on `dma.ama`).
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
# DDD difference line
df['dma']
# , which is equivalent to
df['dma.ddd']

# AMA signal line
df['dma.ama']
```

## Built-in Commands for Statistics

### `change`, Percentage Change

```
change:<period>@<on>
```

Percentage change between the current and a prior element on a certain series.
It computes the percentage change from the immediately previous element by
default, which is useful when comparing percentage change in a time series of
prices.

- **period?** `int=2` `2` means the start value and the end value of a two-period
  window are compared.
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
# Percentage change of the close column
df['change']

# Percentage change with a custom period
df['change:5@close']

# Percentage change of a nested directive
df['change@(ma:20)']
```

### `increase`, Consecutive Increase or Decrease

```
increase:<repeat>,<direction>@<on>
```

Gets a `bool` series where each item is `True` if the value of `on` increases in
the last `repeat` steps. Use `direction=-1` to detect repeated decreases.

- **repeat?** `int=1`
- **direction?** `int=1` `1` means increasing; `-1` means decreasing.
- **on?** `str='close'` Which column or directive the calculation is based on.

```py
# Whether the 20-period moving average has increased repeatedly for 3 bars
df['increase:3@(ma:20@close)']

# Whether close has decreased repeatedly for 5 bars
df['increase:5,-1@close']
```

### `style`, Candle Color

```
style.<style>@<open>,<close>
```

Gets a `bool` series indicating whether the candlestick of a period is of the
given style. This native form is for candle color only; TA-Lib candlestick
patterns are exposed as `cdl.<pattern>` in the table below.

- **style** `'bullish'` or `'bearish'` (required)
- **open?** `str='open'` The column name for open prices.
- **close?** `str='close'` The column name for close prices.

```py
# Uses default open and close columns
df['style.bullish']
df['style.bearish']

# Specify custom columns
df['style.bearish@open,close']
```

### `repeat`, Consecutive Boolean Condition

```
repeat:<repeat>@<bool_directive>
```

The `repeat` command first gets the result of `bool_directive`, then detects
whether `True` repeats for `repeat` consecutive periods.

- **repeat?** `int=1` Must be larger than `0`.
- **bool_directive** `str | (Directive)` A column name or a directive wrapped in
  parentheses.

```py
# Whether bullish candlesticks repeat for 3 periods
df['repeat:3@(style.bullish)']

# Repeat check on a directive expression
df['repeat:5@(close > ma:20)']
```

## TA-Lib-compatible directives

TA-Lib-related directives use lowercase Volas names, but the `TA-Lib original`
column below lists the upstream TA-Lib function they correspond to. Arguments
before `@` are positional; input series after `@` override the default columns.
Square brackets mean an argument has a default. Required arguments are written
without brackets. Empty argument slots keep earlier defaults, so
`macd.signal:,,5` means fast period `12`, slow period `26`, and signal period `5`.

`matype` follows TA-Lib's integer convention: `0=SMA`, `1=EMA`, `2=WMA`,
`3=DEMA`, `4=TEMA`, `5=TRIMA`, `6=KAMA`, `7=MAMA`, and `8=T3`. Multi-output
indicators also accept short aliases where documented by the parser, for example
`macd.s` for `macd.signal`, `boll.u` for `boll.upper`, `aroon.d` for
`aroon.down`, and `style.<pattern>` as an alias for `cdl.<pattern>`.

```py
# Ordinary defaulted positional arguments
df['macd.signal:12,26,9@close']

# Directive command names are case-insensitive; column names after @ stay as written.
df['RSI:14@close']

# Skip fast and slow defaults, override only the signal period
df['macd.signal:,,5']

# A directive with multiple input series
df['stoch.d:5,3,0,3,0@high,low,close']

# MAVP needs a second series for the variable period
df['mavp:2,30,0@close,periods']
```

The TA-Lib Math Transform group is exposed on `Series` rather than as directive
strings: `acos`, `asin`, `atan`, `ceil`, `cos`, `cosh`, `exp`, `floor`, `ln`,
`log10`, `sin`, `sinh`, `sqrt`, `tan`, and `tanh`.

| Volas directive | TA-Lib original | Meaning | Parameters |
| --- | --- | --- | --- |
| `ma` | `MA` | Generic moving average selected by MA type. | `:<period>[,<matype=0>]@series=close` |
| `ema` | `EMA` | Exponential moving average. | `:<period>@series=close` |
| `wma` | `WMA` | Weighted moving average. | `[:period=30]@series=close` |
| `dema` | `DEMA` | Double exponential moving average. | `[:period=30]@series=close` |
| `tema` | `TEMA` | Triple exponential moving average. | `[:period=30]@series=close` |
| `trima` | `TRIMA` | Triangular moving average. | `[:period=30]@series=close` |
| `kama` | `KAMA` | Kaufman adaptive moving average. | `[:period=30]@series=close` |
| `t3` | `T3` | T3 moving average. | `[:period=5,vfactor=0.7]@series=close` |
| `mama` | `MAMA` | MESA adaptive moving average main line. | `[:fast_limit=0.5,slow_limit=0.05]@series=close` |
| `mama.fama` | `MAMA` | Following adaptive moving average line. | `[:fast_limit=0.5,slow_limit=0.05]@series=close` |
| `mavp` | `MAVP` | Moving average with per-row variable periods. | `[:min=2,max=30,matype=0]@series=close,periods` |
| `sar` | `SAR` | Parabolic SAR. | `[:acceleration=0.02,maximum=0.2]@high,low` |
| `sarext` | `SAREXT` | Extended Parabolic SAR. | `[:start=0,offset=0,long_init=0.02,long_step=0.02,long_max=0.2,short_init=0.02,short_step=0.02,short_max=0.2]@high,low` |
| `boll` | `BBANDS` | Bollinger middle band. | `[:period=20]@series=close` |
| `boll.upper` | `BBANDS` | Bollinger upper band. | `[:period=20,times=2]@series=close` |
| `boll.lower` | `BBANDS` | Bollinger lower band. | `[:period=20,times=2]@series=close` |
| `accbands` | `ACCBANDS` | Acceleration Bands middle line. | `[:period=20]@series=close` |
| `accbands.upper` | `ACCBANDS` | Acceleration Bands upper line. | `[:period=20]@high,low` |
| `accbands.lower` | `ACCBANDS` | Acceleration Bands lower line. | `[:period=20]@high,low` |
| `midpoint` | `MIDPOINT` | Midpoint over a rolling period. | `[:period=14]@series=close` |
| `midprice` | `MIDPRICE` | Midpoint price over high and low. | `[:period=14]@high,low` |
| `ht_trendline` | `HT_TRENDLINE` | Hilbert Transform instantaneous trendline. | `@series=close` |
| `macd` | `MACD` | MACD line; Volas uses standalone EMA fast minus EMA slow. | `[:fast=12,slow=26]@series=close` |
| `macd.signal` | `MACD` | Signal line of the Volas MACD line. | `[:fast=12,slow=26,signal=9]@series=close` |
| `macd.histogram` | `MACD` | MACD histogram: line minus signal. | `[:fast=12,slow=26,signal=9]@series=close` |
| `macdext` | `MACDEXT` | MACD line with independent MA types. | `[:fast=12,fast_matype=0,slow=26,slow_matype=0]@series=close` |
| `macdext.signal` | `MACDEXT` | MACDEXT signal line. | `[:fast=12,fast_matype=0,slow=26,slow_matype=0,signal=9,signal_matype=0]@series=close` |
| `macdext.histogram` | `MACDEXT` | MACDEXT histogram. | `[:fast=12,fast_matype=0,slow=26,slow_matype=0,signal=9,signal_matype=0]@series=close` |
| `macdfix` | `MACDFIX` | Fixed 12/26 MACD line; Volas uses standalone EMA fast minus EMA slow. | `@series=close` |
| `macdfix.signal` | `MACDFIX` | Signal line of the Volas fixed 12/26 MACD line. | `[:signal=9]@series=close` |
| `macdfix.histogram` | `MACDFIX` | Histogram of the Volas fixed 12/26 MACD line. | `[:signal=9]@series=close` |
| `apo` | `APO` | Absolute price oscillator. | `[:fast=12,slow=26,matype=0]@series=close` |
| `ppo` | `PPO` | Percentage price oscillator. | `[:fast=12,slow=26,matype=0]@series=close` |
| `rsi` | `RSI` | Relative Strength Index. | `:<period>@series=close` |
| `cmo` | `CMO` | Chande Momentum Oscillator. | `[:period=14]@series=close` |
| `cci` | `CCI` | Commodity Channel Index. | `[:period=14]@high,low,close` |
| `imi` | `IMI` | Intraday Momentum Index. | `[:period=14]@open,close` |
| `mfi` | `MFI` | Money Flow Index. | `[:period=14]@high,low,close,volume` |
| `bop` | `BOP` | Balance of Power. | `@open,high,low,close` |
| `willr` | `WILLR` | Williams Percent Range. | `[:period=14]@high,low,close` |
| `mom` | `MOM` | Momentum. | `[:period=10]@series=close` |
| `roc` | `ROC` | Rate of change. | `[:period=10]@series=close` |
| `rocp` | `ROCP` | Rate of change percentage. | `[:period=10]@series=close` |
| `rocr` | `ROCR` | Rate of change ratio. | `[:period=10]@series=close` |
| `rocr100` | `ROCR100` | Rate of change ratio multiplied by 100. | `[:period=10]@series=close` |
| `stoch.k` | `STOCH` | Slow stochastic percent K. | `[:fastk=5,slowk=3,slowk_matype=0,slowd=3,slowd_matype=0]@high,low,close` |
| `stoch.d` | `STOCH` | Slow stochastic percent D. | `[:fastk=5,slowk=3,slowk_matype=0,slowd=3,slowd_matype=0]@high,low,close` |
| `stochf.k` | `STOCHF` | Fast stochastic percent K. | `[:fastk=5,fastd=3,fastd_matype=0]@high,low,close` |
| `stochf.d` | `STOCHF` | Fast stochastic percent D. | `[:fastk=5,fastd=3,fastd_matype=0]@high,low,close` |
| `stochrsi.k` | `STOCHRSI` | Fast stochastic RSI percent K. | `[:rsi=14,fastk=5,fastd=3,fastd_matype=0]@series=close` |
| `stochrsi.d` | `STOCHRSI` | Fast stochastic RSI percent D. | `[:rsi=14,fastk=5,fastd=3,fastd_matype=0]@series=close` |
| `trix` | `TRIX` | One-period ROC of a triple EMA. | `[:period=30]@series=close` |
| `ultosc` | `ULTOSC` | Ultimate Oscillator. | `[:short=7,medium=14,long=28]@high,low,close` |
| `aroon.up` | `AROON` | Aroon up line. | `[:period=14]@high,low` |
| `aroon.down` | `AROON` | Aroon down line. | `[:period=14]@high,low` |
| `aroonosc` | `AROONOSC` | Aroon oscillator. | `[:period=14]@high,low` |
| `plus_dm` | `PLUS_DM` | Plus directional movement. | `[:period=14]@high,low` |
| `minus_dm` | `MINUS_DM` | Minus directional movement. | `[:period=14]@high,low` |
| `plus_di` | `PLUS_DI` | Plus directional indicator. | `[:period=14]@high,low,close` |
| `minus_di` | `MINUS_DI` | Minus directional indicator. | `[:period=14]@high,low,close` |
| `dx` | `DX` | Directional Movement Index. | `[:period=14]@high,low,close` |
| `adx` | `ADX` | Average Directional Movement Index. | `[:period=14]@high,low,close` |
| `adxr` | `ADXR` | Average Directional Movement Index Rating. | `[:period=14]@high,low,close` |
| `obv` | `OBV` | On-Balance Volume. | `@close,volume` |
| `ad` | `AD` | Chaikin Accumulation Distribution line. | `@high,low,close,volume` |
| `adosc` | `ADOSC` | Chaikin Accumulation Distribution oscillator. | `[:fast=3,slow=10]@high,low,close,volume` |
| `tr` | `TRANGE` | True Range. | `@high,low,close` |
| `atr` | `ATR` | Average True Range. | `[:period=14]@high,low,close` |
| `natr` | `NATR` | Normalized Average True Range. | `[:period=14]@high,low,close` |
| `avgprice` | `AVGPRICE` | Average price. | `@open,high,low,close` |
| `medprice` | `MEDPRICE` | Median price. | `@high,low` |
| `typprice` | `TYPPRICE` | Typical price. | `@high,low,close` |
| `wclprice` | `WCLPRICE` | Weighted close price. | `@high,low,close` |
| `ht_dcperiod` | `HT_DCPERIOD` | Hilbert Transform dominant cycle period. | `@series=close` |
| `ht_dcphase` | `HT_DCPHASE` | Hilbert Transform dominant cycle phase. | `@series=close` |
| `ht_phasor` | `HT_PHASOR` | Hilbert Transform phasor in-phase line. | `@series=close` |
| `ht_phasor.quadrature` | `HT_PHASOR` | Hilbert Transform phasor quadrature line. | `@series=close` |
| `ht_sine` | `HT_SINE` | Hilbert Transform sine wave. | `@series=close` |
| `ht_sine.leadsine` | `HT_SINE` | Hilbert Transform lead sine wave. | `@series=close` |
| `ht_trendmode` | `HT_TRENDMODE` | Hilbert Transform trend versus cycle mode. | `@series=close` |
| `linearreg` | `LINEARREG` | Linear regression value. | `[:period=14]@series=close` |
| `linearreg_slope` | `LINEARREG_SLOPE` | Linear regression slope. | `[:period=14]@series=close` |
| `linearreg_intercept` | `LINEARREG_INTERCEPT` | Linear regression intercept. | `[:period=14]@series=close` |
| `linearreg_angle` | `LINEARREG_ANGLE` | Linear regression angle. | `[:period=14]@series=close` |
| `tsf` | `TSF` | Time Series Forecast. | `[:period=14]@series=close` |
| `var` | `VAR` | Variance. | `[:period=5]@series=close` |
| `stddev` | `STDDEV` | Standard deviation. | `[:period=5,nbdev=1]@series=close` |
| `correl` | `CORREL` | Pearson correlation coefficient. | `[:period=30]@series0=close,series1` |
| `beta` | `BETA` | Beta. | `[:period=5]@series0=close,series1` |
| `sum` | `SUM` | Rolling sum. | `[:period=30]@series=close` |
| `maxindex` | `MAXINDEX` | Index of the rolling maximum. | `[:period=30]@series=close` |
| `minindex` | `MININDEX` | Index of the rolling minimum. | `[:period=30]@series=close` |
| `minmax.min` | `MINMAX` | Rolling minimum from the MINMAX pair. | `[:period=30]@series=close` |
| `minmax.max` | `MINMAX` | Rolling maximum from the MINMAX pair. | `[:period=30]@series=close` |
| `minmaxindex.min` | `MINMAXINDEX` | Index of the rolling minimum from the pair. | `[:period=30]@series=close` |
| `minmaxindex.max` | `MINMAXINDEX` | Index of the rolling maximum from the pair. | `[:period=30]@series=close` |
| `cdl.2crows` | `CDL2CROWS` | Two Crows | `@open,high,low,close` |
| `cdl.3blackcrows` | `CDL3BLACKCROWS` | Three Black Crows | `@open,high,low,close` |
| `cdl.3inside` | `CDL3INSIDE` | Three Inside Up/Down | `@open,high,low,close` |
| `cdl.3linestrike` | `CDL3LINESTRIKE` | Three-Line Strike  | `@open,high,low,close` |
| `cdl.3outside` | `CDL3OUTSIDE` | Three Outside Up/Down | `@open,high,low,close` |
| `cdl.3starsinsouth` | `CDL3STARSINSOUTH` | Three Stars In The South | `@open,high,low,close` |
| `cdl.3whitesoldiers` | `CDL3WHITESOLDIERS` | Three Advancing White Soldiers | `@open,high,low,close` |
| `cdl.abandonedbaby` | `CDLABANDONEDBABY` | Abandoned Baby | `[:penetration=0.3]@open,high,low,close` |
| `cdl.advanceblock` | `CDLADVANCEBLOCK` | Advance Block | `@open,high,low,close` |
| `cdl.belthold` | `CDLBELTHOLD` | Belt-hold | `@open,high,low,close` |
| `cdl.breakaway` | `CDLBREAKAWAY` | Breakaway | `@open,high,low,close` |
| `cdl.closingmarubozu` | `CDLCLOSINGMARUBOZU` | Closing Marubozu | `@open,high,low,close` |
| `cdl.concealbabyswall` | `CDLCONCEALBABYSWALL` | Concealing Baby Swallow | `@open,high,low,close` |
| `cdl.counterattack` | `CDLCOUNTERATTACK` | Counterattack | `@open,high,low,close` |
| `cdl.darkcloudcover` | `CDLDARKCLOUDCOVER` | Dark Cloud Cover | `[:penetration=0.5]@open,high,low,close` |
| `cdl.doji` | `CDLDOJI` | Doji | `@open,high,low,close` |
| `cdl.dojistar` | `CDLDOJISTAR` | Doji Star | `@open,high,low,close` |
| `cdl.dragonflydoji` | `CDLDRAGONFLYDOJI` | Dragonfly Doji | `@open,high,low,close` |
| `cdl.engulfing` | `CDLENGULFING` | Engulfing Pattern | `@open,high,low,close` |
| `cdl.eveningdojistar` | `CDLEVENINGDOJISTAR` | Evening Doji Star | `[:penetration=0.3]@open,high,low,close` |
| `cdl.eveningstar` | `CDLEVENINGSTAR` | Evening Star | `[:penetration=0.3]@open,high,low,close` |
| `cdl.gapsidesidewhite` | `CDLGAPSIDESIDEWHITE` | Up/Down-gap side-by-side white lines | `@open,high,low,close` |
| `cdl.gravestonedoji` | `CDLGRAVESTONEDOJI` | Gravestone Doji | `@open,high,low,close` |
| `cdl.hammer` | `CDLHAMMER` | Hammer | `@open,high,low,close` |
| `cdl.hangingman` | `CDLHANGINGMAN` | Hanging Man | `@open,high,low,close` |
| `cdl.harami` | `CDLHARAMI` | Harami Pattern | `@open,high,low,close` |
| `cdl.haramicross` | `CDLHARAMICROSS` | Harami Cross Pattern | `@open,high,low,close` |
| `cdl.highwave` | `CDLHIGHWAVE` | High-Wave Candle | `@open,high,low,close` |
| `cdl.hikkake` | `CDLHIKKAKE` | Hikkake Pattern | `@open,high,low,close` |
| `cdl.hikkakemod` | `CDLHIKKAKEMOD` | Modified Hikkake Pattern | `@open,high,low,close` |
| `cdl.homingpigeon` | `CDLHOMINGPIGEON` | Homing Pigeon | `@open,high,low,close` |
| `cdl.identical3crows` | `CDLIDENTICAL3CROWS` | Identical Three Crows | `@open,high,low,close` |
| `cdl.inneck` | `CDLINNECK` | In-Neck Pattern | `@open,high,low,close` |
| `cdl.invertedhammer` | `CDLINVERTEDHAMMER` | Inverted Hammer | `@open,high,low,close` |
| `cdl.kicking` | `CDLKICKING` | Kicking | `@open,high,low,close` |
| `cdl.kickingbylength` | `CDLKICKINGBYLENGTH` | Kicking - bull/bear determined by the longer marubozu | `@open,high,low,close` |
| `cdl.ladderbottom` | `CDLLADDERBOTTOM` | Ladder Bottom | `@open,high,low,close` |
| `cdl.longleggeddoji` | `CDLLONGLEGGEDDOJI` | Long Legged Doji | `@open,high,low,close` |
| `cdl.longline` | `CDLLONGLINE` | Long Line Candle | `@open,high,low,close` |
| `cdl.marubozu` | `CDLMARUBOZU` | Marubozu | `@open,high,low,close` |
| `cdl.matchinglow` | `CDLMATCHINGLOW` | Matching Low | `@open,high,low,close` |
| `cdl.mathold` | `CDLMATHOLD` | Mat Hold | `[:penetration=0.5]@open,high,low,close` |
| `cdl.morningdojistar` | `CDLMORNINGDOJISTAR` | Morning Doji Star | `[:penetration=0.3]@open,high,low,close` |
| `cdl.morningstar` | `CDLMORNINGSTAR` | Morning Star | `[:penetration=0.3]@open,high,low,close` |
| `cdl.onneck` | `CDLONNECK` | On-Neck Pattern | `@open,high,low,close` |
| `cdl.piercing` | `CDLPIERCING` | Piercing Pattern | `@open,high,low,close` |
| `cdl.rickshawman` | `CDLRICKSHAWMAN` | Rickshaw Man | `@open,high,low,close` |
| `cdl.risefall3methods` | `CDLRISEFALL3METHODS` | Rising/Falling Three Methods | `@open,high,low,close` |
| `cdl.separatinglines` | `CDLSEPARATINGLINES` | Separating Lines | `@open,high,low,close` |
| `cdl.shootingstar` | `CDLSHOOTINGSTAR` | Shooting Star | `@open,high,low,close` |
| `cdl.shortline` | `CDLSHORTLINE` | Short Line Candle | `@open,high,low,close` |
| `cdl.spinningtop` | `CDLSPINNINGTOP` | Spinning Top | `@open,high,low,close` |
| `cdl.stalledpattern` | `CDLSTALLEDPATTERN` | Stalled Pattern | `@open,high,low,close` |
| `cdl.sticksandwich` | `CDLSTICKSANDWICH` | Stick Sandwich | `@open,high,low,close` |
| `cdl.takuri` | `CDLTAKURI` | Takuri (Dragonfly Doji with very long lower shadow) | `@open,high,low,close` |
| `cdl.tasukigap` | `CDLTASUKIGAP` | Tasuki Gap | `@open,high,low,close` |
| `cdl.thrusting` | `CDLTHRUSTING` | Thrusting Pattern | `@open,high,low,close` |
| `cdl.tristar` | `CDLTRISTAR` | Tristar Pattern | `@open,high,low,close` |
| `cdl.unique3river` | `CDLUNIQUE3RIVER` | Unique 3 River | `@open,high,low,close` |
| `cdl.upsidegap2crows` | `CDLUPSIDEGAP2CROWS` | Upside Gap Two Crows | `@open,high,low,close` |
| `cdl.xsidegap3methods` | `CDLXSIDEGAP3METHODS` | Upside/Downside Gap Three Methods | `@open,high,low,close` |

# License

[MIT](LICENSE)

# For Developers

Developer notes, local build commands, dependency groups, and benchmark report
guidance live in [DEVELOPMENT.md](DEVELOPMENT.md).
