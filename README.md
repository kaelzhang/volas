[![ci](https://github.com/kaelzhang/volas/actions/workflows/ci.yml/badge.svg)](https://github.com/kaelzhang/volas/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kaelzhang/volas/branch/main/graph/badge.svg)](https://codecov.io/gh/kaelzhang/volas)
[![PyPI version](https://img.shields.io/pypi/v/volas.svg)](https://pypi.org/project/volas/)
[![Python versions](https://img.shields.io/pypi/pyversions/volas.svg)](https://pypi.org/project/volas/)

# [volas](https://github.com/kaelzhang/volas)

> High-performance, Rust-backed columnar kernel for stock / candlestick (OHLCV) time-series data.

**volas** is a Rust-powered, **pandas-compatible** `DataFrame` for candlestick
(OHLCV) data, with [**242** trading-indicators](INDICATORS.md) built in.

Know pandas? You already know to use volas.

The difference is speed that **volas** beats **EVERY** solution on earth in terms of overall indicator calculating.

## Why volas

- **Drop-in for pandas.** The same `.loc` / `.iloc` / `.at`, `read_csv`,
  `to_numpy` and resampling — change the import, keep your code. (See
  [what's not covered](PANDAS-DIFFERENCES.md#index-limitations))
- **Fastest in the field.** Quicker than pandas, polars and TA-Lib on
  nearly every indicator — and faster than pandas even off the trading desk.
  ([live benchmark report](https://volas.ost.ai))
  - Beats TA-Lib on **153 / 158** covered indicators in batch computation.
  - Refreshes indicators incrementally on each new bar — **~5×** faster
    than TA-Lib in average, and up to **~360x** faster than pandas.
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
- [Missing values (`volas.NA`)](#missing-values-volasna)
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
df['close']
# 0    3.0
# 1    4.0
# 2    5.0
# 3    6.0
# 4    7.0
# 5    8.0
# Name: close, dtype: float64

# An indicator directive -> Series (2-period SMA of `close`)
df['ma:2']
# 0   <NA>
# 1    3.5
# 2    4.5
# 3    5.5
# 4    6.5
# 5    7.5
# Name: ma:2, dtype: float64

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

# 0   <NA>
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
  list or a duplicate name is rejected, and an absent column is never silently filled.
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

Directly gets the column value by `key`, returning a `Series` — and **never
computes**: unlike `df[key]`, which parses an unknown key as an indicator
directive and executes it, `get_column` only fetches an existing column and
raises `KeyError` otherwise. Use it whenever the column name comes from
external data (CSV headers, user input, configuration), so a name that happens
to look like a directive (e.g. `"ma:5"`) can never silently trigger a
computation.

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
# Name: close, dtype: int64
```

### df.append(other: DataFrame | Row) -> DataFrame

Appends rows of `other` (a `DataFrame` or a `Row`) to the end of the caller in
place, returns the same `DataFrame`, and applies the `DatetimeIndex` to the
newly-appended row(s) if possible. Use `copy()` first when the original frame
must stay unchanged.

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
df = df.append(new_bar)  # the new row's ma:20 is stale (a missing placeholder)
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

A `datetime64[ns]` Series exposes the pandas `.dt` accessor: calendar
components (`year` `month` `day` `hour` `minute` `second` `microsecond`
`nanosecond` `quarter` `dayofweek` `dayofyear` `days_in_month`), calendar
predicates (`is_month_start` … `is_year_end`, `is_leap_year`), names
(`day_name()` / `month_name()`), formatting (`strftime(fmt)`), bar alignment
(`floor(freq)` / `ceil(freq)` / `round(freq)` / `normalize()`), and
`isocalendar()`. A missing element yields `NA` in every component:

```py
t = volas.to_datetime(df['time'])
t.dt.hour                  # int64 Series, 0..23
t.dt.dayofweek             # Monday=0 .. Sunday=6
t.dt.floor('15min')        # datetime Series aligned to the 15-minute bar
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

### to_datetime(obj, unit='ns', format=None) -> Series

A top-level function that converts epoch numbers or datetime strings to a
datetime `Series`, mirroring `pandas.to_datetime`. `obj` may be a `Series`, a 1-D
NumPy array, or a list. A **missing** input (a float `NaN`, or a `volas.NA` in an
int column) becomes `NaT`, like `pd.to_datetime`.

- **obj** the values to convert — numeric epochs, datetime strings, or an
  already-datetime `Series` (returned unchanged).
- **unit?** `str = 'ns'` the epoch unit for **numeric** input (`'s'` / `'ms'` /
  `'us'` / `'ns'`); sub-unit fractions are preserved, like `pd.to_datetime`.
- **format?** `str | None = None` an explicit datetime format for **string**
  input (pandas `format=`, e.g. `'%Y-%m-%d %H:%M:%S'`) — faster and unambiguous;
  ignored for numeric input.

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
pandas, it works the same in volas, except for the deliberate
[NA-model divergences](#known-pandas-divergences-the-volasna-model) noted after
the listing.

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
s.sum() / s.mean() / s.min() / s.max() / s.std() / s.var() / s.median()   # skip missing
s.shift(n=1) / s.diff(n=1) / s.fillna(v) / s.ffill() / s.bfill()           # see Missing values: NA keeps the dtype
s.isna() / s.notna() / s.dropna() / s.equals(t)
```

#### Window operations (`rolling` / `expanding` / `ewm`) — compatibility only

> **This surface exists so pandas research / labeling code moves over verbatim.
> It is NOT the recommended way to compute indicators, and it should NOT be
> used in a live trading system**: a window result is a plain Series — it does
> not join the directive cache and is **not** incrementally refreshed by
> `append()` / `fulfill()`; every new bar costs a full `O(n)` recompute.
> Prefer the equivalent directive (`df['ma:20']`, `df['median:30']`,
> `df['stddev:20']`, …): same kernels, plus caching and `O(lookback)` per-bar
> refresh.

```py
s.rolling(window, min_periods=None, center=False)   # int window; min_periods defaults to window
s.expanding(min_periods=1)
s.ewm(com=|span=|halflife=|alpha=, min_periods=0, adjust=True, ignore_na=False)
                                                    # exactly ONE decay spelling

# Rolling / Expanding (pandas semantics: NA skipped, min_periods gates):
.count() .nunique()                                 # -> int64 Series (native NA)
.sum() .mean() .median() .min() .max()
.var(ddof=1) .std(ddof=1) .sem(ddof=1) .skew() .kurt()
.quantile(q, interpolation='linear') .rank(method='average', ascending=True, pct=False)
.first() .last()                                    # dtype-preserving
.corr(other) .cov(other, ddof=1)

# Ewm:
.mean() .sum() .var(bias=False) .std(bias=False) .corr(other) .cov(other, bias=False)
```

`center=True` labels each window at its center — it reads **future** bars
relative to the label. That is exactly what a labeling pass wants, and exactly
what a live signal must never do; it is supported for the former.

Time-based windows (`rolling('5min')` / a `timedelta`) are deliberately not
implemented. For multi-timeframe computation, maintain **two tf-aware
DataFrames** (see [Cumulation](#cumulation-and-datetimeindex)) and `append`
each bar to both — that is the supported, `O(lookback)`-per-bar design;
emulating a coarser timeframe through window arithmetic recomputes everything
on every bar.

Not provided (pandas members that conflict with volas's model): `apply` /
`agg` / `pipe` (arbitrary-Python-per-window), `win_type`, `step`, `on`,
`closed`, `method`, `ewm(times=...)`, `ewm.online()` — `append()` + directives
already cover the streaming use case.

#### Known pandas divergences (the `volas.NA` model)

A handful of APIs diverge from pandas **by design**, because volas stores missing
values natively as [`volas.NA`](#missing-values-volasna) (no `object` dtype, no
silent float upcast):

- **`shift` / `diff` / `fillna` and friends** keep the column's dtype — a missing
  value is `volas.NA`, not an int/bool/str column upcast to float/object.
- **Comparisons** (`==` `!=` `<` `<=` `>` `>=`) return a *non-nullable* bool mask:
  a missing value compares `False` (and `!=` compares `True`), following IEEE /
  NumPy — not pandas-nullable's three-valued `NA`. This keeps masks free of `NA`
  so `df[mask]` and assignment stay total.
- **`to_numpy()`** exports a missing cell as `NaN` (NumPy has no `NA`), so an
  int / bool / datetime column materializes as `float64` / `NaT`. Storage and
  `to_list()` keep the dtype and `volas.NA`.

For the full picture — why volas's type system is built this way, where pandas's
breaks, and the migration gotchas — see
[volas vs pandas — the type system](PANDAS-DIFFERENCES.md).

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

#### Bar labels are the period start

Every time frame lies on a **fixed grid**, and a cumulated bar is labelled with
its period's grid **start** — even when the first raw bar arrives mid-period. A
bar that opens with a `09:07` tick on a `15m` frame is labelled `09:00`, never
`09:07`, so volas bars line up exactly with exchange klines and with pandas
`resample` (`label='left'`).

The grid origins per frame: intraday frames anchor at **midnight** of the
index's (timezone-aware) trading day — a `15m` bar starts at `:00`/`:15`/`:30`/
`:45`, a `4h` bar at `00:00`/`04:00`/…; `1d` starts at midnight; `1w` on
**Monday**; `3d` is a continuous grid from the Unix epoch; `1M` / `1y` on the
calendar month / year. If a daylight-saving transition removes or repeats a
period's boundary, the label resolves to the period's earliest real instant.

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

| Constant | Label | Alignment |
| --- | --- | --- |
| `TimeFrame.s1` | `'1s'` | Civil second. |
| `TimeFrame.m1` | `'1m'` | Civil minute. |
| `TimeFrame.m3` | `'3m'` | Minute-of-hour buckets starting at `00`, `03`, `06`, ... |
| `TimeFrame.m5` | `'5m'` | Minute-of-hour buckets starting at `00`, `05`, `10`, ... |
| `TimeFrame.m15` | `'15m'` | Minute-of-hour buckets starting at `00`, `15`, `30`, `45`. |
| `TimeFrame.m30` | `'30m'` | Minute-of-hour buckets starting at `00` and `30`. |
| `TimeFrame.H1` | `'1h'` | Civil hour. |
| `TimeFrame.H2` | `'2h'` | Hour-of-day buckets starting at `00`, `02`, `04`, ... |
| `TimeFrame.H4` | `'4h'` | Hour-of-day buckets starting at `00`, `04`, `08`, ... |
| `TimeFrame.H6` | `'6h'` | Hour-of-day buckets starting at `00`, `06`, `12`, `18`. |
| `TimeFrame.H8` | `'8h'` | Hour-of-day buckets starting at `00`, `08`, `16`. |
| `TimeFrame.H12` | `'12h'` | Hour-of-day buckets starting at `00` and `12`. |
| `TimeFrame.D1` | `'1d'` | Civil day in the frame timezone. |
| `TimeFrame.D3` | `'3d'` | Continuous 3-day buckets anchored to the Unix epoch; they do not reset at month boundaries. |
| `TimeFrame.W1` | `'1w'` | Continuous Monday-start weeks, including runs that cross month boundaries. |
| `TimeFrame.M1` | `'1M'` | Civil calendar month in the frame timezone. |
| `TimeFrame.Y1` | `'1y'` | Civil calendar year in the frame timezone. |

Every bucket is aligned in the **frame timezone's local wall-clock** while storage
stays UTC: the hour-of-day frames (`2h`/`4h`/`6h`/`8h`/`12h`) start at local `00`
and step in local hours; `3d` counts continuous 3-local-civil-day buckets keyed
from the Unix epoch day in that zone (not reset at month boundaries); `1w` is
Monday-start in local civil time. So a daily/weekly bar follows the local trading
day, and a named zone makes the hour buckets DST-aware.

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

### Differences from pandas (vs pandas)

volas is pandas-shaped on the surface, but its **type system is deliberately
different** in more than the index: missing values keep their dtype, there is no
`object` dtype, value-returning methods stay `Series`, and a lossy conversion
raises instead of degrading silently. **See
[volas vs pandas — the type system](PANDAS-DIFFERENCES.md) for the full
comparison — why volas is built this way, where pandas's type system breaks, and
the migration gotchas.**

The index specifically is a **single level** of one homogeneous label type.
Relative to pandas, volas does **not** support:

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

Writing a fractional value into an integer column **raises** — the int dtype is
kept, and a lossy write errors rather than silently widening to float (see
[Differences from pandas](PANDAS-DIFFERENCES.md); writing `volas.NA` / `None`
keeps the int dtype and marks the cell missing). Writing into a cached directive
column drops its cached status, so a later `fulfill()` can never silently
overwrite your edit.

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

# Integer epochs: to_datetime(unit=...) reads the unit. An epoch is *absolute* —
# anchor it as UTC, then restate the zone for display. 1609770600000 ms == 14:30Z:
e = DataFrame({'t': [1609770600000], 'close': [100.0]})
e['t'] = to_datetime(e['t'], unit='ms')
e.set_index('t').tz_localize('UTC').tz_convert('America/New_York').index
# ['2021-01-04T14:30:00.000000000']

# An offset-aware string is already absolute too — to_datetime resolves the offset:
o = DataFrame({'t': ['2021-01-04T09:30:00+08:00'], 'close': [1.0]})
o['t'] = to_datetime(o['t'])
o.set_index('t').index
# ['2021-01-04T01:30:00.000000000']  (09:30+08:00 == 01:30Z)
```

A frame's time axis is in one of two states (the pandas model): **naive** (an
unanchored wall-clock, `df.tz is None`) or **tz-aware** (anchored, `df.tz` names
the zone — `'UTC'` included). `tz_localize` anchors a naive axis (the instant
moves to match the wall-clock in that zone); `tz_convert` restates an aware
axis in another zone (the instant is unchanged). Each refuses the other state —
converting an unanchored clock or re-anchoring an anchored one would silently
shift instants:

```py
naive = df                                   # df.tz is None
aware = naive.tz_localize('America/New_York')   # anchor: instants move, wall-clock kept
aware.tz_convert('+08:00')                   # restate: instants kept, wall-clock moves
naive.tz_convert('+08:00')                   # TypeError — anchor with tz_localize first
aware.tz_localize('UTC')                     # TypeError — already anchored; use tz_convert
```

`cumulate` to a daily (or coarser) bar aligns buckets to the frame's local
trading day — DST-aware for a named zone — while the raw `.index` numpy export
stays UTC (matching pandas `.values`).

## Missing values (`volas.NA`)

`volas.NA` is the single missing-value marker, and **every dtype supports it** —
crucially, a missing value **never changes the column's dtype**:

| dtype | how missing is stored | element access | console display |
|---|---|---|---|
| `float64` / `float32` | `NaN`, in-band | `np.float64(nan)` | `<NA>` |
| `int64` / `int32` / `bool` / `str` | a validity mask (dtype kept) | `volas.NA` | `<NA>` |
| `datetime64[ns]` | `NaT` | `np.datetime64('NaT')` | `<NA>` |

Whatever the storage, the **console always prints `<NA>`** — one symbol for a
missing value, regardless of dtype (a float `NaN`, a datetime `NaT`, and an int /
bool / str hole all render identically; `to_string(na_rep=...)` overrides it).
Element access and `to_numpy` stay dtype-specific (a float hole reads back as
`np.nan`), so numpy / pandas interop is lossless.

This tracks pandas' own direction ([PDEP-16]) and means volas has **no `object`
dtype**: an `int` / `bool` / `str` column with a hole stays `int` / `bool` /
`str`, where pandas 3.0 upcasts to `float64` / `object`.

```py
import volas
s = volas.DataFrame({'a': [1, None, 3]})['a']
s.dtype                  # 'int64'        (pandas would give float64)
s[1]                     # <NA>           (s[1] is volas.NA; a float hole stays np.nan)
s.sum()                  # np.int64(4)    reductions skip NA
s.fillna(0).to_list()    # [1, 0, 3]
s.isna().to_numpy()      # [False, True, False]
print(s)                 # the missing cell prints as <NA>

# shift / diff keep the int dtype (pandas upcasts to float); the gap is NA:
volas.DataFrame({'a': [10, 20, 30]})['a'].shift(1).to_list()   # [<NA>, 10, 20]
```

- **Producing NA** — `None` (or `volas.NA`) in a constructor list, the `shift` /
  `diff` gap, and the default fill of `where` / `mask`.
- **Consuming NA** — reductions (`sum` / `mean` / `min` / …) and `count` skip it;
  arithmetic propagates it (`x ∘ NA = NA`); `~` / `&` / `|` / `^` use Kleene
  three-valued logic (`NA & False = False`, `NA | True = True`); `cumsum` / `abs`
  / `round` / `clip` / indexing carry it through; `isna` / `notna` / `dropna` /
  `fillna` / `ffill` / `bfill` work on every dtype.
- **Comparisons** treat a missing value IEEE / numpy style: `==`, `<`, `<=`,
  `>`, `>=` involving NA compare `False`, while `!=` compares `True` — so a
  boolean mask is always pure `bool`, clean for `df[mask]`. Note the `!=`
  exception: `s != value` therefore *includes* missing rows.

[PDEP-16]: https://github.com/pandas-dev/pandas/pull/58988

## pandas interop

pandas is **not** a runtime dependency; these bridges import it lazily, only when
called, so `import volas` stays pandas-free.

```py
from volas import from_pandas

df = from_pandas(pandas_df)        # numeric / bool / str / datetime native; a (tz-aware) DatetimeIndex round-trips;
                                   # a nullable Int64 / boolean / string column reads back as int / bool / str + volas.NA
pdf = df.to_pandas()               # -> pandas.DataFrame ('numpy' backend: an int/bool column with NA becomes float64 + NaN)
pdf = df.to_pandas(dtype_backend='numpy_nullable')  # faithful masked Int64 / boolean (a lossless NA round-trip)
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

## Built-in Indicators

The complete directive reference lives in [INDICATORS.md](INDICATORS.md). It
covers Volas-exclusive indicators, built-in statistical commands, and
TA-Lib-compatible directives.

# License

[MIT](LICENSE)

# For Developers

Developer notes, local build commands, dependency groups, and benchmark report
guidance live in [DEVELOPMENT.md](DEVELOPMENT.md).
