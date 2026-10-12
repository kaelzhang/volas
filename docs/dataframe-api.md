# DataFrame API

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

- **data** `dict[str, list | np.ndarray] | DataFrame` the column data, one of:
  - a **dict** mapping each column name to an equal-length list or NumPy array
    (`float`, `int`, `bool`, `datetime64`, or `string`); a NumPy **masked array**'s
    masked cells become `volas.NA`;
  - another volas **`DataFrame`**, which is then copied (like `pandas.DataFrame(df)`).

  The constructor does **not** accept a `pandas.DataFrame` or an Arrow object — bridge
  those with the dedicated [`DataFrame.from_pandas`](#dataframefrom_pandaspdf---dataframe) /
  [`DataFrame.from_arrow`](#dataframefrom_arrowdata---dataframe) instead. To attach a
  [`DatetimeIndex`](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html),
  parse a column with `to_datetime`, promote it with `set_index`, then tag a zone with
  `tz_localize` / `tz_convert`. See [Timezones](cumulation.md#timezones).
- **columns** `Optional[list[str]] = None` Select and order the columns to keep —
  the same projection as `df[[...]]` (a stale cached directive column is refreshed
  first; a bounded-window source gives its visible rows). A name not present raises
  `KeyError` (a directive is never computed here); an empty list or a duplicate name
  is rejected, and an absent column is never silently filled. With `index=`, `dtype=`
  or `window=`, a source `DataFrame` is read like any bulk read: it raises while one
  of its cached directive columns is stale.
- **time_frame** `Optional[str | TimeFrame] = None` If set, makes this a
  **tf-aware** (cumulating) DataFrame at this bar interval: the given rows are bars
  at that frame, each labelled with its **period start**, and the last one is the
  **forming** bar — a later `append` of a finer bar in its period folds into it
  (the last row must not already contain that bar). Requires a `DatetimeIndex`. See
  [Live cumulation](#live-cumulation--a-tf-aware-dataframe).
- **cumulators** `Optional[dict[str, str]] = None` Per-column aggregator overrides
  used when folding (e.g. `{'amount': 'sum'}`), only meaningful together with
  `time_frame`. Defaults to OHLCV semantics (`open`=first, `high`=max, `low`=min,
  `close`=last, `volume`=sum; any other column `last`). Each dict **value** is one of:
  - `'first'` — the first value in the bucket
  - `'last'` — the last value in the bucket
  - `'max'` — the maximum
  - `'min'` — the minimum
  - `'sum'` — the sum
- **window** `Optional[int] = None` Make this a **bounded rolling-window** frame
  showing only the last `window` rows (see [Bounded rolling window](#bounded-rolling-window)).
  Requires `max_lookback`.
- **max_lookback** `Optional[int | list[str]] = None` **Required with `window`** (and
  valid only with it): the hidden-history margin (`window + max_lookback`) that keeps
  cached indicators correct across the automatic front-drop. Recursive indicators
  (EMA/Wilder/ATR/RSI/MACD) stay **bit-exact**; finite-window ones (ma/wma/trima/stddev/…)
  match an unbounded frame to floating-point tolerance (~1e-13). Pass an **int** to state
  the largest indicator lookback you will use, or a **list of indicator directives** to
  derive it from the largest of their lookbacks (e.g. `['atr:14', 'ma:50']` → margin 49),
  so you never hand-compute a compound indicator's warm-up. Each list entry must be an
  indicator directive (`'ma:50'`); a bare/typo'd name (`'ma50'`) is rejected. Sizing the
  margin too small silently breaks the guarantee.

### Bounded rolling window

Pass `window=` to cap the frame at the last `window` rows. This is the live-trading
/ NN-input shape: you keep `append`-ing bars forever, but memory stays **bounded** —
the frame transparently drops old rows once it has accumulated enough, while
retaining a hidden `max_lookback`-row margin so cached indicators stay **consistent**
across each drop — recursive indicators (EMA/ATR/RSI/…) are bit-exact with an unbounded
frame, finite-window ones (ma/wma/…) match it to floating-point tolerance (~1e-13).

```py
# A bounded 30-bar window; the margin is sized from the indicators you declare.
wf = DataFrame(seed, time_frame='15m', window=30, max_lookback=['atr:14'])

for bar in feed:            # runs forever; memory never grows
    wf.append(bar)          # fold a 1m bar into the forming 15m bar
    wf.fulfill()            # refresh the cached atr:14 tail (O(lookback))
    if wf.ready:            # warmed up: all 30 rows have valid history
        x = wf[['close', 'atr:14']].to_numpy('float32')   # the 30×2 feature window
```

Every row-facing surface — `len`, `shape`, `index`, indexing (`[]` / `.iloc` /
`.loc` / `.iat` / `.at`), `head` / `tail`, reductions, `to_numpy`, `to_csv`,
`to_pandas`, `repr`, `exec`, `tz_localize` / `tz_convert`, comparisons, and a windowed
frame used as an `append` source or a constructor source — shows **only** the `window`
rows; the margin is never visible.

Row labels stay unique over the rows the frame **retains** — the last
`window + max_lookback` rows: the visible ones and the hidden margin, never the older rows
the buffer may still hold until its next compaction. An integer or string label that an
earlier row in the margin still holds raises, and says so:

```text
ValueError: cannot append label 20: an earlier row with that label is still retained in the window's hidden margin (max_lookback)
```

A label whose row has already been dropped from the retained rows is accepted again.

#### `ready` — has the window warmed up?

`ready` (a property) is `True` once the frame holds `window + max_lookback` rows — so
every one of the `window` visible rows has a full indicator history behind it and the
cached indicators are valid end-to-end. During the initial warm-up it is `False` and
the visible rows are still filling in; gate your inference / training on it.

```py
wf = DataFrame(seed, time_frame='15m', window=30, max_lookback=['atr:14'])
wf.ready          # False — fewer than 30 + 14 rows accumulated so far
...               # append until warmed
wf.ready          # True  — every visible row now has valid indicator history
```

An **unbounded** frame (no `window=`) has no warm-up contract and is **always `True`**.

#### `fill_into(out, columns=None)` — zero-allocation feature export

`fill_into` writes the window's values straight into a NumPy array **you** own, in
place — nothing is allocated per call. It is the export half of a live inference loop:
paired with one preallocated buffer, an `append → fulfill → fill_into → infer` loop
allocates nothing per bar (unlike `to_numpy`, which mints a fresh matrix every call).

Contract:

- **Only already-cached columns**, named by their **canonical directive string**. A
  directive column must be **materialized first** — access it once and read the name
  back (`name = wf['atr:14'].name`), because a cached directive is stored under its
  canonical form (e.g. `'MA: 5'` → `'ma:5'`), *not* the string you passed, and `columns=`
  matches on the exact stored name. (`max_lookback=['atr:14']` only sizes the margin; it
  does not create the column, and `fill_into` exports cached values, never computes them.)
- **`out`** must be a `float32` **or** `float64` 2-D array whose shape is **exactly**
  `(len(df), k)`, where `k` is the number of exported columns (strides are respected, so a
  non-contiguous or Fortran-order view works too). A wrong shape or dtype raises.
- **`columns`** selects which columns to export, in order (default: every column). A
  **string column has no float meaning and is rejected** — list the numeric ones in
  `columns=` to exclude it.
- A **missing / NA cell becomes `NaN`**.
- An `append` leaves the cached indicators stale, so **call `fulfill()` before
  `fill_into()`** — exporting with an unrefreshed directive column raises.

```py
import numpy as np

wf = DataFrame(seed, time_frame='15m', window=30, max_lookback=['atr:14'])
atr = wf['atr:14'].name               # materialize the directive once, and read its column
                                      # name back: a cached directive lands under its CANONICAL
                                      # form (here 'atr:14', but e.g. 'MA: 5' -> 'ma:5'), not the
                                      # string you passed — and max_lookback only sized the margin
COLS = ['open', 'high', 'low', 'close', atr]    # k = 5 features

# One reusable buffer for the whole run — shape (window, k), the model's input tensor.
buf = np.empty((30, len(COLS)), dtype=np.float32)

for bar in feed:
    wf.append(bar)
    wf.fulfill()                      # refresh atr's stale tail before exporting
    if not wf.ready:
        continue                      # warming up: buf isn't (30, k) yet, indicators not valid
    wf.fill_into(buf, columns=COLS)   # zero-alloc write of the 30×5 window into `buf`
    prediction = model(buf)           # feed the model the same memory every bar
```

> **Shape note.** The shape match is exact against `len(df)`, which **grows during
> warm-up** (1, 2, … up to `window`) and only then settles at `window`. Gating on
> `wf.ready` (as above) sidesteps this: by the time it is `True`, `len(df) == window`,
> so a fixed `(window, k)` buffer always fits. To export mid-warm-up, size `out` to the
> current `len(df)` instead.

`fill_into` is not windowed-only — on a plain frame it exports the whole logical view
the same way; the bounded window is just where reusing one buffer matters most.

### DataFrame.from_pandas(pdf) -> DataFrame

A **volas-specific static method** that builds a `DataFrame` from a `pandas.DataFrame`
(`pdf`) — the inverse of `df.to_pandas()`. pandas is imported lazily (only here, so volas
stays pandas-free at import). A nullable column keeps its dtype + `volas.NA`, and a
`DatetimeIndex` (tz-aware too) round-trips. The index stays the index under its own
name (an unnamed index stays unnamed), independent of the columns — a column that
shares the index's name is kept as a column. See [pandas interop](interop.md#pandas-interop).

```py
df = DataFrame.from_pandas(pandas_df)   # pandas.DataFrame -> volas DataFrame
```

### DataFrame.from_arrow(data) -> DataFrame

A **volas-specific static method** that builds a `DataFrame` from any object exposing
the Arrow **C-Stream** protocol (`__arrow_c_stream__`) — a `pyarrow.Table` /
`RecordBatch` / `RecordBatchReader`, a polars `DataFrame`, etc. The data buffers are
borrowed where the dtypes match (otherwise a column is copied), a multi-chunk source
is concatenated, and the result carries a fresh `RangeIndex`. The producer's buffers
are validated before they are read: a malformed array (buffer sizes, offsets out of
range or not monotonic, a present string that is not valid UTF-8) raises `ValueError`.
A null string slot's bytes are unspecified in Arrow and are never read.

- **data** the Arrow source — any object implementing `__arrow_c_stream__`.

```py
df = DataFrame.from_arrow(pa_table)        # pyarrow.Table     -> DataFrame
df = DataFrame.from_arrow(polars_df)       # polars.DataFrame  -> DataFrame
```

> Arrow is **not** accepted by the `DataFrame(data=...)` constructor (which takes a
> `dict` or another `DataFrame`); build from an Arrow object through `from_arrow`.

### df.exec(directive: str) -> np.ndarray

Evaluates the given directive and returns its values as a numpy ndarray. It is a
pure, **stateless** evaluation — the frame is never modified (no column is
created, and no cache is read or written). A name in the directive reads a data
column or runs a built-in command, never a cached directive column, so the result is
fresh even while the frame is stale (see
[Names](directives.md#names-a-data-column-or-a-built-in-command)). A bounded-window
frame returns its visible rows.

```py
# Compute the directive without touching the frame
df.exec('ma:20')
```

The difference between `df[directive]` and `df.exec(directive)` is that
- `df[directive]` creates a column for the result and caches it (incrementally
  refreshed after an append), while `df.exec(directive)` computes fresh every
  time and leaves the frame untouched — for a cached ndarray, use
  `df[directive].to_numpy()`
- `df[directive]` also accepts other indexing targets (a column name, a list, a
  boolean mask, a slice), while `df.exec(directive)` only accepts a valid
  **volas** directive string
- `df[directive]` returns a `Series` or `DataFrame` object while
  `df.exec(directive)` returns an [`np.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)

### df.get_column(key: str) -> Series

Directly gets the column value by `key`, returning a `Series` — and **never
computes**: unlike `df[key]`, which parses an unknown key as an indicator
directive and executes it, `get_column` only fetches an existing column and
raises `KeyError` otherwise. It is a column read: a cached directive column left
stale by an `append` is refreshed first, exactly as `df[key]` refreshes it. Use it whenever the column name comes from
external data (CSV headers, user input, configuration), so a name that happens
to look like a directive (e.g. `"ma:5"`) can never silently trigger a
computation.

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

### df.append(other: DataFrame | Row | dict) -> DataFrame

Appends rows of `other` to the end of the caller in place, returns the same
`DataFrame`, and applies the `DatetimeIndex` to the newly-appended row(s) if
possible. Use `copy()` first when the original frame must stay unchanged.

`other` is a `DataFrame`, a `Row`, or a **scalar bar dict** — one bar written as
`{column: value}` with the bar's timestamp under the key equal to the index's name
(a `RangeIndex` auto-increments). The dict form is the fast live path — it builds the
bar straight into the frame with no per-bar 1-row `DataFrame`:

```python
df.append({'time_key': ts, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
```

It is **strict**: every data column must be provided (a missing one raises — unlike a
`DataFrame` / `Row`, where a missing column is NA-padded), and an unknown key raises. The
timestamp is read in its own zone: an aware `Timestamp` / `datetime` / offset string is
an instant, a naive one a wall clock — judged against the frame's index like any other
row (a naive bar onto a tz-aware frame raises), a raw epoch-ns count — an integer or a
NumPy `datetime64`, the form instants are exported in — read in the frame's zone. A label
read back from the frame (`df.index[-1]`, an aware `Timestamp` on a tz-aware frame) is
the same label, so it replaces that row. Each value is one cell: an array (even of one
element) raises. A positional (`RangeIndex`) frame continues its labels, a bounded window's too.
Cached directive columns are not supplied — they are padded and refreshed automatically.

Rows are keyed by their index label:

- a row labelled like the frame's **last** row **replaces** it whole (a column the
  row lacks becomes missing) — so an exchange re-sending the bar it is still building
  updates that bar instead of adding a second one;
- a `DatetimeIndex` only ever increases: a row labelled **before** the last row raises
  `ValueError`;
- an integer or string index stays unique: a row labelled like an **earlier** row
  raises `ValueError` (a new label before the last is appended);
- rows without labels of their own (a `RangeIndex`) take the next labels — on an
  integer index the labels after the last one, which must not already be in the index.

What the rows may carry:

- **the frame's data columns only.** A column the frame does not have raises
  `ValueError`; a column the frame lacks in the rows is padded with a missing value. A
  column the frame **caches** (an indicator directive it computes itself) raises
  `ValueError`, in every form — select the data columns first:

  ```text
  ValueError: cannot append columns ['ema:12']: they are cached directives that this DataFrame computes itself. Select the data columns first, e.g. df.append(other[['open', 'high', 'low', 'close', 'volume']]).
  ```

- **values the frame's dtypes hold exactly.** Each value is converted to the column's
  dtype when that changes nothing (an integer `2` into a float column, `3.0` into an
  integer column); a value that would change (`2.5` into an integer column, text into a
  number column) raises `TypeError` (`OverflowError` past the dtype's range). The same
  rule decides a new row and a replacing row.
- **labels of the frame's index kind.** A `DatetimeIndex` takes timestamps only — of
  the same awareness: tz-naive rows onto a tz-aware frame (or the reverse) raise
  `ValueError`, while tz-aware rows in another zone are the same instants and keep the
  frame's zone; a `NaT` label raises. An integer or string index takes labels of its
  own kind; rows with a `RangeIndex` onto a `DatetimeIndex` raise.
- **a source frame that is fresh.** Appending a `DataFrame` reads it like any bulk
  read: while one of its cached directive columns is stale it raises (call its
  `fulfill()` first), and a bounded-window source contributes its visible rows only.

The rows are checked whole before anything is written: on an error the frame is left
exactly as it was — every column, the index and the cached indicators — and the next
append continues as if the rejected one never happened.

```python
times = ['2026-01-01 17:00', '2026-01-01 17:01', '2026-01-01 17:02']
df = DataFrame({'time': to_datetime(times), 'close': [10.0, 11.0, 12.0]}).set_index('time')
df['ma:2']
df.append({'time': Timestamp('2026-01-01 17:02'), 'close': 12.5})  # re-sent 17:02
len(df)                 # 3
df['ma:2'].iloc[-1]     # 11.75
```

Cached indicators stay exact after a replacement, and refreshing one costs what a
newly appended row does — one step per recursive indicator, one window per
finite-window one.

If the caller is a **tf-aware** DataFrame (one built with a `time_frame`, or
the result of `cumulate`), `append` instead **folds** each bar into the open
period: the period's bars are kept in a small frame, a bar with the same timestamp
as one already there replaces it, and the period's cumulation **replaces the last
row**; a bar of a later period starts a new row — see
[Live cumulation](#live-cumulation--a-tf-aware-dataframe).

`append` is lazy: it does not recompute the indicator columns of the new rows.
They stay stale until an indicator-**column read** refreshes them or `df.fulfill()`
is called (see below).

### df.cumulate(time_frame: TimeFrame | str, cumulators: dict | None = None) -> DataFrame

Cumulate (resample) the data frame to a coarser `time_frame`, returning a new
`DataFrame`. Requires a `DatetimeIndex`.

- **time_frame** `TimeFrame | str` the target bar interval, e.g. `TimeFrame.m5`
  or `'5m'`. See [TimeFrame](cumulation.md#timeframe).
- **cumulators?** `dict[str, str] | None = None` per-column aggregator overrides
  (e.g. `{'amount': 'sum'}`). Defaults to OHLCV semantics (`open`=first, `high`=max,
  `low`=min, `close`=last, `volume`=sum; any other column `last`). Each dict **value**
  is one of:
  - `'first'` — the first value in the bucket
  - `'last'` — the last value in the bucket
  - `'max'` — the maximum
  - `'min'` — the minimum
  - `'sum'` — the sum

```py
# from 1-minute klines to 5-minute klines
five_minute = one_minute.cumulate('5m')
fifteen_minute = one_minute.cumulate('15m')

five_minute.append(new_candle_1m)
# appending a 1-minute candle to a 5-minute DataFrame folds it into the 5m bar

fifteen_minute.append(new_candle_1m)
# so 1-minute data conveniently generates 5m and 15m test datasets
```

See [Cumulation and DatetimeIndex](cumulation.md#cumulation-and-datetimeindex) for details.

### df.fulfill() -> None

Batch-refresh every cached indicator column's stale tail in place
(`O(lookback + new rows)` each, not an `O(n)` recompute), and return `None`.

Since `append` is lazy, the cache becomes fresh in one of two ways:

- **Reading an indicator column** — `df['ma:20']`, `df[['ma:20', 'rsi:14']]`,
  `df.get_column('ma:20')`, or `DataFrame(df, columns=[...])` — auto-refreshes just
  those columns' stale tails on access, so a column read is always fresh and cheap.
  The single- and multi-column forms behave identically.
- **Every other read** — `to_numpy()`, `.iloc` / `.loc` / `.at`, row slices and
  boolean masks (`df[a:b]`, `df[mask]`), the reductions (`sum` / `mean` / `max` /
  `describe` / …), `isna` / `notna`, `rolling` / `expanding` / `ewm`, `tz_localize` /
  `tz_convert`, `to_csv`, `repr`, … — does **not** auto-refresh; while the frame is
  stale it **raises**, telling you to call `fulfill()` first. So does a stale frame
  used as an **operand**: the right-hand side of a comparison, a `where` / `mask` or
  `df[mask] = v` condition, an `append` source. `df.exec(d)` reads data columns only,
  so it is never stale. This is deliberate: a half-updated frame fails loud instead
  of silently returning stale values, and you control when the (bounded) refresh
  cost is paid — which matters on a latency-sensitive live path.

```py
df['ma:20']              # cache + read the 20-period SMA (fresh)
df.append(new_bar)       # lazy: the new row's ma:20 is now stale
df['ma:20']              # a column read auto-refreshes only the tail (fresh again)

df.append(new_bar)       # stale again
df.fulfill()             # batch-refresh every cached column's tail
df.to_numpy()            # now fresh (a bulk read would have raised while stale)
```

### df.is_computed(name: str) -> bool

Whether the column `name` is a **directive (computed)** column — one derived from
a directive (e.g. `df['rsi:14']`) and refreshed by `fulfill` — rather than a plain
data column you supply per bar. Raises `KeyError` if `name` is not a column.

Use it to tell the two kinds of column apart, e.g. to export only the raw columns:

```py
df['ma:5']                                              # materialize a directive column
df.is_computed('ma:5')                                  # True
df.is_computed('close')                                 # False
raw = [c for c in df.columns if not df.is_computed(c)]  # ['open', 'high', 'low', 'close', 'volume']
```

### df.to_numpy(dtype=None, na_value=...) -> np.ndarray

The frame as a 2-D NumPy array (rows × columns). It tracks pandas except for one
deliberate guard: an **integer** or **bool** `dtype` over a frame that holds missing
values **raises** instead of silently writing garbage (NumPy cannot store `NA` in an
integer or bool array) — give `na_value` to fill instead.

- **dtype** `str | None` — an optional export cast. `None` (the default) gives the
  honest per-dtype representation; otherwise the target is whatever NumPy resolves the
  spelling to (`'f8'`, `'d'` and `'<f8'` are `float64`, `'|O'` is `object`):
  - `'object'` (or `'O'`) — a lossless 2-D array of typed cells (number / `str` /
    `Timestamp` / `volas.NA`); the only dtype that keeps a `str` or datetime column intact.
  - `'int64'`, `'int32'`, `'int16'`, `'int8'` (and the unsigned `'uint*'`) — the **exact**
    `i64` channel (a large int and a datetime's epoch-ns survive without a `float` round
    trip). Over a frame with missing values this **raises** unless `na_value` is given.
  - `'float64'`, `'float32'`, `'float16'` — the (lossy) `float` channel: a missing cell
    is `NaN`, a datetime past 2⁵³ ns quantises.
  - `'bool'` — boolean: each value's truth (`0.5` → `True`). Over a frame with missing
    values this **raises** unless `na_value` is given.
  - `'datetime64[ns]'` — datetime nanoseconds; a `NaT` cell is preserved.
  - a text dtype (`'U'` / `'S'`) — each typed cell formatted (`1.5` → `'1.5'`).
  - A `str` column rejects every numeric / temporal dtype — use `'object'`.
- **na_value** `Any` — the value substituted for each missing cell. Default: the
  NA-model representation (`NaN` / `NaT` / `volas.NA`).

```py
df = DataFrame({'a': [1, 2, 3], 'b': [1.5, 2.5, 3.5]})

df.to_numpy()                 # -> float64 2-D array
df.to_numpy(dtype='int64')    # exact int64 cast (dense frame)
df.to_numpy(dtype='object')   # typed cells — lossless (numbers / str / Timestamp / volas.NA)

# an integer dtype over a missing value raises — unless na_value fills it
DataFrame({'a': [1, None]}).to_numpy(dtype='int64')               # ValueError
DataFrame({'a': [1, None]}).to_numpy(dtype='int64', na_value=0)   # -> [[1], [0]]  (int64)
```

Notes:
- The default (no `dtype`) is the honest representation: an all-numeric/bool frame
  is a `float64` matrix (a missing cell → `NaN`), a frame containing `str` or mixed
  dtypes is an `object` matrix of typed cells, and a datetime frame is `datetime64[ns]`.
- `dtype='object'` is always lossless — each cell keeps its own typed value (a
  number, a `str`, a `Timestamp`, or `volas.NA`).
- A `str` column has no numeric meaning, so any numeric/temporal `dtype` raises —
  use `dtype='object'` to keep the strings.
- A datetime column is **exempt** from the integer-NA raise: under `dtype='int64'`
  a `NaT` exports as its exact epoch-ns sentinel (datetime never round-trips through
  `float`); `na_value` overrides that sentinel when given.
- `to_numpy()` is a bulk read; it does not auto-refresh stale indicator columns and
  raises if any are stale — call [`df.fulfill()`](#dffulfill---none) first.

For a zero-copy hand-off to Arrow / DLPack consumers, see
[Arrow & DLPack interop](interop.md#arrow--dlpack-interop-zero-copy).

### df.to_arrow() -> pyarrow.Table

A **volas-specific** export to a `pyarrow.Table`, **zero-copy** where the dtypes
match — the numeric / string / datetime column buffers are shared with Arrow, while
`bool` and the null bitmap are repacked. Requires `pyarrow` (imported lazily, only
here). It is a convenience over volas's Arrow **C-Stream** bridge: any Arrow consumer
can read the frame directly through the standard `__arrow_c_stream__` PyCapsule
protocol, with no `to_arrow()` call and without volas depending on pyarrow. A column
name containing a NUL byte cannot cross the Arrow C interface and raises `ValueError`.

```py
import pyarrow as pa
tbl = df.to_arrow()        # -> pyarrow.Table (shares the column buffers)
tbl = pa.table(df)         # identical, via the __arrow_c_stream__ protocol
pdf = pl.from_dataframe(df)  # polars reads it through the same protocol
```

Returns a `pyarrow.Table`. See [Arrow & DLPack interop](interop.md#arrow--dlpack-interop-zero-copy)
for the full zero-copy contract and the DLPack export.

### df.to_pandas(dtype_backend='numpy') -> pandas.DataFrame

Export to a `pandas.DataFrame` (pandas is imported lazily, only here — it is not a
runtime dependency). A `DatetimeIndex` round-trips with its zone; a datetime **column**
exports as UTC-aware `datetime64[ns, UTC]` (columns hold UTC instants and store no
zone — see [Datetime columns are UTC instants](interop.md#datetime-columns-are-utc-instants)).
The reverse bridge is [`DataFrame.from_pandas`](#dataframefrom_pandaspdf---dataframe).

- **dtype_backend?** `str = 'numpy'` how a missing value is carried into pandas:
  - `'numpy'` — the most ecosystem-compatible form: an int / bool column with a missing
    value becomes `float64` / `object` with `NaN` (like `pandas.Int64.to_numpy()`).
  - `'numpy_nullable'` — a faithful, lossless masked round-trip: an int / bool / str
    column stays `Int64` / `boolean` / `string` with the hole as `pandas.NA`.

```py
pdf = df.to_pandas()                                # 'numpy' backend (NaN-based)
pdf = df.to_pandas(dtype_backend='numpy_nullable')  # lossless masked Int64 / boolean / string
```

See [pandas interop](interop.md#pandas-interop) for the round-trip details.

### df.to_csv(path=None, ...) -> str | None

Write the frame as CSV — a subset of pandas `to_csv`. With a `path` it writes the
file and returns `None`; with `path=None` it returns the CSV as a `str`.

- **path?** `str | os.PathLike | None = None` the output file; `None` returns a string.
- **sep?** `str = ','` the field delimiter.
- **index?** `bool = True` write the row index as the first column.
- **header?** `bool = True` write the column-name header row.
- **na_rep?** `str = ''` the token written for a missing value.
- **columns?** `list[str] | None = None` the columns to write, in order; `None`
  (the default) writes **every** column.
- **float_format?** `str | None = None` a printf-style float format, e.g. `'%.2f'`.

By default `to_csv` writes **every column the frame holds — directive (computed)
columns included**. A directive column like `ma:3` is a real column once
materialized (it counts in `df.columns`), so it is exported alongside the raw
OHLCV columns; its warm-up rows render as the empty `na_rep`. Pass `columns=` to
choose exactly what to write — e.g. the raw columns only. Like every bulk read,
`to_csv` first requires the frame to be fresh: call `fulfill()` after an `append`,
or it raises while a cached column is stale.

```py
from volas import DataFrame

df = DataFrame({
    'open':   [1.0, 2, 3, 4, 5],
    'high':   [2.0, 3, 4, 5, 6],
    'low':    [0.5, 1, 2, 3, 4],
    'close':  [1.5, 2.5, 3.5, 4.5, 5.5],
    'volume': [10, 20, 30, 40, 50.0],
})
df['ma:3']                                # materialize a directive (computed) column
df.fulfill()                              # refresh the cache before a bulk export

# Default: EVERY column is written — the ma:3 directive column included:
df.to_csv(index=False)
# 'open,high,low,close,volume,ma:3\n1.0,2.0,0.5,1.5,10.0,\n...,2.5\n...'
#                            ^^^^ directive column present; warm-up rows blank

# Pass columns= to write exactly what you want — here the raw columns only:
df.to_csv(index=False, columns=['open', 'high', 'low', 'close', 'volume'])
```

### Series

`df[col]` and `df[directive]` return a `Series` — a named 1-D column whose API is
pandas-compatible: arithmetic / comparison / logical operators, `.sum()` /
`.mean()` / `.std()` / …, `.shift()` / `.diff()` / `.fillna()`, `.iloc` /
`.loc`, `.to_numpy()` / `.to_list()`. See
[the rest of the pandas-compatible API](pandas-compatibility.md#the-rest-of-the-pandas-compatible-api)
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

### Series.from_arrow(data, name=None) -> Series

A **volas-specific static method** that builds a `Series` from any object exposing the
Arrow **C-Data** array protocol (`__arrow_c_array__`) — a `pyarrow.Array`, a polars
`Series`, etc. The data buffer is borrowed where the dtype matches (otherwise copied);
the result carries a fresh `RangeIndex`. A malformed array raises `ValueError`, exactly
as in [`DataFrame.from_arrow`](#dataframefrom_arrowdata---dataframe).

- **data** the Arrow source — any object implementing `__arrow_c_array__`.
- **name?** `str | None = None` the name for the resulting `Series`.

```py
s = Series.from_arrow(pa_array, name='close')   # pyarrow.Array -> Series
```

### series.to_numpy(dtype=None, na_value=...) -> np.ndarray

The column values as a 1-D NumPy array — `pandas.Series.to_numpy` semantics:

- **dtype** `str | None` — an optional export cast. `None` (the default) gives the
  column's native representation; otherwise any NumPy dtype string accepted by
  `numpy.ndarray.astype`, the common values being:
  - `'int64'`, `'int32'`, `'int16'`, `'int8'` (and the unsigned `'uint64'`, `'uint32'`,
    `'uint16'`, `'uint8'`) — integer. Over a column with missing values this **raises**
    unless `na_value` is given (an NA has no integer representation).
  - `'float64'`, `'float32'`, `'float16'` — floating point; a missing cell is `NaN`.
  - `'bool'` — boolean. Over a column with missing values this **raises** unless
    `na_value` is given (an NA has no boolean representation).
  - `'datetime64[ns]'` — datetime nanoseconds; a missing cell is `NaT`.
  - `'object'` (or `'O'`) — Python objects, each cell its own typed value (lossless).
- **na_value** `Any` — the value to substitute for each missing cell. Default: the
  NA-model representation (`NaN` for the float export, `None` in an object array). With
  an explicit integer `dtype`, the values stay **exact** (a large int is not funnelled
  through `float64`) and the holes become `na_value`.

```py
series = DataFrame({'qty': [1, None, 3]})['qty']    # int64 with a missing value

series.to_numpy()                          # -> array([ 1., nan,  3.])   (float64; a missing int -> NaN)
series.to_numpy(dtype='int64', na_value=0) # -> array([1, 0, 3])         (int64; NA filled, dtype kept)
series.to_numpy(na_value=-1)               # -> array([ 1., -1.,  3.])   (default float export, NA -> -1)

# without na_value, an integer dtype over a missing value raises
series.to_numpy(dtype='int64')
# ValueError: cannot convert a column with missing values to integer NumPy dtype 'int64' ...
```

Notes:
- The default (no `dtype`, no `na_value`) is the dtype-specific export: a missing
  int / bool / datetime cell collapses to `NaN` / `NaT` (NumPy has no `NA`), while a
  dense column keeps its native dtype. A float `NaN` is in-band, so a float column
  cast to an integer dtype likewise raises when any value is `NaN` (pass `na_value`).
- Like pandas, `na_value` only changes the missing cells — without an explicit `dtype`
  an int column with NA still exports `float64` (the default), and `na_value` simply
  fills the `NaN` slots.
- For a **lossless** NA round-trip that keeps the native dtype *and* the missing
  positions (no fill, no float collapse), use the Arrow path (`series.to_arrow()` carries
  the null bitmap) or `series.to_pandas(dtype_backend='numpy_nullable')`; the NA mask
  alone is `series.isna().to_numpy()`.

### series.to_arrow() -> pyarrow.Array

A **volas-specific** export of the column to a `pyarrow.Array`, **zero-copy** where
the dtype matches (the numeric / string / datetime buffer is shared; `bool` and the
null bitmap are repacked). Requires `pyarrow` (imported lazily). It is a convenience
over volas's Arrow **C-Data** bridge: any Arrow consumer can read the series directly
through the standard `__arrow_c_array__` PyCapsule protocol.

```py
import pyarrow as pa
arr = series.to_arrow()    # -> pyarrow.Array (shares the buffer)
arr = pa.array(series)     # identical, via the __arrow_c_array__ protocol
```

Returns a `pyarrow.Array`. The column also exports zero-copy to NumPy / PyTorch / JAX
via DLPack (`np.from_dlpack(series)`) — see
[Arrow & DLPack interop](interop.md#arrow--dlpack-interop-zero-copy).

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
`DataFrame(data, time_frame=..., cumulators=...)` (requires a DatetimeIndex).

The frame keeps the bars of the **open period** in a small DataFrame of their own;
its last row is their cumulation. On a tf-aware frame:

- **df.append(bar)** adds the bar to the open period and **replaces the forming
  last row** (`df.iloc[-1]`) with the period's cumulation. A bar with the **same
  timestamp** as one already in the period replaces it, so a re-sent bar never
  double-counts. A bar of a later period starts a new row.
- A bar of the frame's **own** time frame stamped at the period start — e.g. an
  exchange `15m` kline pushed again and again while it forms — therefore
  **replaces the last row whole**.
- **The last constructed row is the forming bar.** `DataFrame(data,
  time_frame=...)` makes it the first bar of the open period, so a finer bar in
  its period folds into it rather than opening a second row with the same label.
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

volas cannot tell how your rows were produced, so it relies on two conventions and
does **not** check them:

- **A bar is labelled with its period start** — `17:00` for the `17:00`–`17:14:59`
  `15m` bar, never the period end. Shift data labelled by its end time first.
- **The last constructed row must not already contain the finer bars appended
  after it.** A `15m` snapshot fetched at `17:09:30` followed by the `17:05` 1m
  bar counts that minute twice. To continue an unclosed bar with finer bars,
  build the frame with `cumulate` from the finer history — its open period then
  holds exactly those finer bars.

See [Cumulation and DatetimeIndex](cumulation.md#cumulation-and-datetimeindex) for details.

### read_csv(path, sep=',', header=True, parse_dates=None, index_col=None, na_values=None, keep_default_na=True, tz=None, date_unit=None) -> DataFrame

A top-level function that reads a CSV file into a `DataFrame`, inferring per-column
dtypes — a fast, pandas-subset CSV reader.

- **path** `str | os.PathLike` the CSV file path — a string or any `os.PathLike`
  (e.g. `pathlib.Path`).
- **sep?** `str = ','` the field delimiter (a single character); `delimiter` is an
  accepted alias.
- **header?** `bool = True` `True` (or omitted) treats the first row as the header;
  `False` / `None` means no header (columns are named `'0'`…`'n-1'`).
- **parse_dates?** `list[str] | None = None` column names to parse into datetime
  columns, from their text tokens (an all-missing column is all `NaT`).
- **index_col?** `str | int | None = None` a column name or integer position to move
  into the row index; applied *after* `parse_dates`, so naming a parsed date column
  yields a `DatetimeIndex`.
- **na_values?** `str | list[str] | None = None` extra missing-value tokens.
- **keep_default_na?** `bool = True` also treat the default NA tokens as missing.
- **tz?** `str | None = None` the timezone for the `index_col` datetime: a *naive* date
  string is read in `tz` (stored UTC, the index tagged with `tz` — `'UTC'` included,
  so the index is tz-aware); the column may also be listed in `parse_dates`. See
  [Timezones](cumulation.md#timezones). Accepts either:
  - a fixed UTC offset, e.g. `'+08:00'` / `'-05:00'`
  - an IANA timezone name, e.g. `'America/New_York'` / `'Asia/Shanghai'` / `'UTC'`
- **date_unit?** `str | None = None` read `index_col` as an epoch number in this unit,
  exactly from its text (absolute UTC, a fractional epoch kept to the nearest
  nanosecond; `tz` then only sets the display zone). One of:
  - `'s'` — seconds
  - `'ms'` — milliseconds
  - `'us'` — microseconds
  - `'ns'` — nanoseconds

```py
from volas import read_csv

df = read_csv('klines.csv')                        # RangeIndex
df = read_csv('klines.csv',
              parse_dates=['time_key'],            # parse to datetime
              index_col='time_key')                # -> DatetimeIndex
df = read_csv('data.tsv', sep='\t', header=False,  # no header -> '0'..'n-1'
              na_values=['NA', 'null'])
```

### to_datetime(obj, unit='ns', format=None) -> Series

A top-level function that converts epoch numbers or datetime strings to a
datetime `Series`, mirroring `pandas.to_datetime`. `obj` may be a `Series`, a 1-D
NumPy array, or a list. A **missing** input (a float `NaN`, or a `volas.NA` in an
int column) becomes `NaT`, like `pd.to_datetime`.

- **obj** the values to convert — numeric epochs, datetime strings, or an
  already-datetime `Series` (returned unchanged).
- **unit?** `str = 'ns'` the epoch unit for **numeric** input (sub-unit fractions are
  preserved, like `pd.to_datetime`). One of:
  - `'s'` — seconds
  - `'ms'` — milliseconds
  - `'us'` — microseconds
  - `'ns'` — nanoseconds (the default)
- **format?** `str | None = None` an explicit datetime format for **string** input
  (pandas `format=`, e.g. `'%Y-%m-%d %H:%M:%S'`) — faster and unambiguous; ignored for
  numeric input. Any `strftime`/`strptime` directive string; `None` auto-infers. Every
  parsed field is honored: an offset (`%z` / `%:z`) makes the value an absolute instant,
  exactly like an offset-aware string without `format`; time fields the format omits
  are zero. `%Z` (a zone *name*) raises `ValueError` — use a numeric offset.

Naive strings parse as UTC and offset-aware strings (`…+08:00`) are absolute. To
*display* the resulting index in a zone, make it the index and tag the zone with
`tz_localize` / `tz_convert` (see [Timezones](cumulation.md#timezones)).

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
