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
  - another volas **`DataFrame`**, which is then copied (like `pandas.DataFrame(df)`):
    `DataFrame(df)` is `df.copy()` — its cached directive columns, a bounded window and
    a time-frame state included. Re-labelled (`index=`) or re-typed (`dtype=`), it is a
    new series of the source's visible data columns (see [Derived frames](#derived-frames));
    a tf-aware source gives the tf-aware frame of its new rows, the last one the forming
    bar (see [Writing to a time-frame frame](cumulation.md#writing-to-a-time-frame-frame)).

  The constructor does **not** accept a `pandas.DataFrame` or an Arrow object — bridge
  those with the dedicated [`DataFrame.from_pandas`](#dataframefrom_pandaspdf---dataframe) /
  [`DataFrame.from_arrow`](#dataframefrom_arrowdata-copyfalse---dataframe) instead. To attach a
  [`DatetimeIndex`](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html),
  parse a column with `to_datetime`, promote it with `set_index`, then tag a zone with
  `tz_localize` / `tz_convert`. See [Timezones](cumulation.md#timezones).
- **columns** `Optional[list[str]] = None` Select and order the columns to keep —
  the same projection as `df[[...]]` (see [Derived frames](#derived-frames)): the same
  rows, each listed cached directive column as a data column of its values (a stale one
  is refreshed first), never continued; a bounded-window source gives its visible rows.
  On a DataFrame source, valid directives are computed on first request; an invalid entry raises the same error as `df[[...]]`, with no partial source changes.
  Dict sources require existing keys (`KeyError` for a missing key). Duplicate identities raise, an empty list keeps no column, and absent columns are never silently filled.
  With `index=`, `dtype=`
  or `window=`, a source `DataFrame` is read like any bulk read: it raises while one
  of its cached directive columns is stale.
- **index** `Optional[list | np.ndarray] = None` Row labels (int64, string or datetime),
  one per row. A label names one row: a repeated label raises `ValueError`, consecutive
  or not (`NaT` is no label); `set_index`, `read_csv(index_col=...)` and
  `DataFrame.from_pandas` label a frame the same way. Bars sent again (the same
  timestamp twice) are fed to a frame with `append`, where a row labelled like the last
  one replaces it, or to a time-frame frame, whose fold keeps the last of them.
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

  A missing value is never skipped: every aggregate is missing as soon as any value
  in the bucket is (an all-missing bucket too — never `±inf`); the live fold and the
  batch `cumulate` agree. A `datetime` column has no `'sum'`, and a `bool` / `str`
  column only `'first'` / `'last'`; an unsupported pair is rejected when the frame
  is built (see [Cumulation](cumulation.md)).
- **window** `Optional[int] = None` Make this a **bounded rolling-window** frame
  showing only the last `window` rows (see [Bounded rolling window](#bounded-rolling-window)).
  Requires `max_lookback`.
- **max_lookback** `Optional[int | list[str]] = None` **Required with `window`** (and
  valid only with it): the hidden margin of rows kept behind the visible ones, and the
  directives the frame computes exactly (see
  [What a bounded window computes](#bounded-rolling-window)). Pass a **list of
  directives** to **declare** them: the margin is the largest of their
  lookbacks (e.g. `['atr:14', 'ma:50']` → margin 49), and each is computed at
  construction over all the given rows and cached — a running accumulator or a recursive
  one (`ma`, EMA, Wilder
  smoothing, a cumulative line, …) can only be computed this way. Or pass an **int**: the
  margin alone, declaring nothing. A raw-column directive (`'close'`) is also valid:
  its lookback is zero and it stays a data column. Equivalent spellings of a physical
  data column are also data declarations, with no computed warm-up. Unknown columns (`'ma50'`) and an
  empty list are rejected, and so is a
  directive whose column name is a data column of the frame (`'macd:'` beside a data
  column `macd`): rename that column to declare it.

### Bounded rolling window

Pass `window=` to cap the frame at the last `window` rows. This is the live-trading
/ NN-input shape: you keep `append`-ing bars forever, but memory stays **bounded** —
the frame retains its last `window + max_lookback` rows (and at least
`max_lookback + 2`, so a re-sent last row can resume from the row before it); once it
holds twice that it drops its head back to it, amortized O(1) per bar. Every indicator
it computes equals what an unbounded frame fed the same rows computes, **bit for bit**,
across every drop, re-sent row, missing value and write.

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

#### What a bounded window computes

A bounded window holds only its recent rows, so it computes an indicator only where it
can do so exactly. Every directive has one of three **history classes** (owned by the
command table): **class 1** depends on a trailing window with no running accumulator
(`roc`, `llv`, `maxindex`, `willr`, `psy`, `close > open`, …); **class 2** on a running
accumulator over a trailing window (`ma`, `wma`, `boll`, `stddev`, `cci`, `mfi`,
`linearreg`, …); **class 3** on every row before — a recursion, a cumulative line or an
unbounded count (EMA, Wilder smoothing, `obv`, `increase`, …). A directive takes the
highest class among its parts.

- **Declared directives** (`max_lookback=[...]`) are computed at construction over every
  row given — the whole history — and cached. A recursive one keeps a **checkpoint** of
  its history: the same continuation state used for append refresh, including every
  stateful node's kernel state, complete input tail and output ring. It moves with each
  drop of the head. A write replays the directive from that state; missing rows emit
  NaN and leave a recursion at its last complete row. During warm-up the checkpoint
  retains the raw input rows until all nodes have a state.
  Declared chains such as `'ema:5@(rsi:5)'`, `'ema:3 - ema:5'`, `'close > ema:20'`,
  `'ema:2@(maxindex:2)'` and `'close-ma:5+obv'` stay exact across appends, compaction
  and visible writes, including on time-frame frames. CRSI, stochastic momentum,
  and composite averages at every MA type also carry their states and can be declared.
- **A directive requested later** (`wf[d]`, `wf[[...]]`, `wf.exec(d)`) is computed only
  when it is of class 1 and reads at most `max_lookback` rows before each row. Classes 2
  and 3 must be declared; a later request raises, naming the two correct paths (for a
  directive already declared, `exec` points to its column instead):

```text
ValueError: cannot compute 'ema:10' on this windowed frame: each of its values depends on every row before it, and the frame keeps only its last 50 rows. Declare it when constructing the frame — DataFrame(data, window=30, max_lookback=[..., 'ema:10']) — and read its column df['ema:10'], or compute it on an unbounded frame.
ValueError: cannot compute 'ma:5' on this windowed frame: it carries a running accumulator (a class-2 directive). Declare it when constructing the frame — DataFrame(data, window=30, max_lookback=[..., 'ma:5']) — and read its column df['ma:5'], or compute it on an unbounded frame.
```

A frame copied into a window (`DataFrame(df, window=..., max_lookback=...)`) keeps the
cached columns it carries only when their lookbacks fit the margin; otherwise the
construction raises.

Assigning a whole column (`wf['x'] = values`, a scalar, an array or a Series of the
visible rows) writes the visible rows only: the hidden margin keeps the history it
holds — the column's own values, converted exactly to the new values' dtype (a value it
cannot hold exactly, `float32` rounding included, raises `TypeError`), or missing values
for a new column.

A frame built from a tail of another (`DataFrame(df.iloc[-50:], window=..., ...)`) keeps a
recursion that tail carries in a state, from that state: it is exact on append. Its
checkpoint and lead retain the history needed to replay a visible write exactly.

#### `ready` — has the window warmed up?

`ready` (a property) is `True` only when every declared directive is past its warm-up
on every visible row, counting the rows it reads complete (non-missing): the first value
of each declared column is at or before the first visible row. It is updated in O(1)
per append; reading fresh values stays the reader's job (`fulfill()` first). A frame
declaring nothing (an int `max_lookback`) is always `True`. Gate your inference /
training on it.

```py
wf = DataFrame(seed, time_frame='15m', window=30, max_lookback=['atr:14'])
wf.ready          # False — atr:14 has no value yet at the first visible row
...               # append (and fulfill) until warmed
wf.ready          # True  — every visible row has an atr:14 value
```

An **unbounded** frame (no `window=`) has no warm-up contract and is **always `True`**.

#### `fill_into(out, columns=None)` — zero-allocation feature export

`fill_into` writes the window's values straight into a NumPy array **you** own, in
place — nothing is allocated per call. It is the export half of a live inference loop:
paired with one preallocated buffer, an `append → fulfill → fill_into → infer` loop
allocates nothing per bar (unlike `to_numpy`, which mints a fresh matrix every call).

Contract:

- **Only existing columns** (data or cached): materialize a directive first with
  `name = wf['atr:14'].name`. Use `df.columns` or that returned name for the exact
  stored label; equivalent directive spellings resolve to the same physical owner.
  A declaration in `max_lookback` is cached at construction. `fill_into` exports
  existing values and never computes a missing directive.
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
atr = wf['atr:14'].name               # read the declared directive's column name back: a
                                      # cached directive lands under its CANONICAL form (here
                                      # 'atr:14', but e.g. 'MA: 5' -> 'ma:5'), not the string
                                      # you passed
COLS = ['open', 'high', 'low', 'close', atr]    # k = 5 features

# One reusable buffer for the whole run — shape (window, k), the model's input tensor.
buf = np.empty((30, len(COLS)), dtype=np.float32)

for bar in feed:
    wf.append(bar)
    wf.fulfill()                      # refresh atr's stale tail before exporting
    if not wf.ready or len(wf) < 30:
        continue                      # warming up: indicators not valid / buf not (30, k) yet
    wf.fill_into(buf, columns=COLS)   # zero-alloc write of the 30×5 window into `buf`
    prediction = model(buf)           # feed the model the same memory every bar
```

> **Shape note.** The shape match is exact against `len(df)`, which **grows during
> warm-up** (1, 2, … up to `window`) and only then settles at `window`. `ready` speaks
> for the indicators only, so the loop above also waits for `len(wf) == window`, and a
> fixed `(window, k)` buffer then always fits. To export mid-warm-up, size `out` to the
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

### DataFrame.from_arrow(data, copy=False) -> DataFrame

Builds a `DataFrame` from the Arrow **C-Stream** protocol (`__arrow_c_stream__`):
`pyarrow.Table` / `RecordBatch` / `RecordBatchReader`, polars `DataFrame`, etc. Imports
borrow matching buffers by default (`copy=False`), with a fresh
`RangeIndex`. Producer buffers are validated before reading: a malformed array (buffer sizes, offsets out of
range or not monotonic, a present string that is not valid UTF-8) raises `ValueError`.
A null string slot's bytes are unspecified in Arrow and are never read.

- **data** the Arrow source — any object implementing `__arrow_c_stream__`.
- **copy** keyword-only `bool = False`. `True` takes an owned snapshot, including multi-chunk sources. `False` borrows matching numeric and dense ns-datetime buffers. Arrow data is immutable. The producer and every alias must never write borrowed memory again while any imported object or view survives. Other layouts still copy.

```py
df = DataFrame.from_arrow(pa_table)        # pyarrow.Table     -> DataFrame
df = DataFrame.from_arrow(polars_df)       # polars.DataFrame  -> DataFrame
```

> Arrow is **not** accepted by the `DataFrame(data=...)` constructor (which takes a
> `dict` or another `DataFrame`); use `from_arrow` for whole Arrow tables. Arrow arrays inside a `dict` are supported and copied.

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

A data column the dict leaves out is a missing value of the bar, exactly as in a
`DataFrame` / `Row` bar, and an unknown key raises. The
timestamp is read in its own zone: an aware `Timestamp` / `datetime` / offset string is
an instant, a naive one a wall clock — judged against the frame's index like any other
row (a naive bar onto a tz-aware frame raises), a raw epoch-ns count — an integer or a
NumPy `datetime64`, the form instants are exported in — read in the frame's zone. A label
read back from the frame (`df.index[-1]` — a UTC `datetime64` on a tz-aware frame — or a
row's `.name`, an aware `Timestamp`) is
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

  Every aggregate of a bucket holding a missing value is missing, and an aggregator
  the column's dtype does not support raises up front (see
  [Cumulation](cumulation.md)).

Only the data columns are cumulated: a cached directive column of the caller (say
`ma:20` over 1-minute closes) is not carried into the coarse frame — requested there,
it is computed over the coarse bars.

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

How a derived frame keeps the cached columns of its source is stated in
[Derived frames](#derived-frames).

### Derived frames

A frame derived from another either **continues its series** or **starts a new one**:

- A **contiguous row selection** — the rows `[a, b)` in order, however it is written:
  `df.iloc[a:b]`, `df[a:b]`, `df.loc[lo:hi]` (a partial date string too), `df.loc['2024-01']`
  on an increasing index, `head` / `tail`, `iloc[a:b, :]` / `loc[lo:hi, cols]`, a boolean
  mask or position list that selects consecutive rows, `drop` of the rows around them,
  `dropna` removing a warm-up — continues the series. Every cached directive column stays
  cached with its values, and an `append`, a re-sent last row or a `fulfill` continues it
  bit for bit as the source would have — and so does a selection of a selection. For a
  recursion carried by its state (EMA, Wilder smoothing, SAR, a cumulative line, …) the
  selection keeps the nearest checkpoint of its history before its first row and the
  state at its last (the source's own for a selection ending at its last row, else
  computed the first time the column continues, from the nearest anchor before it); it
  keeps the data rows before its
  first row that its columns read — back to that checkpoint — however short it is (a
  missing row there counts as the recursion skips it). After a write, every column is
  recomputed from those, exactly as the source would recompute it. Every recursion,
  including recursive moving-average types and expressions combining recursive nodes,
  carries the continuation owner's state: kernel state, complete input tail, and any
  node output rings. Slices retain their required checkpoint and input tail rather
  than the entire raw prefix.

  A recursion's checkpoints — its **anchors** — are kept every 512 rows of its history,
  formed by the computation that passes them (caching or recomputing the column over its
  whole history). A selection keeps the nearest one before its first row together with
  the rows after it, so taking it replays nothing; the first `append` or `fulfill` of a
  selection ending before its source's last row replays its last states from the
  nearest anchor, at most 512 rows per recursive column.
- A **column projection** — `df[[...]]`, `DataFrame(df, columns=[...])`, `df.loc[:, [...]]`
  — keeps the rows and only the columns it lists, each a **data column**: a
  listed cached directive column keeps its values exactly as the source shows them (under
  its canonical name: `df[['close', 'ma:05']]` gives `['close', 'ma:5']`) and is never
  continued afterwards — even when the projection also holds the directive's inputs. An
  `append` treats it like any data column: a bar without it leaves it missing, a bar with
  it gives that value. An unlisted cached column is dropped, and the projection keeps no
  history before its rows. New directives start at computation position zero on the
  projection's own rows; index labels are preserved. A failed projection leaves the
  source frame, including its caches and freshness, unchanged.
- A **column removal** — `drop(axis=1)` — removes the columns it names and keeps everything
  else as it is: the remaining cached columns stay cached and continue, and the rows only
  the removed ones read before the first row are released.
- **Every other derivation is a new series**: a selection of rows that are not consecutive
  or not in order (a mask with gaps, `iloc[[0, 2, 5]]`, `sort_index(ascending=False)`,
  `nlargest`), `astype`, `rename`, `set_index` / `reset_index`, a new `index=` or `dtype=`,
  `tz_localize` / `tz_convert`, and every **value transform** that builds a frame from
  the cells (`fillna`, `ffill` / `bfill`, `isna` / `notna`, `round`, `abs`, `clip`,
  `cumsum` / `cummax` / `cummin` / `cumprod`, `diff`, `shift`, `rank`, `isin`, `replace`,
  `interpolate`, `where` / `mask`, `df == x` / `df != x`, the `rolling` / `expanding` /
  `ewm` aggregations). A value transform reads every column as a projection gives it: a
  cached column's transformed values are a plain data column under its name, never
  continued. A cached column read as a Series (`df['ma:5']`) is its values: `to_frame()` /
  `reset_index()` make it a data column. The other derivations above hold the source's
  data columns only: a cached column is dropped, and requested again it is computed from
  the new frame's own rows. A 2-D
  selection of rows that are not consecutive (`df.loc[mask, cols]`) naming a cached column
  raises `KeyError` — read its values as a Series, `df.loc[mask, 'ma:5']`.
- `copy()` / `DataFrame(df)` copy everything.

A derivation of a frame whose cached columns are stale raises, like any bulk read (call
`fulfill()` first); `copy()` proceeds. A derived frame of a bounded window is an unbounded
frame of its visible rows (its hidden margin serves as the rows before them); a
time-frame frame's model is kept where the result is still a frame of its bars (see
[Live cumulation](#live-cumulation--a-tf-aware-dataframe)). A bounded window's own drop of
its head is not a derivation: its cached columns stay cached, a recursion with the
checkpoint of its history (see [What a bounded window computes](#bounded-rolling-window)).

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
  - `'numpy'` — the most ecosystem-compatible form: an int column with a missing
    value becomes `float64` with `NaN` (like `pandas.Int64.to_numpy()`); a bool column with a missing value becomes the masked `boolean` dtype (`pd.NA`), never a truthy `NaN`.
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

`df[col]` and `df[directive]` return a named 1-D `Series`. See the
[Series API](series-api.md) for operators, datetime access, and export methods.

`dt.strftime(fmt)` applies the same format language as `Timestamp.strftime(fmt)`.
`%f` writes six fractional digits; `%.f` writes the nonzero fraction in groups of
three digits, up to nanoseconds. Missing values stay `NA`. Format errors are
reported when formatting the first present value; an empty or all-NaT Series
returns its empty or missing result.

See {ref}`Series Arrow import <series-from-arrow>`.

### series.to_numpy(dtype=None, na_value=...) -> np.ndarray

See {ref}`Series NumPy export <series-to-numpy>`.

### series.to_arrow() -> pyarrow.Array

See {ref}`Series Arrow export <series-to-arrow>`.

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
- **Every bar is a complete bar of the frame's data columns.** A data column a bar
  leaves out — in any append form — is a missing value of that bar, so the period's
  aggregate of that column is missing, exactly as a batch `cumulate` of the same bars
  gives; a later bar re-sent with the value replaces it.
- **A write is taken only where the open period follows it exactly**: a new column
  enters the fine bars as missing (the next fold makes its forming cell missing), and
  a new forming value over several folded fine bars raises `ValueError` — see
  [Writing to a time-frame frame](cumulation.md#writing-to-a-time-frame-frame).
- **The last constructed row is the forming bar.** `DataFrame(data,
  time_frame=...)` makes it the first bar of the open period, so a finer bar in
  its period folds into it rather than opening a second row with the same label.
- **df.iloc[-1]** is the current (still-open) period — the live bar.
- **df[directive]** / **df.exec(directive)** computes indicators over the
  cumulated frame including the forming row — lazily, on read: an `append` only
  marks them stale, and the next read recomputes just the tail.
- **df.cumulate(target)** must be a whole multiple of the source frame (e.g.
  `5m→15m`, not `5m→7m`; a week or 3-day bar does not nest into a month/year);
  at the same time frame it re-aggregates the current data rows with the requested
  cumulators (see [Cumulation](cumulation.md)).
- **A derived frame of the same bars stays tf-aware.** A selection ending at the forming
  row (`df.iloc[a:]`, `df.tail(n)`, `df.loc[t:]`) keeps the open period, so the next finer
  bar still folds into it; a projection and `rename` (each column keeps its aggregator)
  keep it too; `tz_localize` / `tz_convert` keep it while every bar is still a period of
  the grid in the new zone. Any other selection of bars in time order (one ending before
  the forming row, a mask), `astype`, `set_index` and a new `index=` / `dtype=` give the
  tf-aware frame of the resulting bars, the last one forming, as
  `DataFrame(bars, time_frame=...)` builds it. Rows out of time order, a
  non-datetime index, or bars that are no longer periods of the grid in the new zone (day
  bars of `Asia/Shanghai` seen in `UTC`) give a plain frame.

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

Empty, blank and missing string cells become `NaT` before format validation.
An empty or all-missing string input therefore does not raise for an invalid
format. Otherwise errors follow input order and name the first failing text.

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
it under. The command name is lowercased, each argument is written as its value, and
default arguments / series are dropped to save space, so every spelling of a directive
has one name (see [Cached column names](directives.md#cached-column-names)).

```py
from volas import directive_stringify

directive_stringify('kdj.j')
# 'kdj.j'

directive_stringify('kdj.j:9,3,2,100@high,close,close')
# 'kdj.j:,,2,100@,close'

# command names are case-insensitive and canonicalize to lowercase; default
# arguments are dropped
directive_stringify('MACD:12,26')
# 'macd'

# arguments are written as their values
directive_stringify('ma:05')
# 'ma:5'
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
