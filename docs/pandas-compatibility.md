# pandas-compatible API

## The rest of the pandas-compatible API

Everything below behaves like its `pandas` counterpart — if you know it from
pandas, it works the same in volas, except for the deliberate
[NA-model divergences](#known-pandas-divergences-the-volasna-model) noted after
the listing.

```py
# --- DataFrame: metadata --------------------------------------------------
df.columns / df.shape / len(df) / df.dtypes      # dtypes -> dict
df.index                          # row labels, as a NumPy array (a tz-aware index: its UTC instants)
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
df.rename({old: new}) / df.astype({col: dtype}) / df.set_index(col)  # astype: data columns only
df[[]]                                  # no columns over the same rows (pandas)
df.astype({col: 'datetime64[s]'})  # numeric epoch -> datetime (unit s|ms|us|ns; truncating)
df.copy() / df.equals(other) / df.to_csv(path=None, ...)   # to_numpy: see its own section (dtype, na_value)

# --- DataFrame: writing ---------------------------------------------------
df[col] = scalar | array | Series          # add / replace a column (Series: same index)
df.loc[mask, col] = value ; df.iloc[i, j] = value ; df.at[label, col] = value

# --- Series ---------------------------------------------------------------
s.name / s.dtype / len(s) / s.tz / s.index
s.to_list()                       # to_numpy has NA caveats -> see its own section (dtype, na_value)
s.iloc[...] / s.loc[...]
s + s, s - 1, -s, ...             # elementwise arithmetic
s > 0, s == t, s != t, ...        # comparison -> bool Series
s & t, s | t, ~s, s ^ t           # logical -> bool Series
s.sum() / s.mean() / s.min() / s.max() / s.std() / s.var() / s.median()   # skip missing
s.shift(n=1) / s.diff(n=1) / s.fillna(v) / s.ffill() / s.bfill()           # see Missing values: NA keeps the dtype
s.isna() / s.notna() / s.dropna() / s.equals(t)
```

### Window operations (`rolling` / `expanding` / `ewm`) — compatibility only

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

With `min_periods=0`, empty rolling / expanding windows return 0 for `sum`
and `nunique`; statistics without an empty identity remain missing. Rolling,
expanding and EWM treat NaN and both infinity signs as missing computation inputs.
The window preparation matches pandas 3.0.3; volas also excludes infinities in
`count()`, whereas pandas builds that method's non-NA mask before window preparation.
pandas `sem()` uses the same raw count as its denominator; volas excludes infinities there too.

Scalar `replace` preserves dtypes: a matched column that cannot hold the new value
raises and leaves the source frame unchanged. Numeric membership and replacement
match equal integer / float values across storage dtypes without rounding large
integers. `rank(na_option='top'|'bottom')` treats missing cells as one tie group,
including percentile ranks. `DataFrame.mode()` keeps one mode per column; a
missing-only column is padded with NA when another column has a mode, and a frame
with only missing values returns zero rows.

`center=True` labels each window at its center — it reads **future** bars
relative to the label. That is exactly what a labeling pass wants, and exactly
what a live signal must never do; it is supported for the former.

Time-based windows (`rolling('5min')` / a `timedelta`) are deliberately not
implemented. For multi-timeframe computation, maintain **two tf-aware
DataFrames** (see [Cumulation](cumulation.md#cumulation-and-datetimeindex)) and `append`
each bar to both — that is the supported, `O(lookback)`-per-bar design;
emulating a coarser timeframe through window arithmetic recomputes everything
on every bar.

Not provided (pandas members that conflict with volas's model): `apply` /
`agg` / `pipe` (arbitrary-Python-per-window), `win_type`, `step`, `on`,
`closed`, `method`, `ewm(times=...)`, `ewm.online()` — `append()` + directives
already cover the streaming use case.

### Known pandas divergences (the `volas.NA` model)

A handful of APIs diverge from pandas **by design**, because volas stores missing
values natively as [`volas.NA`](#missing-values-volasna) (no `object` dtype, no
silent float upcast):

- **`shift` / `diff` / `fillna` and friends** keep the column's dtype — a missing
  value is `volas.NA`, not an int/bool/str column upcast to float/object.
- **Comparisons** (`==` `!=` `<` `<=` `>` `>=`) return a *non-nullable* bool mask:
  a missing value compares `False` (and `!=` compares `True`), following IEEE /
  NumPy — not pandas-nullable's three-valued `NA`. This keeps masks free of `NA`
  so `df[mask]` and assignment stay total. A **directive** boolean
  (`df['close > ma:20']`) is different: it is a signal, so a row whose operand is
  missing (an indicator's warm-up rows) is `volas.NA`, never a made-up `False` —
  see [Directives](directives.md#operators).
- **Cumulation never skips a missing value**: a period's `first` / `last` / `max` /
  `min` / `sum` is missing as soon as one of its values is — pandas `resample`
  skips missing values instead (see [Cumulation](cumulation.md)).
- **Storage keeps the dtype.** Where pandas upcasts an int/bool column with a missing
  value to `float64` / `object`, volas keeps it `int64` / `boolean` with the hole as
  `volas.NA` — so `to_list()` returns exact ints and `volas.NA`. The numpy *export*
  (`to_numpy()`) still follows pandas 3.0 exactly: a missing cell becomes `NaN` / `NaT`
  by default, an integer `dtype=` over missing values raises, and `na_value=` fills —
  see the dedicated `df.to_numpy` / `series.to_numpy` sections above.

For the full picture — why volas's type system is built this way, where pandas's
breaks, and the migration gotchas — see
[volas vs pandas — the type system](https://github.com/kaelzhang/volas/blob/main/PANDAS-DIFFERENCES.md).

The pandas-shaped indexing and writing details have their own sections —
[Indexing & selection](#indexing--selection) and
[Writing & assignment](#writing--assignment).

## Indexing & selection

A pandas-compatible subset for label and positional access. The row index may be
a range, a `DatetimeIndex`, an integer index, or a **string index**.
Both axes of `iloc` / `loc` accept one-dimensional bool masks, including NumPy
arrays with positive or negative strides (`df.iloc[:, column_mask]` also works
for scalar assignment). For array assignment, select one column. A bool dtype
never becomes integer positions; scalar or
multidimensional bool selectors raise. An empty list selects no rows or columns.
A boolean Series key must have the same labels as the selected axis, in the same
order (row masks use only the visible window; column masks use column names).
Index names and display zones are ignored; mixing naive and aware datetime
indexes raises. NumPy masked arrays preserve their mask: any unknown boolean
signal raises until explicitly filled or dropped. This applies to reads and writes.

`DataFrame.equals` and `Series.equals` compare index labels, ignoring index names
and display zones; a naive / aware datetime pair raises. Different labels return
`False`. Series equality also requires the same dtype and values; Series names
are ignored.

Column names in `loc`, `at` and export selections resolve equivalent directive
spellings to the stored physical label; these reads never materialize a missing
column (use `df[directive]` first).

```py
df.iloc[2]          # a Row by position (row.name is its index label)
df.iloc[10:]        # a DataFrame slice by position
df.loc[label]       # a Row by index label
df.loc[lo:hi]       # inclusive label slice (lexicographic for string indexes)
df.at[label, col]   # a scalar by label + column
df.iat[i, j]        # a scalar by position
df.index            # the row labels, as a NumPy array (a tz-aware index: its UTC instants)
```

String (symbol) index — `set_index` on a string column, then look up by symbol:

```py
df = DataFrame({'sym': ['aa', 'bb', 'cc'], 'px': [1.0, 2.0, 3.0]}).set_index('sym')
df.loc['bb']           # the row keyed 'bb'
df.loc['aa':'bb']      # inclusive, lexicographic slice
df.at['cc', 'px']      # 3.0
df.drop(['bb'])        # drop by string label
```

A label slice follows the index's label order, as in pandas. Integer and string
labels only need to be unique, so they may come in any order:

- **increasing labels** select every row with `lo <= label <= hi`; a bound need
  not be a label of the index;
- **decreasing labels** select every row with `lo >= label >= hi`;
- **any other order** locates each bound by its exact label and selects the rows
  between the two positions. A bound that is not a label of the index raises
  `KeyError`, because comparing by value would select (or, in
  `df.loc[lo:hi, col] = v`, overwrite) rows outside the requested labels.

```py
df = DataFrame({'x': [20, 0, 10]}, index=[2, 0, 1])
df.loc[0:1]            # the rows labelled 0 and 1
df.loc[0:1, 'x'] = 99  # writes those two rows only: x == [20, 99, 99]
df.loc[0:5]            # KeyError: 5 is not a label of this unordered index
```

On an increasing datetime index a bound written as a **partial date** names a whole
period, as in pandas: a year (`'2024'`), a month (`'2024-01'`), a day, an hour
(`'2024-01-02 10'`), a minute or a second; a start bound is the period's first instant
and a stop bound its last, so `df.loc['2024-01':'2024-02']` holds both whole months. A
single year or month key (`df.loc['2024-01']`) selects its period's rows the same way.
Either is a contiguous selection, so cached directive columns continue on it (see
[Derived frames](dataframe-api.md#derived-frames)). A **column** selection
(`df[['close', 'ma:5']]`, `df.loc[:, cols]`) and a value transform (`fillna`, `abs`,
`where`, …) give plain data columns, as pandas does: a cached directive column becomes
its values and is never continued afterwards. `drop(axis=1)` only removes the columns it
names: the remaining cached columns keep continuing.

A datetime index holding `NaT` counts as unordered, as in pandas. On a datetime
index that is not increasing (decreasing, unordered, or holding `NaT`), a bound
written as a **string** raises `KeyError`: pandas reads it as a partial-date value
and may select rows that are not contiguous, which volas does not reproduce, so it
refuses rather than select different rows. Use `Timestamp` bounds there, or sort
the index first. A bound of the other label kind (a string against an integer
index, an integer against a string index) raises `TypeError`.

### Differences from pandas (vs pandas)

volas is pandas-shaped on the surface, but its **type system is deliberately
different** in more than the index: missing values keep their dtype, there is no
`object` dtype, value-returning methods stay `Series`, and a lossy conversion
raises instead of degrading silently. **See
[volas vs pandas — the type system](https://github.com/kaelzhang/volas/blob/main/PANDAS-DIFFERENCES.md) for the full
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
- **Duplicate labels** — a label names one row: a frame built with repeated labels
  (`index=`, `set_index`, `read_csv(index_col=...)`, `from_pandas`) raises `ValueError`.
  A bar sent again is appended, where a row labelled like the last one replaces it.

If your workflow needs any of these, keep using pandas; volas targets the
single-level, OHLCV-shaped index that candlestick data uses.

## Writing & assignment

Assign a whole column, or write into a positional / label / boolean selection
(copy-on-write under the hood). A Series value must carry the frame's index (the same
labels; the alignment rule of every operand) and is then paired by position — volas
never re-aligns by label, so a differently-labelled Series raises. A scalar — a Python
`int` / `float` / `bool` / `str`, a NumPy scalar or a 0-d array — fills the column and
keeps its exact type: an int is an `int64` column (never a float), `np.int32(7)` an
`int32` one, `np.float32(0.5)` a `float32` one, `np.array(True)` a `bool` one. An array
of any other shape, a list or a tuple is values, one per row (a one-element array on a
two-row frame is a length error, never a broadcast). A boolean-frame mask
(`df[mask] = v`) pairs with the columns by **name** and must carry the frame's index.
Every cell write (`iloc` / `loc` / `iat` / `at`, `df[mask] = v`) lands in place, all or
nothing, and on a numeric, bool or datetime column costs time in the cells it writes,
not in the frame's length: no column is copied (unless a Series or an export still
shares it), and a label is found by a binary search over an ordered index — on a plain,
time-frame or windowed frame alike. Three writes do cost time in the length: a write
into a `str` column rebuilds the column's contiguous text bytes; a boolean-mask write
reads its whole mask, one flag per row; and on a windowed frame a label-slice write
(`df.loc['a':'b', col] = v`) first lays out the visible rows' labels. Like a bulk read, a write raises while a cached directive column is stale; a mask that
selects no row changes nothing. On a bounded (`window=`) frame a whole-column assignment
writes the visible rows only and keeps the hidden margin's history (see
[Bounded rolling window](dataframe-api.md#bounded-rolling-window)).

```py
df['signal'] = 0.0                      # add / replace a column (scalar | array | Series)
df.iat[3, 0] = 99.0                     # one cell by position (one value: an array raises)
df.at[label, 'close'] = 99.0            # one cell by label + column
df.iloc[10:20, 0] = 0.0                 # a column slice
df.loc[df['close'] > df['open'], 'signal'] = 1.0   # masked column assignment
```

Writing a fractional value into an integer column **raises** — the int dtype is
kept, and a lossy write errors rather than silently widening to float (see
[Differences from pandas](https://github.com/kaelzhang/volas/blob/main/PANDAS-DIFFERENCES.md); writing `volas.NA` / `None`
keeps the int dtype and marks the cell missing). Writing into a cached directive
column drops its cached status, so a later `fulfill()` can never silently
overwrite your edit.

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

Window computations treat NaN and ±inf as missing inputs. One intentional pandas
3.0.3 exception is `rolling.count()` / `expanding.count()`: pandas builds a
`notna()` mask before preparing numeric window values and therefore counts raw
infinities; volas excludes them. pandas `sem()` inherits that raw-infinity
count in its `std / sqrt(count)` denominator. Compare `count()` and `sem()`
against pandas after replacing ±inf with NaN. Element-wise column arithmetic and comparisons keep IEEE behavior.
