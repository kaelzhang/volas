# volas vs pandas — the type system

volas is **pandas-shaped on the surface** (the same `DataFrame` / `Series` verbs,
the same indexing) but its **type system is deliberately different**. The surface
familiarity is a convenience; the type system is the actual product.

This document explains:

1. [Why volas's type system is built this way](#1-why-volass-type-system-is-built-this-way)
2. [Where pandas's type system breaks today](#2-where-pandass-type-system-breaks-today)
3. [The differences in practice — examples and gotchas](#3-the-differences-in-practice)
4. [When to use pandas instead](#4-when-to-use-pandas-instead)

## At a glance

| | pandas | volas |
|---|---|---|
| An `int` column gains one missing value | silently becomes `float64` | stays `int64`, the cell is `volas.NA` |
| Integer precision past 2⁵³ after a missing | lost (`float64`) | exact (dtype never changes) |
| A mixed-type row (`iterrows`) | an `object` Series | a typed `Row` record (each cell keeps its column dtype) |
| `Series([...]).unique()` | `ndarray` (nullable int → `float64`+`NaN`) | a `Series` (keeps dtype + `volas.NA`) |
| A numeric op on a `str` column | mixed: `sum` concatenates, `mean` raises, … | always raises (a `str` is not numeric) |
| `x > y` where `x` is missing | three-valued `<NA>` (nullable) or `False` (numpy) | `False`, a non-nullable bool mask |
| Catch-all `object` dtype | yes (and it is everywhere) | **never** — no `object` dtype exists |
| Missing → dtype change | implicit and silent | never internal; only at an explicit export boundary |
| Datetime storage | `datetime64[ns]` (`i64`) | `i64` UTC epoch-ns (never funneled through `f64`) |

The rest of this document is the *why* behind that table.

---

## 1. Why volas's type system is built this way

volas is built for **live quantitative trading**, where the worst possible failure
is not a crash — it is **silent wrong data** that flows into a signal, an order, or
a backtest and is only noticed after money moves. A `float64` column that should
have been `int64`, a `NaN` that should have been a real value, an all-`NaN` feature
column produced by a typo — none of these raise; they just quietly produce wrong
numbers of the right shape.

So volas's type system is built on one principle: **dtype is honest end to end, and
anything that cannot be done losslessly fails loudly instead of degrading quietly.**
That principle is enforced by four rules.

### C1 — the container is decided by row correspondence

pandas decides a method's return type by an old "value-level vs index-preserving"
distinction, which is why `unique()` returns a bare `ndarray` while `sort_values()`
returns a `Series` — and why, in the nullable era, `unique()` had to start returning
an `ExtensionArray` instead. volas uses one rule: **does the result still line up,
row for row, with column-shaped data?**

- equal-length / element-wise / filtered / sorted / sliced / **de-duplicated** → a
  `Series` (which carries the dtype, the validity, and an index);
- a reduction (one value, no rows) → a **dtype-aware numpy scalar** (`np.int64`,
  `np.float64`, `np.bool_`);
- "I want to leave volas" → an **explicit boundary method** (`to_numpy`, `to_list`,
  `to_pandas`).

numpy is treated as an **export format, not a compute container**. Every pandas
method that returns a numpy array by default is a forced, lossy export; volas keeps
data in dtype-faithful containers until you explicitly ask to leave.

### C2 — missing values are native (`volas.NA`), and never change a dtype

A missing value is a first-class symbol, [`volas.NA`](docs/pandas-compatibility.md#missing-values-volasna),
stored per dtype: a `float` uses in-band `NaN`, an `int` / `bool` / `str` uses a
validity bitmap (the dtype is preserved), a `datetime` uses `NaT`. Adding a missing
value to a column **never** upcasts it. The pandas-3.0 "missing → widen the dtype"
behaviour exists in volas only at the export boundary (`to_numpy`, `to_pandas`),
never as an internal semantic.

### C3 — every `Series` is a single dtype; there is no `object`

A `Series` is always exactly one of `float64 / float32 / int64 / int32 / bool /
str / datetime64[ns]`. There is **no `object` dtype** — the pandas catch-all that
trades away both type safety and performance. The only place a heterogeneous numpy
array can appear is `df.to_numpy()` on a genuinely mixed frame, because numpy itself
has no other way to express it — and that is an export, not storage.

### C4 — a lossy implicit conversion raises immediately

When an operation would require a dtype conversion that cannot be done legally and
losslessly, volas **raises** instead of producing `NaN` / `object` / a precision-lost
value / a placeholder. Promoting `int` to `float` for a fractional fill, or `bool`
to `int`, is lossless and allowed; pushing a `str` column through a numeric kernel,
writing a fractional value into an `int` column, or narrowing out of range is not,
and it errors.

Together these mean: **what you put in is what you get out, with the dtype intact —
or you get a clear exception, never a quiet wrong number.**

---

## 2. Where pandas's type system breaks today

pandas grew up as a general-purpose analytics library on top of numpy, and numpy
has no missing-value concept and no string dtype. The workarounds pandas adopted —
upcasting to `float` and falling back to `object` — are the source of most of its
type-system problems.

### 2.1 A single missing value silently destroys an integer column

This is the canonical pandas footgun:

```py
>>> import pandas as pd
>>> pd.Series([1, None, 3]).dtype
dtype('float64')          # one None turned an int column into floats
```

It is not just cosmetic. `float64` cannot represent every `int64`, so the upcast
**silently loses integer precision past 2⁵³**:

```py
>>> big = 2**53 + 1                       # 9007199254740993
>>> pd.Series([big, None]).iloc[0]
9007199254740992.0                        # the +1 is gone, forever
```

A trade id, an order id, a nanosecond timestamp, a sequence number — any large
integer that shares a column with one missing value is quietly corrupted.

volas keeps the integer and marks only the hole — the value survives exactly:

```py
>>> import volas
>>> volas.DataFrame({'qty': [1, None, 3]})['qty'].to_list()
[1, <NA>, 3]                              # int64 stays int64; the hole is volas.NA
>>> volas.DataFrame({'qty': [2**53 + 1, None]})['qty'].to_list()[0]
9007199254740993                          # exact — no float upcast, no lost bit
```

### 2.2 `object` dtype — type safety and performance, both gone

When pandas cannot find a single numpy dtype, it falls back to `object`: a column of
boxed Python pointers. It is the dtype of every string column, every mixed column,
and — crucially — **every row pandas hands you**:

```py
>>> pdf = pd.DataFrame({'price': [100.0], 'qty': [5]})
>>> next(pdf.iterrows())[1].dtype
dtype('float64')          # qty (an int) silently became a float...
# ...and for a frame with a string column, the row would be dtype: object,
# where every value is a boxed Python object and per-column dtype is lost.
```

`iterrows()` is one of pandas's most infamous traps: it squeezes a heterogeneous
row into a single `Series`, which forces `object` (or a lossy upcast), which is both
type-unsafe and an order of magnitude slower than column-wise work.

volas has no `object` dtype and never squeezes a row into one shared type — a `Row`
is a typed record that keeps each column's own dtype:

```py
>>> import volas
>>> row = volas.DataFrame({'price': [100.0], 'qty': [5]}).iloc[0]
>>> row.to_dict()                         # each cell keeps its column's dtype
{'price': 100.0, 'qty': 5}                # no boxed objects, no int -> float upcast
```

### 2.3 Silent, inconsistent coercion

Because reductions funnel through numpy, applying them to the "wrong" type produces
surprising results rather than errors — and pandas is not even internally consistent
about which:

```py
>>> pd.Series(['a', 'b', 'c']).sum()      # string "sum" is concatenation
'abc'
>>> pd.Series(['a', 'b', 'c']).mean()     # but "mean" raises
TypeError
```

`sum` concatenating strings is a genuine surprise in a numeric pipeline; that `sum`
and `mean` disagree about whether a string column is even a valid input makes it
worse.

volas refuses a numeric reduction on a non-numeric column — the same way for *every*
reduction, so there is no per-op surprise:

```py
>>> import volas
>>> s = volas.DataFrame({'s': ['a', 'b', 'c']})['s']
>>> s.sum()
TypeError: a numeric operation is not supported on a str column
>>> s.mean()
TypeError: a numeric operation is not supported on a str column   # identical to sum()
```

### 2.4 The nullable retrofit is opt-in, incomplete, and splits return types

pandas does have a newer, better answer — the nullable `Int64` / `boolean` /
`string` extension dtypes and the `pd.NA` scalar (PDEP-16). But it is **opt-in** (the
default `int` column is still the un-nullable numpy one), only **partially adopted**
across the API, and it **splits return types**: the same `unique()` returns a numpy
`ndarray` for a default column and an `ExtensionArray` for a nullable one. You end up
having to know, per call, which type system you are in.

volas does not retrofit nullability onto a non-nullable core — **it is nullable from
the ground up**, with one consistent set of return types. The same call returns the
same type whether or not the column has a hole:

```py
>>> import volas
>>> type(volas.DataFrame({'x': [3, 3, 1]})['x'].unique()).__name__      # dense column
'Series'
>>> type(volas.DataFrame({'x': [3, None, 1]})['x'].unique()).__name__   # column with a hole
'Series'                                                                # same return type
>>> volas.DataFrame({'x': [3, None, 1]})['x'].unique().to_list()
[3, <NA>, 1]                                                            # int64 + volas.NA, kept
```

### 2.5 The same column changes dtype between loads (financial-data scenario)

A live pipeline ingests OHLCV one file per session and concatenates the history.
`volume` is a whole-share integer — until one session's feed has a gap. pandas
infers each file's dtype **independently**, so the *same logical column* is `int64`
on clean days and `float64` on the day with a hole:

```py
>>> import io, pandas as pd
>>> clean = "ts,volume\n1,100\n2,200\n3,300\n"
>>> gappy = "ts,volume\n1,100\n2,\n3,300\n"          # one session missing volume
>>> pd.read_csv(io.StringIO(clean)).volume.dtype
dtype('int64')
>>> pd.read_csv(io.StringIO(gappy)).volume.dtype
dtype('float64')                                      # SAME column, different dtype
```

The dtype now depends on *which day you happened to load*, not on what the data
means. Every downstream step inherits that instability: a `groupby` / `merge` key
silently mismatches `int` vs `float`, an `==` check that held yesterday fails today,
and large volumes lose precision on the float days (§2.1).

volas decides a column's dtype from the **values**, and a gap is `volas.NA` without
changing it — the column is the same dtype on every load:

```py
>>> import volas
>>> volas.read_csv('clean.csv')['volume'].dtype
'int64'
>>> col = volas.read_csv('gappy.csv')['volume']        # the day with a hole
>>> col.dtype, col.to_list()
('int64', [100, <NA>, 300])                            # still int64; the hole is NA
```

*Why the type system is built this way:* **C2** makes missingness orthogonal to
dtype, so identical data always types identically — reproducibility a backtest
depends on.

### 2.6 A join silently rewrites values that were never missing

The upcast is not confined to the rows that go missing. **Any** pandas operation
that *introduces* a hole — `merge`, `reindex`, `align`, `concat`, `unstack` —
upcasts the **whole** column to `float64`, rewriting the present, correct values
along with it:

```py
>>> import pandas as pd
>>> fills  = pd.DataFrame({'sym': ['AAA', 'BBB'], 'qty': [1, 2]})
>>> master = pd.DataFrame({'sym': ['AAA'],
...                        'shares_out': [9007199254740993]})   # 64-bit, int64
>>> m = fills.merge(master, on='sym', how='left')   # BBB has no match -> NaN row
>>> m['shares_out'].dtype
dtype('float64')                                    # the column got upcast...
>>> m['shares_out'].iloc[0]                          # ...so AAA's MATCHED value:
9007199254740992.0                                  # the exact int lost its last bit
```

AAA matched perfectly and was never missing, yet its share count is now wrong —
purely because a *different* row (BBB) was unmatched. A number you can see and have
already validated is corrupted by a hole elsewhere in the column.

volas has no path that does this: introducing a missing value keeps the `int` dtype
and touches only the hole, so every present value stays bit-exact (shown here with
`where`, which masks the unkept rows to NA):

```py
>>> import volas
>>> s = volas.DataFrame({'shares_out': [9007199254740993, 9007199254740995]})['shares_out']
>>> keep = volas.DataFrame({'keep': [True, False]})['keep']
>>> s.where(keep).to_list()
[9007199254740993, <NA>]                            # present value exact; only the hole is NA
```

*Why the type system is built this way:* **C2** keeps NA out of the dtype, so an
NA-introducing operation can never silently rewrite a value that was already there.

### 2.7 A missing trade signal silently becomes a trade

A boolean signal column that picks up a missing value — a `reindex`, an `np.nan`
flowing in from an upstream gap, a hand-assembled column — is no longer `bool`; it
is `object`. The moment anything coerces it back into a usable mask, the unknown
bar resolves to a hard `True`, because `bool(np.nan)` is `True`:

```py
>>> import numpy as np, pandas as pd
>>> sig = pd.Series([True, np.nan, False])   # "go long", with one UNKNOWN bar
>>> sig.dtype
dtype('O')                                   # object — no longer a bool column
>>> sig.astype(bool).to_list()               # forced back into a boolean mask...
[True, True, False]                          # the UNKNOWN bar now fires a trade
```

"I don't know" silently became "yes, enter" — the most expensive direction for the
error to take, and invisible until it has already placed orders.

volas keeps a bool column `bool` with the gap as `volas.NA`, and **refuses** to act
on an NA-carrying mask rather than guess its truth value:

```py
>>> import volas
>>> df = volas.DataFrame({'sig': [True, None, False], 'px': [1.0, 2.0, 3.0]})
>>> df['sig'].dtype, df['sig'].to_list()
('bool', [True, <NA>, False])                # stays bool; the gap is volas.NA
>>> df[df['sig']]                            # using the gap as a row mask:
ValueError: boolean mask/condition contains volas.NA; an unknown signal is not
            treated as False — fill or drop the NA before masking
```

An unknown signal can never fire a trade by accident: you must resolve it
explicitly (`df['sig'].fillna(False)`) before it is allowed to act.

*Why the type system is built this way:* **C3** (no `object`) means a bool column
cannot decay into a bag of Python truth-values, and the NA-aware mask check turns
"did a gap just place an order?" from a silent loss into a loud, fixable error.

---

## 3. The differences in practice

Each subsection below is a behaviour volas deliberately changed, with the pandas
contrast and the gotcha to keep in mind. These are the same divergences summarised in
the pandas-compatible API guide's [Known pandas divergences](docs/pandas-compatibility.md#known-pandas-divergences-the-volasna-model).

### Missing values keep the dtype

```py
>>> import volas
>>> s = volas.DataFrame({'qty': [1, None, 3]})['qty']
>>> s.dtype, s.to_list()
('int64', [1, <NA>, 3])                   # int stays int; the hole is volas.NA
>>> s = volas.DataFrame({'qty': [2**53 + 1, None]})['qty']
>>> s.to_list()[0]
9007199254740993                          # exact — no float upcast, no lost bit
```

*Gotcha:* `shift`, `diff`, `fillna`, `where`, `mask`, and assignment all keep the
dtype too — where pandas would upcast an `int` column to `float` (or a `str`/`bool`
column to `object`), volas keeps the dtype and marks the cell `volas.NA`.

A frame-level `fillna(value)` resolves the typed fill **per column, atomically,
and lazily**: a column with no holes is untouched (so the everyday `df.fillna(0)`
skips a holeless `str` column on a mixed frame), but a column whose hole the fill
cannot legally take raises a `TypeError` for the *whole frame* — nothing is
partially written:

```py
>>> volas.DataFrame({'x': [1.0, None], 's': ['a', 'b']}).fillna(0)   # dense str: ok
>>> volas.DataFrame({'x': [1.0, None], 's': ['a', None]}).fillna(0)
TypeError: fill for a str column must be a string
```

volas has no `object` dtype to absorb a mismatched fill, so the error is the only
honest outcome — silently filling the numeric columns while leaving the `str` hole
as NA would defeat the very "no NA left after fillna" expectation the call states.
Fill heterogeneous frames per column (`df['s'] = df['s'].fillna('')`).

### Comparisons return a non-nullable bool mask

```py
>>> s = volas.DataFrame({'a': [1, None, 3]})['a']
>>> (s > 0).to_list()
[True, False, True]                       # the missing row is False, not <NA>
```

pandas-nullable would give `[True, <NA>, True]` (three-valued). volas keeps masks
**free of NA** so that `df[mask]`, `.where`, and masked assignment are always total
(every row is selected or not — never "unknown"). A missing value compares `False`
for `==`/`<`/…​ and `True` for `!=`, following IEEE / numpy.

*Gotcha:* if you are migrating from pandas nullable `Int64`, a comparison that you
expected to propagate `<NA>` will instead resolve to `False`/`True`. This is
documented and intentional; it is what keeps the boolean-mask contract simple.

A comparison never produces an NA mask, but you *can* build a bool column that
carries `volas.NA` (e.g. `DataFrame({'m': [True, None, False]})['m']`). Using such
a mask to filter (`df[mask]`), condition (`.where` / `.mask`), or assign is
**rejected** — not silently read as `False`:

```py
>>> m = volas.DataFrame({'m': [True, None, False]})['m']
>>> df[m]
ValueError: boolean mask/condition contains volas.NA; an unknown signal is not
            treated as False — fill or drop the NA before masking
```

An unknown signal must not silently act like a deliberate negative — in a live
system that would turn a data gap into a trade decision. Fill or drop the NA
(`m.fillna(False)` / `m.dropna()`) to state your intent explicitly.

A **directive** boolean is such a signal, so it does carry `volas.NA`: a comparison
or cross with a missing operand (the warm-up rows of `df['close > ma:20']`) is
missing, and `~ & | ^` in a directive use Kleene logic (see
[Directives](docs/directives.md#operators)). Filtering by it therefore needs the
explicit `fillna(False)` too.

### Numeric operations on non-numeric columns raise

A `str` or `datetime` column has no numeric meaning, so arithmetic and numeric
reductions raise instead of funneling to `NaN`:

```py
>>> volas.DataFrame({'sym': ['a', 'b']})['sym'].sum()
TypeError: a numeric operation is not supported on a str column
>>> volas.DataFrame({'sym': ['a', 'b']})['sym'] + 1
TypeError: ...
```

This is stricter than pandas (whose `str.sum()` concatenates), and on purpose: in a
quant pipeline a string column reaching a numeric kernel is a bug, and a loud error
is far better than a silent `0.0` or an all-`NaN` column.

*Gotcha:* this also applies inside the directive engine — `ma:20@symbol` over a
string `symbol` column raises rather than producing a silent all-`NaN` feature.

### Value-returning methods stay Series, not arrays

```py
>>> u = volas.DataFrame({'a': [3, None, 1, 3, None]})['a'].unique()
>>> u.dtype, u.to_list()
('int64', [3, <NA>, 1])                   # a Series; dtype + NA preserved
```

pandas's `unique()` would return `array([3., nan, 1.])` — a `float64` numpy array
that has lost both the `int` dtype and the distinction between a real value and a
hole. Because volas keeps it a `Series`, the result flows losslessly into the next
operation.

### `to_numpy` / `to_pandas` are explicit, lossy boundaries

numpy genuinely cannot represent `volas.NA` in an integer or boolean array, so when —
and only when — you explicitly ask to leave volas, missing values export as `NaN`
and the dtype widens to `float64` (or `NaT` for datetime):

```py
>>> volas.DataFrame({'a': [1, None, 3]})['a'].to_numpy()
array([ 1., nan,  3.])                    # the pandas-style upcast, but only here
```

*Gotcha:* the upcast you would get implicitly all over pandas happens in volas
**only** at this boundary, where you asked for it. Storage and `to_list()` keep the
dtype and `volas.NA`.

A **bool** column holding a missing value is the exception: it has **no** bare NumPy
export. NumPy has no missing bool, and any placeholder reads as a decided truth — the
`nan` pandas puts in its object array is truthy, so `np.where(mask, …)` or
`arr[mask]` would act on an unknown signal as if it were `True` (the same laundering
the NA-mask rule above rejects). `to_numpy()`, `np.asarray(...)` and `df.exec(...)`
therefore raise for it until you say what a missing cell means:

```py
>>> sig = df['close > ma:20']            # the warm-up rows are volas.NA
>>> np.asarray(sig)
ValueError: cannot export a bool column with missing values to NumPy: ...
>>> sig.to_numpy(na_value=False)         # bool array, unknown -> False
>>> sig.to_numpy(dtype='float64')        # 1.0 / 0.0, unknown -> NaN
>>> sig.to_numpy(dtype='object')         # True / False / volas.NA (whose bool() raises)
>>> df.exec('close > ma:20', na_value=False)
```

A float export (`dtype='float64'`, `np.asarray(x, dtype=np.float64)`, a frame's
default numeric matrix) carries `NaN` as float data. `to_pandas()` gives such a
column pandas' masked `boolean` dtype (`pd.NA`) under either backend.

`DataFrame.to_numpy(dtype=...)` is an export boundary — you are *leaving* volas — so
an explicit `dtype` is honored per cell exactly like pandas (the internal no-lossy
contract governs *computation*, not what you convert *out*). The matrix:

| frame \ dtype | `None` (default) | `"object"` | integer / bool | float |
|---|---|---|---|---|
| numeric / bool | `float64` matrix | typed-cell object | exact cast | cast |
| datetime | 2-D `datetime64[ns]` | `Timestamp` / `NA` | **exact epoch-ns** (NaT → `i64::MIN`) | epoch-ns as float (lossy past 2⁵³, NaT → `NaN`) |
| mixed | object (typed cells) | typed-cell object | exact cast | cast |
| contains `str` | object (typed cells) | strings kept | **error** | **error** |

So `dtype="object"` is always **lossless** (each cell its own typed value —
`Timestamp` / `volas.NA` / str / number), an **integer** dtype takes the exact `i64`
channel (a datetime exports its true epoch-ns; a large `int64` survives past 2⁵³ —
neither round-trips through `float64`), a **float** dtype is your sanctioned lossy
opt-in (epoch-ns as float), and a `str` column rejects any **numeric** dtype (it has
no numeric meaning — use `dtype="object"`). This mirrors `pandas.DataFrame.to_numpy`.

### A `Row` is a typed record, not an `object` Series

Indexing a single row gives a `Row` — a read-only, per-column-typed record — not a
squeezed `object` Series:

```py
>>> row = volas.DataFrame({'price': [100.0], 'qty': [5]}).iloc[0]
>>> row['qty']                            # keeps its int64 dtype
5
```

`row['col']` and `row.to_dict()` read each cell in its own dtype. A `Row` never
gains cross-column arithmetic or reductions (those would force the `object`/`float`
collapse that `iterrows` suffers) — the computation surface is always the column.
Row iteration is an anti-pattern; vectorise over columns instead.

### Datetime is `i64` UTC epoch-nanoseconds

Datetimes are stored as true `i64` UTC epoch-ns and **never** pushed through an `f64`
channel. A contemporary epoch-ns value (~1.7×10¹⁸) is far past 2⁵³, where `f64` steps
in ~256 ns increments — so any timestamp routed through `f64` is quantised and
corrupted. volas keeps datetime a distinct logical type (so it is never summed,
averaged, or compared through the float funnel), with `NaT` as its missing value and
a per-frame timezone that governs display and matching while storage stays UTC.

The timezone lives on the **DatetimeIndex only** — a datetime **column** holds UTC
instants and stores no zone (pandas attaches one to every datetime column). Every
export that can state a zone states UTC: `to_pandas()` gives `datetime64[ns, UTC]`,
Arrow `timestamp[ns, tz=UTC]`, cells are `Timestamp(..., tz='UTC')` and `.dt.tz` is
`'UTC'` (NumPy `datetime64[ns]` has no zone slot and carries the same UTC instants).
So a pandas `datetime64[ns, Asia/Shanghai]` column round-trips as
`datetime64[ns, UTC]`, and a naive pandas column comes back UTC-aware; the instants
are exact, the zone is not kept. `reset_index()` on a tz-aware frame likewise moves the
labels into a UTC column. Restore a zone by converting the instants:
`df.set_index('t').tz_localize('UTC').tz_convert(zone)` (or `.dt.tz_convert(zone)` on
the pandas side).

A tz-aware **index** gives its labels as what they are: `df.index` is an object array of
aware `Timestamp`s (pandas `DatetimeIndex.to_numpy()` of an aware index), so a label read
back — `df.index[-1]`, `DataFrame(..., index=df.index)` — is the same label and aligns
with the frame. These are volas `Timestamp`s, which pandas and NumPy do not convert
(`pd.DatetimeIndex(df.index)`, `.astype('datetime64[ns]')`): cross with `to_pandas()`. A
naive index stays `datetime64[ns]`. Labels given as `index=` keep the one zone their aware
timestamps carry; aware and naive timestamps, or two zones, in one index raise. A label
written to a frame follows contract #17: an aware value is an instant, a naive
`Timestamp` / `datetime` / string a wall clock (a naive / aware mix raises), and a raw
epoch-ns count — an integer or an `np.datetime64` — is read in the frame's zone.

A scalar `Timestamp` compares like pandas': with another instant (a `Timestamp`, a
`datetime` / `pd.Timestamp`, an `np.datetime64`) by instant, naive and aware never
equal; a string or an integer (an epoch included) is not an instant — `==` is `False`,
`!=` `True`, ordering raises `TypeError` (parse it with `Timestamp(...)`; `.loc` and
index selection still parse strings). So `a == b` implies `hash(a) == hash(b)`, with
pandas' one exception: an equal sub-microsecond `np.datetime64` hashes by NumPy's rule.
One divergence: ordering against `NaT` raises `TypeError` (pandas answers `False`).

Text and numbers cross the instant boundary by one rule each. A float epoch is the
float's exact binary value times the unit, rounded to the nearest nanosecond (ties to
even) — pandas' result for `s` / `ms` / `us`, while pandas truncates a fractional
`unit='ns'` input. `Timestamp.strftime('%s')` is the instant's real epoch second
(pandas reads the wall clock in the host's local zone), and `%z` / `isoformat()` keep
a second-precision offset (`+05:21:10`); a `tzinfo` offset with a sub-second part
raises rather than being truncated.

### `append` aligns by name; the live fold demands ordered, present timestamps

`df.append(rows)` aligns columns **by name**, not position. A column the appended
rows are *missing* is padded with dtype-preserving NA (see above); a column they add
that the target lacks is **rejected**, not silently dropped:

```py
>>> volas.DataFrame({'x': [1.0]}).append(volas.DataFrame({'x': [2.0], 'z': [9.0]}))
ValueError: append: column 'z' is not in the target frame — appended rows must not
            introduce a new column ...
```

pandas (`concat`, outer) would instead grow the frame with the new column. volas
rejects it so an exchange adding a field can never quietly lose data.

Rows appended with a default `RangeIndex` carry no labels of their own, so they take
the next row labels: appended to a frame with integer labels `0, 1, 2`, one such row
is labelled `3` (pandas `concat` keeps the row's own `0`, giving `0, 1, 2, 0`). This
keeps a sliced or windowed frame's labels absolute and gap-free; a continued label
that the index already holds raises, as does continuing past the int64 range. Rows
appended with explicit integer labels keep them.

The appended values take the **target's dtypes** when that changes no value (an
integer `2` into a float column, `3.0` into an integer one); a value that would change
raises. pandas `concat` instead upcasts the result column (`int64` + `2.5` →
`float64`, number + text → `object`). The index kind must match too: a `DatetimeIndex`
frame takes timestamps of the same tz awareness only, where pandas `concat` would mix
the labels into an `object` index. A column the target caches (an indicator directive
it computes) is rejected in the rows of every append form — it is not data, and
pandas has no such column: select the data columns first.

A scalar assigned to a whole column keeps its own dtype — a Python `int` is `int64`,
`np.int32` is `int32`, `np.float32` is `float32` — and a one-element array is one value
per row, not a scalar: on a frame of another height it is a length error (pandas
broadcasts a length-1 array).

A row appended with an explicit label equal to the **last** row's label **replaces**
that row. A `DatetimeIndex` only increases — a row labelled before the last raises —
and an integer or string index stays unique — a row labelled like an earlier row
raises (in both cases the frame is left unchanged). pandas keeps both rows — `concat`
happily builds a duplicate or out-of-order label — while volas treats a repeated label
as the same bar sent again, which is what a live feed means by it.

On a `time_frame` frame the appended bars are *folded* into the forming period, so
the live feed must be well-ordered: a `NaT`-timestamped bar (no period — symmetric
with `cumulate`, which also rejects `NaT`) and an **out-of-order** bar (a timestamp
earlier than the forming period's latest bar) both **raise**, rather than silently
producing a `NaT` period row or a non-monotonic index. Re-sending the current forming
bar (same timestamp) is still accepted. Handle late / re-ordered data explicitly
before folding.

### Index limitations

The index is deliberately a **single level** of one homogeneous label type. Relative
to pandas, volas does **not** support:

- **`MultiIndex`** (hierarchical / multi-level indexes), on rows *or* columns —
  columns are a flat list of unique string names.
- **Arbitrary label dtypes** — an index is exactly one of range, datetime
  (`datetime64[ns]`), integer, or string. There is no float, categorical, interval,
  period, timedelta, or mixed-type `object` index.
- **Index algebra** — reindexing, index set operations (union / intersection), and
  automatic alignment-on-index when combining frames (volas aligns by **position**,
  and raises on a mismatch rather than silently reindexing). One rule decides every
  pairing of two labelled operands — arithmetic, comparisons, `where` / `mask`
  conditions, `df[mask] = v` cell masks (whose columns pair by name), a Series
  assigned to a column (`df[col] = s`), `corr` / `cov` and the pairwise window
  statistics: the labels must be
  equal (the same instants; the index name and display zone are ignored), and a
  tz-naive index never pairs with a tz-aware one.
- **Duplicate-label** lookups (label access assumes unique labels).

---

## 4. When to use pandas instead

volas targets the single-level, OHLCV-shaped, dtype-honest frame that candlestick /
market data uses, and it is intentionally narrow. Reach for pandas when your workflow
genuinely needs what volas leaves out:

- hierarchical (`MultiIndex`) rows or columns;
- float / categorical / interval / period / timedelta / `object` indexes, or
  duplicate index labels;
- index algebra — reindexing, union / intersection, automatic alignment-on-label;
- the long tail of general-purpose analytics APIs (pivot, melt, groupby-apply,
  rolling-apply with arbitrary Python, etc.) that are out of volas's scope.

For everything on the numeric / trading hot path, volas's promise is the one pandas
cannot make: **the dtype you see is the dtype you get, missing values are lossless,
and anything lossy is an exception — never a silent wrong number.**

## `pct_change` → the `change` directive

volas has no `Series.pct_change()` / `DataFrame.pct_change()` method: rate-of-change
is an *indicator*, and indicators live in the directive engine (one computation
path). The equivalent of pandas `df['close'].pct_change()` is:

```py
df['change']          # (close - prev_close) / prev_close, first row NaN
df.exec('change')     # same values as a raw ndarray
```

Mapping notes: `change` operates on the `close` column with a 1-bar period
(`pct_change()`'s default `periods=1`); the warm-up row is missing in both. For
a different period or column, see the directive syntax in `docs/directives.md`.

## `cumulate` across a DST transition: one local grid for every bar

Sub-day time frames bucket by real instants on the frame's **local** grid (anchored
at local midnight), and day-and-coarser frames by the local civil calendar. pandas
instead lays absolute-length bins from the first day's local midnight. The two agree
whenever the offset change is a whole multiple of the step — e.g. New York's one-hour
DST under `1h` and finer: on a fall-back day the repeated `01:00` hour is two bars in
both. They differ otherwise: after the transition pandas' bins leave the local grid,
while volas keeps every bar on it —
- a multi-hour frame across a one-hour change (on 2026-11-01 in `America/New_York`,
  pandas `resample('2h')` labels `00:00`, `01:00`, `03:00`, …; volas `00:00`, `02:00`,
  `04:00`, …, the `00:00` bar spanning three real hours that day);
- a half-hour DST (`Australia/Lord_Howe`) under `1h` or coarser, and historical LMT
  offsets that are not whole minutes, under any step.

So a bar's period never depends on where the data happens to start, and live folding
can place each bar on its own.

## int `min` / `max` over an all-NA column

A reduction with no surviving value returns `np.float64(nan)` (reductions return
numpy scalars, and numpy int has no missing representation) — pandas's
numpy-backed behaviour. pandas's *nullable* backend returns `pd.NA` instead;
volas deliberately keeps the numpy-scalar boundary (C1).

## Cumulation never skips a missing value

pandas `resample` skips missing values: one `NaN` high inside an hour gives the
hour the maximum of the other highs, and an all-missing hour sums to `0`. volas's
`cumulate` and the live `time_frame` fold instead make a period's `first` / `last` /
`max` / `min` / `sum` **missing as soon as one of its values is**, for every dtype
(an int / bool / str `volas.NA` or a datetime `NaT` as much as a float `NaN`,
whatever value the missing cell carries). An understated range or volume with no
sign that data was missing is the failure this avoids; the indicators then treat the
bar as a missing row. The supported aggregations per dtype are tabled in
[Cumulation](docs/cumulation.md) (a datetime has no `sum`; `bool` / `str` only
`first` / `last`), and an unsupported pair is rejected when the frame is built.

## Window aggregations (`rolling` / `expanding`): four deliberate divergences

The window API is pandas-aligned and differential-tested method by method
(`test_audit_t14_window.py`), with four deliberate departures:

- **`count()` / `nunique()` return `int64`** with native NA in the warm-up
  region. pandas returns `float64` (NaN-padded) because numpy ints cannot hold
  a missing value; volas ints can, so a count is an integer.
- **`kurt()` is computed with two-pass central moments.** pandas accumulates
  raw power sums (`Σx … Σx⁴`), which loses ~8 significant digits when a
  window's mean dwarfs its spread (a price series at 42.60 ± 0.1 is enough to
  move pandas's kurtosis by 0.4%). volas's values are the numerically correct
  ones; differential tests compare against pandas at a loosened tolerance and
  pin volas against the closed form exactly.
- **`skew()` keeps working after an NA gap at `min_periods < 3`.** pandas's
  sliding skew kernel goes permanently NaN for every window after an interior
  NA gap in that configuration (kernel-state bug); volas emits the correct
  per-window value.
- **`rank()` with `center=True` at the trailing edge.** Both rank each window's
  last value. Where the centered window is clipped at the end of the data, pandas
  3.0.3 still counts rows that have left the window (it can emit a rank above the
  window's own `count()`); volas ranks within the clipped window.
- **`sum()` / `mean()` over a window holding an infinity give the infinity.** A
  window holding `+inf` sums to `+inf` (`-inf` likewise, and both signs to NaN), and
  the next window without it is exact again. pandas returns NaN for every window
  holding an infinity, an artefact of its compensated summation, not a definition.

## Integer exactness

An integer is an exact value, so volas never changes one silently: where pandas /
NumPy would wrap, round or saturate, volas computes exactly or raises.

### Integer overflow raises

pandas and NumPy wrap int64 arithmetic modulo 2⁶⁴ (`max + 1 == min`,
`abs(min) == min`). Every volas integer kernel is checked instead and raises
`OverflowError`, identically in debug and release builds: `+ - *`, `sum` / `prod`,
`cumsum` / `cumprod`, `diff`, `abs`, `round` to a negative number of decimals, and
`//` (`min // -1`). int32 operands overflow at int32 (int32 `sum` / `prod`
accumulate in int64, like pandas). A missing cell is never computed, so its
placeholder never raises.

```py
>>> s = volas.DataFrame({'a': [2**63 - 1]})['a']
>>> s + 1
OverflowError: int64 overflow in `+`; integer results never wrap — cast to
               float64 first if an approximate result is acceptable
```

### Integer results are exact; float64 math past 2⁵³ raises

NumPy compares and reduces int64 through float64 in many places, so `2**53 + 1`
and `2**53` compare equal and `nlargest` / `df.max()` can return the wrong row or
value. volas decides per operation by the kind of result it produces:

| operation | integer input |
|---|---|
| `== != < <= > >=` (int vs int, int vs float, Series or scalar, and in a directive — whose integer literals are exact int64), `between`, `is_monotonic_*`, `nlargest` / `nsmallest`, `sort_values`, `rank`, `idxmin` / `idxmax`, `cummin` / `cummax` | **exact** — compared in the integers' own order (an int against a float by exact value, so `2**53 + 1 > 2**53`) |
| `sum` / `prod` / `min` / `max` (Series and DataFrame), `cumsum` / `cumprod`, `+ - *`, `//` (int // int), `abs`, `round`, `diff`, `fillna` / `where` / `mask` with an integer fill | **exact** — computed in int64 (int32 where pandas keeps it), checked for overflow |
| a DataFrame `sum` / `min` / … over int **and** float columns | float64 result; each integer converts exactly or raises `TypeError` |
| `mean`, `var`, `std`, `median`, `quantile`, `sem`, `skew`, `kurt`, `describe`, `corr` / `cov`, the math transforms (`sqrt`, `floor`, `exp`, …), `/`, `//` by zero, `+ - *` with a float operand, `clip` with a fractional bound, `interpolate`, every `rolling` / `expanding` / `ewm` aggregation except `count` / `first` / `last`, directive arithmetic (`+ - * /`, unary `-`) and every indicator input | **float64 math** — a present int64 with `|x| > 2⁵³` raises `OverflowError` (its float64 image is not the value); within `±2⁵³` every integer and its neighbours are exact, so the result is the float64 result of the true input |

pandas returns a float computed from the rounded input in the last row.

### Conversions are lossless or raise

One conversion rule serves every target-typed write (`iat` / `iloc` / `at` / `loc`
with a scalar or an array, Series and mask assignment, dict / `Row` / DataFrame
`append` and the row replacement of a repeated label, the live time-frame fold),
column ingress (lists, tuples, NumPy arrays of any kind), and `astype` between
numeric dtypes:

| source → target | converts | raises |
|---|---|---|
| int → float64 / float32 | exactly representable (`2**60`) | `TypeError` otherwise (`2**53 + 1` → float64, `2**24 + 1` → float32) |
| float → int64 / int32 | integral and in range (`5.0`) | `TypeError` for a fraction (`2.5`), `OverflowError` out of range (`1e20`) |
| int64 → int32 | in range | `OverflowError` |
| float64 → float32 | rounds to the nearest float32 (float precision) | `OverflowError` for a finite value past float32 (`1e300`, never `inf`) |

pandas / NumPy round, saturate or wrap in each of these cases. Specific ingress
rules:

- an integer past int64 (a Python int, a NumPy `uint64` value, an `object` array
  element) raises `OverflowError` — volas has no `uint64`; a `uint64` array whose
  values all fit int64 (and every narrower unsigned array) widens exactly;
- integers mixed with floats infer float64 (like pandas) only when every integer
  is exact in float64, else `TypeError`;
- tuples and `object` arrays follow the list rules (so `Decimal` raises everywhere);
- complex values raise `TypeError` (pandas keeps `complex128`; NumPy-to-float
  drops the imaginary part with only a warning).

`astype('int64')` from a float still truncates toward zero (`2.7 → 2`, as pandas),
but a value past int64 raises instead of saturating to `±2⁶³`; a float epoch past
int64 is unconvertible for `astype('datetime64[...]')` likewise.

`read_csv` never rounds an integer token either: a column whose integer tokens
float64 cannot hold exactly (past int64, or past 2⁵³ next to a float token) is read
as text, losslessly — pandas reads `uint64` / `object` or rounds to float64. A
`nan` token that is not an NA token (`keep_default_na=False`) is text, and an NA
token in a `True` / `False` column is a missing bool rather than a literal.

## A missing value has no truth value

`&` / `|` / `^` / `~` use Kleene logic over every dtype: a missing operand — a
float `NaN`, an int / bool `volas.NA`, or a missing scalar (`None` / `NaN` /
`volas.NA`) — is unknown, so `True & NaN` is `volas.NA` and `False & NaN` is
`False`. NumPy reads `NaN` as truthy, so pandas gives `True & nan == True`. The same
rule makes `bool()` of a one-element Series holding a missing value raise
`TypeError` (like `bool(pd.NA)`), and makes `any` / `all` skip missing cells on both
the Series and the DataFrame; a DataFrame `any` / `all` covers the numeric and bool
columns only, as `Series.any` raises for a `str` / `datetime` column.
