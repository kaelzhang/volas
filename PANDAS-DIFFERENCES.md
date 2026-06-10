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

A missing value is a first-class symbol, [`volas.NA`](README.md#missing-values-volasna),
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

### 2.4 The nullable retrofit is opt-in, incomplete, and splits return types

pandas does have a newer, better answer — the nullable `Int64` / `boolean` /
`string` extension dtypes and the `pd.NA` scalar (PDEP-16). But it is **opt-in** (the
default `int` column is still the un-nullable numpy one), only **partially adopted**
across the API, and it **splits return types**: the same `unique()` returns a numpy
`ndarray` for a default column and an `ExtensionArray` for a nullable one. You end up
having to know, per call, which type system you are in.

volas does not retrofit nullability onto a non-nullable core — **it is nullable from
the ground up**, with one consistent set of return types.

---

## 3. The differences in practice

Each subsection below is a behaviour volas deliberately changed, with the pandas
contrast and the gotcha to keep in mind. These are the same divergences summarised in
the README's [Known pandas divergences](README.md#known-pandas-divergences-the-volasna-model).

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
  and raises on a mismatch rather than silently reindexing).
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
