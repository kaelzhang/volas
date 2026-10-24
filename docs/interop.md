# pandas, Arrow and DLPack interop

## pandas interop

pandas is **not** a runtime dependency; these bridges import it lazily, only when
called, so `import volas` stays pandas-free.

```py
from volas import DataFrame

df = DataFrame.from_pandas(pandas_df)  # numeric / bool / str / datetime native; a (tz-aware) DatetimeIndex round-trips;
                                   # a nullable Int64 / boolean / string column reads back as int / bool / str + volas.NA
pdf = df.to_pandas()               # -> pandas.DataFrame ('numpy' backend: int with NA becomes float64 + NaN; bool with NA becomes nullable boolean + pd.NA)
pdf = df.to_pandas(dtype_backend='numpy_nullable')  # faithful masked Int64 / boolean (a lossless NA round-trip)
df.to_csv('out.csv', index=True)   # subset of pandas to_csv; returns a str if path=None
```

`to_pandas`'s **dtype_backend** (`'numpy'` vs the lossless `'numpy_nullable'`) governs
how a missing value crosses into pandas — see
[`df.to_pandas`](dataframe-api.md#dfto_pandasdtype_backendnumpy---pandasdataframe) for the per-value
breakdown.

`from_pandas` keeps index labels and their name, including non-default
`RangeIndex` labels and frames containing only an index. Column labels become
strings; labels that collide after stringification (such as `1` and `'1'`) raise
`ValueError`. Signed int32 extension columns, including `int32[pyarrow]`, keep
int32 width. The `numpy_nullable` backend exports text as pandas `string` with
`pd.NA` for missing values.

NumPy input may have any stride: contiguous arrays are read directly, while
strided arrays are copied in logical order into volas's contiguous storage.

CSV escapes every field, including index labels, index names and column headers.
The supported printf float formats `%f`, `%e`, `%g` (and their `.precision` forms)
apply to CSV and `to_string`; `%g` counts significant digits and uses scientific
notation when its rounded exponent is below −4 or at least the precision.
`to_string(max_rows=..., min_rows=...)` clamps `min_rows` to `max_rows`; a zero
`max_rows` means no explicit truncation.

### Datetime columns are UTC instants

A datetime **column** stores absolute instants (UTC epoch-nanoseconds) and no zone of
its own — only the `DatetimeIndex` carries a zone. A pandas column in any zone (or
naive) is read as its instants, and every export that can state a zone states **UTC**:

| export | a datetime column becomes |
|---|---|
| `to_pandas()` (both backends) | `datetime64[ns, UTC]` |
| `pa.table(df)` / `pa.array(s)` / `to_arrow()` | `timestamp[ns, tz=UTC]` |
| `s.to_list()` / `to_dict()` / `items()`, cells, `to_numpy(dtype='object')` | `Timestamp(..., tz='UTC')` (`volas.NA` for `NaT`) |
| `s.dt.tz` | `'UTC'` (`.dt` fields read the UTC wall clock) |
| `to_numpy()` / `__array__` | `datetime64[ns]` of the same UTC instants (NumPy has no zone) |
| `s.dtype` / `df.dtypes`, `repr`, `to_csv` | unchanged: the dtype is still `datetime64[ns]` (no tz in the dtype name), and repr / CSV text show the UTC wall clock without an offset |

So the instant is never ambiguous, but the **source column's zone is not stored**: a
pandas `datetime64[ns, Asia/Shanghai]` column comes back `datetime64[ns, UTC]` (the
same instants). Restore the zone by converting — in pandas, or in volas through the
index:

```py
pdf = df.to_pandas()
pdf['t'] = pdf['t'].dt.tz_convert('Asia/Shanghai')      # pandas: restate the zone

df.set_index('t').tz_localize('UTC').tz_convert('Asia/Shanghai')
# volas: the column's UTC instants become a UTC-anchored index, then restated
```

A naive pandas datetime column is read as UTC instants too, so it also exports as
`datetime64[ns, UTC]` (drop the zone in pandas with `.dt.tz_localize(None)` if needed).

## Arrow & DLPack interop (zero-copy)

A volas column owns one contiguous buffer per dtype (Arrow-native string layout
included), so it crosses to Arrow and DLPack consumers **without a copy** — the
consumer borrows the same bytes, kept alive by volas.

```py
import pyarrow as pa, numpy as np

# Arrow C-Data / C-Stream — pyarrow, polars, … read volas directly via the
# standard PyCapsule protocols (__arrow_c_array__ / __arrow_c_stream__).
pa.array(df['close'])              # Series  -> pyarrow.Array  (shares the buffer)
pa.table(df)                       # DataFrame -> pyarrow.Table (one RecordBatch)
df['close'].to_arrow()             # convenience for pa.array(...)

Series.from_arrow(pa_array, name='close')   # borrows matching buffers by default
Series.from_arrow(pa_array, name='close', copy=True)   # owned snapshot
DataFrame.from_arrow(pa_table)              # Arrow table  -> DataFrame (row count kept even with no columns)

# DLPack — NumPy / PyTorch / JAX borrow a dense numeric (or bool) column.
np.from_dlpack(df['close'])        # zero-copy ndarray view
```

The high-level entry points — [`df.to_arrow`](dataframe-api.md#dfto_arrow---pyarrowtable) /
[`DataFrame.from_arrow`](dataframe-api.md#dataframefrom_arrowdata-copyfalse---dataframe),
{ref}`series.to_arrow <series-to-arrow>` /
{ref}`Series.from_arrow <series-from-arrow>` — are documented under
Usage. They sit on these **standard protocol methods**, which Arrow / array consumers
call automatically (so you rarely call them yourself):

- **`Series.__arrow_c_array__`** — the Arrow C-Data array protocol; returns the
  `(schema, array)` PyCapsule pair, so `pa.array(s)` / `pl.Series(s)` read a column.
- **`Series.__arrow_c_schema__`** — the schema-only half (the column's Arrow dtype).
- **`DataFrame.__arrow_c_stream__`** — the Arrow C-Stream protocol; the frame as one
  `RecordBatch`, so `pa.table(df)` / `pl.from_dataframe(df)` read a frame.
- **`Series.__dlpack__` / `Series.__dlpack_device__`** — the DLPack protocol, so
  `np.from_dlpack(s)` / `torch.from_dlpack(s)` borrow a dense numeric / bool column. The
  borrowed view is **read-only** (a versioned-DLPack flag) so writing through it can't
  bypass copy-on-write and corrupt the frame; a pre-1.0 consumer that cannot receive the
  flag is given an independent copy instead. `np.from_dlpack(s, copy=True)` returns an
  independent **writable** copy. A non-CPU device or a stream is refused (`BufferError`);
  an int/bool column with a missing value (DLPack has no null mask) or a str/datetime
  column raises.

**Arrow import contract.** Arrow data is immutable. `Series.from_arrow(array)`
and `DataFrame.from_arrow(table)` default to `copy=False`, borrowing matching
native numeric (`int32` / `int64` / `float32` / `float64`) and dense ns-datetime
buffers. **The producer and every alias must never write borrowed memory while
any imported object or view survives.** This includes finite-value changes;
volas cannot detect external writes or invalidate their caches.

`copy=True` takes an owned snapshot, isolating columns, cached directives,
resume states and checkpoints from later producer writes. Do not write inputs
concurrently with import: copying does not provide an atomic snapshot across columns.
`copy=False` permits borrowing rather than requiring it: incompatible layouts and
multi-chunk columns may still copy. Direct NumPy imports and ordinary constructors
receiving Arrow arrays copy; pass a `Series.from_arrow(array)` result to the frame
constructor to share its column.

**Zero-copy export contract.** Native numeric and ns-datetime columns export shared,
read-only data buffers; strings also export zero-copy.
The remaining repacks are:

- on **import**, every string column (`string` / `large_string`, sliced or not) — it is
  copied into volas-owned buffers and validated there (in-range monotonic offsets, valid
  UTF-8 for each present value; a null slot's unspecified bytes are dropped), so a
  producer rewriting its buffers after the import cannot reach the volas column;

- **`bool`** — volas stores one byte per value, Arrow one *bit*;
- the **null bitmap** (≤ `n/8` bytes) — including, on export, a one-pass scan that turns
  a missing `float` (in-band `NaN`) into a real Arrow null;
- on **import only**: a `string_view` (materialised to contiguous bytes), a coarser-than-ns timestamp
  (rescaled), a `Decimal128` / `Decimal256` (→ the nearest `f64`, **lossy** past ~15 digits —
  keep prices as strings for exactness), a narrow / unsigned integer (→ `i64`; a `UInt64` past
  `i64::MAX` has no lossless image and raises), a dictionary / categorical column (decoded
  to its values), a typeless `Null` column (→ an all-NA `f64` column), or a
  `Date32` / `Date64` (→ ns datetime). Other Arrow types raise a clear error.

**NA at the boundary.** Every missing value — int, bool, **and float** — crosses as a
real Arrow null (a missing float is the in-band `NaN`, scanned into the null bitmap on
export), so a downstream `is_null` / `null_count` sees them all consistently. DLPack and
an integer `to_numpy` have no null channel, so they raise on a missing value — pass
`na_value=`, or use the Arrow path, which carries the null bitmap losslessly:

```py
df['qty'].to_numpy(dtype='int64')                # raises if any value is NA (pandas-aligned)
df['qty'].to_numpy(dtype='int64', na_value=0)    # or fill the holes with na_value
pa.array(df['qty'])                              # lossless: keeps int64 + the null bitmap
```

(numpy-export-rules)=
### NumPy export rules

Default string Series / `exec` exports are object arrays whose missing cells are
`None`. Explicit `dtype='object'` exports preserve typed cells: missing int,
bool, string and datetime cells are `volas.NA`; floats retain in-band `NaN`.
Default mixed / string Frame exports also use typed cells.

```py
from volas import DataFrame, NA

text = DataFrame({'x': ['a', None]})['x']
integer = DataFrame({'x': [1, None]})['x']
assert text.to_numpy()[1] is None
assert text.to_numpy(dtype='object')[1] is NA
assert integer.to_numpy(dtype='object')[1] is NA
```

The default numeric / bool Frame is a `float64` matrix (including `NaN` for
missing bools); datetime-only Frames retain `datetime64[ns]`; other mixtures
use typed object cells. Series / `exec` defaults retain dense native types,
promote missing integers to `float64` + `NaN`, and reject missing bools.
Every explicit dtype converts valid slots directly from each column's native
representation, with or without `na_value`. Adding NA in another row never
changes a valid value in a target that holds it losslessly (including time and
text targets); integers never pass through the default float64 export.
Default missing representations are chosen separately: integer / bool holes
use NaN, string holes use None, and datetime holes retain NaT. Missing bools
only allow float / complex / object targets unless a fill is supplied. Frames
have two datetime exceptions: integer / bool targets keep NaT's epoch-ns
sentinel, and float / complex targets replace NaT with NaN. Series / `exec`
float / complex casts without a fill retain NumPy's epoch-ns NaT sentinel.
These missing representations are converted into the requested target;
`na_value` replaces the original missing slots when supplied.
A Frame containing strings rejects numeric / temporal targets (even all-NA
strings). These policies take precedence over plain NumPy casts.

With an explicit NumPy dtype and `na_value`, only valid slots are converted, and `na_value`
is filled in the target array. Missing payloads cannot cause conversion errors
or warnings, including in an all-missing column. For example:
`DataFrame({'qty': [1, None]})['qty'].to_numpy(dtype='float64', na_value=0.5)`
returns `[1.0, 0.5]`. Integer exports cast directly from native integer values;
`dtype='int64', na_value=0` preserves values beyond 2⁵³ exactly. Python errors
from NumPy conversion or scalar boxing propagate to the caller.
Each column retains its own source dtype until the common target is resolved
by NumPy. Conversions then read those source arrays independently: a datetime
column keeps its ns unit even beside numeric columns, and numeric columns
interpret epoch counts in the resolved target unit. The output is assembled
only after conversion and target filling. No shared i64 / f64 / object source
matrix participates in explicit casts.
For Frame, Series and `exec`, unsized text dtypes (`'U'`, `'str'`, `'S'`) infer
their width from NumPy's valid-slot conversion, widened to fit the fill.
The resolved dtype is retained before allocation: generic `datetime64` / `M`
and `timedelta64` / `m` keep a datetime source's unit, and `V` / `V0` keep
its native itemsize. All-missing columns resolve from an empty array of the
source dtype without reading missing payloads. Explicit widths such as `'U2'`
keep the requested width, including NumPy's truncation behavior.


**Timezone.** Arrow tables have no *index*, so the frame's index — and its timezone — is
not carried, and `from_arrow` returns a fresh `RangeIndex`. A datetime column exports
as `timestamp[ns, tz=UTC]` (see
[Datetime columns are UTC instants](#datetime-columns-are-utc-instants)); an imported
timestamp column of any zone (or none) becomes its exact instants, the zone itself not
kept. Restate a zone after import with `set_index` + `tz_localize('UTC')` +
`tz_convert`.
