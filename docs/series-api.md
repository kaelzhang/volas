# Series API

`df[col]` and `df[directive]` return a `Series` — a named 1-D column whose API is
pandas-compatible: arithmetic / comparison / logical operators, `.sum()` /
`.mean()` / `.std()` / …, `.shift()` / `.diff()` / `.fillna()`, `.iloc` /
`.loc`, `.to_numpy()` / `.to_list()`. See
[the rest of the pandas-compatible API](pandas-compatibility.md#the-rest-of-the-pandas-compatible-api)
for the full list. There is no public `Series` constructor — a `Series` is
always obtained by indexing a `DataFrame`.

Derived Series preserve checked index labels: selecting a row twice raises
`ValueError`; a reverse slice or sort raises if its resulting datetime labels
no longer strictly increase. Datetime labels always strictly increase without NaT. Integer and
string labels may be reordered while they remain unique; the source is unchanged.

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

(series-from-arrow)=
## Series.from_arrow(data, name=None, copy=False) -> Series

A **volas-specific static method** that builds a `Series` from any object exposing the
Arrow **C-Data** array protocol (`__arrow_c_array__`) — a `pyarrow.Array`, a polars
`Series`, etc. Imports borrow matching buffers by default (`copy=False`), with a fresh
`RangeIndex`. A malformed array raises `ValueError`, exactly as in
[`DataFrame.from_arrow`](dataframe-api.md#dataframefrom_arrowdata-copyfalse---dataframe).

- **data** the Arrow source — any object implementing `__arrow_c_array__`.
- **name?** `str | None = None` the name for the resulting `Series`.
- **copy** keyword-only `bool = False`. `True` takes an owned snapshot. Arrow data is immutable: the producer and every alias must never write borrowed memory while imported objects or views survive. Conversion layouts still copy.

```py
s = Series.from_arrow(pa_array, name='close')   # pyarrow.Array -> Series
```

(series-to-numpy)=
## series.to_numpy(dtype=None, na_value=...) -> np.ndarray

The column values as a 1-D NumPy array — `pandas.Series.to_numpy` semantics:

- **dtype** `str | None` — an optional export cast. `None` (the default) gives the
  column's native representation; otherwise any NumPy dtype string accepted by
  `numpy.ndarray.astype`, the common values being:
  - `'int64'`, `'int32'`, `'int16'`, `'int8'` (and the unsigned `'uint64'`, `'uint32'`,
    `'uint16'`, `'uint8'`) — integer. Over a column with missing values this **raises**
    unless `na_value` is given (an NA has no integer representation).
  - `'float64'`, `'float32'`, `'float16'` — floating point; missing numeric / bool
    cells are `NaN`. Without `na_value`, datetime Series / `exec` follow NumPy:
    `NaT` casts to its epoch-ns sentinel as a float (or complex); Frame float /
    complex exports instead map `NaT` to `NaN`.

  - `'bool'` — boolean. Over a column with missing values this **raises** unless
    `na_value` is given (an NA has no boolean representation).
  - `'datetime64[ns]'` — datetime nanoseconds; a missing cell is `NaT`.
  - `'object'` (or `'O'`) — Python objects, each cell its own typed value (lossless).
- **na_value** `Any` — the value to substitute for each missing cell. Default: the
  NA-model representation (`NaN` for floats; `None` for default string export).
  Explicit `dtype='object'` preserves typed missing cells: int, bool, string and
  datetime use `volas.NA`; float keeps its in-band `NaN`. With
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

Default string Series / `exec` exports use object arrays with `None`; explicit
object exports use typed cells with `volas.NA`.

```py
from volas import DataFrame, NA

text = DataFrame({'x': ['a', None]})['x']
integer = DataFrame({'x': [1, None]})['x']
assert text.to_numpy()[1] is None
assert text.to_numpy(dtype='object')[1] is NA
assert integer.to_numpy(dtype='object')[1] is NA
```

Notes:
- The default (no `dtype`, no `na_value`) is the dtype-specific export: a missing
  int / datetime cell collapses to `NaN` / `NaT` (NumPy has no `NA`), while a dense
  column keeps its native dtype. A **bool** column with a missing cell has no default
  export and **raises** — a `nan` / `None` placeholder would be read as `True` /
  `False` by `np.where` or boolean indexing; pass `na_value=` (e.g. `False`, a bool
  array), `dtype='float64'` (`NaN`) or `dtype='object'` (`volas.NA`). The same rule
  covers `np.asarray(series)` and `df.exec(directive, dtype=..., na_value=...)`. A float `NaN` is in-band, so a float column
  cast to an integer dtype likewise raises when any value is `NaN` (pass `na_value`).
- Like pandas, `na_value` only changes the missing cells — without an explicit `dtype`
  an int column with NA still exports `float64` (the default), and `na_value` simply
  fills the `NaN` slots.
- For a **lossless** NA round-trip that keeps the native dtype *and* the missing
  positions (no fill, no float collapse), use the Arrow path (`series.to_arrow()` carries
  the null bitmap) or `series.to_pandas(dtype_backend='numpy_nullable')`; the NA mask
  alone is `series.isna().to_numpy()`.

(series-to-arrow)=
## series.to_arrow() -> pyarrow.Array

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

