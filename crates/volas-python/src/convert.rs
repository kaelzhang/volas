//! Boundary marshalling between volas columns and NumPy / pandas (export,
//! scalar boxing, the numpy-type cache, and index/label rendering).


use numpy::IntoPyArray;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyDict, PyList};
use volas_core::{
    datetime, Column, DType, DataFrame, Index,
    IndexKind, Scalar, Tz, Validity,
};

#[allow(unused_imports)]
use crate::*;

/// Like [`column_to_numpy`] but **consumes** the column, moving its backing buffer
/// straight into the NumPy array with no copy when the column is uniquely owned —
/// the fresh-result path (`df.exec(directive)`). Falls back to a borrow + copy for
/// the rarer `Str` / `Datetime` columns.
pub(crate) fn column_into_numpy<'py>(py: Python<'py>, col: Column) -> Bound<'py, PyAny> {
    match col {
        Column::F64(a) => a.into_vec().into_pyarray(py).into_any(),
        Column::Bool(a, _) => a.into_vec().into_pyarray(py).into_any(),
        Column::I64(a, _) => a.into_vec().into_pyarray(py).into_any(),
        other => column_to_numpy(py, &other),
    }
}

/// A column as a pandas array for `to_pandas`. With `nullable`, an int/bool
/// column becomes a pandas masked `Int64` / `Int32` / `boolean` (a faithful,
/// lossless NA round-trip); otherwise it is the numpy export (a missing value
/// becomes NaN, like `Int64.to_numpy()`).
pub(crate) fn column_to_pandas<'py>(
    py: Python<'py>,
    pd: &Bound<'py, PyModule>,
    col: &Column,
    nullable: bool,
) -> PyResult<Bound<'py, PyAny>> {
    if !nullable {
        return Ok(column_to_numpy(py, col));
    }
    let arrays = pd.getattr("arrays")?;
    let mask = |val: &Validity, n: usize| {
        (0..n).map(|i| !val.is_valid(i)).collect::<Vec<bool>>().into_pyarray(py)
    };
    match col {
        Column::I64(v, val) => {
            arrays.call_method1("IntegerArray", (v.to_vec().into_pyarray(py), mask(val, v.len())))
        }
        Column::I32(v, val) => {
            arrays.call_method1("IntegerArray", (v.to_vec().into_pyarray(py), mask(val, v.len())))
        }
        Column::Bool(v, val) => {
            arrays.call_method1("BooleanArray", (v.to_vec().into_pyarray(py), mask(val, v.len())))
        }
        // F9b: a str column exports as the pandas string dtype under the
        // nullable backend, so its declared dtype survives a round-trip even
        // when all-NA / empty (an object array of None carries no dtype).
        Column::Str(v, val) => {
            let items: Vec<Bound<'_, PyAny>> = (0..v.len())
                .map(|i| {
                    if val.is_valid(i) {
                        v.get(i).into_pyobject(py).unwrap().into_any()
                    } else {
                        py.None().into_bound(py)
                    }
                })
                .collect();
            let list = PyList::new(py, items)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", "str")?;
            pd.call_method("array", (list,), Some(&kwargs))
        }
        // float (NaN in-band) / datetime have no nullable masked form here.
        _ => Ok(column_to_numpy(py, col)),
    }
}

pub(crate) fn column_to_numpy<'py>(py: Python<'py>, col: &Column) -> Bound<'py, PyAny> {
    match col {
        Column::F64(v) => v.to_vec().into_pyarray(py).into_any(),
        Column::F32(v) => v.to_vec().into_pyarray(py).into_any(),
        // numpy int/bool cannot hold a missing value, so a column with any NA
        // exports as float64 with NaN (pandas `Int64.to_numpy()` semantics); a
        // dense column keeps its native dtype.
        Column::Bool(v, val) if !val.has_nulls() => v.to_vec().into_pyarray(py).into_any(),
        Column::I64(v, val) if !val.has_nulls() => v.to_vec().into_pyarray(py).into_any(),
        Column::I32(v, val) if !val.has_nulls() => v.to_vec().into_pyarray(py).into_any(),
        // F17 (NA-model interop ruling): a bool column with missing exports as an
        // OBJECT array (True / nan / False) — float64 would destroy the bool
        // identity (1.0/0.0). Matches pandas nullable boolean .to_numpy().
        Column::Bool(v, val) => {
            let items: Vec<Bound<'_, PyAny>> = (0..v.len())
                .map(|i| {
                    if val.is_valid(i) {
                        pyo3::types::PyBool::new(py, v[i]).to_owned().into_any()
                    } else {
                        f64::NAN.into_pyobject(py).unwrap().into_any()
                    }
                })
                .collect();
            let list = PyList::new(py, items).expect("build bool list");
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", "object").expect("set dtype=object");
            py.import("numpy")
                .expect("import numpy")
                .call_method("array", (list,), Some(&kwargs))
                .expect("np.array(object)")
        }
        // numpy int cannot hold a missing value -> float64 with NaN (pandas
        // Int64.to_numpy() semantics); a dense column keeps its native dtype.
        Column::I64(..) | Column::I32(..) => {
            col.to_f64_vec().into_pyarray(py).into_any()
        }
        // String columns become NumPy object arrays (pandas `object` dtype).
        Column::Str(v, val) => {
            // a missing cell becomes Python `None` in the object array (pandas parity)
            let items: Vec<Bound<'_, PyAny>> = (0..v.len())
                .map(|i| {
                    if val.is_valid(i) {
                        v.get(i).into_pyobject(py).unwrap().into_any()
                    } else {
                        py.None().into_bound(py)
                    }
                })
                .collect();
            let list = PyList::new(py, items).expect("build str list");
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("dtype", "object")
                .expect("set dtype=object");
            py.import("numpy")
                .expect("import numpy")
                .call_method("array", (list,), Some(&kwargs))
                .expect("np.array(object)")
        }
        // Datetime columns become NumPy datetime64[ns] arrays.
        Column::Datetime(v) => {
            let arr = v.to_vec().into_pyarray(py);
            arr.call_method1("astype", ("datetime64[ns]",))
                .expect("astype datetime64[ns]")
        }
    }
}

/// A column as a NumPy array, honoring an optional export `dtype` and `na_value`
/// (pandas `Series.to_numpy` semantics):
/// - without `na_value`, an integer `dtype` over missing values **raises** (an NA has
///   no integer representation) — otherwise the NA-model default applies (`NaN` / `NaT`);
/// - with `na_value`, missing cells are filled with it: for an explicit `dtype` the
///   **native** values are kept (so a large int stays exact) and the fill happens before
///   the cast; for the default dtype the fill lands in the NA-model array, keeping its
///   dtype (so `na_value=-1` on an int column with NA stays `float64`, like pandas).
pub(crate) fn column_to_numpy_with<'py>(
    py: Python<'py>,
    col: &Column,
    dtype: Option<&str>,
    na_value: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let Some(nv) = na_value else {
        let arr = column_to_numpy(py, col);
        return match dtype {
            Some(dt) => astype_checked(py, arr, col, dt),
            None => Ok(arr),
        };
    };
    if col.null_count() == 0 {
        // `na_value` is irrelevant with no missing cell — a plain typed export.
        let arr = column_to_numpy(py, col);
        return match dtype {
            Some(dt) => arr.call_method1("astype", (dt,)),
            None => Ok(arr),
        };
    }
    let mask = na_mask(py, col);
    match dtype {
        // exact native values (no float funnel) → fill the holes → cast.
        Some(dt) => {
            let base = column_values_native(py, col)?;
            base.set_item(mask, nv)?;
            base.call_method1("astype", (dt,))
        }
        // the NA-model array keeps its default dtype; only the holes change.
        None => {
            let base = column_to_numpy(py, col);
            base.set_item(mask, nv)?;
            Ok(base)
        }
    }
}

/// The boolean NA mask of a column (`True` exactly at the missing cells).
pub(crate) fn na_mask<'py>(py: Python<'py>, col: &Column) -> Bound<'py, PyAny> {
    (0..col.len()).map(|i| !col.is_valid(i)).collect::<Vec<bool>>().into_pyarray(py).into_any()
}

/// The column values as a **native-dtype** NumPy array with no NA collapse (int stays
/// int, datetime stays `datetime64`) — the exact base for an `na_value` fill. The value
/// at a missing slot is unspecified (the caller overwrites it via [`na_mask`]).
fn column_values_native<'py>(py: Python<'py>, col: &Column) -> PyResult<Bound<'py, PyAny>> {
    Ok(match col {
        Column::F64(v) => v.to_vec().into_pyarray(py).into_any(),
        Column::F32(v) => v.to_vec().into_pyarray(py).into_any(),
        Column::I64(v, _) => v.to_vec().into_pyarray(py).into_any(),
        Column::I32(v, _) => v.to_vec().into_pyarray(py).into_any(),
        Column::Bool(v, _) => v.to_vec().into_pyarray(py).into_any(),
        // str has no fixed-width NumPy form — an object array (string or `None`).
        Column::Str(..) => column_to_numpy(py, col),
        Column::Datetime(v) => {
            v.to_vec().into_pyarray(py).call_method1("astype", ("datetime64[ns]",))?
        }
    })
}

/// Cast an exported NumPy array to `dtype`, but **raise** (pandas-aligned) when the
/// column holds missing values and `dtype` is an integer type — an NA has no integer
/// representation, so NumPy would silently emit a `RuntimeWarning` and write garbage.
pub(crate) fn astype_checked<'py>(
    py: Python<'py>,
    arr: Bound<'py, PyAny>,
    col: &Column,
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if col.null_count() > 0 && is_integer_dtype(py, dtype)? {
        return Err(PyValueError::new_err(format!(
            "cannot convert a column with missing values to integer NumPy dtype '{dtype}' \
             (an NA has no integer representation) — pass na_value=, or use a float dtype"
        )));
    }
    arr.call_method1("astype", (dtype,))
}

/// Whether `dtype` names a NumPy signed/unsigned integer type (`kind` `i` / `u`).
pub(crate) fn is_integer_dtype(py: Python<'_>, dtype: &str) -> PyResult<bool> {
    let kind = py
        .import("numpy")?
        .call_method1("dtype", (dtype,))?
        .getattr("kind")?
        .extract::<String>()?;
    Ok(matches!(kind.as_str(), "i" | "u"))
}

/// The i-th element of a column as a Python scalar.
pub(crate) fn scalar_to_py(py: Python<'_>, col: &Column, i: usize) -> Py<PyAny> {
    match col {
        Column::F64(v) => v[i].into_pyobject(py).unwrap().into_any().unbind(),
        Column::F32(v) => (v[i] as f64).into_pyobject(py).unwrap().into_any().unbind(),
        // an int/bool missing cell is the volas.NA symbol (pandas tolist semantics)
        Column::I64(v, val) if val.is_valid(i) => v[i].into_pyobject(py).unwrap().into_any().unbind(),
        Column::I32(v, val) if val.is_valid(i) => (v[i] as i64).into_pyobject(py).unwrap().into_any().unbind(),
        Column::Bool(v, val) if val.is_valid(i) => {
            v[i].into_pyobject(py).unwrap().to_owned().into_any().unbind()
        }
        Column::Str(v, val) if val.is_valid(i) => v.get(i).into_pyobject(py).unwrap().into_any().unbind(),
        Column::I64(..) | Column::I32(..) | Column::Bool(..) | Column::Str(..) => na(py),
        // O1 (-> B): a datetime cell is a volas.Timestamp (matching the index-label
        // scalar type and pandas); a NaT cell is volas.NA (the unified missing
        // singleton). Columns are UTC-naive ns, so the cell tz is UTC.
        Column::Datetime(v) if v[i] == i64::MIN => na(py),
        Column::Datetime(v) => Py::new(py, PyTimestamp { ns: v[i], tz: Tz::Naive })
            .expect("create Timestamp")
            .into_any(),
    }
}

/// Cached numpy scalar **type** objects (`np.float64` etc.), so a boundary box is
/// a single call rather than a re-import + attribute lookup per value. Holds only
/// the numeric types that need boxing; `bool` / `str` / `datetime` are handled
/// directly. Indexed by [`DType`] for `O(1)` lookup.
pub(crate) struct NumpyTypes {
    float64: Py<PyAny>,
    float32: Py<PyAny>,
    int64: Py<PyAny>,
    int32: Py<PyAny>,
    bool_: Py<PyAny>,
}
pub(crate) static NUMPY_TYPES: GILOnceCell<NumpyTypes> = GILOnceCell::new();
pub(crate) fn numpy_types(py: Python<'_>) -> &'static NumpyTypes {
    NUMPY_TYPES.get_or_init(py, || {
        let np = py.import("numpy").expect("import numpy");
        let ty = |n: &str| np.getattr(n).expect("numpy scalar type").unbind();
        NumpyTypes {
            float64: ty("float64"),
            float32: ty("float32"),
            int64: ty("int64"),
            int32: ty("int32"),
            bool_: ty("bool_"),
        }
    })
}

/// Box `value` as the numpy scalar `dtype` (the external-boundary representation,
/// e.g. `np.float64(0.0)`); the call narrows it to the target type. Non-numeric
/// dtypes never reach here.
pub(crate) fn numpy_scalar(py: Python<'_>, dtype: DType, value: &Bound<'_, PyAny>) -> Py<PyAny> {
    let t = numpy_types(py);
    let ty = match dtype {
        DType::F64 => &t.float64,
        DType::F32 => &t.float32,
        DType::I64 => &t.int64,
        DType::I32 => &t.int32,
        DType::Bool => &t.bool_,
        _ => return value.clone().unbind(), // not a numpy-numeric dtype
    };
    ty.bind(py).call1((value,)).expect("numpy scalar box").unbind()
}

/// A boxed scalar as its numpy type (`np.float64` / `np.float32` / `np.int64` /
/// `np.int32` / `np.bool_`).
pub(crate) fn np_f64(py: Python<'_>, x: f64) -> Py<PyAny> {
    numpy_scalar(py, DType::F64, &x.into_pyobject(py).unwrap())
}
pub(crate) fn np_f32(py: Python<'_>, x: f32) -> Py<PyAny> {
    numpy_scalar(py, DType::F32, &x.into_pyobject(py).unwrap())
}
pub(crate) fn np_i64(py: Python<'_>, x: i64) -> Py<PyAny> {
    numpy_scalar(py, DType::I64, &x.into_pyobject(py).unwrap())
}
pub(crate) fn np_i32(py: Python<'_>, x: i32) -> Py<PyAny> {
    numpy_scalar(py, DType::I32, &x.into_pyobject(py).unwrap())
}
pub(crate) fn np_bool(py: Python<'_>, b: bool) -> Py<PyAny> {
    numpy_scalar(py, DType::Bool, &b.into_pyobject(py).unwrap().to_owned())
}

/// Element `i` as a **numpy** scalar (pandas' direct `s[i]` / `iloc` / `at`
/// semantics), matching the column dtype. Bulk paths (`to_list`, iteration) use
/// [`scalar_to_py`] instead, which yields native Python scalars like pandas.
pub(crate) fn np_scalar_to_py(py: Python<'_>, col: &Column, i: usize) -> Py<PyAny> {
    match col {
        // a float carries missing in-band as NaN -> np.float64(nan), like pandas.
        Column::F64(v) => np_f64(py, v[i]),
        Column::F32(v) => np_f32(py, v[i]),
        // an int/bool missing cell surfaces as the volas.NA symbol.
        Column::I64(v, val) => if val.is_valid(i) { np_i64(py, v[i]) } else { na(py) },
        Column::I32(v, val) => if val.is_valid(i) { np_i32(py, v[i]) } else { na(py) },
        Column::Bool(v, val) => if val.is_valid(i) { np_bool(py, v[i]) } else { na(py) },
        // str -> Python str; datetime -> np.datetime64 (already numpy) — as scalar_to_py.
        Column::Str(_, _) | Column::Datetime(_) => scalar_to_py(py, col, i),
    }
}

/// Box a [`Scalar`] reduction result as its matching numpy scalar (the external
/// representation of `sum` / `min` / … — pandas returns `np.int64` / `np.float64`
/// / `np.bool_`).
pub(crate) fn scalar_to_numpy(py: Python<'_>, s: Scalar) -> Py<PyAny> {
    match s {
        Scalar::F64(x) => np_f64(py, x),
        Scalar::F32(x) => np_f32(py, x),
        Scalar::I64(x) => np_i64(py, x),
        Scalar::I32(x) => np_i32(py, x),
        Scalar::Bool(b) => np_bool(py, b),
    }
}

/// The min/max **value** at `want_max`, typed by dtype: numeric/bool reduce to a
/// numpy scalar (native and exact — an i64 extreme survives past 2^53), `str` to
/// a Python `str`, `datetime` to `np.datetime64`. Order-based selection, so it
/// never routes str/datetime through the f64 funnel (which would collapse them).
/// An all-NA str/datetime column yields `volas.NA`; numeric yields `np.float64`
/// NaN (the existing [`Column::extreme`] behaviour). Backs `Series.min`/`max`.
pub(crate) fn extreme_value(py: Python<'_>, col: &Column, want_max: bool) -> Py<PyAny> {
    match col {
        Column::Str(..) | Column::Datetime(..) => match col.arg_extreme(want_max) {
            Some(i) => scalar_to_py(py, col, i),
            None => na(py),
        },
        _ => scalar_to_numpy(py, col.extreme(want_max)),
    }
}

/// Render an index label at position `i` as a **typed** Python object: a
/// [`Timestamp`](PyTimestamp) for a DatetimeIndex (carrying the frame tz, so
/// `df.loc[row.name]` round-trips on the absolute instant), else the int / str
/// label. Display layers render the readable string form separately.
pub(crate) fn label_to_py(py: Python<'_>, index: &Index, i: usize) -> Py<PyAny> {
    match index.kind() {
        IndexKind::Datetime(v, tz) => Py::new(py, PyTimestamp { ns: v[i], tz: *tz })
            .unwrap()
            .into_any(),
        IndexKind::Int64(v) => v[i].into_pyobject(py).unwrap().into_any().unbind(),
        IndexKind::Range(_) => (i as i64).into_pyobject(py).unwrap().into_any().unbind(),
        IndexKind::Str(v) => v.get(i).into_pyobject(py).unwrap().into_any().unbind(),
    }
}

/// Parse a Python timestamp to UTC epoch-ns, interpreting a **naive** string in
/// `tz`. A [`PyTimestamp`] carries its own tz and resolves to an absolute instant
/// (so it matches across zones); an offset-aware string is already absolute; an
/// integer is epoch-ns.
pub(crate) fn parse_ts_in_tz(key: &Bound<'_, PyAny>, tz: Tz) -> PyResult<i64> {
    // None is not a constructable instant: volas has no NaT scalar — a missing
    // instant is volas.NA (decision 2 / D2). A clean ValueError, never a label
    // KeyError (F16).
    if key.is_none() {
        return Err(PyValueError::new_err(
            "None is not a valid timestamp; a missing instant is volas.NA",
        ));
    }
    if let Ok(ts) = key.extract::<PyRef<PyTimestamp>>() {
        return Ok(ts.ns);
    }
    if let Ok(s) = key.extract::<String>() {
        return datetime::parse_ns_in_tz(&s, tz)
            .ok_or_else(|| PyKeyError::new_err(format!("invalid datetime label {s:?}")));
    }
    // A stdlib `datetime.datetime` / `datetime.date` — and their subclasses,
    // notably `pd.Timestamp` and `pd.NaT` (F15/F23): round-trip through
    // `isoformat()`, which the string parser understands for both naive (then
    // interpreted in `tz`) and offset-aware (absolute) forms. `pd.NaT`
    // isoformat()s to "NaT", failing the parse -> a clean ValueError
    // (decision 2), not a label KeyError.
    if key.is_instance_of::<pyo3::types::PyDate>() || key.is_instance_of::<pyo3::types::PyDateTime>()
    {
        let s = key.call_method0("isoformat")?.extract::<String>()?;
        return datetime::parse_ns_in_tz(&s, tz).ok_or_else(|| {
            PyValueError::new_err(format!(
                "{s:?} is not a valid timestamp; a missing instant is volas.NA"
            ))
        });
    }
    if let Ok(i) = key.extract::<i64>() {
        // i64::MIN is the NaT sentinel (D2): a missing instant is volas.NA, not a
        // constructable timestamp. Reject it here — the single chokepoint for
        // Timestamp(...), TimeFrame.unify(...), and datetime label lookups — so a
        // raw NaT can never become a bucketable / civil timestamp (which would
        // render "NaT" yet expose a real 1677 .year, an internal inconsistency).
        if i == i64::MIN {
            return Err(PyValueError::new_err(
                "NaT (i64::MIN) is not a valid timestamp; a missing instant is volas.NA",
            ));
        }
        return Ok(i);
    }
    // A numpy datetime64 scalar (any unit): take its epoch-ns. This lets a Timestamp
    // cell compare against `np.datetime64(...)` and lets `df.loc[np.datetime64(...)]`
    // resolve — the same instant, just spelled in numpy's vocabulary. Detected by
    // dtype.kind == "M" so non-datetime numpy scalars still fall through to the error.
    if key
        .getattr("dtype")
        .and_then(|d| d.getattr("kind"))
        .and_then(|k| k.extract::<String>())
        .map(|k| k == "M")
        .unwrap_or(false)
    {
        let ns = key
            .call_method1("astype", ("datetime64[ns]",))?
            .call_method1("astype", ("int64",))?
            .call_method0("item")?
            .extract::<i64>()?;
        if ns == i64::MIN {
            return Err(PyValueError::new_err(
                "NaT (i64::MIN) is not a valid timestamp; a missing instant is volas.NA",
            ));
        }
        return Ok(ns);
    }
    Err(PyKeyError::new_err(
        "label must be a datetime string, integer, or numpy datetime64",
    ))
}

/// Build the `.index` as a NumPy array. A DatetimeIndex exports its **UTC**
/// instants as `datetime64[ns]` (matching pandas `.values`; the frame tz governs
/// string rendering / matching, not the numeric export); a string index becomes
/// an object array.
pub(crate) fn index_to_numpy<'py>(py: Python<'py>, index: &Index) -> PyResult<Bound<'py, PyAny>> {
    match index.kind() {
        IndexKind::Datetime(v, _) => {
            let arr = v.clone().into_pyarray(py);
            Ok(arr.call_method1("astype", ("datetime64[ns]",))?)
        }
        IndexKind::Int64(v) => Ok(v.clone().into_pyarray(py).into_any()),
        IndexKind::Range(n) => Ok((0..*n as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py)
            .into_any()),
        IndexKind::Str(v) => {
            let list = PyList::new(py, v.as_slice())?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", "object")?;
            Ok(py
                .import("numpy")?
                .call_method("array", (list,), Some(&kwargs))?)
        }
    }
}

/// The `[start, end)` row window of pandas `head` / `tail` over `len` rows.
/// `n` follows Python slicing (pandas-identical): `head(n)` is `iloc[:n]` — a
/// negative `n` drops the last `-n` rows — and `tail(n)` is `iloc[-n:]` — a
/// negative `n` drops the first `-n` rows. So `head(-1)` is "all but the last
/// row" and `tail(-1)` is "all but the first"; past-the-length values clamp.
pub(crate) fn head_tail_window(n: isize, len: usize, is_head: bool) -> (usize, usize) {
    let l = len as isize;
    if is_head {
        let end = if n >= 0 { n.min(l) } else { (l + n).max(0) };
        (0, end as usize)
    } else {
        let start = if n >= 0 { (l - n).max(0) } else { (-n).min(l) };
        (start as usize, len)
    }
}

/// Resolve a possibly-negative index to `[0, len)`.
pub(crate) fn norm_idx(i: isize, len: usize) -> PyResult<usize> {
    let n = len as isize;
    let i = if i < 0 { i + n } else { i };
    if i < 0 || i >= n {
        Err(PyIndexError::new_err("index out of range"))
    } else {
        Ok(i as usize)
    }
}

/// Parse column `dc` to a UTC `Datetime` column, move it into the index, and tag
/// the index with `tz` (PD-20 ingestion). A **naive** string column is
/// interpreted in `tz`; with `date_unit` the column is read as an epoch integer
/// (`"s"`/`"ms"`/`"us"`/`"ns"`, absolute) and `tz` only sets the display zone.
pub(crate) fn build_datetime_index(
    mut df: DataFrame,
    dc: &str,
    tz: Option<&str>,
    date_unit: Option<&str>,
) -> PyResult<DataFrame> {
    let tzv = match tz {
        Some(s) => Tz::parse(s).map_err(pyerr)?,
        None => Tz::Naive,
    };
    let parsed = match date_unit {
        Some(unit) => df
            .column(dc)
            .map_err(pyerr)?
            .epoch_to_datetime(unit)
            .map_err(pyerr)?,
        None => df
            .column(dc)
            .map_err(pyerr)?
            .to_datetime_tz(tzv)
            .map_err(pyerr)?,
    };
    df.set_column(dc, parsed).map_err(pyerr)?;
    let mut df = df.set_index(dc).map_err(pyerr)?;
    // The instants are already correct UTC; tag the display / matching tz.
    if !tzv.is_utc() {
        df = df.tz_convert(tzv).map_err(pyerr)?;
    }
    Ok(df)
}
