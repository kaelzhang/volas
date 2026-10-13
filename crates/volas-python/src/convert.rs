//! Boundary marshalling between volas columns and NumPy / pandas (export,
//! scalar boxing, the numpy-type cache, and index/label rendering).


use numpy::IntoPyArray;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyDict, PyList, PyTuple};
use volas_core::{
    datetime, Column, DType, Index,
    IndexKind, Scalar, Tz, Validity,
};

#[allow(unused_imports)]
use crate::*;

/// Like [`column_to_numpy`] but **consumes** the column, moving its backing buffer
/// straight into the NumPy array with no copy when the column is uniquely owned —
/// the fresh-result path (`df.exec(directive)`). A bool / int column holding a
/// missing cell, and the rarer `Str` / `Datetime` columns, take the borrowing export
/// instead, so a missing cell is exported as missing — never as its payload.
pub(crate) fn column_into_numpy<'py>(py: Python<'py>, col: Column) -> PyResult<Bound<'py, PyAny>> {
    Ok(match col {
        Column::F64(a) => a.into_vec().into_pyarray(py).into_any(),
        Column::Bool(a, val) if !val.has_nulls() => a.into_vec().into_pyarray(py).into_any(),
        Column::I64(a, val) if !val.has_nulls() => a.into_vec().into_pyarray(py).into_any(),
        other => return column_to_numpy(py, &other),
    })
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
    // A datetime column holds UTC instants: pandas gets `datetime64[ns, UTC]` (the
    // NumPy export has no zone slot), under either backend.
    if let Column::Datetime(_) = col {
        return pd
            .call_method1("array", (column_to_numpy(py, col)?,))?
            .call_method1("tz_localize", (Tz::DATETIME_COLUMN.name(),));
    }
    // A bool column holding a missing cell has no NumPy form (see
    // `column_to_numpy`), so it is pandas' masked `boolean` under either backend —
    // `pd.NA`, never a truthy `nan`.
    let bool_na = matches!(col, Column::Bool(_, val) if val.has_nulls());
    if !nullable && !bool_na {
        return column_to_numpy(py, col);
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
        // float (NaN in-band) has no nullable masked form here.
        _ => column_to_numpy(py, col),
    }
}

/// `DataFrame.from_pandas`: build a volas frame from a `pandas.DataFrame` — the faithful
/// inverse of [`column_to_pandas`] / `to_pandas`. Each column is lowered to a value the
/// frame constructor already ingests (a numpy array, a list, or the typed masked
/// `(values, mask, dtype)` channel), then the index is carried natively. pandas is
/// imported lazily here, so volas stays pandas-free at import.
pub(crate) fn frame_from_pandas(py: Python<'_>, pdf: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
    let pd = py.import("pandas")?;
    let types = pd.getattr("api")?.getattr("types")?;

    // Each column -> a constructor-ingestible value, keyed by its (stringified) name.
    let data = PyDict::new(py);
    for c in pdf.getattr("columns")?.try_iter()? {
        let c = c?;
        let series = pdf.get_item(&c)?;
        data.set_item(c.str()?.to_string(), pandas_series_to_values(py, &types, &series)?)?;
    }

    // A default RangeIndex carries no labels — a fresh volas RangeIndex (just the columns).
    let idx = pdf.getattr("index")?;
    if idx.is_instance(&pd.getattr("RangeIndex")?)? {
        return PyDataFrame::new(data.as_any(), None, None, None, None, None, None, None);
    }

    // The index is built as an index directly — never routed through the column
    // namespace, where it would collide with (and replace) a same-named column. An
    // unnamed pandas index stays unnamed.
    let idx_name = {
        let n = idx.getattr("name")?;
        if n.is_none() { None } else { Some(n.str()?.to_string()) }
    };
    let datetime_index = types
        .call_method1("is_datetime64_any_dtype", (idx.getattr("dtype")?,))?
        .extract::<bool>()?;
    let instants = if datetime_index {
        // The absolute (UTC) instants in their native unit (no strftime round-trip);
        // the column coercion scales them to ns, checked.
        idx.getattr("values")?
    } else {
        idx.call_method0("to_numpy")?
    };
    let labels = pyany_to_column(&instants)?;
    let cols = PyDataFrame::new(data.as_any(), None, None, None, None, None, None, None)?.inner;
    // labelled as set_index labels a frame: a repeated label raises
    let df = PyDataFrame::plain(cols.with_labels(&labels, Tz::Naive, idx_name).map_err(pyerr)?);

    // The imported instants are already true UTC; re-attach a tz-aware index's display
    // zone directly (tz_convert would hit the naive-axis guard; localize would shift).
    if datetime_index {
        if let Some(tz) = idx.getattr("tz").ok().filter(|t| !t.is_none()) {
            // pandas renders a fixed offset as 'UTC+08:00'; volas's tz name for that is '+08:00'.
            let mut tzname = tz.str()?.to_string();
            if tzname.starts_with("UTC+") || tzname.starts_with("UTC-") {
                tzname = tzname[3..].to_string();
            }
            return df._set_index_tz(&tzname);
        }
    }
    Ok(df)
}

/// One pandas column -> a value the volas frame constructor ingests: a `datetime64[ns]`
/// numpy array for datetimes; the typed masked `(values, mask, dtype)` tuple for a
/// nullable `Int64` / `Int32` / `boolean` / `string` column (keeping its declared dtype
/// plus `volas.NA`, even all-NA / empty); a plain list for an `object` column; otherwise
/// the numpy array.
fn pandas_series_to_values<'py>(
    py: Python<'py>,
    types: &Bound<'py, PyAny>,
    s: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let dtype = s.getattr("dtype")?;
    // datetime (naive or tz-aware) -> its UTC instants as a NumPy datetime64 array in
    // the native unit; the column coercion scales them to ns, checked (NumPy's own
    // `datetime64[ns]` cast would wrap an out-of-range coarse value).
    if types.call_method1("is_datetime64_any_dtype", (&dtype,))?.extract::<bool>()? {
        return s.getattr("values");
    }
    // A nullable masked int / boolean / string column -> the typed masked channel, so
    // volas keeps the DECLARED dtype + volas.NA (an Int32 stays int32; an all-NA / empty
    // column keeps its dtype). The inverse of `to_pandas(dtype_backend='numpy_nullable')`.
    if types.call_method1("is_extension_array_dtype", (&dtype,))?.extract::<bool>()? {
        let is_int = types.call_method1("is_integer_dtype", (&dtype,))?.extract::<bool>()?;
        let is_bool = types.call_method1("is_bool_dtype", (&dtype,))?.extract::<bool>()?;
        let is_str = types.call_method1("is_string_dtype", (&dtype,))?.extract::<bool>()?;
        if is_int || is_bool || is_str {
            let mask = s.call_method0("isna")?.call_method0("tolist")?;
            let masked = mask.extract::<Vec<bool>>()?;
            let (volas_dtype, filler): (&str, Bound<'py, PyAny>) = if is_int {
                let dt = if dtype.str()?.to_str()? == "Int32" { "int32" } else { "int64" };
                (dt, 0i64.into_pyobject(py)?.into_any())
            } else if is_bool {
                ("bool", false.into_pyobject(py)?.to_owned().into_any())
            } else {
                ("str", "".into_pyobject(py)?.into_any())
            };
            // values: present cells verbatim, missing cells the dtype's filler — so the
            // tuple's value list extracts cleanly as a typed `Vec<dtype>`.
            let values = PyList::empty(py);
            for (cell, &m) in s.call_method0("tolist")?.try_iter()?.zip(masked.iter()) {
                if m {
                    values.append(&filler)?;
                } else {
                    values.append(cell?)?;
                }
            }
            let tuple = PyTuple::new(py, [values.into_any(), mask, volas_dtype.into_pyobject(py)?.into_any()])?;
            return Ok(tuple.into_any());
        }
    }
    // A pandas `object` column of plain strings -> a `str` column (the constructor rejects
    // an object column mixing strings with other values; volas has no `object` dtype).
    if dtype.getattr("kind")?.extract::<String>()? == "O" {
        return s.call_method0("tolist");
    }
    // numeric / bool / other native -> a numpy array.
    s.call_method0("to_numpy")
}

pub(crate) fn column_to_numpy<'py>(py: Python<'py>, col: &Column) -> PyResult<Bound<'py, PyAny>> {
    Ok(match col {
        Column::F64(v) => v.to_vec().into_pyarray(py).into_any(),
        Column::F32(v) => v.to_vec().into_pyarray(py).into_any(),
        // numpy int/bool cannot hold a missing value, so an int column with any NA
        // exports as float64 with NaN (pandas `Int64.to_numpy()` semantics); a
        // dense column keeps its native dtype.
        Column::Bool(v, val) if !val.has_nulls() => v.to_vec().into_pyarray(py).into_any(),
        Column::I64(v, val) if !val.has_nulls() => v.to_vec().into_pyarray(py).into_any(),
        Column::I32(v, val) if !val.has_nulls() => v.to_vec().into_pyarray(py).into_any(),
        // A bool column with a missing cell has no NumPy form: any placeholder would
        // be read as a truth value (a `nan` is truthy, `None` falsy) — an unknown
        // signal must not act like a decided one (the rule that rejects NA masks).
        // The caller states what a missing cell means instead.
        Column::Bool(..) => return Err(bool_na_err()),
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
        Column::Datetime(v) => v.to_vec().into_pyarray(py).call_method1("astype", ("datetime64[ns]",))?,
    })
}

/// The error of exporting a bool column holding a missing value without saying what
/// that cell means.
pub(crate) fn bool_na_err() -> PyErr {
    PyValueError::new_err(
        "cannot export a bool column with missing values to NumPy: NumPy has no missing \
         bool, and any placeholder would be read as True or False — pass na_value= \
         (e.g. na_value=False), dtype='float64' (missing -> NaN), or dtype='object' \
         (missing -> volas.NA)",
    )
}

/// A bool column's cells as an object array, `fill` at each missing cell.
fn bool_cells<'py>(py: Python<'py>, v: &[bool], val: &Validity, fill: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let items = (0..v.len()).map(|i| {
        if val.is_valid(i) {
            pyo3::types::PyBool::new(py, v[i]).to_owned().into_any()
        } else {
            fill.clone()
        }
    });
    let list = PyList::new(py, items)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "object")?;
    py.import("numpy")?.call_method("array", (list,), Some(&kwargs))
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
    // `object` is the lossless export: each cell its own typed value (a UTC
    // `Timestamp`, an exact int, `volas.NA` — or `na_value` — for a missing cell),
    // exactly like a frame's `to_numpy(dtype="object")`.
    if let Some(dt) = dtype {
        if ExportTarget::of(py, dt)? == ExportTarget::Object {
            let cells = (0..col.len()).map(|i| match na_value {
                Some(nv) if !col.is_valid(i) => nv.clone().unbind(),
                _ => scalar_to_py(py, col, i),
            });
            let list = PyList::new(py, cells)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", "object")?;
            return py.import("numpy")?.call_method("array", (list,), Some(&kwargs));
        }
    }
    let Some(nv) = na_value else {
        return match dtype {
            None => column_to_numpy(py, col),
            Some(dt) => astype_checked(py, col, dt),
        };
    };
    if !col.has_missing() {
        // `na_value` is irrelevant with no missing cell — a plain typed export.
        let arr = column_to_numpy(py, col)?;
        return match dtype {
            Some(dt) => arr.call_method1("astype", (dt,)),
            None => Ok(arr),
        };
    }
    // a bool column's cells, the fill at each hole: an object array the cast (or a
    // bool fill) turns into the target — never a NaN coerced into a bool slot
    if let Column::Bool(v, val) = col {
        let cells = bool_cells(py, v, val, nv)?;
        return match dtype {
            Some(dt) => cells.call_method1("astype", (dt,)),
            None if nv.is_instance_of::<pyo3::types::PyBool>() || type_is_numpy_bool(nv) => {
                cells.call_method1("astype", ("bool",))
            }
            None => Ok(cells),
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
            let base = column_to_numpy(py, col)?;
            base.set_item(mask, nv)?;
            Ok(base)
        }
    }
}

/// Whether `v` is a NumPy bool scalar (`numpy.bool_`).
fn type_is_numpy_bool(v: &Bound<'_, PyAny>) -> bool {
    v.get_type().name().is_ok_and(|n| n.to_str().is_ok_and(|n| n == "bool" || n == "bool_"))
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
        Column::Str(..) => column_to_numpy(py, col)?,
        Column::Datetime(v) => {
            v.to_vec().into_pyarray(py).call_method1("astype", ("datetime64[ns]",))?
        }
    })
}

/// Cast an exported NumPy array to `dtype`, but **raise** (pandas-aligned) when the
/// column holds missing values and `dtype` cannot represent one (integer / bool) —
/// NumPy would silently write garbage (or `True` for a NaN) instead.
/// A float target reads a bool column's missing cell as `NaN` (the float channel);
/// any other target of a bool column holding NA has no form for it.
pub(crate) fn astype_checked<'py>(py: Python<'py>, col: &Column, dtype: &str) -> PyResult<Bound<'py, PyAny>> {
    let target = ExportTarget::of(py, dtype)?;
    if col.has_missing() && target.has_no_na() {
        return Err(no_na_target_err("a column", dtype));
    }
    match col {
        Column::Bool(_, val) if val.has_nulls() => match target {
            ExportTarget::Float => col.to_f64_vec().into_pyarray(py).call_method1("astype", (dtype,)),
            _ => Err(bool_na_err()),
        },
        _ => column_to_numpy(py, col)?.call_method1("astype", (dtype,)),
    }
}

/// The error for an NA-holding export into a dtype that cannot represent NA.
pub(crate) fn no_na_target_err(what: &str, dtype: &str) -> PyErr {
    PyValueError::new_err(format!(
        "cannot convert {what} with missing values to NumPy dtype '{dtype}' (an NA has \
         no integer or boolean representation) — pass na_value=, or use a float dtype"
    ))
}

/// The family of a NumPy export `dtype`, decided by NumPy's own resolution of the
/// spelling (`numpy.dtype(dtype).kind`) — so `'f8'`, `'d'`, `'<f8'` and `'float64'`
/// all land in one family, never by matching the spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExportTarget {
    /// `kind` `f` / `c`: floating or complex.
    Float,
    /// `kind` `i` / `u`: signed / unsigned integer.
    Integer,
    /// `kind` `b`.
    Bool,
    /// `kind` `M` / `m`: datetime64 / timedelta64.
    Temporal,
    /// `kind` `O`.
    Object,
    /// Any other kind (`U` / `S` text, `V` void).
    Other,
}

impl ExportTarget {
    pub(crate) fn of(py: Python<'_>, dtype: &str) -> PyResult<ExportTarget> {
        let kind = py
            .import("numpy")?
            .call_method1("dtype", (dtype,))?
            .getattr("kind")?
            .extract::<String>()?;
        Ok(match kind.as_str() {
            "f" | "c" => ExportTarget::Float,
            "i" | "u" => ExportTarget::Integer,
            "b" => ExportTarget::Bool,
            "M" | "m" => ExportTarget::Temporal,
            "O" => ExportTarget::Object,
            _ => ExportTarget::Other,
        })
    }

    /// Whether the target has no representation for a missing value.
    pub(crate) fn has_no_na(self) -> bool {
        matches!(self, ExportTarget::Integer | ExportTarget::Bool)
    }
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
        // singleton). A column holds UTC instants, so the cell is UTC-aware.
        Column::Datetime(v) if v[i] == i64::MIN => na(py),
        Column::Datetime(v) => Py::new(py, PyTimestamp { ns: v[i], tz: Tz::DATETIME_COLUMN })
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
        IndexKind::Range { start, .. } => (start + i as i64).into_pyobject(py).unwrap().into_any().unbind(),
        IndexKind::Str(v) => v.get(i).into_pyobject(py).unwrap().into_any().unbind(),
    }
}

/// The zone a Python timestamp is written in.
pub(crate) enum Zone {
    /// An instant: an aware `Timestamp` / `datetime`, an offset-carrying string.
    Aware(Tz),
    /// A wall clock with no zone: a naive `Timestamp` / `datetime` / string.
    Naive,
    /// No zone either way — a raw epoch-ns count: an integer or a NumPy `datetime64`
    /// (the form a frame exports its instants in), read in the frame's own zone.
    Unzoned,
}

/// The zone `key` (a timestamp as `parse_ts_in_tz` accepts it) is written in.
pub(crate) fn timestamp_zone(key: &Bound<'_, PyAny>) -> PyResult<Zone> {
    if let Ok(ts) = key.extract::<PyRef<PyTimestamp>>() {
        return Ok(if ts.tz.is_aware() { Zone::Aware(ts.tz) } else { Zone::Naive });
    }
    if let Ok(s) = key.extract::<String>() {
        return match datetime::offset_suffix_secs(&s) {
            Some(offset) => Ok(Zone::Aware(Tz::fixed(offset).map_err(pyerr)?)),
            None => Ok(Zone::Naive),
        };
    }
    if key.is_instance_of::<pyo3::types::PyDateTime>() {
        let off = key.call_method0("utcoffset")?;
        if !off.is_none() {
            return Ok(Zone::Aware(Tz::fixed(crate::scalar::offset_secs(&off)?).map_err(pyerr)?));
        }
        return Ok(Zone::Naive);
    }
    if key.extract::<i64>().is_ok() || crate::coerce::numpy_time_value(key, "M").is_some() {
        return Ok(Zone::Unzoned);
    }
    Ok(Zone::Naive)
}

/// The zone row labels given as `index=` are written in: the zone of a pandas
/// `DatetimeIndex`, or the one zone every aware `Timestamp` / `datetime` among them
/// carries (a frame's own aware `.index`); `None` for wall clocks and raw values.
/// Aware and naive timestamps, or two zones, in one index raise.
pub(crate) fn labels_zone(labels: &Bound<'_, PyAny>) -> PyResult<Option<Tz>> {
    if let Ok(tz) = labels.getattr("tz") {
        if !tz.is_none() {
            return Ok(Some(crate::scalar::tz_from_py(&tz)?));
        }
        return Ok(None);
    }
    let raw = labels.getattr("dtype").and_then(|d| d.getattr("kind")).is_ok_and(|k| !k.eq("O").unwrap_or(false));
    let Ok(items) = labels.try_iter() else { return Ok(None) };
    if raw || labels.is_instance_of::<pyo3::types::PyString>() {
        return Ok(None);
    }
    let mut seen: Option<Option<Tz>> = None;
    for item in items {
        let item = item?;
        let timestamp = item.extract::<PyRef<PyTimestamp>>().is_ok() || item.is_instance_of::<pyo3::types::PyDateTime>();
        if !timestamp {
            continue;
        }
        let zone = match timestamp_zone(&item)? {
            Zone::Aware(tz) => Some(tz),
            _ => None,
        };
        match seen {
            None => seen = Some(zone),
            Some(first) if first == zone => {}
            Some(first) => {
                return Err(PyValueError::new_err(match (first, zone) {
                    (Some(a), Some(b)) => format!(
                        "cannot use timestamps in two zones ({} and {}) as one index — \
                         tz_convert them to one zone first",
                        a.name(),
                        b.name()
                    ),
                    _ => "cannot use tz-aware and tz-naive timestamps as one index — \
                          tz_localize the naive ones first"
                        .to_string(),
                }))
            }
        }
    }
    Ok(seen.flatten())
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
        // Text names no instant when it is not a datetime or names a wall clock `tz`
        // skips (a DST gap) or repeats (a fold): a ValueError saying which, with the
        // zone (pandas' NonExistentTimeError / AmbiguousTimeError are ValueErrors); an
        // instant outside the datetime64[ns] range is an OverflowError. A row-label
        // lookup turns the first into a missing label ([`parse_label_in_tz`]).
        return datetime::parse_ns_in_tz(&s, tz).map_err(pyerr);
    }
    // A stdlib `datetime.datetime` / `datetime.date` — and their subclasses,
    // notably `pd.Timestamp` and `pd.NaT` (F15/F23): read from its components
    // (exact to the nanosecond, any second-precision offset) — an aware value at
    // its own offset, a naive one as a wall clock in `tz`. `pd.NaT` has no
    // components -> a clean ValueError (decision 2), not a label KeyError.
    if key.is_instance_of::<pyo3::types::PyDate>() {
        return crate::scalar::datetime_instant(key, tz);
    }
    if let Ok(i) = key.extract::<i64>() {
        // i64::MIN is the NaT sentinel (D2): a missing instant is volas.NA, not a
        // constructable timestamp. Reject it here — the single chokepoint for
        // Timestamp(...), datetime cell writes and datetime label lookups — so a
        // raw NaT can never become a civil timestamp (which would render "NaT"
        // yet expose a real 1677 .year, an internal inconsistency).
        if i == datetime::NAT {
            return Err(PyValueError::new_err(
                "NaT (i64::MIN) is not a valid timestamp; a missing instant is volas.NA",
            ));
        }
        return Ok(i);
    }
    // A numpy datetime64 scalar (any unit): its native-unit value scaled by the core
    // unit owner (checked; NumPy's own ns cast would wrap). This lets a Timestamp cell
    // compare against `np.datetime64(...)` and lets `df.loc[np.datetime64(...)]`
    // resolve — the same instant, just spelled in numpy's vocabulary. Non-datetime
    // numpy scalars fall through to the error.
    if let Some(value) = crate::coerce::numpy_time_value(key, "M") {
        let (unit, count, raw) = value?;
        if raw == datetime::NAT {
            return Err(PyValueError::new_err(
                "NaT (i64::MIN) is not a valid timestamp; a missing instant is volas.NA",
            ));
        }
        return datetime::datetime64_to_ns(raw, &unit, count).map_err(pyerr);
    }
    Err(PyTypeError::new_err(format!(
        "cannot convert {} to a timestamp: expected a datetime string, a datetime, an \
         integer (epoch ns) or a numpy datetime64",
        key.get_type().name()?
    )))
}

/// A row label of a datetime index in `tz` (an integer index reads `Tz::Naive`): the
/// instant [`parse_ts_in_tz`] builds, except that a key which names no instant — text
/// that is not a datetime or names a DST gap / fold wall clock, or a value of no
/// timestamp type — is a missing label: a KeyError carrying the reason, as pandas'
/// `.loc` raises KeyError for it. `None`, NaT and an out-of-range instant keep their
/// ValueError / OverflowError.
pub(crate) fn parse_label_in_tz(key: &Bound<'_, PyAny>, tz: Tz) -> PyResult<i64> {
    parse_ts_in_tz(key, tz).map_err(|e| {
        let py = key.py();
        let text = key.is_instance_of::<pyo3::types::PyString>();
        if (text && e.is_instance_of::<PyValueError>(py)) || e.is_instance_of::<PyTypeError>(py) {
            let label = key.repr().map_or_else(|_| "?".to_string(), |r| r.to_string());
            PyKeyError::new_err(format!("invalid datetime label {label}: {}", e.value(py)))
        } else {
            e
        }
    })
}

/// The `.index` row labels as a NumPy array. A DatetimeIndex exports its **UTC**
/// instants as `datetime64[ns]` (NumPy has no zone slot, the rule for datetime data
/// at the NumPy boundary; such a label read back is the same instant, see
/// [`Zone::Unzoned`]); a string index becomes an object array.
pub(crate) fn index_values<'py>(py: Python<'py>, index: &Index) -> PyResult<Bound<'py, PyAny>> {
    match index.kind() {
        IndexKind::Datetime(v, _) => {
            let arr = v.clone().into_pyarray(py);
            Ok(arr.call_method1("astype", ("datetime64[ns]",))?)
        }
        IndexKind::Int64(v) => Ok(v.clone().into_pyarray(py).into_any()),
        IndexKind::Range { start, len } => Ok((*start..*start + *len as i64)
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

/// A `round` place count: any Python `int`, as Python's `round` takes one. Past ±400
/// places every float result has saturated (the value itself, or a signed 0), so a count
/// beyond `i32` is clamped to it; a non-integer is a `TypeError`.
#[derive(Clone, Copy)]
pub(crate) struct DecimalPlaces(pub(crate) i32);

impl<'py> FromPyObject<'py> for DecimalPlaces {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if !ob.is_instance_of::<pyo3::types::PyInt>() {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "decimals must be an integer, got {}",
                ob.get_type().name()?
            )));
        }
        Ok(DecimalPlaces(match ob.extract::<i64>() {
            Ok(k) => k.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
            Err(_) if ob.gt(0)? => i32::MAX,
            Err(_) => i32::MIN,
        }))
    }
}
