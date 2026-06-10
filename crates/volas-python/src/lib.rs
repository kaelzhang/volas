//! PyO3 bindings: expose the `volas` kernel to Python as `volas_rs.DataFrame` /
//! `volas_rs.Series`, with stock-pandas-style directive indexing and a
//! pandas-compatible indexing surface (`.iloc`, `.index`, `.name`, label lookup).
//!
//! This crate is the only place pyo3 / numpy are used; all logic lives in the
//! `volas-core` / `volas-compute` / `volas-directive` crates.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::create_exception;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyDict, PyList, PySlice, PySliceIndices, PyTuple};

use volas_core::{
    binary_supertype, datetime, fits, stats, BinOp, BoolOp, CmpOp, Column, DType, DataFrame, Index,
    IndexKind, Label, Scalar, Series, Tz, Validity, VolasError,
};
use volas_directive::{execute, parse};
use volas_time::{aggregate_period, AggSpec, Cumulator, TimeFrame};

mod format;
mod readers;
mod timeframe;

use format::{
    cell_to_csv, index_label_csv, parse_float_format, render_frame, render_frame_html, render_row,
    render_series, Dimensions, DisplayOpts, NA_REPR,
};
use readers::read_csv;
use timeframe::{build_agg_spec, resolve_time_frame, PyTimeFrame};

// --- helpers ---------------------------------------------------------------

pub(crate) fn pyerr(e: VolasError) -> PyErr {
    match e {
        VolasError::ColumnNotFound(n) => PyKeyError::new_err(format!("column \"{n}\" not found")),
        VolasError::DType(m) => PyTypeError::new_err(m),
        VolasError::Shape(m) | VolasError::Index(m) | VolasError::Value(m) => {
            PyValueError::new_err(m)
        }
    }
}

// Typed directive exceptions (both subclass ValueError, so existing
// `except ValueError` keeps working while callers can catch the specific type).
create_exception!(
    volas_rs,
    DirectiveError,
    PyValueError,
    "Base class for volas directive errors."
);
create_exception!(
    volas_rs,
    DirectiveSyntaxError,
    DirectiveError,
    "A directive string could not be parsed (with line / column)."
);
create_exception!(
    volas_rs,
    DirectiveValueError,
    DirectiveError,
    "A directive has an unknown command / sub-command or an invalid argument."
);

/// Map a parse-time error to `DirectiveSyntaxError`.
fn syntax_err(e: VolasError) -> PyErr {
    match e {
        VolasError::Value(m) => DirectiveSyntaxError::new_err(m),
        other => DirectiveSyntaxError::new_err(other.to_string()),
    }
}

/// Map a directive execution error to `DirectiveValueError`.
fn value_err(e: VolasError) -> PyErr {
    match e {
        VolasError::Value(m) => DirectiveValueError::new_err(m),
        VolasError::ColumnNotFound(n) => {
            DirectiveValueError::new_err(format!("column \"{n}\" not found"))
        }
        other => pyerr(other),
    }
}

fn directive_uses_default_series(node: &volas_directive::types::Node) -> bool {
    match node {
        volas_directive::types::Node::Name(_) => true,
        volas_directive::types::Node::Command(cmd) => cmd.series.iter().all(
            |series| matches!(series, volas_directive::types::Node::Name(name) if name.is_empty()),
        ),
        _ => false,
    }
}

/// Parse a pandas-style dtype string to a volas [`DType`].
fn parse_dtype(s: &str) -> PyResult<DType> {
    Ok(match s {
        "float" | "float64" | "float_" | "double" | "f64" => DType::F64,
        "float32" | "single" | "f32" => DType::F32,
        "int" | "int64" | "int_" | "long" | "i64" => DType::I64,
        "int32" | "i32" => DType::I32,
        "bool" | "boolean" => DType::Bool,
        "str" | "string" | "object" | "O" => DType::Utf8,
        "datetime" | "datetime64" | "datetime64[ns]" => DType::Datetime,
        _ => return Err(PyValueError::new_err(format!("unknown dtype {s:?}"))),
    })
}

/// The epoch unit a `datetime64[...]` dtype string implies, or `None` when `s` is
/// not a datetime dtype. Bare `datetime` / `datetime64` / `datetime64[ns]` mean
/// nanoseconds; `datetime64[s|ms|us]` carry their own unit (pandas-aligned).
fn datetime_unit_of(s: &str) -> Option<&'static str> {
    match s {
        "datetime" | "datetime64" | "datetime64[ns]" => Some("ns"),
        "datetime64[s]" => Some("s"),
        "datetime64[ms]" => Some("ms"),
        "datetime64[us]" => Some("us"),
        _ => None,
    }
}

fn pyany_to_column(v: &Bound<'_, PyAny>) -> PyResult<Column> {
    // A volas Series is an internal column, not a numpy boundary value: clone its
    // Column directly (preserving dtype + volas.NA). Without this it would fall
    // through to the array path below, invoking Series.__array__ and re-importing
    // the lossy float64 export — contract C1/C2 (parity with `df['x'] = series`).
    if let Ok(s) = v.extract::<PyRef<PySeries>>() {
        return Ok(s.inner.data.clone());
    }
    // A numpy datetime64 array -> a Datetime column (carried as epoch-ns). This is the
    // pandas-aligned ingestion (`pd.DataFrame` accepts datetime64 arrays) and the no-copy
    // path `from_pandas` relies on. Normalise any datetime64[unit] to ns, then take its
    // int64 view (datetime64[ns] is int64 epoch-ns underneath).
    if v.getattr("dtype")
        .and_then(|d| d.getattr("kind"))
        .and_then(|k| k.extract::<String>())
        .map(|k| k == "M")
        .unwrap_or(false)
    {
        let ns = v.call_method1("astype", ("datetime64[ns]",))?;
        let view = ns.call_method1("view", ("int64",))?;
        let a = view.extract::<PyReadonlyArray1<i64>>()?;
        return Ok(Column::datetime(a.as_slice()?.to_vec())); // int64 ns view == epoch-ns
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<f64>>() {
        return Ok(Column::f64(a.as_slice()?.to_vec()));
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<f32>>() {
        return Ok(Column::f32(a.as_slice()?.to_vec()));
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<i64>>() {
        return Ok(Column::i64(a.as_slice()?.to_vec()));
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<i32>>() {
        return Ok(Column::i32(a.as_slice()?.to_vec()));
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<bool>>() {
        return Ok(Column::bool(a.as_slice()?.to_vec()));
    }
    // A Python list infers its dtype like pandas/NumPy: all-bool -> bool, all-int
    // -> int64, else float64 (empty -> float64). Bool / int are tried before float
    // because a strict bool/int extraction won't match a float list.
    if let Ok(vv) = v.extract::<Vec<bool>>() {
        if !vv.is_empty() {
            return Ok(Column::bool(vv));
        }
    }
    if let Ok(vv) = v.extract::<Vec<i64>>() {
        if !vv.is_empty() {
            return Ok(Column::i64(vv));
        }
    }
    if let Ok(vv) = v.extract::<Vec<f64>>() {
        return Ok(Column::f64(vv));
    }
    if let Ok(vv) = v.extract::<Vec<String>>() {
        return Ok(Column::str(vv));
    }
    // A list containing `None` carries missing values: infer the present values'
    // dtype (bool -> int -> float, like the dense case) and mark each `None` cell
    // `volas.NA`. int/bool keep their dtype (pandas upcasts to float/object).
    if let Ok(vv) = v.extract::<Vec<Option<bool>>>() {
        if vv.iter().any(Option::is_some) {
            return Ok(option_bool_column(vv));
        }
    }
    if let Ok(vv) = v.extract::<Vec<Option<i64>>>() {
        if vv.iter().any(Option::is_some) {
            return Ok(option_i64_column(vv));
        }
    }
    if let Ok(vv) = v.extract::<Vec<Option<f64>>>() {
        // a float carries missing in-band as NaN (also the all-`None` fallback).
        return Ok(Column::f64(vv.iter().map(|x| x.unwrap_or(f64::NAN)).collect()));
    }
    if let Ok(vv) = v.extract::<Vec<Option<String>>>() {
        let validity = Validity::from_valid_iter(vv.len(), vv.iter().map(Option::is_some));
        return Ok(Column::str_with(vv.into_iter().map(Option::unwrap_or_default).collect(), validity));
    }
    // A list may carry the `volas.NA` symbol itself (not `None`) — normalise it to
    // `None` and retry, so `to_list()` output round-trips back into a frame.
    if let Ok(list) = v.downcast::<PyList>() {
        let py = v.py();
        let na_obj = na(py);
        let na_bound = na_obj.bind(py);
        if list.iter().any(|item| item.is(na_bound)) {
            let items: Vec<Bound<'_, PyAny>> = list
                .iter()
                .map(|item| if item.is(na_bound) { py.None().into_bound(py) } else { item })
                .collect();
            let normalized = PyList::new(py, items)?;
            return pyany_to_column(normalized.as_any());
        }
    }
    Err(PyTypeError::new_err(
        "column values must be a 1-D numeric array, a list of numbers, or a list of strings",
    ))
}

/// Build a `Bool` column from `Option`s, marking `None` cells `volas.NA`.
fn option_bool_column(vv: Vec<Option<bool>>) -> Column {
    let validity = Validity::from_valid_iter(vv.len(), vv.iter().map(Option::is_some));
    Column::bool_with(vv.iter().map(|x| x.unwrap_or(false)).collect(), validity)
}

/// Build an `I64` column from `Option`s, marking `None` cells `volas.NA`.
fn option_i64_column(vv: Vec<Option<i64>>) -> Column {
    let validity = Validity::from_valid_iter(vv.len(), vv.iter().map(Option::is_some));
    Column::i64_with(vv.iter().map(|x| x.unwrap_or(0)).collect(), validity)
}

/// Move an `Arc<Vec<T>>` out to an owned `Vec<T>` without copying when it is
/// uniquely owned; clone only if it is still shared.
fn arc_into_vec<T: Clone>(arc: Arc<Vec<T>>) -> Vec<T> {
    Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())
}

/// Like [`column_to_numpy`] but **consumes** the column, moving its backing `Vec`
/// straight into the NumPy array with no copy when the column is uniquely owned —
/// the fresh-result path (`df.exec(directive)`). Falls back to a borrow + copy for
/// the rarer `Str` / `Datetime` columns.
fn column_into_numpy<'py>(py: Python<'py>, col: Column) -> Bound<'py, PyAny> {
    match col {
        Column::F64(a) => arc_into_vec(a).into_pyarray(py).into_any(),
        Column::Bool(a, _) => arc_into_vec(a).into_pyarray(py).into_any(),
        Column::I64(a, _) => arc_into_vec(a).into_pyarray(py).into_any(),
        other => column_to_numpy(py, &other),
    }
}

/// A column as a pandas array for `to_pandas`. With `nullable`, an int/bool
/// column becomes a pandas masked `Int64` / `Int32` / `boolean` (a faithful,
/// lossless NA round-trip); otherwise it is the numpy export (a missing value
/// becomes NaN, like `Int64.to_numpy()`).
fn column_to_pandas<'py>(
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
        // float (NaN in-band) / str / datetime have no nullable masked form here.
        _ => Ok(column_to_numpy(py, col)),
    }
}

fn column_to_numpy<'py>(py: Python<'py>, col: &Column) -> Bound<'py, PyAny> {
    match col {
        Column::F64(v) => v.to_vec().into_pyarray(py).into_any(),
        Column::F32(v) => v.to_vec().into_pyarray(py).into_any(),
        // numpy int/bool cannot hold a missing value, so a column with any NA
        // exports as float64 with NaN (pandas `Int64.to_numpy()` semantics); a
        // dense column keeps its native dtype.
        Column::Bool(v, val) if !val.has_nulls() => v.to_vec().into_pyarray(py).into_any(),
        Column::I64(v, val) if !val.has_nulls() => v.to_vec().into_pyarray(py).into_any(),
        Column::I32(v, val) if !val.has_nulls() => v.to_vec().into_pyarray(py).into_any(),
        Column::Bool(..) | Column::I64(..) | Column::I32(..) => {
            col.to_f64_vec().into_pyarray(py).into_any()
        }
        // String columns become NumPy object arrays (pandas `object` dtype).
        Column::Str(v, val) => {
            // a missing cell becomes Python `None` in the object array (pandas parity)
            let items: Vec<Bound<'_, PyAny>> = (0..v.len())
                .map(|i| {
                    if val.is_valid(i) {
                        v[i].clone().into_pyobject(py).unwrap().into_any()
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

/// The i-th element of a column as a Python scalar.
fn scalar_to_py(py: Python<'_>, col: &Column, i: usize) -> Py<PyAny> {
    match col {
        Column::F64(v) => v[i].into_pyobject(py).unwrap().into_any().unbind(),
        Column::F32(v) => (v[i] as f64).into_pyobject(py).unwrap().into_any().unbind(),
        // an int/bool missing cell is the volas.NA symbol (pandas tolist semantics)
        Column::I64(v, val) if val.is_valid(i) => v[i].into_pyobject(py).unwrap().into_any().unbind(),
        Column::I32(v, val) if val.is_valid(i) => (v[i] as i64).into_pyobject(py).unwrap().into_any().unbind(),
        Column::Bool(v, val) if val.is_valid(i) => {
            v[i].into_pyobject(py).unwrap().to_owned().into_any().unbind()
        }
        Column::Str(v, val) if val.is_valid(i) => v[i].clone().into_pyobject(py).unwrap().into_any().unbind(),
        Column::I64(..) | Column::I32(..) | Column::Bool(..) | Column::Str(..) => na(py),
        Column::Datetime(v) => py
            .import("numpy")
            .expect("import numpy")
            .call_method1("datetime64", (v[i], "ns"))
            .expect("np.datetime64")
            .into_any()
            .unbind(),
    }
}

/// Cached numpy scalar **type** objects (`np.float64` etc.), so a boundary box is
/// a single call rather than a re-import + attribute lookup per value. Holds only
/// the numeric types that need boxing; `bool` / `str` / `datetime` are handled
/// directly. Indexed by [`DType`] for `O(1)` lookup.
struct NumpyTypes {
    float64: Py<PyAny>,
    float32: Py<PyAny>,
    int64: Py<PyAny>,
    int32: Py<PyAny>,
    bool_: Py<PyAny>,
}
static NUMPY_TYPES: GILOnceCell<NumpyTypes> = GILOnceCell::new();
fn numpy_types(py: Python<'_>) -> &'static NumpyTypes {
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

/// `volas.NA` — the singleton missing-value marker shown to users and returned by
/// element access on a missing int/bool cell. A pure symbol: physical storage
/// stays dtype-optimal (a float keeps `NaN`, an int/bool a validity bit).
#[pyclass(frozen, name = "NAType", module = "volas_rs")]
struct NaType;

#[pymethods]
impl NaType {
    fn __repr__(&self) -> &'static str {
        "<NA>"
    }
    // pandas' NA raises on truthiness; mirror it so `if s[i]:` can't silently
    // treat a missing value as False.
    fn __bool__(&self) -> PyResult<bool> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "boolean value of volas.NA is ambiguous",
        ))
    }
}

static NA_SINGLETON: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

/// The cached `volas.NA` singleton object.
fn na(py: Python<'_>) -> Py<PyAny> {
    NA_SINGLETON
        .get_or_init(py, || Py::new(py, NaType).expect("create volas.NA").into_any())
        .clone_ref(py)
}

/// Box `value` as the numpy scalar `dtype` (the external-boundary representation,
/// e.g. `np.float64(0.0)`); the call narrows it to the target type. Non-numeric
/// dtypes never reach here.
fn numpy_scalar(py: Python<'_>, dtype: DType, value: &Bound<'_, PyAny>) -> Py<PyAny> {
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
fn np_f64(py: Python<'_>, x: f64) -> Py<PyAny> {
    numpy_scalar(py, DType::F64, &x.into_pyobject(py).unwrap())
}
fn np_f32(py: Python<'_>, x: f32) -> Py<PyAny> {
    numpy_scalar(py, DType::F32, &x.into_pyobject(py).unwrap())
}
fn np_i64(py: Python<'_>, x: i64) -> Py<PyAny> {
    numpy_scalar(py, DType::I64, &x.into_pyobject(py).unwrap())
}
fn np_i32(py: Python<'_>, x: i32) -> Py<PyAny> {
    numpy_scalar(py, DType::I32, &x.into_pyobject(py).unwrap())
}
fn np_bool(py: Python<'_>, b: bool) -> Py<PyAny> {
    numpy_scalar(py, DType::Bool, &b.into_pyobject(py).unwrap().to_owned())
}

/// Element `i` as a **numpy** scalar (pandas' direct `s[i]` / `iloc` / `at`
/// semantics), matching the column dtype. Bulk paths (`to_list`, iteration) use
/// [`scalar_to_py`] instead, which yields native Python scalars like pandas.
fn np_scalar_to_py(py: Python<'_>, col: &Column, i: usize) -> Py<PyAny> {
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
fn scalar_to_numpy(py: Python<'_>, s: Scalar) -> Py<PyAny> {
    match s {
        Scalar::F64(x) => np_f64(py, x),
        Scalar::F32(x) => np_f32(py, x),
        Scalar::I64(x) => np_i64(py, x),
        Scalar::I32(x) => np_i32(py, x),
        Scalar::Bool(b) => np_bool(py, b),
    }
}

/// Render an index label at position `i` as a **typed** Python object: a
/// [`Timestamp`](PyTimestamp) for a DatetimeIndex (carrying the frame tz, so
/// `df.loc[row.name]` round-trips on the absolute instant), else the int / str
/// label. Display layers render the readable string form separately.
fn label_to_py(py: Python<'_>, index: &Index, i: usize) -> Py<PyAny> {
    match index.kind() {
        IndexKind::Datetime(v, tz) => Py::new(py, PyTimestamp { ns: v[i], tz: *tz })
            .unwrap()
            .into_any(),
        IndexKind::Int64(v) => v[i].into_pyobject(py).unwrap().into_any().unbind(),
        IndexKind::Range(_) => (i as i64).into_pyobject(py).unwrap().into_any().unbind(),
        IndexKind::Str(v) => v[i].clone().into_pyobject(py).unwrap().into_any().unbind(),
    }
}

/// Parse a Python timestamp to UTC epoch-ns, interpreting a **naive** string in
/// `tz`. A [`PyTimestamp`] carries its own tz and resolves to an absolute instant
/// (so it matches across zones); an offset-aware string is already absolute; an
/// integer is epoch-ns.
pub(crate) fn parse_ts_in_tz(key: &Bound<'_, PyAny>, tz: Tz) -> PyResult<i64> {
    if let Ok(ts) = key.extract::<PyRef<PyTimestamp>>() {
        return Ok(ts.ns);
    }
    if let Ok(s) = key.extract::<String>() {
        return datetime::parse_ns_in_tz(&s, tz)
            .ok_or_else(|| PyKeyError::new_err(format!("invalid datetime label {s:?}")));
    }
    if let Ok(i) = key.extract::<i64>() {
        return Ok(i);
    }
    Err(PyKeyError::new_err(
        "label must be a datetime string or integer",
    ))
}

/// ``volas.Timestamp(value, tz=None)`` — a typed datetime label carrying its own
/// timezone, resolving to an absolute **UTC** instant.
///
/// Use it for precise / cross-tz ``.loc`` / ``.loc[a:b]`` / ``.at`` lookups: a
/// Timestamp built in one zone matches the right row of a frame displayed in
/// another, because both compare on the UTC axis. (A bare string label is
/// instead interpreted in the index's own tz.)
///
/// Args:
///     value (str | int): a datetime string (e.g. ``'2021-01-04 09:30'``) or
///         epoch nanoseconds. A naive string is interpreted in ``tz``.
///     tz (str, optional): the zone the value is given in, e.g.
///         ``'America/New_York'`` or ``'+08:00'`` (default UTC).
///
/// Usage::
///
///     ts = volas.Timestamp('2021-01-04 09:30', tz='America/New_York')
///     df.at[ts, 'close']    # matches the right row across timezones
#[pyclass(name = "Timestamp")]
pub struct PyTimestamp {
    /// UTC epoch-ns (the absolute instant).
    ns: i64,
    /// The zone `value` was specified in (for display).
    tz: Tz,
}

#[pymethods]
impl PyTimestamp {
    // Constructor — args & usage live in the class docstring (pyo3 does not
    // surface a `#[new]` doc comment to Python).
    #[new]
    #[pyo3(signature = (value, tz = None))]
    fn new(value: &Bound<'_, PyAny>, tz: Option<String>) -> PyResult<Self> {
        let tzv = match tz {
            Some(s) => Tz::parse(&s).map_err(pyerr)?,
            None => Tz::Utc,
        };
        Ok(PyTimestamp {
            ns: parse_ts_in_tz(value, tzv)?,
            tz: tzv,
        })
    }

    /// The absolute instant as UTC epoch nanoseconds.
    #[getter]
    fn value(&self) -> i64 {
        self.ns
    }

    /// The timezone name, or `None` if UTC / unspecified.
    #[getter]
    fn tz(&self) -> Option<String> {
        match self.tz {
            Tz::Utc => None,
            other => Some(other.name()),
        }
    }

    /// The wall-clock as a NumPy `datetime64[ns]` (UTC instant).
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let arr = vec![self.ns].into_pyarray(py);
        Ok(arr.call_method1("astype", ("datetime64[ns]",))?)
    }

    /// Calendar year in the timestamp's timezone (pandas `Timestamp.year`).
    #[getter]
    fn year(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).0
    }
    /// Calendar month, 1..=12 (pandas `Timestamp.month`).
    #[getter]
    fn month(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).1
    }
    /// Day of month, 1..=31 (pandas `Timestamp.day`).
    #[getter]
    fn day(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).2
    }
    /// Hour, 0..=23 (pandas `Timestamp.hour`).
    #[getter]
    fn hour(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).3
    }
    /// Minute, 0..=59 (pandas `Timestamp.minute`).
    #[getter]
    fn minute(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).4
    }
    /// Second, 0..=59 (pandas `Timestamp.second`).
    #[getter]
    fn second(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).5
    }

    /// Day of week with Monday=0 .. Sunday=6 (pandas `Timestamp.weekday()`).
    fn weekday(&self) -> i64 {
        let (y, mo, d, ..) = datetime::civil_parts_tz(self.ns, self.tz);
        (datetime::days_from_civil(y, mo, d) + 3).rem_euclid(7)
    }

    /// Format the wall-clock time with a `strftime` format string (pandas
    /// `Timestamp.strftime`). Raises `ValueError` on an invalid format.
    fn strftime(&self, fmt: &str) -> PyResult<String> {
        datetime::strftime(self.ns, self.tz, fmt)
            .ok_or_else(|| PyValueError::new_err("invalid strftime format string"))
    }

    fn __repr__(&self) -> String {
        match self.tz {
            Tz::Utc => format!("Timestamp('{}')", datetime::format_ns(self.ns)),
            other => format!(
                "Timestamp('{}', tz='{}')",
                datetime::format_ns_tz(self.ns, other),
                other.name()
            ),
        }
    }

    /// The readable wall-clock string (pandas `str(Timestamp)` form) — the
    /// object form stays in `repr`.
    fn __str__(&self) -> String {
        datetime::format_ns_tz(self.ns, self.tz)
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: pyo3::basic::CompareOp) -> PyResult<bool> {
        let rhs = parse_ts_in_tz(other, Tz::Utc)?;
        Ok(op.matches(self.ns.cmp(&rhs)))
    }

    fn __hash__(&self) -> i64 {
        self.ns
    }
}

/// Parse a Python timestamp (datetime string or epoch-ns integer) to UTC ns,
/// interpreting a naive string as UTC.
pub(crate) fn parse_ts(key: &Bound<'_, PyAny>) -> PyResult<i64> {
    parse_ts_in_tz(key, Tz::Utc)
}

/// Parse a Python label to the [`Label`] kind expected by `index`: a string for
/// a string index, a parsed datetime (in the index's tz) / integer for the
/// numeric kinds.
pub(crate) fn parse_label(key: &Bound<'_, PyAny>, index: &Index) -> PyResult<Label> {
    match index.kind() {
        IndexKind::Str(_) => key
            .extract::<String>()
            .map(Label::Str)
            .map_err(|_| PyKeyError::new_err("label must be a string for a string index")),
        IndexKind::Datetime(_, tz) => parse_ts_in_tz(key, *tz).map(Label::I64),
        _ => parse_ts(key).map(Label::I64),
    }
}

/// Build the `.index` as a NumPy array. A DatetimeIndex exports its **UTC**
/// instants as `datetime64[ns]` (matching pandas `.values`; the frame tz governs
/// string rendering / matching, not the numeric export); a string index becomes
/// an object array.
fn index_to_numpy<'py>(py: Python<'py>, index: &Index) -> PyResult<Bound<'py, PyAny>> {
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
        None => Tz::Utc,
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

/// Raise if the frame has stale computed columns after an `append`. The per-column
/// `df[directive]` access auto-refreshes; bulk / positional reads (`to_numpy`,
/// `.iloc` / `.loc` / `.at` / `.iat`) do not, so they must be fresh — call
/// `fulfill()` first. Keeps the read path O(1) and never returns silent NaN.
fn ensure_fresh(df: &DataFrame) -> PyResult<()> {
    if df.has_stale_computed() {
        Err(PyValueError::new_err(
            "frame has stale computed (directive) columns after append; \
             call fulfill() before a bulk or positional read",
        ))
    } else {
        Ok(())
    }
}

/// Parse an optional printf-style `float_format` spec (shared by `to_csv` /
/// `to_string`), raising a `ValueError` on an unsupported form.
fn parse_ff(float_format: Option<&str>) -> PyResult<Option<(Option<usize>, char)>> {
    match float_format {
        Some(f) => Ok(Some(parse_float_format(f).ok_or_else(|| {
            PyValueError::new_err(format!("unsupported float_format \"{f}\""))
        })?)),
        None => Ok(None),
    }
}

// --- Series ----------------------------------------------------------------

/// ``volas.Series`` — a single named, indexed column (usually obtained from
/// ``df['col']`` or a directive like ``df['ma:5']``).
///
/// Supports NaN-skipping reductions (``mean`` / ``sum`` / ``min`` / ``max`` /
/// ``std`` / ``var`` / ``median``), element-wise arithmetic / comparison /
/// boolean operators (``+ - * /``, ``< <= == != >= >``, ``& | ^ ~``) against a
/// scalar or another equal-length Series, the TA-Lib math transforms
/// (``sin`` / ``sqrt`` / ``ln`` / …), and ``shift`` / ``diff`` / ``fillna`` /
/// ``isna`` / ``notna`` / ``dropna``. Index by position via ``s.iloc[...]`` or
/// label via ``s.loc[...]``; export with ``to_numpy`` / ``to_list``.
///
/// Usage::
///
///     close = df['close']
///     close.mean()            # NaN-skipping mean
///     (close - df['open'])    # element-wise difference
///     close.shift(1)          # lag by one bar
///     close.iloc[-1]          # last value
#[pyclass(name = "Series")]
pub struct PySeries {
    pub(crate) inner: Series,
}

#[pymethods]
impl PySeries {
    /// The series name — the column it was drawn from, or ``None``.
    ///
    /// Returns:
    ///     str | None
    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }

    /// The dtype name (``'float64'``, ``'bool'``, ``'int64'``, ``'object'`` or
    /// ``'datetime64[ns]'``).
    ///
    /// Returns:
    ///     str
    #[getter]
    fn dtype(&self) -> String {
        self.inner.dtype().to_string()
    }

    /// The row index shared with the parent frame, as a NumPy array (a
    /// ``datetime64[ns]`` array for a DatetimeIndex, an object array for a string
    /// index).
    ///
    /// Returns:
    ///     numpy.ndarray
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        index_to_numpy(py, &self.inner.index)
    }

    /// The DatetimeIndex timezone name, or `None` for a tz-naive / non-datetime
    /// index (mirrors `df.tz`).
    #[getter]
    fn tz(&self) -> Option<String> {
        match self.inner.index.tz() {
            Tz::Utc => None,
            other => Some(other.name()),
        }
    }

    /// Positional (integer-location) accessor: ``s.iloc[i]`` returns the i-th
    /// value (negative indices count from the end); ``s.iloc[a:b]`` returns a
    /// sub-series. Read-only.
    ///
    /// Usage::
    ///
    ///     s.iloc[0]      # first value
    ///     s.iloc[-1]     # last value
    ///     s.iloc[1:4]    # a sub-series
    #[getter]
    fn iloc(&self) -> SeriesILoc {
        SeriesILoc {
            inner: self.inner.clone(),
        }
    }

    /// Label-based accessor: ``s.loc[label]`` returns the value at an index
    /// label; ``s.loc[a:b]`` returns the (stop-inclusive) label slice. Read-only.
    ///
    /// Usage::
    ///
    ///     s.loc[20210104]              # by integer label
    ///     s.loc['2021-01-04':'2021-02-01']  # inclusive datetime slice
    #[getter]
    fn loc(&self) -> SeriesLoc {
        SeriesLoc {
            inner: self.inner.clone(),
        }
    }

    /// The shape as a 1-tuple `(len,)` (pandas `Series.shape`).
    #[getter]
    fn shape(&self) -> (usize,) {
        (self.inner.len(),)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Guard the ambiguous `if series:` footgun: a Series has a single truth value
    /// only when it holds exactly one element (pandas-style).
    fn __bool__(&self) -> PyResult<bool> {
        match self.inner.len() {
            1 => Ok(to_bool_vec(&self.inner.data)[0]),
            _ => Err(PyValueError::new_err(
                "The truth value of a Series is ambiguous — use s.any() or s.all()",
            )),
        }
    }

    /// The values as a typed NumPy array; `dtype` casts (e.g. `'float32'`).
    #[pyo3(signature = (dtype = None))]
    fn to_numpy<'py>(&self, py: Python<'py>, dtype: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        let arr = column_to_numpy(py, &self.inner.data);
        match dtype {
            Some(dt) => Ok(arr.call_method1("astype", (dt,))?),
            None => Ok(arr),
        }
    }

    /// NumPy array protocol, so `np.isnan(series)` etc. work directly. Honors a
    /// requested `dtype` (casts).
    #[pyo3(signature = (dtype = None, copy = None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<PyObject>,
        copy: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = copy;
        let arr = column_to_numpy(py, &self.inner.data);
        match dtype {
            Some(dt) => Ok(arr.call_method1("astype", (dt,))?),
            None => Ok(arr),
        }
    }

    // Reductions return numpy scalars (pandas' boundary representation). The
    // dtype-preserving ones (sum/prod/min/max) carry the column's result dtype
    // (np.int64 for an int column, etc.); the always-float statistics box np.float64.

    // Each numeric reduction first asserts the column is numeric — a str/datetime
    // reduction used to funnel through to_f64_vec and silently return 0.0 / NaN,
    // which the API contract (C4) forbids (V3).

    /// NaN-skipping mean (pandas `mean`) -> `np.float64`.
    fn mean(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, self.mean_f64()))
    }
    /// Sum (pandas `sum`), dtype-preserving: float -> `np.float64`, int / bool ->
    /// `np.int64` (computed natively).
    fn sum(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(scalar_to_numpy(py, self.inner.data.sum()))
    }
    /// Product (pandas `prod`), dtype-preserving.
    fn prod(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(scalar_to_numpy(py, self.inner.data.prod()))
    }
    /// Minimum (pandas `min`), dtype-preserving (int -> `np.int64`, etc.).
    fn min(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(scalar_to_numpy(py, self.inner.data.extreme(false)))
    }
    /// Maximum (pandas `max`), dtype-preserving.
    fn max(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(scalar_to_numpy(py, self.inner.data.extreme(true)))
    }
    /// Sample variance (`ddof=1`, pandas `var`) -> `np.float64`.
    fn var(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, self.var_f64()))
    }
    /// Sample standard deviation (`ddof=1`, pandas `std`) -> `np.float64`.
    fn std(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, self.var_f64().sqrt()))
    }
    /// Median (pandas `median`) -> `np.float64`.
    fn median(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, self.median_f64()))
    }
    /// Standard error of the mean (`ddof=1`, pandas `sem`) -> `np.float64`.
    fn sem(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, stats::sem(&self.inner.data.to_f64_vec())))
    }
    /// Adjusted Fisher-Pearson skewness (pandas `skew`) -> `np.float64`.
    fn skew(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, stats::skew(&self.inner.data.to_f64_vec())))
    }
    /// Excess kurtosis, Fisher's definition (pandas `kurt`) -> `np.float64`.
    fn kurt(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, stats::kurt(&self.inner.data.to_f64_vec())))
    }

    /// Pairwise Pearson correlation with `other` (pandas `corr`); positional
    /// alignment (volas does not reindex), dropping NaN pairs.
    fn corr(&self, other: &PySeries) -> PyResult<f64> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        other.inner.data.require_numeric().map_err(pyerr)?;
        Ok(stats::corr(&self.inner.data.to_f64_vec(), &other.inner.data.to_f64_vec()))
    }

    /// Pairwise sample covariance with `other`, ddof=1 (pandas `cov`); positional
    /// alignment, dropping NaN pairs.
    fn cov(&self, other: &PySeries) -> PyResult<f64> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        other.inner.data.require_numeric().map_err(pyerr)?;
        Ok(stats::cov(&self.inner.data.to_f64_vec(), &other.inner.data.to_f64_vec()))
    }

    /// Summary statistics (pandas `describe`): a Series indexed by
    /// `count / mean / std / min / 25% / 50% / 75% / max`.
    fn describe(&self) -> PyResult<PySeries> {
        // describe is a numeric summary (mean/std/quantiles); a str/datetime column
        // would funnel through to_f64_vec to nonsense, so it raises (C4) until a
        // dtype-aware categorical/datetime describe is designed.
        self.inner.data.require_numeric().map_err(pyerr)?;
        let v = self.inner.data.to_f64_vec();
        let count = non_nan(&self.inner.data).len() as f64;
        let vals = vec![
            count,
            self.mean_f64(),
            self.var_f64().sqrt(),
            stats::extreme(&v, false).unwrap_or(f64::NAN),
            self.quantile_f64(0.25)?,
            self.quantile_f64(0.5)?,
            self.quantile_f64(0.75)?,
            stats::extreme(&v, true).unwrap_or(f64::NAN),
        ];
        let labels = describe_labels();
        Ok(PySeries {
            inner: Series::new(
                self.inner.name.clone(),
                Column::f64(vals),
                Arc::new(Index::str(labels)),
            ),
        })
    }

    /// Number of non-missing values (pandas `count`) -> `int`.
    fn count(&self) -> usize {
        self.inner.data.count()
    }

    /// Number of distinct non-missing values (pandas `nunique`) -> `int`.
    fn nunique(&self) -> usize {
        self.inner.data.nunique()
    }

    /// The distinct values in order of first appearance (pandas `unique`), as a
    /// **`Series`** that preserves the dtype and `volas.NA` (API contract C1: a
    /// variable-length column result stays a `Series`, not a numpy array that would
    /// collapse a nullable int/bool to float64 + NaN). One missing slot is kept if
    /// the series has any NA; the result carries a fresh `RangeIndex` (the distinct
    /// values have no row correspondence to the original).
    fn unique(&self) -> PySeries {
        let idx = self.inner.data.unique_indices();
        PySeries {
            inner: Series::new(
                self.inner.name.clone(),
                self.inner.data.take(&idx),
                Arc::new(Index::range(idx.len())),
            ),
        }
    }

    /// Sort by value (pandas `sort_values`), stable, with missing values last; the
    /// index follows the permutation.
    #[pyo3(signature = (ascending = true))]
    fn sort_values(&self, ascending: bool) -> PySeries {
        self.reindexed(&self.inner.data.argsort(ascending))
    }

    /// First `n` rows (pandas `head`).
    #[pyo3(signature = (n = 5))]
    fn head(&self, n: usize) -> PySeries {
        self.sliced(0, n.min(self.inner.len()))
    }

    /// Last `n` rows (pandas `tail`).
    #[pyo3(signature = (n = 5))]
    fn tail(&self, n: usize) -> PySeries {
        let len = self.inner.len();
        self.sliced(len.saturating_sub(n), len)
    }

    /// True if any element is truthy (NaN skipped) — pandas `any` -> `np.bool_`.
    fn any(&self, py: Python<'_>) -> Py<PyAny> {
        let r = match &self.inner.data {
            // skipna: a NA bool is its `false` placeholder in the buffer, so read the
            // validity — only a *present* true counts (matching pandas nullable any).
            Column::Bool(v, val) => v.iter().enumerate().any(|(i, &b)| val.is_valid(i) && b),
            other => other.to_f64_vec().iter().any(|&x| !x.is_nan() && x != 0.0),
        };
        np_bool(py, r)
    }

    /// True if every non-missing element is truthy (empty / all-NA -> True) — pandas
    /// `all` -> `np.bool_`, default `skipna=True`.
    fn all(&self, py: Python<'_>) -> Py<PyAny> {
        let r = match &self.inner.data {
            // skipna: a NA is ignored (vacuously satisfies), only a present false fails.
            Column::Bool(v, val) => v.iter().enumerate().all(|(i, &b)| !val.is_valid(i) || b),
            other => other.to_f64_vec().iter().all(|&x| x.is_nan() || x != 0.0),
        };
        np_bool(py, r)
    }

    /// The values as a Python list of typed scalars (pandas `to_list`).
    fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Py<PyAny>> = (0..self.inner.len())
            .map(|i| scalar_to_py(py, &self.inner.data, i))
            .collect();
        PyList::new(py, items)
    }

    /// Alias of [`to_list`](Self::to_list) (the numpy / pandas spelling).
    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.to_list(py)
    }

    /// Shift the values by ``n`` rows, padding vacated cells with NaN.
    ///
    /// Args:
    ///     n (int): rows to shift; positive shifts down (default 1), negative up.
    ///
    /// Usage::
    ///
    ///     s.shift(1)    # lag by one bar
    ///     s.shift(-1)   # lead by one bar
    ///
    /// Returns:
    ///     Series: a new series of the same length.
    #[pyo3(signature = (n = 1))]
    fn shift(&self, n: isize) -> PySeries {
        col_to_series(&self.inner, self.inner.data.shift(n))
    }

    /// Discrete difference ``x[i] - x[i-n]`` (equivalent to ``s - s.shift(n)``).
    ///
    /// Args:
    ///     n (int): periods to difference; the first ``n`` rows are NaN
    ///         (default 1). Negative ``n`` differences against later rows.
    ///
    /// Returns:
    ///     Series: a new series of the same length.
    #[pyo3(signature = (n = 1))]
    fn diff(&self, n: isize) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.diff(n).map_err(pyerr)?))
    }

    /// Replace a missing cell with `value` (pandas `fillna`). Fills the numeric
    /// family (float / int / bool, promoting the dtype only when the fill needs
    /// it); a non-numeric `str` / `datetime` column with a missing cell raises a
    /// `TypeError` (volas has no `object` dtype to hold a mixed column). For
    /// directional fill use `ffill` / `bfill` (pandas 3.0 removed `fillna(method=)`).
    fn fillna(&self, value: f64) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.fillna(value).map_err(pyerr)?))
    }

    /// Forward-fill NaN cells from the last valid value (pandas `ffill`).
    fn ffill(&self) -> PySeries {
        self.fill_dir(true)
    }

    /// Backward-fill NaN cells from the next valid value (pandas `bfill`).
    fn bfill(&self) -> PySeries {
        self.fill_dir(false)
    }

    /// Boolean mask of missing (`volas.NA`) values, across every dtype (a float
    /// `NaN`, an int/bool validity hole, a datetime `NaT`).
    fn isna(&self) -> PySeries {
        let c = &self.inner.data;
        bool_series(&self.inner, (0..c.len()).map(|i| !c.is_valid(i)).collect())
    }

    /// Boolean mask of present (non-missing) values.
    fn notna(&self) -> PySeries {
        let c = &self.inner.data;
        bool_series(&self.inner, (0..c.len()).map(|i| c.is_valid(i)).collect())
    }

    /// Drop missing (NaN) elements (carries their index labels with them).
    fn dropna(&self) -> PySeries {
        let c = &self.inner.data;
        let keep: Vec<usize> = (0..c.len()).filter(|&i| c.is_valid(i)).collect();
        let data = self.inner.data.take(&keep);
        let index = Arc::new(self.inner.index.take(&keep));
        PySeries {
            inner: Series::new(self.inner.name.clone(), data, index),
        }
    }

    /// pandas-style equality: **same dtype** and value-equal (NaN equals NaN).
    fn equals(&self, other: &PySeries) -> bool {
        self.inner.data.dtype() == other.inner.data.dtype()
            && self.inner.data.equals(&other.inner.data)
    }

    /// Cast to a dtype (`'float64'` / `'int64'` / `'bool'` / `'str'` /
    /// `'datetime64[ns]'` / ...), pandas `astype`.
    fn astype(&self, dtype: &str) -> PyResult<PySeries> {
        let col = if let Some(unit) = datetime_unit_of(dtype) {
            match &self.inner.data {
                Column::Datetime(_) | Column::Str(_, _) => {
                    self.inner.data.to_datetime().map_err(pyerr)?
                }
                _ => self.inner.data.epoch_to_datetime(unit).map_err(pyerr)?,
            }
        } else {
            self.inner
                .data
                .cast(parse_dtype(dtype)?)
                .map_err(pyerr)?
        };
        Ok(PySeries {
            inner: Series::new(self.inner.name.clone(), col, Arc::clone(&self.inner.index)),
        })
    }

    /// Cumulative sum (pandas `cumsum`, skipna=True), dtype-preserving.
    fn cumsum(&self) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.cumsum().map_err(pyerr)?))
    }

    /// Cumulative maximum (pandas `cummax`, skipna=True), dtype-preserving.
    fn cummax(&self) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.cummax().map_err(pyerr)?))
    }

    /// Cumulative minimum (pandas `cummin`, skipna=True), dtype-preserving.
    fn cummin(&self) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.cummin().map_err(pyerr)?))
    }

    /// Cumulative product (pandas `cumprod`, skipna=True), dtype-preserving.
    fn cumprod(&self) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.cumprod().map_err(pyerr)?))
    }

    /// Round each value to `decimals` places (pandas `round`), dtype-preserving:
    /// banker's (half-to-even) for floats, integer-exact for ints; NaN stays NaN.
    #[pyo3(signature = (decimals = 0))]
    fn round(&self, decimals: i32) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.round(decimals).map_err(pyerr)?))
    }

    /// Numerical rank (pandas `rank`, 1-based, NaN kept as NaN). Ties resolve by
    /// `method` (`'average'` | `'min'` | `'max'` | `'first'` | `'dense'`); `pct`
    /// returns ranks scaled to (0, 1].
    #[pyo3(signature = (method = "average", ascending = true, pct = false))]
    fn rank(&self, method: &str, ascending: bool, pct: bool) -> PyResult<PySeries> {
        self.inner.data.require_numeric().map_err(pyerr)?; // str/datetime rank -> error (C4)
        let m = match method {
            "average" => stats::RankMethod::Average,
            "min" => stats::RankMethod::Min,
            "max" => stats::RankMethod::Max,
            "first" => stats::RankMethod::First,
            "dense" => stats::RankMethod::Dense,
            other => return Err(PyValueError::new_err(format!("rank: unknown method '{other}'"))),
        };
        Ok(f64_series(
            &self.inner,
            stats::rank(&self.inner.data.to_f64_vec(), m, ascending, pct),
        ))
    }

    /// Element-wise absolute value (pandas `abs`), dtype-preserving.
    fn abs(&self) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.abs().map_err(pyerr)?))
    }

    /// Clip values into `[lower, upper]` (either bound optional), dtype-preserving;
    /// NaN stays NaN. An int column with a non-integral bound promotes to float
    /// (pandas `clip`).
    #[pyo3(signature = (lower = None, upper = None))]
    fn clip(&self, lower: Option<f64>, upper: Option<f64>) -> PyResult<PySeries> {
        Ok(col_to_series(&self.inner, self.inner.data.clip(lower, upper).map_err(pyerr)?))
    }

    /// The `q`-quantile in `[0, 1]` (linear interpolation, NaN-skipping) — pandas
    /// `quantile` -> `np.float64`.
    #[pyo3(signature = (q = 0.5))]
    fn quantile(&self, py: Python<'_>, q: f64) -> PyResult<Py<PyAny>> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        Ok(self.box_float(py, self.quantile_f64(q)?))
    }

    /// The index **label** of the maximum value (NaN-skipping); raises on an
    /// all-NA series (pandas `idxmax`).
    fn idxmax(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(label_to_py(
            py,
            &self.inner.index,
            argext(&self.inner.data, true)?,
        ))
    }

    /// The index **label** of the minimum value (NaN-skipping); raises on an
    /// all-NA series (pandas `idxmin`).
    fn idxmin(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(label_to_py(
            py,
            &self.inner.index,
            argext(&self.inner.data, false)?,
        ))
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Add, false)
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Sub, false)
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Mul, false)
    }
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_div(&self.inner, other, false)
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Add, true)
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Sub, true)
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_arith(&self.inner, other, BinOp::Mul, true)
    }
    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_div(&self.inner, other, true)
    }
    fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_floordiv(&self.inner, other, false)
    }
    fn __rfloordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_floordiv(&self.inner, other, true)
    }

    // Element-wise comparisons -> bool Series (pandas-style), dtype-aware.
    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Lt)
    }
    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Le)
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Eq)
    }
    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Ne)
    }
    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Ge)
    }
    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, CmpOp::Gt)
    }

    // Element-wise boolean logic -> bool Series (operands coerced to bool).
    fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::And)
    }
    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::Or)
    }
    fn __xor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::Xor)
    }
    fn __rand__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::And)
    }
    fn __ror__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::Or)
    }
    fn __rxor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, BoolOp::Xor)
    }
    fn __invert__(&self) -> PySeries {
        col_to_series(&self.inner, self.inner.data.not())
    }

    /// `series[key]`: an integer position, a datetime label, or a slice.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // boolean mask -> the True rows, as a new Series (pandas `s[bool_mask]`)
        if let Some(mask) = bool_mask_key(key)? {
            let sub = self.inner.filter_mask(&mask).map_err(pyerr)?;
            return Ok(Py::new(py, PySeries { inner: sub })?.into_any());
        }
        if let Ok(i) = key.extract::<isize>() {
            let i = norm_idx(i, self.inner.len())?;
            return Ok(np_scalar_to_py(py, &self.inner.data, i));
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            return Ok(Py::new(py, slice_series(&self.inner, slice)?)?.into_any());
        }
        // label lookup
        let label = parse_label(key, &self.inner.index)?;
        let pos = self
            .inner
            .index
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        Ok(np_scalar_to_py(py, &self.inner.data, pos))
    }

    /// In-place assignment by boolean mask (`s[mask] = v`) or integer position
    /// (`s[i] = v`). Follows pandas 3.0 dtype rules: the column dtype is kept when
    /// the value fits (an integral number stays in an int series), `NaN` upcasts
    /// an int series to float, and a lossy write (e.g. `2.5` into an int series)
    /// raises `TypeError`.
    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = self.inner.len();
        let positions: Vec<usize> = if let Some(mask) = bool_mask_key(key)? {
            if mask.len() != n {
                return Err(PyValueError::new_err(format!(
                    "boolean mask length {} != series length {n}",
                    mask.len()
                )));
            }
            mask.iter()
                .enumerate()
                .filter_map(|(i, &m)| m.then_some(i))
                .collect()
        } else if let Ok(i) = key.extract::<isize>() {
            vec![norm_idx(i, n)?]
        } else {
            return Err(PyTypeError::new_err(
                "Series assignment takes a boolean mask or an integer position",
            ));
        };
        // One assignment path for every value kind (number, bool, string, datetime
        // string, None/NaN): convert to a typed single-cell column for this dtype
        // and scatter it — identical rules to the DataFrame indexers and mask
        // assignment (keep dtype, update validity, lossy values error).
        self.inner.data = scatter_scalar(&self.inner.data, &positions, value)?;
        Ok(())
    }

    /// pandas `Series.where`: keep self where `cond` is True, else `other`
    /// (default NaN). `cond` is a boolean Series; `other` is a scalar or a
    /// (same-index) Series.
    #[pyo3(name = "where", signature = (cond, other = None))]
    fn where_(&self, cond: &PySeries, other: Option<&Bound<'_, PyAny>>) -> PyResult<PySeries> {
        self.select_where(cond, other, false)
    }

    /// pandas `Series.mask`: the inverse of `where` — replace with `other` where
    /// `cond` is True, keep self elsewhere.
    #[pyo3(signature = (cond, other = None))]
    fn mask(&self, cond: &PySeries, other: Option<&Bound<'_, PyAny>>) -> PyResult<PySeries> {
        self.select_where(cond, other, true)
    }

    /// pandas-style vertical repr (`label   value` rows + a
    /// `Name: <name>, dtype: <dtype>` footer), truncating to 5 head + 5 tail rows
    /// past 60 (`display.max_rows` / `min_rows`). `str` and `repr` are identical.
    fn __repr__(&self) -> String {
        let truncate = if self.inner.len() > 60 { Some(5) } else { None };
        render_series(&self.inner, NA_REPR, None, truncate, true)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Render the whole series as text (pandas `Series.to_string`): no truncation
    /// by default and no `Name/dtype` footer; `max_rows` truncates.
    #[pyo3(signature = (na_rep = NA_REPR, float_format = None, max_rows = None))]
    fn to_string(
        &self,
        na_rep: &str,
        float_format: Option<&str>,
        max_rows: Option<usize>,
    ) -> PyResult<String> {
        let ff = parse_ff(float_format)?;
        let truncate = match max_rows {
            Some(m) if self.inner.len() > m => Some((m / 2).max(1)),
            _ => None,
        };
        Ok(render_series(&self.inner, na_rep, ff, truncate, false))
    }

    // --- TA-Lib "Math Transform" group: element-wise, NaN-preserving (a NaN or an
    // out-of-domain input — e.g. sqrt of a negative, asin outside [-1, 1] — yields
    // NaN, matching TA-Lib). Implemented as Series methods, not directives.
    /// Element-wise arc cosine (TA-Lib ACOS).
    fn acos(&self) -> PyResult<PySeries> {
        self.map_f64(f64::acos)
    }
    /// Element-wise arc sine (TA-Lib ASIN).
    fn asin(&self) -> PyResult<PySeries> {
        self.map_f64(f64::asin)
    }
    /// Element-wise arc tangent (TA-Lib ATAN).
    fn atan(&self) -> PyResult<PySeries> {
        self.map_f64(f64::atan)
    }
    /// Element-wise ceiling (TA-Lib CEIL).
    fn ceil(&self) -> PyResult<PySeries> {
        self.map_f64(f64::ceil)
    }
    /// Element-wise cosine (TA-Lib COS).
    fn cos(&self) -> PyResult<PySeries> {
        self.map_f64(f64::cos)
    }
    /// Element-wise hyperbolic cosine (TA-Lib COSH).
    fn cosh(&self) -> PyResult<PySeries> {
        self.map_f64(f64::cosh)
    }
    /// Element-wise base-e exponential (TA-Lib EXP).
    fn exp(&self) -> PyResult<PySeries> {
        self.map_f64(f64::exp)
    }
    /// Element-wise floor (TA-Lib FLOOR).
    fn floor(&self) -> PyResult<PySeries> {
        self.map_f64(f64::floor)
    }
    /// Element-wise natural logarithm (TA-Lib LN).
    fn ln(&self) -> PyResult<PySeries> {
        self.map_f64(f64::ln)
    }
    /// Element-wise base-10 logarithm (TA-Lib LOG10).
    fn log10(&self) -> PyResult<PySeries> {
        self.map_f64(f64::log10)
    }
    /// Element-wise sine (TA-Lib SIN).
    fn sin(&self) -> PyResult<PySeries> {
        self.map_f64(f64::sin)
    }
    /// Element-wise hyperbolic sine (TA-Lib SINH).
    fn sinh(&self) -> PyResult<PySeries> {
        self.map_f64(f64::sinh)
    }
    /// Element-wise square root (TA-Lib SQRT).
    fn sqrt(&self) -> PyResult<PySeries> {
        self.map_f64(f64::sqrt)
    }
    /// Element-wise tangent (TA-Lib TAN).
    fn tan(&self) -> PyResult<PySeries> {
        self.map_f64(f64::tan)
    }
    /// Element-wise hyperbolic tangent (TA-Lib TANH).
    fn tanh(&self) -> PyResult<PySeries> {
        self.map_f64(f64::tanh)
    }
}

impl PySeries {
    /// A new Series whose data and index are gathered by `idx` (backs
    /// `sort_values` and any fancy-index reorder).
    fn reindexed(&self, idx: &[usize]) -> PySeries {
        PySeries {
            inner: Series::new(
                self.inner.name.clone(),
                self.inner.data.take(idx),
                Arc::new(self.inner.index.take(idx)),
            ),
        }
    }

    /// A new Series with the `[start, end)` row slice of both data and index
    /// (backs `head` / `tail`).
    fn sliced(&self, start: usize, end: usize) -> PySeries {
        PySeries {
            inner: Series::new(
                self.inner.name.clone(),
                self.inner.data.slice(start, end),
                Arc::new(self.inner.index.slice(start, end)),
            ),
        }
    }

    /// Box a float statistic as the column's float dtype: `np.float32` for an f32
    /// column, else `np.float64` (pandas: `f32.mean() -> np.float32`).
    fn box_float(&self, py: Python<'_>, value: f64) -> Py<PyAny> {
        if self.inner.data.dtype() == DType::F32 {
            np_f32(py, value as f32)
        } else {
            np_f64(py, value)
        }
    }

    // Raw f64 reduction values (the public methods box these as numpy scalars;
    // `describe` reuses them, so they stay unboxed here).
    fn mean_f64(&self) -> f64 {
        let v = non_nan(&self.inner.data);
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    }
    fn var_f64(&self) -> f64 {
        let v = non_nan(&self.inner.data);
        let n = v.len();
        if n < 2 {
            return f64::NAN;
        }
        let mean = v.iter().sum::<f64>() / n as f64;
        v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64
    }
    fn median_f64(&self) -> f64 {
        let mut v = non_nan(&self.inner.data);
        let n = v.len();
        if n == 0 {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        }
    }
    /// The `q`-quantile as a raw f64 (linear interpolation, NaN-skipping).
    fn quantile_f64(&self, q: f64) -> PyResult<f64> {
        if !(0.0..=1.0).contains(&q) {
            return Err(PyValueError::new_err("quantile: q must be in [0, 1]"));
        }
        let mut v = non_nan(&self.inner.data);
        if v.is_empty() {
            return Ok(f64::NAN);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pos = q * (v.len() - 1) as f64;
        let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
        Ok(v[lo] + (v[hi] - v[lo]) * (pos - lo as f64))
    }

    /// The boolean mask for a `where` / `mask` condition, validating it matches
    /// this series' length (pandas requires equal-shape conditionals).
    fn cond_mask(&self, cond: &PySeries) -> PyResult<Vec<bool>> {
        if !matches!(cond.inner.data, Column::Bool(..)) {
            return Err(PyTypeError::new_err(format!(
                "where/mask: `cond` must be a boolean Series, got {}",
                cond.inner.data.dtype()
            )));
        }
        let c = to_bool_vec(&cond.inner.data);
        if c.len() != self.inner.len() {
            return Err(PyValueError::new_err(format!(
                "Array conditional must be same shape as self ({} != {})",
                c.len(),
                self.inner.len()
            )));
        }
        Ok(c)
    }

    /// `where` (`invert = false`) / `mask` (`invert = true`) shared core: pick
    /// `self` where the (possibly inverted) condition holds, else `other`, in the
    /// promoted dtype. `mask` is `where(!cond)`.
    fn select_where(
        &self,
        cond: &PySeries,
        other: Option<&Bound<'_, PyAny>>,
        invert: bool,
    ) -> PyResult<PySeries> {
        let mut c = self.cond_mask(cond)?;
        if invert {
            c.iter_mut().for_each(|b| *b = !*b);
        }
        let (other_col, other_dt) = where_other_resolve(other, &self.inner)?;
        // A float column keeps its float dtype (it absorbs any fill); a same-dtype
        // `other` (incl. the default NA, and bool/str/datetime) keeps that dtype so
        // it is not funneled to f64; a mixed int/float promotes by the supertype.
        let self_dt = self.inner.data.dtype();
        let target = if self_dt.is_float() {
            self_dt
        } else if self_dt == other_dt {
            self_dt
        } else {
            binary_supertype(self_dt, other_dt)
        };
        Ok(col_to_series(
            &self.inner,
            self.inner.data.select(&c, &other_col, target).map_err(pyerr)?,
        ))
    }

    /// Apply an element-wise `f64 -> f64` map, preserving name and index. The single
    /// guard for the whole Math Transform family: a `str` / `datetime` column would
    /// funnel through `to_f64_vec` to silent `NaN` (str) or `sin(epoch-as-f64)`
    /// (datetime), which the contract (C4) forbids, so it raises here.
    fn map_f64(&self, f: impl Fn(f64) -> f64) -> PyResult<PySeries> {
        self.inner.data.require_numeric().map_err(pyerr)?;
        let data = Column::f64(self.inner.data.to_f64_vec().iter().map(|&x| f(x)).collect());
        Ok(PySeries {
            inner: Series::new(self.inner.name.clone(), data, Arc::clone(&self.inner.index)),
        })
    }

    /// Directional fill (`forward` = ffill, else bfill), dtype-aware over every
    /// dtype (int / bool / str / datetime / float). Shared by `ffill` / `bfill`.
    fn fill_dir(&self, forward: bool) -> PySeries {
        col_to_series(&self.inner, self.inner.data.fill_dir(forward))
    }
}

/// `series.iloc[...]` positional indexer.
#[pyclass]
pub struct SeriesILoc {
    inner: Series,
}

#[pymethods]
impl SeriesILoc {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(i) = key.extract::<isize>() {
            let i = norm_idx(i, self.inner.len())?;
            return Ok(np_scalar_to_py(py, &self.inner.data, i));
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            return Ok(Py::new(py, slice_series(&self.inner, slice)?)?.into_any());
        }
        Err(PyIndexError::new_err(
            "iloc key must be an integer or slice",
        ))
    }
}

fn slice_series(s: &Series, slice: &Bound<'_, PySlice>) -> PyResult<PySeries> {
    let len = s.len();
    let info = slice.indices(len as isize)?;
    let (start, stop, step) = (info.start, info.stop, info.step);
    let positions = strided(start, stop, step);
    let data = s.data.take(&positions);
    let index = Arc::new(s.index.take(&positions));
    Ok(PySeries {
        inner: Series::new(s.name.clone(), data, index),
    })
}

/// The RHS of a Series binary op as an `f64` vector — another Series (aligned by
/// position) or a broadcast scalar. Two Series must share an index (see
/// [`require_aligned`]); volas never silently aligns by label.
/// The right-hand operand of a Series arithmetic op as a length-aligned column.
/// A Series must share the index; a scalar broadcasts with a *type-based* dtype
/// (Python `int`/`bool` -> i64, `float` -> f64) so `int_series + 2 -> int64` but
/// `+ 2.0 -> float64`, matching pandas. Anything else is unsupported.
fn series_rhs_col(s: &Series, other: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(o) = other.extract::<PyRef<PySeries>>() {
        require_aligned(&s.index, &o.inner.index)?;
        Ok(o.inner.data.clone())
    } else if let Ok(b) = other.extract::<bool>() {
        Ok(Column::i64(vec![b as i64; s.len()]))
    } else if let Ok(i) = other.extract::<i64>() {
        Ok(Column::i64(vec![i; s.len()]))
    } else if let Ok(x) = other.extract::<f64>() {
        Ok(Column::f64(vec![x; s.len()]))
    } else {
        Err(PyTypeError::new_err(
            "unsupported operand for a Series operation",
        ))
    }
}

/// Guard a positional Series binary op: the two operands must share an index.
/// Same-frame columns share the index handle (`Arc::ptr_eq`, O(1)); otherwise the
/// indexes are compared by value. A mismatch is an error rather than a silently
/// misaligned (positional) result — volas does not auto-align by label.
fn require_aligned(a: &Arc<Index>, b: &Arc<Index>) -> PyResult<()> {
    if Arc::ptr_eq(a, b) || **a == **b {
        Ok(())
    } else {
        Err(PyValueError::new_err(
            "operands have different indexes; volas aligns by position, not by \
             label — reindex or slice them to a common index first",
        ))
    }
}

/// A new F64 `Series` carrying `s`'s name and index.
fn f64_series(s: &Series, out: Vec<f64>) -> PySeries {
    PySeries {
        inner: Series::new(s.name.clone(), Column::f64(out), Arc::clone(&s.index)),
    }
}

/// A new `Series` carrying `s`'s name and index over an already-built column.
/// Used by the dtype-preserving transforms (the typed Column ops decide dtype).
fn col_to_series(s: &Series, data: Column) -> PySeries {
    PySeries {
        inner: Series::new(s.name.clone(), data, Arc::clone(&s.index)),
    }
}

/// The pandas `describe` row labels (the index of a describe result).
fn describe_labels() -> Vec<String> {
    ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// A new Bool `Series` carrying `s`'s name and index.
fn bool_series(s: &Series, out: Vec<bool>) -> PySeries {
    PySeries {
        inner: Series::new(s.name.clone(), Column::bool(out), Arc::clone(&s.index)),
    }
}

/// A Series `+ - *` op against `other` (scalar / aligned Series), dtype-preserving
/// via the typed [`Column::binary`]. `swap` puts `other` on the left (the reflected
/// `__radd__` etc.). True division is separate ([`series_div`], always float).
fn series_arith(
    s: &Series,
    other: &Bound<'_, PyAny>,
    op: BinOp,
    swap: bool,
) -> PyResult<PySeries> {
    let rhs = series_rhs_col(s, other)?;
    let (lhs, rhs) = if swap { (&rhs, &s.data) } else { (&s.data, &rhs) };
    Ok(col_to_series(s, lhs.binary(rhs, op).map_err(pyerr)?))
}

/// A Series `/` op (always float). `swap` reflects it (`__rtruediv__`).
fn series_div(s: &Series, other: &Bound<'_, PyAny>, swap: bool) -> PyResult<PySeries> {
    let rhs = series_rhs_col(s, other)?;
    let (lhs, rhs) = if swap { (&rhs, &s.data) } else { (&s.data, &rhs) };
    Ok(col_to_series(s, lhs.div(rhs).map_err(pyerr)?))
}

/// A Series `//` op (floor division, dtype-preserving). `swap` reflects it
/// (`__rfloordiv__`).
fn series_floordiv(s: &Series, other: &Bound<'_, PyAny>, swap: bool) -> PyResult<PySeries> {
    let rhs = series_rhs_col(s, other)?;
    let (lhs, rhs) = if swap { (&rhs, &s.data) } else { (&s.data, &rhs) };
    Ok(col_to_series(s, lhs.floordiv(rhs).map_err(pyerr)?))
}

/// Element-wise comparison -> bool Series (positional), dtype-aware via
/// [`Column::compare`]: `str` / `datetime` / `bool` compare by native value (no f64
/// funnel), numeric as f64. A missing slot follows IEEE (`!=` true, else false).
/// The right operand is built to the left column's dtype (a str scalar for a str
/// column, a parsed timestamp for a datetime column, a number for a numeric one).
fn series_cmp(s: &Series, other: &Bound<'_, PyAny>, op: CmpOp) -> PyResult<PySeries> {
    let rhs = compare_rhs_col(s, other)?;
    Ok(col_to_series(s, s.data.compare(&rhs, op).map_err(pyerr)?))
}

/// Build the right operand of a comparison as a column matching the left column's
/// dtype: a `Series` contributes its own column (index-aligned); a scalar is
/// broadcast and typed by the left dtype.
fn compare_rhs_col(s: &Series, other: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(o) = other.extract::<PyRef<PySeries>>() {
        require_aligned(&s.index, &o.inner.index)?;
        return Ok(o.inner.data.clone());
    }
    cmp_scalar_col(other, s.data.dtype(), s.len())
}

/// Broadcast a comparison scalar to `n` rows, typed for a `dtype` column: a `str`
/// scalar for a `Str` column, a parsed timestamp for a `Datetime` column,
/// otherwise a bool / int / float column. A scalar that cannot match the dtype is
/// a `TypeError` (rather than a silent all-`False` mask).
fn cmp_scalar_col(v: &Bound<'_, PyAny>, dtype: DType, n: usize) -> PyResult<Column> {
    match dtype {
        DType::Utf8 => {
            let s = v.extract::<String>().map_err(|_| {
                PyTypeError::new_err("cannot compare a str column with a non-string scalar")
            })?;
            Ok(Column::str(vec![s; n]))
        }
        DType::Datetime => Ok(Column::datetime(vec![parse_ts(v)?; n])),
        _ if v.extract::<bool>().is_ok() => Ok(Column::bool(vec![v.extract::<bool>()?; n])),
        _ if v.extract::<i64>().is_ok() => Ok(Column::i64(vec![v.extract::<i64>()?; n])),
        _ => {
            let x = v
                .extract::<f64>()
                .map_err(|_| PyTypeError::new_err("unsupported operand for a comparison"))?;
            Ok(Column::f64(vec![x; n]))
        }
    }
}

/// The non-NaN `f64` values of a column (for NaN-skipping reductions).
fn non_nan(col: &Column) -> Vec<f64> {
    col.to_f64_vec()
        .into_iter()
        .filter(|x| !x.is_nan())
        .collect()
}

/// The position of the first maximum (`want_max`) or minimum non-NaN value; errors
/// on an all-NA column. Backs `Series.idxmax` / `idxmin`.
fn argext(col: &Column, want_max: bool) -> PyResult<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, x) in col.to_f64_vec().into_iter().enumerate() {
        if x.is_nan() {
            continue;
        }
        match best {
            Some((_, b)) if (want_max && x <= b) || (!want_max && x >= b) => {}
            _ => best = Some((i, x)),
        }
    }
    best.map(|(i, _)| i)
        .ok_or_else(|| PyValueError::new_err("Encountered all NA values"))
}

/// A column coerced to bool (a `Bool` column as-is, else `x != 0.0`).
fn to_bool_vec(col: &Column) -> Vec<bool> {
    match col {
        Column::Bool(v, _) => v.to_vec(),
        other => other.to_f64_vec().iter().map(|&x| x != 0.0).collect(),
    }
}

/// Recognise a boolean-mask key (`s[mask]` / `df[mask] = v`): a boolean Series, a
/// boolean ndarray, or a non-empty `list[bool]`. Returns `None` for any other key
/// so the caller can fall through to its label / position / column handling.
fn bool_mask_key(key: &Bound<'_, PyAny>) -> PyResult<Option<Vec<bool>>> {
    if let Ok(s) = key.extract::<PyRef<PySeries>>() {
        return Ok(match &s.inner.data {
            Column::Bool(m, _) => Some(m.to_vec()),
            _ => None,
        });
    }
    if let Ok(arr) = key.extract::<PyReadonlyArray1<bool>>() {
        return Ok(Some(arr.as_slice()?.to_vec()));
    }
    match key.extract::<Vec<bool>>() {
        Ok(m) if !m.is_empty() => Ok(Some(m)),
        _ => Ok(None),
    }
}

/// Resolve the `other` argument of `where` / `mask` to a length-`n` fill column
/// plus the dtype it contributes to the result. A scalar broadcasts (its dtype is
/// value-based: an integral value contributes int); a Series contributes its own
/// dtype (index-aligned); the default (`None`) fills a dtype-preserving NA.
fn where_other_resolve(
    other: Option<&Bound<'_, PyAny>>,
    s: &Series,
) -> PyResult<(Column, DType)> {
    let n = s.len();
    match other {
        // the default `other` is a dtype-preserving all-NA column (str -> NA str,
        // datetime -> NaT, int/bool -> their NA), so a str/datetime `where` keeps
        // its kept values instead of funneling them to NaN.
        None => {
            let dt = s.data.dtype();
            Ok((Column::na_of(dt, n), dt))
        }
        // an explicit `None` / `NaN` / `volas.NA` fill is the same dtype-preserving
        // NA as the default, so `where(mask, volas.NA)` keeps the column's dtype.
        Some(o) if is_na_like_py(o) => {
            let dt = s.data.dtype();
            Ok((Column::na_of(dt, n), dt))
        }
        Some(o) => {
            if let Ok(ser) = o.extract::<PyRef<PySeries>>() {
                require_aligned(&s.index, &ser.inner.index)?;
                let dt = ser.inner.data.dtype();
                Ok((ser.inner.data.clone(), dt))
            } else if let Ok(b) = o.extract::<bool>() {
                // a bool fill contributes a bool result (checked before f64, since
                // Python bool is an int subclass)
                Ok((Column::bool(vec![b; n]), DType::Bool))
            } else if let Ok(x) = o.extract::<f64>() {
                let dt = if fits(DType::I64, x) { DType::I64 } else { DType::F64 };
                Ok((Column::f64(vec![x; n]), dt))
            } else {
                Err(PyTypeError::new_err(
                    "where/mask: `other` must be a number or a Series",
                ))
            }
        }
    }
}

/// Element-wise boolean logic -> bool Series (both operands coerced to bool).
fn series_logical(s: &Series, other: &Bound<'_, PyAny>, op: BoolOp) -> PyResult<PySeries> {
    let n = s.data.len();
    let rhs: Column = if let Ok(o) = other.extract::<PyRef<PySeries>>() {
        require_aligned(&s.index, &o.inner.index)?;
        o.inner.data.clone()
    } else if let Ok(b) = other.extract::<bool>() {
        Column::bool(vec![b; n])
    } else if let Ok(x) = other.extract::<f64>() {
        Column::bool(vec![x != 0.0; n])
    } else {
        return Err(PyTypeError::new_err(
            "unsupported operand for a Series logical op",
        ));
    };
    // Kleene three-valued logic, propagating volas.NA.
    Ok(col_to_series(s, s.data.logical(&rhs, op)))
}

fn strided(start: isize, stop: isize, step: isize) -> Vec<usize> {
    let mut out = Vec::new();
    if step > 0 {
        let mut i = start;
        while i < stop {
            out.push(i as usize);
            i += step;
        }
    } else if step < 0 {
        let mut i = start;
        while i > stop {
            out.push(i as usize);
            i += step;
        }
    }
    out
}

// --- Row -------------------------------------------------------------------

/// A single DataFrame row (the result of `df.iloc[i]` / `df.loc[label]`): a
/// faithful 1-row frame carrying its index label and every column's *typed*
/// value (no lossy f64 coercion, no flag pair to remember the index kind).
#[pyclass(name = "Row")]
pub struct PyRow {
    inner: DataFrame,
}

#[pymethods]
impl PyRow {
    /// The row's index label.
    #[getter]
    fn name(&self, py: Python<'_>) -> Py<PyAny> {
        label_to_py(py, self.inner.index(), 0)
    }

    /// A single value by column name (``row[col]``).
    ///
    /// Returns:
    ///     the typed scalar at that column.
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let col = self.inner.column(key).map_err(pyerr)?;
        Ok(np_scalar_to_py(py, col, 0))
    }

    /// The row's values as a ``(1, n_columns)`` float64 NumPy array. Only valid for
    /// an all-numeric row — a str / datetime cell cannot be represented as float64
    /// without a silent NaN, so it errors (contract R2 / C4); read those via
    /// ``to_dict()`` or ``row[col]`` instead.
    ///
    /// Returns:
    ///     numpy.ndarray
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        for c in self.inner.columns() {
            c.require_numeric().map_err(pyerr)?;
        }
        let (data, h, w) = self.inner.to_row_major_f64();
        Ok(ndarray::Array2::from_shape_vec((h, w), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py))
    }

    /// The row as a typed `{column: value}` dict (pandas `Series.to_dict`).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            d.set_item(name, scalar_to_py(py, col, 0))?;
        }
        Ok(d)
    }

    /// Vertical repr — `column   value` lines plus a `Name: <row label>` footer.
    /// No `dtype:` is printed: a Row is a typed record, not a Series, and has no
    /// single dtype (pandas prints `dtype: object` only because its row IS an
    /// object Series). `str` and `repr` are identical.
    fn __repr__(&self) -> String {
        render_row(&self.inner, true)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Render the row as text without the `Name` footer (pandas
    /// `Series.to_string`).
    fn to_string(&self) -> String {
        render_row(&self.inner, false)
    }
}

// --- DataFrame -------------------------------------------------------------

/// ``volas.DataFrame`` — an ordered, named, time-indexed OHLCV table with
/// indicator-directive indexing and pandas-compatible positional / label access.
///
/// Construct from a dict of columns, or read a CSV::
///
///     df = volas.DataFrame({'close': [10.0, 11.0], 'volume': [100, 120]})
///     df = volas.read_csv('ohlcv.csv', index_col='time')
///
/// The headline feature is string indexing: a plain column name returns that
/// column, and an *indicator directive* is computed on demand, cached, and
/// incrementally refreshed thereafter::
///
///     df['close']            # a column, as a Series
///     df['ma:5']             # SMA(5) of close (directive) — computed & cached
///     df['macd.signal']      # MACD signal line
///     df['close > open']     # a boolean directive -> bool Series
///     df[['open', 'close']]  # a sub-frame
///     df[df['close'] > 100]  # boolean-mask row filter
///
/// Positional / label access mirrors pandas via ``.iloc`` / ``.loc`` (2-D get +
/// set) and the scalar ``.iat`` / ``.at``; common transforms (``head``,
/// ``tail``, ``dropna``, ``sort_index``, ``reset_index``, ``set_index``,
/// ``rename``, ``astype``, ``to_numpy``, ``to_pandas``, ``to_csv``) follow the
/// pandas spelling. ``cumulate`` resamples to a coarser timeframe; ``append``
/// grows the frame in place for live streaming.
///
/// Args:
///     data (dict[str, Sequence] | DataFrame): a dict of column name -> equal-length
///         values, or another volas DataFrame to copy (its index, aliases and tf-state are
///         carried — like ``df.copy()``). A pandas DataFrame is not accepted; use
///         ``from_pandas``. Build a DatetimeIndex from a column with ``read_csv`` or
///         ``to_datetime`` + ``set_index`` (+ ``tz_localize`` / ``tz_convert``).
///     columns (list[str], optional): select and order the columns to keep (like
///         ``df[[...]]``); a name not present raises ``KeyError``. An empty list or a
///         duplicate name is rejected, and an absent column is never NaN-filled.
///     time_frame (str | TimeFrame, optional): make this a tf-aware (cumulating) frame at
///         this bar interval; the given rows are taken as already-final bars and later
///         ``append``s fold finer bars into them. Requires a DatetimeIndex.
///     cumulators (dict[str, str], optional): per-column aggregator overrides for folding
///         (e.g. ``{'amount': 'sum'}``); only meaningful together with ``time_frame``.
/// Live cumulation state carried by a tf-aware DataFrame (set via the
/// `time_frame` constructor arg or `cumulate`): the target frame, the per-column
/// aggregators, and the raw fine bars of the still-open (forming) period —
/// `df.iloc[-1]` is that period's running bar, which `append` keeps updating.
#[derive(Clone)]
struct TfState {
    time_frame: TimeFrame,
    cumulators: AggSpec,
    /// Raw fine bars of the current open period (`None` until the first
    /// folded append), kept so a re-sent forming bar updates (deduped) rather
    /// than double-counts.
    open: Option<DataFrame>,
}

#[pyclass(name = "DataFrame")]
pub struct PyDataFrame {
    pub(crate) inner: DataFrame,
    /// Cumulation state when this is a tf-aware frame; `None` for a plain frame.
    tf: Option<TfState>,
}

/// Read cell `i` of a directive-result column as f64 (`Bool` -> 0/1, `I64` -> as f64),
/// for the finite-memory-vs-recursive refresh probe. NaN for other dtypes.
fn col_value(col: &Column, i: usize) -> f64 {
    match col {
        Column::F64(v) => v[i],
        Column::Bool(v, _) => {
            if v[i] {
                1.0
            } else {
                0.0
            }
        }
        Column::I64(v, _) => v[i] as f64,
        _ => f64::NAN,
    }
}

impl PyDataFrame {
    /// Wrap a core frame as a plain (non-cumulating) DataFrame — the default for
    /// every derived frame (slices, projections, head/tail, ...).
    pub(crate) fn plain(inner: DataFrame) -> Self {
        PyDataFrame { inner, tf: None }
    }

    /// Element-wise comparison backing `__eq__` / `__ne__`: against another
    /// DataFrame (identical column names + shared index) or a scalar (broadcast),
    /// producing a bool DataFrame. Compared by position; never auto-aligned.
    fn compare(&self, other: &Bound<'_, PyAny>, op: CmpOp) -> PyResult<PyDataFrame> {
        let cols: Vec<Column> = if let Ok(o) = other.extract::<PyRef<PyDataFrame>>() {
            if self.inner.names() != o.inner.names() {
                return Err(PyValueError::new_err(
                    "cannot compare DataFrames with different columns",
                ));
            }
            require_aligned(self.inner.index(), o.inner.index())?;
            self.inner
                .columns()
                .iter()
                .zip(o.inner.columns())
                .map(|(a, b)| a.compare(b, op))
                .collect::<Result<_, _>>()
                .map_err(pyerr)?
        } else {
            // a scalar is broadcast and typed per column; a column whose dtype the
            // scalar cannot match is a TypeError (no silent all-False mask).
            self.inner
                .columns()
                .iter()
                .map(|c| c.compare(&cmp_scalar_col(other, c.dtype(), c.len())?, op).map_err(pyerr))
                .collect::<PyResult<_>>()?
        };
        self.with_columns(cols)
    }

    /// Rebuild a plain frame from `cols`, reusing this frame's names and index (the
    /// columns must be height-aligned). Backs `compare` / `fillna` / `mask_na`.
    fn with_columns(&self, cols: Vec<Column>) -> PyResult<PyDataFrame> {
        DataFrame::new(
            self.inner.names().to_vec(),
            cols,
            Some((**self.inner.index()).clone()),
        )
        .map(PyDataFrame::plain)
        .map_err(pyerr)
    }

    /// One column as a `PySeries` (carrying its name + the frame index), for
    /// column-wise delegation to Series methods.
    fn col_as_series(&self, name: &str, col: &Column) -> PySeries {
        PySeries {
            inner: Series::new(Some(name.to_string()), col.clone(), Arc::clone(self.inner.index())),
        }
    }

    /// Apply a Series transform to every column -> a new frame (pandas column-wise
    /// `df.cumsum()` etc.). Each column's own dtype rule applies; a column the op
    /// rejects (e.g. a string column under a numeric transform) propagates its error.
    fn map_cols(&self, op: impl Fn(&PySeries) -> PyResult<PySeries>) -> PyResult<PyDataFrame> {
        let cols = self
            .inner
            .names()
            .iter()
            .zip(self.inner.columns())
            .map(|(name, col)| Ok(op(&self.col_as_series(name, col))?.inner.data))
            .collect::<PyResult<Vec<_>>>()?;
        self.with_columns(cols)
    }

    /// Reduce each numeric column to a scalar -> a Series indexed by column name
    /// (pandas column-wise `df.sem()` etc.; non-numeric columns are skipped).
    fn reduce_cols(&self, op: impl Fn(&Column) -> f64) -> PySeries {
        let mut names = Vec::new();
        let mut vals = Vec::new();
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            if col.dtype().is_numeric() {
                names.push(name.clone());
                vals.push(op(col));
            }
        }
        PySeries {
            inner: Series::new(None, Column::f64(vals), Arc::new(Index::str(names))),
        }
    }

    /// Directional fill (`forward` = ffill, else bfill) over every column,
    /// delegating to the per-column validity-aware `Column::fill_dir` so int /
    /// bool / str holes carry directionally too (like the Series version), not
    /// only float NaN. Backs `ffill` / `bfill`.
    fn fill_dir(&self, forward: bool) -> PyResult<PyDataFrame> {
        let cols: Vec<Column> = self
            .inner
            .columns()
            .iter()
            .map(|c| c.fill_dir(forward))
            .collect();
        self.with_columns(cols)
    }

    /// Pairwise matrix (corr / cov) over the numeric columns; result column `j`
    /// is `[op(col_i, col_j) for i]`, indexed and labelled by the column names.
    /// Backs `corr` / `cov`.
    fn corr_cov(&self, op: fn(&[f64], &[f64]) -> f64) -> PyResult<PyDataFrame> {
        let numeric: Vec<(String, Vec<f64>)> = self
            .inner
            .names()
            .iter()
            .zip(self.inner.columns())
            .filter(|(_, c)| c.dtype().is_numeric())
            .map(|(n, c)| (n.clone(), c.to_f64_vec()))
            .collect();
        let names: Vec<String> = numeric.iter().map(|(n, _)| n.clone()).collect();
        let cols: Vec<Column> = numeric
            .iter()
            .map(|(_, cj)| Column::f64(numeric.iter().map(|(_, ci)| op(ci, cj)).collect()))
            .collect();
        DataFrame::new(names.clone(), cols, Some(Index::str(names)))
            .map(PyDataFrame::plain)
            .map_err(pyerr)
    }

    /// `df.where` / `df.mask` shared core: per-cell keep/replace against a
    /// same-shape boolean frame, dtype-preservingly per column (the default
    /// `other = None` keeps each column's dtype and fills NA; an explicit numeric
    /// `other` promotes via the supertype).
    fn where_mask(
        &self,
        cond: &PyDataFrame,
        other: Option<f64>,
        is_where: bool,
    ) -> PyResult<PyDataFrame> {
        if cond.inner.width() != self.inner.width() || cond.inner.height() != self.inner.height() {
            return Err(PyValueError::new_err(
                "where/mask: `cond` must have the same shape as the frame",
            ));
        }
        // the condition must be boolean — a numeric mask is rejected (pandas-shaped)
        if let Some(cc) = cond
            .inner
            .columns()
            .iter()
            .find(|c| !matches!(c, Column::Bool(..)))
        {
            return Err(PyTypeError::new_err(format!(
                "where/mask: `cond` must be a boolean frame, got a {} column",
                cc.dtype()
            )));
        }
        let other_dt = match other {
            Some(x) if fits(DType::I64, x) => DType::I64,
            _ => DType::F64,
        };
        let cols = self
            .inner
            .columns()
            .iter()
            .zip(cond.inner.columns())
            .map(|(keep_col, cond_col)| {
                let mut c = to_bool_vec(cond_col);
                if !is_where {
                    c.iter_mut().for_each(|b| *b = !*b);
                }
                let kd = keep_col.dtype();
                // the default `other` is a dtype-preserving NA (keeps str / datetime /
                // int values); an explicit numeric fill promotes via the supertype.
                let (other_col, target) = match other {
                    None => (Column::na_of(kd, keep_col.len()), kd),
                    Some(x) => {
                        let target = if kd.is_float() { kd } else { binary_supertype(kd, other_dt) };
                        (Column::f64(vec![x; keep_col.len()]), target)
                    }
                };
                keep_col.select(&c, &other_col, target)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(pyerr)?;
        self.with_columns(cols)
    }

    /// Rebuild `inner` from new columns, preserving names and index (drops the
    /// directive cache, which a write would stale anyway). Backs mask assignment.
    fn rebuild_with(&mut self, cols: Vec<Column>) -> PyResult<()> {
        self.inner = DataFrame::new(
            self.inner.names().to_vec(),
            cols,
            Some((**self.inner.index()).clone()),
        )
        .map_err(pyerr)?;
        Ok(())
    }

    /// `df[row_mask] = v`: set every column's True rows to the scalar, keeping each
    /// column's dtype (pandas' whole-row boolean assignment), via the shared
    /// `scatter_scalar` primitive. Atomic — if any column would take the value
    /// lossily, the per-column map errors and nothing is written.
    fn assign_row_mask(&mut self, mask: &[bool], value: &Bound<'_, PyAny>) -> PyResult<()> {
        if mask.len() != self.inner.height() {
            return Err(PyValueError::new_err(format!(
                "boolean mask length {} != frame height {}",
                mask.len(),
                self.inner.height()
            )));
        }
        let positions: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| m.then_some(i))
            .collect();
        let cols = self
            .inner
            .columns()
            .iter()
            .map(|c| scatter_scalar(c, &positions, value))
            .collect::<PyResult<Vec<_>>>()?;
        self.rebuild_with(cols)
    }

    /// `df[bool_frame] = v`: per-cell assignment where the mask is True, keeping
    /// each column's dtype. Atomic, like `assign_row_mask`. The condition frame
    /// must be boolean — the same contract as `DataFrame.where` (a numeric / string
    /// mask is rejected up front, not coerced through `x != 0.0`).
    fn assign_cell_mask(&mut self, cond: &PyDataFrame, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if cond.inner.width() != self.inner.width() || cond.inner.height() != self.inner.height() {
            return Err(PyValueError::new_err(
                "df[mask] = v: `mask` must have the same shape as the frame",
            ));
        }
        if let Some(cc) = cond
            .inner
            .columns()
            .iter()
            .find(|c| !matches!(c, Column::Bool(..)))
        {
            return Err(PyTypeError::new_err(format!(
                "df[mask] = v: `mask` must be a boolean frame, got a {} column",
                cc.dtype()
            )));
        }
        let cols = self
            .inner
            .columns()
            .iter()
            .zip(cond.inner.columns())
            .map(|(col, cond_col)| {
                let positions: Vec<usize> = to_bool_vec(cond_col)
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &m)| m.then_some(i))
                    .collect();
                scatter_scalar(col, &positions, value)
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.rebuild_with(cols)
    }

    /// Per-cell missing mask -> a bool frame; backs `isna` (want_na=true) /
    /// `notna`. Reads the column validity (every dtype), so an int/bool/str NA
    /// and a datetime NaT are detected, not just a float NaN.
    fn mask_na(&self, want_na: bool) -> PyResult<PyDataFrame> {
        let cols = self
            .inner
            .columns()
            .iter()
            .map(|c| Column::bool((0..c.len()).map(|i| (!c.is_valid(i)) == want_na).collect()))
            .collect();
        self.with_columns(cols)
    }

    /// Fold incoming fine bars into a tf-aware frame: each bar either extends the
    /// open period's forming bar (update `inner`'s last row in place + mark its
    /// computed tail stale) or rolls over into a new period (append a fresh
    /// forming row). Assumes `self.tf` is `Some`. A re-sent forming bar (same
    /// timestamp) updates the period rather than double-counting it.
    fn fold_append(&mut self, fine: &DataFrame) -> PyResult<()> {
        let last_dt = |df: &DataFrame| -> i64 {
            match df.index().kind() {
                IndexKind::Datetime(v, _) => v[v.len() - 1],
                _ => unreachable!("checked by caller"),
            }
        };
        let PyDataFrame { inner, tf } = self;
        let tfs = tf.as_mut().expect("fold_append on a plain frame");
        let frame = tfs.time_frame;
        let (fine_ts, tz) = match fine.index().kind() {
            IndexKind::Datetime(v, tz) => (v.clone(), *tz),
            _ => {
                return Err(PyValueError::new_err(
                    "append to a time_frame DataFrame requires a DatetimeIndex",
                ))
            }
        };
        for i in 0..fine.height() {
            let bar_ts = fine_ts[i];
            let key = frame.unify_tz(bar_ts, tz);
            let same_period = tfs
                .open
                .as_ref()
                .is_some_and(|open| frame.unify_tz(last_dt(open), tz) == key);
            let bar = fine.slice(i, i + 1);
            if same_period {
                let open = tfs.open.as_mut().unwrap();
                // A re-sent forming bar (same ts) replaces the last open bar.
                if last_dt(open) == bar_ts {
                    *open = open.slice(0, open.height() - 1);
                }
                open.append(&bar).map_err(pyerr)?;
                let agg = aggregate_period(open, &tfs.cumulators).map_err(pyerr)?;
                let last = inner.height() - 1;
                // `assign_positions` invalidates each written column's dependent
                // directive columns, so the forming row's indicators recompute
                // correctly on the next read — no explicit invalidate needed.
                for (name, col) in agg.names().iter().zip(agg.columns()) {
                    if let Some(j) = inner.column_pos(name) {
                        inner.assign_positions(j, &[last], col).map_err(pyerr)?;
                    }
                }
            } else {
                // Roll over: the previous forming bar (if any) is already final in
                // `inner`; start a new open period and append its forming row.
                let agg = aggregate_period(&bar, &tfs.cumulators).map_err(pyerr)?;
                tfs.open = Some(bar);
                inner.append(&agg).map_err(pyerr)?;
            }
        }
        Ok(())
    }

    fn wrap_series(&self, name: String, col: Column) -> PySeries {
        PySeries {
            inner: Series::new(Some(name), col, Arc::clone(self.inner.index())),
        }
    }

    /// Recompute the stale tail of cached directive columns in place — all of
    /// them if `only` is `None`, else just the named one. O(lookback + new rows)
    /// per column. Done against the real (non-computed) columns so a bare-name
    /// directive recomputes and no cached buffer is pinned.
    fn refresh_computed(&mut self, only: Option<&str>) -> PyResult<()> {
        let height = self.inner.height();
        let stale = self.inner.stale_computed_columns(only);
        if stale.is_empty() {
            return Ok(());
        }
        let mut base: Option<DataFrame> = None;
        for (name, meta) in stale {
            let (lb, vr) = (meta.lookback, meta.valid_rows);
            if meta.state.is_some() {
                if height == vr + 1 {
                    if let Some(value) = volas_directive::exec::execute_resume_default_series_one(
                        &self.inner,
                        &meta.directive,
                        vr,
                    ) {
                        self.inner
                            .update_computed_f64_value(&name, vr, value)
                            .map_err(pyerr)?;
                        continue;
                    }
                }
                if let Some((tail, new_state)) =
                    volas_directive::exec::execute_resume_default_series(
                        &self.inner,
                        &meta.directive,
                        vr,
                    )
                {
                    self.inner
                        .update_computed_tail(&name, vr, &tail)
                        .map_err(pyerr)?;
                    self.inner.set_computed_state(&name, Some(new_state));
                    continue;
                }
            }
            let node = parse(&meta.directive).map_err(value_err)?;
            // State-carry fast-path (additive): if this column carries a recursive
            // state, continue the recursion over only the new rows `[vr, height)` —
            // O(new rows), bit-identical to a full recompute — then refresh the carried
            // state. This is the high-performance append path for recursive indicators
            // (and continues correctly across a head-dropping slice, since the state is
            // self-contained and the resume never reads before `vr`). On `None` (no
            // resume kernel for this directive) we fall through to the existing
            // probe / full-recompute path unchanged — always correct.
            if let Some(state) = &meta.state {
                // Default-series resumes only read canonical input columns, so they
                // can skip building a non-computed base frame on the single-column
                // append hot path. Explicit series may reference stale computed
                // columns, so those still use the base-frame fallback below.
                if directive_uses_default_series(&node) {
                    if let Some((tail, new_state)) = volas_directive::exec::execute_resume(
                        &self.inner,
                        &node,
                        state,
                        vr,
                        meta.origin,
                    ) {
                        self.inner
                            .update_computed_tail(&name, vr, &tail)
                            .map_err(pyerr)?;
                        self.inner.set_computed_state(&name, Some(new_state));
                        continue;
                    }
                }
            }
            if base.is_none() {
                let computed_names: HashSet<String> =
                    self.inner.computed_names().into_iter().collect();
                let real_names: Vec<String> = self
                    .inner
                    .names()
                    .iter()
                    .filter(|n| !computed_names.contains(*n))
                    .cloned()
                    .collect();
                base = Some(self.inner.select(&real_names).map_err(pyerr)?);
            }
            let base = base
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("internal base frame was not initialized"))?;
            if let Some(state) = &meta.state {
                if let Some((tail, new_state)) =
                    volas_directive::exec::execute_resume(base, &node, state, vr, meta.origin)
                {
                    self.inner
                        .update_computed_tail(&name, vr, &tail)
                        .map_err(pyerr)?;
                    self.inner.set_computed_state(&name, Some(new_state));
                    continue;
                }
            }
            // A finite-memory indicator (SMA, ROC, price transforms, CDL, …) depends
            // only on a fixed trailing window, so a windowed recompute is exact and
            // O(lookback). A recursive / stateful one (EMA / Wilder / MACD / SAR /
            // cumulative OBV / HT / index) depends on the WHOLE prefix `[0, i]`, so a
            // window re-warms-up and silently diverges (the bug). Probe with a
            // `2*lookback` window that overlaps the last KNOWN row (`vr-1`): if it
            // reproduces that cached value the window is exact, else recompute the full
            // column from row 0 — O(n) but exact for every indicator. (A slice that
            // dropped its head only has the visible rows, so a stateful indicator there
            // cannot be continued past the missing history.)
            let (recomputed, off) = if lb > 0 && vr > 2 * lb {
                let start = vr - 2 * lb;
                let windowed = execute(&base.slice(start, height), &node).map_err(value_err)?;
                let cached_val = col_value(self.inner.column(&name).map_err(pyerr)?, vr - 1);
                let probe = col_value(&windowed, vr - 1 - start);
                if probe.is_finite()
                    && (probe - cached_val).abs() <= 1e-9 * cached_val.abs().max(1.0)
                {
                    (windowed, vr - start)
                } else {
                    (execute(&base, &node).map_err(value_err)?, vr)
                }
            } else {
                (execute(&base, &node).map_err(value_err)?, vr)
            };
            // Write the stale tail back into the column at its original dtype.
            let tail = recomputed.slice(off, recomputed.len());
            self.inner
                .update_computed_tail(&name, vr, &tail)
                .map_err(pyerr)?;
            // The column is now valid for all rows. If this directive supports a
            // resume, (re)capture its recursive state so the NEXT append takes the
            // O(new-rows) fast-path. This repopulates state dropped by an invalidating
            // base-column write or a head-dropping slice. `None` leaves it on the
            // fallback. (`recomputed` is the full column on the full-recompute branch
            // and the window tail otherwise; `initial_state` derives the cumulative
            // family's state from the raw inputs in `base`, so either is fine.)
            let new_state = volas_directive::exec::initial_state(&base, &node, &recomputed);
            if new_state.is_some() {
                self.inner.set_computed_state(&name, new_state);
            }
        }
        Ok(())
    }
}

#[pymethods]
impl PyDataFrame {
    // Constructor — the user-facing argument list & usage live in the class
    // docstring (pyo3 does not surface a `#[new]` doc comment to Python).
    #[new]
    #[pyo3(signature = (data, columns = None, time_frame = None, cumulators = None, dtype = None))]
    fn new(
        data: &Bound<'_, PyAny>,
        columns: Option<Vec<String>>,
        time_frame: Option<&Bound<'_, PyAny>>,
        cumulators: Option<&Bound<'_, PyDict>>,
        dtype: Option<&str>,
    ) -> PyResult<Self> {
        // `columns`, when given, selects and orders the columns — a strict projection, like
        // `df[[...]]`: a name not present raises KeyError, and an empty list or a duplicate
        // name is rejected. It never silently NaN-fills an absent column.
        if let Some(cols) = &columns {
            if cols.is_empty() {
                return Err(PyValueError::new_err("columns must not be empty"));
            }
            let mut seen = HashSet::with_capacity(cols.len());
            for c in cols {
                if !seen.insert(c.as_str()) {
                    return Err(PyValueError::new_err(format!(
                        "duplicate column \"{c}\" in columns"
                    )));
                }
            }
        }
        // `data` is polymorphic over volas's own inputs: another volas DataFrame (copied —
        // index, aliases and any tf-state carried, exactly like `df.copy()`), or a dict of
        // columns (a fresh RangeIndex); with `columns` the frame is projected onto them. A
        // pandas DataFrame is deliberately NOT accepted here — use `from_pandas`, which keeps
        // volas pandas-free at import. To build a DatetimeIndex from a column, parse it with
        // `to_datetime` then `set_index` (or use `read_csv`).
        let (df, tf) = if let Ok(other) = data.extract::<PyRef<PyDataFrame>>() {
            match &columns {
                None => (other.inner.clone(), other.tf.clone()),
                Some(cols) => {
                    // Project the frame, and a tf-aware frame's forming-period state, onto
                    // `cols`. The cumulator spec is per-column with a default, so the dropped
                    // columns' rules simply go unused — folding stays correct on the kept ones.
                    let inner = other.inner.select(cols).map_err(pyerr)?;
                    let tf = match &other.tf {
                        None => None,
                        Some(t) => Some(TfState {
                            time_frame: t.time_frame,
                            cumulators: t.cumulators.clone(),
                            open: t
                                .open
                                .as_ref()
                                .map(|o| o.select(cols))
                                .transpose()
                                .map_err(pyerr)?,
                        }),
                    };
                    (inner, tf)
                }
            }
        } else if let Ok(dict) = data.downcast::<PyDict>() {
            let (names, vcols) = match &columns {
                None => {
                    let mut names = Vec::new();
                    let mut vcols = Vec::new();
                    for (k, v) in dict.iter() {
                        names.push(k.extract::<String>()?);
                        vcols.push(pyany_to_column(&v)?);
                    }
                    (names, vcols)
                }
                Some(cols) => {
                    // Strict select: build only the named columns, in order.
                    let mut vcols = Vec::with_capacity(cols.len());
                    for name in cols {
                        let v = dict.get_item(name)?.ok_or_else(|| {
                            PyKeyError::new_err(format!("column \"{name}\" not found"))
                        })?;
                        vcols.push(pyany_to_column(&v)?);
                    }
                    (cols.clone(), vcols)
                }
            };
            (DataFrame::new(names, vcols, None).map_err(pyerr)?, None)
        } else {
            return Err(PyTypeError::new_err(
                "DataFrame(data): data must be a dict of columns or a volas DataFrame \
                 (for a pandas DataFrame use from_pandas)",
            ));
        };
        // `dtype=` casts every column to a single dtype (pandas `DataFrame(data,
        // dtype=...)`), e.g. dtype='float32'.
        let df = match dtype {
            None => df,
            Some(dt_str) => {
                let dt = parse_dtype(dt_str)?;
                let cols = df
                    .columns()
                    .iter()
                    .map(|c| c.cast(dt))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(pyerr)?;
                DataFrame::new(df.names().to_vec(), cols, Some((**df.index()).clone()))
                    .map_err(pyerr)?
            }
        };
        // A `time_frame` makes this a cumulating frame: the given rows are taken as
        // already-final bars at that frame (not re-aggregated), and later `append`s fold
        // finer bars into them. Requires a DatetimeIndex (build one with `set_index` first).
        if let Some(tf_obj) = time_frame {
            let frame = resolve_time_frame(tf_obj)?;
            if !matches!(df.index().kind(), IndexKind::Datetime(..)) {
                return Err(PyValueError::new_err(
                    "time_frame requires a DatetimeIndex \
                     (build one with to_datetime(df[col]) then df.set_index(col))",
                ));
            }
            let spec = build_agg_spec(cumulators)?;
            return Ok(PyDataFrame {
                inner: df,
                tf: Some(TfState {
                    time_frame: frame,
                    cumulators: spec,
                    open: None,
                }),
            });
        }
        if cumulators.is_some() {
            return Err(PyValueError::new_err("cumulators requires time_frame"));
        }
        Ok(PyDataFrame { inner: df, tf })
    }

    /// The DatetimeIndex timezone name (`"+08:00"` / `"America/New_York"`), or
    /// `None` for a tz-naive (UTC-default) or non-datetime index — mirroring
    /// pandas `df.index.tz`.
    #[getter]
    fn tz(&self) -> Option<String> {
        match self.inner.index().tz() {
            Tz::Utc => None,
            other => Some(other.name()),
        }
    }

    /// Reinterpret the index wall-clock as `tz` (pandas `tz_localize`): the
    /// displayed wall-clock is unchanged, each instant is recomputed. Use when
    /// data was ingested without a tz. Returns a new frame.
    fn tz_localize(&self, tz: &str) -> PyResult<PyDataFrame> {
        let tzv = Tz::parse(tz).map_err(pyerr)?;
        Ok(PyDataFrame::plain(
            self.inner.tz_localize(tzv).map_err(pyerr)?,
        ))
    }

    /// Change the index display / matching tz without moving any instant (pandas
    /// `tz_convert`). Returns a new frame.
    fn tz_convert(&self, tz: &str) -> PyResult<PyDataFrame> {
        let tzv = Tz::parse(tz).map_err(pyerr)?;
        Ok(PyDataFrame::plain(
            self.inner.tz_convert(tzv).map_err(pyerr)?,
        ))
    }

    /// The column names, in order.
    ///
    /// Returns:
    ///     list[str]
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    /// The frame dimensions as ``(n_rows, n_columns)`` (pandas ``shape``).
    ///
    /// Returns:
    ///     tuple[int, int]
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.height(), self.inner.width())
    }

    /// The row index as a NumPy array (``datetime64[ns]`` for a DatetimeIndex,
    /// an object array for a string index, else an integer array).
    ///
    /// Returns:
    ///     numpy.ndarray
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        index_to_numpy(py, self.inner.index())
    }

    // The indexers hold a live reference to this frame (`Py<PyDataFrame>`), not a
    // snapshot, so `df.iloc[...] = ` / `df.loc[...] = ` mutate the frame in place
    // (copy-on-write under the hood) and reads always see the current rows.

    /// Purely integer-location indexing for selection and assignment.
    ///
    /// Get ``df.iloc[i]`` (a row), ``df.iloc[a:b]`` (a sub-frame),
    /// ``df.iloc[i, j]`` (a cell), ``df.iloc[:, j]`` (a column as a Series), or
    /// ``df.iloc[rows, cols]`` (a sub-frame). Assign ``df.iloc[rows, j] = value``
    /// (copy-on-write; a prior ``copy()`` is unaffected).
    ///
    /// Usage::
    ///
    ///     df.iloc[0]            # first row
    ///     df.iloc[-1, 3]        # last row, 4th column -> scalar
    ///     df.iloc[:, 0]         # first column as a Series
    ///     df.iloc[10:20, 0:2]   # a block
    ///     df.iloc[mask, 1] = 0  # assign a column where a boolean mask is True
    #[getter]
    fn iloc(slf: Bound<'_, Self>) -> DataFrameILoc {
        DataFrameILoc {
            parent: slf.unbind(),
        }
    }

    /// Label-based indexing for selection and assignment.
    ///
    /// Get ``df.loc[label]`` (a row), ``df.loc[a:b]`` (a stop-inclusive label
    /// slice), ``df.loc[label, col]`` (a cell), ``df.loc[:, col]`` (a column),
    /// or ``df.loc[mask, col]``. Assign ``df.loc[mask, 'signal'] = 1``
    /// (copy-on-write).
    ///
    /// Usage::
    ///
    ///     df.loc['2021-01-04']               # row by datetime label
    ///     df.loc['2021-01':'2021-03']        # inclusive label slice
    ///     df.loc[df['close'] > df['open'], 'signal'] = 1
    #[getter]
    fn loc(slf: Bound<'_, Self>) -> DataFrameLoc {
        DataFrameLoc {
            parent: slf.unbind(),
        }
    }

    /// Fast scalar access by integer position: ``df.iat[i, j]`` to get or set a
    /// single cell (copy-on-write).
    ///
    /// Usage::
    ///
    ///     df.iat[0, 3]        # the cell at row 0, column 3
    ///     df.iat[0, 3] = 1.5  # set it
    #[getter]
    fn iat(slf: Bound<'_, Self>) -> DataFrameIat {
        DataFrameIat {
            parent: slf.unbind(),
        }
    }

    /// Fast scalar access by label + column name: ``df.at[label, col]`` to get
    /// or set a single cell (copy-on-write).
    ///
    /// Usage::
    ///
    ///     df.at['2021-01-04', 'close']         # one cell
    ///     df.at['2021-01-04', 'close'] = 100.0 # set it
    #[getter]
    fn at(slf: Bound<'_, Self>) -> DataFrameAt {
        DataFrameAt {
            parent: slf.unbind(),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.height()
    }

    /// `name in df` — whether a column exists (alias-aware).
    fn __contains__(&self, key: &str) -> bool {
        self.inner.has_column(key)
    }

    /// `for x in df` — iterate the column names (pandas semantics).
    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let names = PyList::new(py, self.inner.names())?;
        Ok(names.try_iter()?.into_any().unbind())
    }

    /// Guard the ambiguous `if df:` footgun: a DataFrame has no single truth
    /// value (pandas-style).
    fn __bool__(&self) -> PyResult<bool> {
        Err(PyValueError::new_err(
            "The truth value of a DataFrame is ambiguous — use len(df) or an explicit reduction",
        ))
    }

    /// Element-wise `==` -> a bool DataFrame (pandas semantics), not identity. The
    /// operand is another DataFrame (same columns + shared index) or a scalar.
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        self.compare(other, CmpOp::Eq)
    }

    /// Element-wise `!=` -> a bool DataFrame.
    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        self.compare(other, CmpOp::Ne)
    }

    /// First `n` rows (pandas `head`).
    #[pyo3(signature = (n = 5))]
    fn head(&self, n: usize) -> PyDataFrame {
        PyDataFrame::plain(self.inner.slice(0, n.min(self.inner.height())))
    }

    /// Last `n` rows (pandas `tail`).
    #[pyo3(signature = (n = 5))]
    fn tail(&self, n: usize) -> PyDataFrame {
        let h = self.inner.height();
        PyDataFrame::plain(self.inner.slice(h.saturating_sub(n), h))
    }

    /// Per-column count of non-missing values (pandas `count`) -> a Series indexed
    /// by column name (`int64`), reading each column's validity.
    fn count(&self) -> PySeries {
        let names: Vec<String> = self.inner.names().to_vec();
        let counts: Vec<i64> = self.inner.columns().iter().map(|c| c.count() as i64).collect();
        PySeries {
            inner: Series::new(None, Column::i64(counts), Arc::new(Index::str(names))),
        }
    }

    /// Per-column dtypes as `{name: dtype_str}` (pandas `dtypes`).
    #[getter]
    fn dtypes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            d.set_item(name, col.dtype().to_string())?;
        }
        Ok(d)
    }

    /// Drop rows containing missing values, across every dtype (via the column
    /// validity). `how='any'` (default) drops a row if any column is missing
    /// there; `how='all'` only if every column is missing. An invalid `how`
    /// raises `ValueError`.
    #[pyo3(signature = (how = "any"))]
    fn dropna(&self, how: &str) -> PyResult<PyDataFrame> {
        if how != "any" && how != "all" {
            return Err(PyValueError::new_err(format!(
                "dropna: invalid `how` {how:?} (expected 'any' or 'all')"
            )));
        }
        let cols = self.inner.columns();
        let total = cols.len();
        let keep: Vec<usize> = (0..self.inner.height())
            .filter(|&i| {
                let nan = cols.iter().filter(|c| !c.is_valid(i)).count();
                match how {
                    "all" => nan < total.max(1),
                    _ => nan == 0,
                }
            })
            .collect();
        Ok(PyDataFrame::plain(take_frame(&self.inner, &keep)))
    }

    /// Replace missing values with `value` in every column (pandas `fillna`),
    /// delegating to the per-column validity-aware `Column::fillna` so int / bool
    /// holes are filled dtype-preserving (like the Series version), not just float
    /// NaN. A `str` / `datetime` column with a missing cell raises a `TypeError`
    /// (a numeric fill cannot apply; volas has no `object` dtype) — a dense
    /// (no-hole) str / datetime column is untouched. For directional fill use
    /// `ffill` / `bfill` (pandas 3.0 removed `fillna(method=)`).
    fn fillna(&self, value: f64) -> PyResult<PyDataFrame> {
        let cols: Vec<Column> = self
            .inner
            .columns()
            .iter()
            .map(|c| c.fillna(value))
            .collect::<volas_core::Result<_>>()
            .map_err(pyerr)?;
        self.with_columns(cols)
    }

    /// Forward-fill missing cells in every column (pandas `ffill`), dtype-aware.
    fn ffill(&self) -> PyResult<PyDataFrame> {
        self.fill_dir(true)
    }

    /// Backward-fill missing cells in every column (pandas `bfill`), dtype-aware.
    fn bfill(&self) -> PyResult<PyDataFrame> {
        self.fill_dir(false)
    }

    /// Round each float column to `decimals` places (pandas `round`), banker's
    /// rounding; non-float columns are unchanged.
    #[pyo3(signature = (decimals = 0))]
    fn round(&self, decimals: i32) -> PyResult<PyDataFrame> {
        // Round numeric columns dtype-preservingly (banker's f64, integer-exact
        // i64); leave bool / str / datetime untouched, like pandas df.round.
        let cols: Vec<Column> = self
            .inner
            .columns()
            .iter()
            .map(|c| {
                if c.dtype().is_numeric() {
                    c.round(decimals)
                } else {
                    Ok(c.clone())
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(pyerr)?;
        self.with_columns(cols)
    }

    // --- column-wise numeric transforms (-> a new frame, dtype-preserving per
    // column, pandas df.cumsum() etc.). cumulatives / abs / clip keep dtype;
    // diff / shift / rank are always float. -------------------------------------

    /// Column-wise cumulative sum (pandas `cumsum`), dtype-preserving.
    fn cumsum(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.cumsum())
    }
    /// Column-wise cumulative maximum (pandas `cummax`).
    fn cummax(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.cummax())
    }
    /// Column-wise cumulative minimum (pandas `cummin`).
    fn cummin(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.cummin())
    }
    /// Column-wise cumulative product (pandas `cumprod`).
    fn cumprod(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.cumprod())
    }
    /// Column-wise absolute value (pandas `abs`), dtype-preserving.
    fn abs(&self) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.abs())
    }
    /// Column-wise clip into `[lower, upper]` (pandas `clip`), dtype-preserving.
    #[pyo3(signature = (lower = None, upper = None))]
    fn clip(&self, lower: Option<f64>, upper: Option<f64>) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.clip(lower, upper))
    }
    /// Column-wise discrete difference (pandas `diff`), dtype-preserving; the gap
    /// is missing (`volas.NA` for int/bool).
    #[pyo3(signature = (n = 1))]
    fn diff(&self, n: isize) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.diff(n))
    }
    /// Column-wise shift by `n` rows (pandas `shift`), dtype-preserving; the
    /// vacated cells are missing (`volas.NA` for int/bool).
    #[pyo3(signature = (n = 1))]
    fn shift(&self, n: isize) -> PyResult<PyDataFrame> {
        self.map_cols(|s| Ok(s.shift(n)))
    }
    /// Column-wise rank (pandas `rank`); always float.
    #[pyo3(signature = (method = "average", ascending = true, pct = false))]
    fn rank(&self, method: &str, ascending: bool, pct: bool) -> PyResult<PyDataFrame> {
        self.map_cols(|s| s.rank(method, ascending, pct))
    }

    // --- column-wise reductions (-> a Series indexed by column name; numeric
    // columns only, pandas df.sem() etc.). -------------------------------------

    /// Per-column standard error of the mean (pandas `sem`).
    fn sem(&self) -> PySeries {
        self.reduce_cols(|c| stats::sem(&c.to_f64_vec()))
    }
    /// Per-column unbiased skewness (pandas `skew`).
    fn skew(&self) -> PySeries {
        self.reduce_cols(|c| stats::skew(&c.to_f64_vec()))
    }
    /// Per-column unbiased excess kurtosis (pandas `kurt`).
    fn kurt(&self) -> PySeries {
        self.reduce_cols(|c| stats::kurt(&c.to_f64_vec()))
    }

    /// Per-column summary statistics over the numeric columns (pandas `describe`):
    /// a frame indexed by `count / mean / std / min / 25% / 50% / 75% / max`.
    fn describe(&self) -> PyResult<PyDataFrame> {
        let mut names = Vec::new();
        let mut cols = Vec::new();
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            if col.dtype().is_numeric() {
                let s = PySeries {
                    inner: Series::new(Some(name.clone()), col.clone(), Arc::clone(self.inner.index())),
                };
                names.push(name.clone());
                cols.push(s.describe()?.inner.data);
            }
        }
        DataFrame::new(names, cols, Some(Index::str(describe_labels())))
            .map(PyDataFrame::plain)
            .map_err(pyerr)
    }

    /// Pairwise Pearson correlation matrix over the numeric columns (pandas
    /// `corr`): a square frame indexed and labelled by those column names.
    fn corr(&self) -> PyResult<PyDataFrame> {
        self.corr_cov(stats::corr)
    }

    /// Pairwise sample covariance matrix over the numeric columns (pandas `cov`).
    fn cov(&self) -> PyResult<PyDataFrame> {
        self.corr_cov(stats::cov)
    }

    /// pandas `DataFrame.where`: keep each cell where `cond` is True, else `other`
    /// (default NaN). `cond` is a same-shape boolean frame (e.g. from `isna`);
    /// columns are taken as float. The inverse is `mask`.
    #[pyo3(name = "where", signature = (cond, other = None))]
    fn where_(&self, cond: &PyDataFrame, other: Option<f64>) -> PyResult<PyDataFrame> {
        self.where_mask(cond, other, true)
    }

    /// pandas `DataFrame.mask`: replace each cell with `other` where `cond` is
    /// True, keep it elsewhere — the inverse of `where`.
    #[pyo3(signature = (cond, other = None))]
    fn mask(&self, cond: &PyDataFrame, other: Option<f64>) -> PyResult<PyDataFrame> {
        self.where_mask(cond, other, false)
    }

    /// Boolean mask of missing cells (every dtype via the column validity) -> a
    /// bool DataFrame (pandas `isna`).
    fn isna(&self) -> PyResult<PyDataFrame> {
        self.mask_na(true)
    }

    /// Boolean mask of present (non-NaN) cells -> a bool DataFrame (pandas `notna`).
    fn notna(&self) -> PyResult<PyDataFrame> {
        self.mask_na(false)
    }

    /// Sort rows by index label (pandas `sort_index`).
    #[pyo3(signature = (ascending = true))]
    fn sort_index(&self, ascending: bool) -> PyDataFrame {
        let perm = self.inner.index().argsort(ascending);
        PyDataFrame::plain(take_frame(&self.inner, &perm))
    }

    /// Move the row index into an `'index'` column and restore a RangeIndex
    /// (pandas `reset_index`); `drop=True` discards the old index.
    #[pyo3(signature = (drop = false))]
    fn reset_index(&self, drop: bool) -> PyResult<PyDataFrame> {
        let h = self.inner.height();
        let (names, columns): (Vec<String>, Vec<Column>) = if drop {
            (self.inner.names().to_vec(), self.inner.columns().to_vec())
        } else {
            // Restore the index's name as the new column label (pandas parity);
            // an unnamed index falls back to "index".
            let label = self
                .inner
                .index()
                .name()
                .unwrap_or("index")
                .to_string();
            let mut names = vec![label];
            names.extend(self.inner.names().iter().cloned());
            let mut cols = vec![self.inner.index().to_column()];
            cols.extend(self.inner.columns().iter().cloned());
            (names, cols)
        };
        Ok(PyDataFrame::plain(
            DataFrame::new(names, columns, Some(Index::range(h))).map_err(pyerr)?,
        ))
    }

    /// `df[key] = value`. With a column name, add or replace that column —
    /// `value` may be a scalar (broadcast), a 1-D array / list, or a Series
    /// (positional, length must equal the frame height). With a boolean mask and
    /// a scalar fill, assign by mask: a boolean Series / array sets whole rows
    /// (`df[df['a'] > 0] = 0`), a boolean frame sets cells (`df[df.isna()] = 0`).
    /// Copy-on-write: a prior `copy()` is unaffected.
    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        // Boolean-mask assignment with a scalar fill: df[mask] = v
        if let Some(mask) = bool_mask_key(key)? {
            return self.assign_row_mask(&mask, value);
        }
        if let Ok(cond) = key.extract::<PyRef<PyDataFrame>>() {
            return self.assign_cell_mask(&cond, value);
        }
        // Column assignment: df[name] = value
        let name: String = key.extract().map_err(|_| {
            PyTypeError::new_err("DataFrame key must be a column name or a boolean mask")
        })?;
        let h = self.inner.height();
        let col = if let Ok(s) = value.extract::<PyRef<PySeries>>() {
            s.inner.data.clone()
        } else if let Ok(b) = value.extract::<bool>() {
            Column::bool(vec![b; h])
        } else if let Ok(scalar) = value.extract::<f64>() {
            Column::f64(vec![scalar; h])
        } else {
            pyany_to_column(value)?
        };
        // Overwriting an EXISTING column may invalidate any cached indicator derived
        // from it (e.g. `df['close'] = …` stales `ma:20`); mark those for recompute on
        // next access. Adding a brand-new column cannot affect existing caches.
        let existed = self.inner.has_column(&name);
        self.inner.set_column(&name, col).map_err(pyerr)?;
        if existed {
            self.inner.invalidate_computed_on_write(&name);
        }
        Ok(())
    }

    // `df[key]` — column name / indicator directive / list / boolean mask /
    // slice. The user-facing usage lives in the class docstring (pyo3 implements
    // `__getitem__` as a type slot and does not surface its doc comment).
    fn __getitem__(&mut self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // boolean mask (Series or numpy)
        if let Ok(s) = key.extract::<PyRef<PySeries>>() {
            if let Column::Bool(mask, _) = &s.inner.data {
                let sub = self.inner.filter_mask(mask).map_err(pyerr)?;
                return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
            }
        }
        if let Ok(arr) = key.extract::<PyReadonlyArray1<bool>>() {
            let sub = self.inner.filter_mask(arr.as_slice()?).map_err(pyerr)?;
            return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
        }
        // boolean mask as a plain Python list (df[[True, False, ...]]). An empty
        // list is an empty column projection, not a mask, so it falls through.
        if let Ok(mask) = key.extract::<Vec<bool>>() {
            if !mask.is_empty() {
                if mask.len() != self.inner.height() {
                    return Err(PyIndexError::new_err(format!(
                        "boolean index has wrong length: {} instead of {}",
                        mask.len(),
                        self.inner.height()
                    )));
                }
                let sub = self.inner.filter_mask(&mask).map_err(pyerr)?;
                return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
            }
        }
        // label / positional slice: df[:'date'], df[1:5]
        if let Ok(slice) = key.downcast::<PySlice>() {
            let sub = slice_frame(&self.inner, slice)?;
            return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
        }
        // column name or directive
        if let Ok(name) = key.extract::<String>() {
            // Existing column — a real column, or a cached directive: refresh its
            // stale tail if cached (no-op for a plain data column), then return.
            if self.inner.has_column(&name) {
                self.refresh_computed(Some(&name))?;
                let col = self.inner.column(&name).map_err(pyerr)?.clone();
                return Ok(Py::new(py, self.wrap_series(name, col))?.into_any());
            }
            // A directive: materialize (auto-cache) under its canonical name; on
            // later access its stale tail is refreshed incrementally, so the
            // result is always fresh AND cheap (O(lookback), not O(n)).
            let node = parse(&name).map_err(syntax_err)?;
            let canonical = volas_directive::stringify(&node);
            if self.inner.has_column(&canonical) {
                self.refresh_computed(Some(&canonical))?;
            } else {
                let col = execute(&self.inner, &node).map_err(value_err)?;
                let lookback = volas_directive::lookback::lookback(&node);
                // Capture the recursive state (if this directive supports an O(new-rows)
                // resume) BEFORE moving the column in, so a later append/fulfill can
                // continue without a full recompute. `None` for non-resumable directives.
                let state = volas_directive::exec::initial_state(&self.inner, &node, &col);
                self.inner.set_column(&canonical, col).map_err(pyerr)?;
                self.inner
                    .set_computed(&canonical, canonical.clone(), lookback);
                self.inner.set_computed_state(&canonical, state);
            }
            let col = self.inner.column(&canonical).map_err(pyerr)?.clone();
            return Ok(Py::new(py, self.wrap_series(canonical, col))?.into_any());
        }
        // list of names / directives
        if let Ok(list) = key.extract::<Vec<String>>() {
            let mut cols = Vec::with_capacity(list.len());
            for n in &list {
                let col = if self.inner.has_column(n) {
                    self.inner.column(n).map_err(pyerr)?.clone()
                } else {
                    let node = parse(n).map_err(syntax_err)?;
                    execute(&self.inner, &node).map_err(value_err)?
                };
                cols.push(col);
            }
            let idx = (*self.inner.index().as_ref()).clone();
            let df = DataFrame::new(list, cols, Some(idx)).map_err(pyerr)?;
            return Ok(Py::new(py, PyDataFrame::plain(df))?.into_any());
        }
        Err(PyKeyError::new_err(
            "key must be a column name, directive, list, boolean mask, or slice",
        ))
    }

    /// Evaluate an indicator directive and return its values as a NumPy array.
    ///
    /// Unlike ``df['ma:5']`` (which returns a Series and caches the column),
    /// ``exec`` returns the raw array; pass ``create_column=True`` to also cache
    /// it on the frame under its canonical name.
    ///
    /// Args:
    ///     directive (str): the directive, e.g. ``'macd'``, ``'boll.upper:20'``,
    ///         ``'close > open'``.
    ///     create_column (bool): if True, materialize and cache the result as a
    ///         column (default False).
    ///
    /// Usage::
    ///
    ///     df.exec('ma:5')               # ndarray of SMA(5)
    ///     df.exec('kdj.j', create_column=True)  # also caches the column
    ///
    /// Returns:
    ///     numpy.ndarray
    #[pyo3(signature = (directive, create_column = false))]
    fn exec<'py>(
        &mut self,
        py: Python<'py>,
        directive: &str,
        create_column: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.inner.has_column(directive) {
            let col = self.inner.column(directive).map_err(pyerr)?.clone();
            return Ok(column_to_numpy(py, &col));
        }
        let node = parse(directive).map_err(syntax_err)?;
        if create_column {
            // Materialize + cache under the canonical name, exactly like `df[directive]`.
            let canonical = volas_directive::stringify(&node);
            if self.inner.has_column(&canonical) {
                self.refresh_computed(Some(&canonical))?;
            } else {
                let col = execute(&self.inner, &node).map_err(value_err)?;
                let lookback = volas_directive::lookback::lookback(&node);
                let state = volas_directive::exec::initial_state(&self.inner, &node, &col);
                self.inner.set_column(&canonical, col).map_err(pyerr)?;
                self.inner
                    .set_computed(&canonical, canonical.clone(), lookback);
                self.inner.set_computed_state(&canonical, state);
            }
            let col = self.inner.column(&canonical).map_err(pyerr)?.clone();
            Ok(column_to_numpy(py, &col))
        } else {
            let col = execute(&self.inner, &node).map_err(value_err)?;
            Ok(column_into_numpy(py, col))
        }
    }

    /// Gets a column from the frame by name (alias-aware), as a Series.
    ///
    /// Args:
    ///     key (str): the column name.
    ///
    /// Returns:
    ///     Series
    fn get_column(&self, key: &str) -> PyResult<PySeries> {
        let col = self.inner.column(key).map_err(pyerr)?.clone();
        Ok(self.wrap_series(key.to_string(), col))
    }

    /// A copy of the frame — preserving the cached directive columns / cursor and
    /// (for a tf-aware frame) the cumulation state, so the copy keeps folding.
    fn copy(&self) -> PyDataFrame {
        PyDataFrame {
            inner: self.inner.clone(),
            tf: self.tf.clone(),
        }
    }

    /// Convert to a `pandas.DataFrame`. pandas is imported lazily (only here), so
    /// volas stays pandas-free at import.
    #[pyo3(signature = (dtype_backend = "numpy"))]
    fn to_pandas<'py>(&self, py: Python<'py>, dtype_backend: &str) -> PyResult<Bound<'py, PyAny>> {
        ensure_fresh(&self.inner)?;
        // 'numpy' (default): an int/bool column with NA exports as float64+NaN, the
        // most ecosystem-compatible form. 'numpy_nullable': a faithful, lossless
        // masked Int64 / boolean. Mirrors pandas' own `dtype_backend`.
        let nullable = match dtype_backend {
            "numpy" => false,
            "numpy_nullable" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "dtype_backend must be 'numpy' or 'numpy_nullable', got {other:?}"
                )))
            }
        };
        let pd = py.import("pandas")?;
        let data = PyDict::new(py);
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            data.set_item(name, column_to_pandas(py, &pd, col, nullable)?)?;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("index", index_to_numpy(py, self.inner.index())?)?;
        let pdf = pd.call_method("DataFrame", (data,), Some(&kwargs))?;
        // A tz-aware frame exports a UTC-naive datetime64 index (index_to_numpy); restore the
        // display zone so the pandas index is tz-aware — a faithful round-trip with from_pandas.
        if let Some(tz) = self.tz() {
            let aware = pdf
                .getattr("index")?
                .call_method1("tz_localize", ("UTC",))?
                .call_method1("tz_convert", (&tz,))?;
            pdf.setattr("index", aware)?;
        }
        // Carry the index name onto the pandas index (pandas parity).
        if let Some(name) = self.inner.index().name() {
            let renamed = pdf.getattr("index")?.call_method1("rename", (name,))?;
            pdf.setattr("index", renamed)?;
        }
        Ok(pdf)
    }

    /// Write the frame as CSV (pandas-subset). With no `path`, returns the CSV
    /// string. Datetime columns are written as formatted strings (round-trips
    /// with `read_csv`).
    #[pyo3(signature = (path = None, sep = ",", index = true, header = true, na_rep = "", columns = None, float_format = None))]
    fn to_csv(
        &self,
        path: Option<String>,
        sep: &str,
        index: bool,
        header: bool,
        na_rep: &str,
        columns: Option<Vec<String>>,
        float_format: Option<&str>,
    ) -> PyResult<Option<String>> {
        ensure_fresh(&self.inner)?;
        let ff = parse_ff(float_format)?;
        let names = self.inner.names();
        let positions: Vec<usize> = match &columns {
            Some(cols) => cols
                .iter()
                .map(|n| {
                    self.inner
                        .column_pos(n)
                        .ok_or_else(|| PyKeyError::new_err(format!("column \"{n}\" not found")))
                })
                .collect::<PyResult<_>>()?,
            None => (0..self.inner.width()).collect(),
        };
        let mut out = String::new();
        if header {
            if index {
                // pandas writes the index name, or an empty field for an unnamed index.
                out.push_str(self.inner.index().name().unwrap_or(""));
                out.push_str(sep);
            }
            let hdr: Vec<&str> = positions.iter().map(|&j| names[j].as_str()).collect();
            out.push_str(&hdr.join(sep));
            out.push('\n');
        }
        for i in 0..self.inner.height() {
            if index {
                out.push_str(&index_label_csv(self.inner.index(), i));
                out.push_str(sep);
            }
            let cells: Vec<String> = positions
                .iter()
                .map(|&j| cell_to_csv(&self.inner.columns()[j], i, na_rep, ff))
                .collect();
            out.push_str(&cells.join(sep));
            out.push('\n');
        }
        match path {
            Some(p) => {
                std::fs::write(&p, out).map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(None)
            }
            None => Ok(Some(out)),
        }
    }

    /// Drop rows by index label (`axis=0`) or columns by name (`axis=1`) —
    /// returns a new DataFrame. Row labels are parsed against the index kind.
    #[pyo3(signature = (labels, axis = 0))]
    fn drop(&self, py: Python<'_>, labels: Vec<Py<PyAny>>, axis: i64) -> PyResult<PyDataFrame> {
        if axis == 1 {
            let drop_names: Vec<String> = labels
                .iter()
                .map(|l| l.bind(py).extract::<String>())
                .collect::<PyResult<_>>()?;
            let keep: Vec<String> = self
                .inner
                .names()
                .iter()
                .filter(|n| !drop_names.contains(n))
                .cloned()
                .collect();
            return Ok(PyDataFrame::plain(self.inner.select(&keep).map_err(pyerr)?));
        }
        let index = self.inner.index();
        let targets: Vec<Label> = labels
            .iter()
            .map(|l| parse_label(l.bind(py), index))
            .collect::<PyResult<_>>()?;
        let positions: Vec<usize> = (0..self.inner.height())
            .filter(|&i| !targets.contains(&index.label_at(i)))
            .collect();
        Ok(PyDataFrame::plain(take_frame(&self.inner, &positions)))
    }

    /// Append the rows of another DataFrame or a single Row **in place** and
    /// return the same frame (amortized O(1), like ``list.append`` — the live
    /// single-bar hot path, no full-column copy).
    ///
    /// On a **time_frame** frame (see the constructor / ``cumulate``) the rows
    /// are treated as *finer* bars and folded into the current period: a bar in
    /// the open period updates the forming last row (``df.iloc[-1]``), a bar in a
    /// new period rolls over into a fresh row. A re-sent forming bar (same
    /// timestamp) updates rather than double-counts.
    ///
    /// Missing columns are NaN-padded; cached directive columns go stale until
    /// ``fulfill()``. A snapshot taken via ``copy()`` / ``iloc`` is unaffected
    /// (it pays one copy-on-write the next time *it* is appended to).
    ///
    /// Args:
    ///     other (DataFrame | Row): the rows to append (fine bars if tf-aware).
    ///
    /// Usage::
    ///
    ///     df.append(bar)           # append / fold one bar
    ///     df.append(other_frame)   # append / fold many bars
    ///
    /// Returns:
    ///     DataFrame: ``self`` (enabling chaining).
    fn append<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        if let Ok(df) = other.extract::<PyRef<PyDataFrame>>() {
            if slf.as_ptr() == other.as_ptr() {
                // `df.append(df)` needs an owned snapshot before taking `self` mutably.
                let other_inner = df.inner.clone();
                drop(df);
                let mut me = slf.borrow_mut();
                if me.tf.is_some() {
                    me.fold_append(&other_inner)?;
                } else {
                    me.inner.append(&other_inner).map_err(pyerr)?;
                }
                return Ok(slf);
            }
            // Normal live path: append a distinct one-row frame without cloning it.
            let mut me = slf.borrow_mut();
            if me.tf.is_some() {
                me.fold_append(&df.inner)?;
            } else {
                me.inner.append(&df.inner).map_err(pyerr)?;
            }
            return Ok(slf);
        }
        if let Ok(row) = other.extract::<PyRef<PyRow>>() {
            let mut me = slf.borrow_mut();
            if me.tf.is_some() {
                me.fold_append(&row.inner)?;
            } else {
                me.inner.append(&row.inner).map_err(pyerr)?;
            }
            return Ok(slf);
        }
        Err(PyTypeError::new_err("append expects a DataFrame or Row"))
    }

    /// The frame as a 2-D NumPy array (pandas `to_numpy`). With no `dtype`: a fast
    /// `float64` matrix when every column is numeric, else a lossless `object`
    /// array (string columns kept, not NaN-poisoned). `dtype` casts (e.g.
    /// `'float32'`); requesting a float over a string column raises.
    #[pyo3(signature = (dtype = None))]
    fn to_numpy<'py>(&self, py: Python<'py>, dtype: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        ensure_fresh(&self.inner)?;
        let has_str = self
            .inner
            .columns()
            .iter()
            .any(|c| matches!(c, Column::Str(_, _)));
        let (h, w) = (self.inner.height(), self.inner.width());

        if let Some(dt) = dtype {
            let floaty = dt.contains("float") || dt == "f32" || dt == "f64" || dt == "double";
            if has_str && floaty {
                return Err(PyValueError::new_err(format!(
                    "cannot convert a string column to {dt}"
                )));
            }
            let (data, h, w) = self.inner.to_row_major_f64();
            let arr = ndarray::Array2::from_shape_vec((h, w), data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_pyarray(py);
            return Ok(arr.call_method1("astype", (dt,))?);
        }

        if !has_str {
            // fast all-numeric path: a float64 matrix
            let (data, h, w) = self.inner.to_row_major_f64();
            let arr = ndarray::Array2::from_shape_vec((h, w), data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(arr.into_pyarray(py).into_any());
        }

        // mixed/string frame: a lossless 2-D object array (each cell typed)
        let rows = PyList::empty(py);
        for i in 0..h {
            let row = PyList::empty(py);
            for col in self.inner.columns() {
                row.append(scalar_to_py(py, col, i))?;
            }
            rows.append(row)?;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "object")?;
        let _ = w;
        Ok(py
            .import("numpy")?
            .call_method("array", (rows,), Some(&kwargs))?)
    }

    /// Value equality (same columns + index + values, `NaN == NaN`).
    fn equals(&self, other: &PyDataFrame) -> bool {
        self.inner.equals(&other.inner)
    }

    /// Resample to a coarser timeframe (OHLCV cumulation / down-sampling),
    /// returning a **tf-aware** DataFrame you can keep ``append``-ing finer bars
    /// into (the forming period is the live last row).
    ///
    /// Requires a DatetimeIndex. Each column is aggregated with a sensible
    /// default (open=first, high=max, low=min, close=last, volume=sum); override
    /// per column via ``cumulators``. If the source already has a ``time_frame``,
    /// the target must be a whole multiple of it (e.g. 5m→15m, not 5m→7m, and not
    /// a week/3-day into a month); cumulating to the *same* frame is a ``copy()``.
    ///
    /// Args:
    ///     time_frame (str | TimeFrame): the target bucket, e.g. ``'1d'``,
    ///         ``'15m'``, ``'1w'``.
    ///     cumulators (dict[str, str], optional): per-column aggregator
    ///         overrides, e.g. ``{'volume': 'sum', 'close': 'last'}``.
    ///
    /// Usage::
    ///
    ///     daily = df.cumulate('1d')          # a tf-aware 1d frame
    ///     daily.append(intraday_bar)         # folds into the forming day
    ///
    /// Returns:
    ///     DataFrame: the resampled, tf-aware frame.
    #[pyo3(signature = (time_frame, cumulators = None))]
    fn cumulate(
        &self,
        time_frame: &Bound<'_, PyAny>,
        cumulators: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let target = resolve_time_frame(time_frame)?;
        if let Some(tfs) = &self.tf {
            // Same frame is a no-op resample == copy() (keeps the cursor & state).
            if target == tfs.time_frame {
                return Ok(self.copy());
            }
            if !tfs.time_frame.can_coarsen(target) {
                return Err(PyValueError::new_err(format!(
                    "cannot cumulate {} -> {}: the target is not a whole multiple of the source frame",
                    tfs.time_frame.label(),
                    target.label()
                )));
            }
        }
        let spec = build_agg_spec(cumulators)?;
        let mut cum = Cumulator::new(target, spec.clone());
        cum.append(&self.inner).map_err(pyerr)?;
        let frame = cum.frame().map_err(pyerr)?;
        // The result is a fresh frame (no cached directive columns -> cursor 0)
        // that carries the open period's fine bars so further appends fold in.
        Ok(PyDataFrame {
            inner: frame,
            tf: Some(TfState {
                time_frame: target,
                cumulators: spec,
                open: cum.open_clone(),
            }),
        })
    }

    /// Rename columns (pandas `rename(columns={old: new})`), returning a new
    /// frame.
    #[pyo3(signature = (columns))]
    fn rename(&self, columns: &Bound<'_, PyDict>) -> PyResult<PyDataFrame> {
        let mut mapping = HashMap::new();
        for (k, v) in columns.iter() {
            mapping.insert(k.extract::<String>()?, v.extract::<String>()?);
        }
        Ok(PyDataFrame::plain(
            self.inner.rename(&mapping).map_err(pyerr)?,
        ))
    }

    /// Move a column into the row index (pandas `set_index(col)`), returning a
    /// new frame. A datetime / int / string column becomes the matching index.
    #[pyo3(signature = (keys))]
    fn set_index(&self, keys: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::plain(
            self.inner.set_index(keys).map_err(pyerr)?,
        ))
    }

    /// Cast columns to new dtypes (pandas `astype({col: dtype})`), returning a
    /// new frame.
    fn astype(&self, dtypes: &Bound<'_, PyDict>) -> PyResult<PyDataFrame> {
        let mut df = self.inner.clone();
        let mut mapping = HashMap::new();
        for (k, v) in dtypes.iter() {
            let name = k.extract::<String>()?;
            let dt = v.extract::<String>()?;
            if let Some(unit) = datetime_unit_of(&dt) {
                // datetime target: parse a string column, or scale a numeric epoch
                // column by the dtype's unit (truncating, like a NumPy
                // `datetime64[unit]` cast).
                let col = df.column(&name).map_err(pyerr)?.clone();
                let converted = match &col {
                    Column::Datetime(_) | Column::Str(_, _) => col.to_datetime().map_err(pyerr)?,
                    _ => col.epoch_to_datetime(unit).map_err(pyerr)?,
                };
                df.set_column(&name, converted).map_err(pyerr)?;
            } else {
                mapping.insert(name, parse_dtype(&dt)?);
            }
        }
        if !mapping.is_empty() {
            df = df.astype(&mapping).map_err(pyerr)?;
        }
        Ok(PyDataFrame::plain(df))
    }

    /// Define a column / directive alias: `as_name` resolves to `src_name`
    /// everywhere a column is looked up (mutates in place, pandas-like).
    fn alias(&mut self, as_name: &str, src_name: &str) -> PyResult<()> {
        self.inner = self.inner.with_alias(as_name, src_name).map_err(pyerr)?;
        Ok(())
    }

    /// Refresh the stale tail of every materialized (auto-cached) directive
    /// column at once (e.g. before a bulk `to_numpy` / row read). Per-column
    /// access already auto-refreshes; this is the batch form. In place,
    /// incremental — O(lookback + new rows) per column, not O(n).
    fn fulfill(&mut self) -> PyResult<()> {
        self.refresh_computed(None)
    }

    /// pandas-style aligned-table repr: a left-justified index column + right-
    /// justified data columns, truncating to 5 head + 5 tail rows past 60
    /// (`display.max_rows` / `min_rows`) with a `[N rows x M columns]` footer.
    /// `str` and `repr` are identical.
    fn __repr__(&self) -> PyResult<String> {
        ensure_fresh(&self.inner)?;
        let truncate = if self.inner.height() > 60 { Some(5) } else { None };
        let opts = DisplayOpts {
            header: true,
            index: true,
            na_rep: NA_REPR,
            float_format: None,
            dimensions: Dimensions::OnTruncate,
            truncate,
        };
        let cols: Vec<usize> = (0..self.inner.width()).collect();
        Ok(render_frame(&self.inner, &cols, &opts))
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }

    /// Render the whole frame as text (pandas `DataFrame.to_string`), implementing
    /// the core parameters. No truncation by default; `max_rows` truncates to 5
    /// head + 5 tail (or `min_rows`). Legacy / non-applicable pandas params
    /// (`sparsify`, `index_names`, `col_space`, `justify`, `formatters`,
    /// `line_width`, `encoding`, `decimal`, `buf`) are intentionally omitted.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (columns = None, header = true, index = true, na_rep = NA_REPR, float_format = None, max_rows = None, min_rows = None, show_dimensions = false))]
    fn to_string(
        &self,
        columns: Option<Vec<String>>,
        header: bool,
        index: bool,
        na_rep: &str,
        float_format: Option<&str>,
        max_rows: Option<usize>,
        min_rows: Option<usize>,
        show_dimensions: bool,
    ) -> PyResult<String> {
        ensure_fresh(&self.inner)?;
        let ff = parse_ff(float_format)?;
        let col_pos: Vec<usize> = match &columns {
            Some(cols) => cols
                .iter()
                .map(|n| {
                    self.inner
                        .column_pos(n)
                        .ok_or_else(|| PyKeyError::new_err(format!("column \"{n}\" not found")))
                })
                .collect::<PyResult<_>>()?,
            None => (0..self.inner.width()).collect(),
        };
        let truncate = match max_rows {
            Some(m) if self.inner.height() > m => Some((min_rows.unwrap_or(m) / 2).max(1)),
            _ => None,
        };
        let opts = DisplayOpts {
            header,
            index,
            na_rep,
            float_format: ff,
            dimensions: if show_dimensions {
                Dimensions::Always
            } else {
                Dimensions::Never
            },
            truncate,
        };
        Ok(render_frame(&self.inner, &col_pos, &opts))
    }

    /// Rich HTML table for Jupyter (`_repr_html_`). pandas defines this only on
    /// DataFrame — a Series falls back to its text repr — so volas matches and
    /// exposes it on DataFrame alone.
    fn _repr_html_(&self) -> PyResult<String> {
        ensure_fresh(&self.inner)?;
        Ok(render_frame_html(&self.inner))
    }
}

/// `df.iloc[...]` positional indexer.
// --- indexer assignment helpers (PD-12) ------------------------------------

/// Whether `v` is a missing-value scalar: Python `None`, a `NaN` float, or the
/// `volas.NA` singleton. The one predicate every scalar boundary shares, so the
/// canonical `volas.NA` symbol (what `to_list()` returns) is usable wherever
/// `None` is — constructor, Series setitem, DataFrame indexers, mask assignment,
/// and `where` / `mask` `other`.
fn is_na_like_py(v: &Bound<'_, PyAny>) -> bool {
    let py = v.py();
    v.is_none() || v.is(na(py).bind(py)) || v.extract::<f64>().is_ok_and(|x| x.is_nan())
}

/// Build a length-1 [`Column`] from a Python scalar, coerced toward the target
/// column's dtype (so a string can land in a datetime column, etc.). An `I64`
/// target given a float yields an `F64` value — core then widens the column.
fn scalar_to_column(v: &Bound<'_, PyAny>, target: DType) -> PyResult<Column> {
    // `None` / `NaN` / `volas.NA` -> a typed single-cell NA (marks the position
    // missing while keeping the dtype), so `x[i] = None / nan / volas.NA` is
    // uniform across every dtype and every assignment surface.
    if is_na_like_py(v) {
        return Ok(Column::na_of(target, 1));
    }
    match target {
        DType::F64 => {
            let x = v
                .extract::<f64>()
                .map_err(|_| PyTypeError::new_err("expected a number"))?;
            Ok(Column::f64(vec![x]))
        }
        DType::I64 => {
            if let Ok(i) = v.extract::<i64>() {
                Ok(Column::i64(vec![i]))
            } else {
                let x = v
                    .extract::<f64>()
                    .map_err(|_| PyTypeError::new_err("expected a number"))?;
                Ok(Column::f64(vec![x]))
            }
        }
        DType::F32 => {
            let x = v
                .extract::<f64>()
                .map_err(|_| PyTypeError::new_err("expected a number"))?;
            Ok(Column::f32(vec![x as f32]))
        }
        DType::I32 => match v.extract::<i64>() {
            Ok(i) => match i32::try_from(i) {
                Ok(v32) => Ok(Column::i32(vec![v32])),
                Err(_) => Ok(Column::i64(vec![i])), // out of i32 range -> i64 (core widens)
            },
            Err(_) => {
                let x = v
                    .extract::<f64>()
                    .map_err(|_| PyTypeError::new_err("expected a number"))?;
                Ok(Column::f64(vec![x]))
            }
        },
        DType::Bool => {
            let b = v
                .extract::<bool>()
                .map_err(|_| PyTypeError::new_err("expected a bool"))?;
            Ok(Column::bool(vec![b]))
        }
        DType::Utf8 => {
            let s = v
                .extract::<String>()
                .map_err(|_| PyTypeError::new_err("expected a string"))?;
            Ok(Column::str(vec![s]))
        }
        DType::Datetime => Ok(Column::datetime(vec![parse_ts(v)?])),
    }
}

/// Assign a Python **scalar** into `col` at `positions`, via the shared
/// `scalar_to_column` + [`Column::scatter`] primitive — the single assignment
/// path behind Series setitem and DataFrame boolean-mask assignment (the
/// `.loc/.iloc/.at/.iat` indexers reach `scatter` through `assign_positions`).
///
/// A column with **no selected positions** is returned unchanged, so a typed fill
/// (a string into a str column, say) errors only when it actually targets a cell
/// of an incompatible column — the mixed-frame atomic rule: nothing is written
/// unless every targeted column accepts the value.
fn scatter_scalar(
    col: &Column,
    positions: &[usize],
    value: &Bound<'_, PyAny>,
) -> PyResult<Column> {
    if positions.is_empty() {
        return Ok(col.clone());
    }
    let src = scalar_to_column(value, col.dtype())?;
    col.scatter(positions, &src).map_err(pyerr)
}

/// Convert an array-like assignment value (list / NumPy array / `Series`) to a
/// [`Column`]; core coerces it toward the target dtype.
fn value_to_column(v: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(s) = v.extract::<PyRef<PySeries>>() {
        return Ok(s.inner.data.clone());
    }
    pyany_to_column(v)
}

/// Resolve a `df[...] = value` right-hand side for `n` selected rows: a scalar is
/// broadcast (length-1 column), an array-like must match `n`.
fn resolve_assignment(v: &Bound<'_, PyAny>, target: DType, n: usize) -> PyResult<Column> {
    // A Python str has `__len__` but is a scalar here; everything else with
    // `__len__` (list / ndarray / Series) is array-like.
    let is_str = v.extract::<String>().is_ok();
    let arraylike = !is_str && v.hasattr("__len__").unwrap_or(false);
    if arraylike {
        let col = value_to_column(v)?;
        if col.len() != n {
            return Err(PyValueError::new_err(format!(
                "cannot assign {} values to {n} selected rows",
                col.len()
            )));
        }
        Ok(col)
    } else {
        scalar_to_column(v, target)
    }
}

/// If `sel` is a boolean mask of length `height` (NumPy bool array, bool `Series`,
/// or `list[bool]`), return the selected row positions; else `None`.
fn as_bool_mask(sel: &Bound<'_, PyAny>, height: usize) -> Option<Vec<usize>> {
    let collect = |bits: &[bool]| -> Option<Vec<usize>> {
        (bits.len() == height).then(|| {
            bits.iter()
                .enumerate()
                .filter_map(|(i, &b)| b.then_some(i))
                .collect()
        })
    };
    if let Ok(a) = sel.extract::<PyReadonlyArray1<bool>>() {
        return a.as_slice().ok().and_then(collect);
    }
    if let Ok(ser) = sel.extract::<PyRef<PySeries>>() {
        if let Column::Bool(v, _) = &ser.inner.data {
            return collect(v);
        }
        return None;
    }
    if let Ok(v) = sel.extract::<Vec<bool>>() {
        return collect(&v);
    }
    None
}

/// Resolve an `iloc` row selector (int / slice / int-list / bool-mask) to row
/// positions.
fn iloc_positions(sel: &Bound<'_, PyAny>, height: usize) -> PyResult<Vec<usize>> {
    if let Some(pos) = as_bool_mask(sel, height) {
        return Ok(pos);
    }
    if let Ok(i) = sel.extract::<isize>() {
        return Ok(vec![norm_idx(i, height)?]);
    }
    if let Ok(slice) = sel.downcast::<PySlice>() {
        let info = slice.indices(height as isize)?;
        return Ok(strided(info.start, info.stop, info.step));
    }
    if let Ok(idxs) = sel.extract::<Vec<isize>>() {
        return idxs.into_iter().map(|i| norm_idx(i, height)).collect();
    }
    Err(PyTypeError::new_err(
        "iloc row selector must be an int, slice, int list, or boolean mask",
    ))
}

/// Resolve a `loc` row selector (bool-mask / label-slice / label / label-list) to
/// row positions.
fn loc_positions(sel: &Bound<'_, PyAny>, index: &Index, height: usize) -> PyResult<Vec<usize>> {
    if let Some(pos) = as_bool_mask(sel, height) {
        return Ok(pos);
    }
    if let Ok(slice) = sel.downcast::<PySlice>() {
        let (lo, hi) = label_bounds(slice, index)?;
        let (a, b) = index.label_slice(lo.as_ref(), hi.as_ref());
        return Ok((a..b).collect());
    }
    if let Ok(list) = sel.downcast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            let label = parse_label(&item, index)?;
            out.push(
                index
                    .position_of(&label)
                    .ok_or_else(|| PyKeyError::new_err("label not found"))?,
            );
        }
        return Ok(out);
    }
    let label = parse_label(sel, index)?;
    let pos = index
        .position_of(&label)
        .ok_or_else(|| PyKeyError::new_err("label not found"))?;
    Ok(vec![pos])
}

/// One axis of a 2-D `iloc` / `loc` get: a single scalar position (the axis is
/// reduced away, pandas-style) or a list of positions (the axis is kept).
enum AxisSel {
    One(usize),
    Many(Vec<usize>),
}

/// Resolve an `iloc` row axis: a bare int reduces the axis (`AxisSel::One`); a
/// slice / int-list / boolean mask keeps it (`AxisSel::Many`).
fn iloc_row_axis(sel: &Bound<'_, PyAny>, height: usize) -> PyResult<AxisSel> {
    if let Ok(i) = sel.extract::<isize>() {
        return Ok(AxisSel::One(norm_idx(i, height)?));
    }
    Ok(AxisSel::Many(iloc_positions(sel, height)?))
}

/// Resolve an `iloc` column axis (int / slice / int-list) to column positions.
fn iloc_col_axis(sel: &Bound<'_, PyAny>, width: usize) -> PyResult<AxisSel> {
    if let Ok(j) = sel.extract::<isize>() {
        return Ok(AxisSel::One(norm_idx(j, width)?));
    }
    if let Ok(slice) = sel.downcast::<PySlice>() {
        let info = slice.indices(width as isize)?;
        return Ok(AxisSel::Many(strided(info.start, info.stop, info.step)));
    }
    if let Ok(idxs) = sel.extract::<Vec<isize>>() {
        return Ok(AxisSel::Many(
            idxs.into_iter()
                .map(|j| norm_idx(j, width))
                .collect::<PyResult<_>>()?,
        ));
    }
    Err(PyTypeError::new_err(
        "iloc column selector must be an int, slice, or int list",
    ))
}

/// Resolve a `loc` row axis: a single label reduces the axis (`AxisSel::One`); a
/// label-slice / label-list / boolean mask keeps it (`AxisSel::Many`).
fn loc_row_axis(sel: &Bound<'_, PyAny>, index: &Index, height: usize) -> PyResult<AxisSel> {
    if as_bool_mask(sel, height).is_some()
        || sel.downcast::<PySlice>().is_ok()
        || sel.downcast::<PyList>().is_ok()
    {
        return Ok(AxisSel::Many(loc_positions(sel, index, height)?));
    }
    let label = parse_label(sel, index)?;
    let pos = index
        .position_of(&label)
        .ok_or_else(|| PyKeyError::new_err("label not found"))?;
    Ok(AxisSel::One(pos))
}

/// Resolve a `loc` column axis (name / name-list / inclusive name-slice) to
/// column positions.
fn loc_col_axis(sel: &Bound<'_, PyAny>, df: &DataFrame) -> PyResult<AxisSel> {
    let pos_of = |name: &str| {
        df.column_pos(name)
            .ok_or_else(|| PyKeyError::new_err(format!("column {name:?} not found")))
    };
    if let Ok(name) = sel.extract::<String>() {
        return Ok(AxisSel::One(pos_of(&name)?));
    }
    if let Ok(names) = sel.extract::<Vec<String>>() {
        return Ok(AxisSel::Many(
            names.iter().map(|n| pos_of(n)).collect::<PyResult<_>>()?,
        ));
    }
    if let Ok(slice) = sel.downcast::<PySlice>() {
        let start = slice.getattr("start")?;
        let stop = slice.getattr("stop")?;
        let lo = if start.is_none() {
            0
        } else {
            pos_of(&start.extract::<String>()?)?
        };
        let hi = if stop.is_none() {
            df.width().saturating_sub(1)
        } else {
            pos_of(&stop.extract::<String>()?)?
        };
        return Ok(AxisSel::Many((lo..=hi).collect()));
    }
    Err(PyTypeError::new_err(
        "loc column selector must be a name, name list, or name slice",
    ))
}

/// Project `df` onto `cols` (by position), carrying the index — the column-axis
/// counterpart of `DataFrame::take` (which selects rows).
fn project_cols(df: &DataFrame, cols: &[usize]) -> PyResult<DataFrame> {
    let names: Vec<String> = cols.iter().map(|&j| df.names()[j].clone()).collect();
    let data: Vec<Column> = cols.iter().map(|&j| df.columns()[j].clone()).collect();
    let idx = (*df.index().as_ref()).clone();
    DataFrame::new(names, data, Some(idx)).map_err(pyerr)
}

/// Build a 2-D `iloc` / `loc` get result from already-resolved row & column
/// positions, reproducing pandas's shape rules: scalar×scalar -> cell,
/// rows×col -> a column Series, row×cols -> the row (volas's 1-row frame), and
/// rows×cols -> a sub-frame.
fn select_2d(py: Python<'_>, df: &DataFrame, rows: AxisSel, cols: AxisSel) -> PyResult<Py<PyAny>> {
    match (rows, cols) {
        (AxisSel::One(i), AxisSel::One(j)) => Ok(np_scalar_to_py(py, &df.columns()[j], i)),
        (AxisSel::Many(r), AxisSel::One(j)) => {
            let sub = project_cols(df, &[j])?.take(&r);
            let name = sub.names()[0].clone();
            let col = sub.columns()[0].clone();
            let series = PySeries {
                inner: Series::new(Some(name), col, Arc::clone(sub.index())),
            };
            Ok(Py::new(py, series)?.into_any())
        }
        (AxisSel::One(i), AxisSel::Many(c)) => {
            let sub = project_cols(df, &c)?.take(&[i]);
            Ok(Py::new(py, PyRow { inner: sub })?.into_any())
        }
        (AxisSel::Many(r), AxisSel::Many(c)) => {
            let sub = project_cols(df, &c)?.take(&r);
            Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any())
        }
    }
}

/// Split a `df.loc[rows, col]` / `df.iloc[rows, col]` assignment key into its two
/// parts, with a clear error directing to the supported 2-tuple form.
fn split_row_col<'py>(
    key: &Bound<'py, PyAny>,
    accessor: &str,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let tup = key.downcast::<PyTuple>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{accessor} assignment needs a (rows, column) key, e.g. df.{accessor}[mask, 'col'] = value"
        ))
    })?;
    if tup.len() != 2 {
        return Err(PyTypeError::new_err(format!(
            "{accessor} assignment key must be (rows, column)"
        )));
    }
    Ok((tup.get_item(0)?, tup.get_item(1)?))
}

#[pyclass]
pub struct DataFrameILoc {
    parent: Py<PyDataFrame>,
}

#[pymethods]
impl DataFrameILoc {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let pf = self.parent.borrow(py);
        ensure_fresh(&pf.inner)?;
        // 2-D positional get: df.iloc[rows, cols], symmetric with __setitem__.
        if let Ok(tup) = key.downcast::<PyTuple>() {
            if tup.len() == 2 {
                let rows = iloc_row_axis(&tup.get_item(0)?, pf.inner.height())?;
                let cols = iloc_col_axis(&tup.get_item(1)?, pf.inner.width())?;
                return select_2d(py, &pf.inner, rows, cols);
            }
        }
        if let Ok(i) = key.extract::<isize>() {
            let i = norm_idx(i, pf.inner.height())?;
            return Ok(Py::new(py, row_at(&pf.inner, i))?.into_any());
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            let info = slice.indices(pf.inner.height() as isize)?;
            let sub = positional_slice(&pf.inner, &info);
            return Ok(Py::new(py, PyDataFrame::plain(sub))?.into_any());
        }
        // int-list / boolean-mask row selection -> sub-frame.
        let positions = iloc_positions(key, pf.inner.height())?;
        Ok(Py::new(py, PyDataFrame::plain(take_frame(&pf.inner, &positions)))?.into_any())
    }

    /// `df.iloc[i, j] = scalar` or `df.iloc[rows, j] = scalar | array` (positional;
    /// copy-on-write). `rows` is an int / slice / int-list / boolean mask; `j` is a
    /// column position.
    fn __setitem__(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut pf = self.parent.borrow_mut(py);
        ensure_fresh(&pf.inner)?;
        let (rows, col) = split_row_col(key, "iloc")?;
        let height = pf.inner.height();
        let j = norm_idx(col.extract::<isize>()?, pf.inner.width())?;
        let positions = iloc_positions(&rows, height)?;
        let target = pf.inner.columns()[j].dtype();
        let val = resolve_assignment(value, target, positions.len())?;
        pf.inner
            .assign_positions(j, &positions, &val)
            .map_err(pyerr)
    }
}

fn row_at(df: &DataFrame, i: usize) -> PyRow {
    // `take` materializes the index label (Range -> Int64([i])) and preserves
    // every column's dtype — a faithful 1-row frame.
    PyRow {
        inner: df.take(&[i]),
    }
}

fn take_frame(df: &DataFrame, positions: &[usize]) -> DataFrame {
    // Delegates to core `take`, which carries column aliases onto the new frame.
    df.take(positions)
}

/// A positional slice: a contiguous `step == 1` slice uses `DataFrame::slice` (a
/// contiguous copy); a strided slice gathers the individual positions.
fn positional_slice(df: &DataFrame, info: &PySliceIndices) -> DataFrame {
    if info.step == 1 {
        df.slice(info.start.max(0) as usize, info.stop.max(0) as usize)
    } else {
        take_frame(df, &strided(info.start, info.stop, info.step))
    }
}

/// Slice a frame by a Python slice — positional for integer bounds, label-based
/// (DatetimeIndex) for string bounds.
fn slice_frame(df: &DataFrame, slice: &Bound<'_, PySlice>) -> PyResult<DataFrame> {
    let start_obj = slice.getattr("start")?;
    let stop_obj = slice.getattr("stop")?;
    let is_label = start_obj.extract::<String>().is_ok() || stop_obj.extract::<String>().is_ok();
    if is_label {
        let index = df.index();
        let lo = if start_obj.is_none() {
            None
        } else {
            Some(parse_label(&start_obj, index)?)
        };
        let hi = if stop_obj.is_none() {
            None
        } else {
            Some(parse_label(&stop_obj, index)?)
        };
        let (a, b) = index.label_slice(lo.as_ref(), hi.as_ref());
        Ok(df.slice(a, b))
    } else {
        let info = slice.indices(df.height() as isize)?;
        Ok(positional_slice(df, &info))
    }
}

fn label_bounds(
    slice: &Bound<'_, PySlice>,
    index: &Index,
) -> PyResult<(Option<Label>, Option<Label>)> {
    let start_obj = slice.getattr("start")?;
    let stop_obj = slice.getattr("stop")?;
    let lo = if start_obj.is_none() {
        None
    } else {
        Some(parse_label(&start_obj, index)?)
    };
    let hi = if stop_obj.is_none() {
        None
    } else {
        Some(parse_label(&stop_obj, index)?)
    };
    Ok((lo, hi))
}

/// `df.loc[...]` label indexer.
#[pyclass]
pub struct DataFrameLoc {
    parent: Py<PyDataFrame>,
}

#[pymethods]
impl DataFrameLoc {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let pf = self.parent.borrow(py);
        ensure_fresh(&pf.inner)?;
        let index = pf.inner.index();
        // 2-D label get: df.loc[rows, col], symmetric with __setitem__.
        if let Ok(tup) = key.downcast::<PyTuple>() {
            if tup.len() == 2 {
                let rows = loc_row_axis(&tup.get_item(0)?, index, pf.inner.height())?;
                let cols = loc_col_axis(&tup.get_item(1)?, &pf.inner)?;
                return select_2d(py, &pf.inner, rows, cols);
            }
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            let (lo, hi) = label_bounds(slice, index)?;
            let (a, b) = index.label_slice(lo.as_ref(), hi.as_ref());
            return Ok(Py::new(py, PyDataFrame::plain(pf.inner.slice(a, b)))?.into_any());
        }
        // boolean-mask / label-list row selection -> sub-frame.
        if as_bool_mask(key, pf.inner.height()).is_some() || key.downcast::<PyList>().is_ok() {
            let positions = loc_positions(key, index, pf.inner.height())?;
            return Ok(
                Py::new(py, PyDataFrame::plain(take_frame(&pf.inner, &positions)))?.into_any(),
            );
        }
        let label = parse_label(key, index)?;
        let pos = index
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        Ok(Py::new(py, row_at(&pf.inner, pos))?.into_any())
    }

    /// `df.loc[rows, col] = scalar | array` (label-based; copy-on-write). `rows` is
    /// a boolean mask, a label slice, a single label, or a label list; `col` is a
    /// single column name. The classic `df.loc[mask, 'signal'] = 1`.
    fn __setitem__(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut pf = self.parent.borrow_mut(py);
        ensure_fresh(&pf.inner)?;
        let (rows, col) = split_row_col(key, "loc")?;
        let colname: String = col.extract().map_err(|_| {
            PyTypeError::new_err("loc assignment column must be a single column name")
        })?;
        let height = pf.inner.height();
        let positions = {
            let index = pf.inner.index();
            loc_positions(&rows, index, height)?
        };
        let j = pf
            .inner
            .column_pos(&colname)
            .ok_or_else(|| PyKeyError::new_err(format!("column {colname:?} not found")))?;
        let target = pf.inner.columns()[j].dtype();
        let val = resolve_assignment(value, target, positions.len())?;
        pf.inner
            .assign_positions(j, &positions, &val)
            .map_err(pyerr)
    }
}

/// `df.iat[i, j]` scalar access by position.
#[pyclass]
pub struct DataFrameIat {
    parent: Py<PyDataFrame>,
}

#[pymethods]
impl DataFrameIat {
    fn __getitem__(&self, py: Python<'_>, key: (isize, isize)) -> PyResult<Py<PyAny>> {
        let pf = self.parent.borrow(py);
        ensure_fresh(&pf.inner)?;
        let i = norm_idx(key.0, pf.inner.height())?;
        let j = norm_idx(key.1, pf.inner.width())?;
        Ok(np_scalar_to_py(py, &pf.inner.columns()[j], i))
    }

    /// `df.iat[i, j] = scalar` — set a single cell by position (copy-on-write).
    fn __setitem__(
        &self,
        py: Python<'_>,
        key: (isize, isize),
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut pf = self.parent.borrow_mut(py);
        ensure_fresh(&pf.inner)?;
        let i = norm_idx(key.0, pf.inner.height())?;
        let j = norm_idx(key.1, pf.inner.width())?;
        let target = pf.inner.columns()[j].dtype();
        let val = scalar_to_column(value, target)?;
        pf.inner.assign_positions(j, &[i], &val).map_err(pyerr)
    }
}

/// `df.at[label, col]` scalar access by label + column name.
#[pyclass]
pub struct DataFrameAt {
    parent: Py<PyDataFrame>,
}

#[pymethods]
impl DataFrameAt {
    fn __getitem__(&self, py: Python<'_>, key: (Py<PyAny>, String)) -> PyResult<Py<PyAny>> {
        let pf = self.parent.borrow(py);
        ensure_fresh(&pf.inner)?;
        let index = pf.inner.index();
        let label = parse_label(key.0.bind(py), index)?;
        let i = index
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        let col = pf.inner.column(&key.1).map_err(pyerr)?;
        Ok(np_scalar_to_py(py, col, i))
    }

    /// `df.at[label, col] = scalar` — set a single cell by label + column name
    /// (copy-on-write).
    fn __setitem__(
        &self,
        py: Python<'_>,
        key: (Py<PyAny>, String),
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut pf = self.parent.borrow_mut(py);
        ensure_fresh(&pf.inner)?;
        let i = {
            let index = pf.inner.index();
            let label = parse_label(key.0.bind(py), index)?;
            index
                .position_of(&label)
                .ok_or_else(|| PyKeyError::new_err("label not found"))?
        };
        let j = pf
            .inner
            .column_pos(&key.1)
            .ok_or_else(|| PyKeyError::new_err(format!("column {:?} not found", key.1)))?;
        let target = pf.inner.columns()[j].dtype();
        let val = scalar_to_column(value, target)?;
        pf.inner.assign_positions(j, &[i], &val).map_err(pyerr)
    }
}

/// `series.loc[...]` label indexer.
#[pyclass]
pub struct SeriesLoc {
    inner: Series,
}

#[pymethods]
impl SeriesLoc {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(slice) = key.downcast::<PySlice>() {
            let (lo, hi) = label_bounds(slice, &self.inner.index)?;
            let (a, b) = self.inner.index.label_slice(lo.as_ref(), hi.as_ref());
            let positions: Vec<usize> = (a..b).collect();
            let data = self.inner.data.take(&positions);
            let index = Arc::new(self.inner.index.take(&positions));
            return Ok(Py::new(
                py,
                PySeries {
                    inner: Series::new(self.inner.name.clone(), data, index),
                },
            )?
            .into_any());
        }
        let label = parse_label(key, &self.inner.index)?;
        let pos = self
            .inner
            .index
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        Ok(np_scalar_to_py(py, &self.inner.data, pos))
    }
}

/// Convert epoch numbers or datetime strings to a datetime `Series`, mirroring
/// `pandas.to_datetime`. Numeric input is read as an epoch in `unit`
/// (`"s"`/`"ms"`/`"us"`/`"ns"`), preserving sub-`unit` fractions; string input is parsed
/// (naive strings as UTC; offset-aware strings are absolute); an already-datetime input is
/// returned unchanged. To attach a timezone, `set_index` the result and then `df.tz_localize`
/// (read a naive wall-clock in a zone) or `df.tz_convert` (re-display an absolute instant).
/// Accepts a volas `Series` (its name and index are preserved), a 1-D NumPy array, or a list.
#[pyfunction]
#[pyo3(signature = (obj, unit = "ns", format = None))]
fn to_datetime(obj: &Bound<'_, PyAny>, unit: &str, format: Option<&str>) -> PyResult<PySeries> {
    let (col, name, index) = match obj.extract::<PyRef<PySeries>>() {
        Ok(s) => (
            s.inner.data.clone(),
            s.inner.name.clone(),
            Arc::clone(&s.inner.index),
        ),
        Err(_) => {
            let col = pyany_to_column(obj)?;
            let n = col.len();
            (col, None, Arc::new(Index::range(n)))
        }
    };
    let converted = match col {
        c @ Column::Datetime(_) => c,
        Column::Str(v, val) => match format {
            // An explicit format parses faster and unambiguously (pandas `format=`).
            // A missing (NA) cell maps to NaT, not parsed from its "" placeholder.
            Some(fmt) => {
                let ns = v
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        // a missing (NA) or empty/blank cell -> NaT, like the
                        // default path; a non-empty value must match the format.
                        if !val.is_valid(i) || s.trim().is_empty() {
                            return Ok(i64::MIN);
                        }
                        datetime::parse_ns_format(s, fmt).ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "\"{s}\" does not match format \"{fmt}\""
                            ))
                        })
                    })
                    .collect::<PyResult<Vec<i64>>>()?;
                Column::datetime(ns)
            }
            None => Column::Str(v, val).to_datetime().map_err(pyerr)?,
        },
        c => c.epoch_to_datetime_rounded(unit).map_err(pyerr)?,
    };
    Ok(PySeries {
        inner: Series::new(name, converted, index),
    })
}

/// Get the canonical full name of a `directive` — the actual column name volas caches it
/// under. The command name is lowercased and default arguments / series are dropped.
///
/// Usage::
///
///     volas.directive_stringify('MACD:12,26')   # -> "macd"
#[pyfunction]
fn directive_stringify(directive: &str) -> PyResult<String> {
    let node = parse(directive).map_err(syntax_err)?;
    Ok(volas_directive::stringify(&node))
}

/// Get the lookback period of a `directive` — the minimum number of prior rows it needs
/// before it can emit a (non-NaN) value.
///
/// Usage::
///
///     volas.directive_lookback('boll:20')   # -> 19
#[pyfunction]
fn directive_lookback(directive: &str) -> PyResult<usize> {
    let node = parse(directive).map_err(syntax_err)?;
    Ok(volas_directive::lookback::lookback(&node))
}

/// The compiled module backing the `volas` package.
#[pymodule]
fn volas_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PySeries>()?;
    m.add_class::<PyRow>()?;
    m.add_class::<PyTimestamp>()?;
    m.add_class::<DataFrameILoc>()?;
    m.add_class::<DataFrameLoc>()?;
    m.add_class::<DataFrameIat>()?;
    m.add_class::<DataFrameAt>()?;
    m.add_class::<SeriesILoc>()?;
    m.add_class::<SeriesLoc>()?;
    m.add_class::<PyTimeFrame>()?;
    m.add("DirectiveError", m.py().get_type::<DirectiveError>())?;
    m.add(
        "DirectiveSyntaxError",
        m.py().get_type::<DirectiveSyntaxError>(),
    )?;
    m.add(
        "DirectiveValueError",
        m.py().get_type::<DirectiveValueError>(),
    )?;
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(to_datetime, m)?)?;
    m.add_function(wrap_pyfunction!(directive_stringify, m)?)?;
    m.add_function(wrap_pyfunction!(directive_lookback, m)?)?;
    m.add_class::<NaType>()?;
    m.add("NA", na(m.py()))?;
    Ok(())
}
