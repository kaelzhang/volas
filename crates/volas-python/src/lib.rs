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
use pyo3::types::{PyDict, PyList, PySlice};

use volas_core::{datetime, Column, DataFrame, DType, Index, Label, Series, VolasError};
use volas_directive::{execute, parse};

mod readers;
mod timeframe;

use readers::read_csv;
use timeframe::{build_agg_spec, resolve_time_frame, PyCumulator, PyTimeFrame};

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

/// Parse a pandas-style dtype string to a volas [`DType`].
fn parse_dtype(s: &str) -> PyResult<DType> {
    Ok(match s {
        "float" | "float64" | "float_" | "double" | "f64" => DType::F64,
        "int" | "int64" | "int_" | "long" | "i64" => DType::I64,
        "bool" | "boolean" => DType::Bool,
        "str" | "string" | "object" | "O" => DType::Utf8,
        "datetime" | "datetime64" | "datetime64[ns]" => DType::Datetime,
        _ => return Err(PyValueError::new_err(format!("unknown dtype {s:?}"))),
    })
}

fn pyany_to_column(v: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(a) = v.extract::<PyReadonlyArray1<f64>>() {
        return Ok(Column::f64(a.as_slice()?.to_vec()));
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<i64>>() {
        return Ok(Column::i64(a.as_slice()?.to_vec()));
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
    Err(PyTypeError::new_err(
        "column values must be a 1-D numeric array, a list of numbers, or a list of strings",
    ))
}

fn column_to_numpy<'py>(py: Python<'py>, col: &Column) -> Bound<'py, PyAny> {
    match col {
        Column::F64(v) => v.to_vec().into_pyarray(py).into_any(),
        Column::Bool(v) => v.to_vec().into_pyarray(py).into_any(),
        Column::I64(v) => v.to_vec().into_pyarray(py).into_any(),
        // String columns become NumPy object arrays (pandas `object` dtype).
        Column::Str(v) => {
            let list = PyList::new(py, v.as_slice()).expect("build str list");
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", "object").expect("set dtype=object");
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
        Column::I64(v) => v[i].into_pyobject(py).unwrap().into_any().unbind(),
        Column::Bool(v) => v[i].into_pyobject(py).unwrap().to_owned().into_any().unbind(),
        Column::Str(v) => v[i].clone().into_pyobject(py).unwrap().into_any().unbind(),
        Column::Datetime(v) => py
            .import("numpy")
            .expect("import numpy")
            .call_method1("datetime64", (v[i], "ns"))
            .expect("np.datetime64")
            .into_any()
            .unbind(),
    }
}

/// Render an index label at position `i` as a Python object (a datetime string
/// for a DatetimeIndex, else the integer label).
fn label_to_py(py: Python<'_>, index: &Index, i: usize) -> Py<PyAny> {
    match index {
        Index::Datetime(v) => datetime::format_ns(v[i])
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
        Index::Int64(v) => v[i].into_pyobject(py).unwrap().into_any().unbind(),
        Index::Range(_) => (i as i64).into_pyobject(py).unwrap().into_any().unbind(),
        Index::Str(v) => v[i].clone().into_pyobject(py).unwrap().into_any().unbind(),
    }
}

/// Parse a Python timestamp (datetime string or epoch-ns integer) to i64 ns.
pub(crate) fn parse_ts(key: &Bound<'_, PyAny>) -> PyResult<i64> {
    if let Ok(s) = key.extract::<String>() {
        return datetime::parse_ns(&s)
            .ok_or_else(|| PyKeyError::new_err(format!("invalid datetime label {s:?}")));
    }
    if let Ok(i) = key.extract::<i64>() {
        return Ok(i);
    }
    Err(PyKeyError::new_err("label must be a datetime string or integer"))
}

/// Parse a Python label to the [`Label`] kind expected by `index`: a string for
/// a string index, a parsed datetime / integer for the numeric kinds.
pub(crate) fn parse_label(key: &Bound<'_, PyAny>, index: &Index) -> PyResult<Label> {
    match index {
        Index::Str(_) => key
            .extract::<String>()
            .map(Label::Str)
            .map_err(|_| PyKeyError::new_err("label must be a string for a string index")),
        _ => parse_ts(key).map(Label::I64),
    }
}

/// Build the `.index` as a NumPy array (datetime64[ns] for a DatetimeIndex,
/// an object array for a string index).
fn index_to_numpy<'py>(py: Python<'py>, index: &Index) -> PyResult<Bound<'py, PyAny>> {
    match index {
        Index::Datetime(v) => {
            let arr = v.clone().into_pyarray(py);
            Ok(arr.call_method1("astype", ("datetime64[ns]",))?)
        }
        Index::Int64(v) => Ok(v.clone().into_pyarray(py).into_any()),
        Index::Range(n) => Ok((0..*n as i64).collect::<Vec<_>>().into_pyarray(py).into_any()),
        Index::Str(v) => {
            let list = PyList::new(py, v.as_slice())?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", "object")?;
            Ok(py.import("numpy")?.call_method("array", (list,), Some(&kwargs))?)
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

// --- Series ----------------------------------------------------------------

/// `volas.Series` — a single named, indexed column.
#[pyclass(name = "Series")]
pub struct PySeries {
    pub(crate) inner: Series,
}

#[pymethods]
impl PySeries {
    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }

    #[getter]
    fn dtype(&self) -> String {
        self.inner.dtype().to_string()
    }

    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        index_to_numpy(py, &self.inner.index)
    }

    #[getter]
    fn iloc(&self) -> SeriesILoc {
        SeriesILoc {
            inner: self.inner.clone(),
        }
    }

    #[getter]
    fn loc(&self) -> SeriesLoc {
        SeriesLoc {
            inner: self.inner.clone(),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
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

    /// NaN-skipping mean of the values.
    fn mean(&self) -> f64 {
        let v = non_nan(&self.inner.data);
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    }

    /// NaN-skipping sum (0.0 when empty / all-NaN, matching pandas).
    fn sum(&self) -> f64 {
        non_nan(&self.inner.data).iter().sum()
    }

    /// NaN-skipping minimum (NaN when empty / all-NaN).
    fn min(&self) -> f64 {
        non_nan(&self.inner.data)
            .into_iter()
            .fold(f64::NAN, |m, x| if m.is_nan() { x } else { m.min(x) })
    }

    /// NaN-skipping maximum (NaN when empty / all-NaN).
    fn max(&self) -> f64 {
        non_nan(&self.inner.data)
            .into_iter()
            .fold(f64::NAN, |m, x| if m.is_nan() { x } else { m.max(x) })
    }

    /// NaN-skipping sample variance (`ddof=1`; NaN with fewer than 2 values).
    fn var(&self) -> f64 {
        let v = non_nan(&self.inner.data);
        let n = v.len();
        if n < 2 {
            return f64::NAN;
        }
        let mean = v.iter().sum::<f64>() / n as f64;
        v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64
    }

    /// NaN-skipping sample standard deviation (`ddof=1`).
    fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// NaN-skipping median.
    fn median(&self) -> f64 {
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

    /// Shift values by `n` rows (positive = down; vacated cells -> NaN).
    #[pyo3(signature = (n = 1))]
    fn shift(&self, n: isize) -> PySeries {
        let a = self.inner.data.to_f64_vec();
        let len = a.len();
        let mut out = vec![f64::NAN; len];
        if n >= 0 {
            let n = (n as usize).min(len);
            out[n..].copy_from_slice(&a[..len - n]);
        } else {
            let n = ((-n) as usize).min(len);
            out[..len - n].copy_from_slice(&a[n..]);
        }
        f64_series(&self.inner, out)
    }

    /// First difference `x[i] - x[i-n]` (the first `n` rows -> NaN).
    #[pyo3(signature = (n = 1))]
    fn diff(&self, n: isize) -> PySeries {
        let a = self.inner.data.to_f64_vec();
        let len = a.len();
        let mut out = vec![f64::NAN; len];
        if n >= 0 {
            let n = n as usize;
            for i in n..len {
                out[i] = a[i] - a[i - n];
            }
        } else {
            let n = (-n) as usize;
            for i in 0..len.saturating_sub(n) {
                out[i] = a[i] - a[i + n];
            }
        }
        f64_series(&self.inner, out)
    }

    /// Replace NaN with `value` (F64 columns; others returned unchanged).
    fn fillna(&self, value: f64) -> PySeries {
        let col = match &self.inner.data {
            Column::F64(v) => {
                Column::f64(v.iter().map(|&x| if x.is_nan() { value } else { x }).collect())
            }
            other => other.clone(),
        };
        PySeries {
            inner: Series::new(self.inner.name.clone(), col, Arc::clone(&self.inner.index)),
        }
    }

    /// Boolean mask of missing (NaN) values (non-F64 columns -> all False).
    fn isna(&self) -> PySeries {
        let out = match &self.inner.data {
            Column::F64(v) => v.iter().map(|x| x.is_nan()).collect(),
            other => vec![false; other.len()],
        };
        bool_series(&self.inner, out)
    }

    /// Boolean mask of present (non-NaN) values.
    fn notna(&self) -> PySeries {
        let out = match &self.inner.data {
            Column::F64(v) => v.iter().map(|x| !x.is_nan()).collect(),
            other => vec![true; other.len()],
        };
        bool_series(&self.inner, out)
    }

    /// Drop missing (NaN) elements (carries their index labels with them).
    fn dropna(&self) -> PySeries {
        let keep: Vec<usize> = match &self.inner.data {
            Column::F64(v) => v
                .iter()
                .enumerate()
                .filter_map(|(i, x)| (!x.is_nan()).then_some(i))
                .collect(),
            other => (0..other.len()).collect(),
        };
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

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_binop(&self.inner, other, |a, b| a + b)
    }
    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_binop(&self.inner, other, |a, b| a - b)
    }
    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_binop(&self.inner, other, |a, b| a * b)
    }
    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_binop(&self.inner, other, |a, b| a / b)
    }
    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_binop(&self.inner, other, |a, b| b + a)
    }
    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_binop(&self.inner, other, |a, b| b - a)
    }
    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_binop(&self.inner, other, |a, b| b * a)
    }
    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_binop(&self.inner, other, |a, b| b / a)
    }

    // Element-wise comparisons -> bool Series (pandas-style).
    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, |a, b| a < b)
    }
    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, |a, b| a <= b)
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, |a, b| a == b)
    }
    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, |a, b| a != b)
    }
    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, |a, b| a >= b)
    }
    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, |a, b| a > b)
    }

    // Element-wise boolean logic -> bool Series (operands coerced to bool).
    fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, |a, b| a && b)
    }
    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, |a, b| a || b)
    }
    fn __xor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, |a, b| a ^ b)
    }
    fn __rand__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, |a, b| a && b)
    }
    fn __ror__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, |a, b| a || b)
    }
    fn __rxor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, |a, b| a ^ b)
    }
    fn __invert__(&self) -> PySeries {
        let out: Vec<bool> = to_bool_vec(&self.inner.data).iter().map(|&b| !b).collect();
        PySeries {
            inner: Series::new(
                self.inner.name.clone(),
                Column::bool(out),
                Arc::clone(&self.inner.index),
            ),
        }
    }

    /// `series[key]`: an integer position, a datetime label, or a slice.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(i) = key.extract::<isize>() {
            let i = norm_idx(i, self.inner.len())?;
            return Ok(scalar_to_py(py, &self.inner.data, i));
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
        Ok(scalar_to_py(py, &self.inner.data, pos))
    }

    fn __repr__(&self) -> String {
        format!(
            "Series(name={:?}, dtype={}, len={})",
            self.inner.name,
            self.inner.dtype(),
            self.inner.len()
        )
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
            return Ok(scalar_to_py(py, &self.inner.data, i));
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            return Ok(Py::new(py, slice_series(&self.inner, slice)?)?.into_any());
        }
        Err(PyIndexError::new_err("iloc key must be an integer or slice"))
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

/// The RHS of a Series binary op as an `f64` vector — another Series (positional,
/// no index alignment) or a broadcast scalar.
fn series_rhs_f64(other: &Bound<'_, PyAny>, len: usize) -> PyResult<Vec<f64>> {
    if let Ok(o) = other.extract::<PyRef<PySeries>>() {
        Ok(o.inner.data.to_f64_vec())
    } else if let Ok(scalar) = other.extract::<f64>() {
        Ok(vec![scalar; len])
    } else {
        Err(PyTypeError::new_err("unsupported operand for a Series operation"))
    }
}

/// A new F64 `Series` carrying `s`'s name and index.
fn f64_series(s: &Series, out: Vec<f64>) -> PySeries {
    PySeries {
        inner: Series::new(s.name.clone(), Column::f64(out), Arc::clone(&s.index)),
    }
}

/// A new Bool `Series` carrying `s`'s name and index.
fn bool_series(s: &Series, out: Vec<bool>) -> PySeries {
    PySeries {
        inner: Series::new(s.name.clone(), Column::bool(out), Arc::clone(&s.index)),
    }
}

fn series_binop(
    s: &Series,
    other: &Bound<'_, PyAny>,
    f: impl Fn(f64, f64) -> f64,
) -> PyResult<PySeries> {
    let a = s.data.to_f64_vec();
    let rhs = series_rhs_f64(other, a.len())?;
    let n = a.len().min(rhs.len());
    let mut out = vec![f64::NAN; a.len()];
    for i in 0..n {
        out[i] = f(a[i], rhs[i]);
    }
    Ok(PySeries {
        inner: Series::new(s.name.clone(), Column::f64(out), Arc::clone(&s.index)),
    })
}

/// Element-wise comparison -> bool Series (positional; NaN compares `false` for
/// ordering/equality and `true` for `!=`, matching IEEE / pandas element-wise).
fn series_cmp(
    s: &Series,
    other: &Bound<'_, PyAny>,
    f: impl Fn(f64, f64) -> bool,
) -> PyResult<PySeries> {
    let a = s.data.to_f64_vec();
    let rhs = series_rhs_f64(other, a.len())?;
    let n = a.len().min(rhs.len());
    let mut out = vec![false; a.len()];
    for i in 0..n {
        out[i] = f(a[i], rhs[i]);
    }
    Ok(PySeries {
        inner: Series::new(s.name.clone(), Column::bool(out), Arc::clone(&s.index)),
    })
}

/// The non-NaN `f64` values of a column (for NaN-skipping reductions).
fn non_nan(col: &Column) -> Vec<f64> {
    col.to_f64_vec().into_iter().filter(|x| !x.is_nan()).collect()
}

/// A column coerced to bool (a `Bool` column as-is, else `x != 0.0`).
fn to_bool_vec(col: &Column) -> Vec<bool> {
    match col {
        Column::Bool(v) => v.to_vec(),
        other => other.to_f64_vec().iter().map(|&x| x != 0.0).collect(),
    }
}

/// Element-wise boolean logic -> bool Series (both operands coerced to bool).
fn series_logical(
    s: &Series,
    other: &Bound<'_, PyAny>,
    f: impl Fn(bool, bool) -> bool,
) -> PyResult<PySeries> {
    let a = to_bool_vec(&s.data);
    let rhs = if let Ok(o) = other.extract::<PyRef<PySeries>>() {
        to_bool_vec(&o.inner.data)
    } else if let Ok(b) = other.extract::<bool>() {
        vec![b; a.len()]
    } else if let Ok(x) = other.extract::<f64>() {
        vec![x != 0.0; a.len()]
    } else {
        return Err(PyTypeError::new_err("unsupported operand for a Series logical op"));
    };
    let n = a.len().min(rhs.len());
    let mut out = vec![false; a.len()];
    for i in 0..n {
        out[i] = f(a[i], rhs[i]);
    }
    Ok(PySeries {
        inner: Series::new(s.name.clone(), Column::bool(out), Arc::clone(&s.index)),
    })
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

    /// A scalar by column name (`row[col]`).
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let col = self.inner.column(key).map_err(pyerr)?;
        Ok(scalar_to_py(py, col, 0))
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
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

    fn __repr__(&self) -> String {
        format!("Row(columns={:?})", self.inner.names())
    }
}

// --- DataFrame -------------------------------------------------------------

/// `volas.DataFrame` — an ordered, named, time-indexed table with directive
/// indexing and pandas-compatible positional / label access.
#[pyclass(name = "DataFrame")]
pub struct PyDataFrame {
    pub(crate) inner: DataFrame,
}

impl PyDataFrame {
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
        let stale: Vec<_> = self
            .inner
            .computed_columns()
            .into_iter()
            .filter(|(n, m)| m.valid_rows < height && only.is_none_or(|o| o == n))
            .collect();
        if stale.is_empty() {
            return Ok(());
        }
        let computed_names: HashSet<String> = self
            .inner
            .computed_columns()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        let real_names: Vec<String> = self
            .inner
            .names()
            .iter()
            .filter(|n| !computed_names.contains(*n))
            .cloned()
            .collect();
        let base = self.inner.select(&real_names).map_err(pyerr)?;
        for (name, meta) in stale {
            let start = meta.valid_rows.saturating_sub(meta.lookback);
            let slice = base.slice(start, height);
            let node = parse(&meta.directive).map_err(value_err)?;
            let recomputed = execute(&slice, &node).map_err(value_err)?;
            // The recomputed slice is F64 or Bool (directive result); write its
            // stale tail back into the column at its original dtype.
            let tail = recomputed.slice(meta.valid_rows - start, recomputed.len());
            self.inner
                .update_computed_tail(&name, meta.valid_rows, &tail)
                .map_err(pyerr)?;
        }
        Ok(())
    }
}

#[pymethods]
impl PyDataFrame {
    #[new]
    #[pyo3(signature = (data, date_col = None))]
    fn new(data: &Bound<'_, PyDict>, date_col: Option<String>) -> PyResult<Self> {
        let mut names = Vec::new();
        let mut columns = Vec::new();
        for (k, v) in data.iter() {
            names.push(k.extract::<String>()?);
            columns.push(pyany_to_column(&v)?);
        }
        let mut df = DataFrame::new(names, columns, None).map_err(pyerr)?;
        // Same string -> DatetimeIndex path as read_csv: parse the named column
        // to datetime, then move it into the index.
        if let Some(dc) = date_col {
            let parsed = df.column(&dc).map_err(pyerr)?.to_datetime().map_err(pyerr)?;
            df.set_column(&dc, parsed).map_err(pyerr)?;
            df = df.set_index(&dc).map_err(pyerr)?;
        }
        Ok(PyDataFrame { inner: df })
    }

    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.height(), self.inner.width())
    }

    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        index_to_numpy(py, self.inner.index())
    }

    #[getter]
    fn iloc(&self) -> DataFrameILoc {
        DataFrameILoc {
            inner: self.inner.clone(),
        }
    }

    #[getter]
    fn loc(&self) -> DataFrameLoc {
        DataFrameLoc {
            inner: self.inner.clone(),
        }
    }

    #[getter]
    fn iat(&self) -> DataFrameIat {
        DataFrameIat {
            inner: self.inner.clone(),
        }
    }

    #[getter]
    fn at(&self) -> DataFrameAt {
        DataFrameAt {
            inner: self.inner.clone(),
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

    /// First `n` rows (pandas `head`).
    #[pyo3(signature = (n = 5))]
    fn head(&self, n: usize) -> PyDataFrame {
        PyDataFrame {
            inner: self.inner.slice(0, n.min(self.inner.height())),
        }
    }

    /// Last `n` rows (pandas `tail`).
    #[pyo3(signature = (n = 5))]
    fn tail(&self, n: usize) -> PyDataFrame {
        let h = self.inner.height();
        PyDataFrame {
            inner: self.inner.slice(h.saturating_sub(n), h),
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

    /// Drop rows containing missing values. `how='any'` (default) drops a row if
    /// any F64 column is NaN there; `how='all'` only if every column is NaN.
    #[pyo3(signature = (how = "any"))]
    fn dropna(&self, how: &str) -> PyDataFrame {
        let cols = self.inner.columns();
        let total = cols.len();
        let keep: Vec<usize> = (0..self.inner.height())
            .filter(|&i| {
                let nan = cols
                    .iter()
                    .filter(|c| matches!(c, Column::F64(v) if v[i].is_nan()))
                    .count();
                match how {
                    "all" => nan < total.max(1),
                    _ => nan == 0,
                }
            })
            .collect();
        PyDataFrame {
            inner: take_frame(&self.inner, &keep),
        }
    }

    /// Sort rows by index label (pandas `sort_index`).
    #[pyo3(signature = (ascending = true))]
    fn sort_index(&self, ascending: bool) -> PyDataFrame {
        let perm = self.inner.index().argsort(ascending);
        PyDataFrame {
            inner: take_frame(&self.inner, &perm),
        }
    }

    /// Move the row index into an `'index'` column and restore a RangeIndex
    /// (pandas `reset_index`); `drop=True` discards the old index.
    #[pyo3(signature = (drop = false))]
    fn reset_index(&self, drop: bool) -> PyResult<PyDataFrame> {
        let h = self.inner.height();
        let (names, columns): (Vec<String>, Vec<Column>) = if drop {
            (self.inner.names().to_vec(), self.inner.columns().to_vec())
        } else {
            let mut names = vec!["index".to_string()];
            names.extend(self.inner.names().iter().cloned());
            let mut cols = vec![self.inner.index().to_column()];
            cols.extend(self.inner.columns().iter().cloned());
            (names, cols)
        };
        Ok(PyDataFrame {
            inner: DataFrame::new(names, columns, Some(Index::Range(h))).map_err(pyerr)?,
        })
    }

    /// `df[name] = value` — add or replace a column. `value` may be a scalar
    /// (broadcast), a 1-D array / list, or a Series (positional, length must
    /// equal the frame height). Copy-on-write: a prior `copy()` is unaffected.
    fn __setitem__(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
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
        self.inner.set_column(name, col).map_err(pyerr)
    }

    fn __getitem__(&mut self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        // boolean mask (Series or numpy)
        if let Ok(s) = key.extract::<PyRef<PySeries>>() {
            if let Column::Bool(mask) = &s.inner.data {
                let sub = self.inner.filter_mask(mask).map_err(pyerr)?;
                return Ok(Py::new(py, PyDataFrame { inner: sub })?.into_any());
            }
        }
        if let Ok(arr) = key.extract::<PyReadonlyArray1<bool>>() {
            let sub = self.inner.filter_mask(arr.as_slice()?).map_err(pyerr)?;
            return Ok(Py::new(py, PyDataFrame { inner: sub })?.into_any());
        }
        // label / positional slice: df[:'date'], df[1:5]
        if let Ok(slice) = key.downcast::<PySlice>() {
            let sub = slice_frame(&self.inner, slice)?;
            return Ok(Py::new(py, PyDataFrame { inner: sub })?.into_any());
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
                self.inner.set_column(&canonical, col).map_err(pyerr)?;
                self.inner.set_computed(&canonical, canonical.clone(), lookback);
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
            return Ok(Py::new(py, PyDataFrame { inner: df })?.into_any());
        }
        Err(PyKeyError::new_err(
            "key must be a column name, directive, list, boolean mask, or slice",
        ))
    }

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
                self.inner.set_column(&canonical, col).map_err(pyerr)?;
                self.inner.set_computed(&canonical, canonical.clone(), lookback);
            }
            let col = self.inner.column(&canonical).map_err(pyerr)?.clone();
            Ok(column_to_numpy(py, &col))
        } else {
            let col = execute(&self.inner, &node).map_err(value_err)?;
            Ok(column_to_numpy(py, &col))
        }
    }

    /// The minimum number of prior rows a directive needs (its lookback).
    #[staticmethod]
    fn directive_lookback(directive: &str) -> PyResult<usize> {
        let node = parse(directive).map_err(syntax_err)?;
        Ok(volas_directive::lookback::lookback(&node))
    }

    /// The canonical string form of a directive (default args / series dropped).
    #[staticmethod]
    fn directive_stringify(directive: &str) -> PyResult<String> {
        let node = parse(directive).map_err(syntax_err)?;
        Ok(volas_directive::stringify(&node))
    }

    fn get_column(&self, key: &str) -> PyResult<PySeries> {
        let col = self.inner.column(key).map_err(pyerr)?.clone();
        Ok(self.wrap_series(key.to_string(), col))
    }

    /// A copy of the frame.
    fn copy(&self) -> PyDataFrame {
        PyDataFrame {
            inner: self.inner.clone(),
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
            return Ok(PyDataFrame {
                inner: self.inner.select(&keep).map_err(pyerr)?,
            });
        }
        let index = self.inner.index();
        let targets: Vec<Label> = labels
            .iter()
            .map(|l| parse_label(l.bind(py), index))
            .collect::<PyResult<_>>()?;
        let positions: Vec<usize> = (0..self.inner.height())
            .filter(|&i| !targets.contains(&index.label_at(i)))
            .collect();
        Ok(PyDataFrame {
            inner: take_frame(&self.inner, &positions),
        })
    }

    /// Append the rows of another DataFrame or a single Row, returning a new
    /// DataFrame (pandas semantics; not in place).
    /// Append `other`'s rows in place (amortized O(1), like `list.append`) and
    /// return the same frame. Missing columns are NaN-padded; computed columns
    /// go stale until `fulfill`. This is the live single-bar hot path — it grows
    /// one frame with no full-column copy (a snapshot taken via `copy` / `iloc`
    /// still pays one copy-on-write the next time *it* is appended to).
    fn append<'py>(
        slf: Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        // Extract `other` into an owned frame first so its borrow is released
        // before we mutably borrow `slf` (so `df.append(df)` cannot deadlock).
        let other_inner = if let Ok(df) = other.extract::<PyRef<PyDataFrame>>() {
            df.inner.clone()
        } else if let Ok(row) = other.extract::<PyRef<PyRow>>() {
            row.inner.clone()
        } else {
            return Err(PyTypeError::new_err("append expects a DataFrame or Row"));
        };
        slf.borrow_mut().inner.append(&other_inner).map_err(pyerr)?;
        Ok(slf)
    }

    /// The frame as a 2-D NumPy array (pandas `to_numpy`). With no `dtype`: a fast
    /// `float64` matrix when every column is numeric, else a lossless `object`
    /// array (string columns kept, not NaN-poisoned). `dtype` casts (e.g.
    /// `'float32'`); requesting a float over a string column raises.
    #[pyo3(signature = (dtype = None))]
    fn to_numpy<'py>(&self, py: Python<'py>, dtype: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        ensure_fresh(&self.inner)?;
        let has_str = self.inner.columns().iter().any(|c| matches!(c, Column::Str(_)));
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
        Ok(py.import("numpy")?.call_method("array", (rows,), Some(&kwargs))?)
    }

    /// Value equality (same columns + index + values, `NaN == NaN`).
    fn equals(&self, other: &PyDataFrame) -> bool {
        self.inner.equals(&other.inner)
    }

    /// Resample to a coarser `time_frame` (OHLCV cumulation). Requires a
    /// DatetimeIndex; `cumulators` overrides per-column aggregators, e.g.
    /// `{'volume': 'sum'}`.
    #[pyo3(signature = (time_frame, cumulators = None))]
    fn cumulate(
        &self,
        time_frame: &Bound<'_, PyAny>,
        cumulators: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let tf = resolve_time_frame(time_frame)?;
        let spec = build_agg_spec(cumulators)?;
        let out = volas_time::cumulate(&self.inner, tf, &spec).map_err(pyerr)?;
        Ok(PyDataFrame { inner: out })
    }

    /// Rename columns (pandas `rename(columns={old: new})`), returning a new
    /// frame.
    #[pyo3(signature = (columns))]
    fn rename(&self, columns: &Bound<'_, PyDict>) -> PyResult<PyDataFrame> {
        let mut mapping = HashMap::new();
        for (k, v) in columns.iter() {
            mapping.insert(k.extract::<String>()?, v.extract::<String>()?);
        }
        Ok(PyDataFrame {
            inner: self.inner.rename(&mapping).map_err(pyerr)?,
        })
    }

    /// Move a column into the row index (pandas `set_index(col)`), returning a
    /// new frame. A datetime / int / string column becomes the matching index.
    #[pyo3(signature = (keys))]
    fn set_index(&self, keys: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.set_index(keys).map_err(pyerr)?,
        })
    }

    /// Cast columns to new dtypes (pandas `astype({col: dtype})`), returning a
    /// new frame.
    fn astype(&self, dtypes: &Bound<'_, PyDict>) -> PyResult<PyDataFrame> {
        let mut mapping = HashMap::new();
        for (k, v) in dtypes.iter() {
            mapping.insert(k.extract::<String>()?, parse_dtype(&v.extract::<String>()?)?);
        }
        Ok(PyDataFrame {
            inner: self.inner.astype(&mapping).map_err(pyerr)?,
        })
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

    fn __repr__(&self) -> String {
        format!(
            "DataFrame(columns={:?}, shape=({}, {}))",
            self.inner.names(),
            self.inner.height(),
            self.inner.width()
        )
    }
}

/// `df.iloc[...]` positional indexer.
#[pyclass]
pub struct DataFrameILoc {
    inner: DataFrame,
}

#[pymethods]
impl DataFrameILoc {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ensure_fresh(&self.inner)?;
        if let Ok(i) = key.extract::<isize>() {
            let i = norm_idx(i, self.inner.height())?;
            return Ok(Py::new(py, row_at(&self.inner, i))?.into_any());
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            let info = slice.indices(self.inner.height() as isize)?;
            let positions = strided(info.start, info.stop, info.step);
            let sub = take_frame(&self.inner, &positions);
            return Ok(Py::new(py, PyDataFrame { inner: sub })?.into_any());
        }
        Err(PyIndexError::new_err("iloc key must be an integer or slice"))
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
        let positions = strided(info.start, info.stop, info.step);
        Ok(take_frame(df, &positions))
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
    inner: DataFrame,
}

#[pymethods]
impl DataFrameLoc {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        ensure_fresh(&self.inner)?;
        let index = self.inner.index();
        if let Ok(slice) = key.downcast::<PySlice>() {
            let (lo, hi) = label_bounds(slice, index)?;
            let (a, b) = index.label_slice(lo.as_ref(), hi.as_ref());
            return Ok(Py::new(py, PyDataFrame { inner: self.inner.slice(a, b) })?.into_any());
        }
        let label = parse_label(key, index)?;
        let pos = index
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        Ok(Py::new(py, row_at(&self.inner, pos))?.into_any())
    }
}

/// `df.iat[i, j]` scalar access by position.
#[pyclass]
pub struct DataFrameIat {
    inner: DataFrame,
}

#[pymethods]
impl DataFrameIat {
    fn __getitem__(&self, py: Python<'_>, key: (isize, isize)) -> PyResult<Py<PyAny>> {
        ensure_fresh(&self.inner)?;
        let i = norm_idx(key.0, self.inner.height())?;
        let j = norm_idx(key.1, self.inner.width())?;
        Ok(scalar_to_py(py, &self.inner.columns()[j], i))
    }
}

/// `df.at[label, col]` scalar access by label + column name.
#[pyclass]
pub struct DataFrameAt {
    inner: DataFrame,
}

#[pymethods]
impl DataFrameAt {
    fn __getitem__(&self, py: Python<'_>, key: (Py<PyAny>, String)) -> PyResult<Py<PyAny>> {
        ensure_fresh(&self.inner)?;
        let index = self.inner.index();
        let label = parse_label(key.0.bind(py), index)?;
        let i = index
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        let col = self.inner.column(&key.1).map_err(pyerr)?;
        Ok(scalar_to_py(py, col, i))
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
        Ok(scalar_to_py(py, &self.inner.data, pos))
    }
}


/// The compiled module backing the `volas` package.
#[pymodule]
fn volas_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PySeries>()?;
    m.add_class::<PyRow>()?;
    m.add_class::<DataFrameILoc>()?;
    m.add_class::<DataFrameLoc>()?;
    m.add_class::<DataFrameIat>()?;
    m.add_class::<DataFrameAt>()?;
    m.add_class::<SeriesILoc>()?;
    m.add_class::<SeriesLoc>()?;
    m.add_class::<PyTimeFrame>()?;
    m.add_class::<PyCumulator>()?;
    m.add("DirectiveError", m.py().get_type::<DirectiveError>())?;
    m.add("DirectiveSyntaxError", m.py().get_type::<DirectiveSyntaxError>())?;
    m.add("DirectiveValueError", m.py().get_type::<DirectiveValueError>())?;
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    Ok(())
}
