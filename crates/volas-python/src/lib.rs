//! PyO3 bindings: expose the `volas` kernel to Python as `volas_rs.DataFrame` /
//! `volas_rs.Series`, with stock-pandas-style directive indexing and a
//! pandas-compatible indexing surface (`.iloc`, `.index`, `.name`, label lookup).
//!
//! This crate is the only place pyo3 / numpy are used; all logic lives in the
//! `volas-core` / `volas-compute` / `volas-directive` crates.

use std::collections::HashMap;
use std::sync::Arc;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice};

use volas_core::{datetime, Column, DataFrame, DType, Index, Series, VolasError};
use volas_directive::{execute, parse};
use volas_time::{Agg, AggSpec, Cumulator, TimeFrame};

// --- helpers ---------------------------------------------------------------

fn pyerr(e: VolasError) -> PyErr {
    match e {
        VolasError::ColumnNotFound(n) => PyKeyError::new_err(format!("column \"{n}\" not found")),
        VolasError::DType(m) => PyTypeError::new_err(m),
        VolasError::Shape(m) | VolasError::Index(m) | VolasError::Value(m) => {
            PyValueError::new_err(m)
        }
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
    }
}

/// Parse a Python label (datetime string or integer) to the i64 used by the index.
fn parse_label(key: &Bound<'_, PyAny>) -> PyResult<i64> {
    if let Ok(s) = key.extract::<String>() {
        return datetime::parse_ns(&s)
            .ok_or_else(|| PyKeyError::new_err(format!("invalid datetime label {s:?}")));
    }
    if let Ok(i) = key.extract::<i64>() {
        return Ok(i);
    }
    Err(PyKeyError::new_err("label must be a datetime string or integer"))
}

/// Build the `.index` as a NumPy array (datetime64[ns] for a DatetimeIndex).
fn index_to_numpy<'py>(py: Python<'py>, index: &Index) -> PyResult<Bound<'py, PyAny>> {
    match index {
        Index::Datetime(v) => {
            let arr = v.clone().into_pyarray(py);
            Ok(arr.call_method1("astype", ("datetime64[ns]",))?)
        }
        Index::Int64(v) => Ok(v.clone().into_pyarray(py).into_any()),
        Index::Range(n) => Ok((0..*n as i64).collect::<Vec<_>>().into_pyarray(py).into_any()),
    }
}

/// Resolve a possibly-negative index to `[0, len)`.
fn norm_idx(i: isize, len: usize) -> PyResult<usize> {
    let n = len as isize;
    let i = if i < 0 { i + n } else { i };
    if i < 0 || i >= n {
        Err(PyIndexError::new_err("index out of range"))
    } else {
        Ok(i as usize)
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

    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        column_to_numpy(py, &self.inner.data)
    }

    /// NumPy array protocol, so `np.isnan(series)` etc. work directly.
    #[pyo3(signature = (dtype = None, copy = None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<PyObject>,
        copy: Option<PyObject>,
    ) -> Bound<'py, PyAny> {
        let _ = (dtype, copy);
        column_to_numpy(py, &self.inner.data)
    }

    /// NaN-skipping mean of the values.
    fn mean(&self) -> f64 {
        let v = self.inner.data.to_f64_vec();
        let (sum, cnt) = v
            .iter()
            .filter(|x| !x.is_nan())
            .fold((0.0, 0usize), |(s, c), &x| (s + x, c + 1));
        if cnt == 0 {
            f64::NAN
        } else {
            sum / cnt as f64
        }
    }

    /// pandas-style equality (NaN equals NaN, by value).
    fn equals(&self, other: &PySeries) -> bool {
        let a = self.inner.data.to_f64_vec();
        let b = other.inner.data.to_f64_vec();
        a.len() == b.len()
            && a.iter()
                .zip(&b)
                .all(|(&x, &y)| x == y || (x.is_nan() && y.is_nan()))
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
        let label = parse_label(key)?;
        let pos = self
            .inner
            .index
            .position_of(label)
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

fn series_binop(
    s: &Series,
    other: &Bound<'_, PyAny>,
    f: impl Fn(f64, f64) -> f64,
) -> PyResult<PySeries> {
    let a = s.data.to_f64_vec();
    let rhs = if let Ok(o) = other.extract::<PyRef<PySeries>>() {
        o.inner.data.to_f64_vec()
    } else if let Ok(scalar) = other.extract::<f64>() {
        vec![scalar; a.len()]
    } else {
        return Err(PyTypeError::new_err(
            "unsupported operand for Series arithmetic",
        ));
    };
    let n = a.len().min(rhs.len());
    let mut out = vec![f64::NAN; a.len()];
    for i in 0..n {
        out[i] = f(a[i], rhs[i]);
    }
    Ok(PySeries {
        inner: Series::new(s.name.clone(), Column::f64(out), Arc::clone(&s.index)),
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

/// A single DataFrame row (the result of `df.iloc[i]`); carries its index label
/// as `.name` and can be appended back.
#[pyclass(name = "Row")]
pub struct PyRow {
    names: Vec<String>,
    values: Vec<f64>,
    label: Py<PyAny>,
    index_value: i64,
    is_datetime: bool,
}

#[pymethods]
impl PyRow {
    #[getter]
    fn name(&self, py: Python<'_>) -> Py<PyAny> {
        self.label.clone_ref(py)
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        // not commonly used; expose the values as 1xN
        let data = self.values.clone();
        let w = data.len();
        ndarray::Array2::from_shape_vec((1, w), data)
            .unwrap()
            .into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!("Row(name={:?}, columns={:?})", "...", self.names)
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
    fn eval(&self, key: &str) -> Result<Column, VolasError> {
        if self.inner.has_column(key) {
            Ok(self.inner.column(key)?.clone())
        } else {
            let node = parse(key)?;
            execute(&self.inner, &node)
        }
    }

    fn wrap_series(&self, name: String, col: Column) -> PySeries {
        PySeries {
            inner: Series::new(Some(name), col, Arc::clone(self.inner.index())),
        }
    }
}

#[pymethods]
impl PyDataFrame {
    #[new]
    #[pyo3(signature = (data, date_col = None))]
    fn new(data: &Bound<'_, PyDict>, date_col: Option<String>) -> PyResult<Self> {
        let mut names = Vec::new();
        let mut columns = Vec::new();
        let mut index: Option<Index> = None;
        for (k, v) in data.iter() {
            let name: String = k.extract()?;
            if Some(&name) == date_col.as_ref() {
                let strings: Vec<String> = v.extract().map_err(|_| {
                    PyTypeError::new_err(format!("date_col {name:?} must be datetime strings"))
                })?;
                let mut epochs = Vec::with_capacity(strings.len());
                for s in &strings {
                    epochs.push(datetime::parse_ns(s).ok_or_else(|| {
                        PyValueError::new_err(format!("could not parse datetime {s:?}"))
                    })?);
                }
                index = Some(Index::Datetime(epochs));
                continue;
            }
            names.push(name);
            columns.push(pyany_to_column(&v)?);
        }
        let df = DataFrame::new(names, columns, index).map_err(pyerr)?;
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

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
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
            let col = self.eval(&name).map_err(pyerr)?;
            return Ok(Py::new(py, self.wrap_series(name, col))?.into_any());
        }
        // list of names / directives
        if let Ok(list) = key.extract::<Vec<String>>() {
            let mut cols = Vec::with_capacity(list.len());
            for n in &list {
                cols.push(self.eval(n).map_err(pyerr)?);
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
        &self,
        py: Python<'py>,
        directive: &str,
        create_column: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = create_column;
        let col = self.eval(directive).map_err(pyerr)?;
        Ok(column_to_numpy(py, &col))
    }

    /// The minimum number of prior rows a directive needs (its lookback).
    #[staticmethod]
    fn directive_lookback(directive: &str) -> PyResult<usize> {
        let node = parse(directive).map_err(pyerr)?;
        Ok(volas_directive::lookback::lookback(&node))
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

    /// Drop rows by index label (`axis=0`) — returns a new DataFrame.
    #[pyo3(signature = (labels, axis = 0))]
    fn drop(&self, labels: Vec<i64>, axis: i64) -> PyResult<PyDataFrame> {
        let _ = axis;
        let positions: Vec<usize> = (0..self.inner.height())
            .filter(|&i| {
                let lab = match self.inner.index().as_ref() {
                    Index::Range(_) => i as i64,
                    Index::Int64(v) | Index::Datetime(v) => v[i],
                };
                !labels.contains(&lab)
            })
            .collect();
        Ok(PyDataFrame {
            inner: take_frame(&self.inner, &positions),
        })
    }

    /// Append the rows of another DataFrame or a single Row, returning a new
    /// DataFrame (pandas semantics; not in place).
    fn append(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let mut inner = self.inner.clone();
        if let Ok(df) = other.extract::<PyRef<PyDataFrame>>() {
            inner.append(&df.inner).map_err(pyerr)?;
            return Ok(PyDataFrame { inner });
        }
        if let Ok(row) = other.extract::<PyRef<PyRow>>() {
            let mut names = Vec::new();
            let mut cols = Vec::new();
            for (n, v) in row.names.iter().zip(&row.values) {
                names.push(n.clone());
                cols.push(Column::f64(vec![*v]));
            }
            let one_index = if row.is_datetime {
                Index::Datetime(vec![row.index_value])
            } else {
                Index::Int64(vec![row.index_value])
            };
            let one = DataFrame::new(names, cols, Some(one_index)).map_err(pyerr)?;
            inner.append(&one).map_err(pyerr)?;
            return Ok(PyDataFrame { inner });
        }
        Err(PyTypeError::new_err("append expects a DataFrame or Row"))
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (data, h, w) = self.inner.to_row_major_f64();
        let arr = ndarray::Array2::from_shape_vec((h, w), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
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
        if let Ok(i) = key.extract::<isize>() {
            let i = norm_idx(i, self.inner.height())?;
            return Ok(Py::new(py, row_at(&self.inner, py, i))?.into_any());
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

fn row_at(df: &DataFrame, py: Python<'_>, i: usize) -> PyRow {
    let names = df.names().to_vec();
    let values: Vec<f64> = df.columns().iter().map(|c| c.get_f64(i)).collect();
    let label = label_to_py(py, df.index(), i);
    let (index_value, is_datetime) = match df.index().as_ref() {
        Index::Datetime(v) => (v[i], true),
        Index::Int64(v) => (v[i], false),
        Index::Range(_) => (i as i64, false),
    };
    PyRow {
        names,
        values,
        label,
        index_value,
        is_datetime,
    }
}

fn take_frame(df: &DataFrame, positions: &[usize]) -> DataFrame {
    let names = df.names().to_vec();
    let cols: Vec<Column> = df.columns().iter().map(|c| c.take(positions)).collect();
    let index = df.index().take(positions);
    DataFrame::new(names, cols, Some(index)).expect("take keeps shape")
}

/// Slice a frame by a Python slice — positional for integer bounds, label-based
/// (DatetimeIndex) for string bounds.
fn slice_frame(df: &DataFrame, slice: &Bound<'_, PySlice>) -> PyResult<DataFrame> {
    let start_obj = slice.getattr("start")?;
    let stop_obj = slice.getattr("stop")?;
    let is_label = start_obj.extract::<String>().is_ok() || stop_obj.extract::<String>().is_ok();
    if is_label {
        let lo = if start_obj.is_none() {
            None
        } else {
            Some(parse_label(&start_obj)?)
        };
        let hi = if stop_obj.is_none() {
            None
        } else {
            Some(parse_label(&stop_obj)?)
        };
        let (a, b) = df.index().label_slice(lo, hi);
        Ok(df.slice(a, b))
    } else {
        let info = slice.indices(df.height() as isize)?;
        let positions = strided(info.start, info.stop, info.step);
        Ok(take_frame(df, &positions))
    }
}

fn label_bounds(slice: &Bound<'_, PySlice>) -> PyResult<(Option<i64>, Option<i64>)> {
    let start_obj = slice.getattr("start")?;
    let stop_obj = slice.getattr("stop")?;
    let lo = if start_obj.is_none() {
        None
    } else {
        Some(parse_label(&start_obj)?)
    };
    let hi = if stop_obj.is_none() {
        None
    } else {
        Some(parse_label(&stop_obj)?)
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
        if let Ok(slice) = key.downcast::<PySlice>() {
            let (lo, hi) = label_bounds(slice)?;
            let (a, b) = self.inner.index().label_slice(lo, hi);
            return Ok(Py::new(py, PyDataFrame { inner: self.inner.slice(a, b) })?.into_any());
        }
        let label = parse_label(key)?;
        let pos = self
            .inner
            .index()
            .position_of(label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        Ok(Py::new(py, row_at(&self.inner, py, pos))?.into_any())
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
        let label = parse_label(key.0.bind(py))?;
        let i = self
            .inner
            .index()
            .position_of(label)
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
            let (lo, hi) = label_bounds(slice)?;
            let (a, b) = self.inner.index.label_slice(lo, hi);
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
        let label = parse_label(key)?;
        let pos = self
            .inner
            .index
            .position_of(label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        Ok(scalar_to_py(py, &self.inner.data, pos))
    }
}

// --- TimeFrame / cumulation -------------------------------------------------

/// `volas.TimeFrame` — an OHLCV sampling period.
#[pyclass(name = "TimeFrame")]
#[derive(Clone)]
pub struct PyTimeFrame {
    inner: TimeFrame,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyTimeFrame {
    fn __str__(&self) -> String {
        self.inner.label().to_string()
    }
    fn __repr__(&self) -> String {
        format!("TimeFrame.{}", self.inner.label())
    }
    #[getter]
    fn minutes(&self) -> i64 {
        self.inner.minutes()
    }
    /// Unify a timestamp (datetime string or epoch-ns int) to its period key.
    fn unify(&self, ts: &Bound<'_, PyAny>) -> PyResult<i64> {
        Ok(self.inner.unify(parse_label(ts)?))
    }

    #[classattr]
    fn s1() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Sec1 }
    }
    #[classattr]
    fn m1() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Min1 }
    }
    #[classattr]
    fn m3() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Min3 }
    }
    #[classattr]
    fn m5() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Min5 }
    }
    #[classattr]
    fn m15() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Min15 }
    }
    #[classattr]
    fn m30() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Min30 }
    }
    #[classattr]
    fn H1() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Hour1 }
    }
    #[classattr]
    fn H2() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Hour2 }
    }
    #[classattr]
    fn H4() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Hour4 }
    }
    #[classattr]
    fn H6() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Hour6 }
    }
    #[classattr]
    fn H8() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Hour8 }
    }
    #[classattr]
    fn H12() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Hour12 }
    }
    #[classattr]
    fn D1() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Day1 }
    }
    #[classattr]
    fn D3() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Day3 }
    }
    #[classattr]
    fn W1() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Week1 }
    }
    #[classattr]
    fn M1() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Month1 }
    }
    #[classattr]
    fn Y1() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Year1 }
    }
}

/// Resolve a `TimeFrame` from a `PyTimeFrame` or a label string.
fn resolve_time_frame(obj: &Bound<'_, PyAny>) -> PyResult<TimeFrame> {
    if let Ok(tf) = obj.extract::<PyRef<PyTimeFrame>>() {
        return Ok(tf.inner);
    }
    if let Ok(s) = obj.extract::<String>() {
        return TimeFrame::from_label(&s).map_err(pyerr);
    }
    Err(PyTypeError::new_err(
        "time_frame must be a TimeFrame or a label string like '5m'",
    ))
}

/// Build an aggregation spec from the OHLCV defaults plus optional overrides
/// (`{'volume': 'sum', 'open': 'first', ...}`).
fn build_agg_spec(cumulators: Option<&Bound<'_, PyDict>>) -> PyResult<AggSpec> {
    let mut spec = AggSpec::ohlcv();
    if let Some(dict) = cumulators {
        for (k, v) in dict.iter() {
            let name: String = k.extract()?;
            let agg_name: String = v.extract().map_err(|_| {
                PyTypeError::new_err("cumulator values must be aggregator names like 'sum'")
            })?;
            spec.set(name, Agg::from_name(&agg_name).map_err(pyerr)?);
        }
    }
    Ok(spec)
}

/// `volas.Cumulator` — a stateful, incremental OHLCV cumulator (live cum_append):
/// feed fine bars with `.append`, read the cumulated frame from `.frame`.
#[pyclass(name = "Cumulator")]
pub struct PyCumulator {
    inner: Cumulator,
}

#[pymethods]
impl PyCumulator {
    #[new]
    #[pyo3(signature = (time_frame, cumulators = None))]
    fn new(
        time_frame: &Bound<'_, PyAny>,
        cumulators: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let tf = resolve_time_frame(time_frame)?;
        let spec = build_agg_spec(cumulators)?;
        Ok(PyCumulator {
            inner: Cumulator::new(tf, spec),
        })
    }

    /// Feed fine bars (a DataFrame with a DatetimeIndex).
    fn append(&mut self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let df = data
            .extract::<PyRef<PyDataFrame>>()
            .map_err(|_| PyTypeError::new_err("Cumulator.append expects a DataFrame"))?;
        self.inner.append(&df.inner).map_err(pyerr)
    }

    /// The current cumulated frame (closed periods + the open period as the live
    /// last row).
    #[getter]
    fn frame(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.frame().map_err(pyerr)?,
        })
    }

    /// The current open period aggregated into a single live bar, or `None`.
    #[getter]
    fn last(&self) -> PyResult<Option<PyDataFrame>> {
        Ok(self
            .inner
            .last()
            .map_err(pyerr)?
            .map(|inner| PyDataFrame { inner }))
    }
}

/// Read a CSV file into a `DataFrame`, inferring per-column dtypes.
///
/// A pandas-subset of `pandas.read_csv`:
/// - `sep` / `delimiter` — field delimiter (single character; default `,`).
/// - `header` — `True`/omitted = first row is the header; `None`/`False` = no
///   header (columns named `"0".."n-1"`).
/// - `na_values` / `keep_default_na` — extra / default missing-value tokens.
/// - `parse_dates` — column names to parse into datetime columns.
/// - `index_col` — a column name or integer position to move into the row index;
///   applied after `parse_dates`, so naming a parsed date column yields a
///   `DatetimeIndex`.
#[pyfunction]
#[pyo3(signature = (
    path,
    sep = None,
    delimiter = None,
    header = Some(true),
    parse_dates = None,
    index_col = None,
    na_values = None,
    keep_default_na = true,
))]
#[allow(clippy::too_many_arguments)]
fn read_csv(
    path: String,
    sep: Option<String>,
    delimiter: Option<String>,
    header: Option<bool>,
    parse_dates: Option<Vec<String>>,
    index_col: Option<Bound<'_, PyAny>>,
    na_values: Option<Bound<'_, PyAny>>,
    keep_default_na: bool,
) -> PyResult<PyDataFrame> {
    // Resolve the delimiter (a single byte).
    let delim_str = delimiter.or(sep).unwrap_or_else(|| ",".to_string());
    let delim_bytes = delim_str.as_bytes();
    if delim_bytes.len() != 1 {
        return Err(PyValueError::new_err(
            "sep / delimiter must be a single-byte character",
        ));
    }

    // na_values: a string or a list of strings.
    let na_list: Vec<String> = match na_values {
        None => Vec::new(),
        Some(obj) => {
            if let Ok(s) = obj.extract::<String>() {
                vec![s]
            } else if let Ok(v) = obj.extract::<Vec<String>>() {
                v
            } else {
                return Err(PyTypeError::new_err(
                    "na_values must be a string or a list of strings",
                ));
            }
        }
    };

    let opts = volas_io::ReadCsvOptions {
        delimiter: delim_bytes[0],
        has_header: matches!(header, Some(true)),
        na_values: na_list,
        keep_default_na,
    };
    let mut df = volas_io::read_csv(&path, &opts).map_err(pyerr)?;

    // parse_dates: convert each named column to a datetime column in place.
    if let Some(cols) = parse_dates {
        for name in &cols {
            let parsed = df.column(name).map_err(pyerr)?.to_datetime().map_err(pyerr)?;
            df.set_column(name, parsed).map_err(pyerr)?;
        }
    }

    // index_col: move a column (by name or position) into the row index.
    if let Some(ic) = index_col {
        let name = if let Ok(s) = ic.extract::<String>() {
            s
        } else if let Ok(i) = ic.extract::<isize>() {
            let pos = norm_idx(i, df.width())?;
            df.names()[pos].clone()
        } else {
            return Err(PyTypeError::new_err(
                "index_col must be a column name or an integer position",
            ));
        };
        df = df.set_index(&name).map_err(pyerr)?;
    }

    Ok(PyDataFrame { inner: df })
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
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    Ok(())
}
