//! PyO3 bindings: expose the `volas-core` kernel to Python as
//! `volas_rs.DataFrame` / `volas_rs.Series` (re-exported by the `volas` package).
//!
//! This crate is the only place pyo3 / numpy are used; all logic lives in
//! `volas-core`.

use std::sync::Arc;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use volas_core::directive::{execute, parse};
use volas_core::{Column, DataFrame, Series, VolasError};

/// Map a core error to the closest Python exception.
fn pyerr(e: VolasError) -> PyErr {
    match e {
        VolasError::ColumnNotFound(n) => PyKeyError::new_err(format!("column \"{n}\" not found")),
        VolasError::DType(m) => PyTypeError::new_err(m),
        VolasError::Shape(m) | VolasError::Index(m) | VolasError::Value(m) => {
            PyValueError::new_err(m)
        }
    }
}

/// Convert a Python column value (1-D numpy array or list of numbers) to a [`Column`].
fn pyany_to_column(v: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(a) = v.extract::<PyReadonlyArray1<f64>>() {
        return Ok(Column::F64(a.as_slice()?.to_vec()));
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<i64>>() {
        return Ok(Column::I64(a.as_slice()?.to_vec()));
    }
    if let Ok(a) = v.extract::<PyReadonlyArray1<bool>>() {
        return Ok(Column::Bool(a.as_slice()?.to_vec()));
    }
    if let Ok(vv) = v.extract::<Vec<f64>>() {
        return Ok(Column::F64(vv));
    }
    Err(PyTypeError::new_err(
        "column values must be a 1-D numeric array or a list of numbers",
    ))
}

/// Export a column to a 1-D NumPy array of the appropriate dtype.
fn column_to_numpy<'py>(py: Python<'py>, col: &Column) -> Bound<'py, PyAny> {
    match col {
        Column::F64(v) => v.clone().into_pyarray(py).into_any(),
        Column::Bool(v) => v.clone().into_pyarray(py).into_any(),
        Column::I64(v) => v.clone().into_pyarray(py).into_any(),
    }
}

/// `volas.Series` — a single named, indexed column.
#[pyclass(name = "Series")]
pub struct PySeries {
    inner: Series,
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

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Export the values to a 1-D NumPy array.
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        column_to_numpy(py, &self.inner.data)
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

/// `volas.DataFrame` — an ordered, named, time-indexed table of columns with
/// stock-pandas-style directive indexing.
#[pyclass(name = "DataFrame")]
pub struct PyDataFrame {
    inner: DataFrame,
}

impl PyDataFrame {
    /// Resolve `key` to a column: an existing column, or a computed directive.
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
    fn new(data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut names = Vec::with_capacity(data.len());
        let mut columns = Vec::with_capacity(data.len());
        for (k, v) in data.iter() {
            names.push(k.extract::<String>()?);
            columns.push(pyany_to_column(&v)?);
        }
        let df = DataFrame::new(names, columns, None).map_err(pyerr)?;
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

    fn __len__(&self) -> usize {
        self.inner.height()
    }

    /// `df[key]`:
    /// - a boolean Series / numpy bool array -> filtered DataFrame
    /// - a column name or directive string -> Series
    /// - a list of names / directives -> DataFrame
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
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
        if let Ok(name) = key.extract::<String>() {
            let col = self.eval(&name).map_err(pyerr)?;
            return Ok(Py::new(py, self.wrap_series(name, col))?.into_any());
        }
        if let Ok(names) = key.extract::<Vec<String>>() {
            let mut cols = Vec::with_capacity(names.len());
            for n in &names {
                cols.push(self.eval(n).map_err(pyerr)?);
            }
            let index = (*self.inner.index().as_ref()).clone();
            let df = DataFrame::new(names, cols, Some(index)).map_err(pyerr)?;
            return Ok(Py::new(py, PyDataFrame { inner: df })?.into_any());
        }
        Err(PyKeyError::new_err(
            "key must be a column name, directive, list of those, or a boolean mask",
        ))
    }

    /// Execute a directive and return the result as a NumPy array.
    #[pyo3(signature = (directive, create_column = false))]
    fn exec<'py>(
        &self,
        py: Python<'py>,
        directive: &str,
        create_column: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = create_column; // column caching is not implemented in v1
        let col = self.eval(directive).map_err(pyerr)?;
        Ok(column_to_numpy(py, &col))
    }

    /// Get a column by name as a Series (raises KeyError if missing).
    fn get_column(&self, key: &str) -> PyResult<PySeries> {
        let col = self.inner.column(key).map_err(pyerr)?.clone();
        Ok(self.wrap_series(key.to_string(), col))
    }

    /// Append the rows of another DataFrame (matched by column name), in place.
    fn append(&mut self, other: &PyDataFrame) -> PyResult<()> {
        self.inner.append(&other.inner).map_err(pyerr)
    }

    /// Export to a 2-D (row-major) NumPy `float64` array.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (data, h, w) = self.inner.to_row_major_f64();
        let arr = ndarray::Array2::from_shape_vec((h, w), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py))
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

/// The compiled module backing the `volas` package.
#[pymodule]
fn volas_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PySeries>()?;
    Ok(())
}
