//! PyO3 bindings: expose the `volas-core` kernel to Python as
//! `volas_rs.DataFrame` / `volas_rs.Series` (re-exported by the `volas` package).
//!
//! This crate is the only place pyo3 / numpy are used; all logic lives in
//! `volas-core`.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
        match &self.inner.data {
            Column::F64(v) => v.clone().into_pyarray(py).into_any(),
            Column::Bool(v) => v.clone().into_pyarray(py).into_any(),
            Column::I64(v) => v.clone().into_pyarray(py).into_any(),
        }
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

/// `volas.DataFrame` — an ordered, named, time-indexed table of columns.
#[pyclass(name = "DataFrame")]
pub struct PyDataFrame {
    inner: DataFrame,
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

    /// `df[key]`: a column name -> Series, or a list of names -> DataFrame.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(name) = key.extract::<String>() {
            let s = self.inner.series(&name).map_err(pyerr)?;
            return Ok(Py::new(py, PySeries { inner: s })?.into_any());
        }
        if let Ok(names) = key.extract::<Vec<String>>() {
            let sub = self.inner.select(&names).map_err(pyerr)?;
            return Ok(Py::new(py, PyDataFrame { inner: sub })?.into_any());
        }
        Err(PyKeyError::new_err(
            "key must be a column name or a list of column names",
        ))
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
