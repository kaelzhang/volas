//! `volas.Row` — a single DataFrame row (the result of `df.iloc[i]` /
//! `df.loc[label]`), a faithful 1-row frame carrying its typed cells + index label.

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use volas_core::DataFrame;

use crate::format::render_row;
#[allow(unused_imports)]
use crate::*;

/// A single DataFrame row (the result of `df.iloc[i]` / `df.loc[label]`): a
/// faithful 1-row frame carrying its index label and every column's *typed*
/// value (no lossy f64 coercion, no flag pair to remember the index kind).
#[pyclass(name = "Row")]
pub struct PyRow {
    pub(crate) inner: DataFrame,
}

#[pymethods]
impl PyRow {
    /// The row's index label.
    #[getter]
    pub(crate) fn name(&self, py: Python<'_>) -> Py<PyAny> {
        label_to_py(py, self.inner.index(), 0)
    }

    /// A single value by column name (``row[col]``).
    ///
    /// Returns:
    ///     the typed scalar at that column.
    pub(crate) fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
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
    pub(crate) fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        for c in self.inner.columns() {
            c.require_numeric().map_err(pyerr)?;
        }
        // F14: a Row is a single 1-D record -> shape (n,), like pandas
        // df.iloc[0].to_numpy() (was a 2-D (1, n) frame export).
        let (data, _h, _w) = self.inner.to_row_major_f64();
        Ok(data.into_pyarray(py))
    }

    /// The row as a typed `{column: value}` dict (pandas `Series.to_dict`).
    pub(crate) fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
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
    pub(crate) fn __repr__(&self) -> String {
        render_row(&self.inner, true)
    }

    pub(crate) fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Render the row as text without the `Name` footer (pandas
    /// `Series.to_string`).
    // Pandas-parity method name, exposed to Python; not a Rust `Display` impl.
    #[allow(clippy::inherent_to_string)]
    pub(crate) fn to_string(&self) -> String {
        render_row(&self.inner, false)
    }
}
