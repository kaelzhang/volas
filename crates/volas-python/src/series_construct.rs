//! `Series` accessors and construction-adjacent surface (name / dtype / index /
//! timezone / shape / the `.iloc`/`.loc`/`.iat`/`.at` accessors / `.dt`).

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volas_core::{
    Column, Series, Tz,
};

#[allow(unused_imports)]
use crate::*;

#[pymethods]
impl PySeries {
    /// The series name — the column it was drawn from, or ``None``.
    ///
    /// Returns:
    ///     str | None
    #[getter]
    pub(crate) fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }

    /// The dtype name (``'float64'``, ``'float32'``, ``'int64'``, ``'int32'``,
    /// ``'bool'``, ``'str'``, or ``'datetime64[ns]'`` — never ``'object'``, which
    /// volas has no dtype for).
    ///
    /// Returns:
    ///     str
    #[getter]
    pub(crate) fn dtype(&self) -> String {
        self.inner.dtype().to_string()
    }

    /// The row index shared with the parent frame, as a NumPy array (a
    /// ``datetime64[ns]`` array for a DatetimeIndex, an object array for a string
    /// index).
    ///
    /// Returns:
    ///     numpy.ndarray
    #[getter]
    pub(crate) fn index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        index_to_numpy(py, &self.inner.index)
    }

    /// The DatetimeIndex timezone name, or `None` for a tz-naive / non-datetime
    /// index (mirrors `df.tz`).
    #[getter]
    pub(crate) fn tz(&self) -> Option<String> {
        match self.inner.index.tz() {
            Tz::Naive => None,
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
    pub(crate) fn iloc(&self) -> SeriesILoc {
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
    pub(crate) fn loc(&self) -> SeriesLoc {
        SeriesLoc {
            inner: self.inner.clone(),
        }
    }

    /// The shape as a 1-tuple `(len,)` (pandas `Series.shape`).
    #[getter]
    pub(crate) fn shape(&self) -> (usize,) {
        (self.inner.len(),)
    }

    pub(crate) fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Guard the ambiguous `if series:` footgun: a Series has a single truth value
    /// only when it holds exactly one element (pandas-style).
    pub(crate) fn __bool__(&self) -> PyResult<bool> {
        match self.inner.len() {
            1 => Ok(to_bool_vec(&self.inner.data)[0]),
            _ => Err(PyValueError::new_err(
                "The truth value of a Series is ambiguous — use s.any() or s.all()",
            )),
        }
    }

    /// Anchor a NAIVE DatetimeIndex's wall-clock in `tz` (pandas `tz_localize`)
    /// — the Series-level mirror of `df.tz_localize` (F27).
    pub(crate) fn tz_localize(&self, tz: &str) -> PyResult<PySeries> {
        let tzv = Tz::parse(tz).map_err(pyerr)?;
        let df = volas_core::DataFrame::new(
            vec![self.inner.name.clone().unwrap_or_else(|| "x".to_string())],
            vec![self.inner.data.clone()],
            Some((*self.inner.index).clone()),
        )
        .map_err(pyerr)?
        .tz_localize(tzv)
        .map_err(pyerr)?;
        Ok(PySeries {
            inner: Series::new(
                self.inner.name.clone(),
                df.columns()[0].clone(),
                Arc::clone(df.index()),
            ),
        })
    }

    /// Restate an AWARE DatetimeIndex in another zone (pandas `tz_convert`) —
    /// the Series-level mirror of `df.tz_convert` (F27).
    pub(crate) fn tz_convert(&self, tz: &str) -> PyResult<PySeries> {
        let tzv = Tz::parse(tz).map_err(pyerr)?;
        let df = volas_core::DataFrame::new(
            vec![self.inner.name.clone().unwrap_or_else(|| "x".to_string())],
            vec![self.inner.data.clone()],
            Some((*self.inner.index).clone()),
        )
        .map_err(pyerr)?
        .tz_convert(tzv)
        .map_err(pyerr)?;
        Ok(PySeries {
            inner: Series::new(
                self.inner.name.clone(),
                df.columns()[0].clone(),
                Arc::clone(df.index()),
            ),
        })
    }

    /// Scalar positional accessor (pandas `iat`): `s.iat[i]` == `s.iloc[i]`.
    #[getter]
    pub(crate) fn iat(&self) -> SeriesILoc {
        SeriesILoc { inner: self.inner.clone() }
    }

    /// Scalar label accessor (pandas `at`): `s.at[label]` == `s.loc[label]`.
    #[getter]
    pub(crate) fn at(&self) -> SeriesLoc {
        SeriesLoc { inner: self.inner.clone() }
    }

    /// The datetime accessor (pandas `Series.dt`): per-element calendar
    /// components / predicates / names / floor-ceil-round, on a
    /// `datetime64[ns]` Series only.
    #[getter]
    pub(crate) fn dt(&self) -> PyResult<crate::dt::PyDt> {
        if !matches!(self.inner.data, Column::Datetime(_)) {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "Can only use .dt accessor with datetimelike values",
            ));
        }
        Ok(crate::dt::PyDt { series: self.inner.clone() })
    }
}
