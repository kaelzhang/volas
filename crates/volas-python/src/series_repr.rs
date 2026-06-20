//! `Series` text rendering (`__repr__` / `__str__` / `to_string`).


use pyo3::prelude::*;

use crate::format::{render_series, NA_REPR};
#[allow(unused_imports)]
use crate::*;

#[pymethods]
impl PySeries {

    /// pandas-style vertical repr (`label   value` rows + a
    /// `Name: <name>, dtype: <dtype>` footer), truncating to 5 head + 5 tail rows
    /// past 60 (`display.max_rows` / `min_rows`). `str` and `repr` are identical.
    pub(crate) fn __repr__(&self) -> String {
        let truncate = if self.inner.len() > 60 { Some(5) } else { None };
        render_series(&self.inner, NA_REPR, None, truncate, true)
    }

    pub(crate) fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Render the whole series as text (pandas `Series.to_string`): no truncation
    /// by default and no `Name/dtype` footer; `max_rows` truncates.
    #[pyo3(signature = (na_rep = NA_REPR, float_format = None, max_rows = None))]
    pub(crate) fn to_string(
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
}
