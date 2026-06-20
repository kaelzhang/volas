//! `DataFrame` text / HTML rendering (`__repr__` / `to_string` / `_repr_html_`).


use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;

use crate::format::{
    render_frame, render_frame_html, Dimensions,
    DisplayOpts, NA_REPR,
};
#[allow(unused_imports)]
use crate::*;

#[pymethods]
impl PyDataFrame {

    /// pandas-style aligned-table repr: a left-justified index column + right-
    /// justified data columns, truncating to 5 head + 5 tail rows past 60
    /// (`display.max_rows` / `min_rows`) with a `[N rows x M columns]` footer.
    /// `str` and `repr` are identical.
    pub(crate) fn __repr__(&self) -> PyResult<String> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        let truncate = if df.height() > 60 { Some(5) } else { None };
        let opts = DisplayOpts {
            header: true,
            index: true,
            na_rep: NA_REPR,
            float_format: None,
            dimensions: Dimensions::OnTruncate,
            truncate,
        };
        let cols: Vec<usize> = (0..df.width()).collect();
        Ok(render_frame(df, &cols, &opts))
    }

    pub(crate) fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }

    /// Render the whole frame as text (pandas `DataFrame.to_string`), implementing
    /// the core parameters. No truncation by default; `max_rows` truncates to 5
    /// head + 5 tail (or `min_rows`). Legacy / non-applicable pandas params
    /// (`sparsify`, `index_names`, `col_space`, `justify`, `formatters`,
    /// `line_width`, `encoding`, `decimal`, `buf`) are intentionally omitted.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (columns = None, header = true, index = true, na_rep = NA_REPR, float_format = None, max_rows = None, min_rows = None, show_dimensions = false))]
    pub(crate) fn to_string(
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
        let view = self.logical();
        let df = view.as_ref();
        let ff = parse_ff(float_format)?;
        let col_pos: Vec<usize> = match &columns {
            Some(cols) => cols
                .iter()
                .map(|n| {
                    df.column_pos(n)
                        .ok_or_else(|| PyKeyError::new_err(format!("column \"{n}\" not found")))
                })
                .collect::<PyResult<_>>()?,
            None => (0..df.width()).collect(),
        };
        let truncate = match max_rows {
            Some(m) if df.height() > m => Some((min_rows.unwrap_or(m) / 2).max(1)),
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
        Ok(render_frame(df, &col_pos, &opts))
    }

    /// Rich HTML table for Jupyter (`_repr_html_`). pandas defines this only on
    /// DataFrame — a Series falls back to its text repr — so volas matches and
    /// exposes it on DataFrame alone.
    pub(crate) fn _repr_html_(&self) -> PyResult<String> {
        ensure_fresh(&self.inner)?;
        Ok(render_frame_html(self.logical().as_ref()))
    }
}
