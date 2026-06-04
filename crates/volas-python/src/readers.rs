//! `volas.read_csv` — the CSV reader binding.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::{norm_idx, pyerr, PyDataFrame};

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
pub fn read_csv(
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
