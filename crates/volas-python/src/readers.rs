//! `volas.read_csv` — the CSV reader binding.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::{build_datetime_index, norm_idx, pyerr, PyDataFrame};

/// Read a CSV file into a ``DataFrame``, inferring per-column dtypes.
///
/// A pandas-subset of ``pandas.read_csv``.
///
/// Args:
///     path (str): path to the CSV file.
///     sep (str, optional): field delimiter (single character; default ``','``).
///     delimiter (str, optional): alias for ``sep``.
///     header (bool, optional): ``True`` / omitted = the first row is the header;
///         ``None`` / ``False`` = no header (columns named ``"0".."n-1"``).
///     parse_dates (list[str], optional): column names to parse into datetime
///         columns.
///     index_col (str | int, optional): a column name or integer position to
///         move into the row index; applied after ``parse_dates``, so naming a
///         parsed date column yields a DatetimeIndex.
///     na_values (str | list[str], optional): extra missing-value tokens.
///     keep_default_na (bool): also treat the default tokens as missing
///         (default True).
///     tz (str, optional): timezone for the ``index_col`` datetime — a *naive*
///         date string is interpreted in ``tz`` (stored UTC, the index tagged);
///         accepts a fixed offset (``'+08:00'``) or IANA name
///         (``'America/New_York'``). For tz ingestion pass the date column via
///         ``index_col`` and do *not* also list it in ``parse_dates``.
///     date_unit (str, optional): read ``index_col`` as an epoch integer in this
///         unit (``'s'`` / ``'ms'`` / ``'us'`` / ``'ns'``, absolute UTC); ``tz``
///         then only sets the display zone.
///
/// Usage::
///
///     df = volas.read_csv('ohlcv.csv')
///     df = volas.read_csv('ohlcv.csv', index_col='time', tz='America/New_York')
///
/// Returns:
///     DataFrame
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
    tz = None,
    date_unit = None,
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
    tz: Option<String>,
    date_unit: Option<String>,
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

    // `header`: omitted / `True` -> the first row is the header; `None` / `False`
    // -> no header (columns named "0".."n-1"). This matches pandas for the
    // `None` / bool forms (the integer `header=N` position form is not supported;
    // `header=0`, pandas's default, is the omitted / `True` case).
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
        // With tz / date_unit, parse + tag the datetime index; else a plain move.
        df = if tz.is_some() || date_unit.is_some() {
            build_datetime_index(df, &name, tz.as_deref(), date_unit.as_deref())?
        } else {
            df.set_index(&name).map_err(pyerr)?
        };
    } else if tz.is_some() || date_unit.is_some() {
        return Err(PyValueError::new_err(
            "tz / date_unit require index_col (the column to use as the datetime index)",
        ));
    }

    Ok(PyDataFrame { inner: df })
}
