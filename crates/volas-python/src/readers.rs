//! `volas.read_csv` — the CSV reader binding.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use volas_core::Tz;
use volas_io::DateParse;

use crate::{norm_idx, pyerr, PyDataFrame};

/// Read a CSV file into a ``DataFrame``, inferring per-column dtypes.
///
/// A pandas-subset of ``pandas.read_csv``.
///
/// Args:
///     path (str | os.PathLike): path to the CSV file (a ``str`` or any
///         ``os.PathLike``, e.g. ``pathlib.Path``).
///     sep (str, optional): field delimiter (single character; default ``','``).
///     delimiter (str, optional): alias for ``sep``.
///     header (bool, optional): ``True`` / omitted = the first row is the header;
///         ``None`` / ``False`` = no header (columns named ``"0".."n-1"``).
///     parse_dates (list[str], optional): column names to parse into datetime
///         columns, from their text tokens (an all-missing column is all NaT).
///     index_col (str | int, optional): a column name or integer position to
///         move into the row index; applied after ``parse_dates``, so naming a
///         parsed date column yields a DatetimeIndex.
///     na_values (str | list[str], optional): extra missing-value tokens.
///     keep_default_na (bool): also treat the default tokens as missing
///         (default True).
///     tz (str, optional): timezone for the ``index_col`` datetime — a *naive*
///         date string is interpreted in ``tz`` (stored UTC, the index tagged
///         with ``tz``, ``'UTC'`` included);
///         accepts a fixed offset (``'+08:00'``) or IANA name
///         (``'America/New_York'``). The column may also be listed in
///         ``parse_dates``.
///     date_unit (str, optional): read ``index_col`` as an epoch number in this
///         unit, exactly from its text (``'s'`` / ``'ms'`` / ``'us'`` / ``'ns'``,
///         absolute UTC; a
///         fractional epoch is kept to the nearest nanosecond); ``tz`` then only
///         sets the display zone.
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
    py: Python<'_>,
    path: std::path::PathBuf,
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
    let table = volas_io::read_csv_table(&path, &opts).map_err(pyerr)?;

    // index_col: a column name or position (resolved against the header).
    let index_name = match index_col {
        None => None,
        Some(ic) => Some(if let Ok(s) = ic.extract::<String>() {
            s
        } else if let Ok(i) = ic.extract::<isize>() {
            table.names()[norm_idx(i, table.names().len())?].clone()
        } else {
            return Err(PyTypeError::new_err(
                "index_col must be a column name or an integer position",
            ));
        }),
    };
    let tzv = tz.as_deref().map(|name| crate::timezone::parse(py, name)).transpose()?;
    if index_name.is_none() && (tzv.is_some() || date_unit.is_some()) {
        return Err(PyValueError::new_err(
            "tz / date_unit require index_col (the column to use as the datetime index)",
        ));
    }

    // Declared date columns are parsed from their own tokens, before inference:
    // parse_dates as datetime text, and the index column with tz (naive text read
    // in tz) or date_unit (an exact epoch number).
    let mut dates: Vec<(String, DateParse)> = parse_dates
        .unwrap_or_default()
        .into_iter()
        .map(|name| (name, DateParse::Text(Tz::Naive)))
        .collect();
    if let Some(name) = index_name.as_ref().filter(|_| tzv.is_some() || date_unit.is_some()) {
        let parse = match &date_unit {
            Some(unit) => DateParse::Epoch(unit.clone()),
            None => DateParse::Text(tzv.clone().unwrap_or_default()),
        };
        dates.retain(|(n, _)| n != name);
        dates.push((name.clone(), parse));
    }
    let mut df = table.into_frame(&dates).map_err(pyerr)?;

    if let Some(name) = index_name {
        df = df.set_index(&name).map_err(pyerr)?;
        // the instants are already UTC; any tz (UTC included) tags the index aware
        if let Some(tz) = tzv {
            df = df.set_index_tz(tz).map_err(pyerr)?;
        }
    }
    PyDataFrame::validate_schema(&df)?;
    Ok(PyDataFrame::plain(df))
}
