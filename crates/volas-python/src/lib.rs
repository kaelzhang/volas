//! PyO3 bindings: expose the `volas` kernel to Python as `volas_rs.DataFrame` /
//! `volas_rs.Series`, with stock-pandas-style directive indexing and a
//! pandas-compatible indexing surface (`.iloc`, `.index`, `.name`, label lookup).
//!
//! This crate is the only place pyo3 / numpy are used; all logic lives in the
//! `volas-core` / `volas-compute` / `volas-directive` crates.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::create_exception;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySlice, PySliceIndices, PyTuple};

use volas_core::{datetime, Column, DataFrame, DType, Index, Label, Series, Tz, VolasError};
use volas_directive::{execute, parse};

mod readers;
mod timeframe;

use readers::read_csv;
use timeframe::{build_agg_spec, resolve_time_frame, PyCumulator, PyTimeFrame};

// --- helpers ---------------------------------------------------------------

pub(crate) fn pyerr(e: VolasError) -> PyErr {
    match e {
        VolasError::ColumnNotFound(n) => PyKeyError::new_err(format!("column \"{n}\" not found")),
        VolasError::DType(m) => PyTypeError::new_err(m),
        VolasError::Shape(m) | VolasError::Index(m) | VolasError::Value(m) => {
            PyValueError::new_err(m)
        }
    }
}

// Typed directive exceptions (both subclass ValueError, so existing
// `except ValueError` keeps working while callers can catch the specific type).
create_exception!(
    volas_rs,
    DirectiveError,
    PyValueError,
    "Base class for volas directive errors."
);
create_exception!(
    volas_rs,
    DirectiveSyntaxError,
    DirectiveError,
    "A directive string could not be parsed (with line / column)."
);
create_exception!(
    volas_rs,
    DirectiveValueError,
    DirectiveError,
    "A directive has an unknown command / sub-command or an invalid argument."
);

/// Map a parse-time error to `DirectiveSyntaxError`.
fn syntax_err(e: VolasError) -> PyErr {
    match e {
        VolasError::Value(m) => DirectiveSyntaxError::new_err(m),
        other => DirectiveSyntaxError::new_err(other.to_string()),
    }
}

/// Map a directive execution error to `DirectiveValueError`.
fn value_err(e: VolasError) -> PyErr {
    match e {
        VolasError::Value(m) => DirectiveValueError::new_err(m),
        VolasError::ColumnNotFound(n) => {
            DirectiveValueError::new_err(format!("column \"{n}\" not found"))
        }
        other => pyerr(other),
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
    // A Python list infers its dtype like pandas/NumPy: all-bool -> bool, all-int
    // -> int64, else float64 (empty -> float64). Bool / int are tried before float
    // because a strict bool/int extraction won't match a float list.
    if let Ok(vv) = v.extract::<Vec<bool>>() {
        if !vv.is_empty() {
            return Ok(Column::bool(vv));
        }
    }
    if let Ok(vv) = v.extract::<Vec<i64>>() {
        if !vv.is_empty() {
            return Ok(Column::i64(vv));
        }
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

/// Move an `Arc<Vec<T>>` out to an owned `Vec<T>` without copying when it is
/// uniquely owned; clone only if it is still shared.
fn arc_into_vec<T: Clone>(arc: Arc<Vec<T>>) -> Vec<T> {
    Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())
}

/// Like [`column_to_numpy`] but **consumes** the column, moving its backing `Vec`
/// straight into the NumPy array with no copy when the column is uniquely owned —
/// the fresh-result path (`df.exec(directive)`). Falls back to a borrow + copy for
/// the rarer `Str` / `Datetime` columns.
fn column_into_numpy<'py>(py: Python<'py>, col: Column) -> Bound<'py, PyAny> {
    match col {
        Column::F64(a) => arc_into_vec(a).into_pyarray(py).into_any(),
        Column::Bool(a) => arc_into_vec(a).into_pyarray(py).into_any(),
        Column::I64(a) => arc_into_vec(a).into_pyarray(py).into_any(),
        other => column_to_numpy(py, &other),
    }
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
        Index::Datetime(v, tz) => datetime::format_ns_tz(v[i], *tz)
            .into_pyobject(py)
            .unwrap()
            .into_any()
            .unbind(),
        Index::Int64(v) => v[i].into_pyobject(py).unwrap().into_any().unbind(),
        Index::Range(_) => (i as i64).into_pyobject(py).unwrap().into_any().unbind(),
        Index::Str(v) => v[i].clone().into_pyobject(py).unwrap().into_any().unbind(),
    }
}

/// Parse a Python timestamp to UTC epoch-ns, interpreting a **naive** string in
/// `tz`. A [`PyTimestamp`] carries its own tz and resolves to an absolute instant
/// (so it matches across zones); an offset-aware string is already absolute; an
/// integer is epoch-ns.
pub(crate) fn parse_ts_in_tz(key: &Bound<'_, PyAny>, tz: Tz) -> PyResult<i64> {
    if let Ok(ts) = key.extract::<PyRef<PyTimestamp>>() {
        return Ok(ts.ns);
    }
    if let Ok(s) = key.extract::<String>() {
        return datetime::parse_ns_in_tz(&s, tz)
            .ok_or_else(|| PyKeyError::new_err(format!("invalid datetime label {s:?}")));
    }
    if let Ok(i) = key.extract::<i64>() {
        return Ok(i);
    }
    Err(PyKeyError::new_err("label must be a datetime string or integer"))
}

/// ``volas.Timestamp(value, tz=None)`` — a typed datetime label carrying its own
/// timezone, resolving to an absolute **UTC** instant.
///
/// Use it for precise / cross-tz ``.loc`` / ``.loc[a:b]`` / ``.at`` lookups: a
/// Timestamp built in one zone matches the right row of a frame displayed in
/// another, because both compare on the UTC axis. (A bare string label is
/// instead interpreted in the index's own tz.)
///
/// Args:
///     value (str | int): a datetime string (e.g. ``'2021-01-04 09:30'``) or
///         epoch nanoseconds. A naive string is interpreted in ``tz``.
///     tz (str, optional): the zone the value is given in, e.g.
///         ``'America/New_York'`` or ``'+08:00'`` (default UTC).
///
/// Usage::
///
///     ts = volas.Timestamp('2021-01-04 09:30', tz='America/New_York')
///     df.at[ts, 'close']    # matches the right row across timezones
#[pyclass(name = "Timestamp")]
pub struct PyTimestamp {
    /// UTC epoch-ns (the absolute instant).
    ns: i64,
    /// The zone `value` was specified in (for display).
    tz: Tz,
}

#[pymethods]
impl PyTimestamp {
    // Constructor — args & usage live in the class docstring (pyo3 does not
    // surface a `#[new]` doc comment to Python).
    #[new]
    #[pyo3(signature = (value, tz = None))]
    fn new(value: &Bound<'_, PyAny>, tz: Option<String>) -> PyResult<Self> {
        let tzv = match tz {
            Some(s) => Tz::parse(&s).map_err(pyerr)?,
            None => Tz::Utc,
        };
        Ok(PyTimestamp {
            ns: parse_ts_in_tz(value, tzv)?,
            tz: tzv,
        })
    }

    /// The absolute instant as UTC epoch nanoseconds.
    #[getter]
    fn value(&self) -> i64 {
        self.ns
    }

    /// The timezone name, or `None` if UTC / unspecified.
    #[getter]
    fn tz(&self) -> Option<String> {
        match self.tz {
            Tz::Utc => None,
            other => Some(other.name()),
        }
    }

    /// The wall-clock as a NumPy `datetime64[ns]` (UTC instant).
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let arr = vec![self.ns].into_pyarray(py);
        Ok(arr.call_method1("astype", ("datetime64[ns]",))?)
    }

    fn __repr__(&self) -> String {
        match self.tz {
            Tz::Utc => format!("Timestamp('{}')", datetime::format_ns(self.ns)),
            other => format!(
                "Timestamp('{}', tz='{}')",
                datetime::format_ns_tz(self.ns, other),
                other.name()
            ),
        }
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: pyo3::basic::CompareOp) -> PyResult<bool> {
        let rhs = parse_ts_in_tz(other, Tz::Utc)?;
        Ok(op.matches(self.ns.cmp(&rhs)))
    }

    fn __hash__(&self) -> i64 {
        self.ns
    }
}

/// Parse a Python timestamp (datetime string or epoch-ns integer) to UTC ns,
/// interpreting a naive string as UTC.
pub(crate) fn parse_ts(key: &Bound<'_, PyAny>) -> PyResult<i64> {
    parse_ts_in_tz(key, Tz::Utc)
}

/// Parse a Python label to the [`Label`] kind expected by `index`: a string for
/// a string index, a parsed datetime (in the index's tz) / integer for the
/// numeric kinds.
pub(crate) fn parse_label(key: &Bound<'_, PyAny>, index: &Index) -> PyResult<Label> {
    match index {
        Index::Str(_) => key
            .extract::<String>()
            .map(Label::Str)
            .map_err(|_| PyKeyError::new_err("label must be a string for a string index")),
        Index::Datetime(_, tz) => parse_ts_in_tz(key, *tz).map(Label::I64),
        _ => parse_ts(key).map(Label::I64),
    }
}

/// Build the `.index` as a NumPy array. A DatetimeIndex exports its **UTC**
/// instants as `datetime64[ns]` (matching pandas `.values`; the frame tz governs
/// string rendering / matching, not the numeric export); a string index becomes
/// an object array.
fn index_to_numpy<'py>(py: Python<'py>, index: &Index) -> PyResult<Bound<'py, PyAny>> {
    match index {
        Index::Datetime(v, _) => {
            let arr = v.clone().into_pyarray(py);
            Ok(arr.call_method1("astype", ("datetime64[ns]",))?)
        }
        Index::Int64(v) => Ok(v.clone().into_pyarray(py).into_any()),
        Index::Range(n) => Ok((0..*n as i64).collect::<Vec<_>>().into_pyarray(py).into_any()),
        Index::Str(v) => {
            let list = PyList::new(py, v.as_slice())?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", "object")?;
            Ok(py.import("numpy")?.call_method("array", (list,), Some(&kwargs))?)
        }
    }
}

/// Resolve a possibly-negative index to `[0, len)`.
pub(crate) fn norm_idx(i: isize, len: usize) -> PyResult<usize> {
    let n = len as isize;
    let i = if i < 0 { i + n } else { i };
    if i < 0 || i >= n {
        Err(PyIndexError::new_err("index out of range"))
    } else {
        Ok(i as usize)
    }
}

/// Parse column `dc` to a UTC `Datetime` column, move it into the index, and tag
/// the index with `tz` (PD-20 ingestion). A **naive** string column is
/// interpreted in `tz`; with `date_unit` the column is read as an epoch integer
/// (`"s"`/`"ms"`/`"us"`/`"ns"`, absolute) and `tz` only sets the display zone.
pub(crate) fn build_datetime_index(
    mut df: DataFrame,
    dc: &str,
    tz: Option<&str>,
    date_unit: Option<&str>,
) -> PyResult<DataFrame> {
    let tzv = match tz {
        Some(s) => Tz::parse(s).map_err(pyerr)?,
        None => Tz::Utc,
    };
    let parsed = match date_unit {
        Some(unit) => df.column(dc).map_err(pyerr)?.epoch_to_datetime(unit).map_err(pyerr)?,
        None => df.column(dc).map_err(pyerr)?.to_datetime_tz(tzv).map_err(pyerr)?,
    };
    df.set_column(dc, parsed).map_err(pyerr)?;
    let mut df = df.set_index(dc).map_err(pyerr)?;
    // The instants are already correct UTC; tag the display / matching tz.
    if !tzv.is_utc() {
        df = df.tz_convert(tzv).map_err(pyerr)?;
    }
    Ok(df)
}

/// Raise if the frame has stale computed columns after an `append`. The per-column
/// `df[directive]` access auto-refreshes; bulk / positional reads (`to_numpy`,
/// `.iloc` / `.loc` / `.at` / `.iat`) do not, so they must be fresh — call
/// `fulfill()` first. Keeps the read path O(1) and never returns silent NaN.
fn ensure_fresh(df: &DataFrame) -> PyResult<()> {
    if df.has_stale_computed() {
        Err(PyValueError::new_err(
            "frame has stale computed (directive) columns after append; \
             call fulfill() before a bulk or positional read",
        ))
    } else {
        Ok(())
    }
}

// --- Series ----------------------------------------------------------------

/// ``volas.Series`` — a single named, indexed column (usually obtained from
/// ``df['col']`` or a directive like ``df['ma:5']``).
///
/// Supports NaN-skipping reductions (``mean`` / ``sum`` / ``min`` / ``max`` /
/// ``std`` / ``var`` / ``median``), element-wise arithmetic / comparison /
/// boolean operators (``+ - * /``, ``< <= == != >= >``, ``& | ^ ~``) against a
/// scalar or another equal-length Series, the TA-Lib math transforms
/// (``sin`` / ``sqrt`` / ``ln`` / …), and ``shift`` / ``diff`` / ``fillna`` /
/// ``isna`` / ``notna`` / ``dropna``. Index by position via ``s.iloc[...]`` or
/// label via ``s.loc[...]``; export with ``to_numpy`` / ``to_list``.
///
/// Usage::
///
///     close = df['close']
///     close.mean()            # NaN-skipping mean
///     (close - df['open'])    # element-wise difference
///     close.shift(1)          # lag by one bar
///     close.iloc[-1]          # last value
#[pyclass(name = "Series")]
pub struct PySeries {
    pub(crate) inner: Series,
}

#[pymethods]
impl PySeries {
    /// The series name — the column it was drawn from, or ``None``.
    ///
    /// Returns:
    ///     str | None
    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name.clone()
    }

    /// The dtype name (``'float64'``, ``'bool'``, ``'int64'``, ``'object'`` or
    /// ``'datetime64[ns]'``).
    ///
    /// Returns:
    ///     str
    #[getter]
    fn dtype(&self) -> String {
        self.inner.dtype().to_string()
    }

    /// The row index shared with the parent frame, as a NumPy array (a
    /// ``datetime64[ns]`` array for a DatetimeIndex, an object array for a string
    /// index).
    ///
    /// Returns:
    ///     numpy.ndarray
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        index_to_numpy(py, &self.inner.index)
    }

    /// The DatetimeIndex timezone name, or `None` for a tz-naive / non-datetime
    /// index (mirrors `df.tz`).
    #[getter]
    fn tz(&self) -> Option<String> {
        match self.inner.index.tz() {
            Tz::Utc => None,
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
    fn iloc(&self) -> SeriesILoc {
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
    fn loc(&self) -> SeriesLoc {
        SeriesLoc {
            inner: self.inner.clone(),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// The values as a typed NumPy array; `dtype` casts (e.g. `'float32'`).
    #[pyo3(signature = (dtype = None))]
    fn to_numpy<'py>(&self, py: Python<'py>, dtype: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        let arr = column_to_numpy(py, &self.inner.data);
        match dtype {
            Some(dt) => Ok(arr.call_method1("astype", (dt,))?),
            None => Ok(arr),
        }
    }

    /// NumPy array protocol, so `np.isnan(series)` etc. work directly. Honors a
    /// requested `dtype` (casts).
    #[pyo3(signature = (dtype = None, copy = None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<PyObject>,
        copy: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = copy;
        let arr = column_to_numpy(py, &self.inner.data);
        match dtype {
            Some(dt) => Ok(arr.call_method1("astype", (dt,))?),
            None => Ok(arr),
        }
    }

    /// NaN-skipping mean of the values.
    fn mean(&self) -> f64 {
        let v = non_nan(&self.inner.data);
        if v.is_empty() {
            f64::NAN
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    }

    /// NaN-skipping sum (0.0 when empty / all-NaN, matching pandas).
    fn sum(&self) -> f64 {
        non_nan(&self.inner.data).iter().sum()
    }

    /// NaN-skipping minimum (NaN when empty / all-NaN).
    fn min(&self) -> f64 {
        non_nan(&self.inner.data)
            .into_iter()
            .fold(f64::NAN, |m, x| if m.is_nan() { x } else { m.min(x) })
    }

    /// NaN-skipping maximum (NaN when empty / all-NaN).
    fn max(&self) -> f64 {
        non_nan(&self.inner.data)
            .into_iter()
            .fold(f64::NAN, |m, x| if m.is_nan() { x } else { m.max(x) })
    }

    /// NaN-skipping sample variance (`ddof=1`; NaN with fewer than 2 values).
    fn var(&self) -> f64 {
        let v = non_nan(&self.inner.data);
        let n = v.len();
        if n < 2 {
            return f64::NAN;
        }
        let mean = v.iter().sum::<f64>() / n as f64;
        v.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64
    }

    /// NaN-skipping sample standard deviation (`ddof=1`).
    fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// NaN-skipping median.
    fn median(&self) -> f64 {
        let mut v = non_nan(&self.inner.data);
        let n = v.len();
        if n == 0 {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        }
    }

    /// The values as a Python list of typed scalars (pandas `to_list`).
    fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Py<PyAny>> = (0..self.inner.len())
            .map(|i| scalar_to_py(py, &self.inner.data, i))
            .collect();
        PyList::new(py, items)
    }

    /// Alias of [`to_list`](Self::to_list) (the numpy / pandas spelling).
    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.to_list(py)
    }

    /// Shift the values by ``n`` rows, padding vacated cells with NaN.
    ///
    /// Args:
    ///     n (int): rows to shift; positive shifts down (default 1), negative up.
    ///
    /// Usage::
    ///
    ///     s.shift(1)    # lag by one bar
    ///     s.shift(-1)   # lead by one bar
    ///
    /// Returns:
    ///     Series: a new series of the same length.
    #[pyo3(signature = (n = 1))]
    fn shift(&self, n: isize) -> PySeries {
        let a = self.inner.data.to_f64_vec();
        let len = a.len();
        let mut out = vec![f64::NAN; len];
        if n >= 0 {
            let n = (n as usize).min(len);
            out[n..].copy_from_slice(&a[..len - n]);
        } else {
            let n = ((-n) as usize).min(len);
            out[..len - n].copy_from_slice(&a[n..]);
        }
        f64_series(&self.inner, out)
    }

    /// Discrete difference ``x[i] - x[i-n]`` (equivalent to ``s - s.shift(n)``).
    ///
    /// Args:
    ///     n (int): periods to difference; the first ``n`` rows are NaN
    ///         (default 1). Negative ``n`` differences against later rows.
    ///
    /// Returns:
    ///     Series: a new series of the same length.
    #[pyo3(signature = (n = 1))]
    fn diff(&self, n: isize) -> PySeries {
        let a = self.inner.data.to_f64_vec();
        let len = a.len();
        let mut out = vec![f64::NAN; len];
        if n >= 0 {
            let n = n as usize;
            for i in n..len {
                out[i] = a[i] - a[i - n];
            }
        } else {
            let n = (-n) as usize;
            for i in 0..len.saturating_sub(n) {
                out[i] = a[i] - a[i + n];
            }
        }
        f64_series(&self.inner, out)
    }

    /// Replace missing (NaN) values with a constant.
    ///
    /// Args:
    ///     value (float): the value written into every NaN cell.
    ///
    /// Returns:
    ///     Series: a new series (non-float columns are returned unchanged).
    fn fillna(&self, value: f64) -> PySeries {
        let col = match &self.inner.data {
            Column::F64(v) => {
                Column::f64(v.iter().map(|&x| if x.is_nan() { value } else { x }).collect())
            }
            other => other.clone(),
        };
        PySeries {
            inner: Series::new(self.inner.name.clone(), col, Arc::clone(&self.inner.index)),
        }
    }

    /// Boolean mask of missing (NaN) values (non-F64 columns -> all False).
    fn isna(&self) -> PySeries {
        let out = match &self.inner.data {
            Column::F64(v) => v.iter().map(|x| x.is_nan()).collect(),
            other => vec![false; other.len()],
        };
        bool_series(&self.inner, out)
    }

    /// Boolean mask of present (non-NaN) values.
    fn notna(&self) -> PySeries {
        let out = match &self.inner.data {
            Column::F64(v) => v.iter().map(|x| !x.is_nan()).collect(),
            other => vec![true; other.len()],
        };
        bool_series(&self.inner, out)
    }

    /// Drop missing (NaN) elements (carries their index labels with them).
    fn dropna(&self) -> PySeries {
        let keep: Vec<usize> = match &self.inner.data {
            Column::F64(v) => v
                .iter()
                .enumerate()
                .filter_map(|(i, x)| (!x.is_nan()).then_some(i))
                .collect(),
            other => (0..other.len()).collect(),
        };
        let data = self.inner.data.take(&keep);
        let index = Arc::new(self.inner.index.take(&keep));
        PySeries {
            inner: Series::new(self.inner.name.clone(), data, index),
        }
    }

    /// pandas-style equality: **same dtype** and value-equal (NaN equals NaN).
    fn equals(&self, other: &PySeries) -> bool {
        self.inner.data.dtype() == other.inner.data.dtype()
            && self.inner.data.equals(&other.inner.data)
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

    // Element-wise comparisons -> bool Series (pandas-style).
    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, |a, b| a < b)
    }
    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, |a, b| a <= b)
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, |a, b| a == b)
    }
    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, |a, b| a != b)
    }
    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, |a, b| a >= b)
    }
    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_cmp(&self.inner, other, |a, b| a > b)
    }

    // Element-wise boolean logic -> bool Series (operands coerced to bool).
    fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, |a, b| a && b)
    }
    fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, |a, b| a || b)
    }
    fn __xor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, |a, b| a ^ b)
    }
    fn __rand__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, |a, b| a && b)
    }
    fn __ror__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, |a, b| a || b)
    }
    fn __rxor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PySeries> {
        series_logical(&self.inner, other, |a, b| a ^ b)
    }
    fn __invert__(&self) -> PySeries {
        let out: Vec<bool> = to_bool_vec(&self.inner.data).iter().map(|&b| !b).collect();
        PySeries {
            inner: Series::new(
                self.inner.name.clone(),
                Column::bool(out),
                Arc::clone(&self.inner.index),
            ),
        }
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
        let label = parse_label(key, &self.inner.index)?;
        let pos = self
            .inner
            .index
            .position_of(&label)
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

    // --- TA-Lib "Math Transform" group: element-wise, NaN-preserving (a NaN or an
    // out-of-domain input — e.g. sqrt of a negative, asin outside [-1, 1] — yields
    // NaN, matching TA-Lib). Implemented as Series methods, not directives.
    /// Element-wise arc cosine (TA-Lib ACOS).
    fn acos(&self) -> PySeries {
        self.map_f64(f64::acos)
    }
    /// Element-wise arc sine (TA-Lib ASIN).
    fn asin(&self) -> PySeries {
        self.map_f64(f64::asin)
    }
    /// Element-wise arc tangent (TA-Lib ATAN).
    fn atan(&self) -> PySeries {
        self.map_f64(f64::atan)
    }
    /// Element-wise ceiling (TA-Lib CEIL).
    fn ceil(&self) -> PySeries {
        self.map_f64(f64::ceil)
    }
    /// Element-wise cosine (TA-Lib COS).
    fn cos(&self) -> PySeries {
        self.map_f64(f64::cos)
    }
    /// Element-wise hyperbolic cosine (TA-Lib COSH).
    fn cosh(&self) -> PySeries {
        self.map_f64(f64::cosh)
    }
    /// Element-wise base-e exponential (TA-Lib EXP).
    fn exp(&self) -> PySeries {
        self.map_f64(f64::exp)
    }
    /// Element-wise floor (TA-Lib FLOOR).
    fn floor(&self) -> PySeries {
        self.map_f64(f64::floor)
    }
    /// Element-wise natural logarithm (TA-Lib LN).
    fn ln(&self) -> PySeries {
        self.map_f64(f64::ln)
    }
    /// Element-wise base-10 logarithm (TA-Lib LOG10).
    fn log10(&self) -> PySeries {
        self.map_f64(f64::log10)
    }
    /// Element-wise sine (TA-Lib SIN).
    fn sin(&self) -> PySeries {
        self.map_f64(f64::sin)
    }
    /// Element-wise hyperbolic sine (TA-Lib SINH).
    fn sinh(&self) -> PySeries {
        self.map_f64(f64::sinh)
    }
    /// Element-wise square root (TA-Lib SQRT).
    fn sqrt(&self) -> PySeries {
        self.map_f64(f64::sqrt)
    }
    /// Element-wise tangent (TA-Lib TAN).
    fn tan(&self) -> PySeries {
        self.map_f64(f64::tan)
    }
    /// Element-wise hyperbolic tangent (TA-Lib TANH).
    fn tanh(&self) -> PySeries {
        self.map_f64(f64::tanh)
    }
}

impl PySeries {
    /// Apply an element-wise `f64 -> f64` map, preserving name and index. Non-F64
    /// columns are coerced to f64 first. Shared by the Math Transform methods.
    fn map_f64(&self, f: impl Fn(f64) -> f64) -> PySeries {
        let data = Column::f64(self.inner.data.to_f64_vec().iter().map(|&x| f(x)).collect());
        PySeries {
            inner: Series::new(self.inner.name.clone(), data, Arc::clone(&self.inner.index)),
        }
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

/// The RHS of a Series binary op as an `f64` vector — another Series (positional,
/// no index alignment) or a broadcast scalar.
fn series_rhs_f64(other: &Bound<'_, PyAny>, len: usize) -> PyResult<Vec<f64>> {
    if let Ok(o) = other.extract::<PyRef<PySeries>>() {
        Ok(o.inner.data.to_f64_vec())
    } else if let Ok(scalar) = other.extract::<f64>() {
        Ok(vec![scalar; len])
    } else {
        Err(PyTypeError::new_err("unsupported operand for a Series operation"))
    }
}

/// A new F64 `Series` carrying `s`'s name and index.
fn f64_series(s: &Series, out: Vec<f64>) -> PySeries {
    PySeries {
        inner: Series::new(s.name.clone(), Column::f64(out), Arc::clone(&s.index)),
    }
}

/// A new Bool `Series` carrying `s`'s name and index.
fn bool_series(s: &Series, out: Vec<bool>) -> PySeries {
    PySeries {
        inner: Series::new(s.name.clone(), Column::bool(out), Arc::clone(&s.index)),
    }
}

fn series_binop(
    s: &Series,
    other: &Bound<'_, PyAny>,
    f: impl Fn(f64, f64) -> f64,
) -> PyResult<PySeries> {
    let a = s.data.to_f64_vec();
    let rhs = series_rhs_f64(other, a.len())?;
    let n = a.len().min(rhs.len());
    let mut out = vec![f64::NAN; a.len()];
    for i in 0..n {
        out[i] = f(a[i], rhs[i]);
    }
    Ok(PySeries {
        inner: Series::new(s.name.clone(), Column::f64(out), Arc::clone(&s.index)),
    })
}

/// Element-wise comparison -> bool Series (positional; NaN compares `false` for
/// ordering/equality and `true` for `!=`, matching IEEE / pandas element-wise).
fn series_cmp(
    s: &Series,
    other: &Bound<'_, PyAny>,
    f: impl Fn(f64, f64) -> bool,
) -> PyResult<PySeries> {
    let a = s.data.to_f64_vec();
    let rhs = series_rhs_f64(other, a.len())?;
    let n = a.len().min(rhs.len());
    let mut out = vec![false; a.len()];
    for i in 0..n {
        out[i] = f(a[i], rhs[i]);
    }
    Ok(PySeries {
        inner: Series::new(s.name.clone(), Column::bool(out), Arc::clone(&s.index)),
    })
}

/// The non-NaN `f64` values of a column (for NaN-skipping reductions).
fn non_nan(col: &Column) -> Vec<f64> {
    col.to_f64_vec().into_iter().filter(|x| !x.is_nan()).collect()
}

/// A column coerced to bool (a `Bool` column as-is, else `x != 0.0`).
fn to_bool_vec(col: &Column) -> Vec<bool> {
    match col {
        Column::Bool(v) => v.to_vec(),
        other => other.to_f64_vec().iter().map(|&x| x != 0.0).collect(),
    }
}

/// Element-wise boolean logic -> bool Series (both operands coerced to bool).
fn series_logical(
    s: &Series,
    other: &Bound<'_, PyAny>,
    f: impl Fn(bool, bool) -> bool,
) -> PyResult<PySeries> {
    let a = to_bool_vec(&s.data);
    let rhs = if let Ok(o) = other.extract::<PyRef<PySeries>>() {
        to_bool_vec(&o.inner.data)
    } else if let Ok(b) = other.extract::<bool>() {
        vec![b; a.len()]
    } else if let Ok(x) = other.extract::<f64>() {
        vec![x != 0.0; a.len()]
    } else {
        return Err(PyTypeError::new_err("unsupported operand for a Series logical op"));
    };
    let n = a.len().min(rhs.len());
    let mut out = vec![false; a.len()];
    for i in 0..n {
        out[i] = f(a[i], rhs[i]);
    }
    Ok(PySeries {
        inner: Series::new(s.name.clone(), Column::bool(out), Arc::clone(&s.index)),
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

/// A single DataFrame row (the result of `df.iloc[i]` / `df.loc[label]`): a
/// faithful 1-row frame carrying its index label and every column's *typed*
/// value (no lossy f64 coercion, no flag pair to remember the index kind).
#[pyclass(name = "Row")]
pub struct PyRow {
    inner: DataFrame,
}

#[pymethods]
impl PyRow {
    /// The row's index label.
    #[getter]
    fn name(&self, py: Python<'_>) -> Py<PyAny> {
        label_to_py(py, self.inner.index(), 0)
    }

    /// A single value by column name (``row[col]``).
    ///
    /// Returns:
    ///     the typed scalar at that column.
    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        let col = self.inner.column(key).map_err(pyerr)?;
        Ok(scalar_to_py(py, col, 0))
    }

    /// The row's values as a ``(1, n_columns)`` float64 NumPy array.
    ///
    /// Returns:
    ///     numpy.ndarray
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (data, h, w) = self.inner.to_row_major_f64();
        Ok(ndarray::Array2::from_shape_vec((h, w), data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py))
    }

    /// The row as a typed `{column: value}` dict (pandas `Series.to_dict`).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            d.set_item(name, scalar_to_py(py, col, 0))?;
        }
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!("Row(columns={:?})", self.inner.names())
    }
}

// --- DataFrame -------------------------------------------------------------

/// ``volas.DataFrame`` — an ordered, named, time-indexed OHLCV table with
/// indicator-directive indexing and pandas-compatible positional / label access.
///
/// Construct from a dict of columns, or read a CSV::
///
///     df = volas.DataFrame({'close': [10.0, 11.0], 'volume': [100, 120]})
///     df = volas.read_csv('ohlcv.csv', index_col='time')
///
/// The headline feature is string indexing: a plain column name returns that
/// column, and an *indicator directive* is computed on demand, cached, and
/// incrementally refreshed thereafter::
///
///     df['close']            # a column, as a Series
///     df['ma:5']             # SMA(5) of close (directive) — computed & cached
///     df['macd.signal']      # MACD signal line
///     df['close > open']     # a boolean directive -> bool Series
///     df[['open', 'close']]  # a sub-frame
///     df[df['close'] > 100]  # boolean-mask row filter
///
/// Positional / label access mirrors pandas via ``.iloc`` / ``.loc`` (2-D get +
/// set) and the scalar ``.iat`` / ``.at``; common transforms (``head``,
/// ``tail``, ``dropna``, ``sort_index``, ``reset_index``, ``set_index``,
/// ``rename``, ``astype``, ``to_numpy``, ``to_pandas``, ``to_csv``) follow the
/// pandas spelling. ``cumulate`` resamples to a coarser timeframe; ``append``
/// grows the frame in place for live streaming.
///
/// Args:
///     data (dict[str, Sequence]): column name -> equal-length values.
///     date_col (str, optional): a column to parse into a DatetimeIndex.
///     tz (str, optional): timezone for that index (requires ``date_col``).
///     date_unit (str, optional): read ``date_col`` as an epoch integer in this
///         unit (``'s'`` / ``'ms'`` / ``'us'`` / ``'ns'``); requires ``date_col``.
#[pyclass(name = "DataFrame")]
pub struct PyDataFrame {
    pub(crate) inner: DataFrame,
}

impl PyDataFrame {
    fn wrap_series(&self, name: String, col: Column) -> PySeries {
        PySeries {
            inner: Series::new(Some(name), col, Arc::clone(self.inner.index())),
        }
    }

    /// Recompute the stale tail of cached directive columns in place — all of
    /// them if `only` is `None`, else just the named one. O(lookback + new rows)
    /// per column. Done against the real (non-computed) columns so a bare-name
    /// directive recomputes and no cached buffer is pinned.
    fn refresh_computed(&mut self, only: Option<&str>) -> PyResult<()> {
        let height = self.inner.height();
        // Materialize the computed-column set once (per tick on the live append path)
        // and derive both the stale list and the name set from it, instead of cloning
        // the computed map twice.
        let computed = self.inner.computed_columns();
        let stale: Vec<_> = computed
            .iter()
            .filter(|(n, m)| m.valid_rows < height && only.is_none_or(|o| o == n))
            .cloned()
            .collect();
        if stale.is_empty() {
            return Ok(());
        }
        let computed_names: HashSet<String> = computed.iter().map(|(n, _)| n.clone()).collect();
        let real_names: Vec<String> = self
            .inner
            .names()
            .iter()
            .filter(|n| !computed_names.contains(*n))
            .cloned()
            .collect();
        let base = self.inner.select(&real_names).map_err(pyerr)?;
        for (name, meta) in stale {
            let start = meta.valid_rows.saturating_sub(meta.lookback);
            let slice = base.slice(start, height);
            let node = parse(&meta.directive).map_err(value_err)?;
            let recomputed = execute(&slice, &node).map_err(value_err)?;
            // The recomputed slice is F64 or Bool (directive result); write its
            // stale tail back into the column at its original dtype.
            let tail = recomputed.slice(meta.valid_rows - start, recomputed.len());
            self.inner
                .update_computed_tail(&name, meta.valid_rows, &tail)
                .map_err(pyerr)?;
        }
        Ok(())
    }
}

#[pymethods]
impl PyDataFrame {
    // Constructor — the user-facing argument list & usage live in the class
    // docstring (pyo3 does not surface a `#[new]` doc comment to Python).
    #[new]
    #[pyo3(signature = (data, date_col = None, tz = None, date_unit = None))]
    fn new(
        data: &Bound<'_, PyDict>,
        date_col: Option<String>,
        tz: Option<String>,
        date_unit: Option<String>,
    ) -> PyResult<Self> {
        let mut names = Vec::new();
        let mut columns = Vec::new();
        for (k, v) in data.iter() {
            names.push(k.extract::<String>()?);
            columns.push(pyany_to_column(&v)?);
        }
        let mut df = DataFrame::new(names, columns, None).map_err(pyerr)?;
        // Same string -> DatetimeIndex path as read_csv, with timezone ingestion:
        // parse the named column to a UTC datetime, move it into the index, then
        // tag the index tz (see `build_datetime_index`).
        if let Some(dc) = date_col {
            df = build_datetime_index(df, &dc, tz.as_deref(), date_unit.as_deref())?;
        } else if tz.is_some() || date_unit.is_some() {
            return Err(PyValueError::new_err(
                "tz / date_unit require date_col (the column to use as the datetime index)",
            ));
        }
        Ok(PyDataFrame { inner: df })
    }

    /// The DatetimeIndex timezone name (`"+08:00"` / `"America/New_York"`), or
    /// `None` for a tz-naive (UTC-default) or non-datetime index — mirroring
    /// pandas `df.index.tz`.
    #[getter]
    fn tz(&self) -> Option<String> {
        match self.inner.index().tz() {
            Tz::Utc => None,
            other => Some(other.name()),
        }
    }

    /// Reinterpret the index wall-clock as `tz` (pandas `tz_localize`): the
    /// displayed wall-clock is unchanged, each instant is recomputed. Use when
    /// data was ingested without a tz. Returns a new frame.
    fn tz_localize(&self, tz: &str) -> PyResult<PyDataFrame> {
        let tzv = Tz::parse(tz).map_err(pyerr)?;
        Ok(PyDataFrame {
            inner: self.inner.tz_localize(tzv).map_err(pyerr)?,
        })
    }

    /// Change the index display / matching tz without moving any instant (pandas
    /// `tz_convert`). Returns a new frame.
    fn tz_convert(&self, tz: &str) -> PyResult<PyDataFrame> {
        let tzv = Tz::parse(tz).map_err(pyerr)?;
        Ok(PyDataFrame {
            inner: self.inner.tz_convert(tzv).map_err(pyerr)?,
        })
    }

    /// The column names, in order.
    ///
    /// Returns:
    ///     list[str]
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner.names().to_vec()
    }

    /// The frame dimensions as ``(n_rows, n_columns)`` (pandas ``shape``).
    ///
    /// Returns:
    ///     tuple[int, int]
    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.height(), self.inner.width())
    }

    /// The row index as a NumPy array (``datetime64[ns]`` for a DatetimeIndex,
    /// an object array for a string index, else an integer array).
    ///
    /// Returns:
    ///     numpy.ndarray
    #[getter]
    fn index<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        index_to_numpy(py, self.inner.index())
    }

    // The indexers hold a live reference to this frame (`Py<PyDataFrame>`), not a
    // snapshot, so `df.iloc[...] = ` / `df.loc[...] = ` mutate the frame in place
    // (copy-on-write under the hood) and reads always see the current rows.

    /// Purely integer-location indexing for selection and assignment.
    ///
    /// Get ``df.iloc[i]`` (a row), ``df.iloc[a:b]`` (a sub-frame),
    /// ``df.iloc[i, j]`` (a cell), ``df.iloc[:, j]`` (a column as a Series), or
    /// ``df.iloc[rows, cols]`` (a sub-frame). Assign ``df.iloc[rows, j] = value``
    /// (copy-on-write; a prior ``copy()`` is unaffected).
    ///
    /// Usage::
    ///
    ///     df.iloc[0]            # first row
    ///     df.iloc[-1, 3]        # last row, 4th column -> scalar
    ///     df.iloc[:, 0]         # first column as a Series
    ///     df.iloc[10:20, 0:2]   # a block
    ///     df.iloc[mask, 1] = 0  # assign a column where a boolean mask is True
    #[getter]
    fn iloc(slf: Bound<'_, Self>) -> DataFrameILoc {
        DataFrameILoc { parent: slf.unbind() }
    }

    /// Label-based indexing for selection and assignment.
    ///
    /// Get ``df.loc[label]`` (a row), ``df.loc[a:b]`` (a stop-inclusive label
    /// slice), ``df.loc[label, col]`` (a cell), ``df.loc[:, col]`` (a column),
    /// or ``df.loc[mask, col]``. Assign ``df.loc[mask, 'signal'] = 1``
    /// (copy-on-write).
    ///
    /// Usage::
    ///
    ///     df.loc['2021-01-04']               # row by datetime label
    ///     df.loc['2021-01':'2021-03']        # inclusive label slice
    ///     df.loc[df['close'] > df['open'], 'signal'] = 1
    #[getter]
    fn loc(slf: Bound<'_, Self>) -> DataFrameLoc {
        DataFrameLoc { parent: slf.unbind() }
    }

    /// Fast scalar access by integer position: ``df.iat[i, j]`` to get or set a
    /// single cell (copy-on-write).
    ///
    /// Usage::
    ///
    ///     df.iat[0, 3]        # the cell at row 0, column 3
    ///     df.iat[0, 3] = 1.5  # set it
    #[getter]
    fn iat(slf: Bound<'_, Self>) -> DataFrameIat {
        DataFrameIat { parent: slf.unbind() }
    }

    /// Fast scalar access by label + column name: ``df.at[label, col]`` to get
    /// or set a single cell (copy-on-write).
    ///
    /// Usage::
    ///
    ///     df.at['2021-01-04', 'close']         # one cell
    ///     df.at['2021-01-04', 'close'] = 100.0 # set it
    #[getter]
    fn at(slf: Bound<'_, Self>) -> DataFrameAt {
        DataFrameAt { parent: slf.unbind() }
    }

    fn __len__(&self) -> usize {
        self.inner.height()
    }

    /// `name in df` — whether a column exists (alias-aware).
    fn __contains__(&self, key: &str) -> bool {
        self.inner.has_column(key)
    }

    /// `for x in df` — iterate the column names (pandas semantics).
    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let names = PyList::new(py, self.inner.names())?;
        Ok(names.try_iter()?.into_any().unbind())
    }

    /// First `n` rows (pandas `head`).
    #[pyo3(signature = (n = 5))]
    fn head(&self, n: usize) -> PyDataFrame {
        PyDataFrame {
            inner: self.inner.slice(0, n.min(self.inner.height())),
        }
    }

    /// Last `n` rows (pandas `tail`).
    #[pyo3(signature = (n = 5))]
    fn tail(&self, n: usize) -> PyDataFrame {
        let h = self.inner.height();
        PyDataFrame {
            inner: self.inner.slice(h.saturating_sub(n), h),
        }
    }

    /// Per-column dtypes as `{name: dtype_str}` (pandas `dtypes`).
    #[getter]
    fn dtypes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            d.set_item(name, col.dtype().to_string())?;
        }
        Ok(d)
    }

    /// Drop rows containing missing values. `how='any'` (default) drops a row if
    /// any F64 column is NaN there; `how='all'` only if every column is NaN.
    #[pyo3(signature = (how = "any"))]
    fn dropna(&self, how: &str) -> PyDataFrame {
        let cols = self.inner.columns();
        let total = cols.len();
        let keep: Vec<usize> = (0..self.inner.height())
            .filter(|&i| {
                let nan = cols
                    .iter()
                    .filter(|c| matches!(c, Column::F64(v) if v[i].is_nan()))
                    .count();
                match how {
                    "all" => nan < total.max(1),
                    _ => nan == 0,
                }
            })
            .collect();
        PyDataFrame {
            inner: take_frame(&self.inner, &keep),
        }
    }

    /// Sort rows by index label (pandas `sort_index`).
    #[pyo3(signature = (ascending = true))]
    fn sort_index(&self, ascending: bool) -> PyDataFrame {
        let perm = self.inner.index().argsort(ascending);
        PyDataFrame {
            inner: take_frame(&self.inner, &perm),
        }
    }

    /// Move the row index into an `'index'` column and restore a RangeIndex
    /// (pandas `reset_index`); `drop=True` discards the old index.
    #[pyo3(signature = (drop = false))]
    fn reset_index(&self, drop: bool) -> PyResult<PyDataFrame> {
        let h = self.inner.height();
        let (names, columns): (Vec<String>, Vec<Column>) = if drop {
            (self.inner.names().to_vec(), self.inner.columns().to_vec())
        } else {
            let mut names = vec!["index".to_string()];
            names.extend(self.inner.names().iter().cloned());
            let mut cols = vec![self.inner.index().to_column()];
            cols.extend(self.inner.columns().iter().cloned());
            (names, cols)
        };
        Ok(PyDataFrame {
            inner: DataFrame::new(names, columns, Some(Index::Range(h))).map_err(pyerr)?,
        })
    }

    /// `df[name] = value` — add or replace a column. `value` may be a scalar
    /// (broadcast), a 1-D array / list, or a Series (positional, length must
    /// equal the frame height). Copy-on-write: a prior `copy()` is unaffected.
    fn __setitem__(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let h = self.inner.height();
        let col = if let Ok(s) = value.extract::<PyRef<PySeries>>() {
            s.inner.data.clone()
        } else if let Ok(b) = value.extract::<bool>() {
            Column::bool(vec![b; h])
        } else if let Ok(scalar) = value.extract::<f64>() {
            Column::f64(vec![scalar; h])
        } else {
            pyany_to_column(value)?
        };
        self.inner.set_column(name, col).map_err(pyerr)
    }

    // `df[key]` — column name / indicator directive / list / boolean mask /
    // slice. The user-facing usage lives in the class docstring (pyo3 implements
    // `__getitem__` as a type slot and does not surface its doc comment).
    fn __getitem__(&mut self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
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
        // boolean mask as a plain Python list (df[[True, False, ...]]). An empty
        // list is an empty column projection, not a mask, so it falls through.
        if let Ok(mask) = key.extract::<Vec<bool>>() {
            if !mask.is_empty() {
                if mask.len() != self.inner.height() {
                    return Err(PyIndexError::new_err(format!(
                        "boolean index has wrong length: {} instead of {}",
                        mask.len(),
                        self.inner.height()
                    )));
                }
                let sub = self.inner.filter_mask(&mask).map_err(pyerr)?;
                return Ok(Py::new(py, PyDataFrame { inner: sub })?.into_any());
            }
        }
        // label / positional slice: df[:'date'], df[1:5]
        if let Ok(slice) = key.downcast::<PySlice>() {
            let sub = slice_frame(&self.inner, slice)?;
            return Ok(Py::new(py, PyDataFrame { inner: sub })?.into_any());
        }
        // column name or directive
        if let Ok(name) = key.extract::<String>() {
            // Existing column — a real column, or a cached directive: refresh its
            // stale tail if cached (no-op for a plain data column), then return.
            if self.inner.has_column(&name) {
                self.refresh_computed(Some(&name))?;
                let col = self.inner.column(&name).map_err(pyerr)?.clone();
                return Ok(Py::new(py, self.wrap_series(name, col))?.into_any());
            }
            // A directive: materialize (auto-cache) under its canonical name; on
            // later access its stale tail is refreshed incrementally, so the
            // result is always fresh AND cheap (O(lookback), not O(n)).
            let node = parse(&name).map_err(syntax_err)?;
            let canonical = volas_directive::stringify(&node);
            if self.inner.has_column(&canonical) {
                self.refresh_computed(Some(&canonical))?;
            } else {
                let col = execute(&self.inner, &node).map_err(value_err)?;
                let lookback = volas_directive::lookback::lookback(&node);
                self.inner.set_column(&canonical, col).map_err(pyerr)?;
                self.inner.set_computed(&canonical, canonical.clone(), lookback);
            }
            let col = self.inner.column(&canonical).map_err(pyerr)?.clone();
            return Ok(Py::new(py, self.wrap_series(canonical, col))?.into_any());
        }
        // list of names / directives
        if let Ok(list) = key.extract::<Vec<String>>() {
            let mut cols = Vec::with_capacity(list.len());
            for n in &list {
                let col = if self.inner.has_column(n) {
                    self.inner.column(n).map_err(pyerr)?.clone()
                } else {
                    let node = parse(n).map_err(syntax_err)?;
                    execute(&self.inner, &node).map_err(value_err)?
                };
                cols.push(col);
            }
            let idx = (*self.inner.index().as_ref()).clone();
            let df = DataFrame::new(list, cols, Some(idx)).map_err(pyerr)?;
            return Ok(Py::new(py, PyDataFrame { inner: df })?.into_any());
        }
        Err(PyKeyError::new_err(
            "key must be a column name, directive, list, boolean mask, or slice",
        ))
    }

    /// Evaluate an indicator directive and return its values as a NumPy array.
    ///
    /// Unlike ``df['ma:5']`` (which returns a Series and caches the column),
    /// ``exec`` returns the raw array; pass ``create_column=True`` to also cache
    /// it on the frame under its canonical name.
    ///
    /// Args:
    ///     directive (str): the directive, e.g. ``'macd'``, ``'boll.upper:20'``,
    ///         ``'close > open'``.
    ///     create_column (bool): if True, materialize and cache the result as a
    ///         column (default False).
    ///
    /// Usage::
    ///
    ///     df.exec('ma:5')               # ndarray of SMA(5)
    ///     df.exec('kdj.j', create_column=True)  # also caches the column
    ///
    /// Returns:
    ///     numpy.ndarray
    #[pyo3(signature = (directive, create_column = false))]
    fn exec<'py>(
        &mut self,
        py: Python<'py>,
        directive: &str,
        create_column: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.inner.has_column(directive) {
            let col = self.inner.column(directive).map_err(pyerr)?.clone();
            return Ok(column_to_numpy(py, &col));
        }
        let node = parse(directive).map_err(syntax_err)?;
        if create_column {
            // Materialize + cache under the canonical name, exactly like `df[directive]`.
            let canonical = volas_directive::stringify(&node);
            if self.inner.has_column(&canonical) {
                self.refresh_computed(Some(&canonical))?;
            } else {
                let col = execute(&self.inner, &node).map_err(value_err)?;
                let lookback = volas_directive::lookback::lookback(&node);
                self.inner.set_column(&canonical, col).map_err(pyerr)?;
                self.inner.set_computed(&canonical, canonical.clone(), lookback);
            }
            let col = self.inner.column(&canonical).map_err(pyerr)?.clone();
            Ok(column_to_numpy(py, &col))
        } else {
            let col = execute(&self.inner, &node).map_err(value_err)?;
            Ok(column_into_numpy(py, col))
        }
    }

    /// Gets the lookback period of the given directive — the minimum number of
    /// prior rows it needs before it can emit a (non-NaN) value.
    ///
    /// Args:
    ///     directive (str): directive
    ///
    /// Usage::
    ///
    ///     volas.DataFrame.directive_lookback('boll:20')
    ///     # It gets 19
    ///
    /// Returns:
    ///     int
    #[staticmethod]
    fn directive_lookback(directive: &str) -> PyResult<usize> {
        let node = parse(directive).map_err(syntax_err)?;
        Ok(volas_directive::lookback::lookback(&node))
    }

    /// Gets the full (canonical) name of the ``directive``, which is also the
    /// actual column name cached on the frame — default args and series are
    /// filled in.
    ///
    /// Args:
    ///     directive (str): directive
    ///
    /// Usage::
    ///
    ///     volas.DataFrame.directive_stringify('boll')
    ///     # It gets "boll:20@close"
    ///
    /// Returns:
    ///     str
    #[staticmethod]
    fn directive_stringify(directive: &str) -> PyResult<String> {
        let node = parse(directive).map_err(syntax_err)?;
        Ok(volas_directive::stringify(&node))
    }

    /// Gets a column from the frame by name (alias-aware), as a Series.
    ///
    /// Args:
    ///     key (str): the column name.
    ///
    /// Returns:
    ///     Series
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

    /// Convert to a `pandas.DataFrame`. pandas is imported lazily (only here), so
    /// volas stays pandas-free at import.
    fn to_pandas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ensure_fresh(&self.inner)?;
        let pd = py.import("pandas")?;
        let data = PyDict::new(py);
        for (name, col) in self.inner.names().iter().zip(self.inner.columns()) {
            data.set_item(name, column_to_numpy(py, col))?;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("index", index_to_numpy(py, self.inner.index())?)?;
        pd.call_method("DataFrame", (data,), Some(&kwargs))
    }

    /// Write the frame as CSV (pandas-subset). With no `path`, returns the CSV
    /// string. Datetime columns are written as formatted strings (round-trips
    /// with `read_csv`).
    #[pyo3(signature = (path = None, sep = ",", index = true, header = true, na_rep = "", columns = None))]
    fn to_csv(
        &self,
        path: Option<String>,
        sep: &str,
        index: bool,
        header: bool,
        na_rep: &str,
        columns: Option<Vec<String>>,
    ) -> PyResult<Option<String>> {
        ensure_fresh(&self.inner)?;
        let names = self.inner.names();
        let positions: Vec<usize> = match &columns {
            Some(cols) => cols
                .iter()
                .map(|n| {
                    self.inner
                        .column_pos(n)
                        .ok_or_else(|| PyKeyError::new_err(format!("column \"{n}\" not found")))
                })
                .collect::<PyResult<_>>()?,
            None => (0..self.inner.width()).collect(),
        };
        let mut out = String::new();
        if header {
            if index {
                out.push_str("index");
                out.push_str(sep);
            }
            let hdr: Vec<&str> = positions.iter().map(|&j| names[j].as_str()).collect();
            out.push_str(&hdr.join(sep));
            out.push('\n');
        }
        for i in 0..self.inner.height() {
            if index {
                out.push_str(&index_label_csv(self.inner.index(), i));
                out.push_str(sep);
            }
            let cells: Vec<String> = positions
                .iter()
                .map(|&j| cell_to_csv(&self.inner.columns()[j], i, na_rep))
                .collect();
            out.push_str(&cells.join(sep));
            out.push('\n');
        }
        match path {
            Some(p) => {
                std::fs::write(&p, out).map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(None)
            }
            None => Ok(Some(out)),
        }
    }

    /// Drop rows by index label (`axis=0`) or columns by name (`axis=1`) —
    /// returns a new DataFrame. Row labels are parsed against the index kind.
    #[pyo3(signature = (labels, axis = 0))]
    fn drop(&self, py: Python<'_>, labels: Vec<Py<PyAny>>, axis: i64) -> PyResult<PyDataFrame> {
        if axis == 1 {
            let drop_names: Vec<String> = labels
                .iter()
                .map(|l| l.bind(py).extract::<String>())
                .collect::<PyResult<_>>()?;
            let keep: Vec<String> = self
                .inner
                .names()
                .iter()
                .filter(|n| !drop_names.contains(n))
                .cloned()
                .collect();
            return Ok(PyDataFrame {
                inner: self.inner.select(&keep).map_err(pyerr)?,
            });
        }
        let index = self.inner.index();
        let targets: Vec<Label> = labels
            .iter()
            .map(|l| parse_label(l.bind(py), index))
            .collect::<PyResult<_>>()?;
        let positions: Vec<usize> = (0..self.inner.height())
            .filter(|&i| !targets.contains(&index.label_at(i)))
            .collect();
        Ok(PyDataFrame {
            inner: take_frame(&self.inner, &positions),
        })
    }

    /// Append the rows of another DataFrame or a single Row **in place** and
    /// return the same frame (amortized O(1), like ``list.append`` — the live
    /// single-bar hot path, no full-column copy).
    ///
    /// Missing columns are NaN-padded; cached directive columns go stale until
    /// ``fulfill()``. A snapshot taken via ``copy()`` / ``iloc`` is unaffected
    /// (it pays one copy-on-write the next time *it* is appended to).
    ///
    /// Args:
    ///     other (DataFrame | Row): the rows to append.
    ///
    /// Usage::
    ///
    ///     df.append(bar)           # append one Row
    ///     df.append(other_frame)   # append many rows
    ///
    /// Returns:
    ///     DataFrame: ``self`` (enabling chaining).
    fn append<'py>(
        slf: Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        // Extract `other` into an owned frame first so its borrow is released
        // before we mutably borrow `slf` (so `df.append(df)` cannot deadlock).
        let other_inner = if let Ok(df) = other.extract::<PyRef<PyDataFrame>>() {
            df.inner.clone()
        } else if let Ok(row) = other.extract::<PyRef<PyRow>>() {
            row.inner.clone()
        } else {
            return Err(PyTypeError::new_err("append expects a DataFrame or Row"));
        };
        slf.borrow_mut().inner.append(&other_inner).map_err(pyerr)?;
        Ok(slf)
    }

    /// The frame as a 2-D NumPy array (pandas `to_numpy`). With no `dtype`: a fast
    /// `float64` matrix when every column is numeric, else a lossless `object`
    /// array (string columns kept, not NaN-poisoned). `dtype` casts (e.g.
    /// `'float32'`); requesting a float over a string column raises.
    #[pyo3(signature = (dtype = None))]
    fn to_numpy<'py>(&self, py: Python<'py>, dtype: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        ensure_fresh(&self.inner)?;
        let has_str = self.inner.columns().iter().any(|c| matches!(c, Column::Str(_)));
        let (h, w) = (self.inner.height(), self.inner.width());

        if let Some(dt) = dtype {
            let floaty = dt.contains("float") || dt == "f32" || dt == "f64" || dt == "double";
            if has_str && floaty {
                return Err(PyValueError::new_err(format!(
                    "cannot convert a string column to {dt}"
                )));
            }
            let (data, h, w) = self.inner.to_row_major_f64();
            let arr = ndarray::Array2::from_shape_vec((h, w), data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_pyarray(py);
            return Ok(arr.call_method1("astype", (dt,))?);
        }

        if !has_str {
            // fast all-numeric path: a float64 matrix
            let (data, h, w) = self.inner.to_row_major_f64();
            let arr = ndarray::Array2::from_shape_vec((h, w), data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(arr.into_pyarray(py).into_any());
        }

        // mixed/string frame: a lossless 2-D object array (each cell typed)
        let rows = PyList::empty(py);
        for i in 0..h {
            let row = PyList::empty(py);
            for col in self.inner.columns() {
                row.append(scalar_to_py(py, col, i))?;
            }
            rows.append(row)?;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("dtype", "object")?;
        let _ = w;
        Ok(py.import("numpy")?.call_method("array", (rows,), Some(&kwargs))?)
    }

    /// Value equality (same columns + index + values, `NaN == NaN`).
    fn equals(&self, other: &PyDataFrame) -> bool {
        self.inner.equals(&other.inner)
    }

    /// Resample to a coarser timeframe (OHLCV cumulation / down-sampling).
    ///
    /// Requires a DatetimeIndex. Each column is aggregated with a sensible
    /// default (open=first, high=max, low=min, close=last, volume=sum); override
    /// per column via ``cumulators``.
    ///
    /// Args:
    ///     time_frame (str | TimeFrame): the target bucket, e.g. ``'1d'``,
    ///         ``'15m'``, ``'1w'``.
    ///     cumulators (dict[str, str], optional): per-column aggregator
    ///         overrides, e.g. ``{'volume': 'sum', 'close': 'last'}``.
    ///
    /// Usage::
    ///
    ///     daily = df.cumulate('1d')
    ///     df.cumulate('1h', cumulators={'amount': 'sum'})
    ///
    /// Returns:
    ///     DataFrame: the resampled frame.
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

    /// Move a column into the row index (pandas `set_index(col)`), returning a
    /// new frame. A datetime / int / string column becomes the matching index.
    #[pyo3(signature = (keys))]
    fn set_index(&self, keys: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.set_index(keys).map_err(pyerr)?,
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

    /// Define a column / directive alias: `as_name` resolves to `src_name`
    /// everywhere a column is looked up (mutates in place, pandas-like).
    fn alias(&mut self, as_name: &str, src_name: &str) -> PyResult<()> {
        self.inner = self.inner.with_alias(as_name, src_name).map_err(pyerr)?;
        Ok(())
    }

    /// Refresh the stale tail of every materialized (auto-cached) directive
    /// column at once (e.g. before a bulk `to_numpy` / row read). Per-column
    /// access already auto-refreshes; this is the batch form. In place,
    /// incremental — O(lookback + new rows) per column, not O(n).
    fn fulfill(&mut self) -> PyResult<()> {
        self.refresh_computed(None)
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
// --- indexer assignment helpers (PD-12) ------------------------------------

/// Build a length-1 [`Column`] from a Python scalar, coerced toward the target
/// column's dtype (so a string can land in a datetime column, etc.). An `I64`
/// target given a float yields an `F64` value — core then widens the column.
fn scalar_to_column(v: &Bound<'_, PyAny>, target: DType) -> PyResult<Column> {
    match target {
        DType::F64 => {
            let x = v
                .extract::<f64>()
                .map_err(|_| PyTypeError::new_err("expected a number"))?;
            Ok(Column::f64(vec![x]))
        }
        DType::I64 => {
            if let Ok(i) = v.extract::<i64>() {
                Ok(Column::i64(vec![i]))
            } else {
                let x = v
                    .extract::<f64>()
                    .map_err(|_| PyTypeError::new_err("expected a number"))?;
                Ok(Column::f64(vec![x]))
            }
        }
        DType::Bool => {
            let b = v
                .extract::<bool>()
                .map_err(|_| PyTypeError::new_err("expected a bool"))?;
            Ok(Column::bool(vec![b]))
        }
        DType::Utf8 => {
            let s = v
                .extract::<String>()
                .map_err(|_| PyTypeError::new_err("expected a string"))?;
            Ok(Column::str(vec![s]))
        }
        DType::Datetime => Ok(Column::datetime(vec![parse_ts(v)?])),
    }
}

/// Convert an array-like assignment value (list / NumPy array / `Series`) to a
/// [`Column`]; core coerces it toward the target dtype.
fn value_to_column(v: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(s) = v.extract::<PyRef<PySeries>>() {
        return Ok(s.inner.data.clone());
    }
    pyany_to_column(v)
}

/// Resolve a `df[...] = value` right-hand side for `n` selected rows: a scalar is
/// broadcast (length-1 column), an array-like must match `n`.
fn resolve_assignment(v: &Bound<'_, PyAny>, target: DType, n: usize) -> PyResult<Column> {
    // A Python str has `__len__` but is a scalar here; everything else with
    // `__len__` (list / ndarray / Series) is array-like.
    let is_str = v.extract::<String>().is_ok();
    let arraylike = !is_str && v.hasattr("__len__").unwrap_or(false);
    if arraylike {
        let col = value_to_column(v)?;
        if col.len() != n {
            return Err(PyValueError::new_err(format!(
                "cannot assign {} values to {n} selected rows",
                col.len()
            )));
        }
        Ok(col)
    } else {
        scalar_to_column(v, target)
    }
}

/// If `sel` is a boolean mask of length `height` (NumPy bool array, bool `Series`,
/// or `list[bool]`), return the selected row positions; else `None`.
fn as_bool_mask(sel: &Bound<'_, PyAny>, height: usize) -> Option<Vec<usize>> {
    let collect = |bits: &[bool]| -> Option<Vec<usize>> {
        (bits.len() == height).then(|| {
            bits.iter()
                .enumerate()
                .filter_map(|(i, &b)| b.then_some(i))
                .collect()
        })
    };
    if let Ok(a) = sel.extract::<PyReadonlyArray1<bool>>() {
        return a.as_slice().ok().and_then(collect);
    }
    if let Ok(ser) = sel.extract::<PyRef<PySeries>>() {
        if let Column::Bool(v) = &ser.inner.data {
            return collect(v);
        }
        return None;
    }
    if let Ok(v) = sel.extract::<Vec<bool>>() {
        return collect(&v);
    }
    None
}

/// Resolve an `iloc` row selector (int / slice / int-list / bool-mask) to row
/// positions.
fn iloc_positions(sel: &Bound<'_, PyAny>, height: usize) -> PyResult<Vec<usize>> {
    if let Some(pos) = as_bool_mask(sel, height) {
        return Ok(pos);
    }
    if let Ok(i) = sel.extract::<isize>() {
        return Ok(vec![norm_idx(i, height)?]);
    }
    if let Ok(slice) = sel.downcast::<PySlice>() {
        let info = slice.indices(height as isize)?;
        return Ok(strided(info.start, info.stop, info.step));
    }
    if let Ok(idxs) = sel.extract::<Vec<isize>>() {
        return idxs.into_iter().map(|i| norm_idx(i, height)).collect();
    }
    Err(PyTypeError::new_err(
        "iloc row selector must be an int, slice, int list, or boolean mask",
    ))
}

/// Resolve a `loc` row selector (bool-mask / label-slice / label / label-list) to
/// row positions.
fn loc_positions(sel: &Bound<'_, PyAny>, index: &Index, height: usize) -> PyResult<Vec<usize>> {
    if let Some(pos) = as_bool_mask(sel, height) {
        return Ok(pos);
    }
    if let Ok(slice) = sel.downcast::<PySlice>() {
        let (lo, hi) = label_bounds(slice, index)?;
        let (a, b) = index.label_slice(lo.as_ref(), hi.as_ref());
        return Ok((a..b).collect());
    }
    if let Ok(list) = sel.downcast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            let label = parse_label(&item, index)?;
            out.push(
                index
                    .position_of(&label)
                    .ok_or_else(|| PyKeyError::new_err("label not found"))?,
            );
        }
        return Ok(out);
    }
    let label = parse_label(sel, index)?;
    let pos = index
        .position_of(&label)
        .ok_or_else(|| PyKeyError::new_err("label not found"))?;
    Ok(vec![pos])
}

/// One axis of a 2-D `iloc` / `loc` get: a single scalar position (the axis is
/// reduced away, pandas-style) or a list of positions (the axis is kept).
enum AxisSel {
    One(usize),
    Many(Vec<usize>),
}

/// Resolve an `iloc` row axis: a bare int reduces the axis (`AxisSel::One`); a
/// slice / int-list / boolean mask keeps it (`AxisSel::Many`).
fn iloc_row_axis(sel: &Bound<'_, PyAny>, height: usize) -> PyResult<AxisSel> {
    if let Ok(i) = sel.extract::<isize>() {
        return Ok(AxisSel::One(norm_idx(i, height)?));
    }
    Ok(AxisSel::Many(iloc_positions(sel, height)?))
}

/// Resolve an `iloc` column axis (int / slice / int-list) to column positions.
fn iloc_col_axis(sel: &Bound<'_, PyAny>, width: usize) -> PyResult<AxisSel> {
    if let Ok(j) = sel.extract::<isize>() {
        return Ok(AxisSel::One(norm_idx(j, width)?));
    }
    if let Ok(slice) = sel.downcast::<PySlice>() {
        let info = slice.indices(width as isize)?;
        return Ok(AxisSel::Many(strided(info.start, info.stop, info.step)));
    }
    if let Ok(idxs) = sel.extract::<Vec<isize>>() {
        return Ok(AxisSel::Many(
            idxs.into_iter()
                .map(|j| norm_idx(j, width))
                .collect::<PyResult<_>>()?,
        ));
    }
    Err(PyTypeError::new_err(
        "iloc column selector must be an int, slice, or int list",
    ))
}

/// Resolve a `loc` row axis: a single label reduces the axis (`AxisSel::One`); a
/// label-slice / label-list / boolean mask keeps it (`AxisSel::Many`).
fn loc_row_axis(sel: &Bound<'_, PyAny>, index: &Index, height: usize) -> PyResult<AxisSel> {
    if as_bool_mask(sel, height).is_some()
        || sel.downcast::<PySlice>().is_ok()
        || sel.downcast::<PyList>().is_ok()
    {
        return Ok(AxisSel::Many(loc_positions(sel, index, height)?));
    }
    let label = parse_label(sel, index)?;
    let pos = index
        .position_of(&label)
        .ok_or_else(|| PyKeyError::new_err("label not found"))?;
    Ok(AxisSel::One(pos))
}

/// Resolve a `loc` column axis (name / name-list / inclusive name-slice) to
/// column positions.
fn loc_col_axis(sel: &Bound<'_, PyAny>, df: &DataFrame) -> PyResult<AxisSel> {
    let pos_of = |name: &str| {
        df.column_pos(name)
            .ok_or_else(|| PyKeyError::new_err(format!("column {name:?} not found")))
    };
    if let Ok(name) = sel.extract::<String>() {
        return Ok(AxisSel::One(pos_of(&name)?));
    }
    if let Ok(names) = sel.extract::<Vec<String>>() {
        return Ok(AxisSel::Many(
            names.iter().map(|n| pos_of(n)).collect::<PyResult<_>>()?,
        ));
    }
    if let Ok(slice) = sel.downcast::<PySlice>() {
        let start = slice.getattr("start")?;
        let stop = slice.getattr("stop")?;
        let lo = if start.is_none() {
            0
        } else {
            pos_of(&start.extract::<String>()?)?
        };
        let hi = if stop.is_none() {
            df.width().saturating_sub(1)
        } else {
            pos_of(&stop.extract::<String>()?)?
        };
        return Ok(AxisSel::Many((lo..=hi).collect()));
    }
    Err(PyTypeError::new_err(
        "loc column selector must be a name, name list, or name slice",
    ))
}

/// Project `df` onto `cols` (by position), carrying the index — the column-axis
/// counterpart of `DataFrame::take` (which selects rows).
fn project_cols(df: &DataFrame, cols: &[usize]) -> PyResult<DataFrame> {
    let names: Vec<String> = cols.iter().map(|&j| df.names()[j].clone()).collect();
    let data: Vec<Column> = cols.iter().map(|&j| df.columns()[j].clone()).collect();
    let idx = (*df.index().as_ref()).clone();
    DataFrame::new(names, data, Some(idx)).map_err(pyerr)
}

/// Build a 2-D `iloc` / `loc` get result from already-resolved row & column
/// positions, reproducing pandas's shape rules: scalar×scalar -> cell,
/// rows×col -> a column Series, row×cols -> the row (volas's 1-row frame), and
/// rows×cols -> a sub-frame.
fn select_2d(
    py: Python<'_>,
    df: &DataFrame,
    rows: AxisSel,
    cols: AxisSel,
) -> PyResult<Py<PyAny>> {
    match (rows, cols) {
        (AxisSel::One(i), AxisSel::One(j)) => Ok(scalar_to_py(py, &df.columns()[j], i)),
        (AxisSel::Many(r), AxisSel::One(j)) => {
            let sub = project_cols(df, &[j])?.take(&r);
            let name = sub.names()[0].clone();
            let col = sub.columns()[0].clone();
            let series = PySeries {
                inner: Series::new(Some(name), col, Arc::clone(sub.index())),
            };
            Ok(Py::new(py, series)?.into_any())
        }
        (AxisSel::One(i), AxisSel::Many(c)) => {
            let sub = project_cols(df, &c)?.take(&[i]);
            Ok(Py::new(py, PyRow { inner: sub })?.into_any())
        }
        (AxisSel::Many(r), AxisSel::Many(c)) => {
            let sub = project_cols(df, &c)?.take(&r);
            Ok(Py::new(py, PyDataFrame { inner: sub })?.into_any())
        }
    }
}

/// Split a `df.loc[rows, col]` / `df.iloc[rows, col]` assignment key into its two
/// parts, with a clear error directing to the supported 2-tuple form.
fn split_row_col<'py>(
    key: &Bound<'py, PyAny>,
    accessor: &str,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let tup = key.downcast::<PyTuple>().map_err(|_| {
        PyTypeError::new_err(format!(
            "{accessor} assignment needs a (rows, column) key, e.g. df.{accessor}[mask, 'col'] = value"
        ))
    })?;
    if tup.len() != 2 {
        return Err(PyTypeError::new_err(format!(
            "{accessor} assignment key must be (rows, column)"
        )));
    }
    Ok((tup.get_item(0)?, tup.get_item(1)?))
}

#[pyclass]
pub struct DataFrameILoc {
    parent: Py<PyDataFrame>,
}

#[pymethods]
impl DataFrameILoc {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let pf = self.parent.borrow(py);
        ensure_fresh(&pf.inner)?;
        // 2-D positional get: df.iloc[rows, cols], symmetric with __setitem__.
        if let Ok(tup) = key.downcast::<PyTuple>() {
            if tup.len() == 2 {
                let rows = iloc_row_axis(&tup.get_item(0)?, pf.inner.height())?;
                let cols = iloc_col_axis(&tup.get_item(1)?, pf.inner.width())?;
                return select_2d(py, &pf.inner, rows, cols);
            }
        }
        if let Ok(i) = key.extract::<isize>() {
            let i = norm_idx(i, pf.inner.height())?;
            return Ok(Py::new(py, row_at(&pf.inner, i))?.into_any());
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            let info = slice.indices(pf.inner.height() as isize)?;
            let sub = positional_slice(&pf.inner, &info);
            return Ok(Py::new(py, PyDataFrame { inner: sub })?.into_any());
        }
        // int-list / boolean-mask row selection -> sub-frame.
        let positions = iloc_positions(key, pf.inner.height())?;
        Ok(Py::new(py, PyDataFrame { inner: take_frame(&pf.inner, &positions) })?.into_any())
    }

    /// `df.iloc[i, j] = scalar` or `df.iloc[rows, j] = scalar | array` (positional;
    /// copy-on-write). `rows` is an int / slice / int-list / boolean mask; `j` is a
    /// column position.
    fn __setitem__(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut pf = self.parent.borrow_mut(py);
        ensure_fresh(&pf.inner)?;
        let (rows, col) = split_row_col(key, "iloc")?;
        let height = pf.inner.height();
        let j = norm_idx(col.extract::<isize>()?, pf.inner.width())?;
        let positions = iloc_positions(&rows, height)?;
        let target = pf.inner.columns()[j].dtype();
        let val = resolve_assignment(value, target, positions.len())?;
        pf.inner.assign_positions(j, &positions, &val).map_err(pyerr)
    }
}

fn row_at(df: &DataFrame, i: usize) -> PyRow {
    // `take` materializes the index label (Range -> Int64([i])) and preserves
    // every column's dtype — a faithful 1-row frame.
    PyRow {
        inner: df.take(&[i]),
    }
}

fn take_frame(df: &DataFrame, positions: &[usize]) -> DataFrame {
    // Delegates to core `take`, which carries column aliases onto the new frame.
    df.take(positions)
}

/// Format the `i`-th cell of a column as a CSV field (`na_rep` for NaN).
fn cell_to_csv(col: &Column, i: usize, na_rep: &str) -> String {
    match col {
        Column::F64(v) => {
            if v[i].is_nan() {
                na_rep.to_string()
            } else {
                v[i].to_string()
            }
        }
        Column::Bool(v) => if v[i] { "True" } else { "False" }.to_string(),
        Column::I64(v) => v[i].to_string(),
        Column::Str(v) => v[i].clone(),
        Column::Datetime(v) => datetime::format_ns(v[i]),
    }
}

/// Format the `i`-th index label as a CSV field.
fn index_label_csv(index: &Index, i: usize) -> String {
    match index {
        Index::Range(_) => i.to_string(),
        Index::Int64(v) => v[i].to_string(),
        Index::Datetime(v, tz) => datetime::format_ns_tz(v[i], *tz),
        Index::Str(v) => v[i].clone(),
    }
}

/// A positional slice: a contiguous `step == 1` slice uses `DataFrame::slice` (a
/// contiguous copy); a strided slice gathers the individual positions.
fn positional_slice(df: &DataFrame, info: &PySliceIndices) -> DataFrame {
    if info.step == 1 {
        df.slice(info.start.max(0) as usize, info.stop.max(0) as usize)
    } else {
        take_frame(df, &strided(info.start, info.stop, info.step))
    }
}

/// Slice a frame by a Python slice — positional for integer bounds, label-based
/// (DatetimeIndex) for string bounds.
fn slice_frame(df: &DataFrame, slice: &Bound<'_, PySlice>) -> PyResult<DataFrame> {
    let start_obj = slice.getattr("start")?;
    let stop_obj = slice.getattr("stop")?;
    let is_label = start_obj.extract::<String>().is_ok() || stop_obj.extract::<String>().is_ok();
    if is_label {
        let index = df.index();
        let lo = if start_obj.is_none() {
            None
        } else {
            Some(parse_label(&start_obj, index)?)
        };
        let hi = if stop_obj.is_none() {
            None
        } else {
            Some(parse_label(&stop_obj, index)?)
        };
        let (a, b) = index.label_slice(lo.as_ref(), hi.as_ref());
        Ok(df.slice(a, b))
    } else {
        let info = slice.indices(df.height() as isize)?;
        Ok(positional_slice(df, &info))
    }
}

fn label_bounds(
    slice: &Bound<'_, PySlice>,
    index: &Index,
) -> PyResult<(Option<Label>, Option<Label>)> {
    let start_obj = slice.getattr("start")?;
    let stop_obj = slice.getattr("stop")?;
    let lo = if start_obj.is_none() {
        None
    } else {
        Some(parse_label(&start_obj, index)?)
    };
    let hi = if stop_obj.is_none() {
        None
    } else {
        Some(parse_label(&stop_obj, index)?)
    };
    Ok((lo, hi))
}

/// `df.loc[...]` label indexer.
#[pyclass]
pub struct DataFrameLoc {
    parent: Py<PyDataFrame>,
}

#[pymethods]
impl DataFrameLoc {
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let pf = self.parent.borrow(py);
        ensure_fresh(&pf.inner)?;
        let index = pf.inner.index();
        // 2-D label get: df.loc[rows, col], symmetric with __setitem__.
        if let Ok(tup) = key.downcast::<PyTuple>() {
            if tup.len() == 2 {
                let rows = loc_row_axis(&tup.get_item(0)?, index, pf.inner.height())?;
                let cols = loc_col_axis(&tup.get_item(1)?, &pf.inner)?;
                return select_2d(py, &pf.inner, rows, cols);
            }
        }
        if let Ok(slice) = key.downcast::<PySlice>() {
            let (lo, hi) = label_bounds(slice, index)?;
            let (a, b) = index.label_slice(lo.as_ref(), hi.as_ref());
            return Ok(Py::new(py, PyDataFrame { inner: pf.inner.slice(a, b) })?.into_any());
        }
        // boolean-mask / label-list row selection -> sub-frame.
        if as_bool_mask(key, pf.inner.height()).is_some() || key.downcast::<PyList>().is_ok() {
            let positions = loc_positions(key, index, pf.inner.height())?;
            return Ok(
                Py::new(py, PyDataFrame { inner: take_frame(&pf.inner, &positions) })?.into_any(),
            );
        }
        let label = parse_label(key, index)?;
        let pos = index
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        Ok(Py::new(py, row_at(&pf.inner, pos))?.into_any())
    }

    /// `df.loc[rows, col] = scalar | array` (label-based; copy-on-write). `rows` is
    /// a boolean mask, a label slice, a single label, or a label list; `col` is a
    /// single column name. The classic `df.loc[mask, 'signal'] = 1`.
    fn __setitem__(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut pf = self.parent.borrow_mut(py);
        ensure_fresh(&pf.inner)?;
        let (rows, col) = split_row_col(key, "loc")?;
        let colname: String = col.extract().map_err(|_| {
            PyTypeError::new_err("loc assignment column must be a single column name")
        })?;
        let height = pf.inner.height();
        let positions = {
            let index = pf.inner.index();
            loc_positions(&rows, index, height)?
        };
        let j = pf
            .inner
            .column_pos(&colname)
            .ok_or_else(|| PyKeyError::new_err(format!("column {colname:?} not found")))?;
        let target = pf.inner.columns()[j].dtype();
        let val = resolve_assignment(value, target, positions.len())?;
        pf.inner.assign_positions(j, &positions, &val).map_err(pyerr)
    }
}

/// `df.iat[i, j]` scalar access by position.
#[pyclass]
pub struct DataFrameIat {
    parent: Py<PyDataFrame>,
}

#[pymethods]
impl DataFrameIat {
    fn __getitem__(&self, py: Python<'_>, key: (isize, isize)) -> PyResult<Py<PyAny>> {
        let pf = self.parent.borrow(py);
        ensure_fresh(&pf.inner)?;
        let i = norm_idx(key.0, pf.inner.height())?;
        let j = norm_idx(key.1, pf.inner.width())?;
        Ok(scalar_to_py(py, &pf.inner.columns()[j], i))
    }

    /// `df.iat[i, j] = scalar` — set a single cell by position (copy-on-write).
    fn __setitem__(
        &self,
        py: Python<'_>,
        key: (isize, isize),
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut pf = self.parent.borrow_mut(py);
        ensure_fresh(&pf.inner)?;
        let i = norm_idx(key.0, pf.inner.height())?;
        let j = norm_idx(key.1, pf.inner.width())?;
        let target = pf.inner.columns()[j].dtype();
        let val = scalar_to_column(value, target)?;
        pf.inner.assign_positions(j, &[i], &val).map_err(pyerr)
    }
}

/// `df.at[label, col]` scalar access by label + column name.
#[pyclass]
pub struct DataFrameAt {
    parent: Py<PyDataFrame>,
}

#[pymethods]
impl DataFrameAt {
    fn __getitem__(&self, py: Python<'_>, key: (Py<PyAny>, String)) -> PyResult<Py<PyAny>> {
        let pf = self.parent.borrow(py);
        ensure_fresh(&pf.inner)?;
        let index = pf.inner.index();
        let label = parse_label(key.0.bind(py), index)?;
        let i = index
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        let col = pf.inner.column(&key.1).map_err(pyerr)?;
        Ok(scalar_to_py(py, col, i))
    }

    /// `df.at[label, col] = scalar` — set a single cell by label + column name
    /// (copy-on-write).
    fn __setitem__(
        &self,
        py: Python<'_>,
        key: (Py<PyAny>, String),
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut pf = self.parent.borrow_mut(py);
        ensure_fresh(&pf.inner)?;
        let i = {
            let index = pf.inner.index();
            let label = parse_label(key.0.bind(py), index)?;
            index
                .position_of(&label)
                .ok_or_else(|| PyKeyError::new_err("label not found"))?
        };
        let j = pf
            .inner
            .column_pos(&key.1)
            .ok_or_else(|| PyKeyError::new_err(format!("column {:?} not found", key.1)))?;
        let target = pf.inner.columns()[j].dtype();
        let val = scalar_to_column(value, target)?;
        pf.inner.assign_positions(j, &[i], &val).map_err(pyerr)
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
            let (lo, hi) = label_bounds(slice, &self.inner.index)?;
            let (a, b) = self.inner.index.label_slice(lo.as_ref(), hi.as_ref());
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
        let label = parse_label(key, &self.inner.index)?;
        let pos = self
            .inner
            .index
            .position_of(&label)
            .ok_or_else(|| PyKeyError::new_err("label not found"))?;
        Ok(scalar_to_py(py, &self.inner.data, pos))
    }
}


/// The compiled module backing the `volas` package.
#[pymodule]
fn volas_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PySeries>()?;
    m.add_class::<PyRow>()?;
    m.add_class::<PyTimestamp>()?;
    m.add_class::<DataFrameILoc>()?;
    m.add_class::<DataFrameLoc>()?;
    m.add_class::<DataFrameIat>()?;
    m.add_class::<DataFrameAt>()?;
    m.add_class::<SeriesILoc>()?;
    m.add_class::<SeriesLoc>()?;
    m.add_class::<PyTimeFrame>()?;
    m.add_class::<PyCumulator>()?;
    m.add("DirectiveError", m.py().get_type::<DirectiveError>())?;
    m.add("DirectiveSyntaxError", m.py().get_type::<DirectiveSyntaxError>())?;
    m.add("DirectiveValueError", m.py().get_type::<DirectiveValueError>())?;
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    Ok(())
}
