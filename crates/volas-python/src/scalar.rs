//! Scalar Python types: `volas.Timestamp`, the `volas.NA` singleton, and its type.


use numpy::IntoPyArray;
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use volas_core::{
    datetime, Index,
    IndexKind, Label, Tz,
};

#[allow(unused_imports)]
use crate::*;

/// `volas.NA` — the singleton missing-value marker shown to users and returned by
/// element access on a missing int/bool cell. A pure symbol: physical storage
/// stays dtype-optimal (a float keeps `NaN`, an int/bool a validity bit).
#[pyclass(frozen, name = "NAType", module = "volas_rs")]
pub(crate) struct NaType;

#[pymethods]
impl NaType {
    fn __repr__(&self) -> &'static str {
        "<NA>"
    }
    // pandas' NA raises on truthiness; mirror it so `if s[i]:` can't silently
    // treat a missing value as False.
    fn __bool__(&self) -> PyResult<bool> {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "boolean value of volas.NA is ambiguous",
        ))
    }
}

pub(crate) static NA_SINGLETON: GILOnceCell<Py<PyAny>> = GILOnceCell::new();

/// The cached `volas.NA` singleton object.
pub(crate) fn na(py: Python<'_>) -> Py<PyAny> {
    NA_SINGLETON
        .get_or_init(py, || Py::new(py, NaType).expect("create volas.NA").into_any())
        .clone_ref(py)
}

/// A timedelta operand as nanoseconds: an `np.timedelta64` (normalised to ns) or a
/// raw integer nanosecond count. Backs `Timestamp` `+` / `-` arithmetic.
fn delta_ns(delta: &Bound<'_, PyAny>) -> PyResult<i64> {
    if let Ok(n) = delta.extract::<i64>() {
        return Ok(n);
    }
    delta
        .call_method1("astype", ("timedelta64[ns]",))?
        .call_method1("astype", ("int64",))?
        .call_method0("item")?
        .extract::<i64>()
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
    pub(crate) ns: i64,
    /// The zone `value` was specified in (for display).
    pub(crate) tz: Tz,
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

    /// Calendar year in the timestamp's timezone (pandas `Timestamp.year`).
    #[getter]
    fn year(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).0
    }
    /// Calendar month, 1..=12 (pandas `Timestamp.month`).
    #[getter]
    fn month(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).1
    }
    /// Day of month, 1..=31 (pandas `Timestamp.day`).
    #[getter]
    fn day(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).2
    }
    /// Hour, 0..=23 (pandas `Timestamp.hour`).
    #[getter]
    fn hour(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).3
    }
    /// Minute, 0..=59 (pandas `Timestamp.minute`).
    #[getter]
    fn minute(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).4
    }
    /// Second, 0..=59 (pandas `Timestamp.second`).
    #[getter]
    fn second(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).5
    }

    /// Day of week with Monday=0 .. Sunday=6 (pandas `Timestamp.weekday()`).
    fn weekday(&self) -> i64 {
        let (y, mo, d, ..) = datetime::civil_parts_tz(self.ns, self.tz);
        (datetime::days_from_civil(y, mo, d) + 3).rem_euclid(7)
    }

    /// Format the wall-clock time with a `strftime` format string (pandas
    /// `Timestamp.strftime`). Raises `ValueError` on an invalid format.
    fn strftime(&self, fmt: &str) -> PyResult<String> {
        datetime::strftime(self.ns, self.tz, fmt)
            .ok_or_else(|| PyValueError::new_err("invalid strftime format string"))
    }

    /// As a Python `datetime.datetime` (wall-clock in this Timestamp's tz, at
    /// microsecond precision — `datetime`'s maximum; sub-µs ns are truncated).
    fn to_pydatetime<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (y, mo, d, h, mi, s) = datetime::civil_parts_tz(self.ns, self.tz);
        let micros = self.ns.rem_euclid(1_000_000_000) / 1_000;
        py.import("datetime")?
            .getattr("datetime")?
            .call1((y, mo, d, h, mi, s, micros))
    }

    /// `ts + delta` where `delta` is an `np.timedelta64` or an integer count of
    /// nanoseconds, yielding a Timestamp (pandas `Timestamp + Timedelta`).
    fn __add__(&self, delta: &Bound<'_, PyAny>) -> PyResult<PyTimestamp> {
        Ok(PyTimestamp { ns: self.ns.wrapping_add(delta_ns(delta)?), tz: self.tz })
    }

    /// `ts - other`: another Timestamp gives the `np.timedelta64` difference; an
    /// `np.timedelta64` / nanosecond count gives a shifted Timestamp.
    fn __sub__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(o) = other.extract::<PyRef<PyTimestamp>>() {
            return Ok(py
                .import("numpy")?
                .getattr("timedelta64")?
                .call1((self.ns - o.ns, "ns"))?
                .into_any()
                .unbind());
        }
        Ok(Py::new(py, PyTimestamp { ns: self.ns.wrapping_sub(delta_ns(other)?), tz: self.tz })?
            .into_any())
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

    /// The readable wall-clock string (pandas `str(Timestamp)` form) — the
    /// object form stays in `repr`.
    fn __str__(&self) -> String {
        datetime::format_ns_tz(self.ns, self.tz)
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
    match index.kind() {
        IndexKind::Str(_) => key
            .extract::<String>()
            .map(Label::Str)
            .map_err(|_| PyKeyError::new_err("label must be a string for a string index")),
        IndexKind::Datetime(_, tz) => parse_ts_in_tz(key, *tz).map(Label::I64),
        _ => parse_ts(key).map(Label::I64),
    }
}
