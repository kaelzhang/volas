//! Scalar Python types: `volas.Timestamp`, the `volas.NA` singleton, and its type.


use pyo3::exceptions::{PyKeyError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::PyDict;
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

/// Validate a checked-arithmetic result as a representable ns value: inside the
/// i64 range and not `i64::MIN` (the NaT sentinel for both `datetime64[ns]` and
/// `timedelta64[ns]`, D2). `None` (overflow) or the sentinel raises
/// `OverflowError` — Timestamp arithmetic must never wrap into a value that
/// renders as `NaT` yet exposes real civil parts.
fn checked_ts_ns(ns: Option<i64>) -> PyResult<i64> {
    match ns {
        Some(v) if v != i64::MIN => Ok(v),
        _ => Err(PyOverflowError::new_err(
            "Timestamp arithmetic overflows the representable datetime64[ns] range",
        )),
    }
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
    fn new(value: &Bound<'_, PyAny>, tz: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let tzv = match tz {
            Some(obj) => tz_from_py(obj)?,
            // An offset-aware value string carries its own zone — keep it (F25),
            // so '... 09:00:00+08:00' stays a +08:00 instant, not a naive UTC one.
            None => match value.extract::<String>().ok().and_then(|s| datetime::offset_suffix_secs(&s)) {
                Some(0) => Tz::Utc,
                Some(off) => Tz::Offset(off),
                None => Tz::Naive,
            },
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

    /// The timezone name (`"UTC"` / `"+08:00"` / IANA), or `None` if naive
    /// (unanchored) — UTC-anchored and naive are distinct states (F13).
    #[getter]
    fn tz(&self) -> Option<String> {
        match self.tz {
            Tz::Naive => None,
            other => Some(other.name()),
        }
    }

    /// The instant as a NumPy `datetime64[ns]` **scalar** (UTC) — a scalar class
    /// converts to a scalar, matching the stub and pandas `Timestamp.to_numpy()`.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("numpy")?
            .getattr("datetime64")?
            .call1((self.ns, "ns"))
    }

    /// Calendar year in the timestamp's timezone (pandas `Timestamp.year`).
    #[getter]
    pub(crate) fn year(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).0
    }
    /// Calendar month, 1..=12 (pandas `Timestamp.month`).
    #[getter]
    pub(crate) fn month(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).1
    }
    /// Day of month, 1..=31 (pandas `Timestamp.day`).
    #[getter]
    pub(crate) fn day(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).2
    }
    /// Hour, 0..=23 (pandas `Timestamp.hour`).
    #[getter]
    pub(crate) fn hour(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).3
    }
    /// Minute, 0..=59 (pandas `Timestamp.minute`).
    #[getter]
    pub(crate) fn minute(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).4
    }
    /// Second, 0..=59 (pandas `Timestamp.second`).
    #[getter]
    pub(crate) fn second(&self) -> i64 {
        datetime::civil_parts_tz(self.ns, self.tz).5
    }

    /// Day of week with Monday=0 .. Sunday=6 (pandas `Timestamp.weekday()`).
    pub(crate) fn weekday(&self) -> i64 {
        let (y, mo, d, ..) = datetime::civil_parts_tz(self.ns, self.tz);
        (datetime::days_from_civil(y, mo, d) + 3).rem_euclid(7)
    }

    /// Microsecond component, 0..=999_999 (pandas `Timestamp.microsecond`).
    #[getter]
    pub(crate) fn microsecond(&self) -> i64 {
        self.ns.rem_euclid(1_000_000_000) / 1_000
    }
    /// Nanosecond component, 0..=999 (pandas `Timestamp.nanosecond`).
    #[getter]
    pub(crate) fn nanosecond(&self) -> i64 {
        self.ns.rem_euclid(1_000)
    }
    /// Quarter of the year, 1..=4 (pandas `Timestamp.quarter`).
    #[getter]
    pub(crate) fn quarter(&self) -> i64 {
        (self.month() - 1) / 3 + 1
    }
    /// Day of week, Monday=0 (pandas `Timestamp.dayofweek` — property form of
    /// `weekday()`).
    #[getter]
    fn dayofweek(&self) -> i64 {
        self.weekday()
    }
    /// Alias of `dayofweek` (pandas keeps both spellings).
    #[getter]
    fn day_of_week(&self) -> i64 {
        self.weekday()
    }
    /// ISO weekday, Monday=1 .. Sunday=7 (stdlib `datetime.isoweekday()`).
    fn isoweekday(&self) -> i64 {
        self.weekday() + 1
    }
    /// Ordinal day of the year, 1..=366 (pandas `Timestamp.dayofyear`).
    #[getter]
    pub(crate) fn dayofyear(&self) -> i64 {
        let (y, mo, d, ..) = datetime::civil_parts_tz(self.ns, self.tz);
        datetime::days_from_civil(y, mo, d) - datetime::days_from_civil(y, 1, 1) + 1
    }
    /// Alias of `dayofyear`.
    #[getter]
    fn day_of_year(&self) -> i64 {
        self.dayofyear()
    }
    /// ISO week number, 1..=53 (pandas `Timestamp.week`).
    #[getter]
    fn week(&self) -> i64 {
        self.iso_calendar().1
    }
    /// Alias of `week`.
    #[getter]
    fn weekofyear(&self) -> i64 {
        self.iso_calendar().1
    }
    /// The ISO calendar triple `(iso_year, iso_week, iso_weekday)` (stdlib /
    /// pandas `isocalendar()`).
    fn isocalendar(&self) -> (i64, i64, i64) {
        self.iso_calendar()
    }
    /// Days in the timestamp's month, 28..=31 (pandas `Timestamp.days_in_month`).
    #[getter]
    pub(crate) fn days_in_month(&self) -> i64 {
        let (y, mo, ..) = datetime::civil_parts_tz(self.ns, self.tz);
        days_in_month_of(y, mo)
    }
    /// Alias of `days_in_month`.
    #[getter]
    fn daysinmonth(&self) -> i64 {
        self.days_in_month()
    }
    /// Whether this is the first day of its month (pandas `is_month_start`).
    #[getter]
    pub(crate) fn is_month_start(&self) -> bool {
        self.day() == 1
    }
    /// Whether this is the last day of its month (pandas `is_month_end`).
    #[getter]
    pub(crate) fn is_month_end(&self) -> bool {
        self.day() == self.days_in_month()
    }
    /// Whether this is the first day of its quarter (pandas `is_quarter_start`).
    #[getter]
    pub(crate) fn is_quarter_start(&self) -> bool {
        self.day() == 1 && (self.month() - 1) % 3 == 0
    }
    /// Whether this is the last day of its quarter (pandas `is_quarter_end`).
    #[getter]
    pub(crate) fn is_quarter_end(&self) -> bool {
        self.month() % 3 == 0 && self.day() == self.days_in_month()
    }
    /// Whether this is January 1st (pandas `is_year_start`).
    #[getter]
    pub(crate) fn is_year_start(&self) -> bool {
        self.month() == 1 && self.day() == 1
    }
    /// Whether this is December 31st (pandas `is_year_end`).
    #[getter]
    pub(crate) fn is_year_end(&self) -> bool {
        self.month() == 12 && self.day() == 31
    }
    /// Whether the timestamp's year is a leap year (pandas `is_leap_year`).
    #[getter]
    pub(crate) fn is_leap_year(&self) -> bool {
        let y = self.year();
        (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
    }
    /// The English day name, e.g. `"Monday"` (pandas `day_name()`).
    pub(crate) fn day_name(&self) -> &'static str {
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            [self.weekday() as usize]
    }
    /// The English month name, e.g. `"June"` (pandas `month_name()`).
    pub(crate) fn month_name(&self) -> &'static str {
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"]
            [(self.month() - 1) as usize]
    }

    /// The wall-clock calendar date as a python `datetime.date` (pandas `date()`).
    fn date<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (y, mo, d, ..) = datetime::civil_parts_tz(self.ns, self.tz);
        py.import("datetime")?.getattr("date")?.call1((y, mo, d))
    }
    /// The wall-clock time-of-day as a python `datetime.time` (pandas `time()`).
    fn time<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (_, _, _, h, mi, se) = datetime::civil_parts_tz(self.ns, self.tz);
        let micros = self.microsecond();
        py.import("datetime")?.getattr("time")?.call1((h, mi, se, micros))
    }
    /// A new Timestamp with the given wall-clock components replaced (pandas
    /// `replace`); unspecified components are kept. The zone is unchanged.
    #[pyo3(signature = (year = None, month = None, day = None, hour = None, minute = None, second = None))]
    fn replace(
        &self,
        year: Option<i64>,
        month: Option<i64>,
        day: Option<i64>,
        hour: Option<i64>,
        minute: Option<i64>,
        second: Option<i64>,
    ) -> PyResult<Self> {
        let (y, mo, d, h, mi, se) = datetime::civil_parts_tz(self.ns, self.tz);
        let (y, mo, d, h, mi, se) = (
            year.unwrap_or(y), month.unwrap_or(mo), day.unwrap_or(d),
            hour.unwrap_or(h), minute.unwrap_or(mi), second.unwrap_or(se),
        );
        let subsec = self.ns.rem_euclid(1_000_000_000);
        let ns = self
            .tz
            .wall_to_utc_ns(y as i32, mo as u32, d as u32, h as u32, mi as u32, se as u32)
            .ok_or_else(|| PyValueError::new_err("replace produced an invalid wall-clock"))?
            + subsec;
        Ok(PyTimestamp { ns, tz: self.tz })
    }

    /// Floor to a frequency boundary in the timestamp's wall-clock:
    /// `'D' 'h' 'min' 's' 'ms' 'us' 'ns'` with an optional multiple (`'15min'`).
    pub(crate) fn floor(&self, freq: &str) -> PyResult<Self> {
        let unit = parse_freq_ns(freq)?;
        Ok(PyTimestamp { ns: self.wall_floor(unit), tz: self.tz })
    }
    /// Ceil to a frequency boundary (see `floor`).
    pub(crate) fn ceil(&self, freq: &str) -> PyResult<Self> {
        let unit = parse_freq_ns(freq)?;
        let f = self.wall_floor(unit);
        Ok(PyTimestamp { ns: if f == self.ns { f } else { f + unit }, tz: self.tz })
    }
    /// Round to the nearest frequency boundary, ties to even (pandas `round`).
    pub(crate) fn round(&self, freq: &str) -> PyResult<Self> {
        let unit = parse_freq_ns(freq)?;
        let f = self.wall_floor(unit);
        let rem = self.ns - f;
        // round half to even: past the midpoint, or exactly on it with an odd
        // floor multiple (so rounding up lands on the even one).
        let ns = if rem * 2 > unit || (rem * 2 == unit && (f / unit) % 2 != 0) {
            f + unit
        } else {
            f
        };
        Ok(PyTimestamp { ns, tz: self.tz })
    }
    /// Midnight of the timestamp's wall-clock day (pandas `normalize`).
    pub(crate) fn normalize(&self) -> PyResult<Self> {
        self.floor("D")
    }

    /// Anchor a NAIVE timestamp's wall-clock in `tz` (pandas `tz_localize`): the
    /// wall-clock is kept, the instant moves. An already-aware timestamp errors —
    /// use `tz_convert`.
    fn tz_localize(&self, tz: &Bound<'_, PyAny>) -> PyResult<Self> {
        if self.tz.is_aware() {
            return Err(PyTypeError::new_err(format!(
                "Timestamp is already tz-aware ({}); use tz_convert",
                self.tz.name()
            )));
        }
        let tzv = tz_from_py(tz)?;
        let (y, mo, d, h, mi, se) = datetime::civil_parts_tz(self.ns, self.tz);
        let subsec = self.ns.rem_euclid(1_000_000_000);
        let ns = tzv
            .wall_to_utc_ns(y as i32, mo as u32, d as u32, h as u32, mi as u32, se as u32)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "wall-clock does not exist in {} (or is DST-ambiguous)",
                    tzv.name()
                ))
            })?
            + subsec;
        Ok(PyTimestamp { ns, tz: tzv })
    }
    /// Restate an AWARE timestamp in another zone (pandas `tz_convert`): the
    /// instant is kept, only the wall-clock presentation moves. A naive
    /// timestamp errors — anchor it with `tz_localize` first.
    fn tz_convert(&self, tz: &Bound<'_, PyAny>) -> PyResult<Self> {
        if !self.tz.is_aware() {
            return Err(PyTypeError::new_err(
                "cannot tz_convert a tz-naive Timestamp; use tz_localize to anchor it first",
            ));
        }
        Ok(PyTimestamp { ns: self.ns, tz: tz_from_py(tz)? })
    }
    /// stdlib-datetime spelling of `tz_convert`.
    fn astimezone(&self, tz: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.tz_convert(tz)
    }
    /// The UTC offset as a `datetime.timedelta`, or `None` if naive (stdlib).
    fn utcoffset<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if !self.tz.is_aware() {
            return Ok(None);
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("seconds", self.tz.offset_secs_at(self.ns))?;
        Ok(Some(py.import("datetime")?.getattr("timedelta")?.call((), Some(&kwargs))?))
    }
    /// The zone name (`"UTC"` / `"+08:00"` / IANA), or `None` if naive (stdlib).
    fn tzname(&self) -> Option<String> {
        self.tz.is_aware().then(|| self.tz.name())
    }
    /// The DST component of the offset as a `datetime.timedelta`, or `None` if
    /// naive (stdlib `dst()`).
    fn dst<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if !self.tz.is_aware() {
            return Ok(None);
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("seconds", self.tz.dst_secs_at(self.ns))?;
        Ok(Some(py.import("datetime")?.getattr("timedelta")?.call((), Some(&kwargs))?))
    }
    /// The zone as a python `tzinfo` (`datetime.timezone` for UTC / fixed
    /// offsets, `zoneinfo.ZoneInfo` for IANA names), or `None` if naive.
    #[getter]
    fn tzinfo<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self.tz {
            Tz::Naive => Ok(None),
            Tz::Utc => Ok(Some(py.import("datetime")?.getattr("timezone")?.getattr("utc")?)),
            Tz::Offset(secs) => {
                let kwargs = PyDict::new(py);
                kwargs.set_item("seconds", secs)?;
                let delta = py.import("datetime")?.getattr("timedelta")?.call((), Some(&kwargs))?;
                Ok(Some(py.import("datetime")?.getattr("timezone")?.call1((delta,))?))
            }
            Tz::Named(_) => Ok(Some(
                py.import("zoneinfo")?.getattr("ZoneInfo")?.call1((self.tz.name(),))?,
            )),
        }
    }

    /// POSIX epoch seconds as a float (stdlib / pandas `timestamp()`).
    fn timestamp(&self) -> f64 {
        self.ns as f64 / 1e9
    }
    /// The instant as `np.datetime64[ns]` (pandas `to_datetime64`; same as
    /// `to_numpy`).
    fn to_datetime64<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.to_numpy(py)
    }
    /// The ISO-8601 wall-clock string, with the offset suffix when aware
    /// (stdlib / pandas `isoformat()`).
    fn isoformat(&self) -> String {
        let (y, mo, d, h, mi, se) = datetime::civil_parts_tz(self.ns, self.tz);
        let subsec = self.ns.rem_euclid(1_000_000_000);
        let mut out = format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{se:02}");
        if subsec != 0 {
            if subsec % 1_000 == 0 {
                out.push_str(&format!(".{:06}", subsec / 1_000));
            } else {
                out.push_str(&format!(".{subsec:09}"));
            }
        }
        if self.tz.is_aware() {
            let off = self.tz.offset_secs_at(self.ns);
            let sign = if off < 0 { '-' } else { '+' };
            let a = off.abs();
            out.push_str(&format!("{}{:02}:{:02}", sign, a / 3600, (a % 3600) / 60));
        }
        out
    }
    /// The storage resolution, always `"ns"` (pandas `Timestamp.unit`).
    #[getter]
    fn unit(&self) -> &'static str {
        "ns"
    }
    /// volas Timestamps are ns-resolution only: `as_unit('ns')` is the identity,
    /// any other unit raises (no silent precision change).
    fn as_unit(&self, unit: &str) -> PyResult<Self> {
        if unit == "ns" {
            Ok(PyTimestamp { ns: self.ns, tz: self.tz })
        } else {
            Err(PyValueError::new_err(
                "volas Timestamps are ns-resolution; only as_unit('ns') is supported",
            ))
        }
    }
    /// The current instant (pandas `Timestamp.now(tz=None)`): naive UTC
    /// wall-clock by default, or anchored in `tz`.
    #[staticmethod]
    #[pyo3(signature = (tz = None))]
    fn now(tz: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .as_nanos() as i64;
        let tzv = match tz {
            Some(obj) => tz_from_py(obj)?,
            None => Tz::Naive,
        };
        Ok(PyTimestamp { ns, tz: tzv })
    }
    /// Alias of `now` (pandas `Timestamp.today`).
    #[staticmethod]
    #[pyo3(signature = (tz = None))]
    fn today(tz: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Self::now(tz)
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
    /// Overflow raises — arithmetic is a constructor boundary (D2), so a wrapped
    /// result can never resurrect the NaT sentinel or a bogus 1677/2262 instant.
    fn __add__(&self, delta: &Bound<'_, PyAny>) -> PyResult<PyTimestamp> {
        Ok(PyTimestamp { ns: checked_ts_ns(self.ns.checked_add(delta_ns(delta)?))?, tz: self.tz })
    }

    /// `ts - other`: another Timestamp gives the `np.timedelta64` difference; an
    /// `np.timedelta64` / nanosecond count gives a shifted Timestamp. Both are
    /// checked like `__add__` — a difference that exceeds the i64 ns range (or
    /// lands on the NaT sentinel) raises instead of wrapping.
    fn __sub__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(o) = other.extract::<PyRef<PyTimestamp>>() {
            let delta = checked_ts_ns(self.ns.checked_sub(o.ns))?;
            return Ok(py
                .import("numpy")?
                .getattr("timedelta64")?
                .call1((delta, "ns"))?
                .into_any()
                .unbind());
        }
        Ok(Py::new(
            py,
            PyTimestamp { ns: checked_ts_ns(self.ns.checked_sub(delta_ns(other)?))?, tz: self.tz },
        )?
        .into_any())
    }

    fn __repr__(&self) -> String {
        match self.tz {
            Tz::Naive => format!("Timestamp('{}')", datetime::format_ns(self.ns)),
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
        let rhs = parse_ts_in_tz(other, Tz::Naive)?;
        Ok(op.matches(self.ns.cmp(&rhs)))
    }

    fn __hash__(&self) -> i64 {
        self.ns
    }
}

/// Parse a Python timestamp (datetime string or epoch-ns integer) to UTC ns,
/// interpreting a naive string's wall-clock as-is (zero offset).
pub(crate) fn parse_ts(key: &Bound<'_, PyAny>) -> PyResult<i64> {
    parse_ts_in_tz(key, Tz::Naive)
}

/// Resolve a Python `tz=` argument: a string spec (IANA / `"+08:00"` / `"UTC"`),
/// a `zoneinfo.ZoneInfo` (its IANA key), or any `tzinfo` object (its fixed UTC
/// offset) — F29. An empty string is an error (F47, via `Tz::parse`).
pub(crate) fn tz_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Tz> {
    if let Ok(s) = obj.extract::<String>() {
        return Tz::parse(&s).map_err(pyerr);
    }
    // zoneinfo.ZoneInfo carries its IANA name in `.key`.
    if let Ok(key) = obj.getattr("key").and_then(|k| k.extract::<String>()) {
        return Tz::parse(&key).map_err(pyerr);
    }
    // any datetime.tzinfo: take its UTC offset (fixed-offset zones).
    if let Ok(off) = obj
        .call_method1("utcoffset", (obj.py().None(),))
        .and_then(|d| d.call_method0("total_seconds"))
        .and_then(|s| s.extract::<f64>())
    {
        let secs = off as i32;
        return Ok(if secs == 0 { Tz::Utc } else { Tz::Offset(secs) });
    }
    Err(PyTypeError::new_err(
        "tz must be a timezone string (IANA name, '+08:00', 'UTC'), a zoneinfo.ZoneInfo, or a tzinfo",
    ))
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

/// Days in (year, month) of the proleptic Gregorian calendar.
fn days_in_month_of(y: i64, mo: i64) -> i64 {
    match mo {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                29
            } else {
                28
            }
        }
    }
}

/// Parse a pandas-style frequency spec (`'15min'`, `'D'`, `'h'`, `'s'`, …) to a
/// span in nanoseconds.
pub(crate) fn parse_freq_ns(freq: &str) -> PyResult<i64> {
    let f = freq.trim();
    let split = f.find(|c: char| !c.is_ascii_digit()).unwrap_or(f.len());
    let (mult_s, unit_s) = f.split_at(split);
    let mult: i64 = if mult_s.is_empty() { 1 } else { mult_s.parse().map_err(|_| {
        PyValueError::new_err(format!("invalid frequency {freq:?}"))
    })? };
    let unit_ns: i64 = match unit_s {
        "D" | "d" => 86_400_000_000_000,
        "h" | "H" => 3_600_000_000_000,
        "min" | "T" | "m" => 60_000_000_000,
        "s" | "S" => 1_000_000_000,
        "ms" => 1_000_000,
        "us" => 1_000,
        "ns" => 1,
        _ => {
            return Err(PyValueError::new_err(format!(
                "invalid frequency {freq:?} (use D/h/min/s/ms/us/ns with an optional multiple)"
            )))
        }
    };
    if mult <= 0 {
        return Err(PyValueError::new_err(format!("invalid frequency {freq:?}")));
    }
    Ok(mult * unit_ns)
}

impl PyTimestamp {
    /// Floor the instant to a `unit_ns` boundary of the WALL clock (so a daily
    /// floor lands on the local midnight for an anchored zone).
    pub(crate) fn wall_floor(&self, unit_ns: i64) -> i64 {
        let off = self.tz.offset_secs_at(self.ns) as i64 * 1_000_000_000;
        let wall = self.ns + off;
        wall.div_euclid(unit_ns) * unit_ns - off
    }

    /// The ISO calendar triple (iso_year, iso_week, iso_weekday).
    pub(crate) fn iso_calendar(&self) -> (i64, i64, i64) {
        let (y, mo, d, ..) = datetime::civil_parts_tz(self.ns, self.tz);
        datetime::iso_calendar(y, mo, d)
    }
}
