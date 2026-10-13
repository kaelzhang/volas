//! Scalar Python types: `volas.Timestamp`, the `volas.NA` singleton, and its type.


use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::PyDict;
use volas_core::datetime::{Civil, Rounding};
use volas_core::{
    datetime, Index,
    IndexKind, Label, Tz,
};

use crate::coerce::numpy_time_value;

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

/// A timedelta operand as nanoseconds: an `np.timedelta64` in any unit (scaled
/// exactly by the core unit owner — its `NaT`, a calendar unit, or a duration past
/// the i64 range raise) or a raw integer nanosecond count. Backs `Timestamp`
/// `+` / `-` arithmetic.
fn delta_ns(delta: &Bound<'_, PyAny>) -> PyResult<i64> {
    if let Ok(n) = delta.extract::<i64>() {
        return Ok(n);
    }
    let (unit, count, raw) = numpy_time_value(delta, "m").ok_or_else(|| {
        PyTypeError::new_err("a Timestamp operand must be an np.timedelta64 or an integer ns count")
    })??;
    datetime::timedelta64_to_ns(raw, &unit, count).map_err(pyerr)
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
/// Raises:
///     ValueError: text that is not a datetime, or a wall clock ``tz`` skips (a
///         DST gap) or repeats (a DST fold) — the message names which, and the zone
///         (as pandas). The same text used as a row label is a missing label
///         (``KeyError``, carrying that reason).
///     TypeError: a value of no timestamp type.
///     OverflowError: an instant outside the ``datetime64[ns]`` range.
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
            // A value that carries a zone keeps it: a Timestamp its own tz, an
            // aware datetime / pd.Timestamp its tzinfo, an offset-aware string its
            // offset (F25) — '... 09:00:00+08:00' stays a +08:00 instant.
            None => zone_of(value)?,
        };
        Ok(PyTimestamp {
            ns: parse_ts_in_tz(value, &tzv)?,
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
        match &self.tz {
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
    fn year(&self) -> i64 {
        self.civil().year()
    }
    /// Calendar month, 1..=12 (pandas `Timestamp.month`).
    #[getter]
    fn month(&self) -> i64 {
        self.civil().month()
    }
    /// Day of month, 1..=31 (pandas `Timestamp.day`).
    #[getter]
    fn day(&self) -> i64 {
        self.civil().day()
    }
    /// Hour, 0..=23 (pandas `Timestamp.hour`).
    #[getter]
    fn hour(&self) -> i64 {
        self.civil().hour()
    }
    /// Minute, 0..=59 (pandas `Timestamp.minute`).
    #[getter]
    fn minute(&self) -> i64 {
        self.civil().minute()
    }
    /// Second, 0..=59 (pandas `Timestamp.second`).
    #[getter]
    fn second(&self) -> i64 {
        self.civil().second()
    }
    /// Day of week with Monday=0 .. Sunday=6 (pandas `Timestamp.weekday()`).
    fn weekday(&self) -> i64 {
        self.civil().weekday()
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
    fn quarter(&self) -> i64 {
        self.civil().quarter()
    }
    /// Day of week, Monday=0 (pandas `Timestamp.dayofweek`).
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
    fn dayofyear(&self) -> i64 {
        self.civil().day_of_year()
    }
    /// Alias of `dayofyear`.
    #[getter]
    fn day_of_year(&self) -> i64 {
        self.dayofyear()
    }
    /// ISO week number, 1..=53 (pandas `Timestamp.week`).
    #[getter]
    fn week(&self) -> i64 {
        self.civil().iso_calendar().1
    }
    /// Alias of `week`.
    #[getter]
    fn weekofyear(&self) -> i64 {
        self.week()
    }
    /// The ISO calendar triple `(iso_year, iso_week, iso_weekday)` (stdlib / pandas `isocalendar()`).
    fn isocalendar(&self) -> (i64, i64, i64) {
        self.civil().iso_calendar()
    }
    /// Days in the timestamp's month, 28..=31 (pandas `Timestamp.days_in_month`).
    #[getter]
    fn days_in_month(&self) -> i64 {
        self.civil().days_in_month()
    }
    /// Alias of `days_in_month`.
    #[getter]
    fn daysinmonth(&self) -> i64 {
        self.days_in_month()
    }
    /// Whether this is the first day of its month (pandas `is_month_start`).
    #[getter]
    fn is_month_start(&self) -> bool {
        self.civil().is_month_start()
    }
    /// Whether this is the last day of its month (pandas `is_month_end`).
    #[getter]
    fn is_month_end(&self) -> bool {
        self.civil().is_month_end()
    }
    /// Whether this is the first day of its quarter (pandas `is_quarter_start`).
    #[getter]
    fn is_quarter_start(&self) -> bool {
        self.civil().is_quarter_start()
    }
    /// Whether this is the last day of its quarter (pandas `is_quarter_end`).
    #[getter]
    fn is_quarter_end(&self) -> bool {
        self.civil().is_quarter_end()
    }
    /// Whether this is January 1st (pandas `is_year_start`).
    #[getter]
    fn is_year_start(&self) -> bool {
        self.civil().is_year_start()
    }
    /// Whether this is December 31st (pandas `is_year_end`).
    #[getter]
    fn is_year_end(&self) -> bool {
        self.civil().is_year_end()
    }
    /// Whether the timestamp's year is a leap year (pandas `is_leap_year`).
    #[getter]
    fn is_leap_year(&self) -> bool {
        self.civil().is_leap_year()
    }
    /// The English day name, e.g. `"Monday"` (pandas `day_name()`).
    fn day_name(&self) -> &'static str {
        day_name(&self.civil())
    }
    /// The English month name, e.g. `"June"` (pandas `month_name()`).
    fn month_name(&self) -> &'static str {
        month_name(&self.civil())
    }

    /// The wall-clock calendar date as a python `datetime.date` (pandas `date()`).
    fn date<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let c = self.civil();
        py.import("datetime")?.getattr("date")?.call1((c.year(), c.month(), c.day()))
    }
    /// The wall-clock time-of-day as a python `datetime.time` (pandas `time()`).
    fn time<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let c = self.civil();
        let fields = (c.hour(), c.minute(), c.second(), self.microsecond());
        py.import("datetime")?.getattr("time")?.call1(fields)
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
        let fields = [year, month, day, hour, minute, second];
        let ns = self.tz.replace_civil(self.ns, fields).map_err(pyerr)?;
        Ok(PyTimestamp { ns, tz: self.tz.clone() })
    }

    /// Floor to a frequency boundary in the timestamp's wall-clock:
    /// `'D' 'h' 'min' 's' 'ms' 'us' 'ns'` with an optional multiple (`'15min'`).
    pub(crate) fn floor(&self, freq: &str) -> PyResult<Self> {
        self.rounded(parse_freq_ns(freq)?, Rounding::Floor)
    }
    /// Ceil to a frequency boundary (see `floor`).
    pub(crate) fn ceil(&self, freq: &str) -> PyResult<Self> {
        self.rounded(parse_freq_ns(freq)?, Rounding::Ceil)
    }
    /// Round to the nearest frequency boundary, ties to even (pandas `round`).
    pub(crate) fn round(&self, freq: &str) -> PyResult<Self> {
        self.rounded(parse_freq_ns(freq)?, Rounding::HalfEven)
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
        let ns = tzv.localize_from(self.ns, &self.tz).map_err(pyerr)?;
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
        crate::timezone::to_python(py, &self.tz)
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
        let c = self.civil();
        let subsec = self.ns.rem_euclid(1_000_000_000);
        let mut out = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            c.year(), c.month(), c.day(), c.hour(), c.minute(), c.second()
        );
        if subsec != 0 {
            if subsec % 1_000 == 0 {
                out.push_str(&format!(".{:06}", subsec / 1_000));
            } else {
                out.push_str(&format!(".{subsec:09}"));
            }
        }
        if self.tz.is_aware() {
            // second precision when the offset has seconds (LMT `+05:21:10`)
            out.push_str(&volas_core::tz::format_offset(self.tz.offset_secs_at(self.ns), ":"));
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
            Ok(PyTimestamp { ns: self.ns, tz: self.tz.clone() })
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
        datetime::strftime(self.ns, &self.tz, fmt).map_err(pyerr)
    }

    /// As a Python `datetime.datetime` at microsecond precision (`datetime`'s
    /// maximum; sub-µs ns are truncated): naive for a naive Timestamp, else aware
    /// with this Timestamp's `tzinfo` — the same instant, `fold` set on a
    /// DST-repeated wall clock.
    fn to_pydatetime<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dt = py.import("datetime")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("microseconds", self.ns.div_euclid(1_000))?;
        let delta = dt.getattr("timedelta")?.call((), Some(&kwargs))?;
        let epoch = dt.getattr("datetime")?;
        match self.tzinfo(py)? {
            None => epoch.call1((1970, 1, 1))?.call_method1("__add__", (delta,)),
            Some(zone) => {
                let utc = dt.getattr("timezone")?.getattr("utc")?;
                let epoch = epoch.call1((1970, 1, 1, 0, 0, 0, 0, utc))?;
                epoch.call_method1("__add__", (delta,))?.call_method1("astimezone", (zone,))
            }
        }
    }

    /// `ts + delta` where `delta` is an `np.timedelta64` or an integer count of
    /// nanoseconds, yielding a Timestamp (pandas `Timestamp + Timedelta`).
    /// Overflow raises — arithmetic is a constructor boundary (D2), so a wrapped
    /// result can never resurrect the NaT sentinel or a bogus 1677/2262 instant.
    fn __add__(&self, delta: &Bound<'_, PyAny>) -> PyResult<PyTimestamp> {
        let ns = datetime::shift(self.ns, delta_ns(delta)?).map_err(pyerr)?;
        Ok(PyTimestamp { ns, tz: self.tz.clone() })
    }

    /// `ts - other`: another Timestamp gives the `np.timedelta64` difference; an
    /// `np.timedelta64` / nanosecond count gives a shifted Timestamp. Both are
    /// checked like `__add__` — a difference that exceeds the i64 ns range (or
    /// lands on the NaT sentinel) raises instead of wrapping.
    fn __sub__<'py>(&self, py: Python<'py>, other: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(o) = other.extract::<PyRef<PyTimestamp>>() {
            // a difference past the i64 range, or one on the NaT sentinel, raises
            let delta = datetime::instant(self.ns.checked_sub(o.ns)).map_err(pyerr)?;
            return Ok(py
                .import("numpy")?
                .getattr("timedelta64")?
                .call1((delta, "ns"))?
                .into_any()
                .unbind());
        }
        let delta = delta_ns(other)?;
        let ns = datetime::instant(self.ns.checked_sub(delta)).map_err(pyerr)?;
        Ok(Py::new(py, PyTimestamp { ns, tz: self.tz.clone() })?.into_any())
    }

    fn __repr__(&self) -> String {
        match &self.tz {
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
        datetime::format_ns_tz(self.ns, &self.tz)
    }

    /// Compare with another instant (a Timestamp, a `datetime` / `pd.Timestamp` or
    /// an `np.datetime64`). A naive and an aware instant are never equal and cannot
    /// be ordered (`TypeError`), as in pandas. Anything else — a string, an integer
    /// (an epoch included), `None`, a plain `date`, `NaT`, … — is not an instant:
    /// `==` is False, `!=` True and ordering raises `TypeError` (pandas' rule; it
    /// keeps `a == b` implying `hash(a) == hash(b)`). Parse text with `Timestamp(s)`.
    /// NumPy defers a comparison with a `datetime64` / array to this class
    /// instead of first turning its operand into an integer (pandas' value).
    #[classattr]
    fn __array_priority__() -> f64 {
        100.0
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: pyo3::basic::CompareOp,
    ) -> PyResult<Py<PyAny>> {
        use pyo3::basic::CompareOp::{Eq, Ne};
        let np = py.import("numpy")?;
        if other.is_instance(&np.getattr("ndarray")?)? {
            return self.compare_array(py, &np, other, op);
        }
        let Some((ns, aware)) = comparable_instant(other)? else {
            return Ok(py.NotImplemented());
        };
        let answer = if aware == self.tz.is_aware() {
            op.matches(self.ns.cmp(&ns))
        } else {
            match op {
                Eq => false,
                Ne => true,
                _ => {
                    return Err(PyTypeError::new_err(
                        "Cannot compare tz-naive and tz-aware timestamps",
                    ))
                }
            }
        };
        Ok(answer.into_pyobject(py)?.to_owned().into_any().unbind())
    }

    /// Consistent with `==`: a whole-microsecond instant hashes like the equal
    /// `datetime` (so `datetime in {Timestamp}` works); a sub-microsecond one by
    /// its nanosecond value — pandas' rule (an equal sub-microsecond
    /// `np.datetime64` hashes by NumPy's own rule, as it does against pandas).
    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        if self.ns.rem_euclid(1_000) != 0 {
            return self.ns.into_pyobject(py)?.hash();
        }
        self.to_pydatetime(py)?.hash()
    }
}

/// The `(instant, is_aware)` of a comparison operand that is an instant — a
/// Timestamp, a `datetime.datetime` (incl. `pd.Timestamp`) or an `np.datetime64` —
/// or `None` for anything else (a string, an integer, `None`, a plain `date`,
/// `NaT`, …), which is not comparable with a Timestamp.
fn comparable_instant(other: &Bound<'_, PyAny>) -> PyResult<Option<(i64, bool)>> {
    if let Ok(t) = other.extract::<PyRef<PyTimestamp>>() {
        return Ok(Some((t.ns, t.tz.is_aware())));
    }
    if other.is_instance_of::<pyo3::types::PyDateTime>() {
        // `pd.NaT` is a datetime with no offset or components: not an instant
        let Ok(offset) = other.call_method0("utcoffset") else {
            return Ok(None);
        };
        let aware = !offset.is_none();
        return Ok(datetime_instant(other, &Tz::Naive).ok().map(|ns| (ns, aware)));
    }
    match crate::coerce::numpy_time_value(other, "M") {
        Some(value) => {
            let (unit, count, raw) = value?;
            if raw == datetime::NAT {
                return Ok(None);
            }
            Ok(Some((datetime::datetime64_to_ns(raw, &unit, count).map_err(pyerr)?, false)))
        }
        None => Ok(None),
    }
}

/// The zone a constructor value carries: a Timestamp's tz, the canonical key
/// rules of an aware datetime's keyed tzinfo (its fixed offset when unkeyed),
/// an offset-aware string's offset; [`Tz::Naive`] otherwise.
pub(crate) fn zone_of(value: &Bound<'_, PyAny>) -> PyResult<Tz> {
    if let Ok(t) = value.extract::<PyRef<PyTimestamp>>() {
        return Ok(t.tz.clone());
    }
    if value.is_instance_of::<pyo3::types::PyDateTime>() {
        // `pd.NaT` (a datetime without components) has no zone; the instant
        // parse then rejects it
        let offset = value.call_method0("utcoffset")?;
        if offset.is_none() || value.getattr("year")?.extract::<i64>().is_err() {
            return Ok(Tz::Naive);
        }
        let tzinfo = value.getattr("tzinfo")?;
        if ["key", "zone"].iter().any(|name| {
            tzinfo.getattr(*name).and_then(|key| key.extract::<String>()).is_ok()
        }) {
            return tz_from_py(&tzinfo);
        }
        return tz_from_py(&tzinfo).or_else(|_| Tz::fixed(offset_secs(&offset)?).map_err(pyerr));
    }
    Ok(match value.extract::<String>().ok().and_then(|s| datetime::offset_suffix_secs(&s)) {
        Some(off) => Tz::fixed(off).map_err(pyerr)?,
        None => Tz::Naive,
    })
}

/// A `datetime.timedelta` UTC offset as whole seconds; a sub-second offset raises
/// (volas offsets have second precision — truncating would shift the instant).
pub(crate) fn offset_secs(delta: &Bound<'_, PyAny>) -> PyResult<i32> {
    let micros: i64 = delta.getattr("microseconds")?.extract()?;
    if micros != 0 {
        return Err(PyValueError::new_err(
            "a UTC offset with a sub-second part is not supported (volas offsets have second precision)",
        ));
    }
    let days: i64 = delta.getattr("days")?.extract()?;
    let secs: i64 = delta.getattr("seconds")?.extract()?;
    i32::try_from(days * 86_400 + secs)
        .map_err(|_| PyValueError::new_err("a UTC offset must be under a day"))
}

/// A `datetime.datetime` / `datetime.date` (incl. `pd.Timestamp`) as an instant,
/// from its components — exact to the nanosecond (`pd.Timestamp.nanosecond`),
/// never through text. An aware value is read at its own UTC offset; a naive one
/// is a wall clock in `tz`. `pd.NaT` (no components) is a `ValueError`.
pub(crate) fn datetime_instant(value: &Bound<'_, PyAny>, tz: &Tz) -> PyResult<i64> {
    let missing = || PyValueError::new_err("NaT is not a valid timestamp; a missing instant is volas.NA");
    let field = |name: &str| -> PyResult<i64> { value.getattr(name)?.extract().map_err(|_| missing()) };
    let is_dt = value.is_instance_of::<pyo3::types::PyDateTime>();
    let time = |name: &str| if is_dt { field(name) } else { Ok(0) };
    let fields = [field("year")?, field("month")?, field("day")?, time("hour")?, time("minute")?, time("second")?];
    let nanos = time("microsecond")? * 1_000
        + if value.hasattr("nanosecond")? { field("nanosecond")? } else { 0 };
    let offset_zone = match is_dt.then(|| value.call_method0("utcoffset")).transpose()? {
        Some(off) if !off.is_none() => Some(Tz::fixed(offset_secs(&off)?).map_err(pyerr)?),
        _ => None,
    };
    offset_zone.as_ref().unwrap_or(tz).localize_fields(fields, nanos as u32).map_err(pyerr)
}

/// Parse a Python timestamp (datetime string or epoch-ns integer) to UTC ns,
/// interpreting a naive string's wall-clock as-is (zero offset).
pub(crate) fn parse_ts(key: &Bound<'_, PyAny>) -> PyResult<i64> {
    parse_ts_in_tz(key, &Tz::Naive)
}

/// Resolve a Python `tz=` argument: a string spec (IANA / `"+08:00"` / `"UTC"`),
/// a `zoneinfo.ZoneInfo` (its IANA key), or any `tzinfo` object (its fixed UTC
/// offset) — F29. An empty string is an error (F47, via `Tz::parse`).
pub(crate) fn tz_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Tz> {
    if let Ok(s) = obj.extract::<String>() {
        return crate::timezone::parse(obj.py(), &s);
    }
    // zoneinfo.ZoneInfo carries its IANA name in `.key` (pytz in `.zone`).
    for attr in ["key", "zone"] {
        if let Ok(key) = obj.getattr(attr).and_then(|k| k.extract::<String>()) {
            return crate::timezone::from_object(obj, &key);
        }
    }
    // any datetime.tzinfo: take its UTC offset (fixed-offset zones), exactly.
    if let Ok(off) = obj.call_method1("utcoffset", (obj.py().None(),)) {
        if !off.is_none() {
            return Tz::fixed(offset_secs(&off)?).map_err(pyerr);
        }
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
        IndexKind::Datetime(_, tz) => crate::convert::parse_label_in_tz(key, tz).map(Label::I64),
        _ => crate::convert::parse_label_in_tz(key, &Tz::Naive).map(Label::I64),
    }
}

/// The English day name of a civil day, e.g. `"Monday"` (pandas `day_name()`).
pub(crate) fn day_name(c: &Civil) -> &'static str {
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        [c.weekday() as usize]
}

/// The English month name of a civil day, e.g. `"June"` (pandas `month_name()`).
pub(crate) fn month_name(c: &Civil) -> &'static str {
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"]
        [(c.month() - 1) as usize]
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
    datetime::span_ns(mult, unit_ns).map_err(pyerr)
}

impl PyTimestamp {
    /// `self <op> array`, element-wise — pandas' `Timestamp` rule for an ndarray
    /// operand, never the scalar path: a `datetime64` array compares against this
    /// instant as `datetime64[ns]` (an aware Timestamp against that naive array is
    /// all-unequal, and ordering raises `TypeError`); an `object` array compares
    /// each element with this Timestamp; any other array is all-unequal, and its
    /// ordering is left to NumPy.
    fn compare_array(
        &self,
        py: Python<'_>,
        np: &Bound<'_, PyModule>,
        other: &Bound<'_, PyAny>,
        op: pyo3::basic::CompareOp,
    ) -> PyResult<Py<PyAny>> {
        use pyo3::basic::CompareOp::*;
        let name = match op {
            Lt => "lt",
            Le => "le",
            Eq => "eq",
            Ne => "ne",
            Gt => "gt",
            Ge => "ge",
        };
        let shape = other.getattr("shape")?;
        let fill = |value: bool| -> PyResult<Py<PyAny>> {
            let fun = if value { "ones" } else { "zeros" };
            Ok(np.call_method1(fun, (&shape, np.getattr("bool_")?))?.unbind())
        };
        let kind: String = other.getattr("dtype")?.getattr("kind")?.extract()?;
        let operator = py.import("operator")?;
        match kind.as_str() {
            "M" if self.tz.is_aware() => match op {
                Eq => fill(false),
                Ne => fill(true),
                _ => Err(PyTypeError::new_err("Cannot compare tz-naive and tz-aware timestamps")),
            },
            "M" => Ok(operator.call_method1(name, (self.to_numpy(py)?, other))?.unbind()),
            "O" => {
                let me = Py::new(py, PyTimestamp { ns: self.ns, tz: self.tz.clone() })?;
                let flat = other.call_method0("ravel")?;
                let cells = flat
                    .try_iter()?
                    .map(|x| operator.call_method1(name, (&me, x?))?.is_truthy())
                    .collect::<PyResult<Vec<bool>>>()?;
                Ok(numpy::IntoPyArray::into_pyarray(cells, py).call_method1("reshape", (&shape,))?.unbind())
            }
            _ => match op {
                Eq => fill(false),
                Ne => fill(true),
                _ => Ok(py.NotImplemented()),
            },
        }
    }

    /// Round the instant to a `unit_ns` grid of the WALL clock at its offset (so
    /// a daily floor lands on the local midnight for an anchored zone) — the one
    /// kernel behind scalar and `.dt` `floor` / `ceil` / `round` / `normalize`;
    /// a result outside the instant range raises `OverflowError`.
    pub(crate) fn rounded_ns(&self, unit_ns: i64, mode: Rounding) -> PyResult<i64> {
        datetime::round_wall(self.ns, self.tz.offset_secs_at(self.ns), unit_ns, mode).map_err(pyerr)
    }

    fn rounded(&self, unit_ns: i64, mode: Rounding) -> PyResult<Self> {
        Ok(PyTimestamp { ns: self.rounded_ns(unit_ns, mode)?, tz: self.tz.clone() })
    }

    /// The wall-clock calendar view of this instant in its zone — decomposed once
    /// per read; every calendar getter projects it.
    pub(crate) fn civil(&self) -> Civil {
        self.tz.civil(self.ns)
    }
}
