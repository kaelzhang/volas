//! ``Series.dt`` — the datetime accessor (pandas-aligned, Series-only by owner
//! ruling 2026-06-12: no ``DataFrame.dt``).
//!
//! Every member maps the scalar [`PyTimestamp`] kernel over the column, so a
//! component read on a column and on a scalar can never disagree. A ``NaT``
//! cell stays missing in the result (int64/bool/str validity, datetime ``NaT``
//! — `# C2`), matching pandas-nullable semantics.
//!
//! pandas members deliberately NOT exposed (contract waivers):
//! ``date`` / ``time`` / ``timetz`` / ``to_pydatetime`` (object-returning, C3),
//! ``to_period`` / ``freq`` (period dtype out-of-scope), and
//! ``tz_localize`` / ``tz_convert`` (a tz lives on the index or a scalar,
//! never on a value column — D3).

use std::sync::Arc;

use pyo3::prelude::*;

use volas_core::{datetime, Column, DataFrame, Series, Tz, Validity};

use crate::frame::PyDataFrame;
use crate::scalar::PyTimestamp;
use crate::series::PySeries;
use crate::pyerr;

/// The object returned by ``Series.dt``; the getter guarantees the column is
/// ``datetime64[ns]``.
#[pyclass(name = "DatetimeAccessor")]
pub struct PyDt {
    pub(crate) series: Series,
}

impl PyDt {
    fn ns(&self) -> &[i64] {
        match &self.series.data {
            Column::Datetime(v) => v,
            _ => unreachable!("the .dt getter guards the dtype"), // LCOV_EXCL_LINE
        }
    }

    fn ts(ns: i64) -> PyTimestamp {
        PyTimestamp { ns, tz: Tz::Naive } // a value column is tz-less (D3)
    }

    fn wrap(&self, data: Column) -> PySeries {
        PySeries {
            inner: Series::new(self.series.name.clone(), data, Arc::clone(&self.series.index)),
        }
    }

    /// An int64 component column; a NaT cell stays missing (`# C2`).
    fn map_i64(&self, f: impl Fn(&PyTimestamp) -> i64) -> PySeries {
        let v = self.ns();
        let validity = Validity::from_valid_iter(v.len(), v.iter().map(|&ns| ns != i64::MIN));
        let out = v
            .iter()
            .map(|&ns| if ns == i64::MIN { 0 } else { f(&Self::ts(ns)) })
            .collect();
        self.wrap(Column::i64_with(out, validity))
    }

    /// A bool predicate column; a NaT cell stays missing (`# C2`).
    fn map_bool(&self, f: impl Fn(&PyTimestamp) -> bool) -> PySeries {
        let v = self.ns();
        let validity = Validity::from_valid_iter(v.len(), v.iter().map(|&ns| ns != i64::MIN));
        let out = v
            .iter()
            .map(|&ns| ns != i64::MIN && f(&Self::ts(ns)))
            .collect();
        self.wrap(Column::bool_with(out, validity))
    }

    /// A str column; a NaT cell stays missing (`# C2`).
    fn map_str(&self, f: impl Fn(&PyTimestamp) -> String) -> PySeries {
        let v = self.ns();
        let validity = Validity::from_valid_iter(v.len(), v.iter().map(|&ns| ns != i64::MIN));
        let out = v
            .iter()
            .map(|&ns| if ns == i64::MIN { String::new() } else { f(&Self::ts(ns)) })
            .collect();
        self.wrap(Column::str_with(out, validity))
    }

    /// A datetime column; a NaT cell stays NaT.
    fn map_ns(&self, f: impl Fn(&PyTimestamp) -> i64) -> PySeries {
        let out = self
            .ns()
            .iter()
            .map(|&ns| if ns == i64::MIN { i64::MIN } else { f(&Self::ts(ns)) })
            .collect();
        self.wrap(Column::datetime(out))
    }
}

#[pymethods]
impl PyDt {
    /// Calendar year per element (pandas ``.dt.year``); int64, NaT -> NA.
    #[getter]
    fn year(&self) -> PySeries {
        self.map_i64(PyTimestamp::year)
    }
    /// Calendar month, 1..=12 (pandas ``.dt.month``).
    #[getter]
    fn month(&self) -> PySeries {
        self.map_i64(PyTimestamp::month)
    }
    /// Day of month, 1..=31 (pandas ``.dt.day``).
    #[getter]
    fn day(&self) -> PySeries {
        self.map_i64(PyTimestamp::day)
    }
    /// Hour, 0..=23 (pandas ``.dt.hour``).
    #[getter]
    fn hour(&self) -> PySeries {
        self.map_i64(PyTimestamp::hour)
    }
    /// Minute, 0..=59 (pandas ``.dt.minute``).
    #[getter]
    fn minute(&self) -> PySeries {
        self.map_i64(PyTimestamp::minute)
    }
    /// Second, 0..=59 (pandas ``.dt.second``).
    #[getter]
    fn second(&self) -> PySeries {
        self.map_i64(PyTimestamp::second)
    }
    /// Microsecond component, 0..=999_999 (pandas ``.dt.microsecond``).
    #[getter]
    fn microsecond(&self) -> PySeries {
        self.map_i64(PyTimestamp::microsecond)
    }
    /// Nanosecond component, 0..=999 (pandas ``.dt.nanosecond``).
    #[getter]
    fn nanosecond(&self) -> PySeries {
        self.map_i64(PyTimestamp::nanosecond)
    }
    /// Day of week, Monday=0 .. Sunday=6 (pandas ``.dt.dayofweek``).
    #[getter]
    fn dayofweek(&self) -> PySeries {
        self.map_i64(PyTimestamp::weekday)
    }
    /// Alias of ``dayofweek`` (pandas keeps both spellings).
    #[getter]
    fn day_of_week(&self) -> PySeries {
        self.map_i64(PyTimestamp::weekday)
    }
    /// Alias of ``dayofweek`` (a property here, like pandas ``.dt.weekday``).
    #[getter]
    fn weekday(&self) -> PySeries {
        self.map_i64(PyTimestamp::weekday)
    }
    /// Ordinal day of the year, 1..=366 (pandas ``.dt.dayofyear``).
    #[getter]
    fn dayofyear(&self) -> PySeries {
        self.map_i64(PyTimestamp::dayofyear)
    }
    /// Alias of ``dayofyear``.
    #[getter]
    fn day_of_year(&self) -> PySeries {
        self.map_i64(PyTimestamp::dayofyear)
    }
    /// Quarter of the year, 1..=4 (pandas ``.dt.quarter``).
    #[getter]
    fn quarter(&self) -> PySeries {
        self.map_i64(PyTimestamp::quarter)
    }
    /// Days in the element's month, 28..=31 (pandas ``.dt.days_in_month``).
    #[getter]
    fn days_in_month(&self) -> PySeries {
        self.map_i64(PyTimestamp::days_in_month)
    }
    /// Alias of ``days_in_month``.
    #[getter]
    fn daysinmonth(&self) -> PySeries {
        self.map_i64(PyTimestamp::days_in_month)
    }

    /// Whether the element is the first day of its month (pandas
    /// ``.dt.is_month_start``); bool, NaT -> NA.
    #[getter]
    fn is_month_start(&self) -> PySeries {
        self.map_bool(PyTimestamp::is_month_start)
    }
    /// Whether the element is the last day of its month.
    #[getter]
    fn is_month_end(&self) -> PySeries {
        self.map_bool(PyTimestamp::is_month_end)
    }
    /// Whether the element is the first day of its quarter.
    #[getter]
    fn is_quarter_start(&self) -> PySeries {
        self.map_bool(PyTimestamp::is_quarter_start)
    }
    /// Whether the element is the last day of its quarter.
    #[getter]
    fn is_quarter_end(&self) -> PySeries {
        self.map_bool(PyTimestamp::is_quarter_end)
    }
    /// Whether the element is January 1st.
    #[getter]
    fn is_year_start(&self) -> PySeries {
        self.map_bool(PyTimestamp::is_year_start)
    }
    /// Whether the element is December 31st.
    #[getter]
    fn is_year_end(&self) -> PySeries {
        self.map_bool(PyTimestamp::is_year_end)
    }
    /// Whether the element's year is a leap year.
    #[getter]
    fn is_leap_year(&self) -> PySeries {
        self.map_bool(PyTimestamp::is_leap_year)
    }

    /// The English day name per element, e.g. ``"Monday"`` (pandas
    /// ``.dt.day_name()``); str, NaT -> NA.
    fn day_name(&self) -> PySeries {
        self.map_str(|t| t.day_name().to_string())
    }
    /// The English month name per element, e.g. ``"June"``.
    fn month_name(&self) -> PySeries {
        self.map_str(|t| t.month_name().to_string())
    }
    /// Format each element with a ``strftime`` pattern (pandas
    /// ``.dt.strftime``); str, NaT -> NA.
    fn strftime(&self, fmt: &str) -> PyResult<PySeries> {
        let v = self.ns();
        let validity = Validity::from_valid_iter(v.len(), v.iter().map(|&ns| ns != i64::MIN));
        let mut out = Vec::with_capacity(v.len());
        for &ns in v {
            if ns == i64::MIN {
                out.push(String::new());
                continue;
            }
            out.push(datetime::strftime(ns, Tz::Naive, fmt).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid strftime format {fmt:?}"
                ))
            })?);
        }
        Ok(self.wrap(Column::str_with(out, validity)))
    }

    /// Midnight of each element's day (pandas ``.dt.normalize()``); datetime,
    /// NaT stays NaT.
    fn normalize(&self) -> PySeries {
        self.map_ns(|t| t.wall_floor(86_400_000_000_000))
    }
    /// Floor each element to a frequency boundary (``'D' 'h' '15min' …``,
    /// pandas ``.dt.floor``).
    fn floor(&self, freq: &str) -> PyResult<PySeries> {
        let unit = crate::scalar::parse_freq_ns(freq)?;
        Ok(self.map_ns(|t| t.wall_floor(unit)))
    }
    /// Ceil each element to a frequency boundary (see ``floor``).
    fn ceil(&self, freq: &str) -> PyResult<PySeries> {
        let unit = crate::scalar::parse_freq_ns(freq)?;
        Ok(self.map_ns(|t| {
            let f = t.wall_floor(unit);
            if f == t.ns { f } else { f + unit }
        }))
    }
    /// Round each element to the nearest frequency boundary, ties to even
    /// (pandas ``.dt.round``).
    fn round(&self, freq: &str) -> PyResult<PySeries> {
        let unit = crate::scalar::parse_freq_ns(freq)?;
        Ok(self.map_ns(|t| {
            let f = t.wall_floor(unit);
            let rem = t.ns - f;
            if rem * 2 > unit {
                f + unit
            } else if rem * 2 < unit {
                f
            } else if (f / unit) % 2 == 0 {
                f // tie -> the even multiple
            } else {
                f + unit
            }
        }))
    }

    /// The ISO calendar as a DataFrame with ``year`` / ``week`` / ``day``
    /// columns (pandas ``.dt.isocalendar()``); int64 columns, a NaT row is NA
    /// across all three.
    fn isocalendar(&self) -> PyResult<PyDataFrame> {
        let v = self.ns();
        let validity = Validity::from_valid_iter(v.len(), v.iter().map(|&ns| ns != i64::MIN));
        let mut years = Vec::with_capacity(v.len());
        let mut weeks = Vec::with_capacity(v.len());
        let mut days = Vec::with_capacity(v.len());
        for &ns in v {
            let (y, w, d) = if ns == i64::MIN {
                (0, 0, 0)
            } else {
                Self::ts(ns).iso_calendar()
            };
            years.push(y);
            weeks.push(w);
            days.push(d);
        }
        let cols = vec![
            Column::i64_with(years, validity.clone()),
            Column::i64_with(weeks, validity.clone()),
            Column::i64_with(days, validity),
        ];
        let names = vec!["year".to_string(), "week".to_string(), "day".to_string()];
        let df = DataFrame::new(names, cols, Some((*self.series.index).clone()))
            .map_err(pyerr)?;
        Ok(PyDataFrame::plain(df))
    }

    /// The column-level timezone — always ``None``: a tz lives on the index or
    /// a scalar ``Timestamp``, never on a value column (D3).
    #[getter]
    fn tz(&self) -> Option<String> {
        None
    }
    /// The storage unit — always ``'ns'`` (D1, like pandas ``.dt.unit``).
    #[getter]
    fn unit(&self) -> &'static str {
        "ns"
    }
}
