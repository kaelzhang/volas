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

use volas_core::ColumnView;
use std::sync::Arc;

use pyo3::prelude::*;

use volas_core::datetime::{Civil, Rounding};
use volas_core::{datetime, Column, DataFrame, Series, Tz};

use crate::frame::PyDataFrame;
use crate::scalar::{day_name, month_name, PyTimestamp};
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
        match self.series.data().view() {
            ColumnView::Datetime(v) => v,
            _ => unreachable!("the .dt getter guards the dtype"), // LCOV_EXCL_LINE
        }
    }

    fn ts(ns: i64) -> PyTimestamp {
        PyTimestamp { ns, tz: Tz::DATETIME_COLUMN } // a column holds UTC instants
    }

    fn wrap(&self, data: Column) -> PyResult<PySeries> {
        Ok(PySeries {
            inner: Series::new(self.series.name().map(str::to_owned), data, Arc::clone(self.series.index())).map_err(pyerr)?,
        })
    }

    /// A column of one value per element, computed from the element's wall-clock
    /// [`Civil`] view — decomposed ONCE per element, however many fields the value
    /// reads; a NaT cell stays missing (`# C2`).
    fn map_civil<T>(&self, f: impl Fn(&Civil) -> T) -> Vec<Option<T>> {
        self.map_ns(|ns| f(&Tz::DATETIME_COLUMN.civil(ns))) // the UTC wall clock
    }

    /// As [`Self::map_civil`] for a value read from the raw instant itself.
    fn map_ns<T>(&self, f: impl Fn(i64) -> T) -> Vec<Option<T>> {
        self.ns().iter().map(|&ns| (ns != datetime::NAT).then(|| f(ns))).collect()
    }

    /// An int64 calendar-component column.
    fn map_i64(&self, f: impl Fn(&Civil) -> i64) -> PyResult<PySeries> {
        self.wrap(Column::i64_from_options(self.map_civil(f)))
    }

    /// A bool calendar-predicate column.
    fn map_bool(&self, f: impl Fn(&Civil) -> bool) -> PyResult<PySeries> {
        self.wrap(Column::bool_from_options(self.map_civil(f)))
    }

    /// A str calendar-name column.
    fn map_str(&self, f: impl Fn(&Civil) -> &'static str) -> PyResult<PySeries> {
        self.wrap(Column::str_from_options(self.map_civil(|c| f(c).to_string())))
    }

    /// An int64 sub-second component column, read from the instant itself.
    fn sub_second(&self, f: impl Fn(i64) -> i64) -> PyResult<PySeries> {
        self.wrap(Column::i64_from_options(self.map_ns(f)))
    }

    /// A datetime column rounded on the `unit_ns` grid (the scalar kernel); a NaT
    /// cell stays NaT, an element whose result leaves the instant range raises.
    fn rounded(&self, unit_ns: i64, mode: Rounding) -> PyResult<PySeries> {
        let out = self
            .ns()
            .iter()
            .map(|&ns| {
                if ns == datetime::NAT { Ok(ns) } else { Self::ts(ns).rounded_ns(unit_ns, mode) }
            })
            .collect::<PyResult<Vec<i64>>>()?;
        self.wrap(Column::datetime(out))
    }
}

#[pymethods]
impl PyDt {
    /// Calendar year per element (pandas ``.dt.year``); int64, NaT -> NA.
    #[getter]
    fn year(&self) -> PyResult<PySeries> {
        self.map_i64(Civil::year)
    }
    /// Calendar month, 1..=12 (pandas ``.dt.month``).
    #[getter]
    fn month(&self) -> PyResult<PySeries> {
        self.map_i64(Civil::month)
    }
    /// Day of month, 1..=31 (pandas ``.dt.day``).
    #[getter]
    fn day(&self) -> PyResult<PySeries> {
        self.map_i64(Civil::day)
    }
    /// Hour, 0..=23 (pandas ``.dt.hour``).
    #[getter]
    fn hour(&self) -> PyResult<PySeries> {
        self.map_i64(Civil::hour)
    }
    /// Minute, 0..=59 (pandas ``.dt.minute``).
    #[getter]
    fn minute(&self) -> PyResult<PySeries> {
        self.map_i64(Civil::minute)
    }
    /// Second, 0..=59 (pandas ``.dt.second``).
    #[getter]
    fn second(&self) -> PyResult<PySeries> {
        self.map_i64(Civil::second)
    }
    /// Microsecond component, 0..=999_999 (pandas ``.dt.microsecond``).
    #[getter]
    fn microsecond(&self) -> PyResult<PySeries> {
        self.sub_second(|ns| ns.rem_euclid(1_000_000_000) / 1_000)
    }
    /// Nanosecond component, 0..=999 (pandas ``.dt.nanosecond``).
    #[getter]
    fn nanosecond(&self) -> PyResult<PySeries> {
        self.sub_second(|ns| ns.rem_euclid(1_000))
    }
    /// Day of week, Monday=0 .. Sunday=6 (pandas ``.dt.dayofweek``).
    #[getter]
    fn dayofweek(&self) -> PyResult<PySeries> {
        self.map_i64(Civil::weekday)
    }
    /// Alias of ``dayofweek`` (pandas keeps both spellings).
    #[getter]
    fn day_of_week(&self) -> PyResult<PySeries> {
        self.map_i64(Civil::weekday)
    }
    /// Alias of ``dayofweek`` (a property here, like pandas ``.dt.weekday``).
    #[getter]
    fn weekday(&self) -> PyResult<PySeries> {
        self.map_i64(Civil::weekday)
    }
    /// Ordinal day of the year, 1..=366 (pandas ``.dt.dayofyear``).
    #[getter]
    fn dayofyear(&self) -> PyResult<PySeries> {
        self.map_i64(Civil::day_of_year)
    }
    /// Alias of ``dayofyear``.
    #[getter]
    fn day_of_year(&self) -> PyResult<PySeries> {
        self.map_i64(Civil::day_of_year)
    }
    /// Quarter of the year, 1..=4 (pandas ``.dt.quarter``).
    #[getter]
    fn quarter(&self) -> PyResult<PySeries> {
        self.map_i64(Civil::quarter)
    }
    /// Days in the element's month, 28..=31 (pandas ``.dt.days_in_month``).
    #[getter]
    fn days_in_month(&self) -> PyResult<PySeries> {
        self.map_i64(Civil::days_in_month)
    }
    /// Alias of ``days_in_month``.
    #[getter]
    fn daysinmonth(&self) -> PyResult<PySeries> {
        self.map_i64(Civil::days_in_month)
    }

    /// Whether the element is the first day of its month (pandas
    /// ``.dt.is_month_start``); bool, NaT -> NA.
    #[getter]
    fn is_month_start(&self) -> PyResult<PySeries> {
        self.map_bool(Civil::is_month_start)
    }
    /// Whether the element is the last day of its month.
    #[getter]
    fn is_month_end(&self) -> PyResult<PySeries> {
        self.map_bool(Civil::is_month_end)
    }
    /// Whether the element is the first day of its quarter.
    #[getter]
    fn is_quarter_start(&self) -> PyResult<PySeries> {
        self.map_bool(Civil::is_quarter_start)
    }
    /// Whether the element is the last day of its quarter.
    #[getter]
    fn is_quarter_end(&self) -> PyResult<PySeries> {
        self.map_bool(Civil::is_quarter_end)
    }
    /// Whether the element is January 1st.
    #[getter]
    fn is_year_start(&self) -> PyResult<PySeries> {
        self.map_bool(Civil::is_year_start)
    }
    /// Whether the element is December 31st.
    #[getter]
    fn is_year_end(&self) -> PyResult<PySeries> {
        self.map_bool(Civil::is_year_end)
    }
    /// Whether the element's year is a leap year.
    #[getter]
    fn is_leap_year(&self) -> PyResult<PySeries> {
        self.map_bool(Civil::is_leap_year)
    }

    /// The English day name per element, e.g. ``"Monday"`` (pandas
    /// ``.dt.day_name()``); str, NaT -> NA.
    fn day_name(&self) -> PyResult<PySeries> {
        self.map_str(day_name)
    }
    /// The English month name per element, e.g. ``"June"``.
    fn month_name(&self) -> PyResult<PySeries> {
        self.map_str(month_name)
    }
    /// Format each element with a ``strftime`` pattern (pandas
    /// ``.dt.strftime``); str, NaT -> NA.
    fn strftime(&self, fmt: &str) -> PyResult<PySeries> {
        let formatter = datetime::Strftime::new(fmt);
        let out = self
            .map_ns(|ns| formatter.format(ns, &Tz::DATETIME_COLUMN))
            .into_iter()
            .map(Option::transpose)
            .collect::<volas_core::Result<Vec<_>>>()
            .map_err(pyerr)?;
        self.wrap(Column::str_from_options(out))
    }

    /// Midnight of each element's day (pandas ``.dt.normalize()``); datetime,
    /// NaT stays NaT.
    fn normalize(&self) -> PyResult<PySeries> {
        self.rounded(86_400_000_000_000, Rounding::Floor)
    }
    /// Floor each element to a frequency boundary (``'D' 'h' '15min' …``,
    /// pandas ``.dt.floor``).
    fn floor(&self, freq: &str) -> PyResult<PySeries> {
        self.rounded(crate::scalar::parse_freq_ns(freq)?, Rounding::Floor)
    }
    /// Ceil each element to a frequency boundary (see ``floor``).
    fn ceil(&self, freq: &str) -> PyResult<PySeries> {
        self.rounded(crate::scalar::parse_freq_ns(freq)?, Rounding::Ceil)
    }
    /// Round each element to the nearest frequency boundary, ties to even
    /// (pandas ``.dt.round``).
    fn round(&self, freq: &str) -> PyResult<PySeries> {
        self.rounded(crate::scalar::parse_freq_ns(freq)?, Rounding::HalfEven)
    }

    /// The ISO calendar as a DataFrame with ``year`` / ``week`` / ``day``
    /// columns (pandas ``.dt.isocalendar()``); int64 columns, a NaT row is NA
    /// across all three.
    fn isocalendar(&self) -> PyResult<PyDataFrame> {
        // one civil decomposition per element, read three times
        let iso = self.map_civil(Civil::iso_calendar);
        let part = |k: fn((i64, i64, i64)) -> i64| Column::i64_from_options(iso.iter().map(|c| c.map(k)));
        let cols = vec![part(|c| c.0), part(|c| c.1), part(|c| c.2)];
        let names = vec!["year".to_string(), "week".to_string(), "day".to_string()];
        let df = DataFrame::new(names, cols, Some(self.series.index().as_ref().clone()))
            .map_err(pyerr)?;
        Ok(PyDataFrame::plain(df))
    }

    /// The column-level timezone — always ``'UTC'``: a datetime column holds UTC
    /// instants and stores no zone of its own (a display zone lives on the index).
    #[getter]
    fn tz(&self) -> String {
        Tz::DATETIME_COLUMN.name()
    }
    /// The storage unit — always ``'ns'`` (D1, like pandas ``.dt.unit``).
    #[getter]
    fn unit(&self) -> &'static str {
        "ns"
    }
}
