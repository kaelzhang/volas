//! [`Civil`] — the wall-clock calendar view of one instant in one zone, decomposed
//! once ([`Tz::civil`]) and read field by field: every calendar component and
//! predicate (`.dt` and `Timestamp` alike) comes from this one value, so an
//! element is never decomposed more than once and no field has a fallback.

use chrono::{Datelike, NaiveDateTime, Timelike};

use crate::tz::Tz;

/// The wall clock of an instant in a zone (see the module docs). Built only from a
/// real instant, so every field is a valid calendar value by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Civil(NaiveDateTime);

impl Tz {
    /// The wall-clock calendar view of the instant `ns` in this zone (`ns` is a
    /// real instant: callers keep `NaT` out, as for every civil read).
    pub fn civil(&self, ns: i64) -> Civil {
        Civil(self.wall(ns))
    }
}

impl Civil {
    /// Calendar year.
    pub fn year(&self) -> i64 {
        self.0.year() as i64
    }
    /// Month, 1..=12.
    pub fn month(&self) -> i64 {
        self.0.month() as i64
    }
    /// Day of the month, 1..=31.
    pub fn day(&self) -> i64 {
        self.0.day() as i64
    }
    /// Hour, 0..=23.
    pub fn hour(&self) -> i64 {
        self.0.hour() as i64
    }
    /// Minute, 0..=59.
    pub fn minute(&self) -> i64 {
        self.0.minute() as i64
    }
    /// Second, 0..=59.
    pub fn second(&self) -> i64 {
        self.0.second() as i64
    }
    /// Day of the week, Monday = 0 .. Sunday = 6.
    pub fn weekday(&self) -> i64 {
        self.0.weekday().num_days_from_monday() as i64
    }
    /// Ordinal day of the year, 1..=366.
    pub fn day_of_year(&self) -> i64 {
        self.0.ordinal() as i64
    }
    /// Quarter of the year, 1..=4.
    pub fn quarter(&self) -> i64 {
        (self.month() - 1) / 3 + 1
    }
    /// Days in this month, 28..=31.
    pub fn days_in_month(&self) -> i64 {
        self.0.num_days_in_month() as i64
    }
    /// Whether the year is a leap year.
    pub fn is_leap_year(&self) -> bool {
        self.0.date().leap_year()
    }
    /// The first day of its month.
    pub fn is_month_start(&self) -> bool {
        self.day() == 1
    }
    /// The last day of its month.
    pub fn is_month_end(&self) -> bool {
        self.day() == self.days_in_month()
    }
    /// The first day of its quarter.
    pub fn is_quarter_start(&self) -> bool {
        self.is_month_start() && (self.month() - 1) % 3 == 0
    }
    /// The last day of its quarter.
    pub fn is_quarter_end(&self) -> bool {
        self.month() % 3 == 0 && self.is_month_end()
    }
    /// January 1st.
    pub fn is_year_start(&self) -> bool {
        self.month() == 1 && self.day() == 1
    }
    /// December 31st.
    pub fn is_year_end(&self) -> bool {
        self.month() == 12 && self.day() == 31
    }
    /// The ISO calendar triple `(iso_year, iso_week, iso_weekday)`.
    pub fn iso_calendar(&self) -> (i64, i64, i64) {
        let w = self.0.iso_week();
        (w.year() as i64, w.week() as i64, self.weekday() + 1)
    }
}
