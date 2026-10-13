//! Gregorian civil calendar arithmetic, independent of the instant range and
//! time-zone database. A [`Civil`] is a validated wall clock; its decomposition
//! is shared by calendar fields, formatting, and time-zone rules.

use std::fmt;

use crate::tz::Tz;

/// A valid proleptic-Gregorian wall clock, with a nanosecond fraction.
///
/// Civil clocks deliberately have a wider range than datetime64[ns]: an offset
/// can carry the wall clock beyond an instant boundary while the instant itself
/// remains valid. The second count fits i64; nanoseconds are kept separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Civil {
    seconds: i64,
    nanosecond: u32,
    year: i64,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
}

impl Tz {
    /// Decompose one real instant in this zone once for every calendar field.
    pub fn civil(&self, ns: i64) -> Civil {
        self.wall(ns)
    }
}

/// Whether a proleptic-Gregorian year has 366 days.
pub fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// Month length, or `None` for a month outside 1..=12.
pub fn days_in_month(year: i64, month: i64) -> Option<i64> {
    Some(match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => return None,
    })
}

/// Epoch-day count of a valid civil date. Howard Hinnant's era decomposition
/// uses March as month zero, so leap days are the last day in each year.
pub fn checked_days_from_civil(year: i64, month: i64, day: i64) -> Option<i64> {
    if !(1..=days_in_month(year, month)?).contains(&day) {
        return None;
    }
    let year = year.checked_sub(i64::from(month <= 2))?;
    let era = year.div_euclid(400);
    let yoe = year.rem_euclid(400);
    let mp = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    i64::try_from(era as i128 * 146_097 + doe as i128 - 719_468).ok()
}

/// Inverse of [`checked_days_from_civil`]. The five-era adjustment at the i64
/// day limit keeps every intermediate in i64, including the Unix epoch shift.
/// Ordinary instant-derived days never need that adjustment.
#[inline]
pub fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let (z, extra_eras) = match days.checked_add(719_468) {
        Some(z) => (z, 0),
        None => (days - 5 * 146_097 + 719_468, 5),
    };
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe as i64 + (era + extra_eras) * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp as i64 + if mp < 10 { 3 } else { -9 };
    (year + i64::from(month <= 2), month, day as i64)
}

impl Civil {
    /// Validate a wall clock. Leap seconds are not instants on the Unix axis.
    pub fn new(
        year: i64,
        month: i64,
        day: i64,
        hour: i64,
        minute: i64,
        second: i64,
        nanosecond: u32,
    ) -> Option<Self> {
        if !(0..24).contains(&hour)
            || !(0..60).contains(&minute)
            || !(0..60).contains(&second)
            || nanosecond >= 1_000_000_000
        {
            return None;
        }
        let days = checked_days_from_civil(year, month, day)?;
        let seconds = i64::try_from(
            days as i128 * 86_400 + hour as i128 * 3_600 + minute as i128 * 60 + second as i128,
        )
        .ok()?;
        Some(Self {
            seconds,
            nanosecond,
            year,
            month: month as u8,
            day: day as u8,
            hour: hour as u8,
            minute: minute as u8,
            second: second as u8,
        })
    }

    /// Read a Unix second count and its fraction as a wall clock without a zone.
    /// `nanosecond` must be less than one second.
    #[inline]
    pub fn from_epoch_seconds(seconds: i64, nanosecond: u32) -> Self {
        assert!(
            nanosecond < 1_000_000_000,
            "a civil fraction is below one second"
        );
        let (year, month, day) = civil_from_days(seconds.div_euclid(86_400));
        let time = seconds.rem_euclid(86_400);
        Self {
            seconds,
            nanosecond,
            year,
            month: month as u8,
            day: day as u8,
            hour: (time / 3_600) as u8,
            minute: (time / 60 % 60) as u8,
            second: (time % 60) as u8,
        }
    }

    /// Read an epoch-nanosecond count as a wall clock without a zone.
    #[inline]
    pub fn from_epoch_ns(ns: i64) -> Self {
        Self::from_epoch_seconds(
            ns.div_euclid(1_000_000_000),
            ns.rem_euclid(1_000_000_000) as u32,
        )
    }

    /// This wall clock's whole second count, before applying a time-zone offset.
    pub fn epoch_seconds(&self) -> i64 {
        self.seconds
    }
    /// Exact wall-clock nanoseconds, before applying a time-zone offset.
    pub fn epoch_nanoseconds(&self) -> i128 {
        self.seconds as i128 * 1_000_000_000 + self.nanosecond as i128
    }
    /// Fraction of a second, 0..1_000_000_000.
    pub fn nanosecond(&self) -> u32 {
        self.nanosecond
    }
    /// Checked civil-clock arithmetic, preserving the fraction.
    pub fn checked_add_seconds(&self, delta: i64) -> Option<Self> {
        Some(Self::from_epoch_seconds(
            self.seconds.checked_add(delta)?,
            self.nanosecond,
        ))
    }
    /// Replace the time on the same calendar date.
    pub fn with_time(&self, hour: i64, minute: i64, second: i64, nanosecond: u32) -> Option<Self> {
        Self::new(
            self.year(),
            self.month(),
            self.day(),
            hour,
            minute,
            second,
            nanosecond,
        )
    }
    /// Calendar year.
    pub fn year(&self) -> i64 {
        self.year
    }
    /// Month, 1..=12.
    pub fn month(&self) -> i64 {
        self.month as i64
    }
    /// Day of the month, 1..=31.
    pub fn day(&self) -> i64 {
        self.day as i64
    }
    /// Hour, 0..=23.
    pub fn hour(&self) -> i64 {
        self.hour as i64
    }
    /// Minute, 0..=59.
    pub fn minute(&self) -> i64 {
        self.minute as i64
    }
    /// Second, 0..=59.
    pub fn second(&self) -> i64 {
        self.second as i64
    }
    /// Day of the week, Monday = 0 .. Sunday = 6.
    pub fn weekday(&self) -> i64 {
        (self.seconds.div_euclid(86_400) + 3).rem_euclid(7)
    }
    /// Ordinal day of the year, 1..=366.
    pub fn day_of_year(&self) -> i64 {
        const BEFORE: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
        BEFORE[self.month as usize - 1]
            + self.day()
            + i64::from(self.month > 2 && self.is_leap_year())
    }
    /// Quarter of the year, 1..=4.
    pub fn quarter(&self) -> i64 {
        (self.month() - 1) / 3 + 1
    }
    /// Days in this month, 28..=31.
    pub fn days_in_month(&self) -> i64 {
        days_in_month(self.year, self.month()).expect("a Civil has a valid month")
    }
    /// Whether the year is a leap year.
    pub fn is_leap_year(&self) -> bool {
        is_leap_year(self.year)
    }
    /// The first day of its month.
    pub fn is_month_start(&self) -> bool {
        self.day == 1
    }
    /// The last day of its month.
    pub fn is_month_end(&self) -> bool {
        self.day() == self.days_in_month()
    }
    /// The first day of its quarter.
    pub fn is_quarter_start(&self) -> bool {
        self.is_month_start() && (self.month - 1).is_multiple_of(3)
    }
    /// The last day of its quarter.
    pub fn is_quarter_end(&self) -> bool {
        self.month.is_multiple_of(3) && self.is_month_end()
    }
    /// January 1st.
    pub fn is_year_start(&self) -> bool {
        self.month == 1 && self.day == 1
    }
    /// December 31st.
    pub fn is_year_end(&self) -> bool {
        self.month == 12 && self.day == 31
    }
    /// The ISO calendar triple `(iso_year, iso_week, iso_weekday)`.
    pub fn iso_calendar(&self) -> (i64, i64, i64) {
        let weekday = self.weekday() + 1;
        let week = (self.day_of_year() - weekday + 10) / 7;
        if week == 0 {
            (self.year - 1, iso_weeks(self.year - 1), weekday)
        } else if week > iso_weeks(self.year) {
            (self.year + 1, 1, weekday)
        } else {
            (self.year, week, weekday)
        }
    }
}

fn iso_weeks(year: i64) -> i64 {
    let jan1 =
        (checked_days_from_civil(year, 1, 1).expect("civil year has January 1") + 3).rem_euclid(7);
    if jan1 == 3 || (jan1 == 2 && is_leap_year(year)) {
        53
    } else {
        52
    }
}

impl fmt::Display for Civil {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}{}",
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            super::subsec_suffix(self.nanosecond)
        )
    }
}
