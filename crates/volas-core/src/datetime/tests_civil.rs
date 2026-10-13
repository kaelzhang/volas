//! The one civil decomposition and the civil helpers without silent fallbacks.

use super::*;

#[test]
fn civil_view_answers_every_calendar_field() {
    let c = Tz::Utc.civil(parse_ns("2024-12-30 13:14:15.5").unwrap());
    assert_eq!((c.year(), c.month(), c.day()), (2024, 12, 30));
    assert_eq!((c.hour(), c.minute(), c.second()), (13, 14, 15));
    assert_eq!((c.weekday(), c.day_of_year(), c.quarter()), (0, 365, 4));
    assert_eq!(c.iso_calendar(), (2025, 1, 1)); // ISO week 1 of 2025
    assert_eq!(c.days_in_month(), 31);
    assert!(c.is_leap_year() && !c.is_month_start() && !c.is_month_end());
    let feb = Tz::Utc.civil(parse_ns("2023-02-28").unwrap());
    assert!(feb.is_month_end() && !feb.is_leap_year() && feb.days_in_month() == 28);
    let q = Tz::Utc.civil(parse_ns("2024-04-01").unwrap());
    assert!(q.is_quarter_start() && !q.is_quarter_end() && !q.is_year_start());
    let ye = Tz::Utc.civil(parse_ns("2024-12-31").unwrap());
    assert!(ye.is_year_end() && ye.is_quarter_end());
    assert!(Tz::Utc.civil(0).is_year_start());
}

#[test]
fn civil_helpers_have_no_silent_fallback() {
    assert_eq!(days_from_civil(1970, 1, 1), 0);
    assert_eq!(days_from_civil(1677, 9, 21), -106_752);
    assert_eq!(civil_from_days(-106_752), (1677, 9, 21));
    assert_eq!(civil_from_days(106_751), (2262, 4, 11));
    assert_eq!(format_ns(i64::MAX), "2262-04-11 23:47:16.854775807");
    // a zone offset that is not under a day cannot format (no panic)
    assert!(strftime(0, &Tz::Offset(90_000), "%H").is_err());
}

#[test]
#[should_panic(expected = "valid civil date")]
fn days_from_civil_rejects_an_invalid_date_loudly() {
    days_from_civil(2024, 2, 30); // formerly the silent epoch day 0
}

#[test]
fn native_calendar_round_trips_every_day_on_the_instant_axis() {
    for day in -106_753..=106_752 {
        let (y, m, d) = civil_from_days(day);
        assert_eq!(days_from_civil(y, m, d), day);
        let c = Civil::new(y, m, d, 23, 59, 59, 999_999_999).unwrap();
        assert_eq!(Civil::from_epoch_seconds(c.epoch_seconds(), c.nanosecond()), c);
        assert_eq!(c.weekday(), (day + 3).rem_euclid(7));
        let jan1 = days_from_civil(y, 1, 1);
        assert_eq!(c.day_of_year(), day - jan1 + 1);
    }
    for ns in [i64::MIN + 1, -1_000_000_001, -1, 0, 1, i64::MAX] {
        assert_eq!(Civil::from_epoch_ns(ns).epoch_nanoseconds(), ns as i128);
    }
}

#[test]
fn native_calendar_handles_centuries_negative_years_and_wide_clocks() {
    for (year, leap) in [(-400, true), (-100, false), (0, true), (1600, true), (1700, false), (1800, false), (1900, false), (2000, true), (2100, false), (2400, true)] {
        assert_eq!(civil::is_leap_year(year), leap);
        assert_eq!(Civil::new(year, 2, 29, 0, 0, 0, 0).is_some(), leap);
        let c = Civil::new(year, 3, 1, 0, 0, 0, 0).unwrap();
        assert_eq!(c.day_of_year(), if leap { 61 } else { 60 });
        assert_eq!(civil_from_days(days_from_civil(year, 3, 1)), (year, 3, 1));
    }
    for year in [2020, 2021, 2024] {
        for month in 1..=12 {
            assert!(Civil::new(year, month, civil::days_in_month(year, month).unwrap(), 0, 0, 0, 0).unwrap().is_month_end());
        }
    }
    for parts in [(2024, 0, 1, 0, 0, 0, 0), (2024, 13, 1, 0, 0, 0, 0), (2024, 4, 31, 0, 0, 0, 0), (2024, 1, 0, 0, 0, 0, 0), (2024, 1, 1, 24, 0, 0, 0), (2024, 1, 1, 0, -1, 0, 0), (2024, 1, 1, 0, 0, 60, 0), (2024, 1, 1, 0, 0, 0, 1_000_000_000)] {
        let (y, mo, d, h, m, s, n) = parts;
        assert!(Civil::new(y, mo, d, h, m, s, n).is_none());
    }
    assert!(Civil::new(i64::MAX, 1, 1, 0, 0, 0, 0).is_none());
    assert!(Civil::new(1_000_000_000_000, 1, 1, 0, 0, 0, 0).is_none());
    for sec in [i64::MIN, -1, 0, i64::MAX] {
        let c = Civil::from_epoch_seconds(sec, 123);
        assert_eq!(Civil::new(c.year(), c.month(), c.day(), c.hour(), c.minute(), c.second(), 123), Some(c));
        assert_eq!(c.epoch_seconds(), sec);
    }
    assert!(Civil::from_epoch_seconds(i64::MAX, 0).checked_add_seconds(1).is_none());
    let shifted = Civil::from_epoch_ns(-1).checked_add_seconds(1).unwrap();
    assert_eq!(shifted.to_string(), "1970-01-01 00:00:00.999999999");
    for (text, iso) in [("2016-01-01", (2015, 53, 5)), ("2021-01-01", (2020, 53, 5)), ("2021-01-04", (2021, 1, 1)), ("2024-12-30", (2025, 1, 1)), ("2023-01-01", (2022, 52, 7))] {
        assert_eq!(Civil::from_epoch_ns(parse_ns(text).unwrap()).iso_calendar(), iso);
    }
}

#[test]
#[should_panic(expected = "civil fraction")]
fn native_calendar_rejects_an_invalid_fraction_in_an_internal_clock() {
    Civil::from_epoch_seconds(0, 1_000_000_000);
}

#[test]
fn native_epoch_day_extremes_remain_exact() {
    for days in [i64::MIN, i64::MIN + 1, -146_097, -1, 0, 146_097, i64::MAX - 719_468, i64::MAX - 1, i64::MAX] {
        let (year, month, day) = civil_from_days(days);
        assert_eq!(civil::checked_days_from_civil(year, month, day), Some(days));
    }
    assert_eq!(civil::checked_days_from_civil(i64::MIN, 1, 1), None);
}
