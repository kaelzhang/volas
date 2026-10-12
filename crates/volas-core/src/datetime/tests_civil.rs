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
    assert!(strftime(0, Tz::Offset(90_000), "%H").is_err());
}

#[test]
#[should_panic(expected = "valid civil date")]
fn days_from_civil_rejects_an_invalid_date_loudly() {
    days_from_civil(2024, 2, 30); // formerly the silent epoch day 0
}
