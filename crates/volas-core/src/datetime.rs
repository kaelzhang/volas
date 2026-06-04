//! Datetime parsing / formatting for the `DatetimeIndex` (UTC-naive, nanoseconds).

use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, Timelike};

/// Parse a timestamp string to epoch nanoseconds (UTC, naive). Accepts the common
/// `YYYY-MM-DD[ HH:MM:SS]` forms.
pub fn parse_ns(s: &str) -> Option<i64> {
    let s = s.trim();
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S"] {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return dt.and_utc().timestamp_nanos_opt();
        }
    }
    for fmt in ["%Y-%m-%d", "%Y/%m/%d"] {
        if let Ok(d) = NaiveDate::parse_from_str(s, fmt) {
            return d
                .and_hms_opt(0, 0, 0)
                .and_then(|dt| dt.and_utc().timestamp_nanos_opt());
        }
    }
    None
}

/// Decompose epoch nanoseconds into civil UTC parts
/// `(year, month, day, hour, minute, second)` — used by time-frame unification.
pub fn civil_parts(ns: i64) -> (i64, i64, i64, i64, i64, i64) {
    let secs = ns.div_euclid(1_000_000_000);
    let nsub = ns.rem_euclid(1_000_000_000) as u32;
    let dt = DateTime::from_timestamp(secs, nsub)
        .unwrap_or_default()
        .naive_utc();
    (
        dt.year() as i64,
        dt.month() as i64,
        dt.day() as i64,
        dt.hour() as i64,
        dt.minute() as i64,
        dt.second() as i64,
    )
}

/// Format epoch nanoseconds as `YYYY-MM-DD HH:MM:SS` (UTC, naive).
pub fn format_ns(ns: i64) -> String {
    let secs = ns.div_euclid(1_000_000_000);
    let nsub = ns.rem_euclid(1_000_000_000) as u32;
    DateTime::from_timestamp(secs, nsub)
        .map(|dt| dt.naive_utc().format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let ns = parse_ns("2020-02-07 00:00:00").unwrap();
        assert_eq!(format_ns(ns), "2020-02-07 00:00:00");
        // date-only parses to midnight
        assert_eq!(parse_ns("2020-02-07").unwrap(), ns);
        assert!(parse_ns("not a date").is_none());
    }
}
