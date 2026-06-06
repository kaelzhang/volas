//! Datetime parsing / formatting for the `DatetimeIndex`. Storage is **UTC
//! epoch-ns**; per-frame display/matching tz lives in [`crate::tz`].

use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, Timelike};

use crate::tz::Tz;

/// Parse a timestamp string to **UTC** epoch nanoseconds. An **offset-aware**
/// string (RFC3339 `...+HH:MM` / `...Z`) is an absolute instant and converts
/// directly; a **naive** `YYYY-MM-DD[ HH:MM:SS]` string is interpreted as UTC.
pub fn parse_ns(s: &str) -> Option<i64> {
    parse_ns_in_tz(s, Tz::Utc)
}

/// Like [`parse_ns`], but a **naive** string is interpreted in `tz` (then stored
/// as UTC). An **offset-aware** string is already absolute, so `tz` is ignored.
pub fn parse_ns_in_tz(s: &str, tz: Tz) -> Option<i64> {
    let s = s.trim();
    if let Some(ns) = parse_offset_aware(s) {
        return Some(ns);
    }
    let (y, mo, d, h, mi, se) = naive_parts(s)?;
    tz.wall_to_utc_ns(y, mo, d, h, mi, se)
}

/// Parse an **offset-aware** string (RFC3339 / `%z`) to an absolute UTC instant.
fn parse_offset_aware(s: &str) -> Option<i64> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return dt.timestamp_nanos_opt();
    }
    for fmt in [
        "%Y-%m-%d %H:%M:%S%:z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
    ] {
        if let Ok(dt) = DateTime::parse_from_str(s, fmt) {
            return dt.timestamp_nanos_opt();
        }
    }
    None
}

/// Parse the supported **naive** forms into civil parts (no tz applied yet).
fn naive_parts(s: &str) -> Option<(i32, u32, u32, u32, u32, u32)> {
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S"] {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Some((
                dt.year(),
                dt.month(),
                dt.day(),
                dt.hour(),
                dt.minute(),
                dt.second(),
            ));
        }
    }
    for fmt in ["%Y-%m-%d", "%Y/%m/%d"] {
        if let Ok(d) = NaiveDate::parse_from_str(s, fmt) {
            return Some((d.year(), d.month(), d.day(), 0, 0, 0));
        }
    }
    None
}

/// Convert an epoch integer with a unit (`"s"` / `"ms"` / `"us"` / `"ns"`) to
/// UTC epoch-ns. The most robust ingestion path for exchange APIs that return a
/// numeric timestamp. `None` on an unknown unit or on overflow.
pub fn epoch_to_ns(value: i64, unit: &str) -> Option<i64> {
    let scale: i64 = match unit {
        "s" => 1_000_000_000,
        "ms" => 1_000_000,
        "us" => 1_000,
        "ns" => 1,
        _ => return None,
    };
    value.checked_mul(scale)
}

/// Scale a **floating-point** epoch `value` to nanoseconds, rounding to the
/// nearest nanosecond so sub-`unit` fractions are preserved (matching
/// `pandas.to_datetime(..., unit=...)`). Returns `None` on an unknown `unit` or a
/// non-finite / out-of-`i64`-range result.
pub fn epoch_to_ns_f64(value: f64, unit: &str) -> Option<i64> {
    let scale: f64 = match unit {
        "s" => 1_000_000_000.0,
        "ms" => 1_000_000.0,
        "us" => 1_000.0,
        "ns" => 1.0,
        _ => return None,
    };
    let ns = (value * scale).round();
    if ns.is_finite() && ns >= i64::MIN as f64 && ns <= i64::MAX as f64 {
        Some(ns as i64)
    } else {
        None
    }
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

/// Continuous count of days from the Unix epoch (`1970-01-01` = 0) for a civil
/// date — the basis for continuous, month-independent week / multi-day buckets.
/// `1970-01-01` is a Thursday, so a Monday-anchored week index is
/// `(days_from_civil(y, mo, d) + 3).div_euclid(7)`.
pub fn days_from_civil(y: i64, mo: i64, d: i64) -> i64 {
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch is valid");
    let date = NaiveDate::from_ymd_opt(y as i32, mo as u32, d as u32).unwrap_or(epoch);
    date.signed_duration_since(epoch).num_days()
}

/// Format epoch nanoseconds as `YYYY-MM-DD HH:MM:SS` (UTC, naive).
pub fn format_ns(ns: i64) -> String {
    let secs = ns.div_euclid(1_000_000_000);
    let nsub = ns.rem_euclid(1_000_000_000) as u32;
    DateTime::from_timestamp(secs, nsub)
        .map(|dt| dt.naive_utc().format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_default()
}

/// Format epoch nanoseconds as the wall-clock `YYYY-MM-DD HH:MM:SS` in `tz` (the
/// human/string form; bulk numpy export stays UTC).
pub fn format_ns_tz(ns: i64, tz: Tz) -> String {
    if tz.is_utc() {
        return format_ns(ns);
    }
    let (y, mo, d, h, mi, s) = tz.civil_parts(ns);
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02}")
}

/// The wall-clock civil parts `(year, month, day, hour, minute, second)` of the
/// instant `ns` in `tz` (UTC parts when `tz` is UTC).
pub fn civil_parts_tz(ns: i64, tz: Tz) -> (i64, i64, i64, i64, i64, i64) {
    if tz.is_utc() {
        civil_parts(ns)
    } else {
        tz.civil_parts(ns)
    }
}

/// Format the instant `ns` as wall-clock in `tz` with a `strftime` `fmt`. Returns
/// `None` for an invalid format string (so the caller can raise a clean error
/// instead of chrono panicking) or an out-of-range civil date.
pub fn strftime(ns: i64, tz: Tz, fmt: &str) -> Option<String> {
    use chrono::format::{Item, StrftimeItems};
    let (y, mo, d, h, mi, s) = civil_parts_tz(ns, tz);
    let dt = NaiveDate::from_ymd_opt(y as i32, mo as u32, d as u32)?
        .and_hms_opt(h as u32, mi as u32, s as u32)?;
    let items: Vec<Item> = StrftimeItems::new(fmt).collect();
    if items.iter().any(|it| matches!(it, Item::Error)) {
        return None;
    }
    Some(dt.format_with_items(items.iter()).to_string())
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

    #[test]
    fn offset_aware_strings_are_absolute() {
        // `+08:00` is an absolute instant -> stored as UTC (08:00 - 8h = 00:00).
        assert_eq!(
            format_ns(parse_ns("2020-01-01T08:00:00+08:00").unwrap()),
            "2020-01-01 00:00:00"
        );
        // `Z` is UTC.
        assert_eq!(
            format_ns(parse_ns("2020-01-01T00:00:00Z").unwrap()),
            "2020-01-01 00:00:00"
        );
        // space-separated with offset also works.
        assert_eq!(
            format_ns(parse_ns("2020-01-01 09:30:00 -05:00").unwrap()),
            "2020-01-01 14:30:00"
        );
    }

    #[test]
    fn naive_string_interpreted_in_tz() {
        let tz = Tz::parse("+08:00").unwrap();
        // a naive local string is shifted to UTC by the frame tz.
        assert_eq!(
            format_ns(parse_ns_in_tz("2020-01-01 08:00:00", tz).unwrap()),
            "2020-01-01 00:00:00"
        );
        // but an offset-aware string ignores the frame tz (already absolute).
        assert_eq!(
            format_ns(parse_ns_in_tz("2020-01-01T00:00:00Z", tz).unwrap()),
            "2020-01-01 00:00:00"
        );
    }

    #[test]
    fn epoch_units() {
        // 1_577_836_800_000 ms == 2020-01-01 00:00:00 UTC
        assert_eq!(
            format_ns(epoch_to_ns(1_577_836_800_000, "ms").unwrap()),
            "2020-01-01 00:00:00"
        );
        assert_eq!(epoch_to_ns(1_577_836_800, "s").unwrap(), 1_577_836_800_000_000_000);
        assert!(epoch_to_ns(1, "weeks").is_none());
    }

    #[test]
    fn epoch_units_f64_preserves_fraction() {
        // whole seconds match the integer path
        assert_eq!(epoch_to_ns_f64(1_577_836_800.0, "s").unwrap(), 1_577_836_800_000_000_000);
        // a fractional second is preserved (0.5 s == 500_000_000 ns)
        assert_eq!(epoch_to_ns_f64(1_577_836_800.5, "s").unwrap(), 1_577_836_800_500_000_000);
        // ms / us / ns scales
        assert_eq!(epoch_to_ns_f64(1.0, "ms").unwrap(), 1_000_000);
        assert_eq!(epoch_to_ns_f64(1.0, "us").unwrap(), 1_000);
        assert_eq!(epoch_to_ns_f64(1.0, "ns").unwrap(), 1);
        // unknown unit, non-finite, and out-of-range all return None
        assert!(epoch_to_ns_f64(1.0, "weeks").is_none());
        assert!(epoch_to_ns_f64(f64::NAN, "s").is_none());
        assert!(epoch_to_ns_f64(f64::INFINITY, "s").is_none());
        assert!(epoch_to_ns_f64(1e30, "s").is_none());
    }

    #[test]
    fn civil_parts_and_format_ns_tz() {
        assert_eq!(civil_parts(0), (1970, 1, 1, 0, 0, 0));
        // format_ns_tz: the UTC fast path matches format_ns; a named zone shifts.
        assert_eq!(format_ns_tz(0, crate::tz::Tz::Utc), format_ns(0));
        let ny = crate::tz::Tz::parse("America/New_York").unwrap();
        assert!(format_ns_tz(0, ny).starts_with("1969")); // UTC epoch is 1969 in NY
    }
}
