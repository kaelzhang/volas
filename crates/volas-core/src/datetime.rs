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
    if let Some((y, mo, d, h, mi, se, subsec)) = naive_parts(s) {
        // The fractional second is tz-independent (offsets are whole minutes), so
        // it is added after the wall-clock -> UTC conversion of the whole-second part.
        return tz.wall_to_utc_ns(y, mo, d, h, mi, se)?.checked_add(subsec);
    }
    // A bare time-of-day ("09:30" / "09:30:15") means *today* at that wall-clock
    // (pandas `Timestamp('09:00')` parity, F24) — today as seen in `tz`.
    for fmt in ["%H:%M:%S", "%H:%M"] {
        if let Ok(t) = chrono::NaiveTime::parse_from_str(s, fmt) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_nanos() as i64;
            let (y, mo, d, _, _, _) = tz.civil_parts(now);
            return tz.wall_to_utc_ns(
                y as i32,
                mo as u32,
                d as u32,
                t.hour(),
                t.minute(),
                t.second(),
            );
        }
    }
    None
}

/// The fixed offset (seconds east of UTC) carried by an **offset-aware** string,
/// or `None` for a naive one — so a constructor can keep the zone the user wrote
/// (`'... 09:00:00+08:00'` stays a +08:00-zoned instant, F25).
pub fn offset_suffix_secs(s: &str) -> Option<i32> {
    let s = s.trim();
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(chrono::Offset::fix(dt.offset()).local_minus_utc());
    }
    for fmt in [
        "%Y-%m-%d %H:%M:%S%.f%:z",
        "%Y-%m-%d %H:%M:%S%.f%z",
        "%Y-%m-%dT%H:%M:%S%.f%z",
    ] {
        if let Ok(dt) = DateTime::parse_from_str(s, fmt) {
            return Some(chrono::Offset::fix(dt.offset()).local_minus_utc());
        }
    }
    None
}

/// Parse an **offset-aware** string (RFC3339 / `%z`) to an absolute UTC instant.
fn parse_offset_aware(s: &str) -> Option<i64> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return dt.timestamp_nanos_opt();
    }
    // `%.f` parses an optional `.fraction` (up to ns), so each form covers both
    // the whole-second and the fractional spelling.
    for fmt in [
        "%Y-%m-%d %H:%M:%S%.f%:z",
        "%Y-%m-%d %H:%M:%S%.f%z",
        "%Y-%m-%dT%H:%M:%S%.f%z",
    ] {
        if let Ok(dt) = DateTime::parse_from_str(s, fmt) {
            return dt.timestamp_nanos_opt();
        }
    }
    None
}

/// Parse the supported **naive** forms into civil parts plus the fractional
/// second in nanoseconds (no tz applied yet).
fn naive_parts(s: &str) -> Option<(i32, u32, u32, u32, u32, u32, i64)> {
    // Full time, with an optional fractional second (`%.f` also matches its
    // absence) — the form `format_ns` emits, so text output round-trips at ns
    // precision (D1).
    for fmt in [
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y/%m/%d %H:%M:%S%.f",
    ] {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Some((
                dt.year(),
                dt.month(),
                dt.day(),
                dt.hour(),
                dt.minute(),
                dt.second(),
                dt.nanosecond() as i64,
            ));
        }
    }
    // Minute resolution (`14:30` — the everyday intraday spelling, pandas-parity).
    for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y/%m/%d %H:%M"] {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Some((
                dt.year(),
                dt.month(),
                dt.day(),
                dt.hour(),
                dt.minute(),
                0,
                0,
            ));
        }
    }
    for fmt in ["%Y-%m-%d", "%Y/%m/%d"] {
        if let Ok(d) = NaiveDate::parse_from_str(s, fmt) {
            return Some((d.year(), d.month(), d.day(), 0, 0, 0, 0));
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
/// Wall-clock civil parts of `ns`. Callers that produce user-visible output MUST
/// pre-filter `NaT` (`i64::MIN`) — the string layer renders it as `"NaT"` (see
/// [`format_ns`] / [`strftime`]) since a missing instant has no civil date to put
/// in this `i64` tuple.
pub fn civil_parts(ns: i64) -> (i64, i64, i64, i64, i64, i64) {
    let secs = ns.div_euclid(1_000_000_000);
    let nsub = ns.rem_euclid(1_000_000_000) as u32;
    // An i64 ns count spans ~1677..=2262, well inside chrono's range, so this is
    // always `Some` — the previous `unwrap_or_default()` would have silently
    // returned the 1970 epoch for an out-of-range value instead of failing loud.
    // `NaT` (`i64::MIN`) decodes to a real 1677 civil date here, so callers MUST
    // pre-filter it (the binding now rejects raw NaT at the Timestamp / unify
    // boundary), per the doc above.
    let dt = DateTime::from_timestamp(secs, nsub)
        .expect("an i64-ns timestamp is within chrono's representable range")
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

/// The civil date for a continuous epoch-day count — the inverse of
/// [`days_from_civil`], used to map a multi-day bucket index back to the bucket's
/// first calendar day (the period-start label of Day3 / Week1 bars).
pub fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch is valid");
    // An i64-ns timestamp spans ~1677..=2262 (see `civil_parts`), so any day count
    // derived from one is far inside chrono's range.
    let date = epoch + chrono::Duration::days(days);
    (date.year() as i64, date.month() as i64, date.day() as i64)
}

/// The `.fff` / `.ffffff` / `.fffffffff` fractional-second suffix for a
/// sub-second remainder, empty for a whole second. Digits come in groups of
/// three (ms / µs / ns), pandas-style, so a 123 ms value prints `.123` and a
/// 123 ns value prints `.000000123` — text output never silently drops
/// sub-second precision the storage still has (D1).
fn subsec_suffix(nsub: u32) -> String {
    if nsub == 0 {
        String::new()
    } else if nsub % 1_000_000 == 0 {
        format!(".{:03}", nsub / 1_000_000)
    } else if nsub % 1_000 == 0 {
        format!(".{:06}", nsub / 1_000)
    } else {
        format!(".{nsub:09}")
    }
}

/// Format epoch nanoseconds as `YYYY-MM-DD HH:MM:SS[.fff[fff[fff]]]` (UTC,
/// naive). Sub-second precision is preserved (see [`subsec_suffix`]).
pub fn format_ns(ns: i64) -> String {
    if ns == i64::MIN {
        return "NaT".to_string(); // missing instant — not a real 1677 civil date
    }
    let secs = ns.div_euclid(1_000_000_000);
    let nsub = ns.rem_euclid(1_000_000_000) as u32;
    DateTime::from_timestamp(secs, 0)
        .map(|dt| {
            format!(
                "{}{}",
                dt.naive_utc().format("%Y-%m-%d %H:%M:%S"),
                subsec_suffix(nsub)
            )
        })
        .unwrap_or_default()
}

/// Format epoch nanoseconds as the wall-clock
/// `YYYY-MM-DD HH:MM:SS[.fff[fff[fff]]]` in `tz` (the human/string form; bulk
/// numpy export stays UTC). Sub-second precision is preserved like [`format_ns`].
pub fn format_ns_tz(ns: i64, tz: Tz) -> String {
    if ns == i64::MIN {
        return "NaT".to_string();
    }
    if tz.is_utc() {
        return format_ns(ns);
    }
    let (y, mo, d, h, mi, s) = tz.civil_parts(ns);
    let nsub = ns.rem_euclid(1_000_000_000) as u32;
    format!(
        "{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02}{}",
        subsec_suffix(nsub)
    )
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
    // Defensive NaT guard (the civil_parts contract): currently unreachable from the
    // binding because `volas.Timestamp` cannot be constructed as NaT.
    if ns == i64::MIN {
        return Some("NaT".to_string()); // missing instant -> NaT // LCOV_EXCL_LINE
    }
    let (y, mo, d, h, mi, s) = civil_parts_tz(ns, tz);
    let dt = NaiveDate::from_ymd_opt(y as i32, mo as u32, d as u32)?
        .and_hms_opt(h as u32, mi as u32, s as u32)?;
    let items: Vec<Item> = StrftimeItems::new(fmt).collect();
    if items.iter().any(|it| matches!(it, Item::Error)) {
        return None;
    }
    Some(dt.format_with_items(items.iter()).to_string())
}

/// Parse `s` with an explicit `strptime`-style `fmt` to **UTC** epoch ns (naive,
/// like [`parse_ns`]). Tries a full datetime, then a date-only format (→ midnight).
/// Returns `None` when `s` does not match `fmt`.
pub fn parse_ns_format(s: &str, fmt: &str) -> Option<i64> {
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
        return dt.and_utc().timestamp_nanos_opt();
    }
    NaiveDate::parse_from_str(s, fmt)
        .ok()?
        .and_hms_opt(0, 0, 0)?
        .and_utc()
        .timestamp_nanos_opt()
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
        assert_eq!(
            epoch_to_ns(1_577_836_800, "s").unwrap(),
            1_577_836_800_000_000_000
        );
        assert!(epoch_to_ns(1, "weeks").is_none());
    }

    #[test]
    fn epoch_units_f64_preserves_fraction() {
        // whole seconds match the integer path
        assert_eq!(
            epoch_to_ns_f64(1_577_836_800.0, "s").unwrap(),
            1_577_836_800_000_000_000
        );
        // a fractional second is preserved (0.5 s == 500_000_000 ns)
        assert_eq!(
            epoch_to_ns_f64(1_577_836_800.5, "s").unwrap(),
            1_577_836_800_500_000_000
        );
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

/// The ISO calendar triple `(iso_year, iso_week, iso_weekday)` of a civil date.
pub fn iso_calendar(y: i64, mo: i64, d: i64) -> (i64, i64, i64) {
    let date = NaiveDate::from_ymd_opt(y as i32, mo as u32, d as u32).unwrap_or_default();
    let iw = date.iso_week();
    (
        iw.year() as i64,
        iw.week() as i64,
        date.weekday().number_from_monday() as i64,
    )
}

#[cfg(test)]
mod offset_suffix_tests {
    use super::*;

    #[test]
    fn offset_suffix_forms() {
        assert_eq!(offset_suffix_secs("2021-01-01T09:00:00+08:00"), Some(28800));
        assert_eq!(offset_suffix_secs("2021-01-01 09:00:00+08:00"), Some(28800));
        assert_eq!(offset_suffix_secs("2021-01-01T09:00:00+0800"), Some(28800));
        assert_eq!(offset_suffix_secs("2021-01-01 09:00:00"), None);
    }
}
