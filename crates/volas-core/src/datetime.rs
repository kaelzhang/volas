//! Datetime parsing / formatting for the `DatetimeIndex`. Storage is **UTC
//! epoch-ns**; per-frame display/matching tz lives in [`crate::tz`].

use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, TimeZone, Timelike};

use crate::error::{Result, VolasError};
use crate::tz::Tz;

const NS_PER_SEC: i64 = 1_000_000_000;

/// `NaT` — the missing-instant sentinel of a `Datetime` column (`i64::MIN`, NumPy /
/// pandas `iNaT`). It marks a missing value; it is never a constructable instant.
pub const NAT: i64 = i64::MIN;

/// The instant-axis range gate — the ONE rule every instant built from outside data
/// or from arithmetic passes: a datetime64[ns] instant lies in
/// `[i64::MIN + 1, i64::MAX]`. An overflowed computation (`None`) and a result on
/// the [`NAT`] sentinel both raise, so nothing wraps into a far-away instant,
/// saturates to the range edge, or becomes missing silently.
pub fn instant(ns: Option<i64>) -> Result<i64> {
    match ns {
        Some(v) if v != NAT => Ok(v),
        _ => Err(out_of_range()),
    }
}

/// [`instant`] for an exact wide intermediate (civil seconds × 10⁹ + fraction,
/// rounding arithmetic), so no step of the computation can wrap first.
pub fn instant_wide(ns: i128) -> Result<i64> {
    instant(i64::try_from(ns).ok())
}

/// `ns + delta_ns` through the [`instant`] gate (offset / timedelta arithmetic).
pub fn shift(ns: i64, delta_ns: i64) -> Result<i64> {
    instant(ns.checked_add(delta_ns))
}

fn out_of_range() -> VolasError {
    VolasError::Overflow(
        "the instant is outside the datetime64[ns] range \
         (1677-09-21 00:12:43.145224193 .. 2262-04-11 23:47:16.854775807)"
            .into(),
    )
}

/// The span of `mult` × `unit_ns` nanoseconds (a frequency such as `'15min'`),
/// checked: a span past the i64 range raises instead of wrapping.
pub fn span_ns(mult: i64, unit_ns: i64) -> Result<i64> {
    mult.checked_mul(unit_ns).ok_or_else(|| {
        VolasError::Overflow(format!(
            "a frequency of {mult} × {unit_ns} ns exceeds the datetime64[ns] range"
        ))
    })
}

/// How [`round_wall`] picks a grid point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rounding {
    /// The grid point at or before the value.
    Floor,
    /// The grid point at or after the value.
    Ceil,
    /// The nearest grid point; an exact midpoint goes to the even multiple.
    HalfEven,
}

/// Round the instant `ns` to a multiple of `unit_ns` on the wall clock that is
/// `offset_secs` east of UTC, returning the instant of that wall-clock grid point
/// (read back with the same offset). Exact wide arithmetic; a result outside the
/// instant range raises.
pub fn round_wall(ns: i64, offset_secs: i32, unit_ns: i64, mode: Rounding) -> Result<i64> {
    let off = offset_secs as i128 * NS_PER_SEC as i128;
    let wall = ns as i128 + off;
    let unit = unit_ns as i128;
    let floor = wall.div_euclid(unit) * unit;
    let rounded = match mode {
        Rounding::Floor => floor,
        Rounding::Ceil if floor == wall => floor,
        Rounding::Ceil => floor + unit,
        Rounding::HalfEven => {
            let twice = (wall - floor) * 2;
            let odd = (floor / unit) % 2 != 0;
            if twice > unit || (twice == unit && odd) {
                floor + unit
            } else {
                floor
            }
        }
    };
    instant_wide(rounded - off)
}

/// Parse a timestamp string to **UTC** epoch nanoseconds. An **offset-aware**
/// string (RFC3339 `...+HH:MM` / `...Z`) is an absolute instant and converts
/// directly; a **naive** `YYYY-MM-DD[ HH:MM:SS]` string is interpreted as UTC.
pub fn parse_ns(s: &str) -> Result<i64> {
    parse_ns_in_tz(s, Tz::Utc)
}

/// Like [`parse_ns`], but a **naive** string is interpreted in `tz` (then stored
/// as UTC). An **offset-aware** string is already absolute, so `tz` is ignored.
/// A `Value` error for text that is not a datetime (or a wall clock `tz` skips /
/// repeats); an `Overflow` error for an instant outside the datetime64[ns] range.
pub fn parse_ns_in_tz(s: &str, tz: Tz) -> Result<i64> {
    let s = s.trim();
    if let Some(dt) = parse_offset_aware(s) {
        return instant_of(&dt);
    }
    if let Some(wall) = naive_wall(s) {
        return tz.localize(wall);
    }
    // A bare time-of-day ("09:30" / "09:30:15") means *today* at that wall-clock
    // (pandas `Timestamp('09:00')` parity, F24) — today as seen in `tz`.
    for fmt in ["%H:%M:%S", "%H:%M"] {
        if let Ok(t) = chrono::NaiveTime::parse_from_str(s, fmt) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as i64);
            let (y, mo, d, _, _, _) = tz.civil_parts(now);
            let today = NaiveDate::from_ymd_opt(y as i32, mo as u32, d as u32)
                .expect("a decomposed civil date is valid");
            return tz.localize(today.and_time(t));
        }
    }
    Err(VolasError::Value(format!("could not parse datetime {s:?}")))
}

/// The UTC instant of an offset-aware chrono datetime, through the range gate
/// (exact: whole seconds and the fraction are combined without an intermediate
/// that could leave the range first).
pub(crate) fn instant_of<T: TimeZone>(dt: &DateTime<T>) -> Result<i64> {
    fixed_wall_instant(&dt.naive_utc(), 0)
}

/// The instant of the wall clock `wall` read at the fixed offset `offset_secs`
/// east of UTC — exact wide arithmetic, so a wall clock past the range whose
/// instant is inside it (or the reverse) is judged on the instant alone.
pub(crate) fn fixed_wall_instant(wall: &NaiveDateTime, offset_secs: i32) -> Result<i64> {
    let utc = wall.and_utc();
    instant_wide(
        (utc.timestamp() as i128 - offset_secs as i128) * NS_PER_SEC as i128
            + utc.timestamp_subsec_nanos() as i128,
    )
}

/// The fixed offset (seconds east of UTC) carried by an **offset-aware** string,
/// or `None` for a naive one — so a constructor can keep the zone the user wrote
/// (`'... 09:00:00+08:00'` stays a +08:00-zoned instant, F25).
pub fn offset_suffix_secs(s: &str) -> Option<i32> {
    parse_offset_aware(s.trim()).map(|dt| dt.offset().local_minus_utc())
}

/// Parse an **offset-aware** string (RFC3339 / `%z`).
fn parse_offset_aware(s: &str) -> Option<DateTime<chrono::FixedOffset>> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt);
    }
    // `%.f` parses an optional `.fraction` (up to ns), so each form covers both
    // the whole-second and the fractional spelling.
    [
        "%Y-%m-%d %H:%M:%S%.f%:z",
        "%Y-%m-%d %H:%M:%S%.f%z",
        "%Y-%m-%dT%H:%M:%S%.f%z",
    ]
    .iter()
    .find_map(|fmt| DateTime::parse_from_str(s, fmt).ok())
}

/// Parse the supported **naive** forms into a wall-clock datetime (with its
/// fractional second; no tz applied yet).
fn naive_wall(s: &str) -> Option<NaiveDateTime> {
    // Full time, with an optional fractional second (`%.f` also matches its
    // absence) — the form `format_ns` emits, so text output round-trips at ns
    // precision (D1); then minute resolution (`14:30`, the everyday intraday
    // spelling, pandas-parity).
    [
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y/%m/%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y/%m/%d %H:%M",
    ]
    .iter()
    .find_map(|fmt| NaiveDateTime::parse_from_str(s, fmt).ok())
    .or_else(|| {
        ["%Y-%m-%d", "%Y/%m/%d"]
            .iter()
            .find_map(|fmt| NaiveDate::parse_from_str(s, fmt).ok())
            .map(|d| d.and_time(chrono::NaiveTime::MIN))
    })
}

/// A fixed-length time unit (NumPy `datetime64` / `timedelta64` vocabulary) as
/// the rational `(numerator, denominator)` number of nanoseconds it spans.
fn unit_ns(unit: &str) -> Option<(i128, i128)> {
    const S: i128 = NS_PER_SEC as i128;
    Some(match unit {
        "W" => (604_800 * S, 1),
        "D" => (86_400 * S, 1),
        "h" => (3_600 * S, 1),
        "m" => (60 * S, 1),
        "s" => (S, 1),
        "ms" => (1_000_000, 1),
        "us" | "μs" => (1_000, 1),
        "ns" => (1, 1),
        "ps" => (1, 1_000),
        "fs" => (1, 1_000_000),
        "as" => (1, 1_000_000_000),
        _ => return None,
    })
}

/// `value` × `count` fixed `unit`s as an exact nanosecond count (wide): a
/// sub-nanosecond unit must divide exactly — precision is never dropped.
fn fixed_unit_ns(value: i64, unit: &str, count: i64) -> Result<i128> {
    let (num, den) = unit_ns(unit)
        .ok_or_else(|| VolasError::Value(format!("unsupported time unit {unit:?}")))?;
    let scaled = value as i128 * count as i128 * num;
    if scaled % den != 0 {
        return Err(VolasError::Value(format!(
            "{value} × {count}{unit} is not a whole number of nanoseconds"
        )));
    }
    Ok(scaled / den)
}

/// A NumPy `datetime64[count unit]` value (not `NaT`) as an instant: fixed units
/// scale exactly, the calendar units `Y` / `M` count years / months from 1970;
/// the result passes the [`instant`] gate.
pub fn datetime64_to_ns(value: i64, unit: &str, count: i64) -> Result<i64> {
    let months = match unit {
        "Y" => value as i128 * count as i128 * 12,
        "M" => value as i128 * count as i128,
        _ => return instant_wide(fixed_unit_ns(value, unit, count)?),
    };
    let date = i32::try_from(1970 + months.div_euclid(12))
        .ok()
        .and_then(|year| NaiveDate::from_ymd_opt(year, months.rem_euclid(12) as u32 + 1, 1))
        .ok_or_else(out_of_range)?;
    instant_of(&date.and_time(chrono::NaiveTime::MIN).and_utc())
}

/// A NumPy `datetime64[count unit]` array's raw values as a `Datetime` column's
/// instants: `NaT` stays the missing value, every other value passes
/// [`datetime64_to_ns`]. A nanosecond array is a plain copy.
pub fn datetime64_values_to_ns(values: &[i64], unit: &str, count: i64) -> Result<Vec<i64>> {
    if unit == "ns" && count == 1 {
        return Ok(values.to_vec());
    }
    values
        .iter()
        .map(|&v| if v == NAT { Ok(NAT) } else { datetime64_to_ns(v, unit, count) })
        .collect()
}

/// A NumPy `timedelta64[count unit]` value as a nanosecond duration. Its `NaT`
/// and the calendar units (a month or year has no fixed length) raise; a duration
/// past the i64 range raises.
pub fn timedelta64_to_ns(value: i64, unit: &str, count: i64) -> Result<i64> {
    if value == NAT {
        return Err(VolasError::Value(
            "NaT is not a valid timedelta; a missing duration has no arithmetic".into(),
        ));
    }
    if matches!(unit, "Y" | "M") {
        return Err(VolasError::Value(format!(
            "a timedelta64[{unit}] has no fixed length in nanoseconds"
        )));
    }
    instant_wide(fixed_unit_ns(value, unit, count)?)
}

/// The epoch units [`epoch_to_ns`] / [`epoch_to_ns_f64`] accept.
fn epoch_unit(unit: &str) -> Result<&str> {
    match unit {
        "s" | "ms" | "us" | "ns" => Ok(unit),
        _ => Err(VolasError::Value(format!(
            "unknown epoch unit {unit:?} (use \"s\" / \"ms\" / \"us\" / \"ns\")"
        ))),
    }
}

/// Convert an epoch integer with a unit (`"s"` / `"ms"` / `"us"` / `"ns"`) to a
/// UTC instant. The most robust ingestion path for exchange APIs that return a
/// numeric timestamp. An unknown unit is a `Value` error; an instant outside the
/// datetime64[ns] range (or on the `NaT` sentinel) an `Overflow` error.
pub fn epoch_to_ns(value: i64, unit: &str) -> Result<i64> {
    instant_wide(fixed_unit_ns(value, epoch_unit(unit)?, 1)?)
        .map_err(|_| epoch_out_of_range(value, unit))
}

fn epoch_out_of_range(value: impl std::fmt::Display, unit: &str) -> VolasError {
    VolasError::Overflow(format!(
        "epoch {value} {unit} is outside the datetime64[ns] range"
    ))
}

/// Scale a **floating-point** epoch `value` to nanoseconds, rounding to the
/// nearest nanosecond so sub-`unit` fractions are preserved (matching
/// `pandas.to_datetime(..., unit=...)`). An unknown unit or a non-finite value is
/// a `Value` error; an out-of-range result an `Overflow` error.
pub fn epoch_to_ns_f64(value: f64, unit: &str) -> Result<i64> {
    let (scale, _) = unit_ns(epoch_unit(unit)?).expect("an epoch unit is a fixed unit");
    if !value.is_finite() {
        return Err(VolasError::Value(format!("{value} is not a finite epoch")));
    }
    instant(crate::numeric::Numeric::try_from_f64((value * scale as f64).round()))
        .map_err(|_| epoch_out_of_range(value, unit))
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
    } else if nsub.is_multiple_of(1_000_000) {
        format!(".{:03}", nsub / 1_000_000)
    } else if nsub.is_multiple_of(1_000) {
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

/// Parse `s` with an explicit `strptime`-style `fmt` to **UTC** epoch ns, honoring
/// every field the format parses: an offset (`%z` / `%:z` / …) makes the text an
/// absolute instant, exactly like an offset-aware string on the unformatted path
/// ([`parse_ns`]); without one the wall-clock is read as UTC. Time fields the format
/// omits default to zero (`strptime` semantics: a date-only format is midnight, an
/// hour-only one is on the hour). `%Z` (a zone *name*) is rejected rather than
/// silently skipped — chrono cannot resolve one. A `Value` error when `s` does not
/// match `fmt`.
pub fn parse_ns_format(s: &str, fmt: &str) -> Result<i64> {
    use chrono::format::{parse, Fixed, Item, Parsed, StrftimeItems};
    let items: Vec<Item> = StrftimeItems::new(fmt).collect();
    if items.iter().any(|it| matches!(it, Item::Fixed(Fixed::TimezoneName))) {
        return Err(VolasError::Value(format!(
            "format \"{fmt}\": %Z (a time-zone name) is not supported — use a numeric \
             offset (%z / %:z), or parse without format= and localize"
        )));
    }
    let mismatch = || VolasError::Value(format!("\"{s}\" does not match format \"{fmt}\""));
    let mut p = Parsed::new();
    parse(&mut p, s, items.iter()).map_err(|_| mismatch())?;
    if p.timestamp().is_none() {
        let has_hour = p.hour_div_12().is_some() || p.hour_mod_12().is_some();
        if !has_hour {
            p.set_hour(0).map_err(|_| mismatch())?;
        }
        if p.minute().is_none() {
            p.set_minute(0).map_err(|_| mismatch())?;
        }
    }
    let offset = p.offset().unwrap_or(0);
    let local = p.to_naive_datetime_with_offset(offset).map_err(|_| mismatch())?;
    fixed_wall_instant(&local, offset)
        .map_err(|_| VolasError::Overflow(format!("\"{s}\" is outside the datetime64[ns] range")))
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
        assert!(parse_ns("not a date").is_err());
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
    fn parse_ns_format_honors_every_field_or_errors() {
        let utc = |s: &str| parse_ns(&format!("2024-01-01 {s}")).unwrap();
        let fmt = |s, f| parse_ns_format(s, f);
        assert_eq!(fmt("2024-01-01 08:00+0800", "%Y-%m-%d %H:%M%z").unwrap(), utc("00:00"));
        assert_eq!(fmt("2024-01-01 08:00+08:00", "%Y-%m-%d %H:%M%:z").unwrap(), utc("00:00"));
        assert_eq!(fmt("2024-01-01 09", "%Y-%m-%d %H").unwrap(), utc("09:00"));
        assert_eq!(fmt("2024-01-01", "%Y-%m-%d").unwrap(), utc("00:00"));
        assert_eq!(fmt("1704067200", "%s").unwrap(), utc("00:00"));
        let err = |s, f| parse_ns_format(s, f).unwrap_err().to_string();
        assert!(err("2024-01-01 EST", "%Y-%m-%d %Z").contains("%Z"));
        assert!(err("nope", "%Y-%m-%d").contains("does not match"));
        assert_eq!(fmt("2024-01-01 30", "%Y-%m-%d %M").unwrap(), utc("00:30"));
        assert!(err("2300-01-01", "%Y-%m-%d").contains("outside"));
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
        assert!(epoch_to_ns(1, "weeks").is_err());
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
        assert!(epoch_to_ns_f64(1.0, "weeks").is_err());
        assert!(epoch_to_ns_f64(f64::NAN, "s").is_err());
        assert!(epoch_to_ns_f64(f64::INFINITY, "s").is_err());
        assert!(epoch_to_ns_f64(1e30, "s").is_err());
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
mod tests_gate;

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
