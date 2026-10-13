//! Datetime parsing / formatting for the `DatetimeIndex`. Storage is **UTC
//! epoch-ns**; per-frame display/matching tz lives in [`crate::tz`].

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
    parse_ns_in_tz(s, &Tz::Utc)
}

/// Like [`parse_ns`], but a **naive** string is interpreted in `tz` (then stored
/// as UTC). An **offset-aware** string is already absolute, so `tz` is ignored.
/// A `Value` error for text that is not a datetime (or a wall clock `tz` skips /
/// repeats); an `Overflow` error for an instant outside the datetime64[ns] range.
pub fn parse_ns_in_tz(s: &str, tz: &Tz) -> Result<i64> {
    let s = s.trim();
    if let Some((wall, offset)) = parse_offset_aware(s) {
        return wall.instant(offset);
    }
    if let Some(wall) = parse::naive_wall(s) {
        return tz.localize_with_carry(wall.civil, wall.carry);
    }
    // A bare time-of-day ("09:30" / "09:30:15") means *today* at that wall-clock
    // (pandas `Timestamp('09:00')` parity, F24) — today as seen in `tz`.
    if let Some((h, mi, sec, nano)) = parse::time_parts(s, false) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as i64);
        let today = tz.wall(now);
        let wall = today.with_time(h, mi, sec.min(59), nano).expect("a parsed time is valid");
        return tz.localize_with_carry(wall, i64::from(sec == 60) * NS_PER_SEC);
    }
    Err(VolasError::Value(format!("could not parse datetime {s:?}")))
}

/// The instant of the wall clock `wall` read at the fixed offset `offset_secs`
/// east of UTC — exact wide arithmetic, so a wall clock past the range whose
/// instant is inside it (or the reverse) is judged on the instant alone.
pub(crate) fn fixed_wall_instant(wall: &Civil, offset_secs: i32) -> Result<i64> {
    instant_wide(wall.epoch_nanoseconds() - offset_secs as i128 * NS_PER_SEC as i128)
}

/// The fixed offset (seconds east of UTC) carried by an **offset-aware** string,
/// or `None` for a naive one — so a constructor can keep the zone the user wrote
/// (`'... 09:00:00+08:00'` stays a +08:00-zoned instant, F25).
pub fn offset_suffix_secs(s: &str) -> Option<i32> {
    parse_offset_aware(s.trim()).map(|(_, offset)| offset)
}

/// Parse an **offset-aware** string: a wall clock in one of the naive forms, then
/// (after an optional space) `Z` or a UTC offset in the one offset grammar
/// (`±HH`, `±HHMM[SS]`, `±HH:MM[:SS]` — every form [`format_offset`] writes, so a
/// second-precision offset round-trips). The wall clock and the offset in seconds.
///
/// [`format_offset`]: crate::tz::format_offset
fn parse_offset_aware(s: &str) -> Option<(parse::ParsedWall, i32)> {
    if let Some(wall) = s.strip_suffix(['Z', 'z']).and_then(|w| parse::naive_wall(w.trim_end())) {
        return Some((wall, 0));
    }
    // the offset's sign comes after the date's own dashes (`YYYY-MM-DD`)
    let at = s.rfind(['+', '-']).filter(|&i| i > 10)?;
    let wall = parse::naive_wall(s[..at].trim_end())?;
    Some((wall, crate::tz::parse_offset_seconds(&s[at..])?))
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
    let year = i64::try_from(1970 + months.div_euclid(12)).map_err(|_| out_of_range())?;
    let date = Civil::new(year, months.rem_euclid(12) as i64 + 1, 1, 0, 0, 0, 0)
        .ok_or_else(out_of_range)?;
    fixed_wall_instant(&date, 0)
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
    epoch_scale(unit).map(|_| unit)
}

/// The nanoseconds in one epoch `unit` (`"s"` / `"ms"` / `"us"` / `"ns"`).
fn epoch_scale(unit: &str) -> Result<i128> {
    match unit {
        "s" => Ok(NS_PER_SEC as i128),
        "ms" => Ok(1_000_000),
        "us" => Ok(1_000),
        "ns" => Ok(1),
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

/// Scale a **floating-point** epoch `value` to nanoseconds — the ONE float-epoch
/// rule: the float's exact binary value × the unit, rounded to the nearest
/// nanosecond, ties to even. No intermediate product is rounded first, so an
/// exactly representable fraction (`1577836800.25 s`) is exact and a long epoch
/// keeps every nanosecond its float holds (`1704067200.000001 s` → `…954`, as
/// pandas). An unknown unit or a non-finite value is a `Value` error; an
/// out-of-range result an `Overflow` error.
pub fn epoch_to_ns_f64(value: f64, unit: &str) -> Result<i64> {
    let scale = epoch_scale(unit)?;
    if !value.is_finite() {
        return Err(VolasError::Value(format!("{value} is not a finite epoch")));
    }
    // value = ±m · 2^e exactly (IEEE-754 binary64)
    let bits = value.to_bits();
    let field = ((bits >> 52) & 0x7ff) as i32;
    let frac = (bits & ((1 << 52) - 1)) as i128;
    let (m, e) = if field == 0 { (frac, -1074) } else { (frac | 1 << 52, field - 1075) };
    let product = m * scale; // < 2^83
    let magnitude = match e {
        // a normal float this large is ≥ 2^93 ns, far past the range
        e if e > 40 => None,
        e if e >= 0 => Some(product << e),
        e => Some(shr_half_even(product, -e)),
    };
    magnitude
        .map(|v| if bits >> 63 == 1 { -v } else { v })
        .map_or(Err(out_of_range()), instant_wide)
        .map_err(|_| epoch_out_of_range(value, unit))
}

/// Read an epoch written as decimal TEXT (`"1704067200.123456789"`, `"-1.5"`,
/// `"1.7e9"`) in `unit` — exactly, never through a float: the decimal value × the
/// unit, rounded to the nearest nanosecond (ties to even, the float-epoch rule's
/// rounding). Text that is not a finite decimal number (`inf`, `nan`, a date) is a
/// `Value` error; an instant outside the range an `Overflow` error.
pub fn epoch_text_to_ns(text: &str, unit: &str) -> Result<i64> {
    let scale = epoch_scale(unit)?;
    let t = text.trim();
    let bad = || VolasError::Value(format!("could not parse epoch {t:?} (a decimal number)"));
    let (neg, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let (mantissa, exp) = match body.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i32>().map_err(|_| bad())?),
        None => (body, 0),
    };
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
    if (int.is_empty() && frac.is_empty()) || !digits(int) || !digits(frac) {
        return Err(bad());
    }
    // value = D × 10^pow (D the significant digits), then × the unit's ns
    let all = format!("{int}{frac}");
    let mut sig = all.trim_start_matches('0');
    let mut pow = exp as i64 - frac.len() as i64 + scale.ilog10() as i64;
    // past 36 significant digits only the rounding needs the rest (as a sticky bit)
    let sticky = sig.len() > 36 && sig[36..].bytes().any(|b| b != b'0');
    if sig.len() > 36 {
        pow += (sig.len() - 36) as i64;
        sig = &sig[..36];
    }
    let d = sig.bytes().fold(0i128, |acc, b| acc * 10 + (b - b'0') as i128); // ≤ 36 digits
    let magnitude = if d == 0 {
        Some(0)
    } else if pow >= 0 {
        u32::try_from(pow).ok().and_then(|p| 10i128.checked_pow(p)).and_then(|f| d.checked_mul(f))
    } else if pow < -37 {
        Some(0) // below half a nanosecond (d < 10^36)
    } else {
        let f = 10i128.pow((-pow) as u32);
        let (q, r) = (d / f, d % f);
        let half = f / 2;
        Some(if r > half || (r == half && (sticky || q % 2 == 1)) { q + 1 } else { q })
    };
    magnitude
        .map(|v| if neg { -v } else { v })
        .map_or(Err(out_of_range()), instant_wide)
        .map_err(|_| epoch_out_of_range(t, unit))
}

/// `p / 2^k` rounded half to even, for `0 ≤ p < 2^83`.
fn shr_half_even(p: i128, k: i32) -> i128 {
    if k > 84 {
        return 0; // below half a nanosecond
    }
    let q = p >> k;
    let rem = p - (q << k);
    let half = 1i128 << (k - 1);
    if rem > half || (rem == half && q & 1 == 1) { q + 1 } else { q }
}

/// Civil UTC parts `(year, month, day, hour, minute, second)` of the instant `ns`
/// (see [`Tz::civil_parts`]). `NaT` has no civil date: callers that produce
/// user-visible output keep it out — the string layer renders it as `"NaT"` (see
/// [`format_ns`] / [`strftime`]).
pub fn civil_parts(ns: i64) -> (i64, i64, i64, i64, i64, i64) {
    Tz::Utc.civil_parts(ns)
}

/// Continuous count of days from the Unix epoch (`1970-01-01` = 0) for a VALID
/// civil date — the basis for continuous, month-independent week / multi-day
/// buckets. `1970-01-01` is a Thursday, so a Monday-anchored week index is
/// `(days_from_civil(y, mo, d) + 3).div_euclid(7)`. Callers pass the parts of a
/// decomposed instant; an invalid date is a broken invariant and panics, never a
/// silent epoch day.
pub fn days_from_civil(y: i64, mo: i64, d: i64) -> i64 {
    civil::checked_days_from_civil(y, mo, d)
        .expect("days_from_civil takes the parts of a valid civil date")
}

/// The civil date for a continuous epoch-day count — the inverse of
/// [`days_from_civil`], used to map a multi-day bucket index back to the bucket's
/// first calendar day (the period-start label of Day3 / Week1 bars).
pub fn civil_from_days(days: i64) -> (i64, i64, i64) {
    civil::civil_from_days(days)
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
    Civil::from_epoch_ns(ns).to_string()
}

/// Format epoch nanoseconds as the wall-clock
/// `YYYY-MM-DD HH:MM:SS[.fff[fff[fff]]]` in `tz` (the human/string form; bulk
/// numpy export stays UTC). Sub-second precision is preserved like [`format_ns`].
pub fn format_ns_tz(ns: i64, tz: &Tz) -> String {
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
/// instant `ns` in `tz` ([`Tz::civil_parts`]).
pub fn civil_parts_tz(ns: i64, tz: &Tz) -> (i64, i64, i64, i64, i64, i64) {
    tz.civil_parts(ns)
}

/// The wall clock of `ns` in `tz` as `YYYY-MM-DD HH:MM:SS[.f]`, followed by the
/// instant's UTC offset (`±HH:MM[:SS]`) when `tz` is aware — the text form that
/// identifies the instant on its own (a DST-repeated wall clock stays two distinct
/// rows), as pandas writes an aware index to CSV. A naive `tz` has no suffix.
pub fn format_ns_offset(ns: i64, tz: &Tz) -> String {
    let wall = format_ns_tz(ns, tz);
    if ns == NAT || !tz.is_aware() {
        return wall;
    }
    wall + &crate::tz::format_offset(tz.offset_secs_at(ns), ":")
}

/// Format the instant `ns` as its wall clock in `tz` with a Python / pandas
/// `strftime` `fmt`, keeping every field of the instant: `%f` is the 6-digit
/// microsecond (Python's meaning), `%s` the real epoch second, `%z` / `%:z` the
/// offset at the instant (second precision when it has seconds) and `%Z` its zone
/// name — both empty for a naive `tz`, like pandas. An invalid or unformattable
/// format is a `Value` error, never a panic.
pub fn strftime(ns: i64, tz: &Tz, fmt: &str) -> Result<String> {
    Strftime::new(fmt).format(ns, tz)
}

/// Parse `s` with an explicit `strptime`-style `fmt` to **UTC** epoch ns, honoring
/// every field the format parses: an offset (`%z` / `%:z` / …) makes the text an
/// absolute instant, exactly like an offset-aware string on the unformatted path
/// ([`parse_ns`]); without one the wall-clock is read as UTC. Time fields the format
/// omits default to zero (`strptime` semantics: a date-only format is midnight, an
/// hour-only one is on the hour). `%Z` (a zone *name*) is rejected rather than
/// silently skipped — an abbreviation does not identify a zone. A `Value` error when `s` does not
/// match `fmt`.
pub fn parse_ns_format(s: &str, fmt: &str) -> Result<i64> {
    Strptime::new(fmt).parse(s)
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
            format_ns(parse_ns_in_tz("2020-01-01 08:00:00", &tz).unwrap()),
            "2020-01-01 00:00:00"
        );
        // but an offset-aware string ignores the frame tz (already absolute).
        assert_eq!(
            format_ns(parse_ns_in_tz("2020-01-01T00:00:00Z", &tz).unwrap()),
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
        assert_eq!(format_ns_tz(0, &crate::tz::Tz::Utc), format_ns(0));
        let ny = crate::tz::Tz::parse("America/New_York").unwrap();
        assert!(format_ns_tz(0, &ny).starts_with("1969")); // UTC epoch is 1969 in NY
    }
}

pub mod civil;
mod format;
pub use format::Strftime;
pub use parse::Strptime;
mod parse;
pub use civil::Civil;

#[cfg(test)]
mod tests_civil;

#[cfg(test)]
mod tests_gate;

#[cfg(test)]
mod tests_text;

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

#[cfg(test)]
mod tests_format;
