//! Per-frame timezone for a `DatetimeIndex`.
//!
//! Storage stays **UTC epoch-ns** for every frame (the universal axis that lets
//! crypto / US / HK / A-share frames coexist and align on the absolute instant).
//! A `Tz` only governs **wall-clock <-> instant** conversion — rendering, label
//! matching, and day-bucket alignment in cumulation. Two forms:
//!
//! - [`Tz::Offset`] — a fixed offset east of UTC (DST-free: crypto / A-share /
//!   HK). The hot path is an integer add, so a fixed-offset frame costs ~nothing.
//! - [`Tz::Named`] — a named IANA zone (DST-aware, via `chrono-tz`: US / EU). A
//!   fixed offset would misalign daily buckets across a DST transition, so a real
//!   named zone is required there; the per-bar DST lookup is bounded and isolated
//!   to those frames.
//!
//! Default is [`Tz::Utc`] (backward-compatible).

use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, TimeZone, Timelike};

use crate::error::{Result, VolasError};

/// The timezone attached to a `DatetimeIndex`. Storage is always UTC epoch-ns;
/// this drives wall-clock conversion only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tz {
    /// UTC — no conversion. The default.
    #[default]
    Utc,
    /// A fixed offset east of UTC, in **seconds** (e.g. `28800` for UTC+8). DST-free.
    Offset(i32),
    /// A named IANA zone (DST-aware) resolved through `chrono-tz`.
    Named(chrono_tz::Tz),
}

impl Tz {
    /// Parse a timezone spec: `""` / `"UTC"` / `"Z"` -> [`Tz::Utc`]; a fixed offset
    /// (`"+08:00"`, `"+0800"`, `"+8"`, `"-05:00"`) -> [`Tz::Offset`]; otherwise an
    /// IANA name (`"America/New_York"`, `"Asia/Shanghai"`) -> [`Tz::Named`].
    pub fn parse(s: &str) -> Result<Tz> {
        let s = s.trim();
        if s.is_empty() || s.eq_ignore_ascii_case("utc") || s == "Z" {
            return Ok(Tz::Utc);
        }
        if let Some(off) = parse_offset_seconds(s) {
            return Ok(if off == 0 { Tz::Utc } else { Tz::Offset(off) });
        }
        s.parse::<chrono_tz::Tz>()
            .map(Tz::Named)
            .map_err(|_| VolasError::Value(format!("unknown timezone {s:?}")))
    }

    /// Seconds east of UTC for a fixed zone; `None` for a named (DST) zone, whose
    /// offset depends on the instant.
    pub fn fixed_offset_secs(&self) -> Option<i32> {
        match self {
            Tz::Utc => Some(0),
            Tz::Offset(s) => Some(*s),
            Tz::Named(_) => None,
        }
    }

    /// Wall-clock civil parts `(year, month, day, hour, minute, second)` of a UTC
    /// instant `ns` as seen in this timezone.
    pub fn civil_parts(&self, ns: i64) -> (i64, i64, i64, i64, i64, i64) {
        match self {
            Tz::Utc | Tz::Offset(_) => {
                let off = self.fixed_offset_secs().unwrap_or(0) as i64;
                let local = ns + off * 1_000_000_000;
                let secs = local.div_euclid(1_000_000_000);
                let nsub = local.rem_euclid(1_000_000_000) as u32;
                parts(DateTime::from_timestamp(secs, nsub).unwrap_or_default().naive_utc())
            }
            Tz::Named(tz) => {
                let secs = ns.div_euclid(1_000_000_000);
                let nsub = ns.rem_euclid(1_000_000_000) as u32;
                let dt = DateTime::from_timestamp(secs, nsub)
                    .unwrap_or_default()
                    .with_timezone(tz);
                (
                    dt.year() as i64,
                    dt.month() as i64,
                    dt.day() as i64,
                    dt.hour() as i64,
                    dt.minute() as i64,
                    dt.second() as i64,
                )
            }
        }
    }

    /// UTC epoch-ns of a naive wall-clock time interpreted in this timezone.
    /// Returns `None` if the civil time is invalid or (for a named zone) falls in
    /// a spring-forward gap. An ambiguous fall-back time resolves to the earlier
    /// instant (matching pandas' default).
    pub fn wall_to_utc_ns(&self, y: i32, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> Option<i64> {
        let naive = NaiveDate::from_ymd_opt(y, mo, d)?.and_hms_opt(h, mi, s)?;
        match self {
            Tz::Utc | Tz::Offset(_) => {
                let off = self.fixed_offset_secs().unwrap_or(0) as i64;
                naive
                    .and_utc()
                    .timestamp_nanos_opt()
                    .map(|ns| ns - off * 1_000_000_000)
            }
            Tz::Named(tz) => match tz.from_local_datetime(&naive) {
                chrono::LocalResult::Single(dt) => dt.timestamp_nanos_opt(),
                chrono::LocalResult::Ambiguous(earlier, _) => earlier.timestamp_nanos_opt(),
                chrono::LocalResult::None => None,
            },
        }
    }

    /// A display name: `"UTC"`, a `"+08:00"`-style offset, or the IANA name.
    pub fn name(&self) -> String {
        match self {
            Tz::Utc => "UTC".to_string(),
            Tz::Offset(s) => format_offset(*s),
            Tz::Named(tz) => tz.name().to_string(),
        }
    }

    /// Whether this is anything other than plain UTC (so callers can keep the
    /// zero-cost path when it is).
    pub fn is_utc(&self) -> bool {
        matches!(self, Tz::Utc)
    }
}

fn parts(dt: NaiveDateTime) -> (i64, i64, i64, i64, i64, i64) {
    (
        dt.year() as i64,
        dt.month() as i64,
        dt.day() as i64,
        dt.hour() as i64,
        dt.minute() as i64,
        dt.second() as i64,
    )
}

/// Parse a fixed-offset spec (`+HH:MM`, `+HHMM`, `+HH`, `+H`, sign required) to
/// seconds east of UTC; `None` if it is not an offset.
fn parse_offset_seconds(s: &str) -> Option<i32> {
    let sign = match s.as_bytes().first()? {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let rest = &s[1..];
    let (hh, mm) = if let Some((h, m)) = rest.split_once(':') {
        (h.parse::<i32>().ok()?, m.parse::<i32>().ok()?)
    } else if rest.len() <= 2 {
        (rest.parse::<i32>().ok()?, 0)
    } else if rest.len() == 4 {
        (rest[..2].parse::<i32>().ok()?, rest[2..].parse::<i32>().ok()?)
    } else {
        return None;
    };
    if hh > 23 || mm > 59 {
        return None;
    }
    Some(sign * (hh * 3600 + mm * 60))
}

fn format_offset(secs: i32) -> String {
    let sign = if secs < 0 { '-' } else { '+' };
    let a = secs.abs();
    format!("{}{:02}:{:02}", sign, a / 3600, (a % 3600) / 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_forms() {
        assert_eq!(Tz::parse("").unwrap(), Tz::Utc);
        assert_eq!(Tz::parse("UTC").unwrap(), Tz::Utc);
        assert_eq!(Tz::parse("+00:00").unwrap(), Tz::Utc);
        assert_eq!(Tz::parse("+08:00").unwrap(), Tz::Offset(28800));
        assert_eq!(Tz::parse("+0800").unwrap(), Tz::Offset(28800));
        assert_eq!(Tz::parse("+8").unwrap(), Tz::Offset(28800));
        assert_eq!(Tz::parse("-05:00").unwrap(), Tz::Offset(-18000));
        assert!(matches!(Tz::parse("America/New_York").unwrap(), Tz::Named(_)));
        assert!(Tz::parse("Not/AZone").is_err());
    }

    #[test]
    fn offset_round_trip() {
        // 2020-01-01 08:00:00+08:00 == 2020-01-01 00:00:00 UTC
        let tz = Tz::parse("+08:00").unwrap();
        let utc_ns = tz.wall_to_utc_ns(2020, 1, 1, 8, 0, 0).unwrap();
        assert_eq!(crate::datetime::format_ns(utc_ns), "2020-01-01 00:00:00");
        // and rendering that instant back in +08:00 shows the wall clock
        assert_eq!(tz.civil_parts(utc_ns), (2020, 1, 1, 8, 0, 0));
        assert_eq!(tz.name(), "+08:00");
    }

    #[test]
    fn named_zone_dst() {
        let ny = Tz::parse("America/New_York").unwrap();
        // Winter: EST = UTC-5. 2021-01-01 12:00 NY -> 17:00 UTC.
        let w = ny.wall_to_utc_ns(2021, 1, 1, 12, 0, 0).unwrap();
        assert_eq!(crate::datetime::format_ns(w), "2021-01-01 17:00:00");
        // Summer: EDT = UTC-4. 2021-07-01 12:00 NY -> 16:00 UTC.
        let s = ny.wall_to_utc_ns(2021, 7, 1, 12, 0, 0).unwrap();
        assert_eq!(crate::datetime::format_ns(s), "2021-07-01 16:00:00");
        // round-trip rendering
        assert_eq!(ny.civil_parts(w), (2021, 1, 1, 12, 0, 0));
        assert_eq!(ny.civil_parts(s), (2021, 7, 1, 12, 0, 0));
    }
}
