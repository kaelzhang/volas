//! Per-frame timezone for a `DatetimeIndex`.
//!
//! Storage stays **UTC epoch-ns** for every frame (the universal axis that lets
//! US / HK / A-share equity frames coexist and align on the absolute instant).
//! A `Tz` only governs **wall-clock <-> instant** conversion — rendering, label
//! matching, and day-bucket alignment in cumulation. Two forms:
//!
//! - [`Tz::Offset`] — a fixed offset east of UTC (DST-free: A-share / HK
//!   equities). The hot path is an integer add, so a fixed-offset frame costs ~nothing.
//! - [`Tz::Named`] — a named IANA zone (DST-aware, via `chrono-tz`: US / EU
//!   equities). A
//!   fixed offset would misalign daily buckets across a DST transition, so a real
//!   named zone is required there; the per-bar DST lookup is bounded and isolated
//!   to those frames.
//!
//! Default is [`Tz::Utc`] (backward-compatible).

use chrono::{DateTime, Datelike, NaiveDate, NaiveDateTime, TimeZone, Timelike};

use crate::error::{Result, VolasError};

/// The timezone attached to a `DatetimeIndex`. Storage is always UTC epoch-ns;
/// this drives wall-clock conversion only.
///
/// `Naive` vs `Utc` (the pandas two-state model): a **naive** axis is an
/// unanchored wall-clock (`.tz` is None; `tz_convert` refuses it), while a
/// **UTC-aware** axis is anchored (`.tz` is `"UTC"`; convertible). Both convert
/// wall-clock <-> instant with zero offset, so the hot path is identical — the
/// distinction is semantic state, not arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tz {
    /// No timezone attached — an unanchored wall-clock (pandas "naive"). The
    /// default for ingested data until `tz_localize` anchors it.
    #[default]
    Naive,
    /// UTC — anchored, zero offset.
    Utc,
    /// A fixed offset east of UTC, in **seconds** (e.g. `28800` for UTC+8). DST-free.
    Offset(i32),
    /// A named IANA zone (DST-aware) resolved through `chrono-tz`.
    Named(chrono_tz::Tz),
}

impl Tz {
    /// Parse a timezone spec: `"UTC"` / `"Z"` -> [`Tz::Utc`]; a fixed offset
    /// (`"+08:00"`, `"+0800"`, `"+8"`, `"-05:00"`) -> [`Tz::Offset`]; otherwise an
    /// IANA name (`"America/New_York"`, `"Asia/Shanghai"`) -> [`Tz::Named`].
    /// An empty spec is an error (F47): "no timezone" is expressed by not calling,
    /// never by a silent empty string.
    pub fn parse(s: &str) -> Result<Tz> {
        let s = s.trim();
        if s.is_empty() {
            return Err(VolasError::Value(
                "empty timezone — pass an IANA name, an offset like \"+08:00\", or \"UTC\"".into(),
            ));
        }
        if s.eq_ignore_ascii_case("utc") || s == "Z" {
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
            Tz::Naive | Tz::Utc => Some(0),
            Tz::Offset(s) => Some(*s),
            Tz::Named(_) => None,
        }
    }

    /// Wall-clock civil parts `(year, month, day, hour, minute, second)` of a UTC
    /// instant `ns` as seen in this timezone.
    pub fn civil_parts(&self, ns: i64) -> (i64, i64, i64, i64, i64, i64) {
        match self {
            Tz::Naive | Tz::Utc | Tz::Offset(_) => {
                let off = self.fixed_offset_secs().unwrap_or(0) as i64;
                let local = ns + off * 1_000_000_000;
                let secs = local.div_euclid(1_000_000_000);
                let nsub = local.rem_euclid(1_000_000_000) as u32;
                parts(
                    DateTime::from_timestamp(secs, nsub)
                        .unwrap_or_default()
                        .naive_utc(),
                )
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
    /// a DST anomaly: a spring-forward **gap** (the time does not exist) or a
    /// fall-back **fold** (the time occurs twice — ambiguous; silently picking
    /// one would shift instants invisibly, F33, so it is rejected like the gap).
    pub fn wall_to_utc_ns(&self, y: i32, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> Option<i64> {
        let naive = NaiveDate::from_ymd_opt(y, mo, d)?.and_hms_opt(h, mi, s)?;
        match self {
            Tz::Naive | Tz::Utc | Tz::Offset(_) => {
                let off = self.fixed_offset_secs().unwrap_or(0) as i64;
                naive
                    .and_utc()
                    .timestamp_nanos_opt()
                    .map(|ns| ns - off * 1_000_000_000)
            }
            Tz::Named(tz) => match tz.from_local_datetime(&naive) {
                chrono::LocalResult::Single(dt) => dt.timestamp_nanos_opt(),
                chrono::LocalResult::Ambiguous(_, _) => None,
                chrono::LocalResult::None => None,
            },
        }
    }

    /// UTC epoch-ns of a naive wall-clock interpreted in this timezone, resolved
    /// to the **earliest** real instant — the period-start variant of
    /// [`Tz::wall_to_utc_ns`]. Where that method rejects DST anomalies (correct
    /// for user-entered times, F33), a derived bucket start must always resolve:
    /// a fall-back **fold** takes the first of the two occurrences, and a
    /// spring-forward **gap** (the wall-clock minute does not exist) advances in
    /// 15-minute steps to the first instant that does — the true first moment of
    /// the bucket. Returns `None` only for an invalid civil date.
    pub fn wall_to_utc_ns_earliest(
        &self,
        y: i32,
        mo: u32,
        d: u32,
        h: u32,
        mi: u32,
        s: u32,
    ) -> Option<i64> {
        match self {
            Tz::Naive | Tz::Utc | Tz::Offset(_) => self.wall_to_utc_ns(y, mo, d, h, mi, s),
            Tz::Named(tz) => {
                let mut naive = NaiveDate::from_ymd_opt(y, mo, d)?.and_hms_opt(h, mi, s)?;
                // Real DST gaps are at most ~2h; 16 quarter-hour probes cover any
                // historical rule with margin.
                for _ in 0..16 {
                    match tz.from_local_datetime(&naive) {
                        chrono::LocalResult::Single(dt) => return dt.timestamp_nanos_opt(),
                        chrono::LocalResult::Ambiguous(a, _) => return a.timestamp_nanos_opt(),
                        chrono::LocalResult::None => naive += chrono::Duration::minutes(15),
                    }
                }
                None // LCOV_EXCL_LINE — no real tz rule has a >4h gap
            }
        }
    }

    /// A display name: `"naive"` (unanchored), `"UTC"`, a `"+08:00"`-style
    /// offset, or the IANA name.
    pub fn name(&self) -> String {
        match self {
            Tz::Naive => "naive".to_string(),
            Tz::Utc => "UTC".to_string(),
            Tz::Offset(s) => format_offset(*s),
            Tz::Named(tz) => tz.name().to_string(),
        }
    }

    /// Whether wall-clock == stored instant (zero offset, no DST) — the
    /// zero-cost hot path. True for both the naive and the UTC states.
    pub fn is_utc(&self) -> bool {
        matches!(self, Tz::Naive | Tz::Utc)
    }

    /// Whether the axis is anchored to a zone (pandas "tz-aware"). A naive axis
    /// answers `False`: it can be `tz_localize`d but never `tz_convert`ed.
    pub fn is_aware(&self) -> bool {
        !matches!(self, Tz::Naive)
    }

    /// Seconds east of UTC at the given instant (DST-correct for a named zone).
    pub fn offset_secs_at(&self, ns: i64) -> i32 {
        match self {
            Tz::Naive | Tz::Utc => 0,
            Tz::Offset(s) => *s,
            Tz::Named(tz) => {
                use chrono::Offset;
                let secs = ns.div_euclid(1_000_000_000);
                let nsub = ns.rem_euclid(1_000_000_000) as u32;
                let utc = DateTime::from_timestamp(secs, nsub).unwrap_or_default();
                tz.offset_from_utc_datetime(&utc.naive_utc()).fix().local_minus_utc()
            }
        }
    }

    /// The DST component of the offset at the instant, in seconds (0 for fixed
    /// zones / UTC / naive).
    pub fn dst_secs_at(&self, ns: i64) -> i32 {
        match self {
            Tz::Named(_) => {
                // the DST component = the offset now minus the zone's standard
                // (January) offset for that year — robust without chrono-tz
                // internals, and exact for the IANA zones volas serves.
                let (y, ..) = self.civil_parts(ns);
                let jan = self
                    .wall_to_utc_ns(y as i32, 1, 15, 12, 0, 0)
                    .map(|jns| self.offset_secs_at(jns))
                    .unwrap_or(0);
                self.offset_secs_at(ns) - jan.min(self.offset_secs_at(ns))
            }
            _ => 0,
        }
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
        (
            rest[..2].parse::<i32>().ok()?,
            rest[2..].parse::<i32>().ok()?,
        )
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
        assert!(Tz::parse("").is_err()); // F47: empty spec is an error, never silent
        assert_eq!(Tz::parse("UTC").unwrap(), Tz::Utc);
        assert_eq!(Tz::parse("+00:00").unwrap(), Tz::Utc);
        assert_eq!(Tz::parse("+08:00").unwrap(), Tz::Offset(28800));
        assert_eq!(Tz::parse("+0800").unwrap(), Tz::Offset(28800));
        assert_eq!(Tz::parse("+8").unwrap(), Tz::Offset(28800));
        assert_eq!(Tz::parse("-05:00").unwrap(), Tz::Offset(-18000));
        assert!(matches!(
            Tz::parse("America/New_York").unwrap(),
            Tz::Named(_)
        ));
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

    #[test]
    fn offset_and_dst_at_instant() {
        // fixed offsets: constant offset, zero DST; named zones answer per instant.
        let fixed = Tz::parse("+08:00").unwrap();
        assert_eq!(fixed.offset_secs_at(0), 28800);
        assert_eq!(fixed.dst_secs_at(0), 0);
        assert_eq!(Tz::Utc.offset_secs_at(0), 0);
        let ny = Tz::parse("America/New_York").unwrap();
        let summer = ny.wall_to_utc_ns(2021, 7, 1, 12, 0, 0).unwrap();
        assert_eq!(ny.offset_secs_at(summer), -4 * 3600);   // EDT
        assert_eq!(ny.dst_secs_at(summer), 3600);           // one DST hour
    }

    #[test]
    fn tz_edge_branches() {
        let ny = Tz::parse("America/New_York").unwrap();
        assert_eq!(ny.fixed_offset_secs(), None); // a named zone has no fixed offset
        assert_eq!(Tz::Utc.name(), "UTC");
        // DST fall-back (01:30 on 2020-11-01) is ambiguous -> rejected (F33,
        // symmetric with the gap); the spring-forward gap (02:30 on 2020-03-08)
        // does not exist -> None.
        assert!(ny.wall_to_utc_ns(2020, 11, 1, 1, 30, 0).is_none());
        assert!(ny.wall_to_utc_ns(2020, 3, 8, 2, 30, 0).is_none());
        // the two states: naive is unanchored, UTC is anchored.
        assert!(!Tz::Naive.is_aware());
        assert!(Tz::Utc.is_aware());
        assert_eq!(Tz::Naive.name(), "naive");
        // parse rejects 3-char rests and out-of-range hours.
        assert!(Tz::parse("+123").is_err());
        assert!(Tz::parse("+25:00").is_err());
    }
}
