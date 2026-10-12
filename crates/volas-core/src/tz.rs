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
        parts(self.wall(ns))
    }

    /// The wall clock of the UTC instant `ns` in this timezone. The offset is
    /// applied in chrono's civil range (far wider than the instant axis), so an
    /// instant at either end of the axis never wraps on its way to the wall clock.
    fn wall(&self, ns: i64) -> NaiveDateTime {
        let utc = DateTime::from_timestamp_nanos(ns);
        match self {
            Tz::Naive | Tz::Utc => utc.naive_utc(),
            Tz::Offset(s) => utc.naive_utc() + chrono::TimeDelta::seconds(*s as i64),
            Tz::Named(tz) => utc.with_timezone(tz).naive_local(),
        }
    }

    /// The UTC instant of the wall clock `wall` in this timezone — the one
    /// wall-clock → instant conversion, keeping the fractional second. A named
    /// zone rejects a DST anomaly: a spring-forward **gap** (the wall clock does
    /// not exist) and a fall-back **fold** (it occurs twice — silently picking one
    /// would shift instants invisibly, F33), both `Value` errors. An instant outside
    /// the datetime64[ns] range is an `Overflow` error.
    pub fn localize(&self, wall: NaiveDateTime) -> Result<i64> {
        match self {
            Tz::Naive | Tz::Utc => crate::datetime::fixed_wall_instant(&wall, 0),
            Tz::Offset(s) => crate::datetime::fixed_wall_instant(&wall, *s),
            Tz::Named(tz) => match tz.from_local_datetime(&wall) {
                chrono::LocalResult::Single(dt) => crate::datetime::instant_of(&dt),
                chrono::LocalResult::Ambiguous(_, _) => Err(VolasError::Value(format!(
                    "wall clock {wall} is ambiguous in {} (a DST fold)",
                    tz.name()
                ))),
                chrono::LocalResult::None => Err(VolasError::Value(format!(
                    "wall clock {wall} does not exist in {} (a DST gap)",
                    tz.name()
                ))),
            },
        }
    }

    /// The instant whose wall clock in this zone equals the wall clock of `ns` in
    /// `from` — pandas `tz_localize` of one value (a naive axis reads its stored
    /// value as the wall clock). Sub-second digits are kept; errors as
    /// [`Tz::localize`].
    pub fn localize_from(&self, ns: i64, from: Tz) -> Result<i64> {
        self.localize(from.wall(ns))
    }

    /// The instant of `ns`'s wall clock in this zone with the given civil
    /// fields `[year, month, day, hour, minute, second]` replaced (`None` keeps
    /// one; the sub-second part is kept) — pandas `Timestamp.replace`. A field
    /// that does not fit, or an invalid date, is a `Value` error; then as
    /// [`Tz::localize`].
    pub fn replace_civil(&self, ns: i64, fields: [Option<i64>; 6]) -> Result<i64> {
        let wall = self.wall(ns);
        let keep = [
            wall.year() as i64,
            wall.month() as i64,
            wall.day() as i64,
            wall.hour() as i64,
            wall.minute() as i64,
            wall.second() as i64,
        ];
        let mut f = [0u32; 6];
        for (i, (new, old)) in fields.iter().zip(keep).enumerate() {
            let v = new.unwrap_or(old);
            // the year is signed (i32); the other fields are unsigned
            let fits = if i == 0 { i32::try_from(v).is_ok() } else { u32::try_from(v).is_ok() };
            if !fits {
                return Err(VolasError::Value(format!("replace: {v} is out of range")));
            }
            f[i] = v as u32;
        }
        let new = NaiveDate::from_ymd_opt(f[0] as i32, f[1], f[2])
            .and_then(|d| d.and_hms_nano_opt(f[3], f[4], f[5], wall.nanosecond()))
            .ok_or_else(|| VolasError::Value("replace produced an invalid wall-clock".into()))?;
        self.localize(new)
    }

    /// UTC epoch-ns of a naive wall-clock time interpreted in this timezone
    /// ([`Tz::localize`] of whole civil fields). Returns `None` if the civil time is
    /// invalid, falls in a DST gap / fold, or lies outside the instant range.
    pub fn wall_to_utc_ns(&self, y: i32, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> Option<i64> {
        let naive = NaiveDate::from_ymd_opt(y, mo, d)?.and_hms_opt(h, mi, s)?;
        self.localize(naive).ok()
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
                        chrono::LocalResult::Single(dt) => return crate::datetime::instant_of(&dt).ok(),
                        chrono::LocalResult::Ambiguous(a, _) => return crate::datetime::instant_of(&a).ok(),
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
    fn localize_keeps_the_fraction_and_names_dst_anomalies() {
        let ny = Tz::parse("America/New_York").unwrap();
        let wall = |s: &str| crate::datetime::parse_ns(s).unwrap();
        // a naive axis value re-read as a NY wall clock keeps its nanoseconds
        let ns = ny.localize_from(wall("2021-01-04 09:30:00.123456789"), Tz::Naive).unwrap();
        assert_eq!(ns, wall("2021-01-04 14:30:00.123456789"));
        let gap = ny.localize_from(wall("2020-03-08 02:30:00"), Tz::Naive).unwrap_err();
        assert!(gap.to_string().contains("DST gap"));
        let fold = ny.localize_from(wall("2020-11-01 01:30:00"), Tz::Naive).unwrap_err();
        assert!(fold.to_string().contains("DST fold"));
    }

    #[test]
    fn replace_civil_checks_every_field() {
        let utc = Tz::Utc;
        let t = crate::datetime::parse_ns("2024-01-31 10:00:00.5").unwrap();
        let none = [None; 6];
        let mut f = none;
        f[3] = Some(11);
        assert_eq!(
            utc.replace_civil(t, f).unwrap(),
            crate::datetime::parse_ns("2024-01-31 11:00:00.5").unwrap()
        );
        let mut f = none;
        f[0] = Some(2024 + (1 << 32));
        assert!(matches!(utc.replace_civil(t, f), Err(VolasError::Value(_))));
        let mut f = none;
        f[1] = Some(-1);
        assert!(matches!(utc.replace_civil(t, f), Err(VolasError::Value(_))));
        let mut f = none;
        f[1] = Some(2); // Feb 31
        assert!(matches!(utc.replace_civil(t, f), Err(VolasError::Value(_))));
        let mut f = none;
        f[0] = Some(2262);
        f[1] = Some(4);
        f[2] = Some(12);
        assert!(matches!(utc.replace_civil(t, f), Err(VolasError::Overflow(_))));
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
