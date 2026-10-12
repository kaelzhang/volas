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
    /// The zone of every datetime **column**. A column stores absolute instants
    /// (UTC epoch-ns) and no zone of its own — only a `DatetimeIndex` carries one —
    /// so wherever a column's values leave volas with a zone slot (pandas, Arrow,
    /// `Timestamp` cells, `.dt`), that zone is UTC: the instant is never ambiguous,
    /// and a source zone is restored by converting the instants.
    pub const DATETIME_COLUMN: Tz = Tz::Utc;

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
        if s.starts_with(['+', '-']) {
            let off = parse_offset_seconds(s).ok_or_else(|| {
                VolasError::Value(format!(
                    "invalid UTC offset {s:?} (use ±HH, ±HHMM[SS] or ±HH:MM[:SS], hours < 24)"
                ))
            })?;
            return Tz::fixed(off);
        }
        s.parse::<chrono_tz::Tz>()
            .map(Tz::Named)
            .map_err(|_| VolasError::Value(format!("unknown timezone {s:?}")))
    }

    /// The fixed zone `secs` east of UTC — the one constructor of [`Tz::Offset`]:
    /// a zero offset is [`Tz::Utc`], and an offset of a day or more is a `Value`
    /// error (no real zone has one).
    pub fn fixed(secs: i32) -> Result<Tz> {
        match secs {
            0 => Ok(Tz::Utc),
            s if s.abs() < 86_400 => Ok(Tz::Offset(s)),
            s => Err(VolasError::Value(format!(
                "a UTC offset of {s} s is not under a day"
            ))),
        }
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
        let c = self.civil(ns);
        (c.year(), c.month(), c.day(), c.hour(), c.minute(), c.second())
    }

    /// The wall clock of the UTC instant `ns` in this timezone. The offset is
    /// applied in chrono's civil range (far wider than the instant axis), so an
    /// instant at either end of the axis never wraps on its way to the wall clock.
    pub(crate) fn wall(&self, ns: i64) -> NaiveDateTime {
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
        let merged: [i64; 6] = std::array::from_fn(|i| fields[i].unwrap_or(keep[i]));
        self.localize_fields(merged, wall.nanosecond())
    }

    /// The instant of the wall clock given field by field — `[year, month, day,
    /// hour, minute, second]` plus the nanosecond of the second (a Python
    /// `datetime`'s components) — in this zone. A field that does not fit or an
    /// invalid date is a `Value` error; then as [`Tz::localize`].
    pub fn localize_fields(&self, fields: [i64; 6], nanosecond: u32) -> Result<i64> {
        let bad = |v: i64| VolasError::Value(format!("civil field {v} is out of range"));
        let year = i32::try_from(fields[0]).map_err(|_| bad(fields[0]))?;
        let mut f = [0u32; 5];
        for (slot, &v) in f.iter_mut().zip(&fields[1..]) {
            *slot = u32::try_from(v).map_err(|_| bad(v))?;
        }
        let wall = NaiveDate::from_ymd_opt(year, f[0], f[1])
            .and_then(|d| d.and_hms_nano_opt(f[2], f[3], f[4], nanosecond))
            .ok_or_else(|| VolasError::Value(format!("{fields:?} is not a valid wall clock")))?;
        self.localize(wall)
    }

    /// The zone's name for the instant `ns`, as `strftime('%Z')` prints it:
    /// `"UTC"`, `"UTC±HH:MM[:SS]"` for a fixed offset, the tzdata abbreviation of a
    /// named zone at that instant (`"EST"` / `"EDT"`; a numeric one such as `"-03"`
    /// when tzdata has no letters), and `""` for a naive axis.
    pub fn zone_name_at(&self, ns: i64) -> String {
        match self {
            Tz::Naive => String::new(),
            Tz::Utc => "UTC".to_string(),
            Tz::Offset(s) => format!("UTC{}", format_offset(*s, ":")),
            Tz::Named(tz) => {
                use chrono_tz::OffsetName;
                let utc = DateTime::from_timestamp_nanos(ns);
                match tz.offset_from_utc_datetime(&utc.naive_utc()).abbreviation() {
                    Some(name) => name.to_string(),
                    None => {
                        let off = self.offset_secs_at(ns);
                        let text = format_offset(off, "");
                        // tzdata's numeric names drop a zero minute field (`-03`)
                        if off % 3600 == 0 { text[..3].to_string() } else { text }
                    }
                }
            }
        }
    }

    /// UTC epoch-ns of a naive wall-clock time interpreted in this timezone
    /// ([`Tz::localize`] of whole civil fields). Returns `None` if the civil time is
    /// invalid, falls in a DST gap / fold, or lies outside the instant range.
    pub fn wall_to_utc_ns(&self, y: i32, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> Option<i64> {
        let naive = NaiveDate::from_ymd_opt(y, mo, d)?.and_hms_opt(h, mi, s)?;
        self.localize(naive).ok()
    }

    /// The first real instant of the local civil day `year-month-day` in this
    /// zone — its midnight, resolved as [`Tz::localize_earliest`] (a midnight a
    /// DST gap skips starts the day where the gap ends). An invalid date is a
    /// `Value` error; a start outside the datetime64[ns] range an `Overflow` error.
    pub fn day_start(&self, year: i64, month: i64, day: i64) -> Result<i64> {
        let midnight = i32::try_from(year)
            .ok()
            .zip(u32::try_from(month).ok().zip(u32::try_from(day).ok()))
            .and_then(|(y, (m, d))| NaiveDate::from_ymd_opt(y, m, d))
            .ok_or_else(|| {
                VolasError::Value(format!("{year:04}-{month:02}-{day:02} is not a valid date"))
            })?
            .and_time(chrono::NaiveTime::MIN);
        self.localize_earliest(midnight)
    }

    /// The **earliest** real instant of the wall clock `wall` in this zone — the
    /// calendar-period-start variant of [`Tz::localize`]. Where that method rejects
    /// DST anomalies (correct for user-entered times, F33), a derived period start
    /// must always resolve: a fall-back **fold** takes the first of the two
    /// occurrences, and a spring-forward **gap** (the wall clock does not exist)
    /// resolves to the zone's transition instant ending the gap (from its
    /// transition data, to the second) — the true first moment of the period. An
    /// instant outside the datetime64[ns] range is an `Overflow` error.
    fn localize_earliest(&self, wall: NaiveDateTime) -> Result<i64> {
        let Tz::Named(tz) = self else {
            return self.localize(wall);
        };
        let first = match tz.from_local_datetime(&wall) {
            chrono::LocalResult::Single(dt) | chrono::LocalResult::Ambiguous(dt, _) => dt,
            // inside a gap: the first instant after it, from the zone's transition
            // table (a gap only exists between two known transitions, so it has an end)
            chrono::LocalResult::None => chrono_tz::GapInfo::new(&wall, tz)
                .and_then(|gap| gap.end)
                // LCOV_EXCL_START — unreachable: past the last transition no gap exists
                .ok_or_else(|| {
                    VolasError::Value(format!(
                        "wall clock {wall} lies in a DST gap of {} with no known end",
                        tz.name()
                    ))
                })?,
            // LCOV_EXCL_STOP
        };
        crate::datetime::instant_of(&first)
    }

    /// The start of the **sub-day** period holding the instant `ns` on this zone's
    /// local grid of `step_ns` (a whole divisor of a day, anchored at local
    /// midnight): the latest boundary instant at or before `ns`, where every real
    /// instant whose wall clock is a grid point is a boundary. Periods therefore
    /// follow real instants: a grid wall clock that a fall-back repeats starts a
    /// period at each occurrence (two `01:00` hours, distinct instants), and one
    /// that a spring-forward gap skips is replaced by the instant ending the gap. A
    /// fixed zone is plain arithmetic. A start outside the datetime64[ns] range is
    /// an `Overflow` error.
    pub fn floor_local_grid(&self, ns: i64, step_ns: i64) -> Result<i64> {
        const NS: i128 = 1_000_000_000;
        let step = step_ns as i128;
        // the grid point at or before `at`'s wall clock, read back at offset `off`
        let floor_at = |at: i64, off: i32| -> i128 {
            let off = off as i128 * NS;
            (at as i128 + off).div_euclid(step) * step - off
        };
        let mut at = ns;
        loop {
            let off = self.offset_secs_at(at);
            let start = floor_at(at, off);
            let Tz::Named(_) = self else {
                return crate::datetime::instant_wide(start);
            };
            let start = crate::datetime::instant_wide(start)?;
            // `start` keeps `at`'s offset: it is the real instant of that grid point
            if self.offset_secs_at(start) == off {
                return Ok(start);
            }
            // A transition lies in (start, at]: no grid point is passed after it, so
            // the boundary is the instant ending a gap that skipped the grid point,
            // or else the last boundary before the transition.
            let t = self.transition_into(start, at, off);
            let prev = self.offset_secs_at(t - 1);
            if (start as i128 + off as i128 * NS) >= t as i128 + prev as i128 * NS {
                return Ok(t);
            }
            at = t - 1;
        }
    }

    /// The instant at which this zone's offset becomes `off`, given an instant
    /// `before` under another offset and a later instant `after` under `off`, with
    /// one transition between them (zone transitions fall on whole seconds and are
    /// far more than a day apart).
    fn transition_into(&self, before: i64, after: i64, off: i32) -> i64 {
        const NS: i64 = 1_000_000_000;
        let (mut lo, mut hi) = (before.div_euclid(NS), after.div_euclid(NS));
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if self.offset_secs_at(mid * NS) == off {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi * NS
    }

    /// A display name: `"naive"` (unanchored), `"UTC"`, a `"+08:00"`-style
    /// offset, or the IANA name.
    pub fn name(&self) -> String {
        match self {
            Tz::Naive => "naive".to_string(),
            Tz::Utc => "UTC".to_string(),
            Tz::Offset(s) => format_offset(*s, ":"),
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
                let utc = DateTime::from_timestamp_nanos(ns).naive_utc();
                tz.offset_from_utc_datetime(&utc).fix().local_minus_utc()
            }
        }
    }

    /// The DST component of the offset at the instant, in seconds — the zone's
    /// own transition data (tzdata's `SAVE`), so a southern-hemisphere summer, a
    /// 30-minute DST (Lord Howe) and a negative winter DST (Europe/Dublin) are all
    /// exact. 0 for fixed zones / UTC / naive.
    pub fn dst_secs_at(&self, ns: i64) -> i32 {
        match self {
            Tz::Named(tz) => {
                use chrono_tz::OffsetComponents;
                let utc = DateTime::from_timestamp_nanos(ns).naive_utc();
                tz.offset_from_utc_datetime(&utc).dst_offset().num_seconds() as i32
            }
            _ => 0,
        }
    }
}

/// The one UTC-offset grammar (zone specs and the suffix of an offset-aware
/// string): a required sign, then `H`, `HH`, `HHMM`, `HHMMSS`, `HH:MM` or
/// `HH:MM:SS` in ASCII digits, hours < 24, minutes / seconds < 60. Seconds east of
/// UTC, or `None` for anything else (no inner signs, no other digits).
pub(crate) fn parse_offset_seconds(s: &str) -> Option<i32> {
    let (sign, rest) = match s.as_bytes().first()? {
        b'+' => (1, &s[1..]),
        b'-' => (-1, &s[1..]),
        _ => return None,
    };
    let digits = |f: &str, n: usize| -> Option<i32> {
        (f.len() == n && f.bytes().all(|b| b.is_ascii_digit())).then(|| f.parse().ok())?
    };
    let fields: Vec<&str> = if rest.contains(':') {
        rest.split(':').collect()
    } else if rest.len() == 1 {
        vec![rest]
    } else if rest.is_ascii() && matches!(rest.len(), 2 | 4 | 6) {
        (0..rest.len()).step_by(2).map(|i| &rest[i..i + 2]).collect()
    } else {
        return None;
    };
    let mut parts = [0i32; 3];
    for (i, f) in fields.iter().enumerate() {
        // the hour may be one digit (`+8`, `+8:30`); every other field is two
        let v = if i == 0 && f.len() == 1 { digits(f, 1) } else { digits(f, 2) };
        *parts.get_mut(i)? = v?;
    }
    let [h, m, sec] = parts;
    (h < 24 && m < 60 && sec < 60).then_some(sign * (h * 3600 + m * 60 + sec))
}

/// Render `secs` east of UTC as `±HH<sep>MM`, plus `<sep>SS` when the offset is not
/// a whole minute (a historical LMT offset such as Kolkata's `+05:21:10`) — never
/// truncated, so the text parses back to the same offset.
pub fn format_offset(secs: i32, sep: &str) -> String {
    let sign = if secs < 0 { '-' } else { '+' };
    let a = secs.unsigned_abs();
    let mut out = format!("{sign}{:02}{sep}{:02}", a / 3600, a % 3600 / 60);
    if !a.is_multiple_of(60) {
        out.push_str(&format!("{sep}{:02}", a % 60));
    }
    out
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
        // from the transition data: a southern summer, a 30-minute DST, Dublin's
        // negative winter DST
        let jan = |zone: &str| {
            let tz = Tz::parse(zone).unwrap();
            tz.dst_secs_at(tz.wall_to_utc_ns(2024, 1, 15, 12, 0, 0).unwrap())
        };
        assert_eq!(jan("Australia/Sydney"), 3600);
        assert_eq!(jan("Australia/Lord_Howe"), 1800);
        assert_eq!(jan("Europe/Dublin"), -3600);
    }

    #[test]
    fn period_start_in_a_gap_is_the_transition_instant() {
        let at = |s: &str| crate::datetime::parse_ns(s).unwrap();
        let wall = |s: &str| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").unwrap();
        // Monrovia: -00:44:30 -> UTC at 1972-01-07 00:00 local (a 44:30 gap)
        let monrovia = Tz::parse("Africa/Monrovia").unwrap();
        assert_eq!(monrovia.localize_earliest(wall("1972-01-07 00:00:00")).unwrap(), at("1972-01-07 00:44:30"));
        let ny = Tz::parse("America/New_York").unwrap();
        assert_eq!(ny.localize_earliest(wall("2024-03-10 02:30:00")).unwrap(), at("2024-03-10 07:00:00"));
        // a fold takes the first occurrence; a fixed zone is plain localization
        assert_eq!(ny.localize_earliest(wall("2024-11-03 01:30:00")).unwrap(), at("2024-11-03 05:30:00"));
        assert_eq!(Tz::Utc.localize_earliest(wall("2024-11-03 01:30:00")).unwrap(), at("2024-11-03 01:30:00"));
        // a start before the instant range is an Overflow error, not a panic
        assert!(matches!(Tz::Utc.day_start(1677, 9, 21), Err(VolasError::Overflow(_))));
        // the day start of a gap midnight (São Paulo 2018-11-04 sprang 00:00 -> 01:00)
        let sp = Tz::parse("America/Sao_Paulo").unwrap();
        assert_eq!(sp.day_start(2018, 11, 4).unwrap(), at("2018-11-04 03:00:00"));
        for (y, m, d) in [(2024, 2, 30), (2024, 13, 1), (2024, -1, 1), (1 << 40, 1, 1)] {
            assert!(matches!(Tz::Utc.day_start(y, m, d), Err(VolasError::Value(_))), "{y}-{m}-{d}");
        }
    }

    /// Sub-day periods follow real instants on the local grid: a fall-back
    /// repeated hour starts two periods, a skipped grid point is replaced by the
    /// gap's end, and a fold that is not on the grid stays inside one period.
    #[test]
    fn floor_local_grid_follows_real_instants() {
        let at = |s: &str| crate::datetime::parse_ns(s).unwrap();
        const MIN: i64 = 60_000_000_000;
        let ny = Tz::parse("America/New_York").unwrap();
        // 2026-11-01: 01:00-02:00 repeats (EDT 05:00Z-06:00Z, then EST 06:00Z-07:00Z)
        let h = |s: &str| ny.floor_local_grid(at(s), 60 * MIN).unwrap();
        assert_eq!(h("2026-11-01 04:59:59"), at("2026-11-01 04:00:00")); // 00:00 EDT
        assert_eq!(h("2026-11-01 05:30:00"), at("2026-11-01 05:00:00")); // first 01:00
        assert_eq!(h("2026-11-01 06:30:00"), at("2026-11-01 06:00:00")); // second 01:00
        assert_eq!(h("2026-11-01 07:00:00"), at("2026-11-01 07:00:00")); // 02:00 EST
        // a 2h grid has no point inside the fold: 00:00-02:00 is one 3-hour period
        let h2 = |s: &str| ny.floor_local_grid(at(s), 120 * MIN).unwrap();
        assert_eq!(h2("2026-11-01 06:30:00"), at("2026-11-01 04:00:00"));
        // 2026-03-08: 02:00 EST springs to 03:00 EDT at 07:00Z
        assert_eq!(h("2026-03-08 06:59:59"), at("2026-03-08 06:00:00")); // 01:00 EST
        assert_eq!(h("2026-03-08 07:30:00"), at("2026-03-08 07:00:00")); // 03:00 EDT
        // the 2h grid point 02:00 is skipped: its period starts where the gap ends
        assert_eq!(h2("2026-03-08 07:30:00"), at("2026-03-08 07:00:00"));
        // Lord Howe (30-minute DST). 2025-10-05: 02:00 +10:30 springs to 02:30 +11
        // (15:30Z): the 1h period of 02:00 starts at the gap's end
        let lh = Tz::parse("Australia/Lord_Howe").unwrap();
        let lh1 = |s: &str| lh.floor_local_grid(at(s), 60 * MIN).unwrap();
        assert_eq!(lh1("2025-10-04 15:40:00"), at("2025-10-04 15:30:00"));
        // 2025-04-06: 02:00 +11 falls back to 01:30 +10:30 (15:00Z): 01:00 is not
        // repeated, so the repeated half hour stays in the period of 01:00 +11
        assert_eq!(lh1("2025-04-05 15:10:00"), at("2025-04-05 14:00:00"));
        assert_eq!(lh1("2025-04-05 15:30:00"), at("2025-04-05 15:30:00")); // 02:00 +10:30
        // fixed zones are arithmetic; a start before the range is an Overflow error
        let sh = Tz::parse("+08:00").unwrap();
        assert_eq!(sh.floor_local_grid(at("2024-05-22 03:47:00"), 15 * MIN).unwrap(), at("2024-05-22 03:45:00"));
        assert!(matches!(
            Tz::Utc.floor_local_grid(i64::MIN + 1, 1_000_000_000),
            Err(VolasError::Overflow(_))
        ));
        assert!(matches!(
            ny.floor_local_grid(i64::MIN + 1, 60 * MIN),
            Err(VolasError::Overflow(_))
        ));
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
        assert_eq!(Tz::Naive.fixed_offset_secs(), Some(0));
        assert_eq!(Tz::parse("-05:00").unwrap().fixed_offset_secs(), Some(-18_000));
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
