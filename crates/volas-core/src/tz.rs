//! One owner for time-zone rules: immutable TZif bytes compiled into UTC and
//! local-wall interval indexes. Storage remains UTC epoch-nanoseconds; a zone
//! governs localization, calendar fields, rendering and time-frame boundaries.

use std::sync::Arc;
use std::fmt::Write;
use crate::datetime::{self, Civil};
use crate::error::{Result, VolasError};

mod cursor;
mod database;
mod posix;
mod rules;
mod tzif;
pub use cursor::TzCursor;
pub use rules::ZoneRules;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_rules;
#[cfg(test)]
mod tests_tzif;

/// A wall-clock interpretation of the instant axis. Named zones share immutable
/// rules, so existing values keep their snapshot when the data source changes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Tz {
    /// An unanchored wall clock, eligible for `tz_localize`.
    #[default]
    Naive,
    /// UTC, an anchored zero-offset axis.
    Utc,
    /// A fixed number of seconds east of UTC.
    Offset(i32),
    /// A named IANA zone and its exact TZif snapshot.
    Named(Arc<ZoneRules>),
}

impl Tz {
    /// Datetime columns carry UTC instants; only an index owns a named zone.
    pub const DATETIME_COLUMN: Tz = Tz::Utc;

    /// Resolve a zone for Rust-only use: numeric offset, UTC, or an OS zoneinfo
    /// file (`TZDIR` when set). Each named key keeps its first successful load
    /// for the process; later calls share those rules without reading files.
    /// Python uses its separate interpreter-source registry and passes bytes
    /// to [`Tz::from_tzif`], which does not populate this OS-name registry.
    pub fn parse(s: &str) -> Result<Tz> {
        let s = s.trim();
        if let Some(fixed) = Self::parse_fixed(s) { return fixed; }
        database::load(s)
    }

    /// Parse a fixed-zone specification, returning `None` for a named key.
    pub fn parse_fixed(s: &str) -> Option<Result<Tz>> {
        let s = s.trim();
        if s.is_empty() {
            return Some(Err(VolasError::Value(
                "empty timezone — pass an IANA name, an offset like \"+08:00\", or \"UTC\"".into(),
            )));
        }
        if s.eq_ignore_ascii_case("utc") || s == "Z" { return Some(Ok(Tz::Utc)); }
        if s.starts_with(['+', '-']) {
            return Some(parse_offset_seconds(s).ok_or_else(|| VolasError::Value(format!(
                "invalid UTC offset {s:?} (use ±HH, ±HHMM[SS] or ±HH:MM[:SS], hours < 24)"
            ))).and_then(Self::fixed));
        }
        None
    }

    /// Load caller-supplied TZif v1–v4 bytes. Source and version describe the
    /// adapter's selected data; rules and byte identity belong to this owner.
    /// Each call compiles independent rules; clone the result to share them.
    pub fn from_tzif(key: &str, bytes: Vec<u8>, source: String, version: Option<String>) -> Result<Tz> {
        if !database::valid_key(key) {
            return Err(VolasError::Value(format!("invalid timezone key {key:?}")));
        }
        rules::ZoneRules::load(key, bytes, source, version).map(Tz::Named)
    }

    /// Construct a fixed zone; zero is UTC and offsets must be under a day.
    pub fn fixed(secs: i32) -> Result<Tz> {
        match secs {
            0 => Ok(Tz::Utc),
            s if s.unsigned_abs() < 86_400 => Ok(Tz::Offset(s)),
            s => Err(VolasError::Value(format!("a UTC offset of {s} s is not under a day"))),
        }
    }

    pub fn fixed_offset_secs(&self) -> Option<i32> {
        match self {
            Tz::Naive | Tz::Utc => Some(0),
            Tz::Offset(s) => Some(*s),
            Tz::Named(_) => None,
        }
    }

    /// Reuse an interval position across ordered bulk queries.
    pub fn cursor(&self) -> TzCursor<'_> { TzCursor::new(self) }

    pub fn civil_parts(&self, ns: i64) -> (i64, i64, i64, i64, i64, i64) {
        let c = self.civil(ns);
        (c.year(), c.month(), c.day(), c.hour(), c.minute(), c.second())
    }

    /// Civil arithmetic remains wider than the nanosecond instant axis, so
    /// adding an offset at either axis endpoint cannot wrap.
    pub fn wall(&self, ns: i64) -> Civil {
        Civil::from_epoch_seconds(ns.div_euclid(1_000_000_000) + self.offset_secs_at(ns) as i64,
            ns.rem_euclid(1_000_000_000) as u32)
    }

    /// The unique instant of a wall clock; reject both a gap and a fold.
    pub fn localize(&self, wall: Civil) -> Result<i64> {
        self.localize_with_carry(wall, 0)
    }

    /// Text ingress may normalize a parsed leap second after resolving the
    /// preceding civil second. Apply the carry before the single range check.
    pub(crate) fn localize_with_carry(&self, wall: Civil, carry_ns: i64) -> Result<i64> {
        let wide = wall.epoch_nanoseconds() + carry_ns as i128;
        match self {
            Tz::Named(rules) => self.resolve_local(wide, rules.local(wall.epoch_seconds()), false),
            _ => datetime::instant_wide(wide - self.fixed_offset_secs().unwrap() as i128 * 1_000_000_000),
        }
    }

    fn resolve_local(&self, wall_ns: i128, local: rules::Local, earliest: bool) -> Result<i64> {
        let off = match local {
            rules::Local::Unique(off) => off,
            rules::Local::Fold(off) if earliest => off,
            rules::Local::Gap(at) if earliest => return datetime::instant_wide(at as i128 * 1_000_000_000),
            kind => {
                let wall = Civil::from_epoch_seconds((wall_ns.div_euclid(1_000_000_000)) as i64, wall_ns.rem_euclid(1_000_000_000) as u32);
                let reason = match kind {
                    rules::Local::Fold(_) => "is ambiguous",
                    _ => "does not exist",
                };
                let anomaly = if matches!(kind, rules::Local::Fold(_)) { "fold" } else { "gap" };
                return Err(VolasError::Value(format!("wall clock {wall} {reason} in {} (a DST {anomaly})", self.name())));
            }
        };
        datetime::instant_wide(wall_ns - off as i128 * 1_000_000_000)
    }

    pub fn localize_from(&self, ns: i64, from: &Tz) -> Result<i64> { self.localize(from.wall(ns)) }

    /// Localize an array using interval cursors, keeping fractions and NaT.
    pub fn localize_values(&self, values: &[i64], from: &Tz) -> Result<Vec<i64>> {
        let (mut source, mut target) = (from.cursor(), self.cursor());
        values.iter().map(|&ns| {
            if ns == datetime::NAT { return Ok(ns); }
            let wall = ns as i128 + source.offset_secs_at(ns) as i128 * 1_000_000_000;
            target.localize_wall_ns(wall)
        }).collect()
    }

    pub fn replace_civil(&self, ns: i64, fields: [Option<i64>; 6]) -> Result<i64> {
        let wall = self.wall(ns);
        let keep = [wall.year(), wall.month(), wall.day(), wall.hour(), wall.minute(), wall.second()];
        self.localize_fields(std::array::from_fn(|i| fields[i].unwrap_or(keep[i])), wall.nanosecond())
    }

    pub fn localize_fields(&self, fields: [i64; 6], nanosecond: u32) -> Result<i64> {
        let [y, m, d, h, mi, s] = fields;
        // Keep the public field domain independent of the wider internal
        // calendar; arbitrary giant years are invalid user fields.
        if i32::try_from(y).is_err() {
            return Err(VolasError::Value(format!("civil field {y} is out of range")));
        }
        let wall = Civil::new(y, m, d, h, mi, s, nanosecond)
            .ok_or_else(|| VolasError::Value(format!("{fields:?} is not a valid wall clock")))?;
        self.localize(wall)
    }

    pub fn zone_name_at(&self, ns: i64) -> String {
        let mut out = String::new();
        self.write_zone_name(ns, &mut out);
        out
    }

    /// Write the zone name into an existing text buffer, borrowing named-zone
    /// abbreviations and writing fixed offsets without an intermediate String.
    pub(crate) fn write_zone_name(&self, ns: i64, out: &mut String) {
        match self {
            Tz::Naive => {},
            Tz::Utc => out.push_str("UTC"),
            Tz::Offset(s) => { out.push_str("UTC"); write_offset(*s, ":", out); },
            Tz::Named(rules) => out.push_str(&rules.reading(ns.div_euclid(1_000_000_000)).abbr),
        }
    }

    pub fn wall_to_utc_ns(&self, y: i32, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> Option<i64> {
        self.localize(Civil::new(y as i64, mo as i64, d as i64, h as i64, mi as i64, s as i64, 0)?).ok()
    }

    /// The period a partial date text names in this zone, pandas's partial-string
    /// periods: a year (`2024`), a month (`2024-01`), a day (`2024-01-02`), an hour
    /// (`2024-01-02 10`), a minute (`2024-01-02 10:30`) or a second
    /// (`2024-01-02 10:30:15`), with `T` or a space between the date and the time — as
    /// `[start, end)` instants and how many fields the text gives (1 for a year … 6 for a
    /// second). A day or longer starts like a calendar period ([`Tz::day_start`]); an
    /// hour, a minute or a second names a wall clock, which must be one real instant —
    /// a DST gap or fold is no period. `None` for any other text (an exact label, a
    /// fraction of a second, an offset, a gap / fold wall clock) and for a period outside
    /// the datetime64[ns] range.
    pub fn period_of_text(&self, text: &str) -> Option<(i64, i64, usize)> {
        let text = text.trim();
        let (date, time) = match text.split_once(['T', ' ']) {
            Some((d, t)) => (d, Some(t)),
            None => (text, None),
        };
        let digits = |t: &str, n: usize| -> Option<u32> {
            (t.len() == n && t.bytes().all(|b| b.is_ascii_digit())).then(|| t.parse().ok()).flatten()
        };
        let mut fields: Vec<u32> = Vec::with_capacity(6);
        for (i, part) in date.split('-').enumerate() {
            fields.push(digits(part, if i == 0 { 4 } else { 2 })?);
        }
        if let Some(time) = time {
            if fields.len() != 3 {
                return None;
            }
            for part in time.split(':') {
                fields.push(digits(part, 2)?);
            }
        }
        let given = fields.len();
        if given > 6 {
            return None;
        }
        let at = |i: usize, default: u32| fields.get(i).copied().unwrap_or(default);
        let start = Civil::new(at(0, 0) as i64, at(1, 1) as i64, at(2, 1) as i64,
            at(3, 0) as i64, at(4, 0) as i64, at(5, 0) as i64, 0)?;
        let end = match given {
            1 => Civil::new(start.year() + 1, 1, 1, 0, 0, 0, 0)?,
            2 => Civil::new(start.year() + i64::from(start.month() == 12), start.month() % 12 + 1, 1, 0, 0, 0, 0)?,
            3 => start.checked_add_seconds(86_400)?,
            4 => start.checked_add_seconds(3_600)?,
            5 => start.checked_add_seconds(60)?,
            _ => start.checked_add_seconds(1)?,
        };
        let first = match given {
            1..=3 => self.localize_earliest(start).ok()?,
            _ => self.localize(start).ok()?,
        };
        Some((first, self.localize_earliest(end).ok()?, given))
    }

    /// The first real instant of the local civil day `year-month-day` in this
    /// zone — its midnight, resolved as [`Tz::localize_earliest`] (a midnight a
    /// DST gap skips starts the day where the gap ends). An invalid date is a
    /// `Value` error; a start outside the datetime64[ns] range an `Overflow` error.
    pub fn day_start(&self, year: i64, month: i64, day: i64) -> Result<i64> {
        let midnight = i32::try_from(year).ok().and_then(|_| Civil::new(year, month, day, 0, 0, 0, 0))
            .ok_or_else(|| VolasError::Value(format!("{year:04}-{month:02}-{day:02} is not a valid date")))?;
        self.localize_earliest(midnight)
    }

    fn localize_earliest(&self, wall: Civil) -> Result<i64> {
        match self {
            Tz::Named(rules) => self.resolve_local(wall.epoch_nanoseconds(), rules.local(wall.epoch_seconds()), true),
            _ => self.localize(wall),
        }
    }

    /// The latest real local-grid boundary at or before an instant. Repeated
    /// grid points produce distinct periods; skipped ones resolve to the gap end.
    pub fn floor_local_grid(&self, ns: i64, step_ns: i64) -> Result<i64> {
        self.cursor().floor_local_grid(ns, step_ns)
    }

    pub fn name(&self) -> String {
        match self {
            Tz::Naive => "naive".to_string(),
            Tz::Utc => "UTC".to_string(),
            Tz::Offset(s) => format_offset(*s, ":"),
            Tz::Named(rules) => rules.key().to_string(),
        }
    }

    pub fn is_utc(&self) -> bool { matches!(self, Tz::Naive | Tz::Utc) }
    pub fn is_aware(&self) -> bool { !matches!(self, Tz::Naive) }

    #[inline]
    pub fn offset_secs_at(&self, ns: i64) -> i32 {
        match self {
            Tz::Naive | Tz::Utc => 0,
            Tz::Offset(s) => *s,
            Tz::Named(rules) => rules.reading(ns.div_euclid(1_000_000_000)).offset,
        }
    }

    /// DST seconds derived at load with CPython's zoneinfo algorithm. TZif
    /// stores only total offsets and a DST flag, not a numeric SAVE field.
    pub fn dst_secs_at(&self, ns: i64) -> i32 {
        match self {
            Tz::Named(rules) => rules.reading(ns.div_euclid(1_000_000_000)).dst,
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
    let mut out = String::new();
    write_offset(secs, sep, &mut out);
    out
}

fn write_offset(secs: i32, sep: &str, out: &mut String) {
    let sign = if secs < 0 { '-' } else { '+' };
    let a = secs.unsigned_abs();
    write!(out, "{sign}{:02}{sep}{:02}", a / 3600, a % 3600 / 60).unwrap();
    if !a.is_multiple_of(60) {
        write!(out, "{sep}{:02}", a % 60).unwrap();
    }
}
