//! Civil text parsing. Parsing constructs one validated wall clock; applying a
//! numeric or named zone and checking the instant range stay with their owners.

use super::{
    civil,
    format::{self, Spec, Token, MONTHS, WEEKDAYS},
    Civil,
};
use crate::{
    error::{Result, VolasError},
    tz::parse_offset_seconds,
};

/// Text may use leap-second notation. Keep its extra second outside the civil
/// clock so the zone classifies the preceding wall second, then check only the
/// resulting instant's range after the carry has been applied.
pub(super) struct ParsedWall {
    pub civil: Civil,
    pub carry: i64,
}

impl ParsedWall {
    fn new(civil: Civil, leap: bool) -> Self {
        Self {
            civil,
            carry: i64::from(leap) * 1_000_000_000,
        }
    }

    pub fn instant(&self, offset: i32) -> Result<i64> {
        super::instant_wide(
            self.civil.epoch_nanoseconds() + self.carry as i128 - offset as i128 * 1_000_000_000,
        )
    }
}

/// Supported numeric date and time forms without a zone suffix.
pub(super) fn naive_wall(text: &str) -> Option<ParsedWall> {
    let (date, time) = text
        .split_once(['T', ' '])
        .map_or((text, None), |(d, t)| (d, Some(t)));
    let sep = if date.contains('/') { '/' } else { '-' };
    let mut parts = date.rsplitn(3, sep);
    let day = decimal(parts.next()?)?;
    let month = decimal(parts.next()?)?;
    let year = parts.next()?.parse::<i64>().ok()?;
    let (hour, minute, second, nano) = match time {
        Some(t) => time_parts(t.trim_start(), true)?,
        None => (0, 0, 0, 0),
    };
    Some(ParsedWall::new(
        Civil::new(year, month, day, hour, minute, second.min(59), nano)?,
        second == 60,
    ))
}

pub(super) fn time_parts(text: &str, fractions: bool) -> Option<(i64, i64, i64, u32)> {
    let mut parts = text.split(':');
    let h = decimal(parts.next()?)?;
    let m = decimal(parts.next()?)?;
    let (s, nano) = match parts.next() {
        Some(sec) => match sec.split_once('.') {
            Some((sec, frac)) if fractions => (decimal(sec)?, fraction(frac)?),
            _ => (decimal(sec)?, 0),
        },
        None => (0, 0),
    };
    if parts.next().is_some() || h > 23 || m > 59 || s > 60 {
        return None;
    }
    Some((h, m, s, nano))
}

fn decimal(text: &str) -> Option<i64> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

fn fraction(text: &str) -> Option<u32> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n = text
        .bytes()
        .take(9)
        .fold(0u32, |n, b| n * 10 + u32::from(b - b'0'));
    Some(n * 10u32.pow(9 - text.len().min(9) as u32))
}

#[derive(Default)]
struct Fields {
    year: Option<i64>,
    century: Option<i64>,
    year2: Option<i64>,
    month: Option<i64>,
    day: Option<i64>,
    ordinal: Option<i64>,
    quarter: Option<i64>,
    iso_year: Option<i64>,
    iso_year2: Option<i64>,
    iso_week: Option<i64>,
    week_sun: Option<i64>,
    week_mon: Option<i64>,
    weekday: Option<i64>,
    hour: Option<i64>,
    hour12: Option<i64>,
    pm: Option<i64>,
    minute: Option<i64>,
    second: Option<i64>,
    nano: Option<i64>,
    timestamp: Option<i64>,
    offset: Option<i64>,
}

fn put(field: &mut Option<i64>, value: i64) -> Option<()> {
    if field.is_some_and(|old| old != value) {
        return None;
    }
    *field = Some(value);
    Some(())
}

impl Fields {
    fn numeric(&mut self, code: char, n: i64) -> Option<()> {
        let (field, lo, hi) = match code {
            'Y' => (&mut self.year, i64::MIN, i64::MAX),
            'C' => (&mut self.century, 0, 99),
            'y' => (&mut self.year2, 0, 99),
            'G' => (&mut self.iso_year, i64::MIN, i64::MAX),
            'g' => (&mut self.iso_year2, 0, 99),
            'm' => (&mut self.month, 1, 12),
            'd' | 'e' => (&mut self.day, 1, 31),
            'j' => (&mut self.ordinal, 1, 366),
            'q' => (&mut self.quarter, 1, 4),
            'V' => (&mut self.iso_week, 1, 53),
            'U' => (&mut self.week_sun, 0, 53),
            'W' => (&mut self.week_mon, 0, 53),
            'u' => {
                return if (1..=7).contains(&n) {
                    put(&mut self.weekday, n - 1)
                } else {
                    None
                }
            }
            'w' => {
                return if (0..7).contains(&n) {
                    put(&mut self.weekday, (n + 6) % 7)
                } else {
                    None
                }
            }
            'H' | 'k' => (&mut self.hour, 0, 23),
            'I' | 'l' => (&mut self.hour12, 1, 12),
            'M' => (&mut self.minute, 0, 59),
            'S' => (&mut self.second, 0, 60),
            'f' => (&mut self.nano, 0, 999_999_999),
            's' => (&mut self.timestamp, i64::MIN, i64::MAX),
            _ => unreachable!("only numeric format fields reach this owner"), // LCOV_EXCL_LINE
        };
        if !(lo..=hi).contains(&n) {
            return None;
        }
        put(field, n)
    }

    fn year(&self) -> Option<i64> {
        self.year.or_else(|| {
            self.year2.map(|y| {
                self.century
                    .map_or(if y < 70 { 2000 } else { 1900 }, |c| c * 100)
                    + y
            })
        })
    }

    fn wall(&self) -> Option<ParsedWall> {
        // A fraction refines a parsed second; it cannot invent a missing one.
        if self.nano.is_some() && self.second.is_none() && self.timestamp.is_none() {
            return None;
        }
        let nano = self.nano.unwrap_or(0) as u32;
        let from_days = |days: i64| {
            let (y, m, d) = civil::civil_from_days(days);
            Civil::new(y, m, d, 0, 0, 0, 0)
        };
        let date = if let Some(ts) = self.timestamp {
            Civil::from_epoch_seconds(ts.checked_add(self.offset.unwrap_or(0))?, nano)
        } else {
            let date = if let Some((y, (m, d))) = self.year().zip(self.month.zip(self.day)) {
                Civil::new(y, m, d, 0, 0, 0, 0)?
            } else if let Some((y, ordinal)) = self.year().zip(self.ordinal) {
                if ordinal > if civil::is_leap_year(y) { 366 } else { 365 } {
                    return None;
                }
                from_days(civil::checked_days_from_civil(y, 1, 1)?.checked_add(ordinal - 1)?)?
            } else if let Some((y, (week, weekday))) = self
                .iso_year
                .or_else(|| {
                    self.iso_year2
                        .map(|y| if y < 70 { y + 2000 } else { y + 1900 })
                })
                .zip(self.iso_week.zip(self.weekday))
            {
                let jan4 = civil::checked_days_from_civil(y, 1, 4)?;
                from_days(jan4 - (jan4 + 3).rem_euclid(7) + (week - 1) * 7 + weekday)?
            } else if let Some((y, weekday)) = self.year().zip(self.weekday) {
                let jan1 = civil::checked_days_from_civil(y, 1, 1)?;
                let (week, jan_weekday, weekday) = if let Some(week) = self.week_sun {
                    (week, (jan1 + 4).rem_euclid(7), (weekday + 1) % 7)
                } else {
                    (self.week_mon?, (jan1 + 3).rem_euclid(7), weekday)
                };
                from_days(jan1 + (7 - jan_weekday) % 7 + (week - 1) * 7 + weekday)?
            } else {
                return None;
            };
            let hour = if let Some(hour) = self.hour {
                hour
            } else if self.hour12.is_some() || self.pm.is_some() {
                self.hour12? % 12 + self.pm? * 12
            } else {
                0
            };
            date.with_time(
                hour,
                self.minute.unwrap_or(0),
                self.second.unwrap_or(0).min(59),
                nano,
            )?
        };
        self.matches(date)
            .then_some(ParsedWall::new(date, self.second == Some(60)))
    }

    fn matches(&self, c: Civil) -> bool {
        let equal = |field: Option<i64>, value| field.is_none_or(|n| n == value);
        let (iy, iw, _) = c.iso_calendar();
        equal(self.year, c.year())
            && equal(self.century, c.year().div_euclid(100))
            && equal(self.year2, c.year().rem_euclid(100))
            && equal(self.month, c.month())
            && equal(self.day, c.day())
            && equal(self.ordinal, c.day_of_year())
            && equal(self.quarter, c.quarter())
            && equal(self.iso_year, iy)
            && equal(self.iso_year2, iy.rem_euclid(100))
            && equal(self.iso_week, iw)
            && equal(
                self.week_sun,
                (c.day_of_year() + 6 - (c.weekday() + 1) % 7) / 7,
            )
            && equal(self.week_mon, (c.day_of_year() + 6 - c.weekday()) / 7)
            && equal(self.weekday, c.weekday())
            && equal(self.hour, c.hour())
            && equal(self.hour12, (c.hour() + 11) % 12 + 1)
            && equal(self.pm, c.hour() / 12)
            && equal(self.minute, c.minute())
            && equal(self.second.map(|s| s.min(59)), c.second())
    }
}

fn read_number(text: &mut &str, width: usize, signed: bool) -> Option<i64> {
    *text = text.trim_start();
    let sign = usize::from(signed && text.starts_with(['+', '-']));
    let count = text[sign..]
        .bytes()
        .take(width)
        .take_while(u8::is_ascii_digit)
        .count();
    if count == 0 {
        return None;
    }
    let n = text[..sign + count].parse().ok()?;
    *text = &text[sign + count..];
    Some(n)
}

fn read_name(text: &mut &str, names: &[&str], full: bool) -> Option<i64> {
    let (n, matched) = names.iter().enumerate().find_map(|(n, name)| {
        let name = if full { *name } else { &name[..3] };
        text.get(..name.len())
            .filter(|v| v.eq_ignore_ascii_case(name))
            .map(|_| (n, name.len()))
    })?;
    *text = &text[matched..];
    Some(n as i64)
}

fn read_offset(text: &mut &str, spec: Spec) -> Option<i64> {
    if spec.alternate && text.starts_with(['Z', 'z']) {
        *text = &text[1..];
        return Some(0);
    }
    let end = text
        .bytes()
        .enumerate()
        .take_while(|&(i, b)| {
            (i == 0 && matches!(b, b'+' | b'-')) || (i > 0 && (b.is_ascii_digit() || b == b':'))
        })
        .count();
    let offset = parse_offset_seconds(&text[..end])?;
    let digits = text[..end].bytes().filter(u8::is_ascii_digit).count();
    if digits < 4 && !spec.alternate && spec.colons != 3 {
        return None;
    }
    *text = &text[end..];
    Some(offset as i64)
}

/// The established `%+` form: RFC3339 with optional component padding,
/// case-insensitive T/Z/UTC, and whitespace around separators.
fn read_rfc3339(text: &mut &str, p: &mut Fields) -> Option<()> {
    for (code, separator) in [('Y', None), ('m', Some('-')), ('d', Some('-'))] {
        if let Some(separator) = separator {
            *text = text.trim_start().strip_prefix(separator)?;
        }
        let width = if code == 'Y' {
            if text.trim_start().starts_with(['+', '-']) {
                usize::MAX
            } else {
                4
            }
        } else {
            2
        };
        p.numeric(code, read_number(text, width, code == 'Y')?)?;
    }
    *text = text.strip_prefix(['T', 't', ' '])?;
    for (code, separator) in [('H', None), ('M', Some(':')), ('S', Some(':'))] {
        if let Some(separator) = separator {
            *text = text.trim_start().strip_prefix(separator)?;
        }
        p.numeric(code, read_number(text, 2, false)?)?;
    }
    if let Some(rest) = text.strip_prefix('.') {
        let count = rest.bytes().take_while(u8::is_ascii_digit).count();
        put(&mut p.nano, fraction(&rest[..count])? as i64)?;
        *text = &rest[count..];
    }
    *text = text.trim_start();
    if text.get(..3).is_some_and(|s| s.eq_ignore_ascii_case("UTC")) {
        *text = &text[3..];
        put(&mut p.offset, 0)
    } else {
        read_field(
            text,
            p,
            Spec {
                code: 'z',
                pad: None,
                digits: 0,
                dot: false,
                colons: 1,
                alternate: true,
            },
        )
    }
}

fn read_field(text: &mut &str, p: &mut Fields, spec: Spec) -> Option<()> {
    let code = spec.code;
    match code {
        '+' => read_rfc3339(text, p),
        'a' | 'A' => put(&mut p.weekday, read_name(text, &WEEKDAYS, code == 'A')?),
        'b' | 'B' | 'h' => put(&mut p.month, read_name(text, &MONTHS, code == 'B')? + 1),
        'p' | 'P' => put(&mut p.pm, read_name(text, &["AM", "PM"], true)?),
        'z' => put(&mut p.offset, read_offset(text, spec)?),
        'f' if spec.dot || spec.digits != 0 => {
            if spec.dot {
                if !text.starts_with('.') {
                    return Some(());
                }
                *text = &text[1..];
            }
            let max = if spec.digits == 0 {
                usize::MAX
            } else {
                spec.digits
            };
            let count = text
                .bytes()
                .take(max)
                .take_while(u8::is_ascii_digit)
                .count();
            if count == 0 || (spec.digits > 0 && count != max) {
                return None;
            }
            let nano = fraction(&text[..count])?;
            *text = &text[count..];
            put(&mut p.nano, nano as i64)
        }
        _ => {
            let width = match code {
                'Y' | 'G' if !text.trim_start().starts_with(['+', '-']) => 4,
                'Y' | 'G' | 's' => usize::MAX,
                'j' => 3,
                'f' => 9,
                'q' | 'u' | 'w' => 1,
                _ => 2,
            };
            p.numeric(code, read_number(text, width, matches!(code, 'Y' | 'G'))?)
        }
    }
}

/// An explicit format compiled once for a scalar or a batch of text values.
pub struct Strptime<'a> {
    fmt: &'a str,
    tokens: Option<Vec<Token<'a>>>,
    zone_name: bool,
}

impl<'a> Strptime<'a> {
    /// Compile without changing validation timing for an empty or all-missing batch.
    pub fn new(fmt: &'a str) -> Self {
        let tokens = format::tokens(fmt);
        let zone_name = tokens.as_ref().is_some_and(|tokens| tokens.iter().any(|t| matches!(t, Token::Field(Spec { code: 'Z', .. }))));
        Self { fmt, tokens, zone_name }
    }

    /// Parse one value, retaining its text in mismatch and range errors.
    pub fn parse(&self, s: &str) -> Result<i64> {
        let fmt = self.fmt;
        let mismatch = || VolasError::Value(format!("\"{s}\" does not match format \"{fmt}\""));
        let tokens = self.tokens.as_ref().ok_or_else(mismatch)?;
        if self.zone_name {
            return Err(VolasError::Value(format!(
                "format \"{fmt}\": %Z (a time-zone name) is not supported — use a numeric \
                 offset (%z / %:z), or parse without format= and localize"
            )));
        }
        let mut rest = s;
        let mut p = Fields::default();
        for token in tokens {
            match token {
                Token::Literal(text) => rest = rest.strip_prefix(text).ok_or_else(mismatch)?,
                Token::Space(_) => rest = rest.trim_start(),
                Token::Field(spec) => read_field(&mut rest, &mut p, *spec).ok_or_else(mismatch)?,
            }
        }
        if !rest.is_empty() {
            return Err(mismatch());
        }
        let local = p.wall().ok_or_else(mismatch)?;
        local
            .instant(p.offset.unwrap_or(0) as i32)
            .map_err(|_| VolasError::Overflow(format!("\"{s}\" is outside the datetime64[ns] range")))
    }
}
