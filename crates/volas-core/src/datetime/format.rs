//! The supported strftime vocabulary, shared by formatting and explicit parsing.

use std::fmt::Write;

use super::{Civil, NAT};
use crate::{
    error::{Result, VolasError},
    tz::Tz,
};

pub(super) const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
pub(super) const WEEKDAYS: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Spec {
    pub code: char,
    pub pad: Option<char>,
    pub digits: usize,
    pub dot: bool,
    pub colons: usize,
    pub alternate: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Token<'a> {
    Literal(&'a str),
    Space(&'a str),
    Field(Spec),
}

pub(super) fn tokens(fmt: &str) -> Option<Vec<Token<'_>>> {
    let mut out = Vec::new();
    tokenize(fmt, &mut out)?;
    Some(out)
}

fn tokenize<'a>(fmt: &'a str, out: &mut Vec<Token<'a>>) -> Option<()> {
    let mut at = 0;
    while at < fmt.len() {
        let ch = fmt[at..].chars().next()?;
        if ch != '%' {
            let start = at;
            let space = ch.is_whitespace();
            while at < fmt.len() {
                let ch = fmt[at..].chars().next()?;
                if ch == '%' || ch.is_whitespace() != space {
                    break;
                }
                at += ch.len_utf8();
            }
            out.push(if space {
                Token::Space(&fmt[start..at])
            } else {
                Token::Literal(&fmt[start..at])
            });
            continue;
        }
        at += 1;
        let mut next = || {
            let ch = fmt.get(at..)?.chars().next()?;
            at += ch.len_utf8();
            Some(ch)
        };
        let mut code = next()?;
        let mut spec = Spec {
            code,
            pad: None,
            digits: 0,
            dot: false,
            colons: 0,
            alternate: false,
        };
        if matches!(code, '-' | '_' | '0') {
            spec.pad = Some(code);
            code = next()?;
        } else if code == '#' {
            spec.alternate = true;
            code = next()?;
        }
        while code == ':' {
            spec.colons += 1;
            code = next()?;
        }
        if code == '.' {
            spec.dot = true;
            code = next()?;
        }
        if matches!(code, '3' | '6' | '9') {
            spec.digits = code.to_digit(10)? as usize;
            code = next()?;
        }
        spec.code = code;
        if (spec.colons > 0 && (code != 'z' || spec.colons > 3))
            || (spec.alternate && code != 'z')
            || ((spec.dot || spec.digits > 0) && code != 'f')
            || (spec.pad.is_some()
                && (!"CYGgydemHkIlMSfjUuVWwqs".contains(code) || spec.dot || spec.digits > 0))
        {
            return None;
        }
        let expanded = match code {
            'D' | 'x' => Some("%m/%d/%y"),
            'F' => Some("%Y-%m-%d"),
            'v' => Some("%e-%b-%Y"),
            'R' => Some("%H:%M"),
            'T' | 'X' => Some("%H:%M:%S"),
            'r' => Some("%I:%M:%S %p"),
            'c' => Some("%a %b %e %H:%M:%S %Y"),
            _ => None,
        };
        if let Some(fmt) = expanded {
            tokenize(fmt, out)?;
        } else {
            out.push(match code {
                '%' => Token::Literal("%"),
                't' => Token::Space("\t"),
                'n' => Token::Space("\n"),
                _ if "AaBbChCYGgydemHkIlMSfjUuVWwqspPzZ+".contains(code) => Token::Field(spec),
                _ => return None,
            });
        }
    }
    Some(())
}

/// The numeric field and its default width/padding.
pub(super) fn number(c: Civil, code: char, seconds: i64) -> Option<(i64, usize, char)> {
    let (n, width) = match code {
        'Y' => (c.year(), 4),
        'G' => (c.iso_calendar().0, 4),
        'C' => (c.year().div_euclid(100), 2),
        'y' => (c.year().rem_euclid(100), 2),
        'g' => (c.iso_calendar().0.rem_euclid(100), 2),
        'm' => (c.month(), 2),
        'd' | 'e' => (c.day(), 2),
        'H' | 'k' => (c.hour(), 2),
        'I' | 'l' => ((c.hour() + 11) % 12 + 1, 2),
        'M' => (c.minute(), 2),
        'S' => (c.second(), 2),
        'j' => (c.day_of_year(), 3),
        'q' => (c.quarter(), 1),
        'w' => ((c.weekday() + 1) % 7, 1),
        'u' => (c.weekday() + 1, 1),
        'V' => (c.iso_calendar().1, 2),
        'U' => ((c.day_of_year() + 6 - (c.weekday() + 1) % 7) / 7, 2),
        'W' => ((c.day_of_year() + 6 - c.weekday()) / 7, 2),
        's' => (seconds, 0),
        _ => return None,
    };
    Some((
        n,
        width,
        if matches!(code, 'e' | 'k' | 'l') {
            ' '
        } else {
            '0'
        },
    ))
}

/// Format one instant's civil fields with Python's six-digit `%f` and its real
/// UTC `%s`; all zone fields come from the value's retained rules.
pub struct Strftime<'a> {
    fmt: &'a str,
    tokens: Option<Vec<Token<'a>>>,
}

impl<'a> Strftime<'a> {
    /// Compile once. Validation stays at the first non-NaT value, as on the scalar path.
    pub fn new(fmt: &'a str) -> Self {
        Self { fmt, tokens: tokens(fmt) }
    }

    /// Format one value with the shared compiled vocabulary.
    pub fn format(&self, ns: i64, tz: &Tz) -> Result<String> {
        let fmt = self.fmt;
        if ns == NAT {
            return Ok("NaT".into());
        }
        let invalid = || VolasError::Value(format!("invalid strftime format {fmt:?}"));
        let tokens = self.tokens.as_ref().ok_or_else(invalid)?;
        let off = tz.offset_secs_at(ns);
        if off.unsigned_abs() >= 86_400 {
            return Err(invalid());
        }
        let wall = tz.wall(ns);
        let mut out = String::with_capacity(fmt.len() + 16);
        for token in tokens {
            match token {
                Token::Literal(s) | Token::Space(s) => out.push_str(s),
                Token::Field(spec) => {
                    if spec.alternate {
                        return Err(invalid());
                    }
                    if let Some((n, width, default_pad)) =
                        number(wall, spec.code, ns.div_euclid(1_000_000_000)).or_else(|| {
                            (spec.code == 'f' && spec.pad.is_some()).then_some((
                                wall.nanosecond() as i64,
                                9,
                                '0',
                            ))
                        })
                    {
                        let pad = match spec.pad {
                            Some('-') => None,
                            Some('_') => Some(' '),
                            Some('0') => Some('0'),
                            _ => Some(default_pad),
                        };
                        push_number(&mut out, n, width, pad);
                        continue;
                    }
                    match spec.code {
                        '+' => {
                            write!(out, "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
                                wall.year(), wall.month(), wall.day(), wall.hour(), wall.minute(), wall.second()).unwrap();
                            push_fraction(&mut out, wall.nanosecond());
                            push_offset(&mut out, off, 1);
                        }
                        'a' => out.push_str(&WEEKDAYS[wall.weekday() as usize][..3]),
                        'A' => out.push_str(WEEKDAYS[wall.weekday() as usize]),
                        'b' | 'h' => out.push_str(&MONTHS[wall.month() as usize - 1][..3]),
                        'B' => out.push_str(MONTHS[wall.month() as usize - 1]),
                        'p' => out.push_str(if wall.hour() < 12 { "AM" } else { "PM" }),
                        'P' => out.push_str(if wall.hour() < 12 { "am" } else { "pm" }),
                        'f' => {
                            let n = wall.nanosecond();
                            if spec.dot && spec.digits == 0 {
                                push_fraction(&mut out, n);
                            } else {
                                if spec.dot {
                                    out.push('.');
                                }
                                let width = if spec.digits == 0 { 6 } else { spec.digits };
                                push_number(&mut out, (n / 10u32.pow((9 - width) as u32)) as i64, width, Some('0'));
                            }
                        }
                        'z' if tz.is_aware() => push_offset(&mut out, off, spec.colons),
                        'Z' if tz.is_aware() => tz.write_zone_name(ns, &mut out),
                        'z' | 'Z' => {}
                        _ => unreachable!("the tokenizer admits only supported fields"), // LCOV_EXCL_LINE
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Decimal scratch includes the sign in its width, keeping the original padding
/// before a negative sign (unlike Rust's sign-aware numeric formatter).
fn push_number(out: &mut String, n: i64, width: usize, pad: Option<char>) {
    let mut scratch = [0u8; 20];
    let mut at = scratch.len();
    let mut value = n.unsigned_abs();
    loop {
        at -= 1;
        scratch[at] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 { break; }
    }
    if n < 0 { at -= 1; scratch[at] = b'-'; }
    if let Some(pad) = pad {
        out.extend(std::iter::repeat_n(pad, width.saturating_sub(scratch.len() - at)));
    }
    out.push_str(std::str::from_utf8(&scratch[at..]).unwrap());
}

fn push_fraction(out: &mut String, n: u32) {
    if n == 0 { return; }
    let width = if n.is_multiple_of(1_000_000) { 3 } else if n.is_multiple_of(1_000) { 6 } else { 9 };
    out.push('.');
    push_number(out, (n / 10u32.pow(9 - width)) as i64, width as usize, Some('0'));
}

fn push_offset(out: &mut String, off: i32, colons: usize) {
    let a = off.unsigned_abs();
    out.push(if off < 0 { '-' } else { '+' });
    push_number(out, (a / 3_600) as i64, 2, Some('0'));
    if colons == 3 { return; }
    if colons > 0 { out.push(':'); }
    push_number(out, (a / 60 % 60) as i64, 2, Some('0'));
    if colons == 2 || !a.is_multiple_of(60) {
        if colons > 0 { out.push(':'); }
        push_number(out, (a % 60) as i64, 2, Some('0'));
    }
}

#[cfg(test)]
mod scratch_tests {
    use super::*;

    #[test]
    fn decimal_scratch_keeps_signed_width_and_extreme_integers() {
        for n in [i64::MIN, -10_000, -100, -1, 0, 1, 99, 10_000, i64::MAX] {
            for width in [0usize, 2, 4, 9] {
                for pad in [None, Some('0'), Some(' ')] {
                    let digits = n.to_string();
                    let mut want = String::new();
                    if let Some(pad) = pad { want.extend(std::iter::repeat_n(pad, width.saturating_sub(digits.len()))); }
                    want.push_str(&digits);
                    let mut got = String::new();
                    push_number(&mut got, n, width, pad);
                    assert_eq!(got, want);
                }
            }
        }
    }
}
