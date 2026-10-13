//! The supported strftime vocabulary, shared by formatting and explicit parsing.

use super::{Civil, NAT};
use crate::{
    error::{Result, VolasError},
    tz::{format_offset, Tz},
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
pub(super) fn strftime(ns: i64, tz: &Tz, fmt: &str) -> Result<String> {
    if ns == NAT {
        return Ok("NaT".into());
    }
    let invalid = || VolasError::Value(format!("invalid strftime format {fmt:?}"));
    let tokens = tokens(fmt).ok_or_else(invalid)?;
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
                    let digits = n.to_string();
                    if let Some(pad) = pad {
                        out.extend(std::iter::repeat_n(pad, width.saturating_sub(digits.len())));
                    }
                    out.push_str(&digits);
                    continue;
                }
                match spec.code {
                    '+' => out.push_str(&format!(
                        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}{}{}",
                        wall.year(),
                        wall.month(),
                        wall.day(),
                        wall.hour(),
                        wall.minute(),
                        wall.second(),
                        super::subsec_suffix(wall.nanosecond()),
                        format_offset(off, ":")
                    )),
                    'a' => out.push_str(&WEEKDAYS[wall.weekday() as usize][..3]),
                    'A' => out.push_str(WEEKDAYS[wall.weekday() as usize]),
                    'b' | 'h' => out.push_str(&MONTHS[wall.month() as usize - 1][..3]),
                    'B' => out.push_str(MONTHS[wall.month() as usize - 1]),
                    'p' => out.push_str(if wall.hour() < 12 { "AM" } else { "PM" }),
                    'P' => out.push_str(if wall.hour() < 12 { "am" } else { "pm" }),
                    'f' => {
                        let n = wall.nanosecond();
                        if spec.dot && spec.digits == 0 {
                            out.push_str(&super::subsec_suffix(n));
                        } else {
                            if spec.dot {
                                out.push('.');
                            }
                            let width = if spec.digits == 0 { 6 } else { spec.digits };
                            out.push_str(&format!("{:0width$}", n / 10u32.pow((9 - width) as u32)));
                        }
                    }
                    'z' if tz.is_aware() => match spec.colons {
                        0 => out.push_str(&format_offset(off, "")),
                        1 => out.push_str(&format_offset(off, ":")),
                        2 => out.push_str(&format!(
                            "{}{:02}:{:02}:{:02}",
                            if off < 0 { '-' } else { '+' },
                            off.unsigned_abs() / 3_600,
                            off.unsigned_abs() / 60 % 60,
                            off.unsigned_abs() % 60
                        )),
                        _ => out.push_str(&format!(
                            "{}{:02}",
                            if off < 0 { '-' } else { '+' },
                            off.unsigned_abs() / 3_600
                        )),
                    },
                    'Z' if tz.is_aware() => out.push_str(&tz.zone_name_at(ns)),
                    'z' | 'Z' => {}
                    _ => unreachable!("the tokenizer admits only supported fields"), // LCOV_EXCL_LINE
                }
            }
        }
    }
    Ok(out)
}
