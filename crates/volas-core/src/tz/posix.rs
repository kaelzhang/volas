//! A POSIX TZ rule (RFC 9636 §3.3, the footer of a TZif file): a zone's standard time
//! and, when it observes DST, the yearly rule switching between the two — what a named
//! zone follows past its transition table.

use crate::datetime::{civil::{days_in_month, is_leap_year}, Civil};
use crate::datetime::days_from_civil;

/// One local time type: seconds east of UTC and its abbreviation (a quoted `<+04>` name
/// unquoted).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct TimeType<'a> {
    pub(super) offset: i32,
    pub(super) abbr: &'a str,
}

/// The day of the year a transition falls on.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Day {
    /// `Jn`: day `n` of 1..=365, February 29th never counted.
    Julian(u16),
    /// `n`: zero-based day of the year, February 29th counted.
    Ordinal(u16),
    /// `Mm.w.d`: weekday `d` (0 = Sunday) of week `w` (5 = the last) of month `m`.
    Month(u32, u32, u32),
}

/// A transition date and the local time it happens at, in seconds — measured in the time
/// type in force before it, and anywhere in RFC 9636's −167..=167 hours.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Transition {
    day: Day,
    time: i64,
}

/// A parsed POSIX TZ rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Rule<'a> {
    std: TimeType<'a>,
    /// DST's time type and the transitions into and out of it.
    dst: Option<(TimeType<'a>, Transition, Transition)>,
}

/// What the rule says at an instant: the time type in force and its DST component
/// (DST offset minus standard offset; 0 in standard time).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Reading<'a> {
    pub(super) kind: TimeType<'a>,
    pub(super) dst: i32,
}

impl<'a> Rule<'a> {
    /// Parse a POSIX TZ string as zic writes footers; `None` for anything else.
    pub(super) fn parse(text: &'a str) -> Option<Rule<'a>> {
        let mut p = Parser { s: text, at: 0 };
        let std_abbr = p.abbr()?;
        let std = TimeType { offset: -p.time(24)? as i32, abbr: std_abbr };
        if p.done() {
            return Some(Rule { std, dst: None });
        }
        let dst_abbr = p.abbr()?;
        let dst_offset = if p.peek(b',') { std.offset + 3600 } else { -p.time(24)? as i32 };
        let dst = TimeType { offset: dst_offset, abbr: dst_abbr };
        let start = p.transition()?;
        let end = p.transition()?;
        p.done().then_some(Rule { std, dst: Some((dst, start, end)) })
    }

    /// UTC transitions for one civil year, compiled once at zone load.
    pub(super) fn transitions(&self, year: i64) -> Vec<(i64, Reading<'a>)> {
        let Some((dst, start, end)) = self.dst else { return Vec::new() };
        vec![
            (start.utc(year, self.std.offset), Reading { kind: dst, dst: dst.offset - self.std.offset }),
            (end.utc(year, dst.offset), Reading { kind: self.std, dst: 0 }),
        ]
    }

    /// Initial reading when compiling the rule over a bounded instant axis.
    pub(super) fn reading_at(&self, utc: Civil) -> Reading<'a> {
        let mut result = Reading { kind: self.std, dst: 0 };
        let mut latest = i64::MIN;
        for year in utc.year() - 1..=utc.year() + 1 {
            for (at, reading) in self.transitions(year) {
                if at <= utc.epoch_seconds() && at >= latest {
                    latest = at;
                    result = reading;
                }
            }
        }
        result
    }
}

impl Transition {
    /// Day rules use the native Gregorian calendar, including negative times
    /// and times carried into the following day or year.
    fn utc(&self, year: i64, offset: i32) -> i64 {
        let day = match self.day {
            Day::Julian(n) => days_from_civil(year, 1, 1) + n as i64 - 1
                + i64::from(is_leap_year(year) && n >= 60),
            Day::Ordinal(n) => days_from_civil(year, 1, 1) + n as i64,
            Day::Month(m, w, d) => {
                let first = days_from_civil(year, m as i64, 1);
                let lead = (d as i64 - (first + 4).rem_euclid(7)).rem_euclid(7);
                let mut dom = 1 + lead + 7 * (w as i64 - 1);
                if dom > days_in_month(year, m as i64).expect("validated month") { dom -= 7; }
                first + dom - 1
            }
        };
        day * 86_400 + self.time - offset as i64
    }
}

/// A cursor over a POSIX TZ string.
struct Parser<'a> {
    s: &'a str,
    at: usize,
}

impl<'a> Parser<'a> {
    fn done(&self) -> bool {
        self.at == self.s.len()
    }

    fn peek(&self, b: u8) -> bool {
        self.s.as_bytes().get(self.at) == Some(&b)
    }

    fn eat(&mut self, b: u8) -> bool {
        let hit = self.peek(b);
        self.at += usize::from(hit);
        hit
    }

    /// The run of bytes from here satisfying `f`.
    fn run(&mut self, f: impl Fn(u8) -> bool) -> &'a str {
        let start = self.at;
        while self.s.as_bytes().get(self.at).is_some_and(|&b| f(b)) {
            self.at += 1;
        }
        &self.s[start..self.at]
    }

    /// An abbreviation: three or more letters, or any `<…>`-quoted name.
    fn abbr(&mut self) -> Option<&'a str> {
        if self.eat(b'<') {
            let name = self.run(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-'));
            return (name.len() >= 3 && self.eat(b'>')).then_some(name);
        }
        let name = self.run(|b| b.is_ascii_alphabetic());
        (name.len() >= 3).then_some(name)
    }

    fn number(&mut self) -> Option<i64> {
        self.run(|b| b.is_ascii_digit()).parse().ok()
    }

    /// `[+|-]hh[:mm[:ss]]` in seconds.
    fn time(&mut self, max_hour: i64) -> Option<i64> {
        let sign = if self.eat(b'-') { -1 } else { self.eat(b'+'); 1 };
        let hour = self.number()?;
        if hour > max_hour { return None; }
        let mut secs = hour * 3600;
        for unit in [60, 1] {
            if !self.eat(b':') {
                break;
            }
            let field = self.number()?;
            if field > 59 { return None; }
            secs += field * unit;
        }
        Some(sign * secs)
    }

    /// `,date[/time]`, the time 02:00 when omitted.
    fn transition(&mut self) -> Option<Transition> {
        if !self.eat(b',') {
            return None;
        }
        let day = if self.eat(b'M') {
            let m = self.number()?;
            let w = self.eat(b'.').then(|| self.number()).flatten()?;
            let d = self.eat(b'.').then(|| self.number()).flatten()?;
            ((1..=12).contains(&m) && (1..=5).contains(&w) && d <= 6)
                .then_some(Day::Month(m as u32, w as u32, d as u32))?
        } else if self.eat(b'J') {
            let n = self.number()?;
            (1..=365).contains(&n).then_some(Day::Julian(n as u16))?
        } else {
            let n = self.number()?;
            (n <= 365).then_some(Day::Ordinal(n as u16))?
        };
        let time = if self.eat(b'/') { self.time(167)? } else { 7200 };
        Some(Transition { day, time })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utc(s: &str) -> Civil {
        Civil::from_epoch_ns(crate::datetime::parse_ns(s).unwrap())
    }

    fn offset(rule: &'static str, at: &str) -> (i32, i32) {
        let r = Rule::parse(rule).unwrap().reading_at(utc(at));
        (r.kind.offset, r.dst)
    }

    #[test]
    fn northern_southern_and_negative_dst() {
        let ny = "EST5EDT,M3.2.0,M11.1.0";
        // 2115: second Sunday of March is the 10th, first of November the 3rd
        assert_eq!(offset(ny, "2115-03-10 06:59:59"), (-18000, 0));
        assert_eq!(offset(ny, "2115-03-10 07:00:00"), (-14400, 3600));
        assert_eq!(offset(ny, "2115-11-03 05:59:59"), (-14400, 3600));
        assert_eq!(offset(ny, "2115-11-03 06:00:00"), (-18000, 0));
        // southern: DST across the new year
        let syd = "AEST-10AEDT,M10.1.0,M4.1.0/3";
        assert_eq!(offset(syd, "2150-01-01 00:00:00"), (39600, 3600));
        assert_eq!(offset(syd, "2150-07-01 00:00:00"), (36000, 0));
        // Dublin: standard IST in summer, a negative DST in winter
        let dub = "IST-1GMT0,M10.5.0,M3.5.0/1";
        assert_eq!(offset(dub, "2150-01-15 12:00:00"), (0, -3600));
        assert_eq!(offset(dub, "2150-07-15 12:00:00"), (3600, 0));
        // a numeric name, a default DST offset, minutes and seconds in an offset
        let r = Rule::parse("<+0330>-3:30").unwrap().reading_at(utc("2200-01-01 00:00:00"));
        assert_eq!((r.kind.offset, r.kind.abbr), (12600, "+0330"));
        assert_eq!(offset("AAA-1:02:03BBB,M3.5.0,M10.5.0", "2200-07-01 00:00:00"), (3723 + 3600, 3600));
        let r = Rule::parse("EST5EDT,M3.2.0,M11.1.0").unwrap().reading_at(utc("2200-07-01 00:00:00"));
        assert_eq!(r.kind.abbr, "EDT");
    }

    #[test]
    fn transition_times_past_the_day() {
        // Gaza: the fourth Thursday + 50 h = the Saturday after, 02:00
        let gaza = "EET-2EEST,M3.4.4/50,M10.4.4/50";
        // 2230: fourth Thursday of March is the 25th -> Saturday 27th 02:00 EET (00:00Z)
        assert_eq!(offset(gaza, "2230-03-26 23:59:59"), (7200, 0));
        assert_eq!(offset(gaza, "2230-03-27 00:00:00"), (10800, 3600));
        // Nuuk: a negative transition time (-1 h: 23:00 the day before)
        let nuuk = "<-02>2<-01>,M3.5.0/-1,M10.5.0/0";
        // 2150: last Sunday of March is the 29th; 23:00 -02 on the 28th = 01:00Z on the 29th
        assert_eq!(offset(nuuk, "2150-03-29 00:59:59"), (-7200, 0));
        assert_eq!(offset(nuuk, "2150-03-29 01:00:00"), (-3600, 3600));
    }

    #[test]
    fn julian_and_ordinal_days_and_permanent_dst() {
        // J60 is always March 1st; zero-based 59 is February 29th in a leap year.
        let j = "AAA0BBB,J60/0,J300/0";
        assert_eq!(offset(j, "2104-02-29 23:59:59"), (0, 0));
        assert_eq!(offset(j, "2104-03-01 00:00:00"), (3600, 3600));
        let n = "AAA0BBB,59/0,300/0";
        assert_eq!(offset(n, "2104-02-28 23:59:59"), (0, 0)); // 2104 is leap: day 59 = Feb 29th
        assert_eq!(offset(n, "2104-02-29 00:00:00"), (3600, 3600));
        assert_eq!(offset(n, "2105-02-28 23:59:59"), (0, 0)); // a common year: March 1st
        assert_eq!(offset(n, "2105-03-01 00:00:00"), (3600, 3600));
        // The native calendar also permits civil dates beyond the nanosecond axis.
        let far = Civil::new(3000, 1, 1, 0, 0, 0, 0).unwrap();
        assert_eq!(Rule::parse(j).unwrap().reading_at(far).kind.offset, 0);
        // permanent DST: in at Jan 1st 00:00, out at Dec 31st 25:00 = the next Jan 1st
        let all = "EST5EDT,0/0,J365/25";
        for at in ["2150-01-01 05:00:00", "2150-07-01 00:00:00", "2150-12-31 23:59:59", "2151-01-01 05:00:00"] {
            assert_eq!(offset(all, at), (-14400, 3600), "{at}");
        }
    }

    #[test]
    fn cross_year_rules_keep_the_continuous_annual_recurrence() {
        // Negative times can put next year's start in this December; times
        // above 24 h can put the preceding year's start in this January.
        // The UTC query year must not hide either real transition.
        for (rule, at, before, after) in [
            ("AAA0BBB,J1/-1,J365/23", "2150-12-31 22:00:00", (3600, 3600), (0, 0)),
            ("AAA0BBB,J1/-1,J365/23", "2150-12-31 23:00:00", (0, 0), (3600, 3600)),
            ("AAA-1BBB0,J1/-1,J365/20", "2150-12-31 20:00:00", (0, -3600), (3600, 0)),
            ("AAA-1BBB0,J1/-1,J365/20", "2150-12-31 22:00:00", (3600, 0), (0, -3600)),
            ("AAA0BBB,J365/167,J1/-167", "2150-01-06 23:00:00", (0, 0), (3600, 3600)),
            ("AAA0BBB,J365/167,J1/-167", "2150-12-25 00:00:00", (3600, 3600), (0, 0)),
            ("AAA0BBB1,J365/167,J1/-167", "2150-01-06 23:00:00", (0, 0), (-3600, -3600)),
            ("AAA0BBB1,J365/167,J1/-167", "2150-12-25 02:00:00", (-3600, -3600), (0, 0)),
        ] {
            let parsed = Rule::parse(rule).unwrap();
            let data = crate::tz::tests_tzif::fixture(b'3', &[], &[(0, false, 0)], b"UTC\0", rule);
            let zone = crate::tz::Tz::from_tzif("Test/YearSpill", data, "memory".into(), None).unwrap();
            let boundary = crate::datetime::parse_ns(at).unwrap();
            let mut cursor = zone.cursor();
            for (ns, expected) in [(boundary - 1, before), (boundary, after), (boundary + 1, after)] {
                let reading = parsed.reading_at(Civil::from_epoch_ns(ns));
                assert_eq!((reading.kind.offset, reading.dst), expected, "{rule} at {ns}");
                assert_eq!((zone.offset_secs_at(ns), zone.dst_secs_at(ns)), expected, "{rule} at {ns}");
                assert_eq!(cursor.offset_secs_at(ns), expected.0);
            }
        }
    }

    #[test]
    fn malformed_rules_do_not_parse() {
        for bad in [
            "", "E5", "EST", "EST5EDT", "EST5EDT,M3.2.0", "EST5EDT,M13.2.0,M11.1.0",
            "EST5EDT,M3.6.0,M11.1.0", "EST5EDT,M3.2.7,M11.1.0", "EST5EDT,J0,J365",
            "EST5EDT,366,0", "EST5EDT,M3.2.0,M11.1.0x", "<+04-4", "EST5EDT,M3,M11.1.0",
            "EST5EDT,M3.2,M11.1.0", "EST5EDT,Mx,M11.1.0", "EST5EDT,Jx,J1", "EST5EDT,x,1",
            "EST5EDT,M3.2.0/x,M11.1.0", "EST:", "ESTx",
        ] {
            assert!(Rule::parse(bad).is_none(), "{bad:?}");
        }
    }
}
