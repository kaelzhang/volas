//! A POSIX TZ rule (RFC 8536 §3.3, the footer of a TZif file): a zone's standard time
//! and, when it observes DST, the yearly rule switching between the two — what a named
//! zone follows past its transition table.

use chrono::{Datelike, NaiveDate, NaiveDateTime, TimeDelta};

/// One local time type: seconds east of UTC and its abbreviation (a quoted `<+04>` name
/// unquoted).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct TimeType {
    pub(super) offset: i32,
    pub(super) abbr: &'static str,
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
/// type in force before it, and anywhere in RFC 8536's −167..=167 hours.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Transition {
    day: Day,
    time: i64,
}

/// A parsed POSIX TZ rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Rule {
    std: TimeType,
    /// DST's time type and the transitions into and out of it.
    dst: Option<(TimeType, Transition, Transition)>,
}

/// What the rule says at an instant: the time type in force and its DST component
/// (DST offset minus standard offset; 0 in standard time).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Reading {
    pub(super) kind: TimeType,
    pub(super) dst: i32,
}

impl Rule {
    /// Parse a POSIX TZ string as zic writes footers; `None` for anything else.
    pub(super) fn parse(text: &'static str) -> Option<Rule> {
        let mut p = Parser { s: text, at: 0 };
        let std_abbr = p.abbr()?;
        let std = TimeType { offset: -p.time()? as i32, abbr: std_abbr };
        if p.done() {
            return Some(Rule { std, dst: None });
        }
        let dst_abbr = p.abbr()?;
        let dst_offset = if p.peek(b',') { std.offset + 3600 } else { -p.time()? as i32 };
        let dst = TimeType { offset: dst_offset, abbr: dst_abbr };
        let start = p.transition()?;
        let end = p.transition()?;
        p.done().then_some(Rule { std, dst: Some((dst, start, end)) })
    }

    /// The reading at the UTC wall clock `utc`: the time type of the latest transition at
    /// or before it, among the transitions of its year and the years around it (a
    /// transition's local time can carry it across a year boundary).
    pub(super) fn reading_at(&self, utc: NaiveDateTime) -> Reading {
        let Some((dst, start, end)) = self.dst else {
            return Reading { kind: self.std, dst: 0 };
        };
        let year = utc.year();
        let mut latest: Option<(NaiveDateTime, bool)> = None;
        for y in year - 1..=year + 1 {
            // into DST at standard local time, back out at DST local time
            for (at, into_dst) in [(start.utc(y, self.std.offset), true), (end.utc(y, dst.offset), false)] {
                let Some(at) = at else { continue };
                // at equal instants a later year's entry wins, so permanent DST
                // (`J365/25` meeting the next `0/0`) stays DST
                if at <= utc && latest.is_none_or(|(t, _)| at >= t) {
                    latest = Some((at, into_dst));
                }
            }
        }
        match latest {
            Some((_, true)) => Reading { kind: dst, dst: dst.offset - self.std.offset },
            _ => Reading { kind: self.std, dst: 0 },
        }
    }
}

impl Transition {
    /// The UTC wall clock of this transition in year `y`, its local time read at `offset`;
    /// `None` past chrono's calendar.
    fn utc(&self, y: i32, offset: i32) -> Option<NaiveDateTime> {
        let date = match self.day {
            Day::Julian(n) => {
                let leap = NaiveDate::from_ymd_opt(y, 2, 29).is_some();
                NaiveDate::from_ymd_opt(y, 1, 1)? + TimeDelta::days(n as i64 - 1 + i64::from(leap && n >= 60))
            }
            Day::Ordinal(n) => NaiveDate::from_ymd_opt(y, 1, 1)? + TimeDelta::days(n as i64),
            Day::Month(m, w, d) => {
                let first = NaiveDate::from_ymd_opt(y, m, 1)?;
                let lead = (d + 7 - first.weekday().num_days_from_sunday()) % 7;
                let mut day = first + TimeDelta::days((lead + 7 * (w - 1)) as i64);
                while day.month() != m {
                    day -= TimeDelta::days(7); // week 5: the last such weekday
                }
                day
            }
        };
        date.and_time(chrono::NaiveTime::MIN)
            .checked_add_signed(TimeDelta::seconds(self.time - offset as i64))
    }
}

/// A cursor over a POSIX TZ string.
struct Parser {
    s: &'static str,
    at: usize,
}

impl Parser {
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
    fn run(&mut self, f: impl Fn(u8) -> bool) -> &'static str {
        let start = self.at;
        while self.s.as_bytes().get(self.at).is_some_and(|&b| f(b)) {
            self.at += 1;
        }
        &self.s[start..self.at]
    }

    /// An abbreviation: three or more letters, or any `<…>`-quoted name.
    fn abbr(&mut self) -> Option<&'static str> {
        if self.eat(b'<') {
            let name = self.run(|b| b != b'>');
            return self.eat(b'>').then_some(name);
        }
        let name = self.run(|b| b.is_ascii_alphabetic());
        (name.len() >= 3).then_some(name)
    }

    fn number(&mut self) -> Option<i64> {
        self.run(|b| b.is_ascii_digit()).parse().ok()
    }

    /// `[+|-]hh[:mm[:ss]]` in seconds.
    fn time(&mut self) -> Option<i64> {
        let sign = if self.eat(b'-') { -1 } else { self.eat(b'+'); 1 };
        let mut secs = self.number()? * 3600;
        for unit in [60, 1] {
            if !self.eat(b':') {
                break;
            }
            secs += self.number()? * unit;
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
        let time = if self.eat(b'/') { self.time()? } else { 7200 };
        Some(Transition { day, time })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utc(s: &str) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").unwrap()
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
        // J60 is March 1st in every year; 59 (zero-based) is March 1st only in a leap year
        let j = "AAA0BBB,J60/0,J300/0";
        assert_eq!(offset(j, "2104-02-29 23:59:59"), (0, 0));
        assert_eq!(offset(j, "2104-03-01 00:00:00"), (3600, 3600));
        let n = "AAA0BBB,59/0,300/0";
        assert_eq!(offset(n, "2104-02-28 23:59:59"), (0, 0)); // 2104 is leap: day 59 = Feb 29th
        assert_eq!(offset(n, "2104-02-29 00:00:00"), (3600, 3600));
        assert_eq!(offset(n, "2105-02-28 23:59:59"), (0, 0)); // a common year: March 1st
        assert_eq!(offset(n, "2105-03-01 00:00:00"), (3600, 3600));
        // the transitions of a year past chrono's calendar are skipped, never a panic
        let far = NaiveDate::MAX.and_hms_opt(0, 0, 0).unwrap();
        assert_eq!(Rule::parse(j).unwrap().reading_at(far).kind.offset, 0);
        // permanent DST: in at Jan 1st 00:00, out at Dec 31st 25:00 = the next Jan 1st
        let all = "EST5EDT,0/0,J365/25";
        for at in ["2150-01-01 05:00:00", "2150-07-01 00:00:00", "2150-12-31 23:59:59", "2151-01-01 05:00:00"] {
            assert_eq!(offset(all, at), (-14400, 3600), "{at}");
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
