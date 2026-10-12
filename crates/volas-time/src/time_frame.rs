//! Time frames and their period grid.

use volas_core::{Result, Tz, VolasError};

/// An OHLCV sampling period.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeFrame {
    Sec1,
    Min1,
    Min3,
    Min5,
    Min15,
    Min30,
    Hour1,
    Hour2,
    Hour4,
    Hour6,
    Hour8,
    Hour12,
    Day1,
    Day3,
    Week1,
    Month1,
    Year1,
}

impl TimeFrame {
    /// Parse a label (`"5m"`, `"1h"`, `"1d"`, `"1M"`, `"1w"`, `"1y"`, `"1s"`, plus
    /// upper-case aliases).
    pub fn from_label(s: &str) -> Result<TimeFrame> {
        use TimeFrame::*;
        Ok(match s {
            "1s" | "1S" => Sec1,
            "1m" => Min1,
            "3m" => Min3,
            "5m" => Min5,
            "15m" => Min15,
            "30m" => Min30,
            "1h" | "1H" => Hour1,
            "2h" | "2H" => Hour2,
            "4h" | "4H" => Hour4,
            "6h" | "6H" => Hour6,
            "8h" | "8H" => Hour8,
            "12h" | "12H" => Hour12,
            "1d" | "1D" => Day1,
            "3d" | "3D" => Day3,
            "1w" | "1W" => Week1,
            "1M" => Month1,
            "1y" | "1Y" => Year1,
            _ => {
                return Err(VolasError::Value(format!(
                    "\"{s}\" is an invalid time frame"
                )))
            }
        })
    }

    /// The canonical label, e.g. `"5m"`.
    pub fn label(&self) -> &'static str {
        use TimeFrame::*;
        match self {
            Sec1 => "1s",
            Min1 => "1m",
            Min3 => "3m",
            Min5 => "5m",
            Min15 => "15m",
            Min30 => "30m",
            Hour1 => "1h",
            Hour2 => "2h",
            Hour4 => "4h",
            Hour6 => "6h",
            Hour8 => "8h",
            Hour12 => "12h",
            Day1 => "1d",
            Day3 => "3d",
            Week1 => "1w",
            Month1 => "1M",
            Year1 => "1y",
        }
    }

    /// The UTC epoch-ns instant at which the period holding `ns` **starts** on
    /// `tz`'s local grid — at once the period's identity (two instants share a
    /// period iff their starts are equal) and its bar label (owner decision
    /// 2026-06-12: a bar is labelled with its grid period start, origin + n·delta,
    /// never its first raw timestamp).
    ///
    /// - **Sub-day** frames bucket by real instants on the local grid anchored at
    ///   local midnight (a 15-minute bar starts at :00/:15/:30/:45, multi-hour bars
    ///   at hours 0/k/2k…): a wall-clock hour that a DST fall-back repeats is two
    ///   periods with distinct start instants, and a grid point a spring-forward
    ///   gap skips is replaced by the instant ending the gap
    ///   ([`Tz::floor_local_grid`]).
    /// - **Day-and-longer** frames follow the local civil calendar: Day1 starts at
    ///   local midnight, Day3 on the Unix-epoch-anchored 3-day grid, Week1 on
    ///   Monday, Month1 / Year1 on the calendar; a DST anomaly at that midnight
    ///   resolves to the day's earliest real instant ([`Tz::day_start`]).
    ///
    /// A period starting outside the datetime64[ns] range is an `Overflow` error.
    pub fn period_start(&self, ns: i64, tz: Tz) -> Result<i64> {
        use TimeFrame::*;
        if let Some(step) = self.sub_day_ns() {
            return tz.floor_local_grid(ns, step).map_err(|_| self.out_of_range(ns));
        }
        let (y, mo, d, _, _, _) = tz.civil_parts(ns);
        let days = volas_core::datetime::days_from_civil(y, mo, d);
        let (y, mo, d) = match self {
            Day3 => volas_core::datetime::civil_from_days(days.div_euclid(3) * 3),
            // 1970-01-01 is a Thursday, so `+3` anchors week boundaries on Monday.
            Week1 => volas_core::datetime::civil_from_days((days + 3).div_euclid(7) * 7 - 3),
            Month1 => (y, mo, 1),
            Year1 => (y, 1, 1),
            _ => (y, mo, d), // Day1
        };
        // the civil date of an instant is a valid date: only the range can fail
        tz.day_start(y, mo, d).map_err(|_| self.out_of_range(ns))
    }

    /// The grid step of a sub-day frame in nanoseconds; `None` for the
    /// day-and-longer (calendar) frames.
    fn sub_day_ns(self) -> Option<i64> {
        match self.duration_secs() {
            Some(secs) if secs < 86_400 => Some(secs * 1_000_000_000),
            _ => None,
        }
    }

    /// The `Overflow` error of a period whose start is not a datetime64[ns] instant.
    fn out_of_range(self, ns: i64) -> VolasError {
        VolasError::Overflow(format!(
            "the {} period holding {} starts outside the datetime64[ns] range \
             (1677-09-21 00:12:43.145224193 .. 2262-04-11 23:47:16.854775807)",
            self.label(),
            volas_core::datetime::format_ns(ns)
        ))
    }

    /// Whether this frame can be cleanly coarsened (aggregated) up to `dst` —
    /// i.e. every `dst` bucket boundary is also a boundary of `self`, so each
    /// `dst` bar is a whole number of `self` bars with none straddling.
    ///
    /// Fixed-duration frames (≤ `Week1`) nest by duration divisibility on the
    /// epoch grid. The calendar frames need care: a sub-day / 1-day frame nests
    /// into a week / month / year (its boundaries fall on every UTC midnight,
    /// hence on Monday week-starts and on the 1st), and a month nests into a year
    /// — but a **week or a 3-day bar does NOT nest into a month or year** (ISO
    /// weeks and epoch-anchored 3-day bars straddle calendar boundaries).
    pub fn can_coarsen(self, dst: TimeFrame) -> bool {
        use TimeFrame::*;
        if self == dst {
            return true;
        }
        let day_aligned = matches!(
            self,
            Sec1 | Min1
                | Min3
                | Min5
                | Min15
                | Min30
                | Hour1
                | Hour2
                | Hour4
                | Hour6
                | Hour8
                | Hour12
                | Day1
        );
        match dst {
            Year1 => self == Month1 || day_aligned,
            Month1 => day_aligned,
            Week1 => day_aligned,
            _ => match (self.duration_secs(), dst.duration_secs()) {
                (Some(s), Some(d)) => d > s && d % s == 0,
                _ => false,
            },
        }
    }

    /// The frame's length in its own unit: a sub-day frame is an exact number of
    /// seconds; a frame of a day or longer is a count of days, weeks, months or
    /// years (calendar units, whose wall-clock length is not fixed).
    pub fn span(self) -> Span {
        use TimeFrame::*;
        match self {
            Sec1 => Span::Seconds(1),
            Min1 => Span::Seconds(60),
            Min3 => Span::Seconds(180),
            Min5 => Span::Seconds(300),
            Min15 => Span::Seconds(900),
            Min30 => Span::Seconds(1800),
            Hour1 => Span::Seconds(3600),
            Hour2 => Span::Seconds(7200),
            Hour4 => Span::Seconds(14400),
            Hour6 => Span::Seconds(21600),
            Hour8 => Span::Seconds(28800),
            Hour12 => Span::Seconds(43200),
            Day1 => Span::Days(1),
            Day3 => Span::Days(3),
            Week1 => Span::Weeks(1),
            Month1 => Span::Months(1),
            Year1 => Span::Years(1),
        }
    }

    /// The fixed duration in seconds for the fixed-length frames; `None` for the
    /// variable-length calendar frames (`Month1` / `Year1`).
    fn duration_secs(self) -> Option<i64> {
        match self.span() {
            Span::Seconds(s) => Some(s),
            Span::Days(k) => Some(k * 86400),
            Span::Weeks(k) => Some(k * 604800),
            Span::Months(_) | Span::Years(_) => None,
        }
    }
}

/// A [`TimeFrame`]'s length in its own unit (see [`TimeFrame::span`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Span {
    /// A sub-day frame: an exact duration of this many seconds.
    Seconds(i64),
    /// This many days.
    Days(i64),
    /// This many weeks.
    Weeks(i64),
    /// This many calendar months.
    Months(i64),
    /// This many calendar years.
    Years(i64),
}

#[cfg(test)]
mod tests {
    use super::*;
    use volas_core::datetime;

    /// `period_start` floors an off-grid instant to its grid origin + n·delta
    /// for every frame kind — the bar-label contract (owner decision 2026-06-12).
    #[test]
    fn period_start_floors_to_the_grid() {
        let ns = datetime::parse_ns("2024-05-22 13:47:23").unwrap(); // a Wednesday
        let start = |tf: TimeFrame| tf.period_start(ns, Tz::Utc).unwrap();
        let at = |s: &str| datetime::parse_ns(s).unwrap();
        assert_eq!(start(TimeFrame::Sec1), at("2024-05-22 13:47:23"));
        assert_eq!(start(TimeFrame::Min1), at("2024-05-22 13:47:00"));
        assert_eq!(start(TimeFrame::Min3), at("2024-05-22 13:45:00"));
        assert_eq!(start(TimeFrame::Min5), at("2024-05-22 13:45:00"));
        assert_eq!(start(TimeFrame::Min15), at("2024-05-22 13:45:00"));
        assert_eq!(start(TimeFrame::Min30), at("2024-05-22 13:30:00"));
        assert_eq!(start(TimeFrame::Hour1), at("2024-05-22 13:00:00"));
        assert_eq!(start(TimeFrame::Hour2), at("2024-05-22 12:00:00"));
        assert_eq!(start(TimeFrame::Hour4), at("2024-05-22 12:00:00"));
        assert_eq!(start(TimeFrame::Hour6), at("2024-05-22 12:00:00"));
        assert_eq!(start(TimeFrame::Hour8), at("2024-05-22 08:00:00"));
        assert_eq!(start(TimeFrame::Hour12), at("2024-05-22 12:00:00"));
        assert_eq!(start(TimeFrame::Day1), at("2024-05-22 00:00:00"));
        // Day3 is epoch-anchored: day 19865 (2024-05-22) floors to 19863 (05-20).
        assert_eq!(start(TimeFrame::Day3), at("2024-05-20 00:00:00"));
        // Week1 anchors on Monday: 2024-05-22 is Wednesday -> Monday 05-20.
        assert_eq!(start(TimeFrame::Week1), at("2024-05-20 00:00:00"));
        assert_eq!(start(TimeFrame::Month1), at("2024-05-01 00:00:00"));
        assert_eq!(start(TimeFrame::Year1), at("2024-01-01 00:00:00"));
    }

    /// In a named zone the grid is the LOCAL wall-clock (a Shanghai daily bar
    /// starts at 00:00 +08:00 = 16:00 UTC of the prior day); sub-day bars split a
    /// DST fold by real instants, day bars keep the civil day.
    #[test]
    fn period_start_uses_local_wall_clock_and_resolves_dst() {
        let sh = Tz::parse("Asia/Shanghai").unwrap();
        let ns = datetime::parse_ns("2024-05-22 03:30:00").unwrap(); // 11:30 in Shanghai
        assert_eq!(
            TimeFrame::Day1.period_start(ns, sh).unwrap(),
            datetime::parse_ns("2024-05-21 16:00:00").unwrap() // 05-22 00:00 +08
        );
        // America/New_York 2024-11-03: 01:00-02:00 repeats (fold). Sub-day bars
        // follow real instants: the first 01:00 (EDT) starts at 05:00 UTC, the
        // repeated 01:00 (EST) is its own bar starting at 06:00 UTC.
        let ny = Tz::parse("America/New_York").unwrap();
        let in_fold = datetime::parse_ns("2024-11-03 05:30:00").unwrap(); // first 01:30 EDT
        assert_eq!(
            TimeFrame::Hour1.period_start(in_fold, ny).unwrap(),
            datetime::parse_ns("2024-11-03 05:00:00").unwrap()
        );
        let repeated = datetime::parse_ns("2024-11-03 06:30:00").unwrap(); // second 01:30 EST
        assert_eq!(
            TimeFrame::Hour1.period_start(repeated, ny).unwrap(),
            datetime::parse_ns("2024-11-03 06:00:00").unwrap()
        );
        // a day-and-longer bar stays one civil day across the fold
        assert_eq!(
            TimeFrame::Day1.period_start(repeated, ny).unwrap(),
            TimeFrame::Day1.period_start(in_fold, ny).unwrap()
        );
        // 2024-03-10: 02:00-03:00 EST does not exist (gap). A 02:xx-grid start is
        // unreachable from real timestamps, but a Day1 bar over the gap day still
        // starts at the (existing) local midnight.
        let gap_day = datetime::parse_ns("2024-03-10 12:00:00").unwrap();
        assert_eq!(
            TimeFrame::Day1.period_start(gap_day, ny).unwrap(),
            datetime::parse_ns("2024-03-10 05:00:00").unwrap() // 00:00 EST
        );
        // América/São_Paulo 2018-11-04: midnight itself did not exist (the clock
        // sprang 00:00 -> 01:00). The daily bar's start resolves to the first
        // real instant of the day: 01:00 -02:00 = 03:00 UTC (the gap-probe path).
        let sp = Tz::parse("America/Sao_Paulo").unwrap();
        let mid_day = datetime::parse_ns("2018-11-04 15:00:00").unwrap();
        assert_eq!(
            TimeFrame::Day1.period_start(mid_day, sp).unwrap(),
            datetime::parse_ns("2018-11-04 03:00:00").unwrap()
        );
    }

    /// `civil_from_days` is the exact inverse of `days_from_civil` across a wide
    /// sweep, including the negative (pre-epoch) range.
    #[test]
    fn civil_from_days_roundtrips() {
        for days in (-30000..40000).step_by(97) {
            let (y, mo, d) = datetime::civil_from_days(days);
            assert_eq!(datetime::days_from_civil(y, mo, d), days);
        }
    }

    #[test]
    fn labels_and_parse() {
        assert_eq!(TimeFrame::Min5.label(), "5m");
        assert_eq!(TimeFrame::Month1.label(), "1M");
        assert_eq!(TimeFrame::from_label("5m").unwrap(), TimeFrame::Min5);
        assert_eq!(TimeFrame::from_label("1M").unwrap(), TimeFrame::Month1);
        assert_eq!(TimeFrame::from_label("1H").unwrap(), TimeFrame::Hour1);
        assert!(TimeFrame::from_label("1").is_err());
    }

    /// Two instants share a period iff their period starts are equal.
    #[test]
    fn period_start_groups_same_5min_block() {
        let start = |t: &str| TimeFrame::Min5.period_start(datetime::parse_ns(t).unwrap(), Tz::Utc).unwrap();
        assert_eq!(start("2020-01-01 00:00:00"), start("2020-01-01 00:04:59"));
        assert_ne!(start("2020-01-01 00:00:00"), start("2020-01-01 00:05:00"));
    }

    #[test]
    fn week_is_continuous_and_monday_anchored() {
        // 2024-01-29 is a Monday; the run Mon..Sun (2024-02-04) crosses Feb 1.
        let week = |t: &str| TimeFrame::Week1.period_start(datetime::parse_ns(t).unwrap(), Tz::Utc).unwrap();
        // one continuous week, even across the month boundary
        assert_eq!(week("2024-01-29 00:00:00"), week("2024-02-04 23:59:59"));
        // boundaries land on Monday
        assert_ne!(week("2024-01-29 00:00:00"), week("2024-01-28 23:59:59"));
        assert_ne!(week("2024-01-29 00:00:00"), week("2024-02-05 00:00:00"));
    }

    #[test]
    fn day3_is_continuous_across_month_boundary() {
        // Each calendar day keeps the epoch-anchored 3-day period or starts the
        // next one 3 days later — never a month-reset jump at Feb 1.
        let day3 = |t: &str| TimeFrame::Day3.period_start(datetime::parse_ns(t).unwrap(), Tz::Utc).unwrap();
        const THREE_DAYS: i64 = 3 * 86_400_000_000_000;
        let mut prev = day3("2024-01-28 00:00:00");
        for day in ["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"] {
            let k = day3(&format!("{day} 00:00:00"));
            assert!(k == prev || k == prev + THREE_DAYS, "{day}: period {k}, prev {prev}");
            prev = k;
        }
    }

    /// A period whose start is not a datetime64[ns] instant is an `Overflow` error
    /// (never a panic): the first instants of the axis have no whole-second,
    /// 3-day, week, month or year start.
    #[test]
    fn period_start_outside_the_range_is_an_error() {
        let first = i64::MIN + 1; // 1677-09-21 00:12:43.145224193
        for tf in [TimeFrame::Sec1, TimeFrame::Day1, TimeFrame::Day3, TimeFrame::Week1,
                   TimeFrame::Month1, TimeFrame::Year1] {
            let err = tf.period_start(first, Tz::Utc).unwrap_err();
            assert!(matches!(err, VolasError::Overflow(_)), "{tf:?}: {err}");
            assert!(err.to_string().contains(&format!("the {} period holding", tf.label())), "{err}");
        }
        let ny = Tz::parse("America/New_York").unwrap();
        assert!(TimeFrame::Hour1.period_start(first, ny).is_err());
        // the last instant still has a representable start
        assert!(TimeFrame::Year1.period_start(i64::MAX, Tz::Utc).is_ok());
    }

    #[test]
    fn can_coarsen_truth_table() {
        use TimeFrame::*;
        for (s, d) in [
            (Min5, Min5), // identity (= copy)
            // every fixed-duration source frame tiles a coarser one on the epoch grid
            (Sec1, Min1),
            (Min1, Min5),
            (Min30, Hour1),
            (Hour2, Hour6),
            (Hour8, Day1),
            (Min5, Min15),
            (Min15, Hour1),
            (Min5, Hour1),
            (Hour1, Hour4),
            (Hour4, Hour12),
            (Hour1, Day1),
            (Day1, Day3),
            (Hour12, Day3),
            (Day1, Week1),
            (Min5, Week1),
            (Day1, Month1),
            (Hour1, Month1),
            (Month1, Year1),
            (Day1, Year1),
        ] {
            assert!(s.can_coarsen(d), "{s:?} -> {d:?} should be valid");
        }
        for (s, d) in [
            (Min3, Min5),
            (Hour4, Hour6),
            (Day3, Week1),
            (Week1, Month1),
            (Week1, Year1),
            (Day3, Month1),
            (Day3, Year1),
            (Day1, Hour1), // refining, not coarsening
            (Week1, Day1),
            (Month1, Day1),
        ] {
            assert!(!s.can_coarsen(d), "{s:?} -> {d:?} should be invalid");
        }
    }

    const ALL: [TimeFrame; 17] = [
        TimeFrame::Sec1,
        TimeFrame::Min1,
        TimeFrame::Min3,
        TimeFrame::Min5,
        TimeFrame::Min15,
        TimeFrame::Min30,
        TimeFrame::Hour1,
        TimeFrame::Hour2,
        TimeFrame::Hour4,
        TimeFrame::Hour6,
        TimeFrame::Hour8,
        TimeFrame::Hour12,
        TimeFrame::Day1,
        TimeFrame::Day3,
        TimeFrame::Week1,
        TimeFrame::Month1,
        TimeFrame::Year1,
    ];

    #[test]
    fn label_roundtrips_for_every_frame() {
        for tf in ALL {
            assert_eq!(TimeFrame::from_label(tf.label()).unwrap(), tf);
        }
        // Upper-case aliases also parse.
        for alias in ["1S", "2H", "4H", "6H", "8H", "12H", "3D", "1W", "1Y"] {
            assert!(TimeFrame::from_label(alias).is_ok());
        }
    }

    #[test]
    fn period_start_covers_every_frame() {
        // Every frame's start is at or before the instant, inside its period, and
        // is its own period's start.
        let ns = datetime::parse_ns("2021-07-19 22:47:53").unwrap();
        for tf in ALL {
            let start = tf.period_start(ns, Tz::Utc).unwrap();
            assert!(start <= ns, "{tf:?}");
            assert_eq!(tf.period_start(start, Tz::Utc).unwrap(), start, "{tf:?}");
        }
    }

    #[test]
    fn span_is_exact_seconds_below_a_day_and_calendar_units_above() {
        use TimeFrame::*;
        let seconds = [
            (Sec1, 1), (Min1, 60), (Min3, 180), (Min5, 300), (Min15, 900), (Min30, 1800),
            (Hour1, 3600), (Hour2, 7200), (Hour4, 14400), (Hour6, 21600), (Hour8, 28800),
            (Hour12, 43200),
        ];
        for (tf, s) in seconds {
            assert_eq!(tf.span(), Span::Seconds(s), "{tf:?}");
            assert!(s < 86400);
        }
        assert_eq!(Day1.span(), Span::Days(1));
        assert_eq!(Day3.span(), Span::Days(3));
        assert_eq!(Week1.span(), Span::Weeks(1));
        assert_eq!(Month1.span(), Span::Months(1));
        assert_eq!(Year1.span(), Span::Years(1));
        assert_eq!(seconds.len() + 5, ALL.len());
    }
}
