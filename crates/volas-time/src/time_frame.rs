//! Time frames and their period-key unification.

use volas_core::{Result, Tz, VolasError};

// Magnitude-positional encoding of a truncated civil datetime (matches
// stock-pandas, so `unify("2020-01-02 03:04:05")` for seconds == 20200102030405).
const SEC: i64 = 1;
const MIN: i64 = 100;
const HOUR: i64 = 10_000;
const DAY: i64 = 1_000_000;
const MONTH: i64 = 100_000_000;
const YEAR: i64 = 10_000_000_000;

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

    /// Unify an epoch-ns timestamp to its period key (in **UTC** wall-clock). Two
    /// timestamps are in the same period iff their keys are equal.
    pub fn unify(&self, ns: i64) -> i64 {
        self.unify_tz(ns, Tz::Utc)
    }

    /// Unify an epoch-ns timestamp to its period key in `tz`'s wall-clock, so that
    /// hour+ buckets (e.g. daily bars) align to the local trading day — DST-aware
    /// for a named zone. Storage stays UTC; only the bucketing uses `tz`.
    pub fn unify_tz(&self, ns: i64, tz: Tz) -> i64 {
        let (y, mo, d, h, mi, s) = tz.civil_parts(ns);
        use TimeFrame::*;
        match self {
            Sec1 => s * SEC + mi * MIN + h * HOUR + d * DAY + mo * MONTH + y * YEAR,
            Min1 => mi * MIN + h * HOUR + d * DAY + mo * MONTH + y * YEAR,
            Min3 => (mi / 3) * MIN + h * HOUR + d * DAY + mo * MONTH + y * YEAR,
            Min5 => (mi / 5) * MIN + h * HOUR + d * DAY + mo * MONTH + y * YEAR,
            Min15 => (mi / 15) * MIN + h * HOUR + d * DAY + mo * MONTH + y * YEAR,
            Min30 => (mi / 30) * MIN + h * HOUR + d * DAY + mo * MONTH + y * YEAR,
            Hour1 => h * HOUR + d * DAY + mo * MONTH + y * YEAR,
            Hour2 => (h / 2) * HOUR + d * DAY + mo * MONTH + y * YEAR,
            Hour4 => (h / 4) * HOUR + d * DAY + mo * MONTH + y * YEAR,
            Hour6 => (h / 6) * HOUR + d * DAY + mo * MONTH + y * YEAR,
            Hour8 => (h / 8) * HOUR + d * DAY + mo * MONTH + y * YEAR,
            Hour12 => (h / 12) * HOUR + d * DAY + mo * MONTH + y * YEAR,
            Day1 => d * DAY + mo * MONTH + y * YEAR,
            // Day3 and Week1 are CONTINUOUS, epoch-anchored buckets — never reset
            // at a month boundary (the old `(d/3)`/`(d/7)` civil-field scheme split
            // a real 3-day/week run at the month edge). They key off the continuous
            // day count since the Unix epoch; the key is monotonic and unique per
            // bucket, which is all `group_runs` needs.
            Day3 => volas_core::datetime::days_from_civil(y, mo, d).div_euclid(3),
            // 1970-01-01 is a Thursday, so `+3` anchors week boundaries on Monday.
            Week1 => (volas_core::datetime::days_from_civil(y, mo, d) + 3).div_euclid(7),
            Month1 => mo * MONTH + y * YEAR,
            Year1 => y * YEAR,
        }
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

    /// The fixed duration in seconds for the fixed-length frames; `None` for the
    /// variable-length calendar frames (`Month1` / `Year1`). Used only by
    /// [`can_coarsen`](Self::can_coarsen).
    fn duration_secs(self) -> Option<i64> {
        use TimeFrame::*;
        Some(match self {
            Sec1 => 1,
            Min1 => 60,
            Min3 => 180,
            Min5 => 300,
            Min15 => 900,
            Min30 => 1800,
            Hour1 => 3600,
            Hour2 => 7200,
            Hour4 => 14400,
            Hour6 => 21600,
            Hour8 => 28800,
            Hour12 => 43200,
            Day1 => 86400,
            Day3 => 259200,
            Week1 => 604800,
            Month1 | Year1 => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use volas_core::datetime;

    #[test]
    fn labels_and_parse() {
        assert_eq!(TimeFrame::Min5.label(), "5m");
        assert_eq!(TimeFrame::Month1.label(), "1M");
        assert_eq!(TimeFrame::from_label("5m").unwrap(), TimeFrame::Min5);
        assert_eq!(TimeFrame::from_label("1M").unwrap(), TimeFrame::Month1);
        assert_eq!(TimeFrame::from_label("1H").unwrap(), TimeFrame::Hour1);
        assert!(TimeFrame::from_label("1").is_err());
    }

    #[test]
    fn unify_second_contract() {
        let ns = datetime::parse_ns("2020-01-02 03:04:05").unwrap();
        assert_eq!(TimeFrame::Sec1.unify(ns), 20_200_102_030_405);
    }

    #[test]
    fn unify_groups_same_5min_block() {
        let a = datetime::parse_ns("2020-01-01 00:00:00").unwrap();
        let b = datetime::parse_ns("2020-01-01 00:04:59").unwrap();
        let c = datetime::parse_ns("2020-01-01 00:05:00").unwrap();
        assert_eq!(TimeFrame::Min5.unify(a), TimeFrame::Min5.unify(b));
        assert_ne!(TimeFrame::Min5.unify(a), TimeFrame::Min5.unify(c));
    }

    #[test]
    fn week_is_continuous_and_monday_anchored() {
        // 2024-01-29 is a Monday; the run Mon..Sun (2024-02-04) crosses Feb 1.
        let mon = datetime::parse_ns("2024-01-29 00:00:00").unwrap();
        let sun = datetime::parse_ns("2024-02-04 23:59:59").unwrap();
        let prev_sun = datetime::parse_ns("2024-01-28 23:59:59").unwrap();
        let next_mon = datetime::parse_ns("2024-02-05 00:00:00").unwrap();
        // one continuous week, even across the month boundary (the old (d/7) split it)
        assert_eq!(TimeFrame::Week1.unify(mon), TimeFrame::Week1.unify(sun));
        // boundaries land on Monday
        assert_ne!(
            TimeFrame::Week1.unify(mon),
            TimeFrame::Week1.unify(prev_sun)
        );
        assert_ne!(
            TimeFrame::Week1.unify(mon),
            TimeFrame::Week1.unify(next_mon)
        );
    }

    #[test]
    fn day3_is_continuous_across_month_boundary() {
        // Each calendar day advances the epoch-anchored 3-day bucket by 0 or 1 —
        // never the month-reset jump the old (d/3) scheme produced at Feb 1.
        let mut prev = TimeFrame::Day3.unify(datetime::parse_ns("2024-01-28 00:00:00").unwrap());
        for day in [
            "2024-01-29",
            "2024-01-30",
            "2024-01-31",
            "2024-02-01",
            "2024-02-02",
        ] {
            let k = TimeFrame::Day3.unify(datetime::parse_ns(&format!("{day} 00:00:00")).unwrap());
            assert!(k == prev || k == prev + 1, "{day}: bucket {k}, prev {prev}");
            prev = k;
        }
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
    fn unify_covers_every_branch() {
        // A timestamp whose every civil field is non-trivial, so each frame's
        // truncation arm is exercised and yields a well-formed key.
        let ns = datetime::parse_ns("2021-07-19 22:47:53").unwrap();
        for tf in ALL {
            assert!(tf.unify(ns) > 0);
        }
        // Same-period vs next-period contract for each granularity boundary.
        let mid = datetime::parse_ns("2021-07-19 22:47:53").unwrap();
        let same_hour = datetime::parse_ns("2021-07-19 22:00:00").unwrap();
        assert_eq!(
            TimeFrame::Hour1.unify(mid),
            TimeFrame::Hour1.unify(same_hour)
        );
        let same_day = datetime::parse_ns("2021-07-19 00:00:00").unwrap();
        assert_eq!(TimeFrame::Day1.unify(mid), TimeFrame::Day1.unify(same_day));
        let same_month = datetime::parse_ns("2021-07-01 00:00:00").unwrap();
        assert_eq!(
            TimeFrame::Month1.unify(mid),
            TimeFrame::Month1.unify(same_month)
        );
    }
}
