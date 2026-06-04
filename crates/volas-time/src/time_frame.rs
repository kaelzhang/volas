//! Time frames and their period-key unification.

use volas_core::datetime;
use volas_core::{Result, VolasError};

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
            _ => return Err(VolasError::Value(format!("\"{s}\" is an invalid time frame"))),
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

    /// The (approximate) number of minutes in the frame (parity field).
    pub fn minutes(&self) -> i64 {
        use TimeFrame::*;
        match self {
            Sec1 => 1,
            Min1 => 1,
            Min3 => 3,
            Min5 => 5,
            Min15 => 15,
            Min30 => 30,
            Hour1 => 60,
            Hour2 => 120,
            Hour4 => 240,
            Hour6 => 360,
            Hour8 => 480,
            Hour12 => 720,
            Day1 => 1440,
            Day3 => 4320,
            Week1 => 10080,
            Month1 => 525600,
            Year1 => 525600,
        }
    }

    /// Unify an epoch-ns timestamp to its period key. Two timestamps are in the
    /// same period iff their keys are equal.
    pub fn unify(&self, ns: i64) -> i64 {
        let (y, mo, d, h, mi, s) = datetime::civil_parts(ns);
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
            Day3 => (d / 3) * DAY + mo * MONTH + y * YEAR,
            Week1 => (d / 7) * DAY + mo * MONTH + y * YEAR,
            Month1 => mo * MONTH + y * YEAR,
            Year1 => y * YEAR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn label_roundtrips_and_minutes_for_every_frame() {
        for tf in ALL {
            // label -> from_label is a round-trip, and minutes() is positive.
            assert_eq!(TimeFrame::from_label(tf.label()).unwrap(), tf);
            assert!(tf.minutes() > 0);
        }
        // Spot-check a few minute values and the upper-case aliases.
        assert_eq!(TimeFrame::Hour12.minutes(), 720);
        assert_eq!(TimeFrame::Day3.minutes(), 4320);
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
        assert_eq!(TimeFrame::Hour1.unify(mid), TimeFrame::Hour1.unify(same_hour));
        let same_day = datetime::parse_ns("2021-07-19 00:00:00").unwrap();
        assert_eq!(TimeFrame::Day1.unify(mid), TimeFrame::Day1.unify(same_day));
        let same_month = datetime::parse_ns("2021-07-01 00:00:00").unwrap();
        assert_eq!(TimeFrame::Month1.unify(mid), TimeFrame::Month1.unify(same_month));
    }
}
