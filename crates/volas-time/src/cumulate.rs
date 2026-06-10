//! Cumulation: group fine bars into coarser periods and aggregate.

use volas_core::{DataFrame, Index, IndexKind, Result, Tz, VolasError};

use crate::agg::AggSpec;
use crate::time_frame::TimeFrame;

/// A stateful, incremental OHLCV cumulator (the live `cum_append` engine). The
/// currently-open period's raw bars are held explicitly in `open` — there is no
/// hidden frame metadata.
pub struct Cumulator {
    tf: TimeFrame,
    spec: AggSpec,
    /// Finalized coarse rows (closed periods).
    closed: Option<DataFrame>,
    /// Raw bars of the still-open (last) period.
    open: Option<DataFrame>,
}

impl Cumulator {
    /// A fresh cumulator for `tf` with aggregation `spec`.
    pub fn new(tf: TimeFrame, spec: AggSpec) -> Self {
        Cumulator {
            tf,
            spec,
            closed: None,
            open: None,
        }
    }

    /// Feed fine bars (must have a `DatetimeIndex`). Periods that close are
    /// finalized; the trailing period stays open for the next call.
    pub fn append(&mut self, fine: &DataFrame) -> Result<()> {
        if fine.height() == 0 {
            return Err(VolasError::Value(
                "the data frame to be appended is empty".into(),
            ));
        }

        // raw = the still-open period's bars + the incoming bars.
        let raw = match self.open.take() {
            Some(mut open) => {
                open.append(fine)?;
                open
            }
            None => fine.clone(),
        };

        let (ts, tz): (Vec<i64>, _) = match raw.index().kind() {
            IndexKind::Datetime(v, tz) => (v.clone(), *tz),
            _ => {
                return Err(VolasError::Value(
                    "cumulation target must have a DatetimeIndex".into(),
                ))
            }
        };

        // A NaT (i64::MIN) index row has no timestamp to bucket; without this guard
        // it falls into its own period (silently splitting same-period bars and
        // emitting a NaT-labelled coarse row). A live ingest with a timeless bar is
        // a data-quality problem that must be handled explicitly, not aggregated.
        if ts.iter().any(|&t| t == i64::MIN) {
            return Err(VolasError::Value(
                "cannot cumulate over a DatetimeIndex containing NaT; drop or fill \
                 the missing index timestamps first"
                    .into(),
            ));
        }

        let runs = group_runs(&ts, self.tf, tz);
        let last = runs.len() - 1;

        // Finalize every run except the last.
        for &(a, b) in &runs[..last] {
            let coarse = aggregate_period(&raw.slice(a, b), &self.spec)?;
            self.closed = Some(match self.closed.take() {
                Some(mut closed) => {
                    closed.append(&coarse)?;
                    closed
                }
                None => coarse,
            });
        }

        // The last run remains open.
        let (a, b) = runs[last];
        self.open = Some(raw.slice(a, b));
        Ok(())
    }

    /// The current cumulated frame: closed periods plus the open period
    /// aggregated as the (live) last row.
    pub fn frame(&self) -> Result<DataFrame> {
        let open_coarse = match &self.open {
            Some(open) => Some(aggregate_period(open, &self.spec)?),
            None => None,
        };
        match (&self.closed, open_coarse) {
            (Some(closed), Some(open)) => {
                let mut out = closed.clone();
                out.append(&open)?;
                Ok(out)
            }
            (Some(closed), None) => Ok(closed.clone()), // LCOV_EXCL_LINE
            (None, Some(open)) => Ok(open),
            (None, None) => DataFrame::new(Vec::new(), Vec::new(), None),
        }
    }

    /// The current open period aggregated into a single (live) coarse bar, or
    /// `None` if nothing is open. O(open period) — for live tick reads.
    pub fn last(&self) -> Result<Option<DataFrame>> {
        match &self.open {
            Some(open) => Ok(Some(aggregate_period(open, &self.spec)?)),
            None => Ok(None),
        }
    }

    /// A clone of the still-open period's raw fine bars (`None` if nothing is
    /// open) — lets a caller carry the folding state forward after a one-shot
    /// `cumulate`, so further fine bars fold into the same forming period.
    pub fn open_clone(&self) -> Option<DataFrame> {
        self.open.clone()
    }
}

/// One-shot resample of `df` (must have a `DatetimeIndex`) into `tf` periods.
pub fn cumulate(df: &DataFrame, tf: TimeFrame, spec: &AggSpec) -> Result<DataFrame> {
    let mut cumulator = Cumulator::new(tf, spec.clone());
    cumulator.append(df)?;
    cumulator.frame()
}

/// Split a sorted timestamp slice into `[start, end)` runs of equal period key,
/// bucketing in `tz`'s wall-clock so hour+ periods align to the local trading day.
fn group_runs(ts: &[i64], tf: TimeFrame, tz: Tz) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start = 0;
    let mut key = tf.unify_tz(ts[0], tz);
    for i in 1..ts.len() {
        let k = tf.unify_tz(ts[i], tz);
        if k != key {
            runs.push((start, i));
            start = i;
            key = k;
        }
    }
    runs.push((start, ts.len()));
    runs
}

/// Positions to keep when consecutive same-timestamp rows are deduped (keep the
/// last of each run — a re-sent / corrected tick updates rather than accumulates).
fn dedup_keep_last(ts: &[i64]) -> Vec<usize> {
    let n = ts.len();
    (0..n)
        .filter(|&i| i + 1 == n || ts[i + 1] != ts[i])
        .collect()
}

/// Aggregate one period's raw bars into a single coarse row (1-row frame). The
/// index label is the period's first timestamp.
pub fn aggregate_period(period: &DataFrame, spec: &AggSpec) -> Result<DataFrame> {
    let (ts, tz): (&[i64], _) = match period.index().kind() {
        IndexKind::Datetime(v, tz) => (v, *tz),
        _ => {
            return Err(VolasError::Value(
                "cumulation period must have a DatetimeIndex".into(),
            ))
        }
    };
    let kept = dedup_keep_last(ts);
    let start_ns = ts[kept[0]];

    let names = period.names().to_vec();
    let mut columns = Vec::with_capacity(names.len());
    for (name, col) in names.iter().zip(period.columns()) {
        columns.push(spec.agg_for(name).reduce(col, &kept)?);
    }
    // The aggregated index keeps the source index's name (pandas resample parity).
    let index = Index::datetime(vec![start_ns], tz).with_name(period.index().name().map(String::from));
    DataFrame::new(names, columns, Some(index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use volas_core::{datetime, Column};

    fn frame(
        times: &[&str],
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
        vol: &[f64],
    ) -> DataFrame {
        let idx: Vec<i64> = times
            .iter()
            .map(|t| datetime::parse_ns(t).unwrap())
            .collect();
        DataFrame::new(
            vec![
                "open".into(),
                "high".into(),
                "low".into(),
                "close".into(),
                "volume".into(),
            ],
            vec![
                Column::f64(open.to_vec()),
                Column::f64(high.to_vec()),
                Column::f64(low.to_vec()),
                Column::f64(close.to_vec()),
                Column::f64(vol.to_vec()),
            ],
            Some(Index::datetime(idx, Tz::Utc)),
        )
        .unwrap()
    }

    // Five 1-minute bars in one 5-minute period, plus one in the next.
    fn sample() -> DataFrame {
        frame(
            &[
                "2020-01-01 00:00:00",
                "2020-01-01 00:01:00",
                "2020-01-01 00:02:00",
                "2020-01-01 00:03:00",
                "2020-01-01 00:04:00",
                "2020-01-01 00:05:00",
            ],
            &[10.0, 11.0, 12.0, 13.0, 14.0, 20.0],
            &[15.0, 16.0, 17.0, 18.0, 19.0, 25.0],
            &[5.0, 6.0, 7.0, 8.0, 9.0, 15.0],
            &[11.0, 12.0, 13.0, 14.0, 15.0, 21.0],
            &[100.0, 100.0, 100.0, 100.0, 100.0, 50.0],
        )
    }

    #[test]
    fn cumulate_5m_aggregates_each_group() {
        let df = sample();
        let out = cumulate(&df, TimeFrame::Min5, &AggSpec::ohlcv()).unwrap();
        assert_eq!(out.height(), 2);
        // first period (00:00..00:04): open=10 high=19 low=5 close=15 volume=500
        assert_eq!(out.column("open").unwrap().as_f64().unwrap()[0], 10.0);
        assert_eq!(out.column("high").unwrap().as_f64().unwrap()[0], 19.0);
        assert_eq!(out.column("low").unwrap().as_f64().unwrap()[0], 5.0);
        assert_eq!(out.column("close").unwrap().as_f64().unwrap()[0], 15.0);
        assert_eq!(out.column("volume").unwrap().as_f64().unwrap()[0], 500.0);
        // second period (00:05): single bar
        assert_eq!(out.column("open").unwrap().as_f64().unwrap()[1], 20.0);
        assert_eq!(out.column("volume").unwrap().as_f64().unwrap()[1], 50.0);
        // index = period-start timestamps
        match out.index().kind() {
            IndexKind::Datetime(v, _) => {
                assert_eq!(v[0], datetime::parse_ns("2020-01-01 00:00:00").unwrap());
                assert_eq!(v[1], datetime::parse_ns("2020-01-01 00:05:00").unwrap());
            }
            _ => panic!("expected DatetimeIndex"), // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn cumulate_is_idempotent() {
        let df = sample();
        let once = cumulate(&df, TimeFrame::Min5, &AggSpec::ohlcv()).unwrap();
        let twice = cumulate(&once, TimeFrame::Min5, &AggSpec::ohlcv()).unwrap();
        assert!(once.equals(&twice));
    }

    #[test]
    fn incremental_matches_one_shot() {
        let df = sample();
        let one_shot = cumulate(&df, TimeFrame::Min5, &AggSpec::ohlcv()).unwrap();

        let mut cum = Cumulator::new(TimeFrame::Min5, AggSpec::ohlcv());
        for i in 0..df.height() {
            cum.append(&df.slice(i, i + 1)).unwrap();
        }
        assert!(cum.frame().unwrap().equals(&one_shot));
    }

    #[test]
    fn cumulator_empty_then_open_only() {
        let mut cum = Cumulator::new(TimeFrame::Min5, AggSpec::ohlcv());
        // nothing fed yet: empty frame, no live bar
        assert_eq!(cum.frame().unwrap().height(), 0);
        assert!(cum.last().unwrap().is_none());
        // feed part of the first period: only an open period exists
        cum.append(&sample().slice(0, 2)).unwrap();
        assert!(cum.last().unwrap().is_some());
        assert_eq!(cum.frame().unwrap().height(), 1);
    }

    #[test]
    fn dedup_keeps_last_same_timestamp() {
        // two bars share 00:00:00 — the later one (volume 7) overrides the first.
        let df = frame(
            &[
                "2020-01-01 00:00:00",
                "2020-01-01 00:00:00",
                "2020-01-01 00:01:00",
            ],
            &[10.0, 99.0, 12.0],
            &[15.0, 99.0, 17.0],
            &[5.0, 99.0, 7.0],
            &[11.0, 99.0, 13.0],
            &[100.0, 7.0, 100.0],
        );
        let out = cumulate(&df, TimeFrame::Min5, &AggSpec::ohlcv()).unwrap();
        assert_eq!(out.height(), 1);
        // volume = 7 (dup kept) + 100 = 107, not 100 + 7 + 100
        assert_eq!(out.column("volume").unwrap().as_f64().unwrap()[0], 107.0);
        // open = first kept = the second 00:00 bar's open (99)
        assert_eq!(out.column("open").unwrap().as_f64().unwrap()[0], 99.0);
    }

    #[test]
    fn empty_append_errors() {
        let mut cum = Cumulator::new(TimeFrame::Min5, AggSpec::ohlcv());
        let empty = DataFrame::new(
            vec!["open".into()],
            vec![Column::f64(vec![])],
            Some(Index::datetime(vec![], Tz::Utc)),
        )
        .unwrap();
        assert!(cum.append(&empty).is_err());
    }

    #[test]
    fn non_datetime_index_errors() {
        let df = DataFrame::new(
            vec!["open".into()],
            vec![Column::f64(vec![1.0, 2.0])],
            None, // RangeIndex
        )
        .unwrap();
        assert!(cumulate(&df, TimeFrame::Min5, &AggSpec::ohlcv()).is_err());
    }

    #[test]
    fn aggregate_period_rejects_non_datetime_index() {
        // append's guard shadows this one in the public path, so call it directly.
        let df = DataFrame::new(vec!["open".into()], vec![Column::f64(vec![1.0])], None).unwrap();
        assert!(aggregate_period(&df, &AggSpec::ohlcv()).is_err());
    }
}
