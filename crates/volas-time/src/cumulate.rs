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
        self.spec.check(fine)?; // before any state is taken

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
        if ts.contains(&i64::MIN) {
            return Err(VolasError::Value(
                "cannot cumulate over a DatetimeIndex containing NaT; drop or fill \
                 the missing index timestamps first"
                    .into(),
            ));
        }

        // The consecutive-run grouping below (and the open/close aggregation,
        // which is position-ordered) assumes a non-decreasing timeline. An
        // out-of-order input would silently split a period into duplicate coarse
        // rows and pick open/close by row position instead of time — reject it,
        // symmetric with the live fold path's out-of-order-bar guard. Sorting is
        // deliberately NOT done here: OHLCV open/close and the live cursor are
        // position-sensitive, so reordering must be an explicit caller action.
        if ts.windows(2).any(|w| w[1] < w[0]) {
            return Err(VolasError::Value(
                "cannot cumulate over a non-monotonic DatetimeIndex (bars out of \
                 order); sort the input by time first"
                    .into(),
            ));
        }

        let runs = group_runs(&ts, self.tf, tz)?;
        let last = runs.len() - 1;

        // Finalize every run except the last.
        for &(a, b) in &runs[..last] {
            let coarse = aggregate_period(&raw.slice(a, b), &self.spec, self.tf, tz)?;
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
            Some(open) => Some(aggregate_period(open, &self.spec, self.tf, grid_tz(open))?),
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
            Some(open) => Ok(Some(aggregate_period(open, &self.spec, self.tf, grid_tz(open))?)),
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

/// The zone owning the period grid of the cumulator's own bars: their index zone
/// (every bar it holds went through the datetime-index check of `append`).
fn grid_tz(bars: &DataFrame) -> Tz {
    match bars.index().kind() {
        IndexKind::Datetime(_, tz) => *tz,
        _ => unreachable!("cumulator bars carry a DatetimeIndex"), // LCOV_EXCL_LINE
    }
}

/// Split a sorted timestamp slice into `[start, end)` runs of the same period
/// (equal [`TimeFrame::period_start`] on `tz`'s local grid).
fn group_runs(ts: &[i64], tf: TimeFrame, tz: Tz) -> Result<Vec<(usize, usize)>> {
    let mut runs = Vec::new();
    let mut start = 0;
    let mut key = tf.period_start(ts[0], tz)?;
    for (i, &t) in ts.iter().enumerate().skip(1) {
        let k = tf.period_start(t, tz)?;
        if k != key {
            runs.push((start, i));
            start = i;
            key = k;
        }
    }
    runs.push((start, ts.len()));
    Ok(runs)
}

/// Positions to keep when consecutive same-timestamp rows are deduped (keep the
/// last of each run — a re-sent / corrected tick updates rather than accumulates).
fn dedup_keep_last(ts: &[i64]) -> Vec<usize> {
    let n = ts.len();
    (0..n)
        .filter(|&i| i + 1 == n || ts[i + 1] != ts[i])
        .collect()
}

/// Aggregate one period's raw bars into a single coarse row (1-row frame) on the
/// period grid of `tz` — the zone that owns the grid (a tf frame's own zone, never
/// an incoming bar's). The index label is the period's grid **start**,
/// [`TimeFrame::period_start`] of the first raw timestamp (owner decision
/// 2026-06-12: every TimeFrame has a fixed grid origin and a bar's label is
/// origin + n·delta, e.g. a 15-minute bar labels :00/:15/:30/:45 — never the first
/// raw timestamp, which can sit anywhere inside the period), in `tz`.
pub fn aggregate_period(period: &DataFrame, spec: &AggSpec, tf: TimeFrame, tz: Tz) -> Result<DataFrame> {
    let ts: &[i64] = match period.index().kind() {
        IndexKind::Datetime(v, _) if !v.is_empty() => v,
        IndexKind::Datetime(..) => {
            return Err(VolasError::Value("cumulation period has no bars".into()))
        }
        _ => {
            return Err(VolasError::Value(
                "cumulation period must have a DatetimeIndex".into(),
            ))
        }
    };
    let kept = dedup_keep_last(ts);
    let start_ns = tf.period_start(ts[kept[0]], tz)?;

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
        assert!(aggregate_period(&df, &AggSpec::ohlcv(), TimeFrame::Min5, Tz::Utc).is_err());
        // an empty period has no bar to label: an error, not an index panic
        let empty = frame(&[], &[], &[], &[], &[], &[]);
        let err = aggregate_period(&empty, &AggSpec::ohlcv(), TimeFrame::Min5, Tz::Utc).unwrap_err();
        assert!(err.to_string().contains("no bars"), "{err}");
    }

    /// The period grid belongs to the zone passed in, not to the bars' own index
    /// zone: a UTC-indexed bar aggregated on a +08:00 grid starts at the +08:00
    /// local midnight and carries that zone.
    #[test]
    fn aggregate_period_uses_the_grid_zone() {
        let bar = frame(&["2024-01-02 20:00:00"], &[1.0], &[1.0], &[1.0], &[1.0], &[1.0]);
        let sh = Tz::parse("+08:00").unwrap();
        let out = aggregate_period(&bar, &AggSpec::ohlcv(), TimeFrame::Day1, sh).unwrap();
        match out.index().kind() {
            IndexKind::Datetime(v, tz) => {
                // 20:00Z is 04:00 on 01-03 in +08:00, whose day starts at 01-02 16:00Z
                assert_eq!(v, &vec![datetime::parse_ns("2024-01-02 16:00:00").unwrap()]);
                assert_eq!(*tz, sh);
            }
            other => panic!("{other:?}"), // LCOV_EXCL_LINE
        }
    }

    /// Sub-day periods follow real instants on the local grid: the repeated
    /// 01:00 hour of a New York fall-back is two bars (pandas `resample('1h')`
    /// gives the same two 01:00 bars), while the day stays one bar.
    #[test]
    fn dst_fall_back_repeated_hour_is_two_bars() {
        let ny = Tz::parse("America/New_York").unwrap();
        let mut times = Vec::new();
        let mut t = datetime::parse_ns("2026-11-01 04:00:00").unwrap();
        while t < datetime::parse_ns("2026-11-01 08:00:00").unwrap() {
            times.push(t);
            t += 60_000_000_000;
        }
        let n = times.len();
        let df = DataFrame::new(
            vec!["volume".into()],
            vec![Column::f64(vec![1.0; n])],
            Some(Index::datetime(times, ny)),
        )
        .unwrap();
        let spec = AggSpec::ohlcv();
        let hourly = cumulate(&df, TimeFrame::Hour1, &spec).unwrap();
        let labels: Vec<i64> = match hourly.index().kind() {
            IndexKind::Datetime(v, _) => v.clone(),
            _ => unreachable!(), // LCOV_EXCL_LINE
        };
        let at = |s: &str| datetime::parse_ns(s).unwrap();
        // 00:00 EDT, 01:00 EDT, 01:00 EST, 02:00 EST
        assert_eq!(labels, vec![at("2026-11-01 04:00:00"), at("2026-11-01 05:00:00"),
                                at("2026-11-01 06:00:00"), at("2026-11-01 07:00:00")]);
        assert_eq!(hourly.columns()[0].as_f64().unwrap(), &[60.0, 60.0, 60.0, 60.0][..]);
        assert_eq!(cumulate(&df, TimeFrame::Day1, &spec).unwrap().height(), 1);
    }

    /// A period start outside the instant range raises (never a panic).
    #[test]
    fn a_period_start_outside_the_range_is_an_error() {
        let df = DataFrame::new(
            vec!["volume".into()],
            vec![Column::f64(vec![1.0])],
            Some(Index::datetime(vec![i64::MIN + 1], Tz::Utc)),
        )
        .unwrap();
        let err = cumulate(&df, TimeFrame::Week1, &AggSpec::ohlcv()).unwrap_err();
        assert!(matches!(err, VolasError::Overflow(_)), "{err}");
    }

    /// Owner decision 2026-06-12: a coarse bar is labelled with its grid period
    /// START (origin + n·delta), not the first raw timestamp. A 5-minute period
    /// whose first bar arrives mid-period (09:07) still labels 09:05.
    #[test]
    fn label_is_period_start_not_first_bar() {
        let df = frame(
            &["2024-01-02 09:07:00", "2024-01-02 09:09:00", "2024-01-02 09:11:00"],
            &[1.0, 2.0, 3.0],
            &[1.0, 2.0, 3.0],
            &[1.0, 2.0, 3.0],
            &[1.0, 2.0, 3.0],
            &[1.0, 1.0, 1.0],
        );
        let out = cumulate(&df, TimeFrame::Min5, &AggSpec::ohlcv()).unwrap();
        let labels: Vec<i64> = match out.index().kind() {
            IndexKind::Datetime(v, _) => v.clone(),
            _ => unreachable!(), // LCOV_EXCL_LINE
        };
        assert_eq!(
            labels,
            vec![
                datetime::parse_ns("2024-01-02 09:05:00").unwrap(),
                datetime::parse_ns("2024-01-02 09:10:00").unwrap(),
            ]
        );
    }
}
