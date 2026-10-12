//! Cumulation: group fine bars into coarser periods and aggregate.

use std::borrow::Cow;

use volas_core::{DataFrame, Index, IndexKind, Result, Tz, VolasError};

use crate::agg::AggSpec;
use crate::open_period::{dedup_keep_last, OpenPeriod};
use crate::time_frame::TimeFrame;

/// A stateful, incremental OHLCV cumulator (the live `cum_append` engine). The
/// currently-open period's bars are held explicitly as an [`OpenPeriod`] — there is
/// no hidden frame metadata.
pub struct Cumulator {
    tf: TimeFrame,
    spec: AggSpec,
    /// Finalized coarse rows (closed periods).
    closed: Option<DataFrame>,
    /// The fine bars of the still-open (last) period.
    open: Option<OpenPeriod>,
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
    /// finalized; the trailing period stays open for the next call. All or nothing:
    /// the closed rows and the next open period are built aside, and the cumulator
    /// takes them only once every step has succeeded, so on an error it is unchanged.
    pub fn append(&mut self, fine: &DataFrame) -> Result<()> {
        if fine.height() == 0 {
            return Err(VolasError::Value(
                "the data frame to be appended is empty".into(),
            ));
        }
        self.spec.check(fine)?;

        // raw = the still-open period's bars + the incoming bars.
        let raw = match &self.open {
            Some(open) => {
                let mut raw = open.bars().clone();
                raw.append(fine)?;
                Cow::Owned(raw)
            }
            None => Cow::Borrowed(fine),
        };

        let (ts, tz): (&[i64], Tz) = match raw.index().kind() {
            IndexKind::Datetime(v, tz) => (v, *tz),
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

        let runs = group_runs(ts, self.tf, tz)?;
        let last = runs.len() - 1;

        // Every run except the last closes: its coarse rows, gathered aside.
        let mut closing: Option<DataFrame> = None;
        for &(a, b) in &runs[..last] {
            let coarse = aggregate_period(&raw.slice_data(a, b), &self.spec, self.tf, tz)?;
            match &mut closing {
                Some(rows) => rows.append(&coarse)?,
                None => closing = Some(coarse),
            }
        }
        // The last run remains open (re-sent bars collapsed to the last one).
        let (a, b) = runs[last];
        let open = OpenPeriod::new(raw.slice_data(a, b))?;

        // Commit: one atomic append of the closed rows, then the open period.
        if let Some(rows) = closing {
            match &mut self.closed {
                Some(closed) => closed.append(&rows)?,
                None => self.closed = Some(rows),
            }
        }
        self.open = Some(open);
        Ok(())
    }

    /// The current cumulated frame: closed periods plus the open period
    /// aggregated as the (live) last row.
    pub fn frame(&self) -> Result<DataFrame> {
        let open_coarse = match &self.open {
            Some(open) => Some(aggregate_open(open, &self.spec, self.tf)?),
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
            Some(open) => Ok(Some(aggregate_open(open, &self.spec, self.tf)?)),
            None => Ok(None),
        }
    }

    /// The still-open period's fine bars (`None` if nothing is open) — lets a caller
    /// carry the folding state forward after a one-shot `cumulate`, so further fine
    /// bars fold into the same forming period.
    pub fn open_period(&self) -> Option<&OpenPeriod> {
        self.open.as_ref()
    }
}

/// One-shot resample of `df` (must have a `DatetimeIndex`) into `tf` periods.
pub fn cumulate(df: &DataFrame, tf: TimeFrame, spec: &AggSpec) -> Result<DataFrame> {
    let mut cumulator = Cumulator::new(tf, spec.clone());
    cumulator.append(df)?;
    cumulator.frame()
}

/// The open period aggregated on the grid of its own bars' zone (every bar the
/// cumulator holds went through the datetime-index check of `append`).
fn aggregate_open(open: &OpenPeriod, spec: &AggSpec, tf: TimeFrame) -> Result<DataFrame> {
    aggregate_period(open.bars(), spec, tf, open.bars().index().tz())
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
            cum.append(&df.slice_data(i, i + 1)).unwrap();
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
        cum.append(&sample().slice_data(0, 2)).unwrap();
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

    /// The open period a cumulation hands over holds one bar per timestamp: a re-sent
    /// bar inside the fed bars (its first copy superseded) is not kept, so the open
    /// period reduces to the batch aggregate of the same feed.
    #[test]
    fn the_open_period_keeps_no_superseded_bar() {
        let df = frame(
            &["2020-01-01 00:00:00", "2020-01-01 00:01:00", "2020-01-01 00:01:00"],
            &[10.0, 11.0, 12.0],
            &[12.0, 30.0, 13.0],
            &[9.0, 9.5, 9.5],
            &[11.0, 12.0, 12.5],
            &[10.0, 5.0, 7.0],
        );
        let mut cum = Cumulator::new(TimeFrame::Min5, AggSpec::ohlcv());
        cum.append(&df).unwrap();
        let open = cum.open_period().unwrap();
        assert_eq!(open.bars().height(), 2);
        assert_eq!(open.bars().column("high").unwrap().as_f64().unwrap(), &[12.0, 13.0][..]);
        let batch = cumulate(&df, TimeFrame::Min5, &AggSpec::ohlcv()).unwrap();
        assert!(cum.frame().unwrap().equals(&batch));
        // the next call re-sends 00:01 again: the result is the batch of the last-wins feed
        let resend = frame(&["2020-01-01 00:01:00"], &[12.0], &[13.5], &[9.5], &[12.9], &[8.0]);
        cum.append(&resend).unwrap();
        let out = cum.frame().unwrap();
        assert_eq!(out.column("high").unwrap().as_f64().unwrap(), &[13.5][..]);
        assert_eq!(out.column("volume").unwrap().as_f64().unwrap(), &[18.0][..]);
    }

    /// Periods closing over several calls are kept in order: each call's closed rows
    /// join the earlier ones in one append.
    #[test]
    fn closed_periods_accumulate_across_calls() {
        let df = sample();
        let mut cum = Cumulator::new(TimeFrame::Min1, AggSpec::ohlcv());
        cum.append(&df.slice_data(0, 3)).unwrap();
        cum.append(&df.slice_data(3, 6)).unwrap();
        assert!(cum.frame().unwrap().equals(&cumulate(&df, TimeFrame::Min1, &AggSpec::ohlcv()).unwrap()));
    }

    /// A rejected append leaves the cumulator exactly as it was — its open period
    /// included — whichever step rejects it, so the next append continues the period.
    #[test]
    fn a_rejected_append_changes_nothing() {
        let at = |s: &str| datetime::parse_ns(s).unwrap();
        let vol = |ts: &[i64], v: &[i64]| {
            DataFrame::new(
                vec!["volume".into()],
                vec![Column::i64(v.to_vec())],
                Some(Index::datetime(ts.to_vec(), Tz::Utc)),
            )
            .unwrap()
        };
        let mut cum = Cumulator::new(TimeFrame::Min5, AggSpec::ohlcv());
        cum.append(&vol(&[at("2020-01-01 00:00:00")], &[i64::MAX])).unwrap();
        let before = cum.frame().unwrap();
        // the closing period's sum overflows, after the open period was extended
        let overflow = vol(&[at("2020-01-01 00:01:00"), at("2020-01-01 00:05:00")], &[1, 1]);
        assert!(matches!(cum.append(&overflow).unwrap_err(), VolasError::Overflow(_)));
        // a NaT bar, judged after the open period is extended
        assert!(cum.append(&vol(&[i64::MIN], &[1])).is_err());
        assert!(cum.frame().unwrap().equals(&before));
        cum.append(&vol(&[at("2020-01-01 00:06:00")], &[2])).unwrap();
        let out = cum.frame().unwrap();
        assert_eq!(out.column("volume").unwrap().as_i64().unwrap(), &[i64::MAX, 2][..]);
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
