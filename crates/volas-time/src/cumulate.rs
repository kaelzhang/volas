//! Cumulation: group fine bars into coarser periods and aggregate.

use std::borrow::Cow;

use volas_core::{DataFrame, Index, IndexKind, Result, Tz, VolasError};

use crate::agg::AggSpec;
use crate::open_period::{dedup_keep_last, OpenPeriod};
use crate::PeriodAggregate;
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
    /// Absent only before any bar or while the open period cannot aggregate (an
    /// integer overflow is reported by frame/last, or when the period closes).
    aggregate: Option<PeriodAggregate>,
}

impl Cumulator {
    /// A fresh cumulator for `tf` with aggregation `spec`.
    pub fn new(tf: TimeFrame, spec: AggSpec) -> Self {
        Cumulator {
            tf,
            spec,
            closed: None,
            open: None,
            aggregate: None,
        }
    }

    /// Construct and ingest an initial batch, reserving its closed output for
    /// consuming materialization and the first live append. Validation and deferred
    /// open-period errors for non-empty inputs are the same as `new` followed by
    /// `append`. An empty datetime batch keeps its schema without an open period.
    pub fn from_frame(fine: &DataFrame, tf: TimeFrame, spec: AggSpec) -> Result<Self> {
        let mut cumulator = Self::new(tf, spec);
        if fine.height() == 0 {
            // A one-shot empty result still owns its schema and datetime grid. Live
            // append keeps its non-empty contract; no open period exists to fold yet.
            cumulator.spec.check(fine)?;
            if !matches!(fine.index().kind(), IndexKind::Datetime(..)) {
                return Err(VolasError::Value("cumulation target must have a DatetimeIndex".into()));
            }
            cumulator.closed = Some(fine.slice_data(0, 0));
        } else {
            cumulator.append_inner(fine, true)?;
        }
        Ok(cumulator)
    }

    /// Feed fine bars (must have a `DatetimeIndex`). Periods that close are
    /// finalized; the trailing period stays open for the next call. All or nothing:
    /// the closed rows and the next open period are built aside, and the cumulator
    /// takes them only once every step has succeeded, so on an error it is unchanged.
    pub fn append(&mut self, fine: &DataFrame) -> Result<()> {
        self.append_inner(fine, false)
    }

    fn append_inner(&mut self, fine: &DataFrame, materializing: bool) -> Result<()> {
        if fine.height() == 0 {
            return Err(VolasError::Value(
                "the data frame to be appended is empty".into(),
            ));
        }
        self.spec.check(fine)?;

        if self.append_one(fine)? {
            return Ok(());
        }

        // raw = the still-open period's bars + the incoming bars.
        let raw = match &self.open {
            Some(open) => {
                let mut raw = open.bars().clone();
                raw.append(fine)?;
                Cow::Owned(raw)
            }
            None => match &self.closed {
                // An empty seed has no open bars, but owns the schema and grid.
                // Normal frame admission converts the incoming zone and dtypes
                // before period boundaries are computed.
                Some(empty) => {
                    let mut raw = empty.clone();
                    raw.append(fine)?;
                    Cow::Owned(raw)
                }
                None => Cow::Borrowed(fine),
            },
        };

        let (ts, tz): (&[i64], &Tz) = match raw.index().kind() {
            IndexKind::Datetime(v, tz) => (v, tz),
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

        let runs = PeriodRuns::new(ts, self.tf, tz)?;
        let last = runs.keys.len() - 1;

        // Every run except the last closes: its coarse rows, gathered aside.
        let closing = if last == 0 { None } else { Some(runs.aggregate(&raw, &self.spec, last, tz, materializing)?) };
        // The last run remains open (re-sent bars collapsed to the last one).
        let a = runs.starts[last];
        let open = OpenPeriod::new(raw.slice_data(a, raw.height()))?;
        let aggregate = PeriodAggregate::new(&open, &self.spec, self.tf, tz).ok();

        // Commit: one atomic append of the closed rows, then the open period.
        if let Some(rows) = closing {
            match &mut self.closed {
                Some(closed) => closed.append(&rows)?,
                None => self.closed = Some(rows),
            }
        }
        self.open = Some(open);
        self.aggregate = aggregate;
        Ok(())
    }

    /// The live path never clones or scans the raw open period. Invalid / bulk
    /// feeds take the general transactional path, retaining its error ordering.
    fn append_one(&mut self, fine: &DataFrame) -> Result<bool> {
        let (Some(open), Some(aggregate)) = (&mut self.open, &mut self.aggregate) else { return Ok(false) };
        let IndexKind::Datetime(ts, _) = fine.index().kind() else { return Ok(false) };
        if ts.len() != 1 || ts[0] < open.last_ts() { return Ok(false); }
        let bars = open.bars().admit_data_rows(fine)?;
        let tz = open.bars().index().tz();
        let key = self.tf.period_start(ts[0], &tz)?;
        let open_key = match aggregate.current().index().kind() {
            IndexKind::Datetime(keys, _) => keys[0],
            _ => unreachable!("an aggregate has a datetime index"), // LCOV_EXCL_LINE
        };
        if key != open_key {
            let index = Index::datetime(ts.to_vec(), tz.clone()).with_name(open.bars().index().name().map(String::from));
            let columns = bars.columns().iter().map(|c| c.slice(0, 1)).collect();
            let next = OpenPeriod::new(DataFrame::new(bars.names().to_vec(), columns, Some(index))?)?;
            let next_aggregate = PeriodAggregate::new(&next, &self.spec, self.tf, &tz).ok();
            match &mut self.closed {
                Some(closed) => closed.append(aggregate.current())?,
                None => self.closed = Some(aggregate.current().clone()),
            }
            self.open = Some(next);
            self.aggregate = next_aggregate;
        } else {
            let resent = ts[0] == open.last_ts();
            open.push(&bars)?;
            // append has always deferred errors in the still-open aggregate.
            if aggregate.push(&bars, resent).is_err() { self.aggregate = None; }
        }
        Ok(true)
    }

    /// The current cumulated frame: closed periods plus the open period
    /// aggregated as the (live) last row.
    pub fn frame(&self) -> Result<DataFrame> {
        let open_coarse = self.last()?;
        match (&self.closed, open_coarse) {
            (Some(closed), Some(open)) => {
                let mut out = closed.clone();
                out.append(&open)?;
                Ok(out)
            }
            (Some(closed), None) => Ok(closed.clone()),
            (None, Some(open)) => Ok(open),
            (None, None) => DataFrame::new(Vec::new(), Vec::new(), None),
        }
    }

    /// Consume the output buffers, retaining the raw open bars and running aggregate
    /// for a caller that will continue the live series. Moves closed buffers without
    /// snapshot COW; `from_frame` reserves room for the forming row and live append.
    /// Errors from the open period retain `last`'s timing.
    pub fn into_parts(self) -> Result<(DataFrame, Option<OpenPeriod>, Option<PeriodAggregate>)> {
        let open_coarse = self.last()?;
        let Self { closed, open, aggregate, .. } = self;
        let frame = match (closed, open_coarse) {
            (Some(mut closed), Some(coarse)) => { closed.append(&coarse)?; closed }
            (Some(closed), None) => closed,
            (None, Some(coarse)) => coarse,
            (None, None) => DataFrame::new(Vec::new(), Vec::new(), None)?,
        };
        Ok((frame, open, aggregate))
    }

    /// Consume a one-shot cumulator without snapshotting its closed history.
    pub fn into_frame(self) -> Result<DataFrame> {
        self.into_parts().map(|(frame, _, _)| frame)
    }

    /// The current open period aggregated into a single (live) coarse bar, or
    /// `None` if nothing is open. O(columns) for a valid open aggregate.
    pub fn last(&self) -> Result<Option<DataFrame>> {
        if let Some(aggregate) = &self.aggregate { return Ok(Some(aggregate.current().clone())); }
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

    /// The running state carried with the open period into a live frame.
    pub fn period_aggregate(&self) -> Option<&PeriodAggregate> {
        self.aggregate.as_ref()
    }
}

/// One-shot resample of `df` (must have a `DatetimeIndex`) into `tf` periods.
pub fn cumulate(df: &DataFrame, tf: TimeFrame, spec: &AggSpec) -> Result<DataFrame> {
    Cumulator::from_frame(df, tf, spec.clone())?.into_frame()
}

/// The open period aggregated on the grid of its own bars' zone (every bar the
/// cumulator holds went through the datetime-index check of `append`).
fn aggregate_open(open: &OpenPeriod, spec: &AggSpec, tf: TimeFrame) -> Result<DataFrame> {
    aggregate_period(open.bars(), spec, tf, &open.bars().index().tz())
}

/// Split a sorted timestamp slice into `[start, end)` runs of the same period
/// (equal [`TimeFrame::period_start`] on `tz`'s local grid).
struct PeriodRuns {
    keys: Vec<i64>,
    starts: Vec<usize>,
    kept: Vec<usize>,
    offsets: Vec<usize>,
}

impl PeriodRuns {
    fn new(ts: &[i64], tf: TimeFrame, tz: &Tz) -> Result<Self> {
        let mut runs = Self { keys: Vec::new(), starts: Vec::new(), kept: Vec::with_capacity(ts.len()), offsets: Vec::new() };
        let mut cursor = tf.period_cursor(tz);
        for (i, &stamp) in ts.iter().enumerate() {
            let key = cursor.period_start(stamp)?;
            if runs.keys.last() != Some(&key) {
                runs.keys.push(key);
                runs.starts.push(i);
                runs.offsets.push(runs.kept.len());
            }
            if i + 1 == ts.len() || ts[i + 1] != stamp {
                runs.kept.push(i);
            }
        }
        runs.offsets.push(runs.kept.len());
        Ok(runs)
    }

    fn aggregate(&self, raw: &DataFrame, spec: &AggSpec, count: usize, tz: &Tz, materializing: bool) -> Result<DataFrame> {
        let mut columns = Vec::with_capacity(raw.width());
        let mut error: Option<(usize, VolasError)> = None;
        for (name, col) in raw.names().iter().zip(raw.columns()) {
            // Columns fold independently, but errors still belong to the earliest
            // period, then the earliest column of that period, just as before.
            let end = error.as_ref().map_or(count, |(row, _)| row + 1);
            let groups = self.offsets[..=end].windows(2).map(|w| &self.kept[w[0]..w[1]]);
            // Only the initial materializing batch retains the snapshot output's
            // growth capacity. Ordinary incremental builders keep their old budget.
            // The maximum open string cell bounds the forming endpoint's bytes.
            let spare = if materializing { count } else { 0 };
            let bytes = match col {
                volas_core::Column::Str(v, _) if materializing => (self.starts[count]..raw.height()).map(|i| v.get(i).len()).max().unwrap_or(0),
                _ => 0,
            };
            match col.reduce_groups_with_spare(spec.agg_for(name).as_combine_op(), groups, spare, bytes) {
                Ok(col) => columns.push(col),
                Err(e) if error.as_ref().is_none_or(|(row, _)| e.0 < *row) => error = Some(e),
                Err(_) => {}
            }
        }
        if let Some((_, error)) = error { return Err(error); }
        let capacity = if materializing { (count * 2).max(4) } else { count };
        let mut keys = Vec::with_capacity(capacity);
        keys.extend_from_slice(&self.keys[..count]);
        let index = Index::datetime(keys, tz.clone()).with_name(raw.index().name().map(String::from));
        DataFrame::new(raw.names().to_vec(), columns, Some(index))
    }
}

/// Aggregate one period's raw bars into a single coarse row (1-row frame) on the
/// period grid of `tz` — the zone that owns the grid (a tf frame's own zone, never
/// an incoming bar's). The index label is the period's grid **start**,
/// [`TimeFrame::period_start`] of the first raw timestamp (owner decision
/// 2026-06-12: every TimeFrame has a fixed grid origin and a bar's label is
/// origin + n·delta, e.g. a 15-minute bar labels :00/:15/:30/:45 — never the first
/// raw timestamp, which can sit anywhere inside the period), in `tz`.
pub fn aggregate_period(period: &DataFrame, spec: &AggSpec, tf: TimeFrame, tz: &Tz) -> Result<DataFrame> {
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
    if ts.contains(&i64::MIN) {
        return Err(VolasError::Value("cumulation period cannot contain NaT".into()));
    }
    if ts.windows(2).any(|w| w[0] > w[1]) {
        return Err(VolasError::Value("cumulation period bars must be ordered".into()));
    }
    let kept = dedup_keep_last(ts);
    let mut cursor = tf.period_cursor(tz);
    let start_ns = cursor.period_start(ts[0])?;
    for &ns in &ts[1..] {
        if cursor.period_start(ns)? != start_ns {
            return Err(VolasError::Value("cumulation period contains multiple periods".into()));
        }
    }

    let names = period.names().to_vec();
    let mut columns = Vec::with_capacity(names.len());
    for (name, col) in names.iter().zip(period.columns()) {
        columns.push(spec.agg_for(name).reduce(col, &kept)?);
    }
    // The aggregated index keeps the source index's name (pandas resample parity).
    let index = Index::datetime(vec![start_ns], tz.clone()).with_name(period.index().name().map(String::from));
    DataFrame::new(names, columns, Some(index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use volas_core::{datetime, Column};

    #[test]
    fn consumption_keeps_closed_buffers_and_snapshot_isolation() {
        let fine = DataFrame::new(vec!["volume".into()], vec![Column::f64(vec![1., 2., 3.])],
            Some(Index::datetime(vec![0, 60_000_000_000, 120_000_000_000], Tz::Utc))).unwrap();
        let cum = Cumulator::from_frame(&fine, TimeFrame::Min1, AggSpec::ohlcv()).unwrap();
        let closed = cum.closed.as_ref().unwrap();
        let values = closed.columns()[0].as_f64().unwrap().as_ptr();
        let labels = match closed.index().kind() { IndexKind::Datetime(v, _) => v.as_ptr(), _ => unreachable!() };
        let snapshot = cum.frame().unwrap();
        let (out, open, aggregate) = cum.into_parts().unwrap();
        assert_eq!(values, out.columns()[0].as_f64().unwrap().as_ptr());
        match out.index().kind() { IndexKind::Datetime(v, _) => assert_eq!(labels, v.as_ptr()), _ => unreachable!() }
        assert!(out.equals(&snapshot));
        assert_eq!(open.unwrap().bars().height(), 1);
        assert!(aggregate.is_some());
        let mut changed = out;
        let next = DataFrame::new(vec!["volume".into()], vec![Column::f64(vec![4.])],
            Some(Index::datetime(vec![180_000_000_000], Tz::Utc))).unwrap();
        changed.append(&next).unwrap();
        assert!(snapshot.equals(&fine));
        assert_eq!(changed.height(), 4);

        let empty = Cumulator::new(TimeFrame::Min1, AggSpec::ohlcv()).into_parts().unwrap();
        assert_eq!(empty.0.height(), 0);
        assert!(empty.1.is_none() && empty.2.is_none());
        let closed_only = Cumulator { closed: Some(fine.clone()), ..Cumulator::new(TimeFrame::Min1, AggSpec::ohlcv()) };
        assert!(closed_only.into_frame().unwrap().equals(&fine));
    }

    #[test]
    fn empty_batch_keeps_its_schema_and_grid_without_an_open_period() {
        let empty = DataFrame::new(vec!["volume".into()], vec![Column::i64(vec![])],
            Some(Index::datetime(vec![], Tz::Utc).with_name(Some("time".into())))).unwrap();
        let mut cum = Cumulator::from_frame(&empty, TimeFrame::Min5, AggSpec::ohlcv()).unwrap();
        assert!(cum.frame().unwrap().equals(&empty));
        assert!(cum.last().unwrap().is_none());
        assert!(cum.open_period().is_none());
        assert!(cum.period_aggregate().is_none());
        assert!(cum.append(&empty).is_err());
        let next = DataFrame::new(vec!["volume".into()], vec![Column::i64(vec![2])],
            Some(Index::datetime(vec![0], Tz::Utc).with_name(Some("time".into())))).unwrap();
        cum.append(&next).unwrap();
        assert!(cum.period_aggregate().unwrap().current().equals(&next));
        // The public snapshot accessor and consuming output expose the same revised
        // forming value, without adding a re-sent timestamp to the period's sum.
        let revised = DataFrame::new(vec!["volume".into()], vec![Column::i64(vec![3])],
            Some(Index::datetime(vec![0], Tz::Utc).with_name(Some("time".into())))).unwrap();
        cum.append(&revised).unwrap();
        assert!(cum.period_aggregate().unwrap().current().equals(&revised));
        assert!(cum.into_frame().unwrap().equals(&revised));
        assert!(cumulate(&empty, TimeFrame::Min5, &AggSpec::ohlcv()).unwrap().equals(&empty));
        let plain = DataFrame::new(vec!["volume".into()], vec![Column::i64(vec![])], None).unwrap();
        assert!(Cumulator::from_frame(&plain, TimeFrame::Min5, AggSpec::ohlcv()).is_err());
        let text = DataFrame::new(vec!["volume".into()], vec![Column::str(vec![])],
            Some(Index::datetime(vec![], Tz::Utc))).unwrap();
        assert!(Cumulator::from_frame(&text, TimeFrame::Min5, AggSpec::ohlcv()).is_err());
    }

    #[test]
    fn an_empty_cumulator_uses_its_seed_timezone_for_the_first_bucket() {
        let empty = DataFrame::new(vec!["volume".into()], vec![Column::i64(vec![])],
            Some(Index::datetime(vec![], Tz::Utc))).unwrap();
        let mut cum = Cumulator::from_frame(&empty, TimeFrame::Hour1, AggSpec::ohlcv()).unwrap();
        let next = DataFrame::new(vec!["volume".into()], vec![Column::i64(vec![2])],
            Some(Index::datetime(vec![600_000_000_000], Tz::fixed(1800).unwrap()))).unwrap();
        cum.append(&next).unwrap();
        let out = cum.into_frame().unwrap();
        match out.index().kind() {
            IndexKind::Datetime(stamps, tz) => {
                assert_eq!(stamps, &[0]);
                assert_eq!(tz, &Tz::Utc);
            }
            _ => panic!("expected datetime index"), // LCOV_EXCL_LINE
        }
    }

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
        assert!(aggregate_period(&df, &AggSpec::ohlcv(), TimeFrame::Min5, &Tz::Utc).is_err());
        // an empty period has no bar to label: an error, not an index panic
        let empty = frame(&[], &[], &[], &[], &[], &[]);
        let err = aggregate_period(&empty, &AggSpec::ohlcv(), TimeFrame::Min5, &Tz::Utc).unwrap_err();
        assert!(err.to_string().contains("no bars"), "{err}");
    }

    /// The period grid belongs to the zone passed in, not to the bars' own index
    /// zone: a UTC-indexed bar aggregated on a +08:00 grid starts at the +08:00
    /// local midnight and carries that zone.
    #[test]
    fn aggregate_period_uses_the_grid_zone() {
        let bar = frame(&["2024-01-02 20:00:00"], &[1.0], &[1.0], &[1.0], &[1.0], &[1.0]);
        let sh = Tz::parse("+08:00").unwrap();
        let out = aggregate_period(&bar, &AggSpec::ohlcv(), TimeFrame::Day1, &sh).unwrap();
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
