//! The running aggregate and its before-last checkpoint. Only the latest fine
//! bar can be revised, so even min/max need just this checkpoint, not a tree.

use std::borrow::Cow;
use volas_core::{CombineOp, DataFrame, IndexKind, Result, Tz, VolasError};

use crate::{aggregate_period, AggSpec, OpenPeriod, TimeFrame};

/// The aggregate of an open period, plus the aggregate excluding its latest bar.
/// Kept separately from the raw [`OpenPeriod`] value: writes or schema changes
/// rebuild this state from that value; ordinary appends and re-sends reuse it.
#[derive(Clone)]
pub struct PeriodAggregate {
    current: DataFrame,
    before_last: DataFrame,
    has_before: bool,
    ops: Vec<(usize, usize, CombineOp)>,
    missing: Vec<bool>,
    before_missing: Vec<bool>,
    tf: TimeFrame,
    last_ts: i64,
}

impl PeriodAggregate {
    /// Build the checkpoints once from the period's distinct, ordered fine bars.
    pub fn new(open: &OpenPeriod, spec: &AggSpec, tf: TimeFrame, tz: &Tz) -> Result<Self> {
        let bars = open.bars();
        let current = aggregate_period(bars, spec, tf, tz)?;
        let has_before = bars.height() > 1;
        let before_last = if has_before {
            let kept: Vec<_> = (0..bars.height() - 1).collect();
            let columns = bars.names().iter().zip(bars.columns())
                .map(|(name, col)| spec.agg_for(name).reduce(col, &kept)).collect::<Result<_>>()?;
            DataFrame::new(bars.names().to_vec(), columns, Some((**current.index()).clone()))?
        } else {
            current.clone()
        };
        let ops = bars.names().iter().enumerate().map(|(i, name)| (i, i, spec.agg_for(name).as_combine_op())).collect();
        let before_missing: Vec<_> = bars.columns().iter().map(|c| (0..bars.height() - 1).any(|i| c.is_missing_input(i))).collect();
        let missing = bars.columns().iter().zip(&before_missing).map(|(c, &before)| before || c.is_missing_input(bars.height() - 1)).collect();
        Ok(Self { current, before_last, has_before, ops, missing, before_missing, tf, last_ts: open.last_ts() })
    }

    /// The aggregate row, labelled on the frame's grid.
    pub fn current(&self) -> &DataFrame {
        &self.current
    }

    /// The checkpoint a proposed push continues, without changing either one.
    /// A destination frame validates this prefix and the incoming bar together
    /// before the push is committed; a first-bar correction has no prefix.
    pub fn prefix(&self, resent: bool) -> Option<&DataFrame> {
        if !resent { Some(&self.current) }
        else if self.has_before { Some(&self.before_last) }
        else { None }
    }

    /// Admit one bar for this period and bind its eventual push to this exact
    /// aggregate. A re-send must name the latest bar; a fresh bar must follow it.
    pub fn check_push<'a, 'r>(&'a mut self, bar: &'r DataFrame, resent: bool) -> Result<PeriodPush<'a, 'r>> {
        let bars = self.admit_bar(bar)?;
        let timestamp = timestamp_at(&bars);
        let (key, tz) = self.grid();
        if self.tf.period_start(timestamp, tz)? != key
            || (resent && timestamp != self.last_ts) || (!resent && timestamp <= self.last_ts) {
            return Err(VolasError::Value("period aggregate push requires an increasing bar in the same period, or a re-send of its latest timestamp".into()));
        }
        Ok(PeriodPush { aggregate: self, bars, resent, timestamp })
    }

    /// Reduce the first bar of a later period once, then bind publication to this
    /// aggregate. The caller may append `current()` before landing the restart.
    pub fn check_restart<'a, 'r>(&'a mut self, bar: &'r DataFrame, spec: &AggSpec) -> Result<PeriodRestart<'a, 'r>> {
        let bars = self.admit_bar(bar)?;
        if self.current.names().iter().zip(&self.ops).any(|(name, &(_, _, op))| spec.agg_for(name).as_combine_op() != op) {
            return Err(VolasError::Value("period aggregate restart must keep its aggregation spec".into()));
        }
        let (_, tz) = self.grid();
        let current = aggregate_period(&bars, spec, self.tf, tz)?;
        if timestamp_at(&current) <= timestamp_at(&self.current) {
            return Err(VolasError::Value("period aggregate restart requires a later period".into()));
        }
        Ok(PeriodRestart { aggregate: self, current, bars })
    }

    fn grid(&self) -> (i64, &Tz) {
        match self.current.index().kind() {
            IndexKind::Datetime(ts, tz) => (ts[0], tz),
            _ => unreachable!("a period aggregate has a datetime grid"), // LCOV_EXCL_LINE
        }
    }

    fn admit_bar<'r>(&self, bar: &'r DataFrame) -> Result<Cow<'r, DataFrame>> {
        if bar.height() != 1 {
            return Err(VolasError::Shape("a period aggregate takes exactly one fine bar".into()));
        }
        self.current.admit_data_rows(bar)
    }

    /// Start a new period from its already reduced first bar. The admitted bar
    /// has the same schema, so keep the fold plan and checkpoint buffers. The
    /// previous prefix becomes relevant only after the next distinct bar, when
    /// `push` overwrites it with this new period's first aggregate.
    fn restart(&mut self, current: DataFrame, bar: &DataFrame) {
        self.current = current;
        self.has_before = false;
        self.last_ts = timestamp_at(bar);
        for (missing, col) in self.missing.iter_mut().zip(bar.columns()) { *missing = col.is_missing_input(0); }
    }

    /// Take one admitted complete bar of this period. Validation precedes every
    /// write, including the checkpoint, so an overflowing correction changes
    /// neither aggregate. Work is O(columns), with buffers reused after warm-up.
    pub(crate) fn push_admitted(&mut self, bar: &DataFrame, resent: bool, timestamp: i64) -> Result<()> {
        if resent && !self.has_before {
            self.current.refold_forming_row(0, bar, &self.ops)?;
            for (missing, col) in self.missing.iter_mut().zip(bar.columns()) { *missing = col.is_missing_input(0); }
            self.last_ts = timestamp;
            return Ok(());
        }
        let missing = if resent { &self.before_missing } else { &self.missing };
        self.current.fold_with_checkpoint(&mut self.before_last, bar, &self.ops, missing, resent)?;
        if resent {
            self.missing.copy_from_slice(&self.before_missing);
        } else {
            self.before_missing.copy_from_slice(&self.missing);
            self.has_before = true;
        }
        for (missing, col) in self.missing.iter_mut().zip(bar.columns()) { *missing |= col.is_missing_input(0); }
        self.last_ts = timestamp;
        Ok(())
    }
}

/// A schema- and time-checked push, tied to its aggregate until publication.
pub struct PeriodPush<'a, 'r> {
    aggregate: &'a mut PeriodAggregate,
    bars: Cow<'r, DataFrame>,
    resent: bool,
    timestamp: i64,
}
impl PeriodPush<'_, '_> {
    /// Publish the admitted bar. Arithmetic failures leave the aggregate unchanged.
    pub fn land(self) -> Result<()> { self.aggregate.push_admitted(&self.bars, self.resent, self.timestamp) }
}

/// A checked restart whose reduced row is produced only by the aggregation owner.
pub struct PeriodRestart<'a, 'r> {
    aggregate: &'a mut PeriodAggregate,
    current: DataFrame,
    bars: Cow<'r, DataFrame>,
}
impl PeriodRestart<'_, '_> {
    /// The admitted coarse row before either frame or aggregate is changed.
    pub fn current(&self) -> &DataFrame { &self.current }
    /// Publish the restart after its checked row has been accepted by the caller.
    pub fn land(self) { self.aggregate.restart(self.current, &self.bars); }
}

fn timestamp_at(frame: &DataFrame) -> i64 {
    match frame.index().kind() {
        IndexKind::Datetime(ts, _) => ts[0],
        _ => unreachable!("a period aggregate admits datetime bars only"), // LCOV_EXCL_LINE
    }
}
