//! The running aggregate and its before-last checkpoint. Only the latest fine
//! bar can be revised, so even min/max need just this checkpoint, not a tree.

use std::borrow::Cow;
use volas_core::{AdmittedBar, FoldLayout,CombineOp, DataFrame, IndexKind, Result, Tz, VolasError};

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
    layouts: Option<(FoldLayout, FoldLayout)>,
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
        Ok(Self { current, before_last, has_before, ops, missing, before_missing, tf, last_ts: open.last_ts(), layouts: None })
    }

    /// Publish a first period row and retain its aggregate without admitting or
    /// reducing the same fine bar a second time.
    pub fn start(target: &mut DataFrame, timed: &crate::TimedBar<'_, '_, '_, '_>) -> Result<Self> {
        let current = timed.start(target)?;
        Ok(Self::from_current(current, timed))
    }

    pub(crate) fn from_bar(timed: &crate::TimedBar<'_, '_, '_, '_>, name: Option<String>) -> Result<Self> {
        Ok(Self::from_current(timed.reduced_named(name)?.into_frame(), timed))
    }

    fn from_current(current: DataFrame, timed: &crate::TimedBar<'_, '_, '_, '_>) -> Self {
        let bar = timed.bar();
        let before_last = current.clone();
        let ops = (0..bar.rows().width()).map(|i| (i, i, bar.operation(i).expect("an admitted bar covers its complete schema"))).collect();
        let missing = bar.rows().columns().iter().map(|c| c.is_missing_input(0)).collect();
        Self { current, before_last, has_before: false, ops, missing,
            before_missing: vec![false; bar.rows().width()], tf: timed.time_frame(), last_ts: timed.timestamp(), layouts: None }
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

    /// Start a later period on the same grid and aggregation spec. The source
    /// and the newly reduced destination row both retain the schema proof.
    pub fn restart_bar(&mut self, target: &mut DataFrame, timed: &crate::TimedBar<'_, '_, '_, '_>) -> Result<()> {
        if timed.time_frame() != self.tf || timed.zone() != self.grid().1 || timed.key() <= self.grid().0 {
            return Err(VolasError::Value("period aggregate restart requires a later period on the same grid".into()));
        }
        if !self.layouts.as_ref().is_some_and(|(layout, _)| timed.bar().accepts(layout)) {
            Self::check_operations(&self.ops, timed.bar())?;
        }
        let current = timed.start(target)?;
        self.restart(current, timed.bar().rows());
        Ok(())
    }

    /// Consume a schema-admitted bar. The period key was computed by the time
    /// owner once; cached projected layouts inherit the same source proof.
    pub fn push_bar(&mut self, timed: &crate::OpenBar<'_, '_, '_, '_, '_>) -> Result<()> {
        self.check_open_bar(timed)?;
        self.push_prepared_bar(timed, None)
    }

    fn check_open_bar(&self, timed: &crate::OpenBar<'_, '_, '_, '_, '_>) -> Result<()> {
        if timed.time_frame() != self.tf || timed.zone() != self.grid().1 || timed.key() != self.grid().0 || timed.previous() != self.last_ts {
            return Err(VolasError::Value("period aggregate push requires an increasing bar in the same period, or a re-send of its latest timestamp".into()));
        }
        Ok(())
    }

    // Both public entries have checked time; this binds the snapshots once.
    fn push_prepared_bar(&mut self, timed: &crate::OpenBar<'_, '_, '_, '_, '_>, target: Option<&mut DataFrame>) -> Result<()> {
        let bar = timed.bar();
        let timestamp = timed.timestamp();
        let resent = timed.replaces();
        let missing = if resent { &self.before_missing } else { &self.missing };
        let ops = &self.ops;
        let plan = bar.prepare_checkpoint(&mut self.current, &mut self.before_last, &mut self.layouts, missing, resent, || Self::check_operations(ops, bar))?;
        match target {
            Some(target) => plan.refold(target, self.has_before)?,
            None => plan.land(self.has_before)?,
        }
        if resent && !self.has_before {
            for (missing, col) in self.missing.iter_mut().zip(bar.rows().columns()) { *missing = col.is_missing_input(0); }
        } else {
            if resent { self.missing.copy_from_slice(&self.before_missing); }
            else { self.before_missing.copy_from_slice(&self.missing); self.has_before = true; }
            for (missing, col) in self.missing.iter_mut().zip(bar.rows().columns()) { *missing |= col.is_missing_input(0); }
        }
        self.last_ts = timestamp;
        Ok(())
    }

    /// Validate a proposed aggregate against its displayed destination before
    /// any checkpoint changes, then publish without repeating position checks.
    pub fn refold_bar(&mut self, target: &mut DataFrame, timed: &crate::OpenBar<'_, '_, '_, '_, '_>) -> Result<()> {
        self.check_open_bar(timed)?;
        self.push_prepared_bar(timed, Some(target))
    }

    fn check_operations(ops: &[(usize, usize, CombineOp)], bar: &AdmittedBar<'_, '_>) -> Result<()> {
        if ops.iter().any(|&(_, from, op)| bar.operation(from) != Some(op)) {
            return Err(VolasError::Value("period aggregate bar must keep its aggregation spec".into()));
        }
        Ok(())
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
        self.layouts = None;
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
