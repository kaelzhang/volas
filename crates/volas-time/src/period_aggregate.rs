//! The running aggregate and its before-last checkpoint. Only the latest fine
//! bar can be revised, so even min/max need just this checkpoint, not a tree.

use volas_core::{CombineOp, DataFrame, Result, Tz};

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
        Ok(Self { current, before_last, has_before, ops, missing, before_missing })
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

    /// Start a new period from its already reduced first bar. The admitted bar
    /// has the same schema, so keep the fold plan and checkpoint buffers. The
    /// previous prefix becomes relevant only after the next distinct bar, when
    /// `push` overwrites it with this new period's first aggregate.
    pub fn restart(&mut self, current: DataFrame, bar: &DataFrame) {
        self.current = current;
        self.has_before = false;
        for (missing, col) in self.missing.iter_mut().zip(bar.columns()) { *missing = col.is_missing_input(0); }
    }

    /// Take one admitted complete bar of this period. Validation precedes every
    /// write, including the checkpoint, so an overflowing correction changes
    /// neither aggregate. Work is O(columns), with buffers reused after warm-up.
    pub fn push(&mut self, bar: &DataFrame, resent: bool) -> Result<()> {
        if resent && !self.has_before {
            self.current.refold_forming_row(0, bar, &self.ops)?;
            for (missing, col) in self.missing.iter_mut().zip(bar.columns()) { *missing = col.is_missing_input(0); }
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
        Ok(())
    }
}
