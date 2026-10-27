//! The open (forming) period of a cumulation: the small frame of fine bars whose
//! cumulation is the forming coarse bar.

use volas_core::{AdmittedAppend, AdmittedBar, FoldLayout,DataFrame, IndexKind, Result, VolasError};

/// The fine bars of the period still forming — the small frame a time-frame
/// frame (and a [`crate::Cumulator`]) keeps, whose cumulation is the forming coarse
/// bar. Its invariant, kept by every way one is built or changed: at least one bar,
/// a `DatetimeIndex`, and **one bar per timestamp, timestamps strictly increasing**
/// — a re-sent bar has replaced the bar it repeats, so re-cumulating the period
/// (a re-sent bar's refold) reduces exactly the bars a batch cumulation of the same
/// feed reduces, never a superseded one.
#[derive(Clone, Debug)]
pub struct OpenPeriod {
    bars: DataFrame,
    layout: Option<FoldLayout>,
}

impl OpenPeriod {
    /// The open period of already checked datetime bars. Raw re-sends enter
    /// through [`crate::RawFineInput`], before an ordinary frame is built.
    pub fn new(bars: DataFrame) -> Result<OpenPeriod> {
        let ts = timestamps(&bars)?;
        if ts.is_empty() {
            return Err(VolasError::Value("an open period needs at least one bar".into()));
        }
        Ok(OpenPeriod { bars, layout: None })
    }

    /// The bars, one per timestamp in increasing order.
    pub fn bars(&self) -> &DataFrame {
        &self.bars
    }

    /// The first bar's timestamp (epoch ns).
    pub fn first_ts(&self) -> i64 {
        stamps(&self.bars)[0]
    }

    /// The last bar's timestamp (epoch ns).
    pub fn last_ts(&self) -> i64 {
        let ts = stamps(&self.bars);
        ts[ts.len() - 1]
    }

    /// Take `bars` (checked against this period's columns, dtypes and time order): a bar stamped like the last bar replaces it whole, a
    /// later one is appended, an earlier one is an error — so the invariant holds.
    /// On an error nothing changes.
    pub fn push(&mut self, bars: &DataFrame) -> Result<()> {
        self.bars.append(bars)?;
        // append validates the incoming labels against the last held
        // label before writing. Re-scanning the history here would make a debug
        // build's live correction O(period length) despite that invariant.
        Ok(())
    }

    /// Consume a complete schema-admitted bar without checking its columns again.
    pub fn push_bar(&mut self, bar: &AdmittedBar<'_, '_>) -> Result<()> {
        self.check_bar(bar)?.land()
    }

    /// Prepare this bar's single append boundary before any displayed or
    /// checkpoint row changes. The returned plan holds the period exclusively.
    pub fn check_bar<'f, 'r>(&'f mut self, bar: &'r AdmittedBar<'_, '_>) -> Result<AdmittedAppend<'f, 'r>> {
        bar.prepare_append_cached(&mut self.bars, &mut self.layout)
    }

    /// Admit a timed bar's one index boundary and bind it to this open period.
    /// Aggregates consume its order/re-send decision instead of comparing again.
    pub fn check_timed<'f, 'r, 'a, 'p, 'z>(&'f mut self, timed: &crate::TimedBar<'r, 'a, 'p, 'z>) -> Result<OpenBar<'f, 'r, 'a, 'p, 'z>> {
        let previous = self.last_ts();
        let append = self.check_bar(timed.bar())?;
        Ok(OpenBar { append, bar: timed.bar(), previous, timestamp: timed.timestamp(), key: timed.key(), tf: timed.time_frame(), tz: timed.zone() })
    }

    /// The period with only the columns `names` (a projection keeps every bar).
    pub fn select(&self, names: &[String]) -> Result<OpenPeriod> {
        Ok(OpenPeriod { bars: self.bars.select(names)?, layout: None })
    }
}

/// The bars' timestamps, or an error when the index is not a `DatetimeIndex`.
fn timestamps(bars: &DataFrame) -> Result<&[i64]> {
    match bars.index().kind() {
        IndexKind::Datetime(v, _) => Ok(v),
        _ => Err(VolasError::Value("an open period needs a DatetimeIndex".into())),
    }
}

/// The timestamps of an open period's bars (a `DatetimeIndex` by construction).
fn stamps(bars: &DataFrame) -> &[i64] {
    match bars.index().kind() {
        IndexKind::Datetime(v, _) => v,
        _ => unreachable!("an open period has a DatetimeIndex"), // LCOV_EXCL_LINE
    }
}

#[cfg(test)]
#[path = "open_period_tests.rs"]
mod tests;

/// A timed, schema-admitted bar whose append/re-send decision is bound to the
/// exclusively borrowed open period. Only `OpenPeriod::check_timed` constructs it.
pub struct OpenBar<'f, 'r, 'a, 'p, 'z> {
    append: AdmittedAppend<'f, 'r>,
    bar: &'r AdmittedBar<'a, 'p>,
    previous: i64,
    timestamp: i64,
    key: i64,
    tf: crate::TimeFrame,
    tz: &'z volas_core::Tz,
}
impl<'a, 'p> OpenBar<'_, '_, 'a, 'p, '_> {
    /// The one Index-owner routing decision.
    pub fn replaces(&self) -> bool { self.append.replaces() }
    /// Publish the raw fine bar after the aggregate and destination accept it.
    pub fn land(self) -> Result<()> { self.append.land() }
    pub(crate) fn bar(&self) -> &AdmittedBar<'a, 'p> { self.bar }
    pub(crate) fn previous(&self) -> i64 { self.previous }
    pub(crate) fn timestamp(&self) -> i64 { self.timestamp }
    pub(crate) fn key(&self) -> i64 { self.key }
    pub(crate) fn time_frame(&self) -> crate::TimeFrame { self.tf }
    pub(crate) fn zone(&self) -> &volas_core::Tz { self.tz }
}
