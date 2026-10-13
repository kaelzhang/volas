//! The open (forming) period of a cumulation: the small frame of fine bars whose
//! cumulation is the forming coarse bar.

use volas_core::{DataFrame, IndexKind, Result, VolasError};

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
}

impl OpenPeriod {
    /// The open period of one period's fine `bars`, given in non-decreasing time
    /// order: a run of bars with the same timestamp (a re-sent bar) keeps only its
    /// last bar — the re-send rule of [`crate::cumulate()`] (consecutive equal
    /// timestamps, last wins). Errors on an empty frame, a non-datetime index or bars
    /// out of order.
    pub fn new(bars: DataFrame) -> Result<OpenPeriod> {
        let ts = timestamps(&bars)?;
        if ts.is_empty() {
            return Err(VolasError::Value("an open period needs at least one bar".into()));
        }
        if ts.windows(2).any(|w| w[1] < w[0]) {
            return Err(VolasError::Value(
                "an open period's bars must be in time order".into(),
            ));
        }
        let kept = dedup_keep_last(ts);
        let bars = if kept.len() == ts.len() { bars } else { bars.take(&kept) };
        Ok(OpenPeriod { bars })
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

    /// Take `bars` (already admitted against a frame with this period's columns and
    /// dtypes, in time order): a bar stamped like the last bar replaces it whole, a
    /// later one is appended, an earlier one is an error — so the invariant holds.
    /// On an error nothing changes.
    pub fn push(&mut self, bars: &DataFrame) -> Result<()> {
        self.bars.append_admitted(bars)?;
        // append_admitted validates the incoming labels against the last held
        // label before writing. Re-scanning the history here would make a debug
        // build's live correction O(period length) despite that invariant.
        Ok(())
    }

    /// The period with only the columns `names` (a projection keeps every bar).
    pub fn select(&self, names: &[String]) -> Result<OpenPeriod> {
        Ok(OpenPeriod { bars: self.bars.select(names)? })
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

/// Positions to keep when consecutive same-timestamp rows are deduped (keep the
/// last of each run — a re-sent / corrected tick updates rather than accumulates).
/// The one re-send rule of cumulation: the batch aggregate and the open period both
/// reduce exactly these rows.
pub(crate) fn dedup_keep_last(ts: &[i64]) -> Vec<usize> {
    let n = ts.len();
    (0..n)
        .filter(|&i| i + 1 == n || ts[i + 1] != ts[i])
        .collect()
}

#[cfg(test)]
#[path = "open_period_tests.rs"]
mod tests;
