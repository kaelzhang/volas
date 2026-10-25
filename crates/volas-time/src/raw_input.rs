//! The raw fine-bar admission boundary: consecutive re-sends keep their last row.

use volas_core::{Column, DataFrame, Index, IndexKind, Result, Tz, VolasError};

/// Validated raw fine input for cumulation.
///
/// Ordinary frames always have unique, strictly increasing datetime labels. Raw
/// exchange feeds may repeat their latest timestamp: this entry accepts a
/// non-decreasing timeline and keeps the last row of each equal-timestamp run
/// before building a frame. It never sorts, and rejects missing timestamps.
#[derive(Clone, Debug)]
pub struct RawFineInput {
    pub(crate) frame: DataFrame,
}

impl RawFineInput {
    /// Admit raw columns and UTC epoch-nanosecond timestamps, retaining `tz` and
    /// `index_name` as index metadata. Every column must have one value per raw
    /// timestamp, including rows superseded by a re-send.
    pub fn new(
        names: Vec<String>,
        columns: Vec<Column>,
        timestamps: Vec<i64>,
        tz: Tz,
        index_name: Option<String>,
    ) -> Result<Self> {
        if timestamps.contains(&i64::MIN) {
            return Err(VolasError::Value(
                "cannot cumulate over a DatetimeIndex containing NaT; drop or fill the missing index timestamps first".into(),
            ));
        }
        if timestamps.windows(2).any(|w| w[1] < w[0]) {
            return Err(VolasError::Value(
                "cannot cumulate over a non-monotonic DatetimeIndex (bars out of order); sort the input by time first".into(),
            ));
        }
        // Validate the raw schema and lengths before selecting: malformed input
        // must raise even when the affected row would have been superseded.
        let raw = DataFrame::new(names, columns, Some(Index::new(None, IndexKind::Range { start: 0, len: timestamps.len() })?))?;
        let kept: Vec<usize> = (0..timestamps.len())
            .filter(|&i| i + 1 == timestamps.len() || timestamps[i + 1] != timestamps[i])
            .collect();
        let (columns, timestamps) = if kept.len() == timestamps.len() {
            (raw.columns().to_vec(), timestamps)
        } else {
            (
                raw.columns().iter().map(|c| c.take(&kept)).collect(),
                kept.iter().map(|&i| timestamps[i]).collect(),
            )
        };
        let index = Index::new(index_name, IndexKind::Datetime(timestamps, tz))?;
        Ok(Self { frame: DataFrame::new(raw.names().to_vec(), columns, Some(index))? })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AggSpec, Cumulator, TimeFrame};

    fn raw(ts: Vec<i64>, values: Vec<i64>) -> Result<RawFineInput> {
        RawFineInput::new(vec!["volume".into()], vec![Column::i64(values)], ts, Tz::Utc, Some("time".into()))
    }

    #[test]
    fn raw_resends_are_last_wins_before_frame_construction() {
        let input = raw(vec![0, 0, 60_000_000_000], vec![100, 7, 2]).unwrap();
        let mut cum = Cumulator::from_raw(&input, TimeFrame::Min5, AggSpec::ohlcv()).unwrap();
        assert_eq!(cum.last().unwrap().unwrap().columns()[0].as_i64().unwrap(), &[9]);
        assert_eq!(cum.open_period().unwrap().bars().height(), 2);
        cum.append_raw(&raw(vec![60_000_000_000, 60_000_000_000], vec![5, 8]).unwrap()).unwrap();
        assert_eq!(cum.last().unwrap().unwrap().columns()[0].as_i64().unwrap(), &[15]);
        assert_eq!(cum.frame().unwrap().index().name(), Some("time"));
    }

    #[test]
    fn raw_admission_rejects_nat_disorder_and_malformed_schema() {
        assert!(raw(vec![i64::MIN], vec![1]).unwrap_err().to_string().contains("NaT"));
        assert!(raw(vec![2, 1], vec![1, 2]).unwrap_err().to_string().contains("non-monotonic"));
        assert!(raw(vec![0, 0], vec![1]).is_err());
        assert!(RawFineInput::new(vec!["v".into(), "v".into()], vec![Column::i64(vec![1]); 2], vec![0], Tz::Utc, None).is_err());
        let empty = raw(vec![], vec![]).unwrap();
        assert_eq!(Cumulator::from_raw(&empty, TimeFrame::Min5, AggSpec::ohlcv()).unwrap().into_frame().unwrap().height(), 0);
    }
}
