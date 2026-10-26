// The crate-level documentation is the README, so docs.rs and crates.io show one
// source. Its `rust` code blocks are compiled + run as doctests by `cargo test`.
#![doc = include_str!("../README.md")]

pub use volas_core::{
    Column, ColumnView, RowGroups, DType, DataFrame, Index, IndexKind, Label, Result, Scalar, Series, SliceBound, Tz, Validity,
    VolasError,
};
pub use volas_io::{read_csv, ReadCsvOptions};
pub use volas_time::TimeFrame;

/// The directive engine: parse a directive string (e.g. `"ma:20"`, `"macd.signal"`,
/// `"close > open"`) into an [`Ast`](volas_directive::Ast), then [`execute`] it
/// against a [`DataFrame`] to get a [`Column`].
pub mod directive {
    pub use volas_directive::{execute, lookback::lookback, parse, stringify, Ast, Op};
}

/// Numeric kernels and technical indicators — pure functions over slices, usable
/// without a [`DataFrame`]. They follow the same missing-row rule as the directive
/// engine: a row where any input of a recursive function is NaN outputs NaN, and the
/// recursion continues from the last complete row as if that row did not exist — for
/// full computations, carried states and resumes alike.
pub mod compute {
    pub use volas_compute::{indicators, kernels, window};
}

/// Time-frame cumulation (OHLCV resampling): aggregate finer bars into a coarser
/// [`TimeFrame`].
pub mod time {
    pub use volas_time::{aggregate_period, cumulate, Agg, AggSpec, Cumulator, OpenPeriod, RawFineInput, TimeFrame};
}

/// Checked values cannot be assembled through raw storage variants.
///
/// ```compile_fail
/// use volas::{Column, Validity};
/// let _ = Column::I64(vec![1].into(), Validity::from_valid_iter(2, [true, false]));
/// ```
///
/// Series structure is private, so a caller cannot bypass its length check.
///
/// ```compile_fail
/// use std::sync::Arc;
/// use volas::{Column, Index, Series};
/// let _ = Series { name: None, data: Column::i64(vec![1]),
///     index: Arc::new(Index::new(None, volas::IndexKind::Range { start: 0, len: 2 }).unwrap()) };
/// ```
///
/// The checked constructor reports the same mismatch as an error.
///
/// ```
/// use std::sync::Arc;
/// use volas::{Column, Index, Series};
/// assert!(Series::new(None, Column::i64(vec![1]), Arc::new(Index::new(None, volas::IndexKind::Range { start: 0, len: 2 }).unwrap())).is_err());
/// ```
mod invariant_examples {}

/// Raw append commits are not a published write path.
///
/// ```compile_fail
/// use volas::{Column, DataFrame};
/// let mut dst = DataFrame::new(vec!["x".into()], vec![Column::i64(vec![1])], None).unwrap();
/// let src = DataFrame::new(vec!["x".into()], vec![Column::str(vec!["bad".into()])], None).unwrap();
/// dst.append_admitted(&src).unwrap();
/// ```
mod write_admission_examples {}

/// A folded row needs the target-bound proof returned by checked admission.
///
/// ```compile_fail
/// use volas::{Column, DataFrame};
/// let mut dst = DataFrame::new(vec!["x".into()], vec![Column::i64(vec![1])], None).unwrap();
/// let src = dst.clone();
/// dst.write_folded_row(&src, &[]).unwrap();
/// ```
mod folded_admission_examples {}

/// Raw cell-fold commits cannot bypass their checked target-bound admission.
///
/// ```compile_fail
/// use volas::Column;
/// let mut dst = Column::i64(vec![1]);
/// let src = Column::i64(vec![2]);
/// dst.combine_at(0, volas::time::Agg::Sum.as_combine_op(), &src, 0).unwrap();
/// ```
mod cell_admission_examples {}

/// Running aggregate restarts also require their checked admission.
///
/// ```compile_fail
/// use volas::{Column, DataFrame, Index, IndexKind, TimeFrame, Tz};
/// use volas::time::{AggSpec, Cumulator};
/// let bar = DataFrame::new(vec!["close".into()], vec![Column::i64(vec![1])],
///     Some(Index::new(None, IndexKind::Datetime(vec![1], Tz::Utc)).unwrap())).unwrap();
/// let (_, _, aggregate) = Cumulator::from_frame(&bar, TimeFrame::Min1, AggSpec::ohlcv()).unwrap().into_parts().unwrap();
/// aggregate.unwrap().restart(bar.clone(), &bar);
/// ```
mod aggregate_admission_examples {}

/// Forming-row commits cannot be called without their checked position plan.
///
/// ```compile_fail
/// use volas::{Column, DataFrame};
/// let mut dst = DataFrame::new(vec!["x".into()], vec![Column::i64(vec![1])], None).unwrap();
/// let src = dst.clone();
/// dst.fold_forming_row(0, &src, 0, &[], true).unwrap();
/// ```
mod forming_admission_examples {}

/// Admitted cell commits are private; standalone Column callers use checked writes.
///
/// ```compile_fail
/// use volas::Column;
/// let mut dst = Column::f64(vec![1.]);
/// let src = Column::f64(vec![2.]);
/// dst.set_last_from_admitted(0, &src, 0);
/// ```
///
/// ```compile_fail
/// use volas::Column;
/// let mut dst = Column::f64(vec![1.]);
/// dst.set_last_na_admitted(0);
/// ```
mod last_cell_admission_examples {}

/// Reduction admission freezes positions; a user iterator is not a proof.
///
/// ```compile_fail
/// use volas::{Column, time::Agg};
/// let mut dst = Column::i64(vec![0]);
/// let src = Column::i64(vec![1, 2]);
/// dst.check_reduction(0, Agg::Sum.as_combine_op(), &src, (0..2).map(|i| i)).unwrap().land().unwrap();
/// ```
///
/// Grouped construction cannot trust a user-supplied ExactSizeIterator.
///
/// ```compile_fail
/// use volas::{Column, time::Agg};
/// let src = Column::i64(vec![1]);
/// src.reduce_groups(Agg::Last.as_combine_op(), [&[0][..]].into_iter()).unwrap();
/// ```
mod stable_selection_examples {}

/// Raw placeholder growth is internal to an admitted frame append.
///
/// ```compile_fail
/// use volas::Column;
/// let mut target = Column::i64(vec![1]);
/// target.append_na(usize::MAX);
/// ```
mod placeholder_admission_examples {}
