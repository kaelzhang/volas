// The crate-level documentation is the README, so docs.rs and crates.io show one
// source. Its `rust` code blocks are compiled + run as doctests by `cargo test`.
#![doc = include_str!("../README.md")]

pub use volas_core::{
    Column, ColumnView, DType, DataFrame, Index, IndexKind, Label, Result, Scalar, Series, SliceBound, Tz, Validity,
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
