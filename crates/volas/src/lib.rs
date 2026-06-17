// The crate-level documentation is the README, so docs.rs and crates.io show one
// source. Its `rust` code blocks are compiled + run as doctests by `cargo test`.
#![doc = include_str!("../README.md")]

/// The full `volas-core` surface (data model, ops enums, numeric helpers), for the
/// less-common types not re-exported at the top level.
pub use volas_core as core;

pub use volas_core::{
    Column, DType, DataFrame, Index, IndexKind, Label, Result, Scalar, Series, Tz, VolasError,
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
/// without a [`DataFrame`].
pub mod compute {
    pub use volas_compute::{indicators, kernels, window};
}

/// Time-frame cumulation (OHLCV resampling): aggregate finer bars into a coarser
/// [`TimeFrame`].
pub mod time {
    pub use volas_time::{aggregate_period, cumulate, Agg, AggSpec, Cumulator, TimeFrame};
}
