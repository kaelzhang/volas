//! # volas
//!
//! A Rust-backed, OHLCV-shaped [`DataFrame`] for candlestick / market time-series,
//! with a technical-indicator **directive** engine.
//!
//! volas is intentionally narrow: it is not a general-purpose DataFrame. It targets
//! live OHLCV pipelines — append a new bar, keep indicator columns cached, and
//! recompute only the stale tail (`O(lookback + new rows)`, not `O(n)`).
//!
//! This is the umbrella crate. It re-exports the everyday types at the top level and
//! groups the rest under modules:
//!
//! - top level — the data model ([`DataFrame`], [`Series`], [`Column`], [`Index`],
//!   [`DType`], [`Scalar`], [`Tz`], [`Result`], [`VolasError`]) plus CSV [`read_csv`]
//!   and [`TimeFrame`];
//! - [`directive`] — parse a directive string into an `Ast`, then `execute` it;
//! - [`compute`] — numeric kernels and technical indicators (pure functions);
//! - [`time`] — time-frame cumulation (OHLCV resampling);
//! - [`core`] — the full `volas-core` surface, for the less-common types.
//!
//! ```
//! # fn main() -> Result<(), volas::VolasError> {
//! use volas::{Column, DataFrame};
//! use volas::directive::{execute, parse};
//!
//! let df = DataFrame::new(
//!     vec!["close".to_string()],
//!     vec![Column::f64(vec![1.0, 2.0, 3.0, 4.0])],
//!     None,
//! )?;
//!
//! // `ma:2` is a 2-period simple moving average over `close`.
//! let directive = parse("ma:2")?;
//! let ma = execute(&df, &directive)?;
//! assert_eq!(ma.len(), 4);
//! assert_eq!(ma.to_f64_vec()[3], 3.5); // (3.0 + 4.0) / 2
//! # Ok(())
//! # }
//! ```

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
