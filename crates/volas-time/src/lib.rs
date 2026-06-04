//! volas-time: OHLCV time-frame cumulation (resampling) over the volas data
//! model. Groups fine bars on a `DatetimeIndex` into coarser periods and
//! aggregates each (open=first, high=max, low=min, close=last, volume=sum).
//!
//! Depends on `volas-core` only — a sibling of `volas-io` / `volas-directive`
//! with no cross-sibling coupling.

pub mod agg;
pub mod cumulate;
pub mod time_frame;

pub use agg::{Agg, AggSpec};
pub use cumulate::{cumulate, Cumulator};
pub use time_frame::TimeFrame;
