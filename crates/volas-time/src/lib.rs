//! volas-time: OHLCV time-frame cumulation (resampling) over the volas data
//! model. Groups fine bars on a `DatetimeIndex` into coarser periods and
//! aggregates each (open=first, high=max, low=min, close=last, volume=sum).
//!
//! Depends on `volas-core` only — a sibling of `volas-io` / `volas-directive`
//! with no cross-sibling coupling.

pub mod agg;
pub mod cumulate;
mod raw_input;
pub use raw_input::RawFineInput;
pub mod open_period;
mod period_aggregate;
pub mod time_frame;

pub use agg::{Agg, AggSpec};
pub use cumulate::{aggregate_period, cumulate, Cumulator};
pub use open_period::OpenPeriod;
pub use period_aggregate::{PeriodAggregate, PeriodPush, PeriodRestart};
pub use time_frame::{Span, TimeFrame};

#[cfg(test)]
mod cumulate_regression_tests;
