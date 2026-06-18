//! volas-compute: numeric `kernels` (rolling windows, EWMA, diff) and the
//! technical `indicators`, as pure functions over `f64` / `bool` slices.
//!
//! Depends on `volas-core` only for the shared error type; it does not touch the
//! DataFrame data model.

// This crate is tight numeric kernels. `needless_range_loop` / `explicit_counter_loop`
// are calibrated for general application code; here the index-loop form is the
// idiomatic, clearest way to slide overlapping OHLCV windows (`data[i]` /
// `data[i - period]`) or use the index as a weight (WMA's `j + 1`). Iterator rewrites
// add no clarity and would risk perturbing hot-path codegen that `make asm-diff` only
// guards for the three probed kernels.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::explicit_counter_loop)]

mod buf;
pub mod indicators;
pub mod kernels;
pub mod window;
