//! volas-compute: numeric `kernels` (rolling windows, EWMA, diff) and the
//! technical `indicators`, as pure functions over `f64` / `bool` slices.
//!
//! Depends on `volas-core` only for the shared error type; it does not touch the
//! DataFrame data model.

mod buf;
pub mod indicators;
pub mod kernels;
pub mod window;
