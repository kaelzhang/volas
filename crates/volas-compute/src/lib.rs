//! volas-compute: numeric `kernels` (rolling windows, EWMA, diff) and the
//! technical `indicators`, as pure functions over `f64` / `bool` slices.
//!
//! A row where any input is missing (NaN) does not exist for a recursive computation:
//! it outputs NaN, and the recursion continues from the last complete row (see
//! [`missing`]). [`indicators`] and [`kernels`] apply that rule at every public
//! recursive function — full computations, carried-state captures and resumes alike —
//! and hand finite-window functions through, as window locality already confines a
//! bad row to the windows holding it. [`raw`] holds the same functions without the
//! rule (TA-Lib's semantics: a NaN propagates).
//!
//! Depends on `volas-core` only for the shared error type; it does not touch the
//! DataFrame data model.

mod buf;
mod guard;
pub mod indicators;
pub mod kernels;
pub mod missing;
pub mod raw;
pub mod window;
