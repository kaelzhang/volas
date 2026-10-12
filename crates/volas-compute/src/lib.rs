//! volas-compute: numeric `kernels` (rolling windows, EWMA, diff) and the
//! technical `indicators`, as pure functions over `f64` / `bool` slices.
//!
//! A missing (NaN) input affects exactly the outputs whose definition reads it (see
//! [`missing`]). A finite-window function applies that itself, in [`raw`]: a per-row
//! term that reads a missing input is missing, and a window holding a missing row is
//! missing. A row where any input is missing does not exist for a recursive
//! computation: it outputs NaN, and the recursion continues from the last complete row.
//! [`indicators`] and [`kernels`] apply that rule at every public recursive function —
//! full computations, carried-state captures and resumes alike — and hand finite-window
//! functions through; [`raw`] holds the recursive functions without it (TA-Lib's
//! semantics: a NaN propagates).
//!
//! Depends on `volas-core` only for the shared error type; it does not touch the
//! DataFrame data model.

mod accum;
mod buf;
mod guard;
pub mod indicators;
pub mod kernels;
pub mod missing;
pub mod raw;
pub mod window;
