//! The kernels exactly as defined, over rows taken to be complete.
//!
//! Every function here computes on its inputs as given: a NaN is a value like any
//! other, so a recursive indicator carries it into every later output (TA-Lib's
//! semantics). [`crate::indicators`] and [`crate::kernels`] are the same functions with
//! the missing-row rule of [`crate::missing`] applied — a row missing any input is NaN
//! and the recursion continues from the last complete row. Call these directly only
//! for TA-Lib's behaviour, or over rows already known complete (as the directive
//! engine does after applying the rule itself).

pub mod indicators;
pub mod kernels;
pub(crate) mod smoothing;
