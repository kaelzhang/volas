//! volas-core: the pure-Rust kernel behind the `volas` Python package.
//!
//! This crate has no Python / pyo3 dependency, so it builds and unit-tests with
//! plain `cargo test` and is reusable from other Rust code. It provides the
//! column-oriented [`DataFrame`] / [`Series`] storage plus (later) the indicator
//! and directive engine.

pub mod column;
pub mod compute;
pub mod dataframe;
pub mod dtype;
pub mod error;
pub mod indicators;
pub mod index;
pub mod series;

pub use column::Column;
pub use dataframe::DataFrame;
pub use dtype::DType;
pub use error::{Result, VolasError};
pub use index::Index;
pub use series::Series;
