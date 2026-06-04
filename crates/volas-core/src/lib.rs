//! volas-core: the pure-Rust data model behind the `volas` Python package.
//!
//! Storage primitives only (no pyo3, no numeric kernels, no directive engine) so
//! it builds and unit-tests with plain `cargo test` and is reusable as a library.

pub mod column;
pub mod dataframe;
pub mod datetime;
pub mod dtype;
pub mod error;
pub mod index;
pub mod series;

pub use column::Column;
pub use dataframe::DataFrame;
pub use dtype::DType;
pub use error::{Result, VolasError};
pub use index::Index;
pub use series::Series;
