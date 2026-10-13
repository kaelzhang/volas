//! volas-core: the pure-Rust data model behind the `volas` Python package.
//!
//! Storage primitives only (no pyo3, no numeric kernels, no directive engine) so
//! it builds and unit-tests with plain `cargo test` and is reusable as a library.

pub mod buffer;
pub mod column;
pub mod dataframe;
pub mod datetime;
pub mod decimal;
pub mod dtype;
pub mod error;
pub mod fxhash;
pub mod index;
pub mod numeric;
pub mod series;
pub mod stats;
pub mod strbuf;
pub mod tz;
pub mod validity;

pub use buffer::{Buffer, Element, Float, NanFact};
pub use strbuf::{StrBuffer, StrBufferBuilder};
pub use column::{BinOp, BoolOp, CmpOp, Column, CombineOp, Scalar};
pub use numeric::{binary_supertype, fits, Numeric};
pub use validity::{Bitmap, Validity};
pub use dataframe::{
    Anchors, CellWrite, CellWrites, Checkpoint, CheckpointRef, DataFrame, Directive, History, Reach, Replay, Retained, StaleComputed, StateWords,
};
pub use dtype::DType;
pub use error::{Result, VolasError};
pub use index::{Index, IndexKind, Label, SliceBound};
pub use series::Series;
pub use tz::Tz;
