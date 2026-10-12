//! Error type for the volas core kernel.

use thiserror::Error;

/// Errors raised by the volas core. The Python binding maps each variant to a
/// typed exception (see `volas-python`).
#[derive(Debug, Error, Clone, PartialEq)]
pub enum VolasError {
    /// A column name was not found in the frame.
    #[error("column \"{0}\" not found")]
    ColumnNotFound(String),

    /// A dtype-incompatible operation (e.g. appending Bool onto F64).
    #[error("dtype error: {0}")]
    DType(String),

    /// Mismatched lengths / shapes (e.g. columns of unequal height).
    #[error("shape error: {0}")]
    Shape(String),

    /// Out-of-range or otherwise invalid index access.
    #[error("index error: {0}")]
    Index(String),

    /// A row label the operation needs is not in the index (or is not unique
    /// there), e.g. a label-slice bound on an unordered index.
    #[error("key error: {0}")]
    Key(String),

    /// A bad argument value.
    #[error("value error: {0}")]
    Value(String),

    /// A value outside the range its operation is defined on: an integer result
    /// past its dtype (contract #25), or an integer past the float64-exact range
    /// fed to float64 math (contract #26).
    #[error("overflow: {0}")]
    Overflow(String),

    /// A directive string that could not be tokenized / parsed (a *syntax* error,
    /// annotated with line / column), as opposed to a [`Value`](Self::Value) error
    /// for a well-formed directive with an invalid command / argument.
    #[error("parse error: {0}")]
    Parse(String),
}

/// Convenience result alias used throughout the core.
pub type Result<T> = std::result::Result<T, VolasError>;
