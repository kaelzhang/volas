//! Error type for the volas core kernel.

use std::fmt;

/// Errors raised by the volas core. The Python binding maps each variant to a
/// typed exception (see `volas-python`).
#[derive(Debug, Clone, PartialEq)]
pub enum VolasError {
    /// A column name was not found in the frame.
    ColumnNotFound(String),

    /// A dtype-incompatible operation (e.g. appending Bool onto F64).
    DType(String),

    /// Mismatched lengths / shapes (e.g. columns of unequal height).
    Shape(String),

    /// Out-of-range or otherwise invalid index access.
    Index(String),

    /// A row label the operation needs is not in the index (or is not unique
    /// there), e.g. a label-slice bound on an unordered index.
    Key(String),

    /// A bad argument value.
    Value(String),

    /// A value outside the range its operation is defined on: an integer result
    /// past its dtype (contract #25), or an integer past the float64-exact range
    /// fed to float64 math (contract #26).
    Overflow(String),

    /// A directive string that could not be tokenized / parsed (a *syntax* error,
    /// annotated with line / column), as opposed to a [`Value`](Self::Value) error
    /// for a well-formed directive with an invalid command / argument.
    Parse(String),
}

/// Convenience result alias used throughout the core.
pub type Result<T> = std::result::Result<T, VolasError>;

impl fmt::Display for VolasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ColumnNotFound(name) => write!(f, "column \"{name}\" not found"),
            Self::DType(message) => write!(f, "dtype error: {message}"),
            Self::Shape(message) => write!(f, "shape error: {message}"),
            Self::Index(message) => write!(f, "index error: {message}"),
            Self::Key(message) => write!(f, "key error: {message}"),
            Self::Value(message) => write!(f, "value error: {message}"),
            Self::Overflow(message) => write!(f, "overflow: {message}"),
            Self::Parse(message) => write!(f, "parse error: {message}"),
        }
    }
}

impl std::error::Error for VolasError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_error_keeps_its_public_message_and_error_trait() {
        for (error, text) in [
            (
                VolasError::ColumnNotFound("price".into()),
                "column \"price\" not found",
            ),
            (VolasError::DType("detail".into()), "dtype error: detail"),
            (VolasError::Shape("detail".into()), "shape error: detail"),
            (VolasError::Index("detail".into()), "index error: detail"),
            (VolasError::Key("detail".into()), "key error: detail"),
            (VolasError::Value("detail".into()), "value error: detail"),
            (VolasError::Overflow("detail".into()), "overflow: detail"),
            (VolasError::Parse("detail".into()), "parse error: detail"),
        ] {
            assert_eq!(error.to_string(), text);
            let error: &dyn std::error::Error = &error;
            assert!(error.source().is_none());
        }
    }
}
