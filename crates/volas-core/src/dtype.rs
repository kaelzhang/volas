//! Minimal dtype model for the OHLCV / numeric-time-series domain.

/// The logical data type of a [`crate::Column`].
///
/// v1 keeps this deliberately small. Missing values in `F64` columns are encoded
/// in-band as `NaN` (matching stock-pandas / pandas semantics); a separate
/// validity-bitmap `null` model is a documented future refinement.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DType {
    /// 64-bit float (the OHLCV / indicator type). `NaN` denotes a missing value.
    F64,
    /// Boolean (comparison / signal results).
    Bool,
    /// 64-bit signed integer.
    I64,
    /// Datetime stored as i64 nanoseconds since the Unix epoch.
    Datetime,
}

impl DType {
    /// A short, stable, lower-case name for the dtype.
    pub fn name(&self) -> &'static str {
        match self {
            DType::F64 => "float64",
            DType::Bool => "bool",
            DType::I64 => "int64",
            DType::Datetime => "datetime64[ns]",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
