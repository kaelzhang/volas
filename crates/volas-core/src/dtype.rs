//! Minimal dtype model for the OHLCV / numeric-time-series domain.

/// The logical data type of a [`crate::Column`].
///
/// Deliberately small (the OHLCV / numeric-time-series domain). Missing values
/// are encoded per dtype: `F64` / `F32` use in-band `NaN` (matching stock-pandas
/// / pandas float semantics), `Datetime` uses the `i64::MIN` `NaT` sentinel, and
/// `Bool` / `I64` / `I32` / `Utf8` carry a validity bitmap ([`crate::Validity`])
/// that marks each missing cell.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DType {
    /// 64-bit float (the OHLCV / indicator type). `NaN` denotes a missing value.
    F64,
    /// 32-bit float (narrow storage). `NaN` denotes a missing value.
    F32,
    /// Boolean (comparison / signal results).
    Bool,
    /// 64-bit signed integer.
    I64,
    /// 32-bit signed integer (narrow storage).
    I32,
    /// Datetime stored as i64 nanoseconds since the Unix epoch.
    Datetime,
    /// UTF-8 string. Reported as `str`, matching pandas 3.0's default string
    /// dtype (pandas <= 2.x called these columns `object`).
    Utf8,
}

impl DType {
    /// A short, stable name for the dtype.
    pub fn name(&self) -> &'static str {
        match self {
            DType::F64 => "float64",
            DType::F32 => "float32",
            DType::Bool => "bool",
            DType::I64 => "int64",
            DType::I32 => "int32",
            DType::Datetime => "datetime64[ns]",
            DType::Utf8 => "str",
        }
    }

    /// Whether this is a floating-point dtype (for promotion).
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F64 | DType::F32)
    }

    /// Whether this is a numeric dtype — the float and integer kinds (`F64` / `F32`
    /// / `I64` / `I32`), but **not** `Bool`. The single predicate the DataFrame
    /// numeric APIs (`round`, `describe`, `corr` / `cov`, `sem` / `skew` / `kurt`)
    /// and assignment coercion share, so narrow dtypes are never silently dropped.
    pub fn is_numeric(&self) -> bool {
        matches!(self, DType::F64 | DType::F32 | DType::I64 | DType::I32)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
