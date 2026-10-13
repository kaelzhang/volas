//! Parameter domains owned by the published window helpers.
use std::fmt;

/// Invalid window-helper input, rejected before the kernel reads any rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WindowError {
    Window,
    MinPeriods,
    Alpha,
    Quantile,
    Interpolation(String),
    Rank(String),
    Shape,
    Padding,
}

impl fmt::Display for WindowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Window => f.write_str("rolling window must be >= 1"),
            Self::MinPeriods => f.write_str("min_periods must be <= window"),
            Self::Alpha => f.write_str("ewm alpha must satisfy 0 < alpha <= 1"),
            Self::Quantile => f.write_str("quantile q must be in [0, 1]"),
            Self::Interpolation(s) => write!(f, "interpolation must be one of linear/lower/higher/nearest/midpoint, got {s:?}"),
            Self::Rank(s) => write!(f, "rank method must be 'average', 'min' or 'max', got {s:?}"),
            Self::Shape => f.write_str("window inputs must have equal lengths"),
            Self::Padding => f.write_str("rank data_len must describe the data before trailing padding"),
        }
    }
}
impl std::error::Error for WindowError {}

/// Check the rolling geometry; `usize::MAX` denotes expanding.
pub fn validate_window(window: usize, min_periods: usize) -> Result<(), WindowError> {
    if window == 0 { return Err(WindowError::Window); }
    if min_periods > window { return Err(WindowError::MinPeriods); }
    Ok(())
}

/// Check the resolved decay, including a decay supplied through another spelling.
pub fn validate_alpha(alpha: f64) -> Result<(), WindowError> {
    if !(alpha > 0.0 && alpha <= 1.0) { return Err(WindowError::Alpha); }
    Ok(())
}

pub(super) fn validate_pair(x: &[f64], y: &[f64]) -> Result<(), WindowError> {
    if x.len() != y.len() { return Err(WindowError::Shape); }
    Ok(())
}

pub(super) fn validate_quantile(q: f64, interpolation: &str) -> Result<(), WindowError> {
    if !matches!(interpolation, "linear" | "lower" | "higher" | "nearest" | "midpoint") {
        return Err(WindowError::Interpolation(interpolation.into()));
    }
    if !(0.0..=1.0).contains(&q) { return Err(WindowError::Quantile); }
    Ok(())
}

pub(super) fn validate_rank(method: &str) -> Result<(), WindowError> {
    if !matches!(method, "average" | "min" | "max") { return Err(WindowError::Rank(method.into())); }
    Ok(())
}
