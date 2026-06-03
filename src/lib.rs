//! candles-rs: high-performance Rust core for the `candles` time-series kernel.
//!
//! This crate is the compiled backend for the `candles` Python package.
//! It is currently an empty scaffold; storage and indicator logic will live
//! here.

use pyo3::prelude::*;

/// The Rust extension module backing the `candles` package.
#[pymodule]
fn candles_rs(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
