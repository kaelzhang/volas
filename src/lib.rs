//! volas-rs: high-performance Rust core for the `volas` time-series kernel.
//!
//! This crate is the compiled backend for the `volas` Python package.
//! It is currently an empty scaffold; storage and indicator logic will live
//! here.

use pyo3::prelude::*;

/// The Rust extension module backing the `volas` package.
#[pymodule]
fn volas_rs(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
