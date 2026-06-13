//! PyO3 bindings: expose the `volas` kernel to Python as `volas_rs.DataFrame` /
//! `volas_rs.Series`, with stock-pandas-style directive indexing and a
//! pandas-compatible indexing surface (`.iloc`, `.index`, `.name`, label lookup).
//!
//! This crate is the only place pyo3 / numpy are used; all logic lives in the
//! `volas-core` / `volas-compute` / `volas-directive` crates.

use std::sync::Arc;

use pyo3::create_exception;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use volas_core::{
    datetime, Column, DataFrame, Index, Series, VolasError,
};
use volas_directive::parse;

mod format;
mod readers;
mod timeframe;

use format::parse_float_format;
use readers::read_csv;
use timeframe::PyTimeFrame;

// --- helpers ---------------------------------------------------------------

pub(crate) fn pyerr(e: VolasError) -> PyErr {
    match e {
        VolasError::ColumnNotFound(n) => PyKeyError::new_err(format!("column \"{n}\" not found")),
        VolasError::DType(m) => PyTypeError::new_err(m),
        VolasError::Shape(m) | VolasError::Index(m) | VolasError::Value(m) | VolasError::Parse(m) => {
            PyValueError::new_err(m)
        }
    }
}

// Typed directive exceptions (both subclass ValueError, so existing
// `except ValueError` keeps working while callers can catch the specific type).
create_exception!(
    volas_rs,
    DirectiveError,
    PyValueError,
    "Base class for volas directive errors."
);
create_exception!(
    volas_rs,
    DirectiveSyntaxError,
    DirectiveError,
    "A directive string could not be parsed (with line / column)."
);
create_exception!(
    volas_rs,
    DirectiveValueError,
    DirectiveError,
    "A directive has an unknown command / sub-command or an invalid argument."
);

/// Map a `parse` error to its typed exception. `parse` now both tokenizes and
/// binds, so a tokenizer failure (`VolasError::Parse`, annotated with line /
/// column) is a `DirectiveSyntaxError`, while a well-formed directive with an
/// unknown command / sub-command or an invalid argument is a `DirectiveValueError`.
fn directive_err(e: VolasError) -> PyErr {
    match e {
        VolasError::Parse(m) => DirectiveSyntaxError::new_err(m),
        e => value_err(e),
    }
}

/// Map a directive execution error to `DirectiveValueError`.
fn value_err(e: VolasError) -> PyErr {
    match e {
        VolasError::Value(m) => DirectiveValueError::new_err(m),
        VolasError::ColumnNotFound(n) => {
            DirectiveValueError::new_err(format!("column \"{n}\" not found"))
        }
        other => pyerr(other),
    }
}

fn directive_uses_default_series(node: &volas_directive::types::Ast) -> bool {
    match node {
        volas_directive::types::Node::Name(_) => true,
        volas_directive::types::Node::Command(cmd) => cmd.series.iter().all(
            |series| matches!(series, volas_directive::types::Node::Name(name) if name.is_empty()),
        ),
        _ => false,
    }
}

mod scalar;
mod convert;
mod coerce;
mod series;
mod dt;
mod window;
mod series_support;
mod frame;
mod frame_methods;
mod frame_methods2;
mod frame_index;

pub(crate) use scalar::*;
pub(crate) use convert::*;
pub(crate) use coerce::*;
pub(crate) use series::*;
pub(crate) use series_support::*;
pub(crate) use frame::*;
pub(crate) use frame_methods::{PyEwmFrame, PyExpandingFrame, PyRollingFrame};
pub(crate) use frame_index::*;

/// Raise if the frame has stale computed columns after an `append`. The per-column
/// `df[directive]` access auto-refreshes; bulk / positional reads (`to_numpy`,
/// `.iloc` / `.loc` / `.at` / `.iat`) do not, so they must be fresh — call
/// `fulfill()` first. Keeps the read path O(1) and never returns silent NaN.
pub(crate) fn ensure_fresh(df: &DataFrame) -> PyResult<()> {
    if df.has_stale_computed() {
        Err(PyValueError::new_err(
            "frame has stale computed (directive) columns after append; \
             call fulfill() before a bulk or positional read",
        ))
    } else {
        Ok(())
    }
}

/// Parse an optional printf-style `float_format` spec (shared by `to_csv` /
/// `to_string`), raising a `ValueError` on an unsupported form.
fn parse_ff(float_format: Option<&str>) -> PyResult<Option<(Option<usize>, char)>> {
    match float_format {
        Some(f) => Ok(Some(parse_float_format(f).ok_or_else(|| {
            PyValueError::new_err(format!("unsupported float_format \"{f}\""))
        })?)),
        None => Ok(None),
    }
}

/// Convert epoch numbers or datetime strings to a datetime `Series`, mirroring
/// `pandas.to_datetime` (numeric epochs in `unit`, parsed strings, or a passthrough
/// datetime input). Accepts a volas `Series`, a 1-D NumPy array, or a list.
#[pyfunction]
#[pyo3(signature = (obj, unit = "ns", format = None))]
fn to_datetime(obj: &Bound<'_, PyAny>, unit: &str, format: Option<&str>) -> PyResult<PySeries> {
    let (col, name, index) = match obj.extract::<PyRef<PySeries>>() {
        Ok(s) => (
            s.inner.data.clone(),
            s.inner.name.clone(),
            Arc::clone(&s.inner.index),
        ),
        Err(_) => {
            let col = pyany_to_column(obj)?;
            let n = col.len();
            (col, None, Arc::new(Index::range(n)))
        }
    };
    let converted = match col {
        c @ Column::Datetime(_) => c,
        Column::Str(v, val) => match format {
            // An explicit format parses faster and unambiguously (pandas `format=`).
            // A missing (NA) cell maps to NaT, not parsed from its "" placeholder.
            Some(fmt) => {
                let ns = v
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        // a missing (NA) or empty/blank cell -> NaT, like the
                        // default path; a non-empty value must match the format.
                        if !val.is_valid(i) || s.trim().is_empty() {
                            return Ok(i64::MIN);
                        }
                        datetime::parse_ns_format(s, fmt).ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "\"{s}\" does not match format \"{fmt}\""
                            ))
                        })
                    })
                    .collect::<PyResult<Vec<i64>>>()?;
                Column::datetime(ns)
            }
            None => Column::Str(v, val).to_datetime().map_err(pyerr)?,
        },
        c => c.epoch_to_datetime_rounded(unit).map_err(pyerr)?,
    };
    Ok(PySeries {
        inner: Series::new(name, converted, index),
    })
}

/// Get the canonical full name of a `directive` — the actual column name volas caches it
/// under. The command name is lowercased and default arguments / series are dropped.
///
/// Usage::
///
///     volas.directive_stringify('MACD:12,26')   # -> "macd"
#[pyfunction]
fn directive_stringify(directive: &str) -> PyResult<String> {
    let node = parse(directive).map_err(directive_err)?;
    // Frame-blind: a bare name is taken as a command, so reject a known command with
    // no sub-less form (`kdj`, `stoch`, …) — `df[d]` rejects it at execute instead.
    volas_directive::check_bare_commands(&node).map_err(value_err)?;
    Ok(volas_directive::stringify(&node))
}

/// Get the lookback period of a `directive` — the minimum number of prior rows it needs
/// before it can emit a (non-NaN) value.
///
/// Usage::
///
///     volas.directive_lookback('boll:20')   # -> 19
#[pyfunction]
fn directive_lookback(directive: &str) -> PyResult<usize> {
    // `parse` binds + validates, so a bad directive (e.g. `ma:0`) is rejected here
    // rather than yielding a "plausible" warm-up window that feeds a scheduler /
    // cache before execution fails (P2-02 / V17).
    let node = parse(directive).map_err(directive_err)?;
    // Frame-blind: reject a bare known command with no sub-less form (`kdj`, …).
    volas_directive::check_bare_commands(&node).map_err(value_err)?;
    Ok(volas_directive::lookback::lookback(&node))
}

/// The compiled module backing the `volas` package.
#[pymodule]
fn volas_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataFrame>()?;
    m.add_class::<PySeries>()?;
    m.add_class::<PyRow>()?;
    m.add_class::<PyTimestamp>()?;
    m.add_class::<dt::PyDt>()?;
    m.add_class::<DataFrameILoc>()?;
    m.add_class::<DataFrameLoc>()?;
    m.add_class::<DataFrameIat>()?;
    m.add_class::<DataFrameAt>()?;
    m.add_class::<SeriesILoc>()?;
    m.add_class::<SeriesLoc>()?;
    m.add_class::<window::PyRolling>()?;
    m.add_class::<window::PyExpanding>()?;
    m.add_class::<window::PyEwm>()?;
    m.add_class::<PyRollingFrame>()?;
    m.add_class::<PyExpandingFrame>()?;
    m.add_class::<PyEwmFrame>()?;
    m.add_class::<PyTimeFrame>()?;
    m.add("DirectiveError", m.py().get_type::<DirectiveError>())?;
    m.add(
        "DirectiveSyntaxError",
        m.py().get_type::<DirectiveSyntaxError>(),
    )?;
    m.add(
        "DirectiveValueError",
        m.py().get_type::<DirectiveValueError>(),
    )?;
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(to_datetime, m)?)?;
    m.add_function(wrap_pyfunction!(directive_stringify, m)?)?;
    m.add_function(wrap_pyfunction!(directive_lookback, m)?)?;
    m.add_class::<NaType>()?;
    // Machine-readable vocabularies for the systematic audit's external anchors
    // (SPEC §6.3): the dtype set is generated FROM the Rust enum (a new dtype
    // auto-extends the audit matrix), and the directive command list comes from
    // the spec.rs registry (no hand-written command vocabulary).
    m.add(
        "_dtypes",
        volas_core::DType::ALL
            .iter()
            .map(|d| d.name().to_string())
            .collect::<Vec<_>>(),
    )?;
    m.add("_directive_commands", volas_directive::spec::COMMANDS.to_vec())?;
    m.add("NA", na(m.py()))?;
    Ok(())
}
