//! PyO3 bindings: expose the `volas` kernel to Python as `volas_rs.DataFrame` /
//! `volas_rs.Series`, with stock-pandas-style directive indexing and a
//! pandas-compatible indexing surface (`.iloc`, `.index`, `.name`, label lookup).
//!
//! This crate is the only place pyo3 / numpy are used; all logic lives in the
//! `volas-core` / `volas-compute` / `volas-directive` crates.

use volas_core::ColumnView;
use std::sync::Arc;

use pyo3::create_exception;
use pyo3::exceptions::{PyKeyError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use volas_core::{datetime, Column, Index, Series, VolasError};
use volas_directive::parse;

mod format;
mod readers;
mod timeframe;
mod timezone;

use format::parse_float_format;
use readers::read_csv;
use timeframe::PyTimeFrame;

// --- helpers ---------------------------------------------------------------

pub(crate) fn pyerr(e: VolasError) -> PyErr {
    match e {
        VolasError::ColumnNotFound(n) => PyKeyError::new_err(format!("column \"{n}\" not found")),
        VolasError::Key(m) => PyKeyError::new_err(m),
        VolasError::DType(m) => PyTypeError::new_err(m),
        VolasError::Overflow(m) => PyOverflowError::new_err(m),
        VolasError::Shape(m)
        | VolasError::Index(m)
        | VolasError::Value(m)
        | VolasError::Parse(m) => PyValueError::new_err(m),
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

/// The bound directive a cached column computes (only this binding caches columns, each
/// under its frame-bound [`volas_directive::Ast`]).
fn bound_directive(directive: &volas_core::Directive) -> &volas_directive::Ast {
    directive
        .downcast_ref::<volas_directive::Ast>()
        .expect("a cached column's directive is a bound volas_directive::Ast")
}

mod arrow;
mod coerce;
mod convert;
mod dlpack;
mod dt;
mod frame;
mod frame_access;
mod frame_bounded;
mod frame_construct;
mod frame_derive;
mod frame_index;
mod frame_identity;
mod frame_io;
mod frame_reduce;
mod frame_refresh;
mod frame_repr;
mod frame_reshape;
mod frame_row;
mod frame_tf;
mod frame_transform;
mod frame_window;
mod scalar;
mod series;
mod series_construct;
mod series_io;
mod series_ops;
mod series_reduce;
mod series_repr;
mod series_reshape;
mod series_support;
mod series_transform;
mod window;

pub(crate) use coerce::*;
pub(crate) use convert::*;
pub(crate) use frame::*;
pub(crate) use frame_bounded::{LookbackBound, WindowState};
pub(crate) use frame_derive::{uncarried_cached, RowSel};
pub(crate) use frame_index::*;
pub(crate) use frame_row::*;
pub(crate) use frame_tf::{TfState, TfWrite};
pub(crate) use frame_window::{PyEwmFrame, PyExpandingFrame, PyRollingFrame};
pub(crate) use scalar::*;
pub(crate) use series::*;
pub(crate) use series_support::*;
pub(crate) use volas_directive::Replayer;

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
            s.inner.data().clone(),
            s.inner.name().map(str::to_owned),
            Arc::clone(s.inner.index()),
        ),
        Err(_) => {
            let col = pyany_to_column(obj)?;
            let n = col.len();
            (col, None, Arc::new(Index::new(None, volas_core::IndexKind::Range { start: 0, len: n }).map_err(pyerr)?))
        }
    };
    let converted = match col.view() {
        ColumnView::Datetime(_) => col,
        ColumnView::Str(v, val) => match format {
            // An explicit format parses faster and unambiguously (pandas `format=`).
            // A missing (NA) cell maps to NaT, not parsed from its "" placeholder.
            Some(fmt) => {
                let parser = datetime::Strptime::new(fmt);
                let ns = v
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        // a missing (NA) or empty/blank cell -> NaT, like the
                        // default path; a non-empty value must match the format.
                        if !val.is_valid(i) || s.trim().is_empty() {
                            return Ok(i64::MIN);
                        }
                        parser.parse(s).map_err(pyerr)
                    })
                    .collect::<PyResult<Vec<i64>>>()?;
                Column::datetime(ns)
            }
            None => col.to_datetime().map_err(pyerr)?,
        },
        _ => col.epoch_to_datetime_rounded(unit).map_err(pyerr)?,
    };
    Ok(PySeries {
        inner: Series::new(name, converted, index).map_err(pyerr)?,
    })
}

/// Get the canonical full name of a `directive` — the actual column name volas caches it
/// under. The command name is lowercased, arguments are written as their values, and
/// default arguments / series are dropped: every spelling of a directive has one name.
///
/// Usage::
///
///     volas.directive_stringify('MACD:12,26')   # -> "macd"
#[pyfunction]
fn directive_stringify(directive: &str) -> PyResult<String> {
    // Bound as a form: every supplied argument bound (so the name is the cache column's),
    // an absent required one left absent, so a valid but argument-incomplete form like
    // `donchian.upper` still canonicalizes to its name; an unknown command / sub-command
    // or a bare sub-requiring name is rejected.
    let node = volas_directive::parse_cst(directive).map_err(directive_err)?;
    let form = volas_directive::bind_form(&node).map_err(value_err)?;
    Ok(volas_directive::stringify(&form))
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
    // Frame-blind: reject an unknown command / sub-command or a bare sub-requiring name.
    volas_directive::check_bare(&node).map_err(value_err)?;
    Ok(volas_directive::lookback::lookback(&node))
}

/// The structural facts of a command line, as its spec states them (an audit anchor):
/// `history` (`'window'` / `'accumulating'` / `'recursive'` under the directive's
/// arguments), `accumulates` (a running accumulator by kind), `resume` (`'recompute'` /
/// `'rows'` / `'step'`), `position` (a row-position output) and `bare` (usable as a bare
/// name). Without `columns`, `directive` is one command, explicit or bare. With
/// `columns`, it is any expression: returns its bound lookback / history class and
/// whether adding `shadow` would change a command dependency of that binding.
#[pyfunction]
#[pyo3(signature = (directive, columns=None, shadow=None))]
fn _directive_facts<'py>(
    py: Python<'py>,
    directive: &str,
    columns: Option<Vec<String>>,
    shadow: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    use volas_directive::spec::{History, Resume};
    use volas_directive::types::Node;
    let node = parse(directive).map_err(directive_err)?;
    if let Some(columns) = columns {
        // Harness inventory facts use the same binding owner as execution. A physical
        // directive label is data at every nesting level, with no command warm-up.
        let identities: Vec<_> = columns.iter().map(|c| crate::frame_identity::data_identity(c)).collect();
        let bound = volas_directive::bind_frame_resolved(&node, &|name| {
            identities.iter().position(|id| id.as_ref() == name).map(|i| columns[i].clone())
        }, directive).map_err(directive_err)?;
        let reach = volas_directive::lookback::reach(&bound);
        let class = reach.history.class();
        let facts = pyo3::types::PyDict::new(py);
        facts.set_item("lookback", reach.lookback)?;
        facts.set_item("history_class", class)?;
        facts.set_item("shadows", shadow.is_some_and(|name| {
            volas_directive::uses_command_shadow(&bound, &crate::frame_identity::data_identity(name))
        }))?;
        return Ok(facts);
    }
    let node = match node.node() {
        Node::Name(name) => std::rc::Rc::new(
            volas_directive::bind_command(name, None, &[], &[]).map_err(value_err)?,
        ),
        _ => node,
    };
    let cmd = match node.node() {
        Node::Command(c) => c,
        _ => {
            return Err(DirectiveValueError::new_err(format!(
                "'{directive}' is not a command"
            )))
        }
    };
    let spec = cmd.args.spec();
    let facts = pyo3::types::PyDict::new(py);
    let history = match spec.history(&cmd.args) {
        History::Window => "window",
        History::Accumulating => "accumulating",
        History::Recursive => "recursive",
    };
    let resume = match spec.resume(&cmd.args) {
        Resume::Recompute => "recompute",
        Resume::Rows => "rows",
        Resume::Step => "step",
    };
    facts.set_item("history", history)?;
    facts.set_item("accumulates", spec.accumulates())?;
    facts.set_item("resume", resume)?;
    facts.set_item("position", spec.emits_position())?;
    facts.set_item("bare", volas_directive::is_bare_command(&cmd.name))?;
    Ok(facts)
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
    m.add_function(wrap_pyfunction!(timezone::tz_database, m)?)?;
    m.add_function(wrap_pyfunction!(directive_stringify, m)?)?;
    m.add_function(wrap_pyfunction!(directive_lookback, m)?)?;
    m.add_function(wrap_pyfunction!(_directive_facts, m)?)?;
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
    m.add(
        "_directive_commands",
        volas_directive::spec::COMMANDS.to_vec(),
    )?;
    // The commands usable as a bare name (the spec's fact the R17 ambiguity rule reads).
    m.add(
        "_directive_bare_commands",
        volas_directive::spec::COMMANDS
            .iter()
            .copied()
            .filter(|c| volas_directive::is_bare_command(c))
            .collect::<Vec<_>>(),
    )?;
    let missing = scalar::NA_SINGLETON.get_or_try_init(m.py(), || {
        Py::new(m.py(), scalar::NaType).map(|v| v.into_any())
    })?;
    m.add("NA", missing)?;
    Ok(())
}

mod numpy_export;
