//! Interpreter-specific timezone data resolution and process key identity.
//!
//! Python chooses the bytes; `volas_core::tz` owns their parsing and all rules.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyBytes, PyDict};
use volas_core::Tz;

use crate::pyerr;

fn adapter(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("volas._timezone")
}

/// Resolve named zones from this interpreter, never the Rust OS fallback.
pub(crate) fn parse(py: Python<'_>, key: &str) -> PyResult<Tz> {
    let key = key.trim();
    if let Some(zone) = Tz::parse_fixed(key) {
        return zone.map_err(pyerr);
    }
    named(py, key)
}

fn named(py: Python<'_>, key: &str) -> PyResult<Tz> {
    static CACHE: OnceLock<Mutex<HashMap<String, Tz>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Mutex::default);
    if let Some(zone) = cache.lock_py_attached(py)
        .unwrap_or_else(|error| error.into_inner()).get(key) {
        return Ok(zone.clone());
    }
    // Python I/O may release the GIL. Resolve and compile without the registry
    // lock; racing cold reads are harmless, and every caller uses the winner.
    let helper = adapter(py)?;
    let (data, source, version): (Bound<'_, PyBytes>, String, Option<String>) =
        helper.call_method1("_resolve", (key,))?.extract()?;
    let zone = Tz::from_tzif(key, data.as_bytes().to_vec(), source, version).map_err(pyerr)?;
    helper.call_method1("_pin", (key,))?;
    let mut cache = cache.lock_py_attached(py).unwrap_or_else(|error| error.into_inner());
    Ok(cache.entry(key.to_string()).or_insert(zone).clone())
}

// Preserve the existing mixed datetime.timezone.utc / ZoneInfo("UTC") input
// contract only when the owner confirms canonical UTC over the full ns axis.
fn canonical_utc(zone: Tz) -> Tz {
    if let Tz::Named(rules) = &zone {
        if rules.key() == "UTC" && rules.is_canonical_utc() {
            return Tz::Utc;
        }
    }
    zone
}

/// A keyed tzinfo selects the process's canonical rules for that key.
pub(crate) fn from_object(obj: &Bound<'_, PyAny>, key: &str) -> PyResult<Tz> {
    named(obj.py(), key).map(canonical_utc)
}

pub(crate) fn to_python<'py>(py: Python<'py>, zone: &Tz) -> PyResult<Option<Bound<'py, PyAny>>> {
    match zone {
        Tz::Naive => Ok(None),
        Tz::Utc => Ok(Some(py.import("datetime")?.getattr("timezone")?.getattr("utc")?)),
        Tz::Offset(seconds) => {
            let dt = py.import("datetime")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("seconds", seconds)?;
            let delta = dt.getattr("timedelta")?.call((), Some(&kwargs))?;
            Ok(Some(dt.getattr("timezone")?.call1((delta,))?))
        }
        Tz::Named(rules) => Ok(Some(adapter(py)?.call_method1("_pin", (rules.key(),))?)),
    }
}

/// Report the key's first-use source directory or `tzdata` package and release.
///
/// The optional key makes provenance exact when TZPATH directories contain
/// different subsets of zones. An unused key is initialized by this call.
#[pyfunction]
#[pyo3(signature = (key = "UTC"))]
pub(crate) fn tz_database<'py>(py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyDict>> {
    let zone = named(py, key)?;
    let Tz::Named(rules) = zone else { unreachable!("named() always returns named rules") };
    let result = PyDict::new(py);
    result.set_item("source", rules.source())?;
    result.set_item("version", rules.version())?;
    Ok(result)
}

#[cfg(test)]
mod tests;
