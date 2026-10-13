//! Interpreter-specific timezone data resolution and process key identity.
//!
//! Python chooses the bytes; `volas_core::tz` owns their parsing and all rules.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, TryLockError};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyBytes, PyDict};
use volas_core::Tz;

use crate::pyerr;

#[derive(Default)]
struct ZoneEntry {
    ready: OnceLock<Tz>,
    initialize: Mutex<()>,
}

thread_local! {
    static LOADING_KEYS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

struct LoadingKey {
    nested: bool,
}

impl LoadingKey {
    fn enter(key: &str) -> PyResult<Self> {
        LOADING_KEYS.with(|keys| {
            let mut keys = keys.borrow_mut();
            if keys.iter().any(|loading| loading == key) {
                return Err(PyRuntimeError::new_err(format!(
                    "recursive timezone initialization for key {key:?}",
                )));
            }
            let nested = !keys.is_empty();
            keys.push(key.to_string());
            Ok(Self { nested })
        })
    }
}

impl Drop for LoadingKey {
    fn drop(&mut self) {
        LOADING_KEYS.with(|keys| { keys.borrow_mut().pop(); });
    }
}

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
    // The registry guard never crosses a Python callback. Each key has its own
    // GIL-aware initialization lock. Nested callbacks never wait on another
    // active initializer, which prevents dependency cycles across threads.
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<ZoneEntry>>>> = OnceLock::new();
    let entry = {
        let mut cache = CACHE.get_or_init(Mutex::default)
            .lock_py_attached(py).unwrap_or_else(|error| error.into_inner());
        cache.get(key).cloned().unwrap_or_else(|| {
            let entry = Arc::new(ZoneEntry::default());
            cache.insert(key.to_string(), entry.clone());
            entry
        })
    };
    if let Some(zone) = entry.ready.get() {
        return Ok(zone.clone());
    }
    // Check recursion before waiting on the key's lock; dropping this guard on
    // any Python error permits a later initialization attempt to succeed.
    let loading = LoadingKey::enter(key)?;
    let _initialize = if loading.nested {
        match entry.initialize.try_lock() {
            Ok(guard) => guard,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => {
                if let Some(zone) = entry.ready.get() {
                    return Ok(zone.clone());
                }
                return Err(PyRuntimeError::new_err(format!(
                    "nested timezone initialization for key {key:?} would block; retry after the active initialization completes",
                )));
            }
        }
    } else {
        entry.initialize.lock_py_attached(py)
            .unwrap_or_else(|error| error.into_inner())
    };
    if let Some(zone) = entry.ready.get() {
        return Ok(zone.clone());
    }
    let helper = adapter(py)?;
    let (data, source, version): (Bound<'_, PyBytes>, String, Option<String>) =
        helper.call_method1("_resolve", (key,))?.extract()?;
    let zone = Tz::from_tzif(key, data.as_bytes().to_vec(), source, version).map_err(pyerr)?;
    helper.call_method1("_pin", (key,))?;
    Ok(entry.ready.get_or_init(|| zone).clone())
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
