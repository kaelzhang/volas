//! Physical data labels own canonical directive identities. Schema changes validate
//! uniqueness here; every binder and getter resolves through the same owner.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volas_core::DataFrame;

use crate::PyDataFrame;

pub(crate) fn data_identity(name: &str) -> String {
    volas_directive::canonical_data_name(name).unwrap_or_else(|_| name.to_string())
}

impl PyDataFrame {
    pub(crate) fn validate_data_names<'a>(names: impl IntoIterator<Item = &'a str>) -> PyResult<()> {
        let mut owners = std::collections::HashMap::new();
        for name in names {
            let canonical = data_identity(name);
            if let Some(previous) = owners.insert(canonical.clone(), name) {
                if previous == name {
                    return Err(PyValueError::new_err(format!("duplicate column name \"{name}\"")));
                }
                return Err(PyValueError::new_err(format!(
                    "conflicting data columns '{previous}' and '{name}': both own directive identity '{canonical}'. Rename one column."
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn validate_schema(frame: &DataFrame) -> PyResult<()> {
        Self::validate_data_names(frame.names().iter().filter(|n| !frame.is_computed(n)).map(String::as_str))
    }

    pub(crate) fn data_column_name(frame: &DataFrame, name: &str) -> Option<String> {
        if frame.has_column(name) && !frame.is_computed(name) {
            return Some(name.to_string());
        }
        let canonical = data_identity(name);
        frame.names().iter().find(|actual| !frame.is_computed(actual) && data_identity(actual) == canonical).cloned()
    }

    pub(crate) fn check_data_owner(&self, name: &str) -> PyResult<()> {
        Self::validate_data_names(self.inner.names().iter()
            .filter(|n| n.as_str() != name && !self.inner.is_computed(n))
            .map(String::as_str).chain(std::iter::once(name)))
    }
}
