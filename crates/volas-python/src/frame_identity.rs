//! Physical data labels own canonical directive identities. Schema changes validate
//! uniqueness here; every binder and getter resolves through the same owner.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use volas_core::DataFrame;
use std::borrow::Cow;
use std::collections::HashMap;

use crate::PyDataFrame;

#[inline]
pub(crate) fn data_identity(name: &str) -> Cow<'_, str> {
    volas_directive::canonical_data_name(name).unwrap_or(Cow::Borrowed(name))
}

fn conflicting_owner(previous: &str, name: &str, canonical: &str) -> PyErr {
    if previous == name {
        return PyValueError::new_err(format!("duplicate column name \"{name}\""));
    }
    PyValueError::new_err(format!(
        "conflicting data columns '{previous}' and '{name}': both own directive identity '{canonical}'. Rename one column."
    ))
}

impl PyDataFrame {
    pub(crate) fn validate_data_names<'a>(names: impl IntoIterator<Item = &'a str>) -> PyResult<()> {
        Self::validate_identities(names.into_iter().map(|name| (name, data_identity(name))))
    }

    fn validate_identities<'a>(identities: impl IntoIterator<Item = (&'a str, Cow<'a, str>)>) -> PyResult<()> {
        // Typical OHLCV schemas need neither a map nor heap-owned identities.
        // Spill once for wider schemas, preserving linear-time validation there.
        let mut small: [Option<(Cow<'a, str>, &'a str)>; 8] = [const { None }; 8];
        let mut len = 0;
        let mut owners: Option<HashMap<Cow<'a, str>, &'a str>> = None;
        for (name, canonical) in identities {
            if owners.is_none() {
                for (identity, previous) in small[..len].iter().flatten() {
                    if *identity == canonical {
                        return Err(conflicting_owner(previous, name, &canonical));
                    }
                }
                if len < small.len() {
                    small[len] = Some((canonical, name));
                    len += 1;
                    continue;
                }
                owners = Some(small.iter_mut().filter_map(Option::take).collect());
            }
            let map = owners.as_mut().unwrap();
            if let Some(previous) = map.get(&canonical) {
                return Err(conflicting_owner(previous, name, &canonical));
            }
            map.insert(canonical, name);
        }
        Ok(())
    }

    // Keep the ordinary predicate in its constructor: an out-of-line call would
    // make the newly built frame's address escape before it is returned.
    #[inline(always)]
    pub(crate) fn validate_schema(frame: &DataFrame) -> PyResult<()> {
        // Core already guarantees unique physical labels. If every identity is
        // its label, that same guarantee proves identity uniqueness: no second
        // duplicate check or owner storage is needed for ordinary schemas.
        for (i, label) in frame.names().iter().enumerate() {
            if frame.is_computed(label) { continue; }
            let name = label.as_str();
            let canonical = data_identity(name);
            if canonical != name {
                // The prefix already owns its exact labels. Carry those known
                // identities and this first changed identity into the same
                // checker; the remaining labels are canonicalized once too.
                let known = frame.names()[..i].iter().filter(|n| !frame.is_computed(n))
                    .map(|n| (n.as_str(), Cow::Borrowed(n.as_str())));
                let remaining = frame.names()[i + 1..].iter().filter(|n| !frame.is_computed(n))
                    .map(|n| (n.as_str(), data_identity(n)));
                return Self::validate_identities(known.chain(std::iter::once((name, canonical))).chain(remaining));
            }
        }
        Ok(())
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
