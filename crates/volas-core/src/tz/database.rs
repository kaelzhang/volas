//! Rust-only source adapter. Python supplies bytes from its own TZPATH/package
//! resolver instead of consulting these OS paths.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use crate::error::{Result, VolasError};
use super::{Tz, ZoneRules};

type Registry = Mutex<HashMap<String, Arc<ZoneRules>>>;

pub(super) fn valid_key(key: &str) -> bool {
    !key.is_empty() && !key.contains('\\') && !key.contains('\0')
        && key.split('/').all(|p| !p.is_empty() && p != "." && p != "..")
}

pub(super) fn load(key: &str) -> Result<Tz> {
    if !valid_key(key) { return Err(VolasError::Value(format!("invalid timezone key {key:?}"))); }
    // Unlike the bounded byte cache, named lookups retain the first successful
    // load for the entire process. Explicit from_tzif calls never enter here.
    static NAMED: OnceLock<Registry> = OnceLock::new();
    load_once(NAMED.get_or_init(Registry::default), key, || read(key)).map(Tz::Named)
}

fn load_once(registry: &Registry, key: &str, read: impl FnOnce() -> Result<Arc<ZoneRules>>) -> Result<Arc<ZoneRules>> {
    // Keep the lock through initialization: simultaneous first lookups must
    // neither read the source twice nor acquire different snapshots.
    let mut loaded = registry.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(rules) = loaded.get(key) { return Ok(rules.clone()); }
    let rules = read()?;
    loaded.insert(key.to_string(), rules.clone());
    Ok(rules)
}

fn read(key: &str) -> Result<Arc<ZoneRules>> {
    for path in directories(std::env::var_os("TZDIR")) {
        if let Ok(bytes) = std::fs::read(path.join(key)) {
            return ZoneRules::load(key, bytes, path.to_string_lossy().into_owned(), version(&path));
        }
    }
    Err(VolasError::Value(format!("unknown timezone {key:?}")))
}

fn directories(override_path: Option<std::ffi::OsString>) -> Vec<std::path::PathBuf> {
    match override_path {
        Some(path) => vec![std::path::PathBuf::from(path)],
        None => ["/usr/share/zoneinfo", "/usr/lib/zoneinfo", "/usr/share/lib/zoneinfo", "/etc/zoneinfo"]
            .iter().map(std::path::PathBuf::from).collect(),
    }
}

fn version(directory: &Path) -> Option<String> {
    let text = std::fs::read_to_string(directory.join("tzdata.zi")).ok()?;
    text.lines().find_map(|line| line.strip_prefix("# version ").map(|v| v.trim().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Barrier, atomic::{AtomicUsize, Ordering}};

    fn fixture_rules(key: &str) -> Arc<ZoneRules> {
        let bytes = super::super::tests_tzif::fixture(0, &[], &[(0, false, 0)], b"UTC\0", "");
        ZoneRules::load(key, bytes, "test".into(), None).unwrap()
    }

    #[test]
    fn concurrent_first_load_initializes_once_and_shares_the_arc() {
        let registry = Registry::default();
        let barrier = Barrier::new(8);
        let reads = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8).map(|_| scope.spawn(|| {
                barrier.wait();
                load_once(&registry, "Test/Concurrent", || {
                    reads.fetch_add(1, Ordering::Relaxed);
                    Ok(fixture_rules("Test/Concurrent"))
                }).unwrap()
            })).collect();
            let rules: Vec<_> = handles.into_iter().map(|handle| handle.join().unwrap()).collect();
            assert!(rules.iter().all(|rule| Arc::ptr_eq(&rules[0], rule)));
        });
        assert_eq!(reads.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn failed_initialization_can_retry_and_success_never_reloads() {
        let registry = Registry::default();
        assert!(load_once(&registry, "Test/Retry", || Err(VolasError::Value("unavailable".into()))).is_err());
        let first = load_once(&registry, "Test/Retry", || Ok(fixture_rules("Test/Retry"))).unwrap();
        let second = load_once(&registry, "Test/Retry", || panic!("cached name must not read its source")).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn parsed_names_keep_their_rules_after_byte_cache_eviction() {
        let bytes = super::super::tests_tzif::fixture(0, &[], &[(1, false, 0)], b"TEST\0", "");
        let supplied = Tz::from_tzif("Etc/UTC", bytes.clone(), "test".into(), None).unwrap();
        assert_eq!(supplied.offset_secs_at(0), 1);
        let Tz::Named(first) = Tz::parse("Etc/UTC").unwrap() else { panic!() };
        assert_eq!(Tz::Named(first.clone()).offset_secs_at(0), 0);
        for n in 0..65 {
            Tz::from_tzif(&format!("Test/NativeCache{n}"), bytes.clone(), "test".into(), None).unwrap();
        }
        let Tz::Named(second) = Tz::parse("Etc/UTC").unwrap() else { panic!() };
        assert!(Arc::ptr_eq(&first, &second));

        // Explicit bytes under the same key describe an independent snapshot.
        let supplied = Tz::from_tzif("Etc/UTC", bytes, "test".into(), None).unwrap();
        assert_eq!(supplied.offset_secs_at(0), 1);
        let Tz::Named(third) = Tz::parse("Etc/UTC").unwrap() else { panic!() };
        assert!(Arc::ptr_eq(&first, &third));
        assert_eq!(Tz::Named(third).offset_secs_at(0), 0);
    }

    #[test]
    fn keys_stay_inside_the_data_directory() {
        for key in ["", "/etc/passwd", "../UTC", "a/../UTC", "a/./UTC", "a//UTC", "UTC/", "a\\UTC", "a\0UTC"] {
            assert!(!valid_key(key), "{key:?}");
            assert!(load(key).is_err());
        }
        assert!(valid_key("America/New_York"));
        assert!(load("Not/A_Timezone").is_err());
        assert_eq!(directories(Some("/custom/zoneinfo".into())), [std::path::PathBuf::from("/custom/zoneinfo")]);
        assert_eq!(directories(None)[0], Path::new("/usr/share/zoneinfo"));
    }

    #[test]
    fn version_is_optional_and_read_from_the_selected_directory() {
        let temp = std::env::temp_dir().join(format!("volas-tz-version-{}", std::process::id()));
        std::fs::create_dir_all(&temp).unwrap();
        assert_eq!(version(&temp), None);
        std::fs::write(temp.join("tzdata.zi"), "# source\n# version 2026b\n").unwrap();
        assert_eq!(version(&temp).as_deref(), Some("2026b"));
        std::fs::write(temp.join("tzdata.zi"), "# unknown\n").unwrap();
        assert_eq!(version(&temp), None);
        std::fs::remove_dir_all(temp).unwrap();
    }
}
