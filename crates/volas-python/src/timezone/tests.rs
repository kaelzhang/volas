use std::ffi::CString;
use std::sync::Arc;

use pyo3::ffi::c_str;

use super::*;

#[test]
fn concurrent_first_use_returns_the_stored_arc() {
    const KEY: &str = "America/New_York";
    pyo3::prepare_freethreaded_python();
    let controller = Python::with_gil(|py| {
        let source = CString::new(r#"
import threading
from volas import _timezone

original = _timezone._resolve
barrier = threading.Barrier(8)
reads = 0

def resolve(key):
    global reads
    reads += 1
    data = original(key)
    barrier.wait(timeout=5)
    return data

def forbidden(key):
    raise AssertionError('a cached key must not resolve again')

def forbid_reads():
    _timezone._resolve = forbidden

def restore():
    _timezone._resolve = original

_timezone._resolve = resolve
"#).unwrap();
        PyModule::from_code(py, &source, c_str!("timezone_race.py"), c_str!("timezone_race"))
            .unwrap().unbind()
    });
    // Joining outside the GIL lets every resolver reach the Python barrier.
    let results = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..8).map(|_| {
            scope.spawn(|| Python::with_gil(|py| named(py, KEY)))
        }).collect();
        threads.into_iter().map(|thread| thread.join().unwrap()).collect::<Vec<_>>()
    });
    let (reads, warm) = Python::with_gil(|py| {
        let control = controller.bind(py);
        control.call_method0("forbid_reads").unwrap();
        let warm = named(py, KEY);
        control.call_method0("restore").unwrap();
        (control.getattr("reads").unwrap().extract::<usize>().unwrap(), warm)
    });
    assert_eq!(reads, 8, "all callers must resolve before the first insertion");
    let rules: Vec<_> = results.into_iter().chain([warm]).map(|result| {
        let Tz::Named(rules) = result.unwrap() else { panic!("expected named rules") };
        rules
    }).collect();
    assert!(rules.iter().all(|rule| Arc::ptr_eq(&rules[0], rule)));
}
