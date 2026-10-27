//! Observe entry-point work through buffer facts and the allocator, without hooks.
#[path = "../../../volas-core/tests/support/allocation.rs"]
mod allocation;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use volas_core::Column;

use crate::{columns_to_numpy_with, PyDataFrame};

unsafe extern "C" {
    fn budget_initialize_python();
}

fn initialize() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| unsafe { budget_initialize_python() });
    Python::with_gil(|py| {
        let sys = py.import("sys").unwrap();
        for (attribute, variable) in [
            ("prefix", "VOLAS_BUDGET_PREFIX"),
            ("executable", "VOLAS_BUDGET_EXECUTABLE"),
            ("exec_prefix", "VOLAS_BUDGET_EXEC_PREFIX"),
            ("base_prefix", "VOLAS_BUDGET_BASE_PREFIX"),
            ("base_exec_prefix", "VOLAS_BUDGET_BASE_EXEC_PREFIX"),
        ] {
            assert_eq!(
                sys.getattr(attribute).unwrap().extract::<String>().unwrap(),
                std::env::var(variable).unwrap(),
                "selected Python {attribute}"
            );
        }
        let origin = py
            .import("numpy")
            .unwrap()
            .getattr("__file__")
            .unwrap()
            .extract::<String>()
            .unwrap();
        assert_eq!(
            std::path::Path::new(&origin).canonicalize().unwrap(),
            std::path::Path::new(&std::env::var("VOLAS_BUDGET_NUMPY").unwrap())
                .canonicalize()
                .unwrap(),
            "selected NumPy origin"
        );
        eprintln!(
            "embedded prefix={}, numpy={origin}",
            sys.getattr("prefix").unwrap()
        );
    });
}

fn scan_samples(py: Python<'_>) {
    let clock = py
        .import("time")
        .unwrap()
        .getattr("process_time_ns")
        .unwrap();
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", false).unwrap();
    for datetime in [false, true] {
        // Large enough that the O(1) dtype/shape preparation is negligible next
        // to the mandatory native copy. No missing values: scans cannot exit early.
        let n = 1 << 23;
        let (col, dtype) = if datetime {
            (Column::Datetime(vec![1; n].into()), "datetime64[ns]")
        } else {
            (Column::f64(vec![1.; n]), "float64")
        };
        let descriptor = py
            .import("numpy")
            .unwrap()
            .call_method1("dtype", (dtype,))
            .unwrap();
        let reference = || {
            drop(
                crate::column_to_numpy(py, &col)
                    .unwrap()
                    .call_method("astype", (&descriptor,), Some(&kwargs))
                    .unwrap(),
            );
        };
        let actual = || {
            drop(
                columns_to_numpy_with(py, std::slice::from_ref(&col), n, dtype, None, false)
                    .unwrap(),
            );
        };
        for _ in 0..8 {
            reference();
            actual();
        }
        let elapsed = |operation: &dyn Fn()| {
            let start = clock.call0().unwrap().extract::<u64>().unwrap();
            operation();
            clock.call0().unwrap().extract::<u64>().unwrap() - start
        };
        let mut ratios = Vec::new();
        for pair in 0..200 {
            let (base, measured) = if pair % 2 == 0 {
                (elapsed(&reference), elapsed(&actual))
            } else {
                let measured = elapsed(&actual);
                (elapsed(&reference), measured)
            };
            ratios.push(measured as f64 / base as f64);
        }
        ratios.sort_by(f64::total_cmp);
        let median = (ratios[99] + ratios[100]) / 2.;
        eprintln!(
            "scan budget datetime={datetime}: q1={:.4}, median={median:.4}, q3={:.4}",
            ratios[50], ratios[150]
        );
        eprintln!("CPU_BUDGET {median}");
    }
}

fn scan_budget() {
    let mut samples = Vec::new();
    for run in 0..2 {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "entry_budget::explicit_export_queries_missing_only_when_needed",
                "--exact",
                "--test-threads=1",
                "--nocapture",
            ])
            .env("VOLAS_BUDGET_CPU_WORKER", "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        eprintln!("fresh CPU worker {run}:\n{stderr}");
        assert!(
            output.status.success() && stdout.contains("1 passed; 0 failed; 0 ignored"),
            "CPU worker failed: {stdout}"
        );
        let medians = stderr
            .lines()
            .filter_map(|line| line.strip_prefix("CPU_BUDGET "))
            .map(|value| value.parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(medians.len(), 2, "both source types must be measured");
        assert!(medians.iter().all(|x| x.is_finite() && *x > 0.));
        samples.push(medians);
    }
    for (source, (&first, &second)) in samples[0].iter().zip(&samples[1]).enumerate() {
        assert!(
            !(first > 1.02 && second > 1.02),
            "extra missing-scan CPU work: datetime={}, medians={:?}",
            source == 1,
            [first, second]
        );
    }
}

#[test]
fn explicit_export_queries_missing_only_when_needed() {
    initialize();
    Python::with_gil(|py| {
        crate::volas_rs(&PyModule::new(py, "volas_rs").unwrap()).unwrap();
        if std::env::var_os("VOLAS_BUDGET_CPU_WORKER").is_some() {
            scan_samples(py);
            return;
        }
        scan_budget();
        for frame in [false, true] {
            for dtype in [
                "float64",
                "datetime64[ns]",
                "timedelta64[ns]",
                "U",
                "object",
            ] {
                for filled in [false, true] {
                    for nan in [false, true] {
                        // A fresh owned buffer starts with an unknown NaN fact.
                        // has_missing learns that fact; copying/converting values does not.
                        let col = Column::f64(vec![1., if nan { f64::NAN } else { 2. }, 3.]);
                        let Column::F64(values) = &col else {
                            unreachable!()
                        };
                        assert_eq!(values.known_has_nan(), None);
                        let fill = 0i64.into_pyobject(py).unwrap().into_any();
                        let output = columns_to_numpy_with(
                            py,
                            std::slice::from_ref(&col),
                            3,
                            dtype,
                            filled.then_some(&fill),
                            frame,
                        )
                        .unwrap();
                        assert_eq!(
                            output.getattr("size").unwrap().extract::<usize>().unwrap(),
                            3
                        );
                        assert_eq!(
                            values.known_has_nan(),
                            filled.then_some(nan),
                            "frame={frame}, dtype={dtype}, filled={filled}, nan={nan}",
                        );
                    }
                }
            }
            for filled in [false, true] {
                for missing in [false, true] {
                    let col =
                        Column::Datetime(vec![1, if missing { i64::MIN } else { 2 }, 3].into());
                    let fill = 0i64.into_pyobject(py).unwrap().into_any();
                    for dtype in ["datetime64[ns]", "timedelta64[ns]", "U", "object"] {
                        let out = columns_to_numpy_with(
                            py,
                            std::slice::from_ref(&col),
                            3,
                            dtype,
                            filled.then_some(&fill),
                            frame,
                        )
                        .unwrap();
                        assert_eq!(out.getattr("size").unwrap().extract::<usize>().unwrap(), 3);
                    }
                }
            }
            // Positive control: an integer target needs the query even without a fill.
            let col = Column::f64(vec![1., f64::NAN, 3.]);
            let result =
                columns_to_numpy_with(py, std::slice::from_ref(&col), 3, "int64", None, frame);
            assert!(result.unwrap_err().to_string().contains("missing"));
            let Column::F64(values) = &col else {
                unreachable!()
            };
            assert_eq!(values.known_has_nan(), Some(true));
        }
    });
}

#[test]
fn no_override_constructor_takes_one_snapshot() {
    initialize();
    Python::with_gil(|py| {
        let module = PyModule::new(py, "volas_rs").unwrap();
        crate::volas_rs(&module).unwrap();
        let scope = PyDict::new(py);
        scope
            .set_item("DataFrame", module.getattr("DataFrame").unwrap())
            .unwrap();
        py.run(c"import numpy as np", Some(&scope), None).unwrap();
        for kind in ["plain", "window", "tf", "stale"] {
            scope.set_item("kind", kind).unwrap();
            py.run(
                c"
source = DataFrame({'close': [1., 2., 3.]},
    index=np.array([0, 60, 120], dtype='datetime64[s]'),
    time_frame='1m' if kind == 'tf' else None,
    window=2 if kind == 'window' else None,
    max_lookback=['ema:2'] if kind == 'window' else None)
source['ema:2']
if kind == 'stale':
    source.append(DataFrame({'close': [4.]},
        index=np.array([180], dtype='datetime64[s]')))
",
                Some(&scope),
                None,
            )
            .unwrap();
            let source = scope.get_item("source").unwrap().unwrap();
            let source: Bound<'_, PyDataFrame> = source.downcast_into().unwrap();
            let (reference, calls, bytes) = allocation::measure(|| {
                source
                    .as_any()
                    .extract::<PyRefMut<PyDataFrame>>()
                    .unwrap()
                    .copy()
            });
            assert!(
                calls > 0 && bytes > 0,
                "the fixture must observe snapshot work"
            );
            let (_, twice_calls, twice_bytes) = allocation::measure(|| {
                let source = source.as_any().extract::<PyRefMut<PyDataFrame>>().unwrap();
                (source.copy(), source.copy())
            });
            assert!(
                twice_calls > calls && twice_bytes > bytes,
                "two snapshots must cost more"
            );
            let (copied, actual_calls, actual_bytes) = allocation::measure(|| {
                PyDataFrame::new(source.as_any(), None, None, None, None, None, None, None).unwrap()
            });
            assert_eq!((actual_calls, actual_bytes), (calls, bytes), "kind={kind}");
            eprintln!("{kind}: copy=({calls}, {bytes}), constructor=({actual_calls}, {actual_bytes}), two copies=({twice_calls}, {twice_bytes})");
            assert_eq!(copied.inner.names(), reference.inner.names());
            assert_eq!(copied.inner.height(), reference.inner.height());
        }
    });
}
