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

#[test]
fn explicit_export_queries_missing_only_when_needed() {
    initialize();
    Python::with_gil(|py| {
        crate::volas_rs(&PyModule::new(py, "volas_rs").unwrap()).unwrap();
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
                    for dtype in ["float64", "datetime64[ns]", "timedelta64[ns]", "U", "object"] {
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
source = DataFrame({'close': [1., 2., 3.],
    'time': np.array([1, 'NaT', 3], dtype='datetime64[ns]')},
    index=np.array([0, 60, 120], dtype='datetime64[s]'),
    time_frame='1m' if kind == 'tf' else None,
    window=2 if kind == 'window' else None,
    max_lookback=['ema:2'] if kind == 'window' else None)
source['ema:2']
if kind == 'stale':
    source.append(DataFrame({'close': [4.],
        'time': np.array([4], dtype='datetime64[ns]')},
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
