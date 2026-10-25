//! Consuming NumPy export receives typed storage from the Column owner once.
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use volas_core::{Buffer, ColumnExport, StrBuffer, Validity};

pub(crate) struct NumpyExport<'py>(pub(crate) Python<'py>);
impl<'py> ColumnExport for NumpyExport<'py> {
    type Output = PyResult<Bound<'py, PyAny>>;
    fn f64(self, v: Buffer<f64>) -> Self::Output { Ok(v.into_vec().into_pyarray(self.0).into_any()) }
    fn f32(self, v: Buffer<f32>) -> Self::Output { Ok(v.into_vec().into_pyarray(self.0).into_any()) }
    fn i64(self, v: Buffer<i64>, valid: &Validity) -> Self::Output {
        Ok(if valid.has_nulls() {
            v.iter().enumerate().map(|(i, &x)| if valid.is_valid(i) { x as f64 } else { f64::NAN }).collect::<Vec<_>>().into_pyarray(self.0).into_any()
        } else { v.into_vec().into_pyarray(self.0).into_any() })
    }
    fn i32(self, v: Buffer<i32>, valid: &Validity) -> Self::Output {
        Ok(if valid.has_nulls() {
            v.iter().enumerate().map(|(i, &x)| if valid.is_valid(i) { x as f64 } else { f64::NAN }).collect::<Vec<_>>().into_pyarray(self.0).into_any()
        } else { v.into_vec().into_pyarray(self.0).into_any() })
    }
    fn bool(self, v: Buffer<bool>, valid: &Validity) -> Self::Output {
        if valid.has_nulls() { return Err(crate::convert::bool_na_err()); }
        Ok(v.into_vec().into_pyarray(self.0).into_any())
    }
    fn str(self, v: &StrBuffer, valid: &Validity) -> Self::Output { string_array(self.0, v, valid) }
    fn datetime(self, v: Buffer<i64>) -> Self::Output {
        v.into_vec().into_pyarray(self.0).call_method1("astype", ("datetime64[ns]",))
    }
}

pub(crate) fn string_array<'py>(py: Python<'py>, v: &StrBuffer, valid: &Validity) -> PyResult<Bound<'py, PyAny>> {
    let items: Vec<Bound<'_, PyAny>> = (0..v.len()).map(|i| {
        if valid.is_valid(i) { v.get(i).into_pyobject(py).unwrap().into_any() }
        else { py.None().into_bound(py) }
    }).collect();
    let list = PyList::new(py, items)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "object")?;
    py.import("numpy")?.call_method("array", (list,), Some(&kwargs))
}
