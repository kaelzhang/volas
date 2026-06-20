//! `Series` import / export (NumPy, Arrow, DLPack, list / dict / items).

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList, PyTuple};
use volas_core::{
    Index, Series,
};

#[allow(unused_imports)]
use crate::*;

#[pymethods]
impl PySeries {

    /// The values as a typed NumPy array; `dtype` casts (e.g. `'float32'`). Tracks
    /// `pandas.Series.to_numpy`: an integer `dtype` over missing values **raises** (an
    /// NA has no integer representation) unless `na_value` is given, in which case each
    /// missing cell is filled with it (the values stay exact for an explicit dtype).
    #[pyo3(signature = (dtype = None, na_value = None))]
    pub(crate) fn to_numpy<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&str>,
        na_value: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        column_to_numpy_with(py, &self.inner.data, dtype, na_value.as_ref())
    }

    /// NumPy array protocol, so `np.isnan(series)` etc. work directly. Honors a
    /// requested `dtype` (casts).
    #[pyo3(signature = (dtype = None, copy = None))]
    pub(crate) fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<PyObject>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // numpy 2.0: `copy=False` means "must not copy". `column_to_numpy` always
        // materialises a fresh array, so an explicit `copy=False` cannot be honored —
        // raise rather than silently return a copy (use __dlpack__ / to_arrow for a view).
        if copy == Some(false) {
            return Err(PyValueError::new_err(
                "volas cannot return a zero-copy NumPy array (to_numpy copies); pass \
                 copy=True/None, or use __dlpack__ / to_arrow for a borrowed view",
            ));
        }
        let arr = column_to_numpy(py, &self.inner.data);
        match dtype {
            Some(dt) => {
                let dt: String = dt.call_method0(py, "__str__")?.extract(py)?;
                astype_checked(py, arr, &self.inner.data, &dt)
            }
            None => Ok(arr),
        }
    }

    /// Arrow PyCapsule schema protocol: a lone `arrow_schema` capsule (the column's
    /// dtype), so Arrow consumers can read the type without materialising the data.
    pub(crate) fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        crate::arrow::column_c_schema(py, &self.inner.data)
    }

    /// Arrow PyCapsule array protocol — lets pyarrow / polars consume the series
    /// zero-copy (`pa.array(s)`, `pl.Series(s)`). Returns the `(schema, array)` capsule
    /// pair; `requested_schema` is accepted and ignored (we export the native dtype).
    #[pyo3(signature = (requested_schema = None))]
    pub(crate) fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let _ = requested_schema;
        crate::arrow::column_c_array(py, &self.inner.data)
    }

    /// Build a Series from any object implementing the Arrow array protocol
    /// (`__arrow_c_array__`) — a pyarrow `Array`, a polars `Series`, … — zero-copy where
    /// the dtypes line up. The result carries a fresh `RangeIndex`; `name` labels it.
    #[staticmethod]
    #[pyo3(signature = (data, name = None))]
    pub(crate) fn from_arrow(data: &Bound<'_, PyAny>, name: Option<String>) -> PyResult<PySeries> {
        let col = crate::arrow::column_from_arrow_obj(data)?;
        let n = col.len();
        Ok(PySeries { inner: Series::new(name, col, Arc::new(Index::range(n))) })
    }

    /// Export as a `pyarrow.Array` (zero-copy where dtypes match; requires pyarrow).
    pub(crate) fn to_arrow<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("pyarrow")?.call_method1("array", (slf,))
    }

    /// DLPack export (`np.from_dlpack(s)`, `torch.from_dlpack(s)`): borrow a dense numeric
    /// / bool column. A consumer that negotiates DLPack ≥ 1.0 via `max_version` gets a
    /// **read-only** view; an older consumer gets an independent **copy** (it has no
    /// read-only flag, so a shared borrow could bypass copy-on-write). `copy=True` forces a
    /// writable copy. A non-CPU `dl_device` or a `stream` is refused with `BufferError`, as
    /// is a missing value in an int/bool column or a str / datetime column.
    #[pyo3(signature = (stream = None, max_version = None, dl_device = None, copy = None))]
    pub(crate) fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<PyObject>,
        max_version: Option<(i32, i32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        crate::dlpack::column_to_dlpack(
            py,
            &self.inner.data,
            max_version,
            dl_device,
            copy,
            stream.is_some(),
        )
    }

    /// DLPack device: always CPU (`kDLCPU`, device 0).
    pub(crate) fn __dlpack_device__(&self) -> (i32, i32) {
        crate::dlpack::DEVICE_CPU
    }

    /// `{label: value}` (pandas `to_dict`); a missing value is `volas.NA`.
    pub(crate) fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for i in 0..self.inner.len() {
            d.set_item(
                label_to_py(py, &self.inner.index, i),
                scalar_to_py(py, &self.inner.data, i),
            )?;
        }
        Ok(d)
    }

    /// `[(label, value), ...]` (pandas `items()`, materialised).
    pub(crate) fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let pairs: Vec<(Py<PyAny>, Py<PyAny>)> = (0..self.inner.len())
            .map(|i| {
                (
                    label_to_py(py, &self.inner.index, i),
                    scalar_to_py(py, &self.inner.data, i),
                )
            })
            .collect();
        PyList::new(py, pairs)
    }

    pub(crate) fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items: Vec<Py<PyAny>> = (0..self.inner.len())
            .map(|i| scalar_to_py(py, &self.inner.data, i))
            .collect();
        PyList::new(py, items)
    }

    /// pandas-style equality: **same dtype** and value-equal (NaN equals NaN).
    pub(crate) fn equals(&self, other: &PySeries) -> bool {
        self.inner.data.dtype() == other.inner.data.dtype()
            && self.inner.data.equals(&other.inner.data)
    }
}
