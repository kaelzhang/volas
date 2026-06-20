//! `DataFrame` import / export (NumPy, CSV, pandas, Arrow) and directive evaluation.


use numpy::{IntoPyArray, PyReadwriteArray2};
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList};
use volas_core::{
    Column, DataFrame,
};
use volas_directive::{execute, parse};

use crate::format::{
    cell_to_csv, csv_escape, index_label_csv,
};
#[allow(unused_imports)]
use crate::*;

/// The frame's 2-D NA mask (row-major `h × w`, `True` where the cell is missing) — the
/// index for an `na_value` fill into an exported NumPy matrix.
fn frame_na_mask<'py>(py: Python<'py>, df: &DataFrame) -> PyResult<Bound<'py, PyAny>> {
    let (h, w) = (df.height(), df.width());
    let cols = df.columns();
    let mut data = Vec::with_capacity(h * w);
    for i in 0..h {
        for c in cols {
            data.push(!c.is_valid(i));
        }
    }
    Ok(ndarray::Array2::from_shape_vec((h, w), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py)
        .into_any())
}

/// Guard a `fill_into` destination's shape against the expected `(h, w)`.
fn check_fill_shape(shape: &[usize], h: usize, w: usize) -> PyResult<()> {
    if shape != [h, w] {
        return Err(PyValueError::new_err(format!(
            "fill_into: `out` shape {shape:?} does not match the frame's ({h}, {w})"
        )));
    }
    Ok(())
}

/// Substitute `na_value` into the missing cells (`mask`) of an exported array, in place;
/// a no-op when either the mask or the value is absent.
fn fill_na_2d<'py>(
    arr: Bound<'py, PyAny>,
    mask: Option<&Bound<'py, PyAny>>,
    na_value: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if let (Some(m), Some(nv)) = (mask, na_value) {
        arr.set_item(m, nv)?;
    }
    Ok(arr)
}

/// A lossless 2-D `object` NumPy array of `df`: each cell its own typed Python value
/// (`volas.Timestamp` / `volas.NA` / str / number) via `scalar_to_py`. Backs the
/// default mixed-frame export and `to_numpy(dtype="object")` — taking a `&DataFrame`
/// so a windowed frame passes its logical M view.
fn object_array<'py>(py: Python<'py>, df: &DataFrame) -> PyResult<Bound<'py, PyAny>> {
    let cols = df.columns();
    let (h, w) = (df.height(), df.width());
    let rows = PyList::empty(py);
    for i in 0..h {
        let row = PyList::empty(py);
        for col in cols {
            row.append(scalar_to_py(py, col, i))?;
        }
        rows.append(row)?;
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", "object")?;
    let arr = py.import("numpy")?.call_method("array", (rows,), Some(&kwargs))?;
    // An empty or single-column frame can collapse to the wrong ndim; pin (h, w).
    arr.call_method1("reshape", ((h, w),))
}

#[pymethods]
impl PyDataFrame {

    /// Evaluate an indicator directive and return its values as a NumPy array.
    ///
    /// A pure, stateless evaluation: it parses and computes the directive and leaves
    /// the frame untouched — no column is created, and no cache is read or written.
    /// For a cached, incrementally-refreshed result use ``df[directive]`` (a Series)
    /// or ``df[directive].to_numpy()``.
    ///
    /// Args:
    ///     directive (str): the directive, e.g. ``'macd'``, ``'boll.upper:20'``,
    ///         ``'close > open'``.
    ///
    /// Usage::
    ///
    ///     df.exec('ma:5')   # ndarray of SMA(5); the frame is not modified
    ///
    /// Returns:
    ///     numpy.ndarray
    pub(crate) fn exec<'py>(&self, py: Python<'py>, directive: &str) -> PyResult<Bound<'py, PyAny>> {
        let node = parse(directive).map_err(directive_err)?;
        let col = execute(&self.inner, &node).map_err(value_err)?;
        Ok(column_into_numpy(py, col))
    }

    /// Gets a column from the frame by name (alias-aware), as a Series.
    ///
    /// Args:
    ///     key (str): the column name.
    ///
    /// Returns:
    ///     Series
    pub(crate) fn get_column(&self, key: &str) -> PyResult<PySeries> {
        let col = self.inner.column(key).map_err(pyerr)?.clone();
        Ok(self.present_series(key.to_string(), col))
    }

    /// A copy of the frame — preserving the cached directive columns / cursor and
    /// (for a tf-aware frame) the cumulation state, so the copy keeps folding.
    pub(crate) fn copy(&self) -> PyDataFrame {
        PyDataFrame {
            inner: self.inner.clone(),
            tf: self.tf.clone(),
            window: self.window.clone(),
        }
    }

    /// Convert to a `pandas.DataFrame`. pandas is imported lazily (only here), so
    /// volas stays pandas-free at import.
    #[pyo3(signature = (dtype_backend = "numpy"))]
    pub(crate) fn to_pandas<'py>(&self, py: Python<'py>, dtype_backend: &str) -> PyResult<Bound<'py, PyAny>> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        // 'numpy' (default): an int/bool column with NA exports as float64+NaN, the
        // most ecosystem-compatible form. 'numpy_nullable': a faithful, lossless
        // masked Int64 / boolean. Mirrors pandas' own `dtype_backend`.
        let nullable = match dtype_backend {
            "numpy" => false,
            "numpy_nullable" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "dtype_backend must be 'numpy' or 'numpy_nullable', got {other:?}"
                )))
            }
        };
        let pd = py.import("pandas")?;
        let data = PyDict::new(py);
        for (name, col) in df.names().iter().zip(df.columns()) {
            data.set_item(name, column_to_pandas(py, &pd, col, nullable)?)?;
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("index", index_to_numpy(py, df.index())?)?;
        let pdf = pd.call_method("DataFrame", (data,), Some(&kwargs))?;
        // A tz-aware frame exports a UTC-naive datetime64 index (index_to_numpy); restore the
        // display zone so the pandas index is tz-aware — a faithful round-trip with from_pandas.
        if let Some(tz) = self.tz() {
            let aware = pdf
                .getattr("index")?
                .call_method1("tz_localize", ("UTC",))?
                .call_method1("tz_convert", (&tz,))?;
            pdf.setattr("index", aware)?;
        }
        // Carry the index name onto the pandas index (pandas parity).
        if let Some(name) = df.index().name() {
            let renamed = pdf.getattr("index")?.call_method1("rename", (name,))?;
            pdf.setattr("index", renamed)?;
        }
        Ok(pdf)
    }

    /// Write the frame as CSV (pandas-subset). With no `path`, returns the CSV
    /// string. Datetime columns are written as formatted strings (round-trips
    /// with `read_csv`).
    // One parameter per pandas `to_csv` keyword — a struct would break the API.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (path = None, sep = ",", index = true, header = true, na_rep = "", columns = None, float_format = None))]
    pub(crate) fn to_csv(
        &self,
        path: Option<std::path::PathBuf>,
        sep: &str,
        index: bool,
        header: bool,
        na_rep: &str,
        columns: Option<Vec<String>>,
        float_format: Option<&str>,
    ) -> PyResult<Option<String>> {
        ensure_fresh(&self.inner)?;
        // Windowed: emit only the logical M rows (a zero-cost borrow when unbounded)
        // — the CSV never carries the hidden margin.
        let view = self.logical();
        let df = view.as_ref();
        let ff = parse_ff(float_format)?;
        let names = df.names();
        let positions: Vec<usize> = match &columns {
            Some(cols) => cols
                .iter()
                .map(|n| {
                    df.column_pos(n)
                        .ok_or_else(|| PyKeyError::new_err(format!("column \"{n}\" not found")))
                })
                .collect::<PyResult<_>>()?,
            None => (0..df.width()).collect(),
        };
        let mut out = String::new();
        if header {
            if index {
                // pandas writes the index name, or an empty field for an unnamed index.
                out.push_str(df.index().name().unwrap_or(""));
                out.push_str(sep);
            }
            let hdr: Vec<String> = positions
                .iter()
                .map(|&j| csv_escape(names[j].clone(), sep))
                .collect();
            out.push_str(&hdr.join(sep));
            out.push('\n');
        }
        for i in 0..df.height() {
            if index {
                out.push_str(&index_label_csv(df.index(), i));
                out.push_str(sep);
            }
            let cells: Vec<String> = positions
                .iter()
                .map(|&j| csv_escape(cell_to_csv(&df.columns()[j], i, na_rep, ff), sep))
                .collect();
            out.push_str(&cells.join(sep));
            out.push('\n');
        }
        match path {
            Some(p) => {
                std::fs::write(&p, out).map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(None)
            }
            None => Ok(Some(out)),
        }
    }

    /// Arrow PyCapsule stream protocol — exposes the whole frame to Arrow consumers
    /// (`pa.table(df)`, `pl.from_dataframe(df)`) as a single zero-copy `RecordBatch`.
    /// `requested_schema` is accepted and ignored (the native dtypes are exported).
    #[pyo3(signature = (requested_schema = None))]
    pub(crate) fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = requested_schema;
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        crate::arrow::frame_c_stream(py, df.names(), df.columns())
    }

    /// Build a volas `DataFrame` from a `pandas.DataFrame` (the inverse of
    /// `df.to_pandas()`). Numeric / bool columns are carried natively; a pandas
    /// **nullable** column (`Int64` / `boolean` / `string`) keeps its dtype + `volas.NA`;
    /// a plain-string `object` column becomes a `str` column (a mixed `object` column is
    /// rejected — volas has no `object` dtype). Datetime columns and a datetime *index*
    /// are carried as native `datetime64[ns]` instants (no string round-trip), and a
    /// tz-aware index keeps its zone for display. pandas is imported lazily, so volas
    /// stays pandas-free at import.
    #[staticmethod]
    pub(crate) fn from_pandas(py: Python<'_>, pdf: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        crate::convert::frame_from_pandas(py, pdf)
    }

    /// Build a DataFrame from any object exposing the Arrow stream protocol
    /// (`__arrow_c_stream__`) — a pyarrow `Table`, a polars `DataFrame`, … — zero-copy
    /// where dtypes match. The result carries a fresh `RangeIndex`.
    #[staticmethod]
    pub(crate) fn from_arrow(data: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let (names, cols) = crate::arrow::frame_from_arrow_obj(data)?;
        Ok(PyDataFrame::plain(DataFrame::new(names, cols, None).map_err(pyerr)?))
    }

    /// Export as a `pyarrow.Table` (zero-copy where dtypes match; requires pyarrow).
    pub(crate) fn to_arrow<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("pyarrow")?.call_method1("table", (slf,))
    }

    /// The frame as a 2-D NumPy array (pandas `to_numpy`). `to_numpy` is an export
    /// boundary — leaving volas — so an explicit `dtype` is honored per cell like
    /// pandas (the internal no-lossy contract governs *computation*, not what the
    /// caller asks to convert *out* to). The matrix:
    ///
    /// | frame \\ dtype | `None` (default)        | `"object"`          | int / bool          | float                 |
    /// |---|---|---|---|---|
    /// | numeric / bool | `float64` matrix        | typed-cell object   | exact cast          | cast                  |
    /// | datetime       | 2-D `datetime64[ns]`    | `Timestamp` / `NA`  | **exact epoch-ns** (NaT→`i64::MIN`) | epoch-ns as float (lossy past 2⁵³, NaT→NaN) |
    /// | mixed          | object (typed cells)    | typed-cell object   | exact cast          | cast (str → error)    |
    /// | contains str   | object (typed cells)    | str kept            | **error**           | **error**             |
    ///
    /// So `dtype="object"` is always lossless (each cell its own typed value —
    /// `Timestamp` / `volas.NA` / str / number), an integer dtype takes the exact
    /// `i64` channel (datetime never round-trips through `f64`), a float dtype is
    /// the caller's opt-in lossy export, and a `str` column rejects any numeric
    /// dtype (no numeric meaning) pointing at `dtype="object"`.
    #[pyo3(signature = (dtype = None, na_value = None))]
    pub(crate) fn to_numpy<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&str>,
        na_value: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        ensure_fresh(&self.inner)?;
        // Windowed: export only the logical M rows (a zero-cost borrow when
        // unbounded) — the NN feature matrix is exactly the visible window.
        let view = self.logical();
        let df = view.as_ref();
        let cols = df.columns();
        let has_str = cols.iter().any(|c| matches!(c, Column::Str(..)));

        // `na_value` fill plumbing: with a value AND any missing cell, build the 2-D NA
        // mask once and substitute `na_value` into the holes of whichever array is built.
        let nv = na_value.as_ref();
        let mask = (nv.is_some() && cols.iter().any(|c| c.null_count() > 0))
            .then(|| frame_na_mask(py, df))
            .transpose()?;
        let mask = mask.as_ref();

        if let Some(dt) = dtype {
            // `object`: a lossless typed-cell array (never the f64 channel) — the
            // inspection / interop export that keeps datetime, str and NA intact.
            if dt == "object" || dt == "O" {
                return fill_na_2d(object_array(py, df)?, mask, nv);
            }
            // Any numeric / temporal target: a str column has no numeric value
            // (pandas raises here too), so reject it and point at the object route.
            if has_str {
                return Err(PyValueError::new_err(format!(
                    "cannot convert a string column to {dt}; use dtype='object' to keep strings"
                )));
            }
            // A float target rides the (lossy) f64 channel — the caller opted into
            // float, so a datetime epoch-ns past 2⁵³ quantises and a NaT becomes
            // NaN. A non-float (integer / bool / datetime) target takes the EXACT
            // i64 channel so a datetime exports its true epoch-ns (NaT→i64::MIN)
            // and a large i64 column survives, instead of an f64 round-trip.
            let floaty = dt.contains("float")
                || dt == "f32"
                || dt == "f64"
                || dt == "double"
                || dt == "single";
            // The two channels carry different Vec element types, so each builds its
            // own (type-erased) NumPy matrix; the fill + cast is then shared.
            let arr = if floaty {
                let (data, h, w) = df.to_row_major_f64();
                ndarray::Array2::from_shape_vec((h, w), data)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .into_pyarray(py)
                    .into_any()
            } else {
                // The i64 channel serves both integer and `datetime64` targets. A
                // datetime NaT → `i64::MIN` is its documented exact export, but a plain
                // int/float NA has no integer representation — raise for a true integer
                // target (pandas-aligned) UNLESS `na_value` gives an explicit fill,
                // exempting datetime columns' sentinel.
                if nv.is_none()
                    && is_integer_dtype(py, dt)?
                    && cols
                        .iter()
                        .any(|c| !matches!(c, Column::Datetime(..)) && c.null_count() > 0)
                {
                    return Err(PyValueError::new_err(format!(
                        "cannot convert a frame with missing values to integer NumPy dtype \
                         '{dt}' (an NA has no integer representation) — pass na_value=, or use \
                         a float dtype"
                    )));
                }
                let (data, h, w) = df.to_row_major_i64();
                ndarray::Array2::from_shape_vec((h, w), data)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .into_pyarray(py)
                    .into_any()
            };
            return fill_na_2d(arr, mask, nv)?.call_method1("astype", (dt,));
        }

        // Default (no dtype) — the honest representation chosen by the dtypes.
        // Fast path: an entirely numeric/bool frame is exactly the f64 matrix (an
        // empty frame counts — `all` over no columns is true, checked first).
        let all_numeric = cols.iter().all(|c| {
            matches!(
                c,
                Column::F64(_)
                    | Column::F32(_)
                    | Column::I64(..)
                    | Column::I32(..)
                    | Column::Bool(..)
            )
        });
        if all_numeric {
            let (data, h, w) = df.to_row_major_f64();
            let arr = ndarray::Array2::from_shape_vec((h, w), data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_pyarray(py)
                .into_any();
            return fill_na_2d(arr, mask, nv);
        }
        // A datetime-only frame -> a 2-D `datetime64[ns]` built DIRECTLY from the
        // raw i64 ns buffer (ns-exact, NaT = i64::MIN native). The old path boxed
        // each cell into a Python `Timestamp` then let NumPy re-coerce it, which
        // truncated ns and failed outright on a NaT cell (P1-01 / D1 / D2).
        if cols.iter().all(|c| matches!(c, Column::Datetime(_))) {
            let (data, h, w) = df.to_row_major_i64();
            let arr = ndarray::Array2::from_shape_vec((h, w), data)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_pyarray(py)
                .call_method1("astype", ("datetime64[ns]",))?;
            return fill_na_2d(arr, mask, nv);
        }
        // Any other mix -> a lossless object array, each cell its own typed value.
        fill_na_2d(object_array(py, df)?, mask, nv)
    }

    /// Write the frame's values into a caller-preallocated 2-D NumPy array, in place
    /// — the zero-allocation feature-export hot path for a windowed (NN-input) frame.
    ///
    /// ``out`` must be a C-contiguous ``float32`` or ``float64`` array of shape
    /// ``(len(df), k)`` where ``k`` is the number of selected columns (all numeric /
    /// bool columns by default, or those named in ``columns=``). Each cell is the
    /// column value cast to the array dtype; a missing cell becomes ``NaN``. Unlike
    /// ``to_numpy`` (which allocates a fresh matrix every call), ``fill_into`` reuses
    /// the same buffer across rounds — so a live ``append`` → ``fill_into`` inference
    /// loop allocates nothing per bar.
    ///
    /// Args:
    ///     out (numpy.ndarray): the destination, shape ``(len(df), k)``, dtype
    ///         ``float32`` or ``float64``.
    ///     columns (list[str], optional): the columns to export, in order. Defaults
    ///         to every column (a string column raises — there is no float meaning).
    #[pyo3(signature = (out, columns = None))]
    pub(crate) fn fill_into(&self, out: &Bound<'_, PyAny>, columns: Option<Vec<String>>) -> PyResult<()> {
        ensure_fresh(&self.inner)?;
        let view = self.logical();
        let df = view.as_ref();
        let positions: Vec<usize> = match &columns {
            Some(cols) => cols
                .iter()
                .map(|n| {
                    df.column_pos(n)
                        .ok_or_else(|| PyKeyError::new_err(format!("column \"{n}\" not found")))
                })
                .collect::<PyResult<_>>()?,
            None => (0..df.width()).collect(),
        };
        if let Some(c) = positions.iter().find(|&&j| matches!(df.columns()[j], Column::Str(..))) {
            return Err(PyValueError::new_err(format!(
                "fill_into: column \"{}\" is a string column (no float value) — exclude it via columns=",
                df.names()[*c]
            )));
        }
        let (h, w) = (df.height(), positions.len());
        // f32 first (the common NN dtype), then f64; anything else is rejected.
        if let Ok(mut arr) = out.extract::<PyReadwriteArray2<f32>>() {
            let mut a = arr.as_array_mut();
            check_fill_shape(a.shape(), h, w)?;
            for (jj, &j) in positions.iter().enumerate() {
                let v = df.columns()[j].to_f64_vec();
                for i in 0..h {
                    a[[i, jj]] = v[i] as f32;
                }
            }
            return Ok(());
        }
        if let Ok(mut arr) = out.extract::<PyReadwriteArray2<f64>>() {
            let mut a = arr.as_array_mut();
            check_fill_shape(a.shape(), h, w)?;
            for (jj, &j) in positions.iter().enumerate() {
                let v = df.columns()[j].to_f64_vec();
                for i in 0..h {
                    a[[i, jj]] = v[i];
                }
            }
            return Ok(());
        }
        Err(PyTypeError::new_err(
            "fill_into: `out` must be a float32 or float64 2-D NumPy array",
        ))
    }

    /// Value equality (same columns + index + values, `NaN == NaN`).
    pub(crate) fn equals(&self, other: &PyDataFrame) -> PyResult<bool> {
        ensure_fresh(&self.inner)?;
        ensure_fresh(&other.inner)?;
        Ok(self.logical().as_ref().equals(other.logical().as_ref()))
    }
}
