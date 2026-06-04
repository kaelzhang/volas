//! `volas.TimeFrame` and `volas.Cumulator` — time-frame cumulation bindings.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use volas_time::{Agg, AggSpec, Cumulator, TimeFrame};

use crate::{parse_label, pyerr, PyDataFrame};

/// `volas.TimeFrame` — an OHLCV sampling period.
#[pyclass(name = "TimeFrame")]
#[derive(Clone)]
pub struct PyTimeFrame {
    inner: TimeFrame,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyTimeFrame {
    fn __str__(&self) -> String {
        self.inner.label().to_string()
    }
    fn __repr__(&self) -> String {
        format!("TimeFrame.{}", self.inner.label())
    }
    #[getter]
    fn minutes(&self) -> i64 {
        self.inner.minutes()
    }
    /// Unify a timestamp (datetime string or epoch-ns int) to its period key.
    fn unify(&self, ts: &Bound<'_, PyAny>) -> PyResult<i64> {
        Ok(self.inner.unify(parse_label(ts)?))
    }

    #[classattr]
    fn s1() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Sec1 }
    }
    #[classattr]
    fn m1() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Min1 }
    }
    #[classattr]
    fn m3() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Min3 }
    }
    #[classattr]
    fn m5() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Min5 }
    }
    #[classattr]
    fn m15() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Min15 }
    }
    #[classattr]
    fn m30() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Min30 }
    }
    #[classattr]
    fn H1() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Hour1 }
    }
    #[classattr]
    fn H2() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Hour2 }
    }
    #[classattr]
    fn H4() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Hour4 }
    }
    #[classattr]
    fn H6() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Hour6 }
    }
    #[classattr]
    fn H8() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Hour8 }
    }
    #[classattr]
    fn H12() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Hour12 }
    }
    #[classattr]
    fn D1() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Day1 }
    }
    #[classattr]
    fn D3() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Day3 }
    }
    #[classattr]
    fn W1() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Week1 }
    }
    #[classattr]
    fn M1() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Month1 }
    }
    #[classattr]
    fn Y1() -> PyTimeFrame {
        PyTimeFrame { inner: TimeFrame::Year1 }
    }
}

/// Resolve a `TimeFrame` from a `PyTimeFrame` or a label string.
pub(crate) fn resolve_time_frame(obj: &Bound<'_, PyAny>) -> PyResult<TimeFrame> {
    if let Ok(tf) = obj.extract::<PyRef<PyTimeFrame>>() {
        return Ok(tf.inner);
    }
    if let Ok(s) = obj.extract::<String>() {
        return TimeFrame::from_label(&s).map_err(pyerr);
    }
    Err(PyTypeError::new_err(
        "time_frame must be a TimeFrame or a label string like '5m'",
    ))
}

/// Build an aggregation spec from the OHLCV defaults plus optional overrides
/// (`{'volume': 'sum', 'open': 'first', ...}`).
pub(crate) fn build_agg_spec(cumulators: Option<&Bound<'_, PyDict>>) -> PyResult<AggSpec> {
    let mut spec = AggSpec::ohlcv();
    if let Some(dict) = cumulators {
        for (k, v) in dict.iter() {
            let name: String = k.extract()?;
            let agg_name: String = v.extract().map_err(|_| {
                PyTypeError::new_err("cumulator values must be aggregator names like 'sum'")
            })?;
            spec.set(name, Agg::from_name(&agg_name).map_err(pyerr)?);
        }
    }
    Ok(spec)
}

/// `volas.Cumulator` — a stateful, incremental OHLCV cumulator (live cum_append):
/// feed fine bars with `.append`, read the cumulated frame from `.frame`.
#[pyclass(name = "Cumulator")]
pub struct PyCumulator {
    inner: Cumulator,
}

#[pymethods]
impl PyCumulator {
    #[new]
    #[pyo3(signature = (time_frame, cumulators = None))]
    fn new(
        time_frame: &Bound<'_, PyAny>,
        cumulators: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let tf = resolve_time_frame(time_frame)?;
        let spec = build_agg_spec(cumulators)?;
        Ok(PyCumulator {
            inner: Cumulator::new(tf, spec),
        })
    }

    /// Feed fine bars (a DataFrame with a DatetimeIndex).
    fn append(&mut self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let df = data
            .extract::<PyRef<PyDataFrame>>()
            .map_err(|_| PyTypeError::new_err("Cumulator.append expects a DataFrame"))?;
        self.inner.append(&df.inner).map_err(pyerr)
    }

    /// The current cumulated frame (closed periods + the open period as the live
    /// last row).
    #[getter]
    fn frame(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame {
            inner: self.inner.frame().map_err(pyerr)?,
        })
    }

    /// The current open period aggregated into a single live bar, or `None`.
    #[getter]
    fn last(&self) -> PyResult<Option<PyDataFrame>> {
        Ok(self
            .inner
            .last()
            .map_err(pyerr)?
            .map(|inner| PyDataFrame { inner }))
    }
}
