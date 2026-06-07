//! `volas.TimeFrame` + the time-frame / aggregator resolvers shared by the
//! tf-aware DataFrame (`time_frame` constructor arg, `cumulate`).

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use volas_time::{Agg, AggSpec, TimeFrame};

use crate::{parse_ts, pyerr};

/// ``volas.TimeFrame`` — an OHLCV sampling period.
///
/// Use the predefined class constants as the ``time_frame`` argument to
/// ``df.cumulate`` / ``Cumulator``: ``s1``, ``m1``, ``m3``, ``m5``, ``m15``,
/// ``m30``, ``H1``, ``H2``, ``H4``, ``H6``, ``H8``, ``H12``, ``D1``, ``D3``,
/// ``W1``, ``M1``, ``Y1`` (label strings like ``'5m'`` / ``'1d'`` also work).
///
/// Usage::
///
///     df.cumulate(volas.TimeFrame.D1)   # daily bars
///     df.cumulate('15m')                # same idea, by label
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
    /// Map a timestamp (datetime string or epoch-ns int) to its **period key**:
    /// an opaque, monotonic integer that is equal for two timestamps iff they
    /// fall in the same bar of this timeframe (used internally by cumulation).
    ///
    /// Args:
    ///     ts (str | int): the timestamp to bucket.
    ///
    /// Returns:
    ///     int: the period key (compare for equality; not an epoch).
    fn unify(&self, ts: &Bound<'_, PyAny>) -> PyResult<i64> {
        Ok(self.inner.unify(parse_ts(ts)?))
    }

    #[classattr]
    fn s1() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Sec1,
        }
    }
    #[classattr]
    fn m1() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Min1,
        }
    }
    #[classattr]
    fn m3() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Min3,
        }
    }
    #[classattr]
    fn m5() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Min5,
        }
    }
    #[classattr]
    fn m15() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Min15,
        }
    }
    #[classattr]
    fn m30() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Min30,
        }
    }
    #[classattr]
    fn H1() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Hour1,
        }
    }
    #[classattr]
    fn H2() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Hour2,
        }
    }
    #[classattr]
    fn H4() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Hour4,
        }
    }
    #[classattr]
    fn H6() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Hour6,
        }
    }
    #[classattr]
    fn H8() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Hour8,
        }
    }
    #[classattr]
    fn H12() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Hour12,
        }
    }
    #[classattr]
    fn D1() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Day1,
        }
    }
    #[classattr]
    fn D3() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Day3,
        }
    }
    #[classattr]
    fn W1() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Week1,
        }
    }
    #[classattr]
    fn M1() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Month1,
        }
    }
    #[classattr]
    fn Y1() -> PyTimeFrame {
        PyTimeFrame {
            inner: TimeFrame::Year1,
        }
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

// `volas.Cumulator` was merged into the tf-aware DataFrame: build one with
// `DataFrame(..., time_frame=...)` or `df.cumulate(tf)`, then `append` fine bars
// (the forming period is `df.iloc[-1]`). See `PyDataFrame`.
