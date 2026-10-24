//! Free helpers backing the Series surface (arith / compare / logical
//! resolution, the typed-scalar fill, slicing, and small constructors).

use std::sync::Arc;

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PySlice;
use volas_core::{
    fits, BinOp, BoolOp, CmpOp, Column, DType, Index, Series,
};

#[allow(unused_imports)]
use crate::*;

pub(crate) fn slice_series(s: &Series, slice: &Bound<'_, PySlice>) -> PyResult<PySeries> {
    let len = s.len();
    let info = slice.indices(len as isize)?;
    let (start, stop, step) = (info.start, info.stop, info.step);
    let positions = strided(start, stop, step);
    let data = s.data.take(&positions);
    let index = Arc::new(s.index.take(&positions));
    Ok(PySeries {
        inner: Series::new(s.name.clone(), data, index),
    })
}

/// The RHS of a Series binary op as an `f64` vector — another Series (aligned by
/// position) or a broadcast scalar. Two Series must share an index (see
/// [`require_aligned`]); volas never silently aligns by label.
/// The right-hand operand of a Series arithmetic op as a length-aligned column.
/// A Series must share the index; a scalar broadcasts with a *type-based* dtype
/// (Python `int`/`bool` -> i64, `float` -> f64) so `int_series + 2 -> int64` but
/// `+ 2.0 -> float64`, matching pandas. Anything else is unsupported.
pub(crate) fn series_rhs_col(s: &Series, other: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(o) = other.extract::<PyRef<PySeries>>() {
        require_aligned(&s.index, &o.inner.index)?;
        Ok(o.inner.data.clone())
    } else if other.is_none()
        || other.is(crate::scalar::na(other.py()).bind(other.py()))
        || other.get_type().name().map(|n| n == "NAType").unwrap_or(false)
    {
        // F7: an NA scalar operand poisons the whole result (known op unknown =
        // unknown), dtype-preserved — consistent with column-NA propagation and
        // with accepting np.nan. Broadcast an all-NA column of the series' dtype.
        Ok(Column::na_of(s.data.dtype(), s.len()))
    } else if let Ok(b) = other.extract::<bool>() {
        Ok(Column::i64(vec![b as i64; s.len()]))
    } else if let Ok(i) = other.extract::<i64>() {
        Ok(Column::i64(vec![i; s.len()]))
    } else if let Ok(x) = other.extract::<f64>() {
        Ok(Column::f64(vec![x; s.len()]))
    } else {
        Err(PyTypeError::new_err(
            "unsupported operand for a Series operation",
        ))
    }
}

/// Guard a positional binary op between two labelled operands — the single
/// alignment rule, owned by [`Index::require_aligned`]. Same-frame columns share
/// the index allocation (pointer identity, O(1)); otherwise the labels are compared. A
/// mismatch is an error rather than a silently misaligned (positional) result —
/// volas does not auto-align by label.
pub(crate) fn require_aligned(a: &Index, b: &Index) -> PyResult<()> {
    if std::ptr::eq(a, b) {
        return Ok(());
    }
    a.require_aligned(b).map_err(pyerr)
}

/// Equality returns False for different labels, but a datetime awareness mismatch
/// is still an invalid comparison. Empty datetime slices preserve that metadata:
/// the alignment owner checks it without making a label mismatch an error here.
pub(crate) fn require_comparable_indexes(a: &Index, b: &Index) -> PyResult<()> {
    if !std::ptr::eq(a, b)
        && matches!((a.kind(), b.kind()), (volas_core::IndexKind::Datetime(..), volas_core::IndexKind::Datetime(..)))
    {
        a.slice(0, 0).require_aligned(&b.slice(0, 0)).map_err(pyerr)?;
    }
    Ok(())
}

pub(crate) fn indexes_equal(a: &Index, b: &Index) -> PyResult<bool> {
    require_comparable_indexes(a, b)?;
    Ok(std::ptr::eq(a, b) || a.label_eq(b))
}

/// A new F64 `Series` carrying `s`'s name and index.
pub(crate) fn f64_series(s: &Series, out: Vec<f64>) -> PySeries {
    PySeries {
        inner: Series::new(s.name.clone(), Column::f64(out), Arc::clone(&s.index)),
    }
}

/// A new `Series` carrying `s`'s name and index over an already-built column.
/// Used by the dtype-preserving transforms (the typed Column ops decide dtype).
pub(crate) fn col_to_series(s: &Series, data: Column) -> PySeries {
    PySeries {
        inner: Series::new(s.name.clone(), data, Arc::clone(&s.index)),
    }
}

/// The pandas `describe` row labels (the index of a describe result).
pub(crate) fn describe_labels() -> Vec<String> {
    ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// A new Bool `Series` carrying `s`'s name and index.
pub(crate) fn bool_series(s: &Series, out: Vec<bool>) -> PySeries {
    PySeries {
        inner: Series::new(s.name.clone(), Column::bool(out), Arc::clone(&s.index)),
    }
}

/// A Series `+ - *` op against `other` (scalar / aligned Series), dtype-preserving
/// via the typed [`Column::binary`]. `swap` puts `other` on the left (the reflected
/// `__radd__` etc.). True division is separate ([`series_div`], always float).
pub(crate) fn series_arith(
    s: &Series,
    other: &Bound<'_, PyAny>,
    op: BinOp,
    swap: bool,
) -> PyResult<PySeries> {
    let rhs = series_rhs_col(s, other)?;
    let (lhs, rhs) = if swap { (&rhs, &s.data) } else { (&s.data, &rhs) };
    Ok(col_to_series(s, lhs.binary(rhs, op).map_err(pyerr)?))
}

/// A Series `/` op (always float). `swap` reflects it (`__rtruediv__`).
pub(crate) fn series_div(s: &Series, other: &Bound<'_, PyAny>, swap: bool) -> PyResult<PySeries> {
    let rhs = series_rhs_col(s, other)?;
    let (lhs, rhs) = if swap { (&rhs, &s.data) } else { (&s.data, &rhs) };
    Ok(col_to_series(s, lhs.div(rhs).map_err(pyerr)?))
}

/// A Series `//` op (floor division, dtype-preserving). `swap` reflects it
/// (`__rfloordiv__`).
pub(crate) fn series_floordiv(s: &Series, other: &Bound<'_, PyAny>, swap: bool) -> PyResult<PySeries> {
    let rhs = series_rhs_col(s, other)?;
    let (lhs, rhs) = if swap { (&rhs, &s.data) } else { (&s.data, &rhs) };
    Ok(col_to_series(s, lhs.floordiv(rhs).map_err(pyerr)?))
}

/// Element-wise comparison -> bool Series (positional), dtype-aware via
/// [`Column::compare`]: `str` / `datetime` / `bool` compare by native value (no f64
/// funnel), numeric exactly (no float64 round trip). A missing slot follows IEEE (`!=` true, else false).
/// The right operand is built to the left column's dtype (a str scalar for a str
/// column, a parsed timestamp for a datetime column, a number for a numeric one).
pub(crate) fn series_cmp(s: &Series, other: &Bound<'_, PyAny>, op: CmpOp) -> PyResult<PySeries> {
    let rhs = compare_rhs_col(s, other)?;
    Ok(col_to_series(s, s.data.compare(&rhs, op).map_err(pyerr)?))
}

/// Build the right operand of a comparison as a column matching the left column's
/// dtype: a `Series` contributes its own column (index-aligned); a scalar is
/// broadcast and typed by the left dtype.
pub(crate) fn compare_rhs_col(s: &Series, other: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Ok(o) = other.extract::<PyRef<PySeries>>() {
        require_aligned(&s.index, &o.inner.index)?;
        return Ok(o.inner.data.clone());
    }
    cmp_scalar_col(other, s.data.dtype(), s.len())
}

/// Broadcast a comparison scalar to `n` rows, typed for a `dtype` column: a `str`
/// scalar for a `Str` column, a parsed timestamp for a `Datetime` column,
/// otherwise a bool / int / float column. A scalar that cannot match the dtype is
/// a `TypeError` (rather than a silent all-`False` mask).
pub(crate) fn cmp_scalar_col(v: &Bound<'_, PyAny>, dtype: DType, n: usize) -> PyResult<Column> {
    match dtype {
        DType::Utf8 => {
            let s = v.extract::<String>().map_err(|_| {
                PyTypeError::new_err("cannot compare a str column with a non-string scalar")
            })?;
            Ok(Column::str(vec![s; n]))
        }
        DType::Datetime => Ok(Column::datetime(vec![parse_ts(v)?; n])),
        _ if v.extract::<bool>().is_ok() => Ok(Column::bool(vec![v.extract::<bool>()?; n])),
        _ if v.extract::<i64>().is_ok() => Ok(Column::i64(vec![v.extract::<i64>()?; n])),
        // a float, or an integer past int64 that float64 holds exactly (else a
        // TypeError: comparing against a rounded value would be silently wrong)
        _ if v.extract::<f64>().is_ok() => Ok(broadcast(&scalar_to_column(v, DType::F64)?, n)),
        _ => Err(PyTypeError::new_err("unsupported operand for a comparison")),
    }
}

/// The non-NaN `f64` values of a column for the NaN-skipping float64 reduction
/// `op` (an int64 past 2^53 raises, contract #26).
pub(crate) fn non_nan(col: &Column, op: &str) -> PyResult<Vec<f64>> {
    Ok(col.to_f64_domain(op).map_err(pyerr)?.into_iter().filter(|x| !x.is_nan()).collect())
}

/// A 1-row column repeated to `n` rows.
pub(crate) fn broadcast(cell: &Column, n: usize) -> Column {
    cell.take(&vec![0; n])
}

/// The position of the first maximum (`want_max`) or minimum present value,
/// dtype-aware (numeric by value, str lexically, datetime by raw i64 — NOT the
/// f64 funnel, which loses sub-256ns datetime ordering); errors on an all-NA
/// column. Backs `Series.idxmax` / `idxmin`.
pub(crate) fn argext(col: &Column, want_max: bool) -> PyResult<usize> {
    col.arg_extreme(want_max)
        .ok_or_else(|| PyValueError::new_err("Encountered all NA values"))
}

/// A boolean mask / condition column as a `Vec<bool>`, rejecting any missing
/// cell (O5) through the core truth owner ([`Column::truth_at`], contract #27). A
/// missing condition is an *unknown* signal — silently reading it as `False`
/// would drop a row (filtering) or fill it in the `False` direction (`where` /
/// `mask`) exactly like a deliberate negative, which in a live system turns a
/// data gap into a trade signal. The user must fill or drop the NA first.
pub(crate) fn bool_mask_vec(col: &Column) -> PyResult<Vec<bool>> {
    (0..col.len())
        .map(|i| {
            col.truth_at(i).ok_or_else(|| {
                PyValueError::new_err(
                    "boolean mask/condition contains volas.NA; an unknown signal is not \
                     treated as False — fill or drop the NA before masking",
                )
            })
        })
        .collect()
}

/// A recognized key keeps a NumPy read borrow alive until selection completes.
/// Dense arrays remain borrowed; strided arrays gather in logical order.
pub(crate) enum BoolMask<'py> {
    Array(PyReadonlyArray1<'py, bool>),
    Owned(Vec<bool>),
}

impl BoolMask<'_> {
    pub(crate) fn values(&self) -> std::borrow::Cow<'_, [bool]> {
        match self {
            Self::Array(a) => crate::coerce::numpy_values(a),
            Self::Owned(v) => std::borrow::Cow::Borrowed(v),
        }
    }
}

/// Recognise a boolean-mask key (`s[mask]` / `df[mask] = v`): a boolean Series, a
/// boolean ndarray, or a non-empty `list[bool]`. Returns `None` for any other key
/// so the caller can fall through to its label / position / column handling.
pub(crate) fn bool_mask_key<'a, 'py>(
    key: &Bound<'py, PyAny>,
    labels: impl FnOnce() -> std::borrow::Cow<'a, Index>,
) -> PyResult<Option<BoolMask<'py>>> {
    if let Ok(s) = key.extract::<PyRef<PySeries>>() {
        return Ok(match &s.inner.data {
            Column::Bool(..) => {
                require_aligned(&labels(), &s.inner.index)?;
                Some(BoolMask::Owned(bool_mask_vec(&s.inner.data)?))
            },
            _ => None,
        });
    }
    if let Ok(arr) = key.extract::<PyReadonlyArray1<bool>>() {
        // MaskedArray is an ndarray subclass: read validity before taking its buffer.
        if let Some(col) = crate::coerce::masked_array_column(key)? {
            return Ok(Some(BoolMask::Owned(bool_mask_vec(&col)?)));
        }
        return Ok(Some(BoolMask::Array(arr)));
    }
    match key.extract::<Vec<bool>>() {
        Ok(m) if !m.is_empty() => Ok(Some(BoolMask::Owned(m))),
        _ => Ok(None),
    }
}

/// Resolve the `other` argument of `where` / `mask` to a length-`n` fill column
/// plus the dtype it contributes to the result. A scalar broadcasts (its dtype is
/// value-based: an integral value contributes int); a Series contributes its own
/// dtype (index-aligned); the default (`None`) fills a dtype-preserving NA.
pub(crate) fn where_other_resolve(
    other: Option<&Bound<'_, PyAny>>,
    s: &Series,
) -> PyResult<(Column, DType)> {
    let n = s.len();
    match other {
        // the default `other` is a dtype-preserving all-NA column (str -> NA str,
        // datetime -> NaT, int/bool -> their NA), so a str/datetime `where` keeps
        // its kept values instead of funneling them to NaN.
        None => {
            let dt = s.data.dtype();
            Ok((Column::na_of(dt, n), dt))
        }
        // an explicit `None` / `NaN` / `volas.NA` fill is the same dtype-preserving
        // NA as the default, so `where(mask, volas.NA)` keeps the column's dtype.
        Some(o) if is_na_like_py(o) => {
            let dt = s.data.dtype();
            Ok((Column::na_of(dt, n), dt))
        }
        Some(o) => {
            if let Ok(ser) = o.extract::<PyRef<PySeries>>() {
                require_aligned(&s.index, &ser.inner.index)?;
                let dt = ser.inner.data.dtype();
                return Ok((ser.inner.data.clone(), dt));
            }
            scalar_fill_col(o, s.data.dtype(), n)
        }
    }
}

/// Resolve a scalar `other` fill to a length-`n` column typed to the target
/// `dtype`, plus the dtype it contributes. The shared typed-scalar rule for both
/// the Series and the (per-column) DataFrame `where` / `mask` / `fillna` surfaces:
/// a str scalar into a str column, a Timestamp / datetime string into a datetime
/// column, a bool/number into a numeric-family column. An incompatible scalar (a
/// number into a str or datetime column) is a `TypeError` — never a silent
/// numeric -> non-numeric coercion (C4).
pub(crate) fn scalar_fill_col(
    o: &Bound<'_, PyAny>,
    dtype: DType,
    n: usize,
) -> PyResult<(Column, DType)> {
    match dtype {
        DType::Utf8 => {
            let v = o.extract::<String>().map_err(|_| {
                PyTypeError::new_err("fill for a str column must be a string")
            })?;
            Ok((Column::str(vec![v; n]), DType::Utf8))
        }
        DType::Datetime => {
            // a bare number is not a datetime — require a volas.Timestamp or a
            // datetime string (parse_ts handles both); reject the rest so a number
            // cannot silently become an epoch-ns instant.
            if o.extract::<PyRef<PyTimestamp>>().is_err() && o.extract::<String>().is_err() {
                return Err(PyTypeError::new_err(
                    "fill for a datetime column must be a Timestamp or datetime string, not a number",
                ));
            }
            let ts = parse_ts(o)?;
            Ok((Column::datetime(vec![ts; n]), DType::Datetime))
        }
        // a bool column keeps bool for a bool fill or a 0/1 numeric fill; any other
        // number promotes the (numeric-family) bool column to float — matching
        // Column::fillna, so the Series and DataFrame surfaces agree on bool fills.
        DType::Bool => {
            if let Ok(b) = o.extract::<bool>() {
                return Ok((Column::bool(vec![b; n]), DType::Bool));
            }
            let x = o.extract::<f64>().map_err(|_| {
                PyTypeError::new_err("fill must be a number, a matching-dtype scalar, or a Series")
            })?;
            // F40 (decision 4): a bool column stays bool — 0/1 fills it; any other
            // number is an error, never a silent promotion to float (C3/C4 honesty).
            if x == 0.0 || x == 1.0 {
                Ok((Column::bool(vec![x != 0.0; n]), DType::Bool))
            } else {
                Err(PyTypeError::new_err("fill for a bool column must be a bool (or 0/1)"))
            }
        }
        // a numeric column with a bool fill contributes bool (value-based promotion).
        _ if o.extract::<bool>().is_ok() => {
            Ok((Column::bool(vec![o.extract::<bool>()?; n]), DType::Bool))
        }
        // an integer fill is carried as int64, exactly (contract #26: never via
        // float64, so 2**53 + 1 stays itself); the select converts it losslessly.
        // F1: it contributes the narrowest int width that fits, so an i32 column
        // filled with an integer stays int32 (# C2), not widened to int64.
        _ if o.extract::<i64>().is_ok() => {
            let i = o.extract::<i64>()?;
            let dt = if dtype == DType::I32 && i32::try_from(i).is_ok() { DType::I32 } else { DType::I64 };
            Ok((Column::i64(vec![i; n]), dt))
        }
        _ => {
            o.extract::<f64>().map_err(|_| {
                PyTypeError::new_err("fill must be a number, a matching-dtype scalar, or a Series")
            })?;
            // a float (or an integer past int64 that float64 holds exactly)
            let x = scalar_to_column(o, DType::F64)?.to_f64_vec()[0];
            let dt = match dtype {
                DType::I32 if fits(DType::I32, x) => DType::I32,
                _ if fits(DType::I64, x) => DType::I64,
                _ => DType::F64,
            };
            Ok((Column::f64(vec![x; n]), dt))
        }
    }
}

/// Element-wise boolean logic -> bool Series (both operands coerced to bool).
pub(crate) fn series_logical(s: &Series, other: &Bound<'_, PyAny>, op: BoolOp) -> PyResult<PySeries> {
    let n = s.data.len();
    let rhs: Column = if let Ok(o) = other.extract::<PyRef<PySeries>>() {
        require_aligned(&s.index, &o.inner.index)?;
        o.inner.data.clone()
    } else if is_na_like_py(other) {
        // a missing scalar (None / NaN / volas.NA / pd.NA) is an unknown truth
        // value: Kleene NA, never a present `NaN != 0 == True` (contract #27)
        Column::na_of(DType::Bool, n)
    } else if let Ok(b) = other.extract::<bool>() {
        Column::bool(vec![b; n])
    } else if let Ok(x) = other.extract::<f64>() {
        Column::bool(vec![x != 0.0; n])
    } else {
        return Err(PyTypeError::new_err(
            "unsupported operand for a Series logical op",
        ));
    };
    // Kleene three-valued logic, propagating volas.NA.
    Ok(col_to_series(s, s.data.logical(&rhs, op).map_err(pyerr)?))
}

pub(crate) fn strided(start: isize, stop: isize, step: isize) -> Vec<usize> {
    let mut out = Vec::new();
    if step > 0 {
        let mut i = start;
        while i < stop {
            out.push(i as usize);
            i += step;
        }
    } else if step < 0 {
        let mut i = start;
        while i > stop {
            out.push(i as usize);
            i += step;
        }
    }
    out
}
