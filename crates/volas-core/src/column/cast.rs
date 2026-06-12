//! `Column` dtype casting and datetime conversion (`astype` / `to_datetime`).

use super::*;

impl Column {
    /// Parse this column into a [`Column::Datetime`] (epoch ns). `Str` cells are
    /// parsed via [`datetime::parse_ns`]; an already-`Datetime` column is shared
    /// back (cheap). Errors on an unparseable cell or an unsupported dtype.
    pub fn to_datetime(&self) -> Result<Column> {
        self.to_datetime_tz(crate::tz::Tz::Utc)
    }

    /// Parse a string column to a UTC `Datetime` column, interpreting **naive**
    /// strings in `tz` (offset-aware strings are absolute; an existing `Datetime`
    /// column is already UTC and returned as-is). The tz is then attached to the
    /// *index* (storage stays UTC).
    pub fn to_datetime_tz(&self, tz: crate::tz::Tz) -> Result<Column> {
        match self {
            Column::Datetime(_) => Ok(self.clone()),
            Column::Str(v, val) => {
                let mut out = Vec::with_capacity(v.len());
                for (i, s) in v.iter().enumerate() {
                    // A missing (NA) string cell — or a present but empty/blank one —
                    // parses to NaT, matching read_csv(parse_dates) and pandas
                    // `to_datetime("")`. A non-empty unparseable string still errors.
                    if !val.is_valid(i) || s.trim().is_empty() {
                        out.push(i64::MIN);
                        continue;
                    }
                    let ns = datetime::parse_ns_in_tz(s, tz).ok_or_else(|| {
                        VolasError::Value(format!("could not parse datetime {s:?}"))
                    })?;
                    out.push(ns);
                }
                Ok(Column::datetime(out))
            }
            other => Err(VolasError::DType(format!(
                "cannot parse a {} column as datetime",
                other.dtype()
            ))),
        }
    }

    /// Interpret a numeric (epoch) column as a UTC `Datetime` column, scaling by
    /// `unit` (`"s"` / `"ms"` / `"us"` / `"ns"`). Float epochs are **truncated** to
    /// the whole `unit` (matching a NumPy / pandas `astype('datetime64[unit]')`
    /// cast). The robust ingestion path for exchange APIs that return numeric
    /// timestamps.
    pub fn epoch_to_datetime(&self, unit: &str) -> Result<Column> {
        self.epoch_to_datetime_with(unit, |x| datetime::epoch_to_ns(x as i64, unit))
    }

    /// Like [`epoch_to_datetime`](Self::epoch_to_datetime) but **rounds** float
    /// epochs to the nearest nanosecond, preserving sub-`unit` fractions (matching
    /// `pandas.to_datetime(..., unit=...)`). Identical for integer columns.
    pub fn epoch_to_datetime_rounded(&self, unit: &str) -> Result<Column> {
        self.epoch_to_datetime_with(unit, |x| datetime::epoch_to_ns_f64(x, unit))
    }

    /// Shared epoch → `Datetime` conversion; `f64_to_ns` chooses how float epochs
    /// map to nanoseconds (truncate vs round). Integers always scale exactly.
    fn epoch_to_datetime_with(
        &self,
        unit: &str,
        f64_to_ns: impl Fn(f64) -> Option<i64>,
    ) -> Result<Column> {
        match self {
            // A missing input maps to `NaT` (i64::MIN), not 1970 / an error: read
            // the i64 validity bit, and treat a float `NaN` as missing.
            Column::I64(v, val) => v
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    if !val.is_valid(i) {
                        Ok(i64::MIN)
                    } else {
                        datetime::epoch_to_ns(x, unit).ok_or_else(|| {
                            VolasError::Value(format!(
                            "could not convert epoch with unit {unit:?}: unknown unit or value out of nanosecond range"
                        ))
                        })
                    }
                })
                .collect::<Result<Vec<_>>>()
                .map(Column::datetime),
            Column::F64(v) => v
                .iter()
                .map(|&x| {
                    if x.is_nan() {
                        Ok(i64::MIN)
                    } else {
                        f64_to_ns(x).ok_or_else(|| {
                            VolasError::Value(format!(
                            "could not convert epoch with unit {unit:?}: unknown unit or value out of nanosecond range"
                        ))
                        })
                    }
                })
                .collect::<Result<Vec<_>>>()
                .map(Column::datetime),
            other => Err(VolasError::DType(format!(
                "cannot read a {} column as an epoch timestamp (need int64 / float64)",
                other.dtype()
            ))),
        }
    }

    /// Cast to another dtype (best-effort, pandas `astype`-like). A no-op when
    /// already the target dtype.
    pub fn cast(&self, to: DType) -> Result<Column> {
        if self.dtype() == to {
            return Ok(self.clone());
        }
        // An int/bool source to an int/bool target keeps its missing cells and
        // converts present values exactly (no f64 round-trip; range-checked).
        if matches!(self, Column::I64(..) | Column::I32(..) | Column::Bool(..))
            && matches!(to, DType::I64 | DType::I32 | DType::Bool)
        {
            return self.cast_int_bool(to);
        }
        // `astype` is an EXPLICIT cast, so a str source to a numeric target parses
        // each cell rather than funnelling to a silent all-NaN (`to_f64_vec` reads
        // a str column as NaN): a valid numeric string converts, an empty/missing
        // cell becomes NA, any other non-empty string raises (contract C4).
        if let Column::Str(v, val) = self {
            if matches!(to, DType::F64 | DType::F32 | DType::I64 | DType::I32) {
                return Self::cast_str_to_numeric(v, val, to);
            }
        }
        // D5: a datetime -> float cast is a lossy reinterpret of the i64 epoch-ns
        // (values past 2^53 quantize to ~256ns), so reject it (pandas does too).
        // `astype('int64')` gives the exact ns; an explicit `to_numpy(dtype=...)`
        // export is the deliberate opt-in lossy channel.
        if matches!(self, Column::Datetime(_)) && matches!(to, DType::F64 | DType::F32) {
            return Err(VolasError::DType(
                "cannot cast a datetime column to float (lossy past 2^53 ns); \
                 use astype('int64') for the exact epoch nanoseconds"
                    .to_string(),
            ));
        }
        // Epoch nanoseconds never fit int32 (except meaningless near-1970
        // instants), so a datetime -> int32 cast is rejected like the float one.
        if matches!(self, Column::Datetime(_)) && to == DType::I32 {
            return Err(VolasError::DType(
                "cannot cast a datetime column to int32 (epoch nanoseconds do not fit); \
                 use astype('int64') for the exact epoch nanoseconds"
                    .to_string(),
            ));
        }
        match to {
            DType::F64 => Ok(Column::f64(self.to_f64_vec())),
            DType::I64 => match self {
                Column::F64(v) => {
                    // pandas raises (IntCastingNaNError) rather than silently
                    // turning NaN -> 0 / inf -> i64::MAX, which corrupts data.
                    if let Some(x) = v.iter().copied().find(|x| !x.is_finite()) {
                        return Err(VolasError::Value(format!(
                            "cannot convert non-finite value ({x}) to int64 (NaN / inf); \
                             fill or drop it first"
                        )));
                    }
                    Ok(Column::i64(v.iter().map(|&x| x as i64).collect()))
                }
                // F32 truncates like F64 (consistent with F32 -> I32), finite-checked.
                Column::F32(v) => {
                    if let Some(x) = v.iter().copied().find(|x| !x.is_finite()) {
                        return Err(VolasError::Value(format!(
                            "cannot convert non-finite value ({x}) to int64 (NaN / inf); \
                             fill or drop it first"
                        )));
                    }
                    Ok(Column::i64(v.iter().map(|&x| x as i64).collect()))
                }
                // int/bool sources are handled by `cast_int_bool` above.
                // The exact epoch nanoseconds; a NaT cell stays missing in the
                // int64 validity (# C2) instead of surfacing the i64::MIN sentinel
                // as a value (pandas raises here; volas's native-NA int carries it).
                Column::Datetime(v) => {
                    let validity = Validity::from_valid_iter(
                        v.len(),
                        (0..v.len()).map(|i| self.is_valid(i)),
                    );
                    Ok(Column::i64_with(v.to_vec(), validity))
                }
                // unreachable: int/bool go through cast_int_bool, str through
                // cast_str_to_numeric, and F64/F32/Datetime are handled above.
                _ => unreachable!("non-numeric int64 cast handled earlier"), // LCOV_EXCL_LINE
            },
            // numeric -> bool is pandas-nullable truthiness (owner ruling
            // 2026-06-12): nonzero -> True (incl ±inf), zero -> False, and a
            // missing float cell (NaN) STAYS missing — never numpy's
            // NaN-is-truthy footgun. int sources go through `cast_int_bool`.
            DType::Bool => match self {
                Column::F64(v) => Ok(Column::bool_with(
                    v.iter().map(|&x| x != 0.0).collect(),
                    Validity::from_valid_iter(v.len(), v.iter().map(|x| !x.is_nan())),
                )),
                Column::F32(v) => Ok(Column::bool_with(
                    v.iter().map(|&x| x != 0.0).collect(),
                    Validity::from_valid_iter(v.len(), v.iter().map(|x| !x.is_nan())),
                )),
                // str -> bool has no parse policy (owner ruling: error), and a
                // datetime has no truthiness.
                other => Err(VolasError::DType(format!(
                    "cannot cast a {} column to bool",
                    other.dtype()
                ))),
            },
            DType::F32 => Ok(Column::f32(
                self.to_f64_vec().iter().map(|&x| x as f32).collect(),
            )),
            DType::I32 => self
                .to_f64_vec()
                .iter()
                .map(|&x| {
                    i32::try_from_f64(x).ok_or_else(|| {
                        VolasError::Value(format!(
                            "cannot convert {x} to int32 (non-finite / non-integral / out of range)"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()
                .map(Column::i32),
            // F4: EVERY source's missing cells stay missing in the str target —
            // including the in-band sentinels (float NaN, datetime NaT), which the
            // raw bitmap doesn't cover. Without this, stringify materialises the
            // sentinel as a literal 'NaN'/'NaT' value (isna -> False), a C2/C4
            // placeholder collapse. The unified is_valid covers all three NA forms.
            DType::Utf8 => {
                let n = self.len();
                let validity = crate::column::Validity::from_valid_iter(
                    n,
                    (0..n).map(|i| self.is_valid(i)),
                );
                Ok(Column::str_with(self.to_string_vec(), validity))
            }
            DType::Datetime => self.to_datetime(),
        }
    }

    /// Cast an int/bool column to another int/bool dtype, carrying its validity (a
    /// missing cell stays missing) and converting present values exactly (no f64
    /// round-trip). Only a present, out-of-range value (a large `i64` into `i32`)
    /// errors; a missing cell never does.
    fn cast_int_bool(&self, to: DType) -> Result<Column> {
        let validity = self.validity().cloned().unwrap_or_default();
        let narrow_i32 = |v: &[i64]| -> Result<Vec<i32>> {
            v.iter()
                .enumerate()
                .map(|(i, &x)| {
                    if validity.is_valid(i) {
                        i32::try_from(x).map_err(|_| {
                            VolasError::Value(format!("cannot convert {x} to int32 (out of range)"))
                        })
                    } else {
                        Ok(0)
                    }
                })
                .collect()
        };
        match (self, to) {
            (Column::I64(v, _), DType::I32) => Ok(Column::i32_with(narrow_i32(v)?, validity)),
            (Column::I64(v, _), DType::Bool) => Ok(Column::bool_with(
                v.iter().map(|&x| x != 0).collect(),
                validity,
            )),
            (Column::I32(v, _), DType::I64) => Ok(Column::i64_with(
                v.iter().map(|&x| x as i64).collect(),
                validity,
            )),
            (Column::I32(v, _), DType::Bool) => Ok(Column::bool_with(
                v.iter().map(|&x| x != 0).collect(),
                validity,
            )),
            (Column::Bool(v, _), DType::I64) => Ok(Column::i64_with(
                v.iter().map(|&b| b as i64).collect(),
                validity,
            )),
            (Column::Bool(v, _), DType::I32) => Ok(Column::i32_with(
                v.iter().map(|&b| b as i32).collect(),
                validity,
            )),
            _ => unreachable!("same-dtype is handled in cast"), // LCOV_EXCL_LINE
        }
    }

    /// Parse a str column into a numeric target for explicit `astype` (Q2): a
    /// present, non-empty cell parses (int target -> int literal, so `"1.5"`
    /// raises; float target -> float literal), an empty / whitespace / missing
    /// cell becomes NA, and any other non-empty string raises with the offending
    /// value. Never a silent NaN (contract C4). `to` is numeric (caller guard).
    fn cast_str_to_numeric(v: &[String], val: &Validity, to: DType) -> Result<Column> {
        // The cells we parse: present AND not blank. Blank / missing cells are NA
        // in the result (an int target carries this in its validity; a float
        // target writes NaN).
        let parse_me: Vec<bool> = (0..v.len())
            .map(|i| val.is_valid(i) && !v[i].trim().is_empty())
            .collect();
        let bad = |i: usize, target: &str| {
            VolasError::Value(format!(
                "cannot convert string {:?} to {target} (astype)",
                v[i]
            ))
        };
        match to {
            DType::F64 | DType::F32 => {
                let mut out = vec![f64::NAN; v.len()];
                for (i, slot) in out.iter_mut().enumerate() {
                    if parse_me[i] {
                        *slot = v[i].trim().parse::<f64>().map_err(|_| bad(i, "float64"))?;
                    }
                }
                Ok(if to == DType::F32 {
                    Column::f32(out.iter().map(|&x| x as f32).collect())
                } else {
                    Column::f64(out)
                })
            }
            // int targets: parse an integer literal (no funnel, so "1.5" raises
            // rather than truncating), carrying the blank/missing cells as NA.
            _ => {
                let validity = Validity::from_valid_iter(v.len(), parse_me.iter().copied());
                let mut out = vec![0i64; v.len()];
                for (i, slot) in out.iter_mut().enumerate() {
                    if parse_me[i] {
                        *slot = v[i].trim().parse::<i64>().map_err(|_| bad(i, "int64"))?;
                    }
                }
                if to == DType::I32 {
                    let narrowed = out
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| {
                            if parse_me[i] {
                                i32::try_from(x).map_err(|_| bad(i, "int32"))
                            } else {
                                Ok(0)
                            }
                        })
                        .collect::<Result<Vec<i32>>>()?;
                    Ok(Column::i32_with(narrowed, validity))
                } else {
                    Ok(Column::i64_with(out, validity))
                }
            }
        }
    }
}
