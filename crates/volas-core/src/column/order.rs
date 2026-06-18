//! `Column` structural & order-based operations — slice/take, unique,
//! sort/rank, the typed extreme/running-extreme, append, scatter, and equality.

use super::*;

impl Column {
    /// A contiguous `[start, end)` slice (a fresh buffer).
    pub fn slice(&self, start: usize, end: usize) -> Column {
        match self {
            Column::F64(v) => Column::f64(v[start..end].to_vec()),
            Column::F32(v) => Column::f32(v[start..end].to_vec()),
            Column::Bool(v, val) => {
                Column::bool_with(v[start..end].to_vec(), val.slice(start, end))
            }
            Column::I64(v, val) => Column::i64_with(v[start..end].to_vec(), val.slice(start, end)),
            Column::I32(v, val) => Column::i32_with(v[start..end].to_vec(), val.slice(start, end)),
            Column::Str(v, val) => Column::Str(v.slice(start, end), val.slice(start, end)),
            Column::Datetime(v) => Column::datetime(v[start..end].to_vec()),
        }
    }

    /// Gather the given positions into a new column (fancy indexing).
    pub fn take(&self, idx: &[usize]) -> Column {
        match self {
            Column::F64(v) => Column::f64(idx.iter().map(|&i| v[i]).collect()),
            Column::F32(v) => Column::f32(idx.iter().map(|&i| v[i]).collect()),
            Column::Bool(v, val) => {
                Column::bool_with(idx.iter().map(|&i| v[i]).collect(), val.take(idx))
            }
            Column::I64(v, val) => {
                Column::i64_with(idx.iter().map(|&i| v[i]).collect(), val.take(idx))
            }
            Column::I32(v, val) => {
                Column::i32_with(idx.iter().map(|&i| v[i]).collect(), val.take(idx))
            }
            // Gather straight into one contiguous `StrBuffer` (no intermediate
            // `Vec<String>`, so no per-cell allocation).
            Column::Str(v, val) => {
                Column::Str(idx.iter().map(|&i| v.get(i)).collect(), val.take(idx))
            }
            Column::Datetime(v) => Column::datetime(idx.iter().map(|&i| v[i]).collect()),
        }
    }

    /// Gather optional positions into a new column of the SAME dtype: `Some(i)`
    /// reads row `i`, `None` is a missing cell (dtype-preserving NA — int/bool/str
    /// grow a validity hole, float `NaN`, datetime `NaT`). Backs the window
    /// `first` / `last` aggregations.
    pub fn take_optional(&self, idx: &[Option<usize>]) -> Column {
        let validity = || {
            Validity::from_valid_iter(
                idx.len(),
                idx.iter().map(|p| p.is_some_and(|i| self.is_valid(i))),
            )
        };
        match self {
            Column::F64(v) => {
                Column::f64(idx.iter().map(|p| p.map_or(f64::NAN, |i| v[i])).collect())
            }
            Column::F32(v) => {
                Column::f32(idx.iter().map(|p| p.map_or(f32::NAN, |i| v[i])).collect())
            }
            Column::Bool(v, _) => Column::bool_with(
                idx.iter().map(|p| p.is_some_and(|i| v[i])).collect(),
                validity(),
            ),
            Column::I64(v, _) => Column::i64_with(
                idx.iter().map(|p| p.map_or(0, |i| v[i])).collect(),
                validity(),
            ),
            Column::I32(v, _) => Column::i32_with(
                idx.iter().map(|p| p.map_or(0, |i| v[i])).collect(),
                validity(),
            ),
            // `None` → empty placeholder (the validity marks it NA); gather builds the
            // `StrBuffer` in one pass.
            Column::Str(v, _) => Column::Str(
                idx.iter().map(|p| p.map_or("", |i| v.get(i))).collect(),
                validity(),
            ),
            Column::Datetime(v) => Column::datetime(
                idx.iter().map(|p| p.map_or(i64::MIN, |i| v[i])).collect(),
            ),
        }
    }

    /// Number of present (non-missing) values (pandas `count`): `len - null_count`,
    /// reading the validity for every dtype (a float `NaN`, an int/bool/str NA, a
    /// datetime `NaT`).
    pub fn count(&self) -> usize {
        self.len() - self.null_count()
    }

    /// Number of distinct present values (pandas `nunique`, `dropna=True`).
    pub fn nunique(&self) -> usize {
        self.group_records().iter().filter(|(_, _, na)| !na).count()
    }

    /// First-appearance index of each distinct value (pandas `unique` order),
    /// **including one missing slot** if the column has any NA — so `take`ing these
    /// indices yields the distinct values with a single `NA` where present.
    pub fn unique_indices(&self) -> Vec<usize> {
        self.group_records()
            .iter()
            .map(|(first, _, _)| *first)
            .collect()
    }

    /// Distinct-value group records `(first_index, count, is_na)` in order of first
    /// appearance — the shared basis of `nunique` / `unique` / `value_counts`. Every
    /// missing value (`NaN` / NA / `NaT`) collapses into one `is_na = true` group.
    pub(crate) fn group_records(&self) -> Vec<(usize, usize, bool)> {
        let len = self.len();
        match self {
            Column::F64(v) => group_by(len, |i| float_key(v[i])),
            Column::F32(v) => group_by(len, |i| float_key(v[i] as f64)),
            Column::I64(v, val) => group_by(len, |i| val.is_valid(i).then_some(v[i])),
            Column::I32(v, val) => group_by(len, |i| val.is_valid(i).then_some(v[i] as i64)),
            Column::Bool(v, val) => group_by(len, |i| val.is_valid(i).then_some(v[i] as i64)),
            Column::Str(v, val) => group_by(len, |i| val.is_valid(i).then(|| v.get(i).to_string())),
            Column::Datetime(v) => group_by(len, |i| (v[i] != i64::MIN).then_some(v[i])),
        }
    }

    /// Indices that sort the column (pandas `sort_values`), **stable**, with every
    /// missing value placed last regardless of `ascending` (pandas `na_position =
    /// 'last'`). Present values compare per dtype (float by value, str lexically).
    pub fn argsort(&self, ascending: bool) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.len()).collect();
        idx.sort_by(|&a, &b| match (self.is_valid(a), self.is_valid(b)) {
            (false, false) => Ordering::Equal,
            (false, true) => Ordering::Greater, // NA sinks to the end
            (true, false) => Ordering::Less,
            (true, true) => {
                let o = self.cmp_at(a, b);
                if ascending {
                    o
                } else {
                    o.reverse()
                }
            }
        });
        idx
    }

    /// Order two **present** values at `a` / `b` (helper for [`argsort`](Self::argsort);
    /// floats are non-`NaN` here, so `partial_cmp` is total).
    fn cmp_at(&self, a: usize, b: usize) -> Ordering {
        match self {
            Column::F64(v) => v[a].partial_cmp(&v[b]).unwrap_or(Ordering::Equal),
            Column::F32(v) => v[a].partial_cmp(&v[b]).unwrap_or(Ordering::Equal),
            Column::I64(v, _) => v[a].cmp(&v[b]),
            Column::I32(v, _) => v[a].cmp(&v[b]),
            Column::Bool(v, _) => v[a].cmp(&v[b]),
            // SAFETY: `a`/`b` are present-row indices from `argsort` / `arg_extreme`,
            // always `< len`; the unchecked accessor avoids redundant offset/data checks.
            Column::Str(v, _) => unsafe { v.get_unchecked(a).cmp(v.get_unchecked(b)) },
            Column::Datetime(v) => v[a].cmp(&v[b]),
        }
    }

    /// Position of the maximum (`want_max`) or minimum **present** value,
    /// dtype-aware via [`cmp_at`](Self::cmp_at) — numeric by value, `str`
    /// lexically, `datetime` by raw `i64` (so sub-256ns ordering survives, unlike
    /// the `to_f64_vec` funnel which collapses it past 2^53). Ties keep the FIRST
    /// occurrence (pandas `idxmax`/`idxmin`). `None` if every value is missing.
    /// The typed basis of `idxmax` / `idxmin` / `min` / `max`.
    pub fn arg_extreme(&self, want_max: bool) -> Option<usize> {
        let mut best: Option<usize> = None;
        for i in 0..self.len() {
            if !self.is_valid(i) {
                continue;
            }
            let take = match best {
                None => true,
                Some(b) => {
                    let o = self.cmp_at(i, b);
                    (want_max && o == Ordering::Greater) || (!want_max && o == Ordering::Less)
                }
            };
            if take {
                best = Some(i);
            }
        }
        best
    }

    /// Cumulative running extreme (pandas `cummax` if `want_max`, else `cummin`),
    /// dtype-aware and dtype-preserving via [`cmp_at`](Self::cmp_at): numeric by
    /// value, `str` lexically, `datetime` by instant. A present cell takes the
    /// running extreme of the present values seen so far; a missing cell stays
    /// missing (skipped, not filled — pandas `cummax([3, NA, 5]) == [3, NA, 5]`).
    /// Built by gathering each output position from the cell that holds its
    /// running extreme (or itself, when missing).
    pub fn cum_extreme(&self, want_max: bool) -> Result<Column> {
        let n = self.len();
        let mut positions: Vec<usize> = (0..n).collect(); // missing cells point at self -> stay NA
        let mut best: Option<usize> = None;
        for (i, slot) in positions.iter_mut().enumerate() {
            if self.is_valid(i) {
                best = Some(match best {
                    None => i,
                    Some(b) => {
                        let o = self.cmp_at(i, b);
                        if (want_max && o == Ordering::Greater)
                            || (!want_max && o == Ordering::Less)
                        {
                            i
                        } else {
                            b
                        }
                    }
                });
                *slot = best.expect("just set"); // present cell -> running extreme so far
            }
        }
        Ok(self.take(&positions))
    }

    /// Order-based rank (pandas `rank`, 1-based, missing -> `NaN`), dtype-aware
    /// via [`cmp_at`](Self::cmp_at): numeric by value, `str` lexically, `datetime`
    /// by raw `i64`, `bool` by `false < true`. The result is always `f64` (ties
    /// can average to `x.5`), so rank is order-based, not a numeric-arithmetic op.
    pub fn rank(&self, method: stats::RankMethod, ascending: bool, pct: bool) -> Vec<f64> {
        stats::rank_by(
            self.len(),
            |i| self.is_valid(i),
            |a, b| self.cmp_at(a, b),
            method,
            ascending,
            pct,
        )
    }

    /// Append another column of the same dtype, copy-on-write (grows in place
    /// when the buffer is uniquely owned).
    pub fn append(&mut self, other: &Column) -> Result<()> {
        match (self, other) {
            (Column::F64(a), Column::F64(b)) => {
                a.make_mut().extend_from_slice(b);
                Ok(())
            }
            (Column::F32(a), Column::F32(b)) => {
                a.make_mut().extend_from_slice(b);
                Ok(())
            }
            (Column::Bool(a, av), Column::Bool(b, bv)) => {
                append_validity(av, a.len(), bv, b.len());
                a.make_mut().extend_from_slice(b);
                Ok(())
            }
            (Column::I64(a, av), Column::I64(b, bv)) => {
                append_validity(av, a.len(), bv, b.len());
                a.make_mut().extend_from_slice(b);
                Ok(())
            }
            (Column::I32(a, av), Column::I32(b, bv)) => {
                append_validity(av, a.len(), bv, b.len());
                a.make_mut().extend_from_slice(b);
                Ok(())
            }
            (Column::Str(a, av), Column::Str(b, bv)) => {
                append_validity(av, a.len(), bv, b.len());
                let mut builder = StrBufferBuilder::with_capacity(a.len() + b.len());
                a.iter().chain(b.iter()).for_each(|s| builder.push(s));
                *a = builder.finish();
                Ok(())
            }
            (Column::Datetime(a), Column::Datetime(b)) => {
                a.make_mut().extend_from_slice(b);
                Ok(())
            }
            (s, o) => Err(VolasError::DType(format!(
                "cannot append a {} column onto a {} column",
                o.dtype(),
                s.dtype()
            ))),
        }
    }

    /// Extend a stale computed column with placeholder missing values, avoiding a
    /// temporary one-row [`Column`] allocation on the live append path.
    pub fn append_missing(&mut self, len: usize) -> Result<()> {
        match self {
            Column::F64(v) => {
                v.make_mut().extend(std::iter::repeat(f64::NAN).take(len));
                Ok(())
            }
            Column::F32(v) => {
                v.make_mut().extend(std::iter::repeat(f32::NAN).take(len));
                Ok(())
            }
            // The refresh path overwrites these placeholder rows on recompute, so a
            // dense `false` keeps the validity simple (no lingering NA to clear).
            Column::Bool(v, _) => {
                v.make_mut().extend(std::iter::repeat(false).take(len));
                Ok(())
            }
            other => Err(VolasError::DType(format!(
                "column type {} has no missing-value placeholder",
                other.dtype()
            ))),
        }
    }

    /// Extend a **plain** (non-computed) column with `len` genuine missing values,
    /// dtype-preserving: float / datetime use their in-band sentinel (`NaN` /
    /// `NaT`), while int / bool / str grow the validity bitmap with `len` invalid
    /// bits. Used when a column is absent from an appended frame; a cached
    /// directive instead uses the cheaper [`append_missing`] placeholder, which
    /// `fulfill` overwrites.
    pub fn append_na(&mut self, len: usize) {
        let old = self.len();
        let na_validity = |val: &Validity| {
            Validity::from_valid_iter(
                old + len,
                (0..old + len).map(|i| i < old && val.is_valid(i)),
            )
        };
        match self {
            Column::F64(v) => v.make_mut().extend(std::iter::repeat(f64::NAN).take(len)),
            Column::F32(v) => v.make_mut().extend(std::iter::repeat(f32::NAN).take(len)),
            Column::Datetime(v) => v.make_mut().extend(std::iter::repeat(i64::MIN).take(len)),
            Column::I64(v, val) => {
                *val = na_validity(val);
                v.make_mut().extend(std::iter::repeat(0).take(len));
            }
            Column::I32(v, val) => {
                *val = na_validity(val);
                v.make_mut().extend(std::iter::repeat(0).take(len));
            }
            Column::Bool(v, val) => {
                *val = na_validity(val);
                v.make_mut().extend(std::iter::repeat(false).take(len));
            }
            Column::Str(v, val) => {
                *val = na_validity(val);
                let mut builder = StrBufferBuilder::with_capacity(v.len() + len);
                v.iter().for_each(|s| builder.push(s));
                (0..len).for_each(|_| builder.push(""));
                *v = builder.finish();
            }
        }
    }

    /// Scatter `values` into `self` at `positions` — **the single assignment
    /// primitive** behind every surface (`df.loc / iloc / at / iat = `, Series
    /// setitem, and boolean-mask assignment; a scalar write passes a length-1
    /// `values`, which broadcasts). It **keeps `self`'s dtype** and updates its
    /// validity: each written position takes the source's presence, every other
    /// position keeps its own. The dtype rules are:
    /// - a float target absorbs any numeric source (a missing source -> in-band `NaN`);
    /// - an int target stays int — a present integral value is stored, a missing /
    ///   `NaN` source marks the cell NA (no float widening, per the NA model), a
    ///   present non-integral value is a lossy [`VolasError::DType`];
    /// - a `Bool` / `Str` / `Datetime` target requires a matching-kind source (else
    ///   a `DType` error); a missing source marks the cell NA (`NaT` for datetime).
    ///
    /// `positions` are assumed in bounds (callers validate the mask / index).
    pub fn scatter(&self, positions: &[usize], values: &Column) -> Result<Column> {
        let len = self.len();
        let m = values.len();
        let pick = |k: usize| if m == 1 { 0 } else { k };
        // A numeric target accepts only a numeric / bool source. A `Str` source
        // would funnel through `to_f64_vec` to `NaN` and a `Datetime` source to raw
        // epoch nanos — both silent corruption — so reject them up front. (The
        // `Bool` / `Str` / `Datetime` targets are already strict via their
        // `as_*_vec` helpers, which error on a mismatched-kind source.)
        if self.dtype().is_numeric() && matches!(values.dtype(), DType::Utf8 | DType::Datetime) {
            return Err(VolasError::DType(format!(
                "cannot assign {} values into a {} column",
                values.dtype(),
                self.dtype()
            )));
        }
        // The validity-carrying targets (int / bool / str) share one rule: keep
        // each row's own presence, then stamp every written position with the
        // source's. (Float and datetime carry missing in-band, so they skip this.)
        let scatter_validity = |val: &Validity| -> Validity {
            let mut flags: Vec<bool> = (0..len).map(|i| val.is_valid(i)).collect();
            for (k, &p) in positions.iter().enumerate() {
                flags[p] = values.is_valid(pick(k));
            }
            Validity::from_valid_iter(len, flags)
        };
        match self {
            Column::F64(v) => {
                let src = values.to_f64_vec(); // validity-aware: missing -> NaN
                let mut nv = v.to_vec();
                for (k, &p) in positions.iter().enumerate() {
                    nv[p] = src[pick(k)];
                }
                Ok(Column::f64(nv))
            }
            Column::F32(v) => {
                let src = values.to_f32_vec();
                let mut nv = v.to_vec();
                for (k, &p) in positions.iter().enumerate() {
                    nv[p] = src[pick(k)];
                }
                Ok(Column::f32(nv))
            }
            // Int targets keep their dtype: `as_i*_vec` yields a 0 placeholder for a
            // missing/`NaN` source (the presence is restored by `scatter_validity`)
            // and errors on a present non-integral value — exactly the Series rule.
            Column::I64(v, val) => {
                let src = values.as_i64_vec()?;
                let mut nv = v.to_vec();
                for (k, &p) in positions.iter().enumerate() {
                    nv[p] = src[pick(k)];
                }
                Ok(Column::i64_with(nv, scatter_validity(val)))
            }
            Column::I32(v, val) => {
                let src = values.as_i32_vec()?;
                let mut nv = v.to_vec();
                for (k, &p) in positions.iter().enumerate() {
                    nv[p] = src[pick(k)];
                }
                Ok(Column::i32_with(nv, scatter_validity(val)))
            }
            Column::Bool(v, val) => {
                let src = values.as_bool_vec()?;
                let mut nv = v.to_vec();
                for (k, &p) in positions.iter().enumerate() {
                    nv[p] = src[pick(k)];
                }
                Ok(Column::bool_with(nv, scatter_validity(val)))
            }
            Column::Str(v, val) => {
                let src = values.as_str_vec()?;
                let mut nv = v.to_vec();
                for (k, &p) in positions.iter().enumerate() {
                    nv[p] = src[pick(k)].clone();
                }
                Ok(Column::str_with(nv, scatter_validity(val)))
            }
            Column::Datetime(v) => {
                let src = values.as_datetime_vec()?; // `NaT` (i64::MIN) is in-band missing
                let mut nv = v.to_vec();
                for (k, &p) in positions.iter().enumerate() {
                    nv[p] = src[pick(k)];
                }
                Ok(Column::datetime(nv))
            }
        }
    }

    // --- dtype-preserving numeric transforms (pandas 3.0) ---------------------
    // Each dispatches the kernel over the column's element type so an int column
    // stays int and computes natively (no f64 round-trip). A non-numeric column
    // is a `DType` error.

    /// Value equality where `NaN == NaN` (pandas `equals` semantics), unlike the
    /// derived `PartialEq` (which uses IEEE `NaN != NaN`).
    pub fn equals(&self, other: &Column) -> bool {
        match (self, other) {
            (Column::F64(a), Column::F64(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
            }
            (Column::F32(a), Column::F32(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
            }
            _ => self == other,
        }
    }
}
