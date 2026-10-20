//! `Column` structural & order-based operations — slice/take, unique,
//! sort/rank, the typed extreme/running-extreme, append, scatter, and equality.

use super::*;
use super::fold::collect_with_spare;

impl Column {
    /// A contiguous `[start, end)` slice (a fresh buffer).
    pub fn slice(&self, start: usize, end: usize) -> Column {
        match self {
            Column::F64(v) => Column::F64(v.slice(start, end)),
            Column::F32(v) => Column::F32(v.slice(start, end)),
            Column::Bool(v, val) => {
                Column::bool_masked(v[start..end].to_vec(), val.slice(start, end))
            }
            Column::I64(v, val) => Column::i64_masked(v[start..end].to_vec(), val.slice(start, end)),
            Column::I32(v, val) => Column::i32_masked(v[start..end].to_vec(), val.slice(start, end)),
            Column::Str(v, val) => Column::Str(v.slice(start, end), val.slice(start, end)),
            Column::Datetime(v) => Column::datetime(v[start..end].to_vec()),
        }
    }

    /// Live retention keeps float columns independently writable, with room for
    /// the next append cycle. Other dtypes already have ordinary owned slices.
    pub(crate) fn slice_with_spare(&self, start: usize, end: usize, spare: usize) -> Column {
        match self {
            Column::F64(v) => Column::F64(v.slice_with_spare(start, end, spare)),
            Column::F32(v) => Column::F32(v.slice_with_spare(start, end, spare)),
            _ => self.slice(start, end),
        }
    }

    /// Live compaction reuses unique bounded float allocations; other dtypes
    /// retain their ordinary independent slicing semantics.
    pub(crate) fn compact_range(&mut self, start: usize, end: usize, spare: usize) {
        match self {
            Column::F64(v) => v.compact_range(start, end, spare),
            Column::F32(v) => v.compact_range(start, end, spare),
            _ => *self = self.slice(start, end),
        }
    }

    /// Gather the given positions into a new column (fancy indexing).
    pub fn take(&self, idx: &[usize]) -> Column {
        self.take_with_spare(idx, 0, 0)
    }

    /// Gather into the final buffers with room for rows the builder will append.
    pub(super) fn take_with_spare(&self, idx: &[usize], spare: usize, bytes: usize) -> Column {
        let gather = |v: &[f64]| collect_with_spare(idx.iter().map(|&i| v[i]), spare);
        match self {
            Column::F64(v) => Column::f64(gather(v)),
            Column::F32(v) => Column::f32(collect_with_spare(idx.iter().map(|&i| v[i]), spare)),
            Column::Bool(v, val) => {
                Column::bool_masked(collect_with_spare(idx.iter().map(|&i| v[i]), spare), val.take(idx))
            }
            Column::I64(v, val) => {
                Column::i64_masked(collect_with_spare(idx.iter().map(|&i| v[i]), spare), val.take(idx))
            }
            Column::I32(v, val) => {
                Column::i32_masked(collect_with_spare(idx.iter().map(|&i| v[i]), spare), val.take(idx))
            }
            // Gather straight into one contiguous `StrBuffer` (no intermediate
            // `Vec<String>`, so no per-cell allocation).
            Column::Str(v, val) => {
                if spare == 0 && bytes == 0 {
                    return Column::Str(idx.iter().map(|&i| v.get(i)).collect(), val.take(idx));
                }
                let size = idx.iter().map(|&i| v.get(i).len()).sum::<usize>();
                let capacity = if spare == 0 { size + bytes } else { (size * 2).max(size + bytes) };
                let capacity = if capacity == 0 || spare == 0 { capacity } else { capacity.max(8) };
                let rows = idx.len() + spare + usize::from(spare != 0);
                let mut builder = crate::strbuf::StrBufferBuilder::with_capacities(rows, capacity);
                for &i in idx { builder.push(v.get(i)); }
                Column::Str(builder.finish(), val.take(idx))
            }
            Column::Datetime(v) => Column::datetime(collect_with_spare(idx.iter().map(|&i| v[i]), spare)),
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
            Column::Bool(v, _) => Column::bool_masked(
                idx.iter().map(|p| p.is_some_and(|i| v[i])).collect(),
                validity(),
            ),
            Column::I64(v, _) => Column::i64_masked(
                idx.iter().map(|p| p.map_or(0, |i| v[i])).collect(),
                validity(),
            ),
            Column::I32(v, _) => Column::i32_masked(
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
        self.rank_with_na(method, ascending, pct, None)
    }

    /// Rank missing cells as one tie group, at the top (`Some(true)`) or bottom
    /// (`Some(false)`) independent of sort direction, or keep them missing.
    pub fn rank_with_na(&self, method: stats::RankMethod, ascending: bool, pct: bool, na_top: Option<bool>) -> Vec<f64> {
        stats::rank_by(
            self.len(),
            |i| na_top.is_some() || self.is_valid(i),
            |a, b| match (self.is_valid(a), self.is_valid(b)) {
                (true, true) => self.cmp_at(a, b),
                (false, false) => Ordering::Equal,
                (false, true) => if na_top == Some(ascending) { Ordering::Less } else { Ordering::Greater },
                (true, false) => if na_top == Some(ascending) { Ordering::Greater } else { Ordering::Less },
            },
            method, ascending, pct,
        )
    }

    /// Append another column, copy-on-write (grows in place when the buffer is
    /// uniquely owned). A column of the same dtype appends as-is (the live fast
    /// path); another dtype converts to this one through the lossless owner
    /// ([`convert_lossless`](Self::convert_lossless), contract #20) — `2.0` into an
    /// int column appends `2`, `2.5` raises — so a frame append agrees with a dict /
    /// row append. A failed conversion leaves `self` unchanged.
    pub fn append(&mut self, other: &Column) -> Result<()> {
        if self.dtype() != other.dtype() {
            let converted = other.convert_lossless(self.dtype()).map_err(|e| self.append_error(other, e))?;
            return self.append(&converted);
        }
        match (self, other) {
            (Column::F64(a), Column::F64(b)) => {
                a.append(b);
                Ok(())
            }
            (Column::F32(a), Column::F32(b)) => {
                a.append(b);
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
                a.extend(b.iter());
                Ok(())
            }
            (Column::Datetime(a), Column::Datetime(b)) => {
                a.make_mut().extend_from_slice(b);
                Ok(())
            }
            _ => unreachable!("dtypes matched above"), // LCOV_EXCL_LINE
        }
    }

    /// The error of appending `other`'s values that this column's dtype cannot hold
    /// losslessly: the conversion's own kind (`DType` for an inexact value or a kind
    /// mismatch, `Overflow` past the range), naming both dtypes.
    #[cold]
    pub(super) fn append_error(&self, other: &Column, e: VolasError) -> VolasError {
        let ctx = format!("cannot append a {} column onto a {} column", other.dtype(), self.dtype());
        match e {
            VolasError::Overflow(why) => VolasError::Overflow(format!("{ctx}: {why}")),
            VolasError::DType(why) => VolasError::DType(format!("{ctx}: {why}")),
            other => other, // LCOV_EXCL_LINE (convert_lossless raises DType / Overflow only)
        }
    }

    /// Overwrite the last cell with `src[src_row]` — value and validity — in place: the
    /// row appended under the last row's label replaces it. Another dtype converts
    /// losslessly first, exactly as for [`append`](Self::append). A float write keeps
    /// the column's NaN fact.
    pub fn set_last_from(&mut self, src: &Column, src_row: usize) -> Result<()> {
        let n = self.len();
        let Some(last) = n.checked_sub(1) else {
            return Err(VolasError::Shape("no last row to replace".into()));
        };
        if self.dtype() != src.dtype() {
            let cell = src.slice(src_row, src_row + 1);
            let converted = cell.convert_lossless(self.dtype()).map_err(|e| self.append_error(src, e))?;
            return self.set_last_from(&converted, 0);
        }
        match (self, src) {
            (Column::F64(a), Column::F64(b)) => a.set(last, b[src_row]),
            (Column::F32(a), Column::F32(b)) => a.set(last, b[src_row]),
            (Column::Datetime(a), Column::Datetime(b)) => a.make_mut()[last] = b[src_row],
            (Column::Bool(a, av), Column::Bool(b, bv)) => {
                a.make_mut()[last] = b[src_row];
                av.set(last, n, bv.is_valid(src_row));
            }
            (Column::I64(a, av), Column::I64(b, bv)) => {
                a.make_mut()[last] = b[src_row];
                av.set(last, n, bv.is_valid(src_row));
            }
            (Column::I32(a, av), Column::I32(b, bv)) => {
                a.make_mut()[last] = b[src_row];
                av.set(last, n, bv.is_valid(src_row));
            }
            (Column::Str(a, av), Column::Str(b, bv)) => {
                a.set_last(b.get(src_row));
                av.set(last, n, bv.is_valid(src_row));
            }
            _ => unreachable!("dtypes matched above"), // LCOV_EXCL_LINE
        }
        Ok(())
    }

    /// Mark the last cell missing in place, dtype-preserving (`NaN` / `NaT`, or an
    /// invalid validity bit) — a replacing row that lacks this column. No-op when empty.
    pub fn set_last_na(&mut self) {
        let n = self.len();
        let Some(last) = n.checked_sub(1) else { return };
        match self {
            Column::F64(a) => a.set(last, f64::NAN),
            Column::F32(a) => a.set(last, f32::NAN),
            Column::Datetime(a) => a.make_mut()[last] = i64::MIN,
            Column::Bool(_, v) | Column::I64(_, v) | Column::I32(_, v) | Column::Str(_, v) => {
                v.set(last, n, false)
            }
        }
    }

    /// Extend the column with `len` missing values, in place and dtype-preserving
    /// — every dtype: float / datetime use their in-band sentinel (`NaN` / `NaT`),
    /// int / bool / str grow the validity with `len` holes (amortized O(len) on a
    /// uniquely-owned buffer). One rule for a plain column absent from an appended
    /// frame and for a cached directive's stale rows (its refresh writes each row's
    /// value and presence over the placeholder, whatever dtype the directive has).
    pub fn append_na(&mut self, len: usize) {
        let old = self.len();
        match self {
            Column::F64(v) => v.append_repeated(f64::NAN, len),
            Column::F32(v) => v.append_repeated(f32::NAN, len),
            Column::Datetime(v) => v.make_mut().extend(std::iter::repeat_n(i64::MIN, len)),
            Column::I64(v, val) => {
                val.extend_missing(old, len);
                v.make_mut().extend(std::iter::repeat_n(0, len));
            }
            Column::I32(v, val) => {
                val.extend_missing(old, len);
                v.make_mut().extend(std::iter::repeat_n(0, len));
            }
            Column::Bool(v, val) => {
                val.extend_missing(old, len);
                v.make_mut().extend(std::iter::repeat_n(false, len));
            }
            Column::Str(v, val) => {
                val.extend_missing(old, len);
                v.extend((0..len).map(|_| ""));
            }
        }
    }

    /// Overwrite rows `[from, from + src.len())` (those inside the column) with
    /// `src`'s values and presence, in place — every dtype, `src` of this column's
    /// dtype (else a [`VolasError::DType`] error). Backs a cached directive's tail
    /// write after a refresh. A `str` tail is rewritten by rebuilding from `from`.
    pub fn write_rows(&mut self, from: usize, src: &Column) -> Result<()> {
        let n = self.len();
        let k = src.len().min(n.saturating_sub(from));
        let presence = |val: &mut Validity, sv: &Validity| {
            for i in 0..k {
                val.set(from + i, n, sv.is_valid(i));
            }
        };
        match (self, src) {
            (Column::F64(a), Column::F64(b)) => a.make_mut()[from..from + k].copy_from_slice(&b[..k]),
            (Column::F32(a), Column::F32(b)) => a.make_mut()[from..from + k].copy_from_slice(&b[..k]),
            (Column::Datetime(a), Column::Datetime(b)) => a.make_mut()[from..from + k].copy_from_slice(&b[..k]),
            (Column::I64(a, av), Column::I64(b, bv)) => {
                a.make_mut()[from..from + k].copy_from_slice(&b[..k]);
                presence(av, bv);
            }
            (Column::I32(a, av), Column::I32(b, bv)) => {
                a.make_mut()[from..from + k].copy_from_slice(&b[..k]);
                presence(av, bv);
            }
            (Column::Bool(a, av), Column::Bool(b, bv)) => {
                a.make_mut()[from..from + k].copy_from_slice(&b[..k]);
                presence(av, bv);
            }
            (Column::Str(a, av), Column::Str(b, bv)) => {
                let mut rebuilt = a.slice(0, from);
                rebuilt.extend(b.iter().take(k).chain(a.iter().skip(from + k)));
                *a = rebuilt;
                presence(av, bv);
            }
            (dst, src) => {
                return Err(VolasError::DType(format!(
                    "cannot write {} rows into a {} column",
                    src.dtype(),
                    dst.dtype()
                )))
            }
        }
        Ok(())
    }

    /// `values` as this column stores them: converted to its dtype — the source rule of
    /// [`scatter`](Self::scatter), every error it raises. A numeric column takes a
    /// numeric / bool source through the lossless owner (contract #20): an int past
    /// float64's exact range, a non-integral float into an int, or a finite value past
    /// float32's range raises — never a rounded or saturated write; a missing source
    /// cell stays missing (`NaN` / NA). A `Str` or `Datetime` source into a numeric
    /// column would read as `NaN` / raw epoch nanos — silent corruption — so it is
    /// refused. A `Bool` / `Str` / `Datetime` column takes only its own dtype.
    pub fn assignable(&self, values: &Column) -> Result<Column> {
        let to = self.dtype();
        if to.is_numeric() && matches!(values.dtype(), DType::Utf8 | DType::Datetime) {
            return Err(VolasError::DType(format!(
                "cannot assign {} values into a {} column",
                values.dtype(),
                to
            )));
        }
        let kind = |want: &str| {
            Err(VolasError::DType(match want {
                "bool" => format!("expected a bool column, got {}", values.dtype()),
                _ => format!("cannot select a {} column as {want}", values.dtype()),
            }))
        };
        match (self, values) {
            (Column::Bool(..), Column::Bool(..))
            | (Column::Str(..), Column::Str(..))
            | (Column::Datetime(_), Column::Datetime(_)) => Ok(values.clone()),
            (Column::Bool(..), _) => kind("bool"),
            (Column::Str(..), _) => kind("str"),
            (Column::Datetime(_), _) => kind("datetime"),
            _ => values.convert_lossless(to),
        }
    }

    /// Scatter `values` into `self` at `positions`, in place — **the single assignment
    /// primitive** behind every surface (`df.loc / iloc / at / iat = `, Series
    /// setitem, and boolean-mask assignment; a scalar write passes a length-1
    /// `values`, which broadcasts). It **keeps `self`'s dtype** and updates its
    /// validity: each written position takes the source's presence, every other
    /// position keeps its own. The dtype rules are [`assignable`](Self::assignable)'s:
    /// - a float target absorbs any numeric source (a missing source -> in-band `NaN`);
    /// - an int target stays int — a present integral value is stored, a missing /
    ///   `NaN` source marks the cell NA (no float widening, per the NA model), a
    ///   present non-integral value is a lossy [`VolasError::DType`];
    /// - a `Bool` / `Str` / `Datetime` target requires a matching-kind source (else
    ///   a `DType` error); a missing source marks the cell NA (`NaT` for datetime).
    ///
    /// All or nothing: the source is converted before the first write, so an error
    /// leaves the column unchanged. The write costs O(positions) on a uniquely-owned
    /// buffer (a shared one is copied once, copy-on-write); a float column keeps its
    /// NaN fact. A `str` column rebuilds its contiguous bytes.
    ///
    /// `positions` are assumed in bounds (callers validate the mask / index).
    pub fn scatter(&mut self, positions: &[usize], values: &Column) -> Result<()> {
        let src = self.assignable(values)?;
        self.scatter_assignable(positions, &src);
        Ok(())
    }

    /// [`scatter`](Self::scatter) on a copy, returned (tests compare whole columns).
    #[cfg(test)]
    pub(crate) fn scattered(&self, positions: &[usize], values: &Column) -> Result<Column> {
        let mut out = self.clone();
        out.scatter(positions, values)?;
        Ok(out)
    }

    /// [`scatter`](Self::scatter) of a source already of this column's dtype (from
    /// [`assignable`](Self::assignable)): cannot fail. A later repeat of a position
    /// wins, as a sequence of single writes would.
    pub(crate) fn scatter_assignable(&mut self, positions: &[usize], src: &Column) {
        let len = self.len();
        let pick = |k: usize| if src.len() == 1 { 0 } else { k };
        // The validity-carrying targets (int / bool / str) stamp every written position
        // with the source's presence; float and datetime carry missing in-band.
        let presence = |val: &mut Validity, sv: &Validity| {
            for (k, &p) in positions.iter().enumerate() {
                val.set(p, len, sv.is_valid(pick(k)));
            }
        };
        match (self, src) {
            (Column::F64(v), Column::F64(s)) => v.scatter(positions, |k| s[pick(k)]),
            (Column::F32(v), Column::F32(s)) => v.scatter(positions, |k| s[pick(k)]),
            (Column::Datetime(v), Column::Datetime(s)) => {
                let w = v.make_mut();
                positions.iter().enumerate().for_each(|(k, &p)| w[p] = s[pick(k)]);
            }
            (Column::I64(v, val), Column::I64(s, sv)) => {
                let w = v.make_mut();
                positions.iter().enumerate().for_each(|(k, &p)| w[p] = s[pick(k)]);
                presence(val, sv);
            }
            (Column::I32(v, val), Column::I32(s, sv)) => {
                let w = v.make_mut();
                positions.iter().enumerate().for_each(|(k, &p)| w[p] = s[pick(k)]);
                presence(val, sv);
            }
            (Column::Bool(v, val), Column::Bool(s, sv)) => {
                let w = v.make_mut();
                positions.iter().enumerate().for_each(|(k, &p)| w[p] = s[pick(k)]);
                presence(val, sv);
            }
            (Column::Str(v, val), Column::Str(s, sv)) => {
                let mut cells: Vec<&str> = v.iter().collect();
                positions.iter().enumerate().for_each(|(k, &p)| cells[p] = s.get(pick(k)));
                let rebuilt: StrBuffer = cells.into_iter().collect();
                *v = rebuilt;
                presence(val, sv);
            }
            _ => unreachable!("an assignable source has the column's dtype"), // LCOV_EXCL_LINE
        }
    }

    /// Value equality (pandas `equals`): the same dtype and length, missing in the
    /// same positions, and equal present values. A missing cell is compared by its
    /// presence only — never by the payload it happens to hold (two all-NA int
    /// columns are equal whatever their placeholders) — and `NaN == NaN`, unlike the
    /// derived structural `PartialEq`.
    pub fn equals(&self, other: &Column) -> bool {
        let n = self.len();
        if other.len() != n {
            return false;
        }
        let cells = |same: &dyn Fn(usize) -> bool| {
            (0..n).all(|i| match (self.is_valid(i), other.is_valid(i)) {
                (true, true) => same(i),
                (a, b) => a == b,
            })
        };
        match (self, other) {
            (Column::F64(a), Column::F64(b)) => cells(&|i| a[i] == b[i]),
            (Column::F32(a), Column::F32(b)) => cells(&|i| a[i] == b[i]),
            (Column::I64(a, _), Column::I64(b, _)) => cells(&|i| a[i] == b[i]),
            (Column::I32(a, _), Column::I32(b, _)) => cells(&|i| a[i] == b[i]),
            (Column::Bool(a, _), Column::Bool(b, _)) => cells(&|i| a[i] == b[i]),
            (Column::Str(a, _), Column::Str(b, _)) => cells(&|i| a.get(i) == b.get(i)),
            (Column::Datetime(a), Column::Datetime(b)) => cells(&|i| a[i] == b[i]),
            _ => false,
        }
    }
}
