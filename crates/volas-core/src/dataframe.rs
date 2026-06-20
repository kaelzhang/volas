//! DataFrame: ordered, named columns sharing a single row index.

use std::collections::HashMap;
use std::sync::Arc;

use crate::column::{Column, CombineOp};
use crate::fxhash::FxHashMap;
use crate::error::{Result, VolasError};
use crate::index::{Index, IndexKind};
use crate::series::Series;

/// Metadata for a materialized (cached) directive column: the directive that
/// produced it, its lookback, and how many leading rows currently hold valid
/// values. After an `append`, the new rows are stale (NaN) and `valid_rows` lags
/// `height` until `fulfill` recomputes the tail.
#[derive(Clone, Debug)]
pub struct ComputedMeta {
    /// The (canonical) directive string. `Arc<str>` so the per-bar refresh snapshot
    /// (`stale_computed_columns`) clones it with a refcount bump, not a heap copy.
    pub directive: Arc<str>,
    /// The directive's lookback (warm-up rows).
    pub lookback: usize,
    /// Rows `[0, valid_rows)` currently hold valid values.
    pub valid_rows: usize,
    /// Carried recursive state for an O(new-rows) append resume: a small,
    /// fixed-size per-indicator vector capturing the internal recursive state as
    /// of the last valid row (`valid_rows - 1`), so an `append`/`fulfill` can
    /// continue the recursion over only the new rows, bit-identical to a fresh
    /// full recompute. `None` when the directive has no resume implementation (it
    /// then falls back to the correct full recompute) or the state is unknown
    /// (e.g. after a slice that did not reach the parent's `valid_rows`).
    pub state: Option<Vec<f64>>,
    /// The original-frame row that THIS (possibly sliced) frame's row 0 maps to.
    /// `0` for a freshly-computed column; a contiguous slice from `start` bumps it
    /// by `start`. It lets an absolute-position indicator (the index family —
    /// maxindex/minindex/minmaxindex) keep emitting ABSOLUTE positions after a
    /// head-dropping slice: a sub-frame position `p` is original row `p + origin`,
    /// matching the verbatim-carried (original-absolute) head. Recursive *value*
    /// indicators ignore it (their state is offset-free).
    pub origin: usize,
}

/// A 2-D, column-oriented, time-indexed table. All columns share one index and
/// have equal length (`height`).
#[derive(Clone, Debug)]
pub struct DataFrame {
    // Schema (names + lookup) is `Arc`-shared so a frame clone / same-schema
    // derivation (slice / take / mask / astype) is an O(1) refcount bump, not a
    // rebuild of the name strings + hash map (copy-on-write on mutation).
    names: Arc<Vec<String>>,
    columns: Vec<Column>,
    // These internal name maps use FxHash (not the default SipHash): their keys are
    // internal column / directive names, and they are hit on the live-append hot path.
    name_to_idx: Arc<FxHashMap<String, usize>>,
    index: Arc<Index>,
    height: usize,
    /// Materialized directive columns (name -> meta). Tracked so `fulfill` can
    /// incrementally recompute their tail after an append. Carried through
    /// `clone` / `append`; dropped by shape-changing ops (slice/select/…), where
    /// the columns become plain data.
    computed: FxHashMap<String, ComputedMeta>,
}

impl DataFrame {
    /// Construct a frame from parallel `names` / `columns`, validating shape.
    pub fn new(names: Vec<String>, columns: Vec<Column>, index: Option<Index>) -> Result<Self> {
        if names.len() != columns.len() {
            return Err(VolasError::Shape(format!(
                "{} names but {} columns",
                names.len(),
                columns.len()
            )));
        }
        let height = columns.first().map(|c| c.len()).unwrap_or(0);
        for (n, c) in names.iter().zip(&columns) {
            if c.len() != height {
                return Err(VolasError::Shape(format!(
                    "column \"{}\" has length {} but frame height is {}",
                    n,
                    c.len(),
                    height
                )));
            }
        }
        let index = match index {
            Some(ix) => {
                if ix.len() != height {
                    return Err(VolasError::Shape(format!(
                        "index length {} != frame height {}",
                        ix.len(),
                        height
                    )));
                }
                ix
            }
            None => Index::range(height),
        };
        let mut name_to_idx = FxHashMap::with_capacity_and_hasher(names.len(), Default::default());
        for (i, n) in names.iter().enumerate() {
            name_to_idx.insert(n.clone(), i);
        }
        Ok(DataFrame {
            names: Arc::new(names),
            columns,
            name_to_idx: Arc::new(name_to_idx),
            index: Arc::new(index),
            height,
            computed: FxHashMap::default(),
        })
    }

    /// Number of rows.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of columns.
    pub fn width(&self) -> usize {
        self.columns.len()
    }

    /// Column names in order.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// The `Arc`-shared name vector. Pointer-stable across row-only mutations
    /// (`append` / forming-row folds), so a caller can validate "schema unchanged"
    /// with an O(1) [`Arc::ptr_eq`] instead of an element-wise name comparison.
    pub fn names_arc(&self) -> &Arc<Vec<String>> {
        &self.names
    }

    /// The shared row index.
    pub fn index(&self) -> &Arc<Index> {
        &self.index
    }

    /// Columns in order.
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Position of a column by name.
    pub fn column_pos(&self, name: &str) -> Option<usize> {
        self.name_to_idx.get(name).copied()
    }

    /// Whether a column exists.
    pub fn has_column(&self, name: &str) -> bool {
        self.name_to_idx.contains_key(name)
    }

    /// Whether `name` is a cached directive (computed) column rather than plain data.
    /// Plain columns are supplied per bar; computed ones are derived and refreshed by
    /// `fulfill`. A read-only metadata lookup — off every compute / append hot path.
    pub fn is_computed(&self, name: &str) -> bool {
        self.computed.contains_key(name)
    }

    /// Build a frame that **shares this frame's schema** (names + lookup, both
    /// `Arc`-cloned) over freshly derived `columns` / `index` — for the same-shape
    /// derivations (slice / take / mask / astype), with no name-string or hash-map
    /// rebuild. Computed-column status is dropped; the caller re-attaches it where
    /// the derivation preserves it (a contiguous slice).
    fn same_schema(&self, columns: Vec<Column>, index: Index) -> DataFrame {
        let height = columns.first().map_or(0, |c| c.len());
        DataFrame {
            names: Arc::clone(&self.names),
            name_to_idx: Arc::clone(&self.name_to_idx),
            columns,
            index: Arc::new(index),
            height,
            computed: FxHashMap::default(),
        }
    }

    /// Gather rows by position into a new frame.
    pub fn take(&self, positions: &[usize]) -> DataFrame {
        let columns: Vec<Column> = self.columns.iter().map(|c| c.take(positions)).collect();
        self.same_schema(columns, self.index.take(positions))
    }

    /// Borrow a column by name.
    pub fn column(&self, name: &str) -> Result<&Column> {
        self.column_pos(name)
            .map(|i| &self.columns[i])
            .ok_or_else(|| VolasError::ColumnNotFound(name.to_string()))
    }

    /// Extract a column as a [`Series`] sharing this frame's index.
    pub fn series(&self, name: &str) -> Result<Series> {
        let col = self.column(name)?.clone();
        Ok(Series::new(
            Some(name.to_string()),
            col,
            Arc::clone(&self.index),
        ))
    }

    /// Add a new column or replace an existing one (must match `height`, unless
    /// the frame currently has no columns).
    pub fn set_column(&mut self, name: &str, col: Column) -> Result<()> {
        if self.columns.is_empty() {
            self.height = col.len();
            if self.index.len() != self.height {
                self.index = Arc::new(Index::range(self.height));
            }
        } else if col.len() != self.height {
            return Err(VolasError::Shape(format!(
                "new column \"{}\" has length {} but frame height is {}",
                name,
                col.len(),
                self.height
            )));
        }
        match self.column_pos(name) {
            Some(i) => self.columns[i] = col,
            None => {
                Arc::make_mut(&mut self.name_to_idx).insert(name.to_string(), self.columns.len());
                Arc::make_mut(&mut self.names).push(name.to_string());
                self.columns.push(col);
            }
        }
        Ok(())
    }

    /// Move a column out of the frame and use it as the row index (pandas
    /// `set_index`). The column is removed; its values become the index
    /// (datetime / int64 — see [`Index::from_column`]).
    pub fn set_index(&self, name: &str) -> Result<DataFrame> {
        let pos = self
            .column_pos(name)
            .ok_or_else(|| VolasError::ColumnNotFound(name.to_string()))?;
        // Record the source column's name on the index (pandas keeps it, so
        // `reset_index` can restore the original column label).
        let index = Index::from_column(&self.columns[pos])?.with_name(Some(name.to_string()));
        let mut names = (*self.names).clone();
        let mut columns = self.columns.clone();
        names.remove(pos);
        columns.remove(pos);
        DataFrame::new(names, columns, Some(index))
    }

    /// Change the DatetimeIndex's **display / matching** timezone without moving
    /// any instant (pandas `tz_convert`): stored UTC ns are unchanged; only how
    /// they render and how bare-string `.loc` matches changes. Returns a new frame
    /// (columns shared). Errors if the index is not a DatetimeIndex.
    pub fn tz_convert(&self, tz: crate::tz::Tz) -> Result<DataFrame> {
        match self.index.kind() {
            IndexKind::Datetime(_, cur) => {
                // A naive axis is an unanchored wall-clock — there is no source
                // zone to convert FROM, so converting it would silently relabel
                // wrong instants. Anchor with tz_localize first (pandas parity).
                if !cur.is_aware() {
                    return Err(VolasError::DType(
                        "cannot tz_convert a tz-naive DatetimeIndex; use tz_localize to anchor it first"
                            .into(),
                    ));
                }
                let mut df = self.clone();
                df.index = Arc::new((*self.index).clone().with_tz(tz));
                Ok(df)
            }
            _ => Err(VolasError::DType(
                "tz_convert requires a DatetimeIndex".into(),
            )),
        }
    }

    /// Tag the DatetimeIndex's zone directly, without the naive-axis guard of
    /// [`Self::tz_convert`]. For importers (`from_pandas`) whose instants are
    /// ALREADY true UTC and arrive carrying their zone — not a user-facing API.
    pub fn set_index_tz(&self, tz: crate::tz::Tz) -> Result<DataFrame> {
        match self.index.kind() {
            IndexKind::Datetime(_, _) => {
                let mut df = self.clone();
                df.index = Arc::new((*self.index).clone().with_tz(tz));
                Ok(df)
            }
            _ => Err(VolasError::DType(
                "set_index_tz requires a DatetimeIndex".into(),
            )),
        }
    }

    /// Reinterpret the index's **wall-clock** as `tz` (pandas `tz_localize`): each
    /// instant is recomputed so the displayed wall-clock is unchanged but now
    /// correct for `tz`. Use this when data was ingested without a tz and you need
    /// to attach the right one. Returns a new frame. Errors if the index is not a
    /// DatetimeIndex or a wall-clock does not exist in `tz` (a DST spring-forward
    /// gap).
    pub fn tz_localize(&self, tz: crate::tz::Tz) -> Result<DataFrame> {
        let (values, cur) = match self.index.kind() {
            IndexKind::Datetime(v, cur) => (v.clone(), *cur),
            _ => {
                return Err(VolasError::DType(
                    "tz_localize requires a DatetimeIndex".into(),
                ))
            }
        };
        // Localize anchors an UNanchored wall-clock; an already-aware axis must
        // use tz_convert (re-localizing would silently reinterpret instants).
        if cur.is_aware() {
            return Err(VolasError::DType(format!(
                "index is already tz-aware ({}); use tz_convert",
                cur.name()
            )));
        }
        let mut shifted = Vec::with_capacity(values.len());
        for ns in values {
            let (y, mo, d, h, mi, s) = cur.civil_parts(ns);
            let new = tz
                .wall_to_utc_ns(y as i32, mo as u32, d as u32, h as u32, mi as u32, s as u32)
                .ok_or_else(|| {
                    VolasError::Value(format!(
                        "wall-clock {y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02} does not exist in {} (or is DST-ambiguous)",
                        tz.name()
                    ))
                })?;
            shifted.push(new);
        }
        let mut df = self.clone();
        // tz_localize moves the instants but keeps the index identity (and name).
        df.index = Arc::new(Index::datetime(shifted, tz).with_name(self.index.name().map(String::from)));
        Ok(df)
    }

    /// Select a subset of columns into a new frame sharing this index.
    pub fn select(&self, names: &[String]) -> Result<DataFrame> {
        let mut columns = Vec::with_capacity(names.len());
        for n in names {
            columns.push(self.column(n)?.clone());
        }
        let mut name_to_idx = FxHashMap::with_capacity_and_hasher(names.len(), Default::default());
        for (i, n) in names.iter().enumerate() {
            name_to_idx.insert(n.clone(), i);
        }
        Ok(DataFrame {
            names: Arc::new(names.to_vec()),
            columns,
            name_to_idx: Arc::new(name_to_idx),
            index: Arc::clone(&self.index),
            height: self.height,
            computed: FxHashMap::default(),
        })
    }

    /// A `[start, end)` row slice.
    ///
    /// Deliberately a **value copy** (each column's window is copied), not a
    /// zero-copy view into the parent buffer: a slice is an independent frame, so
    /// slicing the recent tail of a long history does not pin the whole history
    /// alive — the right default for a live system. (A view would be ~1.5x faster
    /// here but would retain the parent's full buffer; we keep the safer copy.)
    pub fn slice(&self, start: usize, end: usize) -> DataFrame {
        let start = start.min(self.height);
        let end = end.max(start).min(self.height);
        let len = end - start;
        let columns: Vec<Column> = self.columns.iter().map(|c| c.slice(start, end)).collect();
        let mut df = self.same_schema(columns, self.index.slice(start, end));
        // SP-9: carry cached-directive columns *as continuable computed columns*
        // through a contiguous slice. The cached values are already correct (they
        // were computed with full history) and are carried verbatim; we re-tag the
        // `ComputedMeta` cursor so a later `append` refreshes the tail incrementally
        // — re-deriving it from the retained raw columns over a `lookback` window,
        // exactly as a non-sliced frame would (the engine re-warms from raw data,
        // never from cached output, so composite recursive indicators continue
        // correctly too). This is only sound when the slice keeps at least
        // `lookback` warm-up rows; a shorter slice would re-warm from its own start
        // (a seed that is *not* `lookback` rows back) and silently diverge, so there
        // we drop the computed status and the column stays plain data (honest:
        // values correct, but not continuable). Non-contiguous derivations
        // (`take` / `filter_mask`) go through `DataFrame::new` and already drop it.
        for (name, meta) in &self.computed {
            if len >= meta.lookback {
                let valid = meta.valid_rows.saturating_sub(start).min(len);
                // Carry the recursive state only when this slice's END reaches the
                // parent's `valid_rows`: the captured state is the internal state as
                // of the parent row `valid_rows - 1`, which is THIS sub-frame's last
                // valid row exactly when `start + len >= valid_rows` (so `valid` ==
                // the parent's last-valid offset). A shorter slice (end before
                // `valid_rows`) would leave the state attached to a row the sub-frame
                // no longer ends on, so we drop it (the column stays correct via the
                // full-recompute fallback, just not O(new-rows) continuable).
                let carried = if end >= meta.valid_rows {
                    meta.state.clone()
                } else {
                    None
                };
                df.computed.insert(
                    name.clone(),
                    ComputedMeta {
                        directive: meta.directive.clone(),
                        lookback: meta.lookback,
                        valid_rows: valid,
                        state: carried,
                        // This sub-frame's row 0 is the parent's row `start`, so its
                        // origin shifts by `start` (an absolute-index resume adds it
                        // back to stay original-absolute, matching the carried head).
                        origin: meta.origin + start,
                    },
                );
            }
        }
        df
    }

    /// A `[start, end)` row slice that does **not** carry the cached-directive
    /// (computed) metadata — for a READ-ONLY derivation that is never appended to
    /// (a refresh probe, a row-select feeding `DataFrame::new`). [`slice`] clones a
    /// `ComputedMeta` per cached column to keep the SP-9 incremental resume across
    /// the slice; a read-only consumer reads only the raw columns and discards the
    /// frame, so that per-column clone (`O(K)` per slice, `O(K²)` per fulfill over a
    /// K-indicator windowed frame) is pure waste. The result's computed columns
    /// become plain data — correct values, but it MUST NOT be appended to (use
    /// [`slice`] for anything that continues live, e.g. window compaction).
    pub fn slice_data(&self, start: usize, end: usize) -> DataFrame {
        let start = start.min(self.height);
        let end = end.max(start).min(self.height);
        let columns: Vec<Column> = self.columns.iter().map(|c| c.slice(start, end)).collect();
        self.same_schema(columns, self.index.slice(start, end))
    }

    /// Filter rows by a boolean mask.
    pub fn filter_mask(&self, mask: &[bool]) -> Result<DataFrame> {
        if mask.len() != self.height {
            return Err(VolasError::Shape(format!(
                "boolean mask length {} != frame height {}",
                mask.len(),
                self.height
            )));
        }
        let idx: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect();
        let columns: Vec<Column> = self.columns.iter().map(|c| c.take(&idx)).collect();
        Ok(self.same_schema(columns, self.index.take(&idx)))
    }

    /// Append the rows of `other` (matched by column name) in place. Columns of
    /// `self` absent from `other` are NaN-padded (so a frame with materialized
    /// directive columns can take raw bars; `fulfill` then refreshes them).
    /// Computed-column metadata is retained, leaving the new rows stale.
    pub fn append(&mut self, other: &DataFrame) -> Result<()> {
        let oh = other.height;
        // Identical schema (same names, same order) — the live-streaming / tf-fold
        // case: append positionally, skipping the per-column name lookup entirely.
        // Matching name vectors guarantee `other`'s column `pos` is this frame's
        // column `pos`.
        if self.names == other.names {
            for (dst, src) in self.columns.iter_mut().zip(&other.columns) {
                dst.append(src)?;
            }
            Arc::make_mut(&mut self.index).extend(&other.index)?;
            self.height += oh;
            return Ok(());
        }
        // Iterate by position to avoid cloning every column name and then
        // re-hashing it back into this same frame on the live append path.
        for pos in 0..self.names.len() {
            let n = &self.names[pos];
            if let Some(other_pos) = other.column_pos(n) {
                self.columns[pos].append(&other.columns[other_pos])?;
            } else {
                // column `n` is missing from `other` — pad the new rows.
                if self.computed.contains_key(n) {
                    // A cached directive (F64 indicator / Bool mask): a cheap stale
                    // placeholder (NaN / `false`); `fulfill` recomputes and overwrites
                    // the appended tail, so a dense placeholder keeps validity simple.
                    self.columns[pos].append_missing(oh)?;
                } else {
                    // A plain column keeps its data semantics: pad with dtype-preserving
                    // NA (int / bool / str grow the validity bitmap; datetime -> NaT;
                    // float -> NaN), never upcasting the dtype or erroring.
                    self.columns[pos].append_na(oh);
                }
            }
        }
        Arc::make_mut(&mut self.index).extend(&other.index)?;
        self.height += oh;
        Ok(())
    }

    /// Assign `values` into column position `col` at the given row `positions`
    /// (copy-on-write via [`Arc::make_mut`]). `values` is broadcast when it has
    /// length 1, otherwise its length must equal `positions.len()`. This backs
    /// `df.loc[...] = `, `df.iloc[...] = `, `df.at[...] = ` and `df.iat[...] = `.
    ///
    /// Dtype handling is delegated to [`Column::scatter`], the single assignment
    /// primitive shared with the Series and boolean-mask surfaces: it **keeps the
    /// target column's dtype** and updates its validity (a write into an existing NA
    /// cell makes it present; a missing / `NaN` source marks the cell NA without
    /// widening an int column to float; a present non-integral value into an int
    /// column is a lossy error).
    ///
    /// A manual write into a cached directive column **drops its computed status**
    /// (it becomes plain data) so a later `fulfill` can never silently clobber the
    /// override.
    pub fn assign_positions(
        &mut self,
        col: usize,
        positions: &[usize],
        values: &Column,
    ) -> Result<()> {
        if col >= self.columns.len() {
            return Err(VolasError::Shape(format!(
                "column position {col} is out of range (width {})",
                self.columns.len()
            )));
        }
        let n = positions.len();
        if values.len() != 1 && values.len() != n {
            return Err(VolasError::Shape(format!(
                "cannot assign {} values to {n} selected rows",
                values.len()
            )));
        }
        for &p in positions {
            if p >= self.height {
                return Err(VolasError::Shape(format!(
                    "row position {p} is out of range (height {})",
                    self.height
                )));
            }
        }
        self.columns[col] = self.columns[col].scatter(positions, values)?;
        self.invalidate_computed_on_write_at(col);
        Ok(())
    }

    /// Fold `src[src_row]` into the forming aggregate at `row`, in place, per the
    /// `(dst_col, src_col, op)` plan — the allocation-free live tf-fold. Unlike
    /// `assign_positions` it neither re-reduces the period nor clones a column
    /// buffer: each cell is combined through [`Column::combine_at`]. A single
    /// conservative cache invalidation follows (the forming row changed, so every
    /// cached directive recomputes on the next read), exactly like a positional
    /// write. The caller guarantees every `op`'s column dtype is fold-eligible
    /// (numeric / datetime); a `Bool` / `Str` column makes `combine_at` error.
    pub fn fold_forming_row(
        &mut self,
        row: usize,
        src: &DataFrame,
        src_row: usize,
        ops: &[(usize, usize, CombineOp)],
    ) -> Result<()> {
        for &(dst_col, src_col, op) in ops {
            self.columns[dst_col].combine_at(row, op, &src.columns[src_col], src_row)?;
        }
        if let Some(&(dst_col, _, _)) = ops.first() {
            self.invalidate_computed_on_write_at(dst_col);
        }
        Ok(())
    }

    /// Rename columns (pandas `rename(columns=...)`), returning a new frame.
    /// Names not in `mapping` are kept; columns and index are shared (cheap).
    pub fn rename(&self, mapping: &HashMap<String, String>) -> Result<DataFrame> {
        let names: Vec<String> = self
            .names
            .iter()
            .map(|n| mapping.get(n).cloned().unwrap_or_else(|| n.clone()))
            .collect();
        DataFrame::new(names, self.columns.clone(), Some((*self.index).clone()))
    }

    /// Cast the named columns to new dtypes (pandas `astype`), returning a new
    /// frame. Untouched columns are shared (cheap).
    pub fn astype(&self, mapping: &HashMap<String, crate::dtype::DType>) -> Result<DataFrame> {
        let mut columns = self.columns.clone();
        for (name, dtype) in mapping {
            let pos = self
                .column_pos(name)
                .ok_or_else(|| VolasError::ColumnNotFound(name.clone()))?;
            columns[pos] = self.columns[pos].cast(*dtype)?;
        }
        Ok(self.same_schema(columns, (*self.index).clone()))
    }

    /// Value equality (pandas `DataFrame.equals`): same column names + order,
    /// same index *labels*, and value-equal columns (`NaN == NaN`). The index
    /// *name* is metadata and is ignored, matching pandas (`.equals` ignores it).
    pub fn equals(&self, other: &DataFrame) -> bool {
        self.height == other.height
            && self.names == other.names
            && self.index.label_eq(&other.index)
            && self
                .columns
                .iter()
                .zip(&other.columns)
                .all(|(a, b)| a.equals(b))
    }

    /// Flatten to a row-major (C-order) 2-D `f64` buffer for NumPy export,
    /// returning `(data, height, width)`. Each column is materialized through the
    /// validity-aware `to_f64_vec`, so a missing cell (int/bool NA, datetime NaT,
    /// str) exports as `NaN` — not the raw placeholder — matching the 1-D
    /// `Series` export and `pandas` `Int64.to_numpy()`.
    pub fn to_row_major_f64(&self) -> (Vec<f64>, usize, usize) {
        let h = self.height;
        let w = self.columns.len();
        let mut out = vec![0.0f64; h * w];
        for (j, c) in self.columns.iter().enumerate() {
            let col = c.to_f64_vec();
            for i in 0..h {
                out[i * w + j] = col[i];
            }
        }
        (out, h, w)
    }

    /// Flatten to a row-major (C-order) 2-D `i64` buffer for an **exact** integer
    /// (or `datetime64[ns]`) NumPy export, returning `(data, height, width)`. A
    /// datetime column contributes its raw epoch-ns — so sub-2⁵³ ns and `NaT`
    /// (which stays `i64::MIN`, the datetime64 sentinel) survive, unlike the
    /// `to_row_major_f64` channel — and an `i64` column its exact value (no
    /// float round-trip past 2⁵³). A float column truncates toward zero. A `str`
    /// column has no integer meaning; the export boundary rejects it before
    /// calling this, so it contributes a `0` placeholder it never reaches.
    pub fn to_row_major_i64(&self) -> (Vec<i64>, usize, usize) {
        let h = self.height;
        let w = self.columns.len();
        let mut out = vec![0i64; h * w];
        for (j, c) in self.columns.iter().enumerate() {
            for i in 0..h {
                out[i * w + j] = match c {
                    Column::Datetime(v) => v[i],
                    Column::I64(v, _) => v[i],
                    Column::I32(v, _) => v[i] as i64,
                    Column::Bool(v, _) => v[i] as i64,
                    Column::F64(v) => v[i] as i64,
                    Column::F32(v) => v[i] as i64,
                    Column::Str(..) => 0,
                };
            }
        }
        (out, h, w)
    }
}

mod computed;

#[cfg(test)]
mod tests;
