//! DataFrame: ordered, named columns sharing a single row index.

use std::collections::HashMap;
use std::sync::Arc;

use crate::column::Column;
use crate::error::{Result, VolasError};
use crate::index::{Index, IndexKind};
use crate::series::Series;

/// Metadata for a materialized (cached) directive column: the directive that
/// produced it, its lookback, and how many leading rows currently hold valid
/// values. After an `append`, the new rows are stale (NaN) and `valid_rows` lags
/// `height` until `fulfill` recomputes the tail.
#[derive(Clone, Debug)]
pub struct ComputedMeta {
    /// The (canonical) directive string.
    pub directive: String,
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
    name_to_idx: Arc<HashMap<String, usize>>,
    index: Arc<Index>,
    height: usize,
    /// Column-name aliases (`alias -> source name`), resolved on lookup. Shared
    /// via `Arc` (cheap clone) and carried through derived frames.
    aliases: Arc<HashMap<String, String>>,
    /// Materialized directive columns (name -> meta). Tracked so `fulfill` can
    /// incrementally recompute their tail after an append. Carried through
    /// `clone` / `append`; dropped by shape-changing ops (slice/select/…), where
    /// the columns become plain data.
    computed: HashMap<String, ComputedMeta>,
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
        let mut name_to_idx = HashMap::with_capacity(names.len());
        for (i, n) in names.iter().enumerate() {
            name_to_idx.insert(n.clone(), i);
        }
        Ok(DataFrame {
            names: Arc::new(names),
            columns,
            name_to_idx: Arc::new(name_to_idx),
            index: Arc::new(index),
            height,
            aliases: Arc::new(HashMap::new()),
            computed: HashMap::new(),
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

    /// The shared row index.
    pub fn index(&self) -> &Arc<Index> {
        &self.index
    }

    /// Columns in order.
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Resolve a name through the alias map (`alias -> source`, else itself).
    fn resolve<'a>(&'a self, name: &'a str) -> &'a str {
        self.aliases.get(name).map(String::as_str).unwrap_or(name)
    }

    /// Position of a column by name (alias-aware).
    pub fn column_pos(&self, name: &str) -> Option<usize> {
        self.name_to_idx.get(self.resolve(name)).copied()
    }

    /// Whether a column exists (alias-aware).
    pub fn has_column(&self, name: &str) -> bool {
        self.name_to_idx.contains_key(self.resolve(name))
    }

    /// Define a column alias (`as_name -> src_name`), returning a new frame.
    /// Errors if `as_name` is already a real column, or `src_name` does not exist.
    pub fn with_alias(&self, as_name: &str, src_name: &str) -> Result<DataFrame> {
        if self.name_to_idx.contains_key(as_name) {
            return Err(VolasError::Value(format!(
                "column \"{as_name}\" already exists"
            )));
        }
        if self.column_pos(src_name).is_none() {
            return Err(VolasError::Value(format!(
                "column \"{src_name}\" not exists"
            )));
        }
        let mut aliases = (*self.aliases).clone();
        aliases.insert(as_name.to_string(), src_name.to_string());
        let mut df = self.clone();
        df.aliases = Arc::new(aliases);
        Ok(df)
    }

    /// Build a frame that **shares this frame's schema** (names + lookup +
    /// aliases, all `Arc`-cloned) over freshly derived `columns` / `index` — for
    /// the same-shape derivations (slice / take / mask / astype), with no
    /// name-string or hash-map rebuild. Computed-column status is dropped; the
    /// caller re-attaches it where the derivation preserves it (a contiguous slice).
    fn same_schema(&self, columns: Vec<Column>, index: Index) -> DataFrame {
        let height = columns.first().map_or(0, |c| c.len());
        DataFrame {
            names: Arc::clone(&self.names),
            name_to_idx: Arc::clone(&self.name_to_idx),
            columns,
            index: Arc::new(index),
            height,
            aliases: Arc::clone(&self.aliases),
            computed: HashMap::new(),
        }
    }

    /// Gather rows by position into a new frame (carries aliases).
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
        let mut df = DataFrame::new(names, columns, Some(index))?;
        df.aliases = Arc::clone(&self.aliases);
        Ok(df)
    }

    /// Change the DatetimeIndex's **display / matching** timezone without moving
    /// any instant (pandas `tz_convert`): stored UTC ns are unchanged; only how
    /// they render and how bare-string `.loc` matches changes. Returns a new frame
    /// (columns shared). Errors if the index is not a DatetimeIndex.
    pub fn tz_convert(&self, tz: crate::tz::Tz) -> Result<DataFrame> {
        match self.index.kind() {
            IndexKind::Datetime(_, _) => {
                let mut df = self.clone();
                df.index = Arc::new((*self.index).clone().with_tz(tz));
                Ok(df)
            }
            _ => Err(VolasError::DType(
                "tz_convert requires a DatetimeIndex".into(),
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
        let mut shifted = Vec::with_capacity(values.len());
        for ns in values {
            let (y, mo, d, h, mi, s) = cur.civil_parts(ns);
            let new = tz
                .wall_to_utc_ns(y as i32, mo as u32, d as u32, h as u32, mi as u32, s as u32)
                .ok_or_else(|| {
                    VolasError::Value(format!(
                        "wall-clock {y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02} does not exist in {}",
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
        let mut name_to_idx = HashMap::with_capacity(names.len());
        for (i, n) in names.iter().enumerate() {
            name_to_idx.insert(n.clone(), i);
        }
        Ok(DataFrame {
            names: Arc::new(names.to_vec()),
            columns,
            name_to_idx: Arc::new(name_to_idx),
            index: Arc::clone(&self.index),
            height: self.height,
            aliases: Arc::clone(&self.aliases),
            computed: HashMap::new(),
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

    /// Record that column `name` is a materialized directive result (valid for
    /// all current rows).
    pub fn set_computed(&mut self, name: &str, directive: String, lookback: usize) {
        self.computed.insert(
            name.to_string(),
            ComputedMeta {
                directive,
                lookback,
                valid_rows: self.height,
                state: None,
                origin: 0,
            },
        );
    }

    /// Attach (or replace) the carried recursive [`ComputedMeta::state`] for a
    /// cached column, enabling an O(new-rows) append resume. No-op if `name` is
    /// not a tracked computed column.
    pub fn set_computed_state(&mut self, name: &str, state: Option<Vec<f64>>) {
        if let Some(meta) = self.computed.get_mut(name) {
            meta.state = state;
        }
    }

    /// Whether any materialized directive column is stale (its `valid_rows` lags
    /// `height` after an `append`), i.e. a bulk read would see NaN until `fulfill`.
    pub fn has_stale_computed(&self) -> bool {
        self.computed.values().any(|m| m.valid_rows < self.height)
    }

    /// Snapshot of the materialized-directive columns (`name`, meta).
    pub fn computed_columns(&self) -> Vec<(String, ComputedMeta)> {
        self.computed
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Snapshot only stale materialized-directive columns. This keeps the live
    /// append/fulfill path from cloning unrelated computed metadata.
    pub fn stale_computed_columns(&self, only: Option<&str>) -> Vec<(String, ComputedMeta)> {
        self.computed
            .iter()
            .filter(|(name, meta)| {
                meta.valid_rows < self.height && only.is_none_or(|target| target == name.as_str())
            })
            .map(|(name, meta)| (name.clone(), meta.clone()))
            .collect()
    }

    /// Names of all materialized-directive columns.
    pub fn computed_names(&self) -> Vec<String> {
        self.computed.keys().cloned().collect()
    }

    /// Overwrite a computed column's rows `[from, from + tail.len())` in place
    /// (copy-on-write) with the recomputed `tail`, and mark it valid up to the
    /// full height. Directive results are F64 or Bool, so both are handled.
    /// O(tail.len()).
    pub fn update_computed_tail(&mut self, name: &str, from: usize, tail: &Column) -> Result<()> {
        let pos = self
            .name_to_idx
            .get(name)
            .copied()
            .ok_or_else(|| VolasError::ColumnNotFound(name.to_string()))?;
        match (&mut self.columns[pos], tail) {
            (Column::F64(arc), Column::F64(t)) => {
                let buf = Arc::make_mut(arc);
                for (i, &v) in t.iter().enumerate() {
                    if from + i < buf.len() {
                        buf[from + i] = v;
                    }
                }
            }
            (Column::Bool(arc, _), Column::Bool(t, _)) => {
                let buf = Arc::make_mut(arc);
                for (i, &v) in t.iter().enumerate() {
                    if from + i < buf.len() {
                        buf[from + i] = v;
                    }
                }
            }
            (col, t) => {
                return Err(VolasError::DType(format!(
                    "computed tail dtype {} does not match column \"{name}\" dtype {}",
                    t.dtype(),
                    col.dtype()
                )))
            }
        }
        if let Some(meta) = self.computed.get_mut(name) {
            meta.valid_rows = self.height;
        }
        Ok(())
    }

    /// Overwrite one F64 computed value and mark the computed column current.
    /// This avoids allocating a one-value tail column on the single-bar append path.
    pub fn update_computed_f64_value(&mut self, name: &str, row: usize, value: f64) -> Result<()> {
        let pos = self
            .name_to_idx
            .get(name)
            .copied()
            .ok_or_else(|| VolasError::ColumnNotFound(name.to_string()))?;
        match &mut self.columns[pos] {
            Column::F64(arc) => {
                let buf = Arc::make_mut(arc);
                if row < buf.len() {
                    buf[row] = value;
                }
            }
            col => {
                return Err(VolasError::DType(format!(
                    "computed scalar dtype F64 does not match column \"{name}\" dtype {}",
                    col.dtype()
                )))
            }
        }
        if let Some(meta) = self.computed.get_mut(name) {
            meta.valid_rows = self.height;
        }
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

    /// Drop the computed (cached-directive) status of the column at `col`, if any.
    fn drop_computed_at(&mut self, col: usize) {
        if let Some(name) = self.names.get(col) {
            self.computed.remove(name);
        }
    }

    /// A user write to column `col` invalidates the directive cache: `col` loses any
    /// computed status, and every OTHER cached directive column is marked fully stale —
    /// it may have been derived from `col`, so it is recomputed on next access (a bulk
    /// read raises until `fulfill`, exactly like an append). Conservative (no per-column
    /// dependency tracking), but writes are rare relative to reads.
    fn invalidate_computed_on_write_at(&mut self, col: usize) {
        self.drop_computed_at(col);
        for meta in self.computed.values_mut() {
            meta.valid_rows = 0;
            // The carried recursive state described row `valid_rows - 1`; resetting
            // `valid_rows` to 0 breaks that correspondence, so drop it. A later refresh
            // recomputes from scratch (correct) and repopulates the state.
            meta.state = None;
        }
    }

    /// Name-based variant for the `df[name] = value` whole-column replace path.
    pub fn invalidate_computed_on_write(&mut self, written_name: &str) {
        self.computed.remove(written_name);
        for meta in self.computed.values_mut() {
            meta.valid_rows = 0;
            meta.state = None;
        }
    }

    /// Rename columns (pandas `rename(columns=...)`), returning a new frame.
    /// Names not in `mapping` are kept; columns and index are shared (cheap).
    pub fn rename(&self, mapping: &HashMap<String, String>) -> Result<DataFrame> {
        let names: Vec<String> = self
            .names
            .iter()
            .map(|n| mapping.get(n).cloned().unwrap_or_else(|| n.clone()))
            .collect();
        let mut df = DataFrame::new(names, self.columns.clone(), Some((*self.index).clone()))?;
        df.aliases = Arc::clone(&self.aliases);
        Ok(df)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;

    fn sample() -> DataFrame {
        DataFrame::new(
            vec!["a".into(), "b".into()],
            vec![
                Column::f64(vec![1.0, 2.0, 3.0]),
                Column::i64(vec![10, 20, 30]),
            ],
            None,
        )
        .unwrap()
    }

    #[test]
    fn build_and_access() {
        let df = sample();
        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 2);
        assert_eq!(df.names(), &["a".to_string(), "b".to_string()]);
        assert_eq!(df.column("a").unwrap().as_f64().unwrap(), &[1.0, 2.0, 3.0]);
        assert!(df.column("missing").is_err());
    }

    #[test]
    fn select_shares_index() {
        let df = sample();
        let sub = df.select(&["b".into()]).unwrap();
        assert_eq!(sub.width(), 1);
        assert_eq!(sub.height(), 3);
        assert!(Arc::ptr_eq(df.index(), sub.index()));
    }

    #[test]
    fn slice_and_filter() {
        let df = sample();
        let s = df.slice(1, 3);
        assert_eq!(s.height(), 2);
        assert_eq!(s.column("a").unwrap().as_f64().unwrap(), &[2.0, 3.0]);

        let f = df.filter_mask(&[true, false, true]).unwrap();
        assert_eq!(f.height(), 2);
        assert_eq!(f.column("b").unwrap().as_i64().unwrap(), &[10, 30]);
    }

    #[test]
    fn append_extends() {
        let mut df = sample();
        let other = sample();
        df.append(&other).unwrap();
        assert_eq!(df.height(), 6);
        assert_eq!(
            df.column("a").unwrap().as_f64().unwrap(),
            &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn set_index_moves_column_out() {
        let df = DataFrame::new(
            vec!["t".into(), "v".into()],
            vec![Column::i64(vec![100, 200]), Column::f64(vec![1.0, 2.0])],
            None,
        )
        .unwrap();
        let indexed = df.set_index("t").unwrap();
        assert_eq!(indexed.names(), &["v".to_string()]);
        // the index carries the source column's name (pandas parity)
        assert_eq!(indexed.index().name(), Some("t"));
        assert_eq!(
            indexed.index().as_ref(),
            &Index::int64(vec![100, 200]).with_name(Some("t".into()))
        );
        assert!(indexed.column("t").is_err());
        // an f64 column cannot be an index
        assert!(df.set_index("v").is_err());
        assert!(df.set_index("missing").is_err());
    }

    #[test]
    fn row_major_export() {
        let df = sample();
        let (data, h, w) = df.to_row_major_f64();
        assert_eq!((h, w), (3, 2));
        assert_eq!(data, vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
        // A missing cell (int NA, bool NA, datetime NaT) exports as NaN, never the
        // raw placeholder — the row-major path honors validity like the 1-D path.
        let na_df = DataFrame::new(
            vec!["i".into(), "b".into(), "t".into()],
            vec![
                Column::i64_with(vec![1, 0, 3], crate::Validity::from_valid_iter(3, [true, false, true])),
                Column::bool_with(vec![true, false, false], crate::Validity::from_valid_iter(3, [true, false, true])),
                Column::datetime(vec![100, i64::MIN, 300]),
            ],
            None,
        )
        .unwrap();
        let (d2, _, _) = na_df.to_row_major_f64(); // row-major, w = 3
        assert_eq!(d2[0], 1.0); // i[0]
        assert!(d2[3].is_nan() && d2[4].is_nan() && d2[5].is_nan()); // row 1: NA, NA, NaT
        assert_eq!(d2[6], 3.0); // i[2]
    }

    #[test]
    fn new_validates_shape() {
        // names / columns count mismatch
        assert!(DataFrame::new(vec!["a".into()], vec![], None).is_err());
        // a column shorter than the frame height
        assert!(DataFrame::new(
            vec!["a".into(), "b".into()],
            vec![Column::f64(vec![1.0, 2.0]), Column::f64(vec![1.0])],
            None,
        )
        .is_err());
        // an index whose length disagrees with the height
        assert!(DataFrame::new(
            vec!["a".into()],
            vec![Column::f64(vec![1.0, 2.0])],
            Some(Index::range(3)),
        )
        .is_err());
    }

    #[test]
    fn series_extracts_a_named_column() {
        let df = sample();
        let s = df.series("a").unwrap();
        assert_eq!(s.name.as_deref(), Some("a"));
        assert_eq!(s.data.as_f64().unwrap(), &[1.0, 2.0, 3.0]);
        assert!(Arc::ptr_eq(&s.index, df.index()));
        assert!(df.series("missing").is_err());
    }

    #[test]
    fn set_column_add_replace_and_errors() {
        // adding the first column to an empty frame seeds the height + index
        let mut empty = DataFrame::new(vec![], vec![], None).unwrap();
        empty.set_column("x", Column::f64(vec![1.0, 2.0])).unwrap();
        assert_eq!(empty.height(), 2);
        assert_eq!(empty.index().as_ref(), &Index::range(2));

        // replace in place, then add a second column
        let mut df = sample();
        df.set_column("a", Column::f64(vec![9.0, 9.0, 9.0]))
            .unwrap();
        assert_eq!(df.column("a").unwrap().as_f64().unwrap(), &[9.0, 9.0, 9.0]);
        df.set_column("c", Column::f64(vec![7.0, 7.0, 7.0]))
            .unwrap();
        assert_eq!(df.width(), 3);

        // a wrong-height column is rejected
        assert!(df.set_column("d", Column::f64(vec![1.0])).is_err());
    }

    #[test]
    fn filter_mask_rejects_wrong_length() {
        assert!(sample().filter_mask(&[true, false]).is_err());
    }

    #[test]
    fn append_pads_missing_columns_by_dtype() {
        // a plain int column missing on append -> stays int64 with an NA at the gap
        // (no upcast to f64; the NA model preserves the dtype).
        let mut df = sample(); // a: f64, b: i64
        let only_a = DataFrame::new(vec!["a".into()], vec![Column::f64(vec![4.0])], None).unwrap();
        df.append(&only_a).unwrap();
        assert_eq!(df.height(), 4);
        let b = df.column("b").unwrap();
        assert_eq!(b.dtype(), DType::I64); // not upcast
        assert!(b.is_valid(0) && !b.is_valid(3)); // the padded row is NA

        // a plain bool column missing on append -> padded bool+NA (was an error).
        let mut g = DataFrame::new(
            vec!["a".into(), "flag".into()],
            vec![Column::f64(vec![1.0]), Column::bool(vec![true])],
            None,
        )
        .unwrap();
        let only_a2 = DataFrame::new(vec!["a".into()], vec![Column::f64(vec![2.0])], None).unwrap();
        g.append(&only_a2).unwrap();
        let flag = g.column("flag").unwrap();
        assert_eq!(flag.dtype(), DType::Bool);
        assert!(flag.is_valid(0) && !flag.is_valid(1));

        // a cached *bool directive* column missing on append -> padded false, stays bool.
        let mut h = DataFrame::new(
            vec!["a".into(), "sig".into()],
            vec![Column::f64(vec![1.0]), Column::bool(vec![true])],
            None,
        )
        .unwrap();
        h.set_computed("sig", "a > 0".into(), 0);
        let only_a3 = DataFrame::new(vec!["a".into()], vec![Column::f64(vec![2.0])], None).unwrap();
        h.append(&only_a3).unwrap();
        assert_eq!(h.column("sig").unwrap().as_bool().unwrap(), &[true, false]);
    }

    #[test]
    fn computed_tail_update_and_dtype_guard() {
        let mut df = sample();
        df.set_computed("a", "ma:2".into(), 1);
        assert_eq!(df.computed_columns().len(), 1);
        // overwrite the tail of the F64 column "a" with an F64 tail
        df.update_computed_tail("a", 1, &Column::f64(vec![8.0, 9.0]))
            .unwrap();
        assert_eq!(df.column("a").unwrap().as_f64().unwrap(), &[1.0, 8.0, 9.0]);
        // an F64 tail into the I64 column "b" is a dtype mismatch
        assert!(df
            .update_computed_tail("b", 0, &Column::f64(vec![1.0]))
            .is_err());
        // an unknown column errors
        assert!(df
            .update_computed_tail("nope", 0, &Column::f64(vec![1.0]))
            .is_err());
    }

    #[test]
    fn slice_carries_computed_only_with_enough_warmup() {
        // a frame with a cached recursive directive of lookback 11 (ema:12)
        let mut df = DataFrame::new(
            vec!["close".into()],
            vec![Column::f64((0..60).map(|i| i as f64).collect())],
            None,
        )
        .unwrap();
        df.set_computed("close", "ema:12".into(), 11);
        // A carried EMA state as of row 59.
        df.set_computed_state("close", Some(vec![42.0]));
        // A slice keeping >= lookback rows AND ending at `valid_rows` carries the column
        // as continuable, threading the recursive state through (the `Some` branch).
        let keep = df.slice(40, 60); // 20 rows >= 11, end == valid_rows (60)
        assert_eq!(keep.computed_columns().len(), 1);
        assert_eq!(keep.computed_columns()[0].1.valid_rows, 20);
        assert_eq!(keep.computed_columns()[0].1.state, Some(vec![42.0]));
        // a TAIL slice keeps >= lookback rows but ends BEFORE `valid_rows`, so the carried
        // state (attached to a row this sub-frame no longer ends on) is dropped — the column
        // stays computed but state-less, continuable only via the full-recompute fallback.
        let tail = df.slice(0, 50); // 50 rows >= 11, end (50) < valid_rows (60)
        assert_eq!(tail.computed_columns().len(), 1);
        assert_eq!(tail.computed_columns()[0].1.state, None);
        // a slice keeping < lookback rows drops the computed status entirely (not continuable).
        let too_short = df.slice(55, 60); // 5 rows < 11
        assert!(too_short.computed_columns().is_empty());
    }

    #[test]
    fn assign_positions_scalar_and_array() {
        let mut df = sample();
        // broadcast a scalar into two rows of the F64 column "a"
        df.assign_positions(0, &[0, 2], &Column::f64(vec![9.0]))
            .unwrap();
        assert_eq!(df.column("a").unwrap().as_f64().unwrap(), &[9.0, 2.0, 9.0]);
        // element-wise array into the I64 column "b" (integral -> stays i64)
        df.assign_positions(1, &[1, 2], &Column::f64(vec![40.0, 50.0]))
            .unwrap();
        assert_eq!(df.column("b").unwrap().as_i64().unwrap(), &[10, 40, 50]);
    }

    #[test]
    fn assign_positions_fractional_into_int_errors() {
        let mut df = sample();
        // a fractional write into the I64 column is lossy and errors (no float
        // widening) — the column stays unchanged, matching the Series scalar path
        assert!(df
            .assign_positions(1, &[0], &Column::f64(vec![1.5]))
            .is_err());
        assert_eq!(df.column("b").unwrap().dtype(), DType::I64);
        assert_eq!(df.column("b").unwrap().as_i64().unwrap(), &[10, 20, 30]);
    }

    #[test]
    fn assign_positions_nan_into_int_keeps_int_na() {
        let mut df = sample();
        // a NaN write into the I64 column keeps int64 and marks the cell NA
        // (Decision 1: no float widening, the native-NA model)
        df.assign_positions(1, &[0], &Column::f64(vec![f64::NAN]))
            .unwrap();
        let b = df.column("b").unwrap();
        assert_eq!(b.dtype(), DType::I64);
        assert!(!b.is_valid(0) && b.is_valid(1) && b.is_valid(2));
        assert_eq!(b.as_i64().unwrap()[1..], [20, 30]);
    }

    #[test]
    fn assign_positions_drops_computed_status() {
        let mut df = sample();
        df.set_computed("a", "ma:2".into(), 1);
        assert_eq!(df.computed_columns().len(), 1);
        // a manual write into the cached column drops its computed status
        df.assign_positions(0, &[0], &Column::f64(vec![7.0]))
            .unwrap();
        assert!(df.computed_columns().is_empty());
    }

    #[test]
    fn tz_convert_keeps_instant_localize_shifts() {
        use crate::tz::Tz;
        // a frame whose datetime index was ingested as UTC instants
        let ns = crate::datetime::parse_ns("2021-01-01 12:00:00").unwrap();
        let df = DataFrame::new(
            vec!["c".into()],
            vec![Column::f64(vec![1.0])],
            Some(Index::datetime(vec![ns], Tz::Utc)),
        )
        .unwrap();

        // tz_convert: instant unchanged, only the tag changes.
        let conv = df
            .tz_convert(Tz::parse("America/New_York").unwrap())
            .unwrap();
        match conv.index().kind() {
            IndexKind::Datetime(v, tz) => {
                assert_eq!(v[0], ns);
                assert_eq!(*tz, Tz::parse("America/New_York").unwrap());
            }
            _ => panic!("datetime"), // LCOV_EXCL_LINE
        }

        // tz_localize: wall-clock 12:00 reinterpreted as NY -> instant moves +5h to UTC.
        let loc = df
            .tz_localize(Tz::parse("America/New_York").unwrap())
            .unwrap();
        match loc.index().kind() {
            IndexKind::Datetime(v, _) => {
                assert_eq!(crate::datetime::format_ns(v[0]), "2021-01-01 17:00:00");
            }
            _ => panic!("datetime"), // LCOV_EXCL_LINE
        }
    }

    #[test]
    fn tz_ops_require_datetime_index() {
        use crate::tz::Tz;
        let df = sample(); // Range index
        assert!(df.tz_convert(Tz::Utc).is_err());
        assert!(df.tz_localize(Tz::Utc).is_err());
    }

    #[test]
    fn assign_positions_length_and_dtype_guards() {
        let mut df = sample();
        // wrong-length array
        assert!(df
            .assign_positions(0, &[0, 1], &Column::f64(vec![1.0, 2.0, 3.0]))
            .is_err());
        // out-of-range row
        assert!(df
            .assign_positions(0, &[9], &Column::f64(vec![1.0]))
            .is_err());
        // a bool into a numeric column COERCES (true -> 1.0), matching the Series
        // `set_float_at` path (Python `bool` is an int subclass)
        df.assign_positions(0, &[0], &Column::bool(vec![true]))
            .unwrap();
        assert_eq!(df.column("a").unwrap().as_f64().unwrap()[0], 1.0);
        // a str / datetime into a numeric column is a hard dtype error (no silent
        // funnel through `to_f64_vec`)
        assert!(df
            .assign_positions(0, &[0], &Column::str(vec!["x".into()]))
            .is_err());
        assert!(df
            .assign_positions(1, &[0], &Column::datetime(vec![123]))
            .is_err());
    }

    #[test]
    fn assign_positions_type_combinations_and_col_out_of_range() {
        let mut df = DataFrame::new(
            vec!["f".into(), "b".into(), "s".into(), "d".into()],
            vec![
                Column::f64(vec![1.0, 2.0, 3.0]),
                Column::bool(vec![true, false, true]),
                Column::str(vec!["a".into(), "b".into(), "c".into()]),
                Column::datetime(vec![10, 20, 30]),
            ],
            None,
        )
        .unwrap();
        // column position out of range
        assert!(df
            .assign_positions(99, &[0], &Column::f64(vec![1.0]))
            .is_err());
        df.assign_positions(0, &[1], &Column::i64(vec![7])).unwrap(); // F64 <- I64
        df.assign_positions(1, &[0], &Column::bool(vec![false]))
            .unwrap(); // Bool <- Bool
        df.assign_positions(2, &[0], &Column::str(vec!["z".into()]))
            .unwrap(); // Str <- Str
        df.assign_positions(3, &[2], &Column::datetime(vec![99]))
            .unwrap(); // Datetime <- Datetime
        assert_eq!(df.columns()[0].as_f64().unwrap()[1], 7.0);
        assert_eq!(df.columns()[3].as_datetime().unwrap()[2], 99);
    }

    #[test]
    fn tz_localize_rejects_nonexistent_wall_time() {
        use crate::tz::Tz;
        // 02:30 on 2020-03-08 does not exist in America/New_York (spring-forward gap).
        let ns = crate::datetime::parse_ns("2020-03-08 02:30:00").unwrap();
        let df = DataFrame::new(
            vec!["c".into()],
            vec![Column::f64(vec![1.0])],
            Some(Index::datetime(vec![ns], Tz::Utc)),
        )
        .unwrap();
        assert!(df
            .tz_localize(Tz::parse("America/New_York").unwrap())
            .is_err());
    }
}
