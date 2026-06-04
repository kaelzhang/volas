//! DataFrame: ordered, named columns sharing a single row index.

use std::collections::HashMap;
use std::sync::Arc;

use crate::column::Column;
use crate::error::{Result, VolasError};
use crate::index::Index;
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
}

/// A 2-D, column-oriented, time-indexed table. All columns share one index and
/// have equal length (`height`).
#[derive(Clone, Debug)]
pub struct DataFrame {
    names: Vec<String>,
    columns: Vec<Column>,
    name_to_idx: HashMap<String, usize>,
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
            None => Index::Range(height),
        };
        let mut name_to_idx = HashMap::with_capacity(names.len());
        for (i, n) in names.iter().enumerate() {
            name_to_idx.insert(n.clone(), i);
        }
        Ok(DataFrame {
            names,
            columns,
            name_to_idx,
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
            return Err(VolasError::Value(format!("column \"{src_name}\" not exists")));
        }
        let mut aliases = (*self.aliases).clone();
        aliases.insert(as_name.to_string(), src_name.to_string());
        let mut df = self.clone();
        df.aliases = Arc::new(aliases);
        Ok(df)
    }

    /// Gather rows by position into a new frame (carries aliases).
    pub fn take(&self, positions: &[usize]) -> DataFrame {
        let columns: Vec<Column> = self.columns.iter().map(|c| c.take(positions)).collect();
        let index = self.index.take(positions);
        let mut df =
            DataFrame::new(self.names.clone(), columns, Some(index)).expect("take keeps shape");
        df.aliases = Arc::clone(&self.aliases);
        df
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
                self.index = Arc::new(Index::Range(self.height));
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
                self.name_to_idx.insert(name.to_string(), self.columns.len());
                self.names.push(name.to_string());
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
        let index = Index::from_column(&self.columns[pos])?;
        let mut names = self.names.clone();
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
        match self.index.as_ref() {
            Index::Datetime(_, _) => {
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
        let (values, cur) = match self.index.as_ref() {
            Index::Datetime(v, cur) => (v.clone(), *cur),
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
        df.index = Arc::new(Index::Datetime(shifted, tz));
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
            names: names.to_vec(),
            columns,
            name_to_idx,
            index: Arc::clone(&self.index),
            height: self.height,
            aliases: Arc::clone(&self.aliases),
            computed: HashMap::new(),
        })
    }

    /// A `[start, end)` row slice.
    pub fn slice(&self, start: usize, end: usize) -> DataFrame {
        let start = start.min(self.height);
        let end = end.max(start).min(self.height);
        let columns: Vec<Column> = self.columns.iter().map(|c| c.slice(start, end)).collect();
        let index = self.index.slice(start, end);
        let mut df =
            DataFrame::new(self.names.clone(), columns, Some(index)).expect("slice keeps shape");
        df.aliases = Arc::clone(&self.aliases);
        // The sliced cached-directive *values* are carried (correct, full-history),
        // but the computed *metadata* is dropped: the columns become plain data.
        // Exact continuation across `slice -> append` for recursive indicators
        // (carrying each indicator's internal recursive state) is a deferred
        // feature; until then a slice is a read-only snapshot, not continuable.
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
        let index = self.index.take(&idx);
        let mut df = DataFrame::new(self.names.clone(), columns, Some(index))?;
        df.aliases = Arc::clone(&self.aliases);
        Ok(df)
    }

    /// Append the rows of `other` (matched by column name) in place. Columns of
    /// `self` absent from `other` are NaN-padded (so a frame with materialized
    /// directive columns can take raw bars; `fulfill` then refreshes them).
    /// Computed-column metadata is retained, leaving the new rows stale.
    pub fn append(&mut self, other: &DataFrame) -> Result<()> {
        let names = self.names.clone();
        let oh = other.height;
        for n in &names {
            let pos = *self.name_to_idx.get(n).expect("name came from self");
            match other.column(n) {
                Ok(oc) => self.columns[pos].append(oc)?,
                Err(_) => {
                    // column `n` is missing from `other` — pad the new rows.
                    let is_computed = self.computed.contains_key(n);
                    match (&self.columns[pos], is_computed) {
                        // F64: NaN marks the gap.
                        (Column::F64(_), _) => {
                            self.columns[pos].append(&Column::f64(vec![f64::NAN; oh]))?;
                        }
                        // A cached *bool* directive: pad a stale `false` placeholder;
                        // `fulfill` rewrites the correct bool tail (the column stays a mask).
                        (Column::Bool(_), true) => {
                            self.columns[pos].append(&Column::bool(vec![false; oh]))?;
                        }
                        // A plain int column: upcast to F64 so NaN can mark the gap
                        // (NaN distinguishes "missing" from a real 0).
                        (Column::I64(v), false) => {
                            let mut f: Vec<f64> = v.iter().map(|&x| x as f64).collect();
                            f.extend(std::iter::repeat(f64::NAN).take(oh));
                            self.columns[pos] = Column::f64(f);
                        }
                        // Plain bool / str / datetime cannot represent "missing".
                        (other_col, _) => {
                            return Err(VolasError::DType(format!(
                                "cannot append: column \"{n}\" ({}) is missing from the \
                                 appended frame and has no missing-value representation",
                                other_col.dtype()
                            )));
                        }
                    }
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
            },
        );
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
            (Column::Bool(arc), Column::Bool(t)) => {
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

    /// Assign `values` into column position `col` at the given row `positions`
    /// (copy-on-write via [`Arc::make_mut`]). `values` is broadcast when it has
    /// length 1, otherwise its length must equal `positions.len()`. This backs
    /// `df.loc[...] = `, `df.iloc[...] = `, `df.at[...] = ` and `df.iat[...] = `.
    ///
    /// Dtype handling, matching pandas where reasonable:
    /// - numeric kinds cross-cast (`I64` <-> `F64`);
    /// - an `I64` column receiving a **fractional** (or NaN) `F64` value upgrades
    ///   the whole column to `F64` (pandas widens an int column on a float write);
    /// - `Bool` / `Str` / `Datetime` targets require a matching-kind value.
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
        // Broadcast a length-1 value across every position.
        let pick = |k: usize| if values.len() == 1 { 0 } else { k };

        // I64 column + a fractional / NaN F64 write -> widen the column to F64.
        if let (Column::I64(arc), Column::F64(src)) = (&self.columns[col], values) {
            if src.iter().any(|x| x.is_nan() || x.fract() != 0.0) {
                let mut f: Vec<f64> = arc.iter().map(|&x| x as f64).collect();
                for (k, &p) in positions.iter().enumerate() {
                    f[p] = src[pick(k)];
                }
                self.columns[col] = Column::f64(f);
                self.drop_computed_at(col);
                return Ok(());
            }
        }

        match (&mut self.columns[col], values) {
            (Column::F64(arc), Column::F64(src)) => {
                let buf = Arc::make_mut(arc);
                for (k, &p) in positions.iter().enumerate() {
                    buf[p] = src[pick(k)];
                }
            }
            (Column::F64(arc), Column::I64(src)) => {
                let buf = Arc::make_mut(arc);
                for (k, &p) in positions.iter().enumerate() {
                    buf[p] = src[pick(k)] as f64;
                }
            }
            (Column::I64(arc), Column::I64(src)) => {
                let buf = Arc::make_mut(arc);
                for (k, &p) in positions.iter().enumerate() {
                    buf[p] = src[pick(k)];
                }
            }
            // All-integral F64 (the widening branch above did not fire) -> store as i64.
            (Column::I64(arc), Column::F64(src)) => {
                let buf = Arc::make_mut(arc);
                for (k, &p) in positions.iter().enumerate() {
                    buf[p] = src[pick(k)] as i64;
                }
            }
            (Column::Bool(arc), Column::Bool(src)) => {
                let buf = Arc::make_mut(arc);
                for (k, &p) in positions.iter().enumerate() {
                    buf[p] = src[pick(k)];
                }
            }
            (Column::Str(arc), Column::Str(src)) => {
                let buf = Arc::make_mut(arc);
                for (k, &p) in positions.iter().enumerate() {
                    buf[p] = src[pick(k)].clone();
                }
            }
            (Column::Datetime(arc), Column::Datetime(src)) => {
                let buf = Arc::make_mut(arc);
                for (k, &p) in positions.iter().enumerate() {
                    buf[p] = src[pick(k)];
                }
            }
            (target, src) => {
                return Err(VolasError::DType(format!(
                    "cannot assign {} values into a {} column",
                    src.dtype(),
                    target.dtype()
                )));
            }
        }
        self.drop_computed_at(col);
        Ok(())
    }

    /// Drop the computed (cached-directive) status of the column at `col`, if any.
    fn drop_computed_at(&mut self, col: usize) {
        if let Some(name) = self.names.get(col) {
            self.computed.remove(name);
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
        let mut df = DataFrame::new(self.names.clone(), columns, Some((*self.index).clone()))?;
        df.aliases = Arc::clone(&self.aliases);
        Ok(df)
    }

    /// Value equality (pandas `DataFrame.equals`): same column names + order,
    /// same index, and value-equal columns (`NaN == NaN`).
    pub fn equals(&self, other: &DataFrame) -> bool {
        self.height == other.height
            && self.names == other.names
            && self.index.as_ref() == other.index.as_ref()
            && self
                .columns
                .iter()
                .zip(&other.columns)
                .all(|(a, b)| a.equals(b))
    }

    /// Flatten to a row-major (C-order) 2-D `f64` buffer for NumPy export,
    /// returning `(data, height, width)`.
    pub fn to_row_major_f64(&self) -> (Vec<f64>, usize, usize) {
        let h = self.height;
        let w = self.columns.len();
        let mut out = vec![0.0f64; h * w];
        for (j, c) in self.columns.iter().enumerate() {
            for i in 0..h {
                out[i * w + j] = c.get_f64(i);
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
        assert_eq!(indexed.index().as_ref(), &Index::Int64(vec![100, 200]));
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
            Some(Index::Range(3)),
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
        assert_eq!(empty.index().as_ref(), &Index::Range(2));

        // replace in place, then add a second column
        let mut df = sample();
        df.set_column("a", Column::f64(vec![9.0, 9.0, 9.0])).unwrap();
        assert_eq!(df.column("a").unwrap().as_f64().unwrap(), &[9.0, 9.0, 9.0]);
        df.set_column("c", Column::f64(vec![7.0, 7.0, 7.0])).unwrap();
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
        // a plain int column missing on append -> upcast to f64 + NaN (EX-11).
        let mut df = sample(); // a: f64, b: i64
        let only_a =
            DataFrame::new(vec!["a".into()], vec![Column::f64(vec![4.0])], None).unwrap();
        df.append(&only_a).unwrap();
        assert_eq!(df.height(), 4);
        let b = df.column("b").unwrap();
        assert!(b.as_f64().is_some()); // upcast to F64
        assert!(b.as_f64().unwrap()[3].is_nan());

        // a plain bool column missing on append -> error (no missing representation).
        let mut g = DataFrame::new(
            vec!["a".into(), "flag".into()],
            vec![Column::f64(vec![1.0]), Column::bool(vec![true])],
            None,
        )
        .unwrap();
        let only_a2 =
            DataFrame::new(vec!["a".into()], vec![Column::f64(vec![2.0])], None).unwrap();
        assert!(g.append(&only_a2).is_err());

        // a cached *bool directive* column missing on append -> padded false, stays bool.
        let mut h = DataFrame::new(
            vec!["a".into(), "sig".into()],
            vec![Column::f64(vec![1.0]), Column::bool(vec![true])],
            None,
        )
        .unwrap();
        h.set_computed("sig", "a > 0".into(), 0);
        let only_a3 =
            DataFrame::new(vec!["a".into()], vec![Column::f64(vec![2.0])], None).unwrap();
        h.append(&only_a3).unwrap();
        assert_eq!(h.column("sig").unwrap().as_bool().unwrap(), &[true, false]);
    }

    #[test]
    fn computed_tail_update_and_dtype_guard() {
        let mut df = sample();
        df.set_computed("a", "ma:2".into(), 1);
        assert_eq!(df.computed_columns().len(), 1);
        // overwrite the tail of the F64 column "a" with an F64 tail
        df.update_computed_tail("a", 1, &Column::f64(vec![8.0, 9.0])).unwrap();
        assert_eq!(df.column("a").unwrap().as_f64().unwrap(), &[1.0, 8.0, 9.0]);
        // an F64 tail into the I64 column "b" is a dtype mismatch
        assert!(df.update_computed_tail("b", 0, &Column::f64(vec![1.0])).is_err());
        // an unknown column errors
        assert!(df.update_computed_tail("nope", 0, &Column::f64(vec![1.0])).is_err());
    }

    #[test]
    fn assign_positions_scalar_and_array() {
        let mut df = sample();
        // broadcast a scalar into two rows of the F64 column "a"
        df.assign_positions(0, &[0, 2], &Column::f64(vec![9.0])).unwrap();
        assert_eq!(df.column("a").unwrap().as_f64().unwrap(), &[9.0, 2.0, 9.0]);
        // element-wise array into the I64 column "b" (integral -> stays i64)
        df.assign_positions(1, &[1, 2], &Column::f64(vec![40.0, 50.0])).unwrap();
        assert_eq!(df.column("b").unwrap().as_i64().unwrap(), &[10, 40, 50]);
    }

    #[test]
    fn assign_positions_widens_int_on_fractional() {
        let mut df = sample();
        // a fractional write into the I64 column widens the whole column to F64
        df.assign_positions(1, &[0], &Column::f64(vec![1.5])).unwrap();
        assert_eq!(df.column("b").unwrap().dtype(), DType::F64);
        assert_eq!(df.column("b").unwrap().as_f64().unwrap(), &[1.5, 20.0, 30.0]);
    }

    #[test]
    fn assign_positions_drops_computed_status() {
        let mut df = sample();
        df.set_computed("a", "ma:2".into(), 1);
        assert_eq!(df.computed_columns().len(), 1);
        // a manual write into the cached column drops its computed status
        df.assign_positions(0, &[0], &Column::f64(vec![7.0])).unwrap();
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
            Some(Index::Datetime(vec![ns], Tz::Utc)),
        )
        .unwrap();

        // tz_convert: instant unchanged, only the tag changes.
        let conv = df.tz_convert(Tz::parse("America/New_York").unwrap()).unwrap();
        match conv.index().as_ref() {
            Index::Datetime(v, tz) => {
                assert_eq!(v[0], ns);
                assert_eq!(*tz, Tz::parse("America/New_York").unwrap());
            }
            _ => panic!("datetime"),
        }

        // tz_localize: wall-clock 12:00 reinterpreted as NY -> instant moves +5h to UTC.
        let loc = df.tz_localize(Tz::parse("America/New_York").unwrap()).unwrap();
        match loc.index().as_ref() {
            Index::Datetime(v, _) => {
                assert_eq!(crate::datetime::format_ns(v[0]), "2021-01-01 17:00:00");
            }
            _ => panic!("datetime"),
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
        assert!(df.assign_positions(0, &[0, 1], &Column::f64(vec![1.0, 2.0, 3.0])).is_err());
        // out-of-range row
        assert!(df.assign_positions(0, &[9], &Column::f64(vec![1.0])).is_err());
        // bool into a numeric column
        assert!(df.assign_positions(0, &[0], &Column::bool(vec![true])).is_err());
    }
}
