//! Index: the row labels shared by a frame and the series drawn from it.

use crate::column::Column;
use crate::error::{Result, VolasError};
use crate::tz::Tz;

/// Row labels plus an optional name.
///
/// `name` is index-wide metadata (pandas `Index.name`): it is recorded by
/// `set_index` (from the source column), restored by `reset_index`, and shown in
/// a frame's / series' repr. It is `None` for an unnamed index (the default
/// `Range` index, a freshly built datetime index, …).
///
/// `kind` is the label storage. A `Datetime` kind carries its own [`Tz`]:
/// storage is always UTC epoch-ns, but the tz governs how those instants render
/// and how bare-string / day-bucket matching maps to wall-clock time (see
/// [`crate::tz`]). Both the name and a datetime kind's tz ride with the shared
/// `Arc<Index>`, so a frame and every series drawn from it agree on them for free.
#[derive(Clone, Debug, PartialEq)]
pub struct Index {
    /// Optional index name (pandas `Index.name`); `None` when unnamed.
    pub name: Option<String>,
    /// The label storage / kind.
    pub kind: IndexKind,
}

/// The label storage backing an [`Index`].
///
/// Defaults to an implicit `0..n` range; a `Datetime` kind is the common OHLCV
/// case (i64 nanoseconds since the Unix epoch); a `Str` kind (pandas
/// object/string index) supports symbol-keyed lookup.
#[derive(Clone, Debug, PartialEq)]
pub enum IndexKind {
    /// Implicit `0..n` integer labels.
    Range(usize),
    /// Explicit integer labels.
    Int64(Vec<i64>),
    /// Datetime labels as i64 nanoseconds since the Unix epoch, with a display /
    /// matching timezone (UTC by default).
    Datetime(Vec<i64>, Tz),
    /// String labels (pandas object/string index).
    Str(Vec<String>),
}

impl IndexKind {
    /// Materialize the numeric labels as `i64`. Numeric kinds only — a string
    /// kind has no i64 labels (its callers guard against it).
    fn to_i64_labels(&self) -> Vec<i64> {
        match self {
            IndexKind::Range(n) => (0..*n as i64).collect(),
            IndexKind::Int64(v) => v.clone(),
            IndexKind::Datetime(v, _) => v.clone(),
            IndexKind::Str(_) => unreachable!("string indexes have no i64 labels"), // LCOV_EXCL_LINE
        }
    }
}

/// A single row-index label: an integer / datetime (`I64`, ns) or a string
/// (`Str`). It decouples label *lookup* from the index's storage so the same
/// `.loc` / `.at` / `.loc[a:b]` / `drop` paths serve every index kind.
#[derive(Clone, Debug, PartialEq)]
pub enum Label {
    /// An integer or datetime-ns label.
    I64(i64),
    /// A string label.
    Str(String),
}

impl Label {
    /// The i64 payload, if this is an `I64` label.
    pub fn as_i64(&self) -> Option<i64> {
        if let Label::I64(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// The string payload, if this is a `Str` label.
    pub fn as_str(&self) -> Option<&str> {
        if let Label::Str(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }
}

impl Index {
    /// An unnamed implicit `0..n` range index.
    pub fn range(n: usize) -> Index {
        Index {
            name: None,
            kind: IndexKind::Range(n),
        }
    }

    /// An unnamed explicit integer index.
    pub fn int64(labels: Vec<i64>) -> Index {
        Index {
            name: None,
            kind: IndexKind::Int64(labels),
        }
    }

    /// An unnamed datetime index (UTC-ns `labels`, tagged with `tz`).
    pub fn datetime(labels: Vec<i64>, tz: Tz) -> Index {
        Index {
            name: None,
            kind: IndexKind::Datetime(labels, tz),
        }
    }

    /// An unnamed string index.
    pub fn str(labels: Vec<String>) -> Index {
        Index {
            name: None,
            kind: IndexKind::Str(labels),
        }
    }

    /// The label storage / kind.
    pub fn kind(&self) -> &IndexKind {
        &self.kind
    }

    /// The index name, if any (pandas `Index.name`).
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Return this index with its name set (builder; consumes `self`).
    pub fn with_name(mut self, name: Option<String>) -> Index {
        self.name = name;
        self
    }

    /// Build an unnamed index from a column (for `set_index`): a `Datetime`
    /// column becomes a `DatetimeIndex`, an `I64` column an `Int64Index`, a `Str`
    /// column a string index. Float / bool columns are not valid labels.
    pub fn from_column(col: &Column) -> Result<Index> {
        Index::from_column_tz(col, Tz::Naive)
    }

    /// Build an unnamed index from a column, tagging a `Datetime` column with
    /// `tz` (otherwise the tz is ignored).
    ///
    /// An `I64` / `Str` column carrying `volas.NA` is rejected: an int/str index
    /// has no missing-label representation, so building one would silently turn
    /// each NA into its physical placeholder (`0` / `""`) — an ordinary label a
    /// `.loc` lookup can match (C2/C4). Datetime is the one nullable index kind:
    /// `NaT` is a physical sentinel inside the label vector itself, rendered as
    /// `NaT` and sorted last.
    pub fn from_column_tz(col: &Column, tz: Tz) -> Result<Index> {
        let kind = match col {
            // Datetime is exempt from the unique-label rule (F34 refinement):
            // real market data legitimately carries duplicate timestamps (resent
            // forming bars, multiple ticks per ts, NaT batches) and cumulate /
            // sort own the dedup semantics. int64/str labels stay strictly unique.
            Column::Datetime(v) => IndexKind::Datetime(v.to_vec(), tz),
            Column::I64(v, _) => {
                require_no_missing_labels(col, "int64")?;
                require_unique_labels(v, "int64")?;
                IndexKind::Int64(v.to_vec())
            }
            Column::Str(v, _) => {
                require_no_missing_labels(col, "str")?;
                require_unique_labels(v, "str")?;
                IndexKind::Str(v.to_vec())
            }
            other => {
                return Err(VolasError::DType(format!(
                    "cannot use a {} column as an index (only datetime / int64 / string)",
                    other.dtype()
                )))
            }
        };
        Ok(Index { name: None, kind })
    }

    /// The timezone of a `Datetime` index ([`Tz::Naive`] for every other kind).
    pub fn tz(&self) -> Tz {
        match &self.kind {
            IndexKind::Datetime(_, tz) => *tz,
            _ => Tz::Naive,
        }
    }

    /// Return this index with its timezone set (no-op for a non-datetime index);
    /// the name is preserved.
    pub fn with_tz(mut self, tz: Tz) -> Index {
        if let IndexKind::Datetime(_, cur) = &mut self.kind {
            *cur = tz;
        }
        self
    }

    /// Number of labels.
    pub fn len(&self) -> usize {
        match &self.kind {
            IndexKind::Range(n) => *n,
            IndexKind::Int64(v) => v.len(),
            IndexKind::Datetime(v, _) => v.len(),
            IndexKind::Str(v) => v.len(),
        }
    }

    /// The label at position `i` (for membership tests like `drop`).
    pub fn label_at(&self, i: usize) -> Label {
        match &self.kind {
            IndexKind::Range(_) => Label::I64(i as i64),
            IndexKind::Int64(v) => Label::I64(v[i]),
            IndexKind::Datetime(v, _) => Label::I64(v[i]),
            IndexKind::Str(v) => Label::Str(v[i].clone()),
        }
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Materialize the numeric labels as `i64`. Numeric indexes only — string
    /// indexes are handled by their own paths (`label_slice` / `append` guard).
    pub fn to_i64_labels(&self) -> Vec<i64> {
        self.kind.to_i64_labels()
    }

    /// A `[start, end)` slice (the name and a datetime tz are preserved).
    pub fn slice(&self, start: usize, end: usize) -> Index {
        let kind = match &self.kind {
            // A range starting at 0 stays an (implicit) range; a non-zero start
            // cannot be expressed as `Range`, so it materializes the actual labels
            // (`start..end`) — e.g. `tail(2)` of `0..7` keeps labels `[5, 6]`, not
            // a reset `[0, 1]`, matching pandas.
            IndexKind::Range(_) if start == 0 => IndexKind::Range(end),
            IndexKind::Range(_) => IndexKind::Int64((start as i64..end as i64).collect()),
            IndexKind::Int64(v) => IndexKind::Int64(v[start..end].to_vec()),
            IndexKind::Datetime(v, tz) => IndexKind::Datetime(v[start..end].to_vec(), *tz),
            IndexKind::Str(v) => IndexKind::Str(v[start..end].to_vec()),
        };
        Index {
            name: self.name.clone(),
            kind,
        }
    }

    /// Gather the given positions (the name and a datetime tz are preserved).
    pub fn take(&self, idx: &[usize]) -> Index {
        let kind = match &self.kind {
            IndexKind::Range(_) => IndexKind::Int64(idx.iter().map(|&i| i as i64).collect()),
            IndexKind::Int64(v) => IndexKind::Int64(idx.iter().map(|&i| v[i]).collect()),
            IndexKind::Datetime(v, tz) => {
                IndexKind::Datetime(idx.iter().map(|&i| v[i]).collect(), *tz)
            }
            IndexKind::Str(v) => IndexKind::Str(idx.iter().map(|&i| v[i].clone()).collect()),
        };
        Index {
            name: self.name.clone(),
            kind,
        }
    }

    /// Value equality of the **labels** (pandas `.equals` index semantics): a
    /// `RangeIndex` equals the same integer labels materialized as `Int64`, but an
    /// integer index never equals a datetime or string index. The index *name* and
    /// a datetime *tz* (display metadata) are ignored.
    pub fn label_eq(&self, other: &Index) -> bool {
        use IndexKind::*;
        match (&self.kind, &other.kind) {
            (Range(_) | Int64(_), Range(_) | Int64(_)) => {
                self.to_i64_labels() == other.to_i64_labels()
            }
            (Datetime(a, _), Datetime(b, _)) => a == b,
            (Str(a), Str(b)) => a == b,
            _ => false, // different label kinds are never equal
        }
    }

    /// The positions that sort the index by label (`sort_index`). Numeric kinds
    /// sort numerically, a string index lexicographically.
    pub fn argsort(&self, ascending: bool) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.len()).collect();
        let cmp_dir = |o: std::cmp::Ordering| if ascending { o } else { o.reverse() };
        match &self.kind {
            IndexKind::Str(v) => idx.sort_by(|&a, &b| cmp_dir(v[a].cmp(&v[b]))),
            // a datetime index sinks a NaT label (i64::MIN) to the end in both
            // directions — matching sort_values / pandas na_position='last' (V9) —
            // rather than sorting it as the smallest i64, which put NaT first.
            IndexKind::Datetime(v, _) => idx.sort_by(|&a, &b| {
                use std::cmp::Ordering::*;
                match (v[a] == i64::MIN, v[b] == i64::MIN) {
                    (true, true) => Equal,
                    (true, false) => Greater,
                    (false, true) => Less,
                    (false, false) => cmp_dir(v[a].cmp(&v[b])),
                }
            }),
            _ => {
                let labels = self.to_i64_labels();
                idx.sort_by(|&a, &b| cmp_dir(labels[a].cmp(&labels[b])));
            }
        }
        idx
    }

    /// Materialize the labels as a [`Column`] (for `reset_index`).
    pub fn to_column(&self) -> Column {
        match &self.kind {
            IndexKind::Range(n) => Column::i64((0..*n as i64).collect()),
            IndexKind::Int64(v) => Column::i64(v.clone()),
            IndexKind::Datetime(v, _) => Column::datetime(v.clone()),
            IndexKind::Str(v) => Column::str(v.clone()),
        }
    }

    /// Concatenate two indexes (extending labels), keeping the left index's name.
    /// Same-kind indexes preserve their kind; mixing numeric kinds yields
    /// `Int64`; mixing a string index with a numeric one is an error.
    pub fn append(&self, other: &Index) -> Result<Index> {
        use IndexKind::*;
        let kind = match (&self.kind, &other.kind) {
            (Range(a), Range(b)) => Range(a + b),
            (Datetime(a, ta), Datetime(b, _)) => Datetime([a.as_slice(), b].concat(), *ta),
            (Str(a), Str(b)) => Str([a.as_slice(), b].concat()),
            (Str(_), _) | (_, Str(_)) => {
                return Err(VolasError::Shape(
                    "cannot append a string index to a non-string index".into(),
                ))
            }
            // remaining: numeric mixes (Range / Int64 / Datetime) -> Int64 labels
            (a, b) => Int64([a.to_i64_labels(), b.to_i64_labels()].concat()),
        };
        Ok(Index {
            name: self.name.clone(),
            kind,
        })
    }

    /// Extend in place by the labels of `other` — the amortized-O(1) counterpart
    /// of [`append`](Self::append), used by the live single-bar hot path; the
    /// growing index keeps its own name (so appending an unnamed bar to a named
    /// index does not drop the name). Same-kind indexes grow their buffer; a
    /// numeric-kind mix collapses to `Int64`; mixing a string index with a
    /// numeric one is an error.
    pub fn extend(&mut self, other: &Index) -> Result<()> {
        use IndexKind::*;
        match (&mut self.kind, &other.kind) {
            (Range(a), Range(b)) => *a += b,
            (Datetime(a, _), Datetime(b, _)) => a.extend_from_slice(b),
            (Int64(a), Int64(b)) => a.extend_from_slice(b),
            (Str(a), Str(b)) => a.extend(b.iter().cloned()),
            (Str(_), _) | (_, Str(_)) => {
                return Err(VolasError::Shape(
                    "cannot append a string index to a non-string index".into(),
                ))
            }
            // numeric-kind mix (Range / Int64 / Datetime) -> Int64 labels
            (slot, b) => {
                let mut labels = slot.to_i64_labels();
                labels.extend(b.to_i64_labels());
                *slot = Int64(labels);
            }
        }
        Ok(())
    }

    /// Position of the first label exactly equal to `label`. Returns `None` if
    /// the label's kind does not match the index's kind.
    pub fn position_of(&self, label: &Label) -> Option<usize> {
        match (&self.kind, label) {
            (IndexKind::Range(n), Label::I64(v)) => {
                if *v >= 0 && (*v as usize) < *n {
                    Some(*v as usize)
                } else {
                    None
                }
            }
            (IndexKind::Int64(vs), Label::I64(v)) => vs.iter().position(|x| x == v),
            (IndexKind::Datetime(vs, _), Label::I64(v)) => vs.iter().position(|x| x == v),
            (IndexKind::Str(vs), Label::Str(s)) => vs.iter().position(|x| x == s),
            _ => None,
        }
    }

    /// `[start, end)` positions covering the inclusive label range `[lo, hi]`
    /// (ascending labels; pandas `.loc` slice semantics). Either bound may be
    /// `None` for open-ended. Numeric indexes compare numerically; a string
    /// index compares lexicographically.
    pub fn label_slice(&self, lo: Option<&Label>, hi: Option<&Label>) -> (usize, usize) {
        match &self.kind {
            IndexKind::Str(labels) => {
                let start = lo.and_then(Label::as_str).map_or(0, |lo| {
                    labels
                        .iter()
                        .position(|x| x.as_str() >= lo)
                        .unwrap_or(labels.len())
                });
                let end = hi.and_then(Label::as_str).map_or(labels.len(), |hi| {
                    labels
                        .iter()
                        .rposition(|x| x.as_str() <= hi)
                        .map_or(0, |p| p + 1)
                });
                (start, end.max(start))
            }
            _ => {
                let labels = self.to_i64_labels();
                let start = lo.and_then(Label::as_i64).map_or(0, |lo| {
                    labels.iter().position(|&x| x >= lo).unwrap_or(labels.len())
                });
                let end = hi.and_then(Label::as_i64).map_or(labels.len(), |hi| {
                    labels.iter().rposition(|&x| x <= hi).map_or(0, |p| p + 1)
                });
                (start, end.max(start))
            }
        }
    }
}

/// Reject building an index from a column that carries `volas.NA`: with no
/// missing-label representation for `int64` / `str`, the NA's physical
/// placeholder (`0` / `""`) would become an ordinary, lookup-matchable label.
fn require_no_missing_labels(col: &Column, kind: &str) -> Result<()> {
    if col.null_count() > 0 {
        return Err(VolasError::Value(format!(
            "cannot use a {kind} column containing volas.NA as an index (a missing \
             label has no {kind} representation); drop or fill the NA rows first"
        )));
    }
    Ok(())
}

/// Label access assumes unique labels (F34, decision 1B): a duplicate-label
/// index makes `.loc[label]` ill-defined (one row or many?), so it is rejected
/// at creation — the same creation-time guard as the NA-label rule above.
fn require_unique_labels<T: std::hash::Hash + Eq>(labels: &[T], kind: &str) -> Result<()> {
    let mut seen = std::collections::HashSet::with_capacity(labels.len());
    for l in labels {
        if !seen.insert(l) {
            return Err(VolasError::Value(format!(
                "cannot use a {kind} column with duplicate labels as an index \
                 (label access assumes unique labels)"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_datetime_and_int_columns() {
        assert_eq!(
            Index::from_column(&Column::datetime(vec![5, 6])).unwrap(),
            Index::datetime(vec![5, 6], Tz::Naive)
        );
        assert_eq!(
            Index::from_column(&Column::i64(vec![1, 2])).unwrap(),
            Index::int64(vec![1, 2])
        );
    }

    #[test]
    fn from_unsupported_column_errors() {
        // float / bool columns are not valid labels (string is — see below)
        assert!(Index::from_column(&Column::f64(vec![1.0])).is_err());
        assert!(Index::from_column(&Column::bool(vec![true])).is_err());
    }

    #[test]
    fn is_empty_labels_and_position_of() {
        assert!(Index::range(0).is_empty());
        assert!(!Index::range(3).is_empty());

        assert_eq!(Index::range(3).to_i64_labels(), vec![0, 1, 2]);
        assert_eq!(Index::int64(vec![5, 6]).to_i64_labels(), vec![5, 6]);
        assert_eq!(
            Index::datetime(vec![10, 20], Tz::Utc).to_i64_labels(),
            vec![10, 20]
        );

        let i64 = Label::I64;
        assert_eq!(Index::range(5).position_of(&i64(3)), Some(3));
        assert_eq!(Index::range(5).position_of(&i64(9)), None);
        assert_eq!(Index::range(5).position_of(&i64(-1)), None);
        assert_eq!(Index::int64(vec![10, 20, 30]).position_of(&i64(20)), Some(1));
        assert_eq!(Index::int64(vec![10, 20]).position_of(&i64(99)), None);
        assert_eq!(
            Index::datetime(vec![100, 200], Tz::Utc).position_of(&i64(200)),
            Some(1)
        );

        // take() on an Int64 index gathers the labels at those positions
        assert_eq!(
            Index::int64(vec![10, 20, 30]).take(&[2, 0]),
            Index::int64(vec![30, 10])
        );
    }

    fn str_index(labels: &[&str]) -> Index {
        Index::str(labels.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn string_index_construction_and_ops() {
        // a string column becomes a string index
        let ix = Index::from_column(&Column::str(vec!["a".into(), "b".into()])).unwrap();
        assert_eq!(ix, str_index(&["a", "b"]));

        let ix = str_index(&["a", "b", "c", "d"]);
        assert_eq!(ix.len(), 4);
        assert!(!ix.is_empty());
        assert_eq!(ix.label_at(2), Label::Str("c".into()));
        assert_eq!(ix.slice(1, 3), str_index(&["b", "c"]));
        assert_eq!(ix.take(&[3, 0]), str_index(&["d", "a"]));
    }

    #[test]
    fn string_index_lookup_and_slice() {
        let ix = str_index(&["aa", "bb", "cc", "dd"]);
        // exact lookup, kind-matched
        assert_eq!(ix.position_of(&Label::Str("cc".into())), Some(2));
        assert_eq!(ix.position_of(&Label::Str("zz".into())), None);
        // a numeric label against a string index never matches
        assert_eq!(ix.position_of(&Label::I64(1)), None);
        // lexicographic slice [bb, cc]
        let lo = Label::Str("bb".into());
        let hi = Label::Str("cc".into());
        assert_eq!(ix.label_slice(Some(&lo), Some(&hi)), (1, 3));
        // open-ended upper bound: [cc, end)
        assert_eq!(ix.label_slice(Some(&lo), None), (1, 4));
    }

    #[test]
    fn string_index_append_rules() {
        let a = str_index(&["x", "y"]);
        let b = str_index(&["z"]);
        assert_eq!(a.append(&b).unwrap(), str_index(&["x", "y", "z"]));
        // mixing a string index with a numeric one is an error
        assert!(a.append(&Index::range(2)).is_err());
        assert!(Index::range(2).append(&a).is_err());
    }

    #[test]
    fn extend_grows_in_place_per_kind() {
        // same-kind grows the buffer in place (the live append hot path)
        let mut r = Index::range(3);
        r.extend(&Index::range(2)).unwrap();
        assert_eq!(r, Index::range(5));

        let mut d = Index::datetime(vec![1, 2], Tz::Utc);
        d.extend(&Index::datetime(vec![3], Tz::Utc)).unwrap();
        assert_eq!(d, Index::datetime(vec![1, 2, 3], Tz::Utc));

        let mut s = str_index(&["a", "b"]);
        s.extend(&str_index(&["c"])).unwrap();
        assert_eq!(s, str_index(&["a", "b", "c"]));

        // a numeric-kind mix collapses to Int64 (matches `append`)
        let mut m = Index::range(2);
        m.extend(&Index::int64(vec![5, 6])).unwrap();
        assert_eq!(m, Index::int64(vec![0, 1, 5, 6]));

        // mixing string with numeric is an error, either way
        assert!(str_index(&["x"]).extend(&Index::range(1)).is_err());
        assert!(Index::range(1).extend(&str_index(&["x"])).is_err());
    }

    #[test]
    fn name_set_and_propagates_through_ops() {
        let ix = Index::datetime(vec![1, 2, 3], Tz::Utc).with_name(Some("date".into()));
        assert_eq!(ix.name(), Some("date"));
        // an unnamed index reports None
        assert_eq!(Index::range(3).name(), None);
        // the name rides through identity-preserving ops
        assert_eq!(ix.slice(0, 2).name(), Some("date"));
        assert_eq!(ix.take(&[2, 0]).name(), Some("date"));
        assert_eq!(ix.clone().with_tz(Tz::Offset(28800)).name(), Some("date"));
        // append / extend keep the left (growing) index's name
        assert_eq!(
            ix.append(&Index::datetime(vec![4], Tz::Utc)).unwrap().name(),
            Some("date")
        );
        let mut g = ix.clone();
        g.extend(&Index::datetime(vec![4], Tz::Utc)).unwrap();
        assert_eq!(g.name(), Some("date"));
        // with_name(None) clears it
        assert_eq!(ix.with_name(None).name(), None);
    }

    #[test]
    fn label_accessors_and_numeric_label_at() {
        // accessors return None on the other variant
        assert_eq!(Label::I64(5).as_i64(), Some(5));
        assert_eq!(Label::I64(5).as_str(), None);
        assert_eq!(Label::Str("x".into()).as_str(), Some("x"));
        assert_eq!(Label::Str("x".into()).as_i64(), None);
        // label_at over the numeric index kinds
        assert_eq!(Index::range(3).label_at(2), Label::I64(2));
        assert_eq!(Index::int64(vec![10, 20]).label_at(1), Label::I64(20));
        assert_eq!(
            Index::datetime(vec![100, 200], Tz::Utc).label_at(0),
            Label::I64(100)
        );
    }

    #[test]
    fn index_kind_branch_coverage() {
        // tz() / with_tz() are no-ops on a non-datetime index.
        assert_eq!(Index::range(3).tz(), Tz::Naive);
        assert!(matches!(
            Index::range(3).with_tz(Tz::Utc).kind,
            IndexKind::Range(3)
        ));
        // slice over the non-range kinds.
        assert_eq!(
            Index::int64(vec![1, 2, 3]).slice(0, 2),
            Index::int64(vec![1, 2])
        );
        assert!(matches!(
            Index::datetime(vec![1, 2], Tz::Utc).slice(0, 1).kind,
            IndexKind::Datetime(_, _)
        ));
        assert_eq!(
            Index::str(vec!["a".into(), "b".into()]).slice(1, 2),
            Index::str(vec!["b".into()])
        );
        // argsort lexicographically over a string index.
        assert_eq!(
            Index::str(vec!["b".into(), "a".into()]).argsort(true),
            vec![1, 0]
        );
        // to_column over every kind.
        assert_eq!(Index::range(2).to_column().len(), 2);
        assert_eq!(Index::datetime(vec![5], Tz::Utc).to_column().len(), 1);
        assert_eq!(Index::str(vec!["x".into()]).to_column().len(), 1);
        // append: same-kind datetime / string, and a numeric mix -> Int64.
        assert!(matches!(
            Index::datetime(vec![1], Tz::Utc)
                .append(&Index::datetime(vec![2], Tz::Utc))
                .unwrap()
                .kind,
            IndexKind::Datetime(_, _)
        ));
        assert!(matches!(
            Index::str(vec!["a".into()])
                .append(&Index::str(vec!["b".into()]))
                .unwrap()
                .kind,
            IndexKind::Str(_)
        ));
        assert!(matches!(
            Index::range(2).append(&Index::range(3)).unwrap().kind,
            IndexKind::Range(5)
        ));
        assert!(matches!(
            Index::range(2).append(&Index::int64(vec![5])).unwrap().kind,
            IndexKind::Int64(_)
        ));
        // mixing a string index with a numeric one is an error.
        assert!(Index::str(vec!["a".into()])
            .append(&Index::range(1))
            .is_err());
    }

    #[test]
    fn label_eq_value_semantics() {
        // a RangeIndex equals the same integer labels materialized as Int64
        assert!(Index::range(3).label_eq(&Index::int64(vec![0, 1, 2])));
        assert!(!Index::range(3).label_eq(&Index::int64(vec![0, 1, 9])));
        // datetime and string indexes compare by value
        assert!(Index::datetime(vec![1, 2], Tz::Utc).label_eq(&Index::datetime(vec![1, 2], Tz::Utc)));
        assert!(!Index::datetime(vec![1, 2], Tz::Utc).label_eq(&Index::datetime(vec![1, 9], Tz::Utc)));
        assert!(Index::str(vec!["a".into()]).label_eq(&Index::str(vec!["a".into()])));
        assert!(!Index::str(vec!["a".into()]).label_eq(&Index::str(vec!["b".into()])));
        // different label kinds are never equal
        assert!(!Index::range(2).label_eq(&Index::datetime(vec![0, 1], Tz::Utc)));
        assert!(!Index::str(vec!["a".into()]).label_eq(&Index::int64(vec![0])));
    }
}
