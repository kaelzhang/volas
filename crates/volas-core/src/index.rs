//! Index: the row labels shared by a frame and the series drawn from it.

use crate::column::Column;
use crate::error::{Result, VolasError};

/// Row labels. Defaults to an implicit `0..n` range; a `Datetime` index is the
/// common OHLCV case (i64 nanoseconds since the Unix epoch); a `Str` index
/// (pandas object/string index) supports symbol-keyed lookup.
#[derive(Clone, Debug, PartialEq)]
pub enum Index {
    /// Implicit `0..n` integer labels.
    Range(usize),
    /// Explicit integer labels.
    Int64(Vec<i64>),
    /// Datetime labels as i64 nanoseconds since the Unix epoch.
    Datetime(Vec<i64>),
    /// String labels (pandas object/string index).
    Str(Vec<String>),
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
    /// Build an index from a column (for `set_index`): a `Datetime` column
    /// becomes a `DatetimeIndex`, an `I64` column an `Int64Index`, a `Str`
    /// column a string index. Float / bool columns are not valid labels.
    pub fn from_column(col: &Column) -> Result<Index> {
        match col {
            Column::Datetime(v) => Ok(Index::Datetime(v.to_vec())),
            Column::I64(v) => Ok(Index::Int64(v.to_vec())),
            Column::Str(v) => Ok(Index::Str(v.to_vec())),
            other => Err(VolasError::DType(format!(
                "cannot use a {} column as an index (only datetime / int64 / string)",
                other.dtype()
            ))),
        }
    }

    /// Number of labels.
    pub fn len(&self) -> usize {
        match self {
            Index::Range(n) => *n,
            Index::Int64(v) => v.len(),
            Index::Datetime(v) => v.len(),
            Index::Str(v) => v.len(),
        }
    }

    /// The label at position `i` (for membership tests like `drop`).
    pub fn label_at(&self, i: usize) -> Label {
        match self {
            Index::Range(_) => Label::I64(i as i64),
            Index::Int64(v) | Index::Datetime(v) => Label::I64(v[i]),
            Index::Str(v) => Label::Str(v[i].clone()),
        }
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Materialize the numeric labels as `i64`. Numeric indexes only — string
    /// indexes are handled by their own paths (`label_slice` / `append` guard).
    pub fn to_i64_labels(&self) -> Vec<i64> {
        match self {
            Index::Range(n) => (0..*n as i64).collect(),
            Index::Int64(v) => v.clone(),
            Index::Datetime(v) => v.clone(),
            Index::Str(_) => unreachable!("string indexes have no i64 labels"),
        }
    }

    /// A `[start, end)` slice.
    pub fn slice(&self, start: usize, end: usize) -> Index {
        match self {
            Index::Range(_) => Index::Range(end.saturating_sub(start)),
            Index::Int64(v) => Index::Int64(v[start..end].to_vec()),
            Index::Datetime(v) => Index::Datetime(v[start..end].to_vec()),
            Index::Str(v) => Index::Str(v[start..end].to_vec()),
        }
    }

    /// Gather the given positions.
    pub fn take(&self, idx: &[usize]) -> Index {
        match self {
            Index::Range(_) => Index::Int64(idx.iter().map(|&i| i as i64).collect()),
            Index::Int64(v) => Index::Int64(idx.iter().map(|&i| v[i]).collect()),
            Index::Datetime(v) => Index::Datetime(idx.iter().map(|&i| v[i]).collect()),
            Index::Str(v) => Index::Str(idx.iter().map(|&i| v[i].clone()).collect()),
        }
    }

    /// Concatenate two indexes (extending labels). Same-kind indexes preserve
    /// their kind; mixing numeric kinds yields `Int64`; mixing a string index
    /// with a numeric one is an error.
    pub fn append(&self, other: &Index) -> Result<Index> {
        use Index::*;
        Ok(match (self, other) {
            (Range(a), Range(b)) => Range(a + b),
            (Datetime(a), Datetime(b)) => Datetime([a.as_slice(), b].concat()),
            (Str(a), Str(b)) => Str([a.as_slice(), b].concat()),
            (Str(_), _) | (_, Str(_)) => {
                return Err(VolasError::Shape(
                    "cannot append a string index to a non-string index".into(),
                ))
            }
            // remaining: numeric mixes (Range / Int64 / Datetime) -> Int64 labels
            (a, b) => Int64([a.to_i64_labels(), b.to_i64_labels()].concat()),
        })
    }

    /// Extend in place by the labels of `other` — the amortized-O(1) counterpart
    /// of [`append`](Self::append), used by the live single-bar hot path. Same-
    /// kind indexes grow their buffer; a numeric-kind mix collapses to `Int64`;
    /// mixing a string index with a numeric one is an error.
    pub fn extend(&mut self, other: &Index) -> Result<()> {
        use Index::*;
        match (&mut *self, other) {
            (Range(a), Range(b)) => *a += b,
            (Datetime(a), Datetime(b)) => a.extend_from_slice(b),
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
        match (self, label) {
            (Index::Range(n), Label::I64(v)) => {
                if *v >= 0 && (*v as usize) < *n {
                    Some(*v as usize)
                } else {
                    None
                }
            }
            (Index::Int64(vs) | Index::Datetime(vs), Label::I64(v)) => {
                vs.iter().position(|x| x == v)
            }
            (Index::Str(vs), Label::Str(s)) => vs.iter().position(|x| x == s),
            _ => None,
        }
    }

    /// `[start, end)` positions covering the inclusive label range `[lo, hi]`
    /// (ascending labels; pandas `.loc` slice semantics). Either bound may be
    /// `None` for open-ended. Numeric indexes compare numerically; a string
    /// index compares lexicographically.
    pub fn label_slice(&self, lo: Option<&Label>, hi: Option<&Label>) -> (usize, usize) {
        match self {
            Index::Str(labels) => {
                let start = lo.and_then(Label::as_str).map_or(0, |lo| {
                    labels.iter().position(|x| x.as_str() >= lo).unwrap_or(labels.len())
                });
                let end = hi.and_then(Label::as_str).map_or(labels.len(), |hi| {
                    labels.iter().rposition(|x| x.as_str() <= hi).map_or(0, |p| p + 1)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_datetime_and_int_columns() {
        assert_eq!(
            Index::from_column(&Column::datetime(vec![5, 6])).unwrap(),
            Index::Datetime(vec![5, 6])
        );
        assert_eq!(
            Index::from_column(&Column::i64(vec![1, 2])).unwrap(),
            Index::Int64(vec![1, 2])
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
        assert!(Index::Range(0).is_empty());
        assert!(!Index::Range(3).is_empty());

        assert_eq!(Index::Range(3).to_i64_labels(), vec![0, 1, 2]);
        assert_eq!(Index::Int64(vec![5, 6]).to_i64_labels(), vec![5, 6]);
        assert_eq!(Index::Datetime(vec![10, 20]).to_i64_labels(), vec![10, 20]);

        let i64 = Label::I64;
        assert_eq!(Index::Range(5).position_of(&i64(3)), Some(3));
        assert_eq!(Index::Range(5).position_of(&i64(9)), None);
        assert_eq!(Index::Range(5).position_of(&i64(-1)), None);
        assert_eq!(Index::Int64(vec![10, 20, 30]).position_of(&i64(20)), Some(1));
        assert_eq!(Index::Int64(vec![10, 20]).position_of(&i64(99)), None);
        assert_eq!(Index::Datetime(vec![100, 200]).position_of(&i64(200)), Some(1));

        // take() on an Int64 index gathers the labels at those positions
        assert_eq!(
            Index::Int64(vec![10, 20, 30]).take(&[2, 0]),
            Index::Int64(vec![30, 10])
        );
    }

    fn str_index(labels: &[&str]) -> Index {
        Index::Str(labels.iter().map(|s| s.to_string()).collect())
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
        assert!(a.append(&Index::Range(2)).is_err());
        assert!(Index::Range(2).append(&a).is_err());
    }

    #[test]
    fn extend_grows_in_place_per_kind() {
        // same-kind grows the buffer in place (the live append hot path)
        let mut r = Index::Range(3);
        r.extend(&Index::Range(2)).unwrap();
        assert_eq!(r, Index::Range(5));

        let mut d = Index::Datetime(vec![1, 2]);
        d.extend(&Index::Datetime(vec![3])).unwrap();
        assert_eq!(d, Index::Datetime(vec![1, 2, 3]));

        let mut s = str_index(&["a", "b"]);
        s.extend(&str_index(&["c"])).unwrap();
        assert_eq!(s, str_index(&["a", "b", "c"]));

        // a numeric-kind mix collapses to Int64 (matches `append`)
        let mut m = Index::Range(2);
        m.extend(&Index::Int64(vec![5, 6])).unwrap();
        assert_eq!(m, Index::Int64(vec![0, 1, 5, 6]));

        // mixing string with numeric is an error, either way
        assert!(str_index(&["x"]).extend(&Index::Range(1)).is_err());
        assert!(Index::Range(1).extend(&str_index(&["x"])).is_err());
    }

    #[test]
    fn label_accessors_and_numeric_label_at() {
        // accessors return None on the other variant
        assert_eq!(Label::I64(5).as_i64(), Some(5));
        assert_eq!(Label::I64(5).as_str(), None);
        assert_eq!(Label::Str("x".into()).as_str(), Some("x"));
        assert_eq!(Label::Str("x".into()).as_i64(), None);
        // label_at over the numeric index kinds
        assert_eq!(Index::Range(3).label_at(2), Label::I64(2));
        assert_eq!(Index::Int64(vec![10, 20]).label_at(1), Label::I64(20));
        assert_eq!(Index::Datetime(vec![100, 200]).label_at(0), Label::I64(100));
    }
}
