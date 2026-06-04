//! Typed aggregators for cumulation (replaces stock-pandas's Python callables).

use std::collections::HashMap;

use volas_core::{Column, Result, VolasError};

/// A column aggregator over a period's rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Agg {
    /// First value.
    First,
    /// Maximum (numeric / datetime).
    Max,
    /// Minimum (numeric / datetime).
    Min,
    /// Last value.
    Last,
    /// Sum (numeric).
    Sum,
}

impl Agg {
    /// Parse an aggregator name (`first` / `max` / `min` / `last` / `sum`).
    pub fn from_name(s: &str) -> Result<Agg> {
        Ok(match s {
            "first" => Agg::First,
            "max" => Agg::Max,
            "min" => Agg::Min,
            "last" => Agg::Last,
            "sum" => Agg::Sum,
            _ => return Err(VolasError::Value(format!("unknown aggregator '{s}'"))),
        })
    }

    fn name(&self) -> &'static str {
        match self {
            Agg::First => "first",
            Agg::Max => "max",
            Agg::Min => "min",
            Agg::Last => "last",
            Agg::Sum => "sum",
        }
    }

    /// Reduce the values at `kept` positions of `col` to a 1-element column of
    /// the appropriate dtype.
    pub fn reduce(&self, col: &Column, kept: &[usize]) -> Result<Column> {
        match self {
            Agg::First => Ok(col.take(&[kept[0]])),
            Agg::Last => Ok(col.take(&[*kept.last().unwrap()])),
            Agg::Max | Agg::Min | Agg::Sum => self.reduce_numeric(col, kept),
        }
    }

    fn reduce_numeric(&self, col: &Column, kept: &[usize]) -> Result<Column> {
        match col {
            Column::F64(v) => {
                let it = kept.iter().map(|&i| v[i]);
                let val = match self {
                    Agg::Max => it.fold(f64::NEG_INFINITY, f64::max),
                    Agg::Min => it.fold(f64::INFINITY, f64::min),
                    Agg::Sum => it.sum(),
                    _ => unreachable!(),
                };
                Ok(Column::f64(vec![val]))
            }
            Column::I64(v) | Column::Datetime(v) => {
                let it = kept.iter().map(|&i| v[i]);
                let val = match self {
                    Agg::Max => it.max().unwrap(),
                    Agg::Min => it.min().unwrap(),
                    Agg::Sum => it.sum(),
                    _ => unreachable!(),
                };
                Ok(if matches!(col, Column::Datetime(_)) {
                    Column::datetime(vec![val])
                } else {
                    Column::i64(vec![val])
                })
            }
            other => Err(VolasError::DType(format!(
                "cannot {} a {} column",
                self.name(),
                other.dtype()
            ))),
        }
    }
}

/// Per-column aggregator assignment, with a default for unlisted columns.
#[derive(Clone, Debug)]
pub struct AggSpec {
    default: Agg,
    by_name: HashMap<String, Agg>,
}

impl AggSpec {
    /// The OHLCV defaults: open=first, high=max, low=min, close=last, volume=sum;
    /// every other column falls back to `last`.
    pub fn ohlcv() -> Self {
        let mut by_name = HashMap::new();
        for (name, agg) in [
            ("open", Agg::First),
            ("high", Agg::Max),
            ("low", Agg::Min),
            ("close", Agg::Last),
            ("volume", Agg::Sum),
        ] {
            by_name.insert(name.to_string(), agg);
        }
        AggSpec {
            default: Agg::Last,
            by_name,
        }
    }

    /// Override (or add) a column's aggregator.
    pub fn set(&mut self, name: impl Into<String>, agg: Agg) {
        self.by_name.insert(name.into(), agg);
    }

    /// The aggregator for a column (the default if unset).
    pub fn agg_for(&self, name: &str) -> Agg {
        self.by_name.get(name).copied().unwrap_or(self.default)
    }
}

impl Default for AggSpec {
    fn default() -> Self {
        AggSpec::ohlcv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_defaults_and_overrides() {
        let mut spec = AggSpec::ohlcv();
        assert_eq!(spec.agg_for("open"), Agg::First);
        assert_eq!(spec.agg_for("volume"), Agg::Sum);
        assert_eq!(spec.agg_for("anything"), Agg::Last);
        spec.set("volume", Agg::Last);
        assert_eq!(spec.agg_for("volume"), Agg::Last);
    }

    #[test]
    fn reduce_numeric_and_first_last() {
        let col = Column::f64(vec![3.0, 1.0, 4.0, 1.0]);
        let kept = [0, 1, 2, 3];
        assert_eq!(Agg::First.reduce(&col, &kept).unwrap(), Column::f64(vec![3.0]));
        assert_eq!(Agg::Last.reduce(&col, &kept).unwrap(), Column::f64(vec![1.0]));
        assert_eq!(Agg::Max.reduce(&col, &kept).unwrap(), Column::f64(vec![4.0]));
        assert_eq!(Agg::Min.reduce(&col, &kept).unwrap(), Column::f64(vec![1.0]));
        assert_eq!(Agg::Sum.reduce(&col, &kept).unwrap(), Column::f64(vec![9.0]));
    }

    #[test]
    fn sum_on_string_errors() {
        let col = Column::str(vec!["a".into(), "b".into()]);
        assert!(Agg::Sum.reduce(&col, &[0, 1]).is_err());
        // first/last still work on strings
        assert_eq!(Agg::First.reduce(&col, &[0, 1]).unwrap(), Column::str(vec!["a".into()]));
    }

    #[test]
    fn from_name_parses_all_and_rejects_unknown() {
        assert_eq!(Agg::from_name("first").unwrap(), Agg::First);
        assert_eq!(Agg::from_name("max").unwrap(), Agg::Max);
        assert_eq!(Agg::from_name("min").unwrap(), Agg::Min);
        assert_eq!(Agg::from_name("last").unwrap(), Agg::Last);
        assert_eq!(Agg::from_name("sum").unwrap(), Agg::Sum);
        assert!(Agg::from_name("median").is_err());
    }

    #[test]
    fn reduce_over_i64_and_datetime_columns() {
        let ints = Column::i64(vec![3, 1, 4, 1]);
        assert_eq!(Agg::Max.reduce(&ints, &[0, 1, 2, 3]).unwrap(), Column::i64(vec![4]));
        assert_eq!(Agg::Min.reduce(&ints, &[0, 1, 2, 3]).unwrap(), Column::i64(vec![1]));
        assert_eq!(Agg::Sum.reduce(&ints, &[0, 1, 2, 3]).unwrap(), Column::i64(vec![9]));

        // Datetime max/min stay in the datetime dtype.
        let ts = Column::datetime(vec![100, 500, 200]);
        assert_eq!(Agg::Max.reduce(&ts, &[0, 1, 2]).unwrap(), Column::datetime(vec![500]));
        assert_eq!(Agg::Min.reduce(&ts, &[0, 1, 2]).unwrap(), Column::datetime(vec![100]));

        // Sum on a bool column is a dtype error (hits the fallback arm).
        let flags = Column::bool(vec![true, false]);
        assert!(Agg::Sum.reduce(&flags, &[0, 1]).is_err());
    }

    #[test]
    fn default_spec_and_max_min_dtype_errors() {
        assert_eq!(AggSpec::default().agg_for("open"), Agg::First);
        // Max / Min on a non-numeric column hit the dtype-error path (exercises name()).
        let flags = Column::bool(vec![true, false]);
        assert!(Agg::Max.reduce(&flags, &[0, 1]).is_err());
        assert!(Agg::Min.reduce(&flags, &[0, 1]).is_err());
    }
}
