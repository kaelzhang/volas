//! Typed aggregators for cumulation (replaces stock-pandas's Python callables).

use std::collections::HashMap;

use volas_core::{Column, CombineOp, DataFrame, Result, VolasError};

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

    /// The in-place single-cell combine op equivalent to this aggregator — the
    /// incremental dual of [`Self::reduce`], used by the live tf-fold to update the
    /// forming bar without re-reducing the whole period.
    pub fn as_combine_op(&self) -> CombineOp {
        match self {
            Agg::First => CombineOp::Keep,
            Agg::Last => CombineOp::Replace,
            Agg::Max => CombineOp::Max,
            Agg::Min => CombineOp::Min,
            Agg::Sum => CombineOp::Sum,
        }
    }

    /// Reduce the values at `kept` positions of `col` to a 1-element column of
    /// `col`'s dtype in one pass ([`Column::reduce_selection`]: the steps the live
    /// time-frame fold runs, dispatched once per column), so the batch and the
    /// incremental aggregate agree exactly — the support table is
    /// [`CombineOp::supports`], a missing input makes any aggregate missing, an
    /// integer sum is checked.
    pub fn reduce(&self, col: &Column, kept: &[usize]) -> Result<Column> {
        col.reduce_selection(self.as_combine_op(), kept)
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

    /// This spec for columns renamed by `mapping` (old name -> new name): each renamed
    /// column keeps the aggregator it had under its old name.
    pub fn renamed(&self, mapping: &HashMap<String, String>) -> AggSpec {
        let mut spec = self.clone();
        for (old, new) in mapping {
            spec.set(new.clone(), self.agg_for(old));
        }
        spec
    }

    /// The aggregator for a column (the default if unset).
    pub fn agg_for(&self, name: &str) -> Agg {
        self.by_name.get(name).copied().unwrap_or(self.default)
    }

    /// Whether both specs assign the same effective aggregator to every given
    /// data column. Redundant overrides and absent OHLCV columns do not matter.
    pub fn equivalent_for(&self, other: &Self, names: &[String]) -> bool {
        names.iter().all(|name| self.agg_for(name) == other.agg_for(name))
    }

    /// Check every column of `df` against the support table
    /// ([`CombineOp::supports`]) before any period is aggregated — a frame whose
    /// spec asks for, say, the `max` of a `str` column is rejected up front, by
    /// `cumulate` and by a `time_frame` frame's construction alike.
    pub fn check(&self, df: &DataFrame) -> Result<()> {
        for (name, col) in df.names().iter().zip(df.columns()) {
            let op = self.agg_for(name).as_combine_op();
            if !op.supports(col.dtype()) {
                return Err(VolasError::DType(format!(
                    "cumulators: cannot {} the {} column {name:?}",
                    op.name(),
                    col.dtype()
                )));
            }
        }
        Ok(())
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
        assert_eq!(
            Agg::First.reduce(&col, &kept).unwrap(),
            Column::f64(vec![3.0])
        );
        assert_eq!(
            Agg::Last.reduce(&col, &kept).unwrap(),
            Column::f64(vec![1.0])
        );
        assert_eq!(
            Agg::Max.reduce(&col, &kept).unwrap(),
            Column::f64(vec![4.0])
        );
        assert_eq!(
            Agg::Min.reduce(&col, &kept).unwrap(),
            Column::f64(vec![1.0])
        );
        assert_eq!(
            Agg::Sum.reduce(&col, &kept).unwrap(),
            Column::f64(vec![9.0])
        );
    }

    #[test]
    fn an_integer_sum_never_wraps() {
        let col = Column::i64(vec![i64::MAX, 1]);
        let err = Agg::Sum.reduce(&col, &[0, 1]).unwrap_err();
        assert!(matches!(err, VolasError::Overflow(_)), "{err}");
        assert_eq!(Agg::Sum.reduce(&col, &[0]).unwrap(), Column::i64(vec![i64::MAX]));
    }

    #[test]
    fn as_combine_op_maps_each_aggregator() {
        assert_eq!(Agg::First.as_combine_op(), CombineOp::Keep);
        assert_eq!(Agg::Last.as_combine_op(), CombineOp::Replace);
        assert_eq!(Agg::Max.as_combine_op(), CombineOp::Max);
        assert_eq!(Agg::Min.as_combine_op(), CombineOp::Min);
        assert_eq!(Agg::Sum.as_combine_op(), CombineOp::Sum);
    }

    #[test]
    fn sum_on_string_errors() {
        let col = Column::str(vec!["a".into(), "b".into()]);
        assert!(Agg::Sum.reduce(&col, &[0, 1]).is_err());
        // first/last still work on strings
        assert_eq!(
            Agg::First.reduce(&col, &[0, 1]).unwrap(),
            Column::str(vec!["a".into()])
        );
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
        assert_eq!(
            Agg::Max.reduce(&ints, &[0, 1, 2, 3]).unwrap(),
            Column::i64(vec![4])
        );
        assert_eq!(
            Agg::Min.reduce(&ints, &[0, 1, 2, 3]).unwrap(),
            Column::i64(vec![1])
        );
        assert_eq!(
            Agg::Sum.reduce(&ints, &[0, 1, 2, 3]).unwrap(),
            Column::i64(vec![9])
        );

        // Datetime max/min stay in the datetime dtype.
        let ts = Column::datetime(vec![100, 500, 200]);
        assert_eq!(
            Agg::Max.reduce(&ts, &[0, 1, 2]).unwrap(),
            Column::datetime(vec![500])
        );
        assert_eq!(
            Agg::Min.reduce(&ts, &[0, 1, 2]).unwrap(),
            Column::datetime(vec![100])
        );

        // Sum on a bool column is a dtype error (hits the fallback arm).
        let flags = Column::bool(vec![true, false]);
        assert!(Agg::Sum.reduce(&flags, &[0, 1]).is_err());
    }

    #[test]
    fn a_missing_input_makes_every_aggregate_missing() {
        use volas_core::Validity;
        let all = [0, 1, 2];
        // float: a NaN anywhere -> NaN (never skipped, never ±inf), first / last too
        let f = Column::f64(vec![1.0, f64::NAN, 3.0]);
        for agg in [Agg::First, Agg::Last, Agg::Max, Agg::Min, Agg::Sum] {
            assert!(agg.reduce(&f, &all).unwrap().get_f64(0).is_nan(), "{agg:?}");
        }
        assert_eq!(Agg::First.reduce(&f, &[0, 2]).unwrap(), Column::f64(vec![1.0]));
        assert_eq!(Agg::Last.reduce(&f, &[0, 2]).unwrap(), Column::f64(vec![3.0]));
        let s = Column::str_from_options([Some("a".into()), None]);
        assert!(!Agg::First.reduce(&s, &[0, 1]).unwrap().is_valid(0));
        let nan = Column::f64(vec![f64::NAN]);
        assert!(Agg::Max.reduce(&nan, &[0]).unwrap().get_f64(0).is_nan(), "all-missing");
        // int / f32 / i32: the same rule, payload-independent, dtype kept
        let ints = Column::i64_with(vec![-5, 999, 2], Validity::from_valid_iter(3, [true, false, true])).unwrap();
        for agg in [Agg::Max, Agg::Min, Agg::Sum] {
            let out = agg.reduce(&ints, &all).unwrap();
            assert_eq!((out.dtype(), out.is_valid(0)), (volas_core::DType::I64, false), "{agg:?}");
        }
        assert_eq!(Agg::Max.reduce(&ints, &[0, 2]).unwrap(), Column::i64(vec![2]));
        assert_eq!(Agg::Sum.reduce(&Column::f32(vec![1.5, 2.0]), &[0, 1]).unwrap(), Column::f32(vec![3.5]));
        assert_eq!(Agg::Min.reduce(&Column::i32(vec![4, -2]), &[0, 1]).unwrap(), Column::i32(vec![-2]));
        // the support table: a datetime has no sum
        let ts = Column::datetime(vec![100, 500]);
        assert!(Agg::Sum.reduce(&ts, &[0, 1]).is_err());
    }

    #[test]
    fn a_spec_is_checked_against_the_frame_up_front() {
        let df = DataFrame::new(
            vec!["close".into(), "sym".into()],
            vec![Column::f64(vec![1.0]), Column::str(vec!["a".into()])],
            None,
        )
        .unwrap();
        let mut spec = AggSpec::ohlcv();
        spec.check(&df).unwrap(); // `sym` falls back to `last`
        spec.set("sym", Agg::Max);
        assert_eq!(
            spec.check(&df).unwrap_err(),
            VolasError::DType("cumulators: cannot max the str column \"sym\"".into())
        );
    }

    #[test]
    fn the_batch_reduce_is_the_fold() {
        use volas_core::Validity;
        let cols = [
            Column::f64(vec![2.0, f64::NAN, 7.0, 1.0]),
            Column::f64(vec![2.0, 9.0, 7.0, 1.0]),
            Column::f32(vec![2.0, 9.0, 7.0, 1.0]),
            Column::i64_with(vec![2, 0, 7, 1], Validity::from_valid_iter(4, [true, false, true, true])).unwrap(),
            Column::i32(vec![2, 9, 7, 1]),
            Column::datetime(vec![2, 9, i64::MIN, 1]),
        ];
        for col in &cols {
            for agg in [Agg::First, Agg::Last, Agg::Max, Agg::Min, Agg::Sum] {
                let (op, batch) = (agg.as_combine_op(), agg.reduce(col, &[0, 1, 2, 3]));
                let mut fold = col.take(&[0]);
                let folded = (1..4).try_for_each(|r| fold.check_fold_cell(0, op, col, r, false).and_then(|write| write.land()));
                match batch {
                    Ok(b) => assert!(folded.is_ok() && b.equals(&fold), "{agg:?} {}", col.dtype()),
                    Err(_) => assert!(folded.is_err(), "{agg:?} {}", col.dtype()),
                }
            }
        }
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
