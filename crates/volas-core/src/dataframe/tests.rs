    use super::*;
    use super::derive::NoReplay;
    use crate::index::IndexKind;
    use crate::CombineOp;
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
    fn duplicate_names_report_the_first_repeated_encounter() {
        for (names, repeated) in [(["b", "a", "a", "b"], "a"), (["a", "b", "b", "a"], "b")] {
            let error = DataFrame::new(names.map(str::to_owned).to_vec(), vec![Column::f64(vec![1.0]); 4], None).unwrap_err();
            let VolasError::Value(message) = error else { panic!("duplicate name must be a value error") };
            assert_eq!(message, format!("duplicate column name \"{repeated}\""));
        }
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
        let s = df.rows(1, 3, &NoReplay);
        assert_eq!(s.height(), 2);
        assert_eq!(s.column("a").unwrap().as_f64().unwrap(), &[2.0, 3.0]);

        let f = df.filter_mask(&[true, false, true], &NoReplay).unwrap();
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
    fn names_arc_is_pointer_stable_across_row_appends() {
        let mut df = sample();
        let before = std::sync::Arc::as_ptr(df.names_arc());
        assert_eq!(df.names_arc().as_slice(), df.names());
        // A row-only append never rebuilds the name vector, so a caller can validate
        // "schema unchanged" with an O(1) pointer check (the live-fold fast path).
        df.append(&sample()).unwrap();
        assert!(std::ptr::eq(before, std::sync::Arc::as_ptr(df.names_arc())));
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
            &Index::new(None, crate::IndexKind::Int64(vec![100, 200])).unwrap().with_name(Some("t".into()))
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
                Column::i64_masked(vec![1, 0, 3], crate::Validity::from_valid_iter(3, [true, false, true])),
                Column::bool_masked(vec![true, false, false], crate::Validity::from_valid_iter(3, [true, false, true])),
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
    fn row_major_i64_export_is_exact() {
        // The integer export takes each column's raw value (no f64 round-trip): a
        // datetime keeps exact epoch-ns with NaT = i64::MIN, a large i64 survives
        // past 2^53, i32 / bool widen, a float truncates toward zero, and a str
        // (rejected by the caller before this) contributes its 0 placeholder.
        let big = (1i64 << 60) + 1;
        let df = DataFrame::new(
            vec!["t".into(), "n".into(), "i32".into(), "b".into(), "f".into(), "f32".into(), "s".into()],
            vec![
                Column::datetime(vec![123, i64::MIN]),
                Column::i64(vec![big, -7]),
                Column::i32(vec![5, -6]),
                Column::bool(vec![true, false]),
                Column::f64(vec![2.9, -2.9]),
                Column::f32(vec![1.9, -1.9]),
                Column::str(vec!["a".into(), "b".into()]),
            ],
            None,
        )
        .unwrap();
        let (data, h, w) = df.to_row_major_i64();
        assert_eq!((h, w), (2, 7));
        // row 0: datetime 123 (exact), i64 2^60+1 (exact), i32 5, bool->1, f64 2.9->2, f32 1.9->1, str->0
        assert_eq!(&data[0..7], &[123, big, 5, 1, 2, 1, 0]);
        // row 1: NaT stays i64::MIN, -7, i32 -6, bool->0, -2.9->-2, -1.9->-1, str->0
        assert_eq!(&data[7..14], &[i64::MIN, -7, -6, 0, -2, -1, 0]);
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
            Some(Index::new(None, crate::IndexKind::Range { start: 0, len: 3 }).unwrap()),
        )
        .is_err());
    }

    #[test]
    fn series_extracts_a_named_column() {
        let df = sample();
        let s = df.series("a").unwrap();
        assert_eq!(s.name(), Some("a"));
        assert_eq!(s.data().as_f64().unwrap(), &[1.0, 2.0, 3.0]);
        assert!(Arc::ptr_eq(s.index(), df.index()));
        assert!(df.series("missing").is_err());
    }

    #[test]
    fn set_column_add_replace_and_errors() {
        // adding the first column to an empty frame seeds the height + index
        let mut empty = DataFrame::new(vec![], vec![], None).unwrap();
        empty.set_column("x", Column::f64(vec![1.0, 2.0])).unwrap();
        assert_eq!(empty.height(), 2);
        assert_eq!(empty.index().as_ref(), &Index::new(None, crate::IndexKind::Range { start: 0, len: 2 }).unwrap());

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
        assert!(sample().filter_mask(&[true, false], &NoReplay).is_err());
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
        h.set_computed("sig", std::sync::Arc::new(()), Reach { lookback: 0, resume: 0, continuation: 0, history: History::Window });
        let only_a3 = DataFrame::new(vec!["a".into()], vec![Column::f64(vec![2.0])], None).unwrap();
        h.append(&only_a3).unwrap();
        assert_eq!(h.column("sig").unwrap().as_bool().unwrap(), &[true, false]);
    }

    #[test]
    fn computed_tail_update_and_dtype_guard() {
        let mut df = sample();
        df.set_computed("a", std::sync::Arc::new(()), Reach { lookback: 1, resume: 0, continuation: 1, history: History::Window });
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
    fn slice_keeps_computed_with_history_before_short_selections() {
        // a frame with a cached recursive directive of lookback 11 (ema:12)
        let mut df = DataFrame::new(
            vec!["close".into()],
            vec![Column::f64((0..60).map(|i| i as f64).collect())],
            None,
        )
        .unwrap();
        df.set_computed("close", std::sync::Arc::new(()), Reach { lookback: 11, resume: 0, continuation: 11, history: History::Window });
        // A carried EMA state as of row 59.
        df.set_computed_states("close", Some(vec![42.0]), None);
        // A slice keeping >= lookback rows AND ending at `valid_rows` carries the column
        // as continuable, threading the recursive state through (the `Some` branch).
        let keep = df.rows(40, 60, &NoReplay); // 20 rows >= 11, end == valid_rows (60)
        assert_eq!(keep.computed_columns().len(), 1);
        assert_eq!(keep.computed_columns()[0].1.valid_rows, 20);
        assert_eq!(keep.computed_columns()[0].1.state.as_ref().map(|words| words.to_vec()), Some(vec![42.0]));
        // a TAIL slice keeps >= lookback rows but ends BEFORE `valid_rows`, so the carried
        // state (attached to a row this sub-frame no longer ends on) is dropped — the column
        // stays computed but state-less, continuable only via the full-recompute fallback.
        let tail = df.rows(0, 50, &NoReplay); // 50 rows >= 11, end (50) < valid_rows (60)
        assert_eq!(tail.computed_columns().len(), 1);
        assert_eq!(tail.computed_columns()[0].1.state.as_ref().map(|words| words.to_vec()), None);
        // A short selection keeps its cached identity and the lead its recompute reads.
        let too_short = df.rows(55, 60, &NoReplay); // 5 rows < 11
        assert_eq!(too_short.computed_columns().len(), 1);
        assert_eq!(too_short.lead_rows(), 12);
        // A column needing one row past its lookback retains that row in its lead
        df.set_computed("close", std::sync::Arc::new(()), Reach { lookback: 11, resume: 0, continuation: 12, history: History::Window });
        assert_eq!(df.rows(49, 60, &NoReplay).computed_columns().len(), 1);
        assert_eq!(df.rows(49, 60, &NoReplay).lead_rows(), 12);
        assert_eq!(df.rows(48, 60, &NoReplay).computed_columns()[0].1.reach.continuation, 12);
    }

    fn meta(df: &DataFrame) -> ComputedMeta {
        df.computed_columns()[0].1.clone()
    }

    fn ten_rows() -> DataFrame {
        let mut df = DataFrame::new(
            vec!["close".into()],
            vec![Column::f64((0..10).map(|i| i as f64).collect())],
            None,
        )
        .unwrap();
        df.set_computed("close", std::sync::Arc::new(()), Reach { lookback: 2, resume: 0, continuation: 2, history: History::Window });
        df
    }

    /// `ten_rows` with its last row appended after the states were captured: the cached
    /// column is valid through row 8 and carries `[8.0]` (row 8) and `[7.0]` (row 7).
    fn stale_last_row() -> DataFrame {
        let mut df = DataFrame::new(
            vec!["x".into(), "close".into()],
            vec![Column::f64((0..9).map(|i| i as f64).collect()); 2],
            None,
        )
        .unwrap();
        df.set_computed("close", std::sync::Arc::new(()), Reach { lookback: 2, resume: 0, continuation: 2, history: History::Window });
        df.set_computed_states("close", Some(vec![8.0]), Some(vec![7.0]));
        df.append(&DataFrame::new(vec!["x".into()], vec![Column::f64(vec![9.0])], None).unwrap())
            .unwrap();
        assert_eq!((df.height(), meta(&df).valid_rows), (10, 9));
        df
    }

    /// A one-row step advances both states, recycling the buffers; a declined step and
    /// an untracked / stateless column leave them alone.
    #[test]
    fn a_step_advances_the_two_states() {
        let mut df = ten_rows();
        assert_eq!(df.step_computed_state("close", |_, _, _, _| Some(1.0)), None); // no state
        assert_eq!(df.step_computed_state("missing", |_, _, _, _| Some(1.0)), None);
        let mut df2 = stale_last_row();
        let got = df2.step_computed_state("close", |frame, prev, origin, next| {
            assert_eq!((frame.height(), prev, origin, next.len()), (10, &[8.0][..], 0, 1));
            next[0] = 9.0;
            Some(9.0)
        });
        assert_eq!(got, Some(9.0));
        df2.update_computed_value("close", 9, 9.0).unwrap();
        assert_eq!((meta(&df2).state.as_ref().map(|words| words.to_vec()), meta(&df2).state_before_last.as_ref().map(|words| words.to_vec())), (Some(vec![9.0]), Some(vec![8.0])));
        // declined: untouched
        assert_eq!(df2.step_computed_state("close", |_, _, _, _| None), None);
        assert_eq!((meta(&df2).state.as_ref().map(|words| words.to_vec()), meta(&df2).state_before_last.as_ref().map(|words| words.to_vec())), (Some(vec![9.0]), Some(vec![8.0])));
        // a Vec resume of one row advances the same way
        let mut df3 = stale_last_row();
        df3.update_computed_value("close", 9, 9.0).unwrap();
        df3.advance_computed_state("close", vec![9.0]);
        assert_eq!((meta(&df3).state.as_ref().map(|words| words.to_vec()), meta(&df3).state_before_last.as_ref().map(|words| words.to_vec())), (Some(vec![9.0]), Some(vec![8.0])));
        df3.advance_computed_state("missing", vec![1.0]); // no-op
        df3.set_computed_states("close", Some(vec![1.0]), None);
        assert_eq!(meta(&df3).state_before_last.as_ref().map(|words| words.to_vec()), None);
        df3.set_computed_states("missing", None, None); // no-op
        // a column with no valid row carries no state
        df3.invalidate_computed_on_write("x");
        df3.set_computed_states("close", Some(vec![1.0]), Some(vec![0.0]));
        assert_eq!((meta(&df3).state.as_ref().map(|words| words.to_vec()), meta(&df3).state_before_last.as_ref().map(|words| words.to_vec())), (None, None));
    }

    /// A state is resumed only from the row it describes: one left on a row that is no
    /// longer the last valid one is refused (it would continue the wrong history).
    #[test]
    fn a_state_of_another_row_is_never_resumed() {
        let mut df = stale_last_row();
        // the row is written without advancing the state: the state now describes row 8
        // while the last valid row is 9
        df.update_computed_value("close", 9, 9.0).unwrap();
        assert_eq!(df.computed_resume_state("close"), None);
        assert_eq!(df.step_computed_state("close", |_, _, _, _| Some(1.0)), None);
        // a later valid write with the state before it is advanced again
        df.advance_computed_state("close", vec![9.0]);
        assert_eq!(df.computed_resume_state("close"), Some((&[9.0][..], 0)));
    }

    /// Replacing the last row drops a column valid through it back by one row: the state
    /// before it becomes its state. A column already stale there keeps both; a write
    /// drops both.
    #[test]
    fn a_replaced_last_row_restores_the_state_before_it() {
        let mut df = ten_rows();
        df.set_computed_states("close", Some(vec![9.0]), Some(vec![8.0]));
        df.invalidate_computed_forming_row(9);
        assert_eq!(meta(&df).valid_rows, 9);
        assert_eq!((meta(&df).state.as_ref().map(|words| words.to_vec()), meta(&df).state_before_last.as_ref().map(|words| words.to_vec())), (Some(vec![8.0]), None));
        assert_eq!(df.computed_resume_state("close"), Some((&[8.0][..], 0)));
        df.invalidate_computed_forming_row(9); // already stale there: unchanged
        assert_eq!(meta(&df).state.as_ref().map(|words| words.to_vec()), Some(vec![8.0]));
        // the step after the replacement recycles the dropped buffer
        df.step_computed_state("close", |_, _, _, next| {
            next[0] = 9.5;
            Some(9.5)
        });
        df.update_computed_value("close", 9, 9.5).unwrap();
        assert_eq!((meta(&df).state.as_ref().map(|words| words.to_vec()), meta(&df).state_before_last.as_ref().map(|words| words.to_vec())), (Some(vec![9.5]), Some(vec![8.0])));
        // a replacement without a state before it leaves none
        df.set_computed_states("close", Some(vec![9.5]), None);
        df.invalidate_computed_forming_row(9);
        assert_eq!(meta(&df).state.as_ref().map(|words| words.to_vec()), None);
        df.set_column("x", Column::f64(vec![0.0; 10])).unwrap();
        df.set_computed("x", std::sync::Arc::new(()), Reach { lookback: 0, resume: 0, continuation: 0, history: History::Window });
        df.invalidate_computed_on_write("x");
        assert_eq!((meta(&df).state.as_ref().map(|words| words.to_vec()), meta(&df).state_before_last.as_ref().map(|words| words.to_vec())), (None, None));
    }

    /// Both states describe rows relative to the last valid one, so a slice holding it
    /// carries both (the earlier one only with its own row inside), and a slice ending
    /// before it — or starting after it — carries neither.
    #[test]
    fn a_slice_carries_both_states_together() {
        let mut df = ten_rows();
        df.set_computed_states("close", Some(vec![9.0]), Some(vec![8.0]));
        let tail = df.rows(5, 10, &NoReplay);
        assert_eq!((meta(&tail).state.as_ref().map(|words| words.to_vec()), meta(&tail).state_before_last.as_ref().map(|words| words.to_vec())), (Some(vec![9.0]), Some(vec![8.0])));
        assert_eq!(tail.computed_resume_state("close"), Some((&[9.0][..], 5)));
        let last_two = df.rows(8, 10, &NoReplay);
        assert_eq!(meta(&last_two).state_before_last.as_ref().map(|words| words.to_vec()), Some(vec![8.0]));
        let last_one = ten_rows_lookback(1);
        let last_one = { let mut d = last_one; d.set_computed_states("close", Some(vec![9.0]), Some(vec![8.0])); d.rows(9, 10, &NoReplay) };
        assert_eq!((meta(&last_one).state.as_ref().map(|words| words.to_vec()), meta(&last_one).state_before_last.as_ref().map(|words| words.to_vec())), (Some(vec![9.0]), None));
        let head = df.rows(0, 9, &NoReplay);
        assert_eq!((meta(&head).state.as_ref().map(|words| words.to_vec()), meta(&head).state_before_last.as_ref().map(|words| words.to_vec())), (None, None));
        // a stale frame (valid through row 8) sliced after its last valid row: the state
        // describes a row before the slice, so it is not carried and nothing is valid
        let mut stale = stale_last_row();
        stale.append(&DataFrame::new(vec!["x".into()], vec![Column::f64(vec![10.0])], None).unwrap())
            .unwrap();
        let after = stale.rows(9, 11, &NoReplay);
        let m = after.computed_columns().into_iter().find(|(n, _)| n == "close").unwrap().1;
        assert_eq!((m.valid_rows, m.state.as_ref().map(|words| words.to_vec()), m.state_before_last.as_ref().map(|words| words.to_vec())), (0, None, None));
        // the history behind a carried column is kept: past warm-up from the first row
        assert_eq!(df.rows(9, 10, &NoReplay).computed_columns().len(), 1);
        assert_eq!(df.rows(9, 10, &NoReplay).lead_rows(), 3);
        assert!(meta(&df.rows(8, 10, &NoReplay)).warm(8));
    }

    fn ten_rows_lookback(lookback: usize) -> DataFrame {
        let mut df = DataFrame::new(
            vec!["close".into()],
            vec![Column::f64((0..10).map(|i| i as f64).collect())],
            None,
        )
        .unwrap();
        df.set_computed("close", std::sync::Arc::new(()), Reach { lookback, resume: 0, continuation: lookback, history: History::Window });
        df
    }

    /// A computed write never truncates and never skips stale rows: rows past the frame,
    /// or a write starting after the valid rows, raise with nothing written.
    #[test]
    fn computed_writes_are_contiguous_and_inside_the_frame() {
        let mut df = stale_last_row();
        let before = df.column("close").unwrap().clone();
        assert!(df.update_computed_tail("close", 9, &Column::f64(vec![1.0, 2.0])).is_err());
        assert!(df.update_computed_value("close", 10, 1.0).is_err());
        assert!(df.update_computed_tail("close", usize::MAX, &Column::f64(vec![1.0])).is_err());
        df.invalidate_computed_on_write("x"); // valid_rows 0
        assert!(df.update_computed_value("close", 5, 1.0).is_err());
        assert!(df.column("close").unwrap().equals(&before));
        assert_eq!(meta(&df).valid_rows, 0);
        // a contiguous write moves the cursor to its end only
        df.update_computed_tail("close", 0, &Column::f64(vec![0.0; 4])).unwrap();
        assert_eq!(meta(&df).valid_rows, 4);
        // an earlier rewrite keeps the cursor
        df.update_computed_value("close", 1, 7.0).unwrap();
        assert_eq!(meta(&df).valid_rows, 4);
        // a data column has no cursor to move
        df.update_computed_tail("x", 0, &Column::f64(vec![5.0])).unwrap();
        assert_eq!(df.column("x").unwrap().get_f64(0), 5.0);
    }

    #[test]
    fn slice_data_drops_computed_carry() {
        // `slice` carries the cached-directive metadata (continuable resume); `slice_data`
        // is the read-only twin that drops it — the boundary the refresh probe relies on.
        // A `slice_data` frame must NOT be appended to (its resume is gone); `slice` is for
        // anything that continues live (window compaction).
        let mut df = DataFrame::new(
            vec!["close".into()],
            vec![Column::f64((0..60).map(|i| i as f64).collect())],
            None,
        )
        .unwrap();
        df.set_computed("close", std::sync::Arc::new(()), Reach { lookback: 11, resume: 0, continuation: 11, history: History::Window });
        df.set_computed_states("close", Some(vec![42.0]), None);
        // slice() carries it (continuable, with the recursive state) ...
        assert_eq!(df.rows(40, 60, &NoReplay).computed_columns().len(), 1);
        // ... slice_data() drops it (same row data, no resume carry) — and the underlying
        // raw values are identical, so a read-only probe sees exactly the same columns.
        let data = df.slice_data(40, 60);
        assert!(data.computed_columns().is_empty());
        assert_eq!(data.height(), 20);
        assert_eq!(data.column("close").unwrap().get_f64(0), 40.0);
    }

    #[test]
    fn write_cells_scalar_and_array() {
        let mut df = sample();
        // broadcast a scalar into two rows of the F64 column "a"
        df.write_one(0, &[0, 2], &Column::f64(vec![9.0]))
            .unwrap();
        assert_eq!(df.column("a").unwrap().as_f64().unwrap(), &[9.0, 2.0, 9.0]);
        // element-wise array into the I64 column "b" (integral -> stays i64)
        df.write_one(1, &[1, 2], &Column::f64(vec![40.0, 50.0]))
            .unwrap();
        assert_eq!(df.column("b").unwrap().as_i64().unwrap(), &[10, 40, 50]);
    }

    #[test]
    fn write_cells_fractional_into_int_errors() {
        let mut df = sample();
        // a fractional write into the I64 column is lossy and errors (no float
        // widening) — the column stays unchanged, matching the Series scalar path
        assert!(df
            .write_one(1, &[0], &Column::f64(vec![1.5]))
            .is_err());
        assert_eq!(df.column("b").unwrap().dtype(), DType::I64);
        assert_eq!(df.column("b").unwrap().as_i64().unwrap(), &[10, 20, 30]);
    }

    #[test]
    fn write_cells_nan_into_int_keeps_int_na() {
        let mut df = sample();
        // a NaN write into the I64 column keeps int64 and marks the cell NA
        // (Decision 1: no float widening, the native-NA model)
        df.write_one(1, &[0], &Column::f64(vec![f64::NAN]))
            .unwrap();
        let b = df.column("b").unwrap();
        assert_eq!(b.dtype(), DType::I64);
        assert!(!b.is_valid(0) && b.is_valid(1) && b.is_valid(2));
        assert_eq!(b.as_i64().unwrap()[1..], [20, 30]);
    }

    #[test]
    fn write_cells_drops_computed_status() {
        let mut df = sample();
        df.set_computed("a", std::sync::Arc::new(()), Reach { lookback: 1, resume: 0, continuation: 1, history: History::Window });
        assert_eq!(df.computed_columns().len(), 1);
        // a manual write into the cached column drops its computed status
        df.write_one(0, &[0], &Column::f64(vec![7.0]))
            .unwrap();
        assert!(df.computed_columns().is_empty());
    }

    #[test]
    fn set_index_tz_requires_datetime_index() {
        let df = DataFrame::new(
            vec!["c".into()],
            vec![Column::f64(vec![1.0])],
            None, // RangeIndex
        )
        .unwrap();
        assert!(df.set_index_tz(crate::tz::Tz::Utc).is_err());
    }

    #[test]
    fn tz_convert_keeps_instant_localize_shifts() {
        use crate::tz::Tz;
        // a frame whose datetime index was ingested as UTC instants
        let ns = crate::datetime::parse_ns("2021-01-01 12:00:00").unwrap();
        let df = DataFrame::new(
            vec!["c".into()],
            vec![Column::f64(vec![1.0])],
            Some(Index::new(None, crate::IndexKind::Datetime(vec![ns], Tz::Utc)).unwrap()),
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

        // tz_localize anchors a NAIVE frame only: an already-aware frame errors
        // (use tz_convert), and a naive frame cannot tz_convert (no source zone).
        assert!(df.tz_localize(Tz::parse("America/New_York").unwrap()).is_err());
        let naive = DataFrame::new(
            vec!["c".into()],
            vec![Column::f64(vec![1.0])],
            Some(Index::new(None, crate::IndexKind::Datetime(vec![ns], Tz::Naive)).unwrap()),
        )
        .unwrap();
        assert!(naive.tz_convert(Tz::Utc).is_err());
        // tz_localize: wall-clock 12:00 reinterpreted as NY -> instant moves +5h to UTC.
        let loc = naive
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
    fn write_cells_length_and_dtype_guards() {
        let mut df = sample();
        // wrong-length array
        assert!(df
            .write_one(0, &[0, 1], &Column::f64(vec![1.0, 2.0, 3.0]))
            .is_err());
        // out-of-range row
        assert!(df
            .write_one(0, &[9], &Column::f64(vec![1.0]))
            .is_err());
        // a bool into a numeric column COERCES (true -> 1.0), matching the Series
        // `set_float_at` path (Python `bool` is an int subclass)
        df.write_one(0, &[0], &Column::bool(vec![true]))
            .unwrap();
        assert_eq!(df.column("a").unwrap().as_f64().unwrap()[0], 1.0);
        // a str / datetime into a numeric column is a hard dtype error (no silent
        // funnel through `to_f64_vec`)
        assert!(df
            .write_one(0, &[0], &Column::str(vec!["x".into()]))
            .is_err());
        assert!(df
            .write_one(1, &[0], &Column::datetime(vec![123]))
            .is_err());
    }

    #[test]
    fn write_cells_type_combinations_and_col_out_of_range() {
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
            .write_one(99, &[0], &Column::f64(vec![1.0]))
            .is_err());
        df.write_one(0, &[1], &Column::i64(vec![7])).unwrap(); // F64 <- I64
        df.write_one(1, &[0], &Column::bool(vec![false]))
            .unwrap(); // Bool <- Bool
        df.write_one(2, &[0], &Column::str(vec!["z".into()]))
            .unwrap(); // Str <- Str
        df.write_one(3, &[2], &Column::datetime(vec![99]))
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
            Some(Index::new(None, crate::IndexKind::Datetime(vec![ns], Tz::Utc)).unwrap()),
        )
        .unwrap();
        assert!(df
            .tz_localize(Tz::parse("America/New_York").unwrap())
            .is_err());
    }

    /// Localizing keeps every sub-second digit (each instant goes through
    /// `Tz::localize_from`, never whole civil seconds), so sub-second rows stay
    /// distinct; a NaT label is rejected at checked construction.
    #[test]
    fn tz_localize_keeps_subseconds_and_nat() {
        use crate::tz::Tz;
        let a = crate::datetime::parse_ns("2026-03-08 06:59:59.123456789").unwrap();
        let df = DataFrame::new(
            vec!["c".into()],
            vec![Column::f64(vec![1.0, 2.0])],
            Some(Index::new(None, crate::IndexKind::Datetime(vec![a, a + 1], Tz::Naive)).unwrap()),
        )
        .unwrap();
        let sh = df.tz_localize(Tz::parse("+08:00").unwrap()).unwrap();
        assert!(Index::new(None, crate::IndexKind::Datetime(vec![a, a + 1, crate::datetime::NAT], Tz::Naive)).is_err());
        let eight_h = 8 * 3_600_000_000_000;
        match sh.index().kind() {
            IndexKind::Datetime(v, _) => {
                assert_eq!(v, &vec![a - eight_h, a + 1 - eight_h])
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn fold_forming_row_combines_in_place() {
        let mut df = DataFrame::new(
            vec!["high".into(), "vol".into()],
            vec![Column::f64(vec![1.0, 2.0]), Column::f64(vec![10.0, 20.0])],
            None,
        )
        .unwrap();
        let bar = DataFrame::new(
            vec!["high".into(), "vol".into()],
            vec![Column::f64(vec![5.0]), Column::f64(vec![3.0])],
            None,
        )
        .unwrap();
        df.fold_forming_row(1, &bar, 0, &[(0, 0, CombineOp::Max), (1, 1, CombineOp::Sum)], true)
            .unwrap();
        assert_eq!(df.columns()[0].get_f64(1), 5.0); // max(2, 5)
        assert_eq!(df.columns()[1].get_f64(1), 23.0); // 20 + 3
        // an empty plan is a no-op (and skips the invalidation).
        df.fold_forming_row(1, &bar, 0, &[], true).unwrap();
        assert_eq!(df.columns()[0].get_f64(1), 5.0);
    }


    #[test]
    fn adding_to_a_zero_column_projection_preserves_the_shared_datetime_index() {
        let index = Index::new(Some("time".into()),
            IndexKind::Datetime(vec![10, 20], crate::Tz::fixed(3600).unwrap())).unwrap();
        let source = DataFrame::new(vec!["close".into()], vec![Column::i64(vec![1, 2])], Some(index)).unwrap();
        let shared = Arc::clone(source.index());
        let mut projected = source.project(&[], 0, source.height()).unwrap();
        assert_eq!((projected.width(), projected.height()), (0, 2));
        projected.set_column("replacement", Column::i64(vec![3, 4])).unwrap();
        assert!(Arc::ptr_eq(projected.index(), &shared));
        assert_eq!(projected.index().to_i64_labels(), vec![10, 20]);
        assert_eq!(projected.index().name(), Some("time"));
        assert_eq!(projected.index().tz(), crate::Tz::fixed(3600).unwrap());
        assert_eq!(projected.column("replacement").unwrap().as_i64().unwrap(), &[3, 4]);
    }

    #[test]
    fn replacement_before_an_unmaterializable_tail_preserves_write_metadata() {
        let index = Index::new(None, IndexKind::Range { start: 0, len: i64::MAX as usize + 1 }).unwrap();
        let mut target = DataFrame::new(vec![], vec![], Some(index)).unwrap();
        target.invalidate_computed_on_write("close");
        let before_stale = target.stale_by;
        let before_index = Arc::clone(target.index());
        let rows = DataFrame::new(vec![], vec![],
            Some(Index::new(None, IndexKind::Int64(vec![i64::MAX, -1])).unwrap())).unwrap();
        assert!(matches!(target.append(&rows), Err(VolasError::Overflow(_))));
        assert_eq!(target.stale_by, before_stale);
        assert!(Arc::ptr_eq(target.index(), &before_index));
        assert_eq!(target.height(), i64::MAX as usize + 1);
    }
