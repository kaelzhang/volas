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
    fn row_major_i64_export_is_exact() {
        // The integer export takes each column's raw value (no f64 round-trip): a
        // datetime keeps exact epoch-ns with NaT = i64::MIN, a large i64 survives
        // past 2^53, a float truncates toward zero, a str (rejected by the caller
        // before this) contributes its 0 placeholder.
        let big = (1i64 << 60) + 1;
        let df = DataFrame::new(
            vec!["t".into(), "n".into(), "f".into(), "s".into()],
            vec![
                Column::datetime(vec![123, i64::MIN]),
                Column::i64(vec![big, -7]),
                Column::f64(vec![2.9, -2.9]),
                Column::str(vec!["a".into(), "b".into()]),
            ],
            None,
        )
        .unwrap();
        let (data, h, w) = df.to_row_major_i64();
        assert_eq!((h, w), (2, 4));
        // row 0: datetime 123 (exact), i64 2^60+1 (exact), f64 2.9->2, str->0
        assert_eq!(&data[0..4], &[123, big, 2, 0]);
        // row 1: NaT stays i64::MIN, -7, -2.9->-2, str->0
        assert_eq!(&data[4..8], &[i64::MIN, -7, -2, 0]);
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
