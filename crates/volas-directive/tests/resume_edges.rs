use volas_core::{Column, DataFrame};
use volas_directive::{
    exec::{execute, execute_resume, execute_resume_step, initial_state},
    parse,
};

fn ohlcv(n: usize) -> DataFrame {
    let close: Vec<f64> = (0..n)
        .map(|i| 50.0 + i as f64 * 0.3 + (i as f64 * 0.21).sin() * 2.0)
        .collect();
    let open: Vec<f64> = close.iter().map(|v| v - 0.3).collect();
    let high: Vec<f64> = close.iter().map(|v| v + 1.2).collect();
    let low: Vec<f64> = close.iter().map(|v| v - 1.4).collect();
    let volume: Vec<f64> = (0..n).map(|i| 1000.0 + i as f64 * 10.0).collect();
    DataFrame::new(
        vec![
            "open".into(),
            "high".into(),
            "low".into(),
            "close".into(),
            "volume".into(),
        ],
        vec![
            Column::f64(open),
            Column::f64(high),
            Column::f64(low),
            Column::f64(close),
            Column::f64(volume),
        ],
        None,
    )
    .unwrap()
}

fn assert_f64_same(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let close = (a - e).abs() <= 1e-10 + 1e-12 * e.abs();
        assert!(
            close || a.to_bits() == e.to_bits() || (a.is_nan() && e.is_nan()),
            "index {i}: {a:?} != {e:?}"
        );
    }
}

fn resume_matches_full(df: &DataFrame, directive: &str, from: usize) {
    let node = parse(directive).unwrap();
    let head = df.slice_data(0, from);
    execute(&head, &node).unwrap();
    let state = initial_state(&head, &node, usize::MAX)
        .unwrap_or_else(|| panic!("no initial state for {directive}"));
    let (tail, _) = execute_resume(df, &node, &state, from, df.height(), 0)
        .unwrap_or_else(|| panic!("no resume for {directive}"));
    let full = execute(df, &node).unwrap().to_f64_vec();
    assert_f64_same(&tail.to_f64_vec(), &full[from..]);
}

/// One step of `directive` at `row` from its state over the rows before.
fn step(df: &DataFrame, directive: &str, row: usize) -> Option<f64> {
    let node = parse(directive).unwrap();
    let head = df.slice_data(0, row);
    let state = initial_state(&head, &node, usize::MAX)?;
    let mut next = state.clone();
    execute_resume_step(df, &node, &state, (row, 0), &mut next)
}

/// A formula of its own row (the price transforms, TR, the momentum family) steps one
/// row with an empty state, exactly as the full kernel computes it.
#[test]
fn a_row_formula_steps_like_its_kernel() {
    let df = ohlcv(20);
    for directive in [
        "avgprice", "medprice", "typprice", "wclprice", "bop", "tr", "mom:3", "roc:3", "rocp:3", "rocr:3",
        "rocr100:3", "mom:30",
    ] {
        let full = execute(&df, &parse(directive).unwrap()).unwrap().to_f64_vec();
        for row in [1, 5, 19] {
            let Some(value) = step(&df, directive, row) else {
                assert!(row < volas_directive::lookback::lookback(&parse(directive).unwrap()));
                continue;
            };
            assert_f64_same(&[value], &[full[row]]);
        }
    }
    // a row past the end declines; TR has no prior bar at row 0
    let node = parse("avgprice").unwrap();
    assert!(execute_resume_step(&df, &node, &[], (20, 0), &mut []).is_none());
    assert!(step(&df, "tr", 0).is_none());
    let zero_prior = DataFrame::new(vec!["close".into()], vec![Column::f64(vec![0.0, 1.0, 2.0])], None).unwrap();
    assert_eq!(step(&zero_prior, "roc:2", 2).unwrap(), 0.0);
}

#[test]
fn a_row_formula_resumes_its_tail_like_its_kernel() {
    let df = ohlcv(12);
    for directive in [
        "avgprice", "medprice", "typprice", "wclprice", "bop", "tr", "mom:3", "roc:3", "rocp:3", "rocr:3", "rocr100:3",
    ] {
        let node = parse(directive).unwrap();
        let from = volas_directive::lookback::lookback(&node);
        let seed = initial_state(&df, &node, from).unwrap();
        let (tail, state) = execute_resume(&df, &node, &seed, from, usize::MAX, 0)
            .unwrap_or_else(|| panic!("tail resume failed for {directive}"));
        assert_eq!(state, initial_state(&df, &node, usize::MAX).unwrap());
        assert_f64_same(&tail.to_f64_vec(), &execute(&df, &node).unwrap().to_f64_vec()[from..]);
    }
}

#[test]
fn a_step_reads_an_integer_column_as_f64() {
    let df = DataFrame::new(vec!["close".into()], vec![Column::i64(vec![10, 11, 13, 12])], None).unwrap();
    assert_eq!(step(&df, "mom:2", 3).unwrap(), 1.0);
}

#[test]
fn node_resume_dispatch_covers_stateless_and_recursive_indicators() {
    let df = ohlcv(180);
    for directive in [
        "avgprice",
        "medprice",
        "typprice",
        "wclprice",
        "bop",
        "tr",
        "mom:5",
        "roc:5",
        "rocp:5",
        "rocr:5",
        "rocr100:5",
        "ema:12",
        "smma:12",
        "dema:12",
        "tema:12",
        "t3:12",
        "trix:12",
        "kama:10",
        "macd",
        "macd.signal",
        "macd.histogram",
        "macdfix",
        "macdfix.signal",
        "macdfix.histogram",
        "rsi:14",
        "cmo:14",
        "atr:14",
        "natr:14",
        "plus_dm:14",
        "minus_dm:14",
        "plus_di:14",
        "minus_di:14",
        "dx:14",
        "adx:14",
        "adxr:14",
        "ht_dcperiod",
        "ht_phasor",
        "ht_phasor.quadrature",
        "ht_dcphase",
        "ht_sine",
        "ht_sine.leadsine",
        "ht_trendline",
        "ht_trendmode",
        "mama",
        "mama.fama",
        "stochrsi.k",
        "stochrsi.d",
        "kdj.k",
        "kdj.d",
        "kdj.j",
        // Group A cumulative / EMA-recursion family.
        "pvt",
        "nvi",
        "pvi",
        "efi:13",
        "tsi:25,13",
        "mass_index:25",
        // the cumulative volume lines, the Keltner middle and bands, SAR, Supertrend
        "obv",
        "ad",
        "wad",
        "keltner",
        "keltner.upper",
        "supertrend",
        "sar",
        "sarext",
    ] {
        resume_matches_full(&df, directive, 120);
    }
}

#[test]
fn execute_dispatch_covers_specialized_indicator_defaults() {
    let df = ohlcv(120);
    for directive in [
        "macdext",
        "macdext.signal",
        "macdext.histogram",
        "stoch.d",
        "stochf.d",
        "stochrsi.d",
    ] {
        let out = execute(&df, &parse(directive).unwrap())
            .unwrap_or_else(|e| panic!("execute failed for {directive}: {e:?}"));
        assert_eq!(out.len(), df.height(), "{directive}");
    }

    for directive in [
        "macdext:12,99",
        "macdext:12,0,26,99",
        "macdext.signal:12,0,26,0,9,99",
    ] {
        // An out-of-range matype is rejected when binding (at parse), so chain
        // parse -> execute to assert the directive is refused somewhere.
        assert!(
            parse(directive).and_then(|n| execute(&df, &n)).is_err(),
            "{directive}"
        );
    }
}

#[test]
fn execute_dispatch_covers_specialized_default_fallbacks() {
    let mut df = ohlcv(120);
    df.set_column("high", {
        let mut high = df.column("high").unwrap().to_f64_vec();
        high[4] = f64::NAN;
        Column::f64(high)
    })
    .unwrap();
    for directive in ["stoch.d", "stochf.d"] {
        let out = execute(&df, &parse(directive).unwrap())
            .unwrap_or_else(|e| panic!("execute failed for {directive}: {e:?}"));
        assert_eq!(out.len(), df.height(), "{directive}");
    }

    let mut df = ohlcv(120);
    df.set_column("close", {
        let mut close = df.column("close").unwrap().to_f64_vec();
        close[20] = f64::NAN;
        Column::f64(close)
    })
    .unwrap();
    let out = execute(&df, &parse("stochrsi.d").unwrap()).unwrap();
    assert_eq!(out.len(), df.height());
}

/// `ma` at a recursive MA type runs as that average's own command, so it continues
/// from that command's state — by the `Vec` resume and by the single-row step — exactly
/// as a full recompute; and `kcw` continues the Keltner band's EMA + ATR pair. Both
/// over a series with a missing row too.
#[test]
fn recursive_moving_averages_and_kcw_continue_from_a_state() {
    let mut df = ohlcv(160);
    for gap in [false, true] {
        if gap {
            let mut close = df.column("close").unwrap().to_f64_vec();
            close[120] = f64::NAN;
            df.set_column("close", Column::f64(close)).unwrap();
        }
        for directive in ["ma:5,1", "ma:5,3", "ma:5,4", "ma:5,6", "ma:5,7", "ma:5,8", "kcw", "kcw:7,4,1.5"] {
            let node = parse(directive).unwrap();
            let full = execute(&df, &node).unwrap().to_f64_vec();
            let exact = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
            for from in [100, 121, 159] {
                let head = df.slice_data(0, from);
                let state = initial_state(&head, &node, usize::MAX).unwrap();
                let (tail, _) = execute_resume(&df, &node, &state, from, usize::MAX, 0).unwrap();
                let tail = tail.to_f64_vec();
                assert!(tail.iter().zip(&full[from..]).all(|(&a, &b)| exact(a, b)), "{directive} from {from}");
                let mut next = state.clone();
                if let Some(v) = execute_resume_step(&df.slice_data(0, from + 1), &node, &state, (from, 0), &mut next) {
                    assert!(exact(v, full[from]), "{directive} step at {from}: {v} vs {}", full[from]);
                } else {
                    assert_eq!(directive, "ma:5,7", "only MAMA has no single-row step");
                }
            }
        }
    }
    // a period of 1 is the identity for every type: no recursion, the SMA's own state
    let node = parse("ma:1,1").unwrap();
    let head = df.slice_data(0, 50);
    assert!(initial_state(&head, &node, usize::MAX).is_some());
}

#[test]
fn resume_decline_paths_cover_question_mark_branches() {
    let df = ohlcv(80);
    let p = |directive: &str| parse(directive).unwrap();
    let tr = initial_state(&df, &p("tr"), 1).unwrap();
    let mom = initial_state(&df, &p("mom:30"), 30).unwrap();

    assert!(execute_resume(&df, &p("tr"), &tr, 1, df.height(), 0).is_some());
    assert!(execute_resume(&df, &p("mom:30"), &mom, 30, df.height(), 0).is_some());

    for directive in [
        "kama:30",
        "rsi:14",
        "cmo:14",
        "atr:14",
        "natr:14",
        "plus_dm:14",
        "minus_dm:14",
        "dx:14",
        "adx:14",
        "adxr:14",
        "ht_phasor",
        "ht_sine",
        // keltner bands compose the EMA + ATR resumes; the ATR resume declines at from 0.
        "keltner.upper:20,10,2",
        // supertrend's resume composes the ATR resume, which declines at from 0.
        "supertrend:10,3",
    ] {
        assert!(
            execute_resume(&df, &p(directive), &[0.0; 8], 0, df.height(), 0).is_none(),
            "{directive}"
        );
    }
}

#[test]
fn a_huge_window_does_not_allocate_its_declared_size_before_warmup() {
    let df = ohlcv(20);
    for text in ["ma:999999999999", "close-ma:999999999999+obv"] {
        assert!(initial_state(&df, &parse(text).unwrap(), df.height()).is_none(), "{text}");
    }
    let seed = initial_state(&df, &parse("ema:999999999999").unwrap(), df.height()).unwrap();
    assert!(seed.len() < 100);
    let chain = initial_state(&df, &parse("llv:999999999999@(ema:3)").unwrap(), df.height()).unwrap();
    assert!(chain.len() < 100);
}
