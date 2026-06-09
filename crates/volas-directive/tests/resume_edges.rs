use volas_core::{Column, DataFrame};
use volas_directive::{
    exec::{
        execute, execute_resume, execute_resume_default_series, execute_resume_default_series_one,
        initial_state,
    },
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
    let head = df.slice(0, from);
    let computed = execute(&head, &node).unwrap();
    let state = initial_state(&head, &node, &computed)
        .unwrap_or_else(|| panic!("no initial state for {directive}"));
    let (tail, _) = execute_resume(df, &node, &state, from, 0)
        .unwrap_or_else(|| panic!("no resume for {directive}"));
    let full = execute(df, &node).unwrap().to_f64_vec();
    assert_f64_same(&tail.to_f64_vec(), &full[from..]);
}

#[test]
fn default_series_scalar_resume_covers_every_directive_arm() {
    let df = ohlcv(20);
    assert!(execute_resume_default_series_one(&df, "ma:2@close", 19).is_none());
    assert!(execute_resume_default_series_one(&df, "avgprice", 20).is_none());

    assert_eq!(
        execute_resume_default_series_one(&df, "avgprice", 2).unwrap(),
        (df.column("open").unwrap().as_f64().unwrap()[2]
            + df.column("high").unwrap().as_f64().unwrap()[2]
            + df.column("low").unwrap().as_f64().unwrap()[2]
            + df.column("close").unwrap().as_f64().unwrap()[2])
            / 4.0
    );
    assert!(execute_resume_default_series_one(&df, "tr", 0)
        .unwrap()
        .is_nan());
    for directive in [
        "medprice",
        "typprice",
        "wclprice",
        "tr",
        "mom:3",
        "roc:3",
        "rocp:3",
        "rocr:3",
        "rocr100:3",
    ] {
        let value = execute_resume_default_series_one(&df, directive, 5)
            .unwrap_or_else(|| panic!("scalar resume failed for {directive}"));
        assert!(value.is_finite(), "{directive}");
    }

    assert!(execute_resume_default_series_one(&df, "mom:30", 5)
        .unwrap()
        .is_nan());
    let zero_prior = DataFrame::new(
        vec!["close".into()],
        vec![Column::f64(vec![0.0, 1.0, 2.0])],
        None,
    )
    .unwrap();
    assert_eq!(
        execute_resume_default_series_one(&zero_prior, "roc:2", 2).unwrap(),
        0.0
    );
}

#[test]
fn default_series_tail_resume_covers_transform_and_ratio_arms() {
    let df = ohlcv(12);
    assert!(execute_resume_default_series(&df, "ma:2@close", 6).is_none());

    for directive in [
        "avgprice",
        "medprice",
        "typprice",
        "wclprice",
        "tr",
        "mom:3",
        "roc:3",
        "rocp:3",
        "rocr:3",
        "rocr100:3",
    ] {
        let (tail, _) = execute_resume_default_series(&df, directive, 0)
            .unwrap_or_else(|| panic!("tail resume failed for {directive}"));
        let full = execute(&df, &parse(directive).unwrap())
            .unwrap()
            .to_f64_vec();
        assert_f64_same(&tail.to_f64_vec(), &full);
    }
}

#[test]
fn node_resume_dispatch_covers_stateless_and_recursive_indicators() {
    let df = ohlcv(180);
    for directive in [
        "avgprice",
        "medprice",
        "typprice",
        "wclprice",
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
        assert!(
            execute(&df, &parse(directive).unwrap()).is_err(),
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

#[test]
fn resume_decline_paths_cover_question_mark_branches() {
    let df = ohlcv(80);
    let p = |directive: &str| parse(directive).unwrap();
    let empty: &[f64] = &[];

    assert!(execute_resume(&df, &p("tr"), empty, 0, 0).is_some());
    assert!(execute_resume(&df, &p("mom:30"), empty, 0, 0).is_some());

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
            execute_resume(&df, &p(directive), &[0.0; 8], 0, 0).is_none(),
            "{directive}"
        );
    }
}
