//! The directive layer's half of a derived frame: the frame a cached directive is
//! computed over, and the replay of a recursion's states a middle selection asks for.

use std::borrow::Cow;

use volas_core::{Column, DataFrame, Replay};
use volas_directive::{execute, parse, refresh_frame, Replayer};

fn closes(n: usize) -> DataFrame {
    let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.7).sin()).collect();
    DataFrame::new(vec!["close".into()], vec![Column::f64(close)], None).unwrap()
}

#[test]
fn a_frame_with_only_data_columns_is_its_own_refresh_frame() {
    let df = closes(5);
    assert!(matches!(refresh_frame(&df), Cow::Borrowed(_)));
    let mut named = df.clone();
    named.set_column("change", Column::f64(vec![0.0; 5])).unwrap(); // a bare command's name
    assert_eq!(refresh_frame(&named).names(), &["close".to_string()]);
}

#[test]
fn a_replay_gives_the_states_a_full_computation_ends_with() {
    let mut df = closes(30);
    let node = parse("ema:5").unwrap();
    df.set_column("ema:5", execute(&df, &node).unwrap()).unwrap();
    df.set_computed("ema:5", std::sync::Arc::new((*node).clone()), volas_directive::lookback::reach(&node));
    let ends = volas_directive::states_at(&df, "ema:5", 20);
    let carried = volas_directive::resume_states(&df.slice_data(0, 20), &node);
    assert_eq!(ends, Some((carried.state.unwrap(), carried.before)));
    // a selection's first row: replayed over its lead, with no state before it
    let sub = df.rows(10, 30, &Replayer);
    let first = volas_directive::resume_states(&df.slice_data(0, 11), &node).state;
    assert_eq!(volas_directive::states_at(&sub, "ema:5", 1), Some((first.unwrap(), None)));
    let boundary = Replayer.boundary(&df, "ema:5", 7).unwrap();
    assert_eq!((boundary.at, boundary.state.map(|state| state.to_vec())), (6, volas_directive::resume_states(&df.slice_data(0, 7), &node).state));
    assert_eq!(Replayer.boundary(&df, "ema:5", 0), None); // no row before the first
    // states farther than the anchor spacing from the nearest point the frame keeps (a
    // history with no anchors) are not replayed
    let mut long = closes(2 * volas_core::dataframe::ANCHOR_SPACING);
    long.set_column("ema:5", execute(&long, &node).unwrap()).unwrap();
    long.set_computed("ema:5", std::sync::Arc::new((*node).clone()), volas_directive::lookback::reach(&node));
    assert_eq!(volas_directive::states_at(&long, "ema:5", long.height()), None);
    // a name that is not a cached column, or one that does not parse, has no history
    assert_eq!(Replayer.boundary(&df, "close", 7), None);
    assert_eq!(volas_directive::states_at(&df, "ema:", 20), None);
}

/// A computation over a long history keeps the checkpoints of a recursion carried by its
/// state at the anchor rows it passes — each the one a capture there gives — and its
/// states are the full computation's; a frame whose row 0 is past position 0 keeps them
/// at the same absolute rows; a directive that is no such recursion keeps none.
#[test]
fn a_computation_over_a_long_history_keeps_its_anchors() {
    use volas_core::dataframe::ANCHOR_SPACING as S;
    use volas_directive::exec::{capture, initial_state};
    let n = 3 * S + 7;
    let df = closes(n);
    for text in ["ema:5", "ema:5@(ma:3)"] {
        let node = parse(text).unwrap();
        let carried = volas_directive::resume_states(&df, &node);
        let rows: Vec<usize> = carried.anchors.iter().map(|cp| cp.at).collect();
        assert_eq!(rows, vec![S - 1, 2 * S - 1, 3 * S - 1], "{text}");
        for cp in &carried.anchors {
            assert_eq!(Some(cp), capture(&df, &node, cp.at + 1).as_ref(), "{text}");
        }
        assert_eq!(carried.state, initial_state(&df, &node, n), "{text}");
        assert_eq!(carried.before, initial_state(&df, &node, n - 1), "{text}");
    }
    let node = parse("ema:5").unwrap();
    let later = df.slice_data(10, n);
    let rows: Vec<usize> = volas_directive::resume_states(&later, &node).anchors.iter().map(|cp| cp.at).collect();
    assert_eq!(rows, vec![S - 1, 2 * S - 1, 3 * S - 1]);
    let finite = parse("ma:5").unwrap();
    assert!(volas_directive::resume_states(&df, &finite).anchors.is_empty());
    assert_eq!(volas_directive::exec::execute_with_states_and_anchors(&df, &finite).unwrap().3.len(), 3);
    // an anchor at the row before the last: the state before the last is the anchor's
    let short = closes(S + 1);
    let carried = volas_directive::resume_states(&short, &node);
    assert_eq!(carried.anchors.iter().map(|cp| cp.at).collect::<Vec<_>>(), vec![S - 1]);
    assert_eq!(carried.before, initial_state(&short, &node, S));
    assert_eq!(carried.state, initial_state(&short, &node, S + 1));
}

/// Every checkpoint carries the same framed state as a full prefix, including chains
/// and complete-row compaction at raw anchor boundaries.
#[test]
fn same_pass_anchors_match_prefix_continuations() {
    use volas_directive::exec::{execute_with_states_and_anchors, initial_state};
    for origin in [0, 17] {
        for missing in [false, true] {
            let n = 1600;
            let close: Vec<f64> = (0..n).map(|i| if missing && (i % 19 == 0 || (490..530).contains(&i)) {
                f64::NAN
            } else { 100.0 + (i as f64 * 0.7).sin() }).collect();
            let df = DataFrame::new(
                ["close", "open", "high", "low", "volume"].map(String::from).to_vec(),
                vec![Column::f64(close.clone()), Column::f64(close.iter().map(|x| x - 0.5).collect()),
                    Column::f64(close.iter().map(|x| x + 1.0).collect()), Column::f64(close.iter().map(|x| x - 1.0).collect()),
                    Column::f64(close.iter().map(|x| x * 10.0).collect())], None).unwrap().with_origin(origin);
            for text in ["ema:20", "ma:20", "crsi", "close>ma:20", "ema:5@(ma:3)", "stochrsi.d:5,5,3,1",
                "dema:20", "tema:20", "t3:20", "kama:20", "rsi:20", "wma:20", "trima:20", "sum:20",
                "vwma:20", "cci:20", "boll.upper:20", "var:20", "stddev:20", "linearreg:20", "mfi:20",
                "hma:20", "bbi", "dpo:20", "kst", "coppock", "cog:20", "correl:20@close,open", "beta:20@close,open",
                "vr:20", "imi:20", "emv:20", "cmf:20", "ultosc", "relative_vigor", "relative_vigor.signal", "dkx", "dkx.ma",
                "wvad", "apo", "ppo", "bias", "dma", "macdext", "stoch.d", "stochf.d"] {
                let node = parse(text).unwrap();
                let (col, state, before, anchors) = execute_with_states_and_anchors(&df, &node).unwrap();
                let expected = execute(&df, &node).unwrap();
                assert!(col.to_f64_vec().iter().zip(expected.to_f64_vec()).all(|(a,b)| a.to_bits()==b.to_bits() || a.is_nan() && b.is_nan()), "{text}");
                assert_eq!(anchors.len(), 3, "{text}, origin {origin}, missing {missing}");
                for cp in anchors {
                    let end = cp.at + 1 - origin;
                    let want = initial_state(&df, &node, end).unwrap();
                    assert_eq!(format!("{:?}", cp.state.unwrap()), format!("{want:?}"), "{text} row {} origin {origin}, missing {missing}", cp.at);
                }
                assert_eq!(format!("{state:?}"), format!("{:?}", initial_state(&df, &node, n)), "{text}");
                assert_eq!(format!("{before:?}"), format!("{:?}", initial_state(&df, &node, n - 1)), "{text}");
            }
        }
    }
}

/// A newer retained boundary checkpoint wins over an older usable anchor.
#[test]
fn replay_starts_from_the_newest_available_checkpoint() {
    use volas_directive::exec::{capture, initial_state};
    let mut df = closes(1200);
    let node = parse("ema:5").unwrap();
    df.set_column("ema:5", execute(&df, &node).unwrap()).unwrap();
    df.set_computed("ema:5", std::sync::Arc::new((*node).clone()), volas_directive::lookback::reach(&node));
    df.keep_computed_anchors("ema:5", vec![capture(&df, &node, 512).unwrap()]);
    df.set_computed_checkpoint("ema:5", Ok(capture(&df, &node, 901).unwrap()));
    let (state, before) = Replayer.continuation(&df, "ema:5", 1101).unwrap();
    assert_eq!(Some(state), initial_state(&df, &node, 1101));
    assert_eq!(before, initial_state(&df, &node, 1100));
}

/// Checkpoints keep kernel inputs, while raw operands still retain the prefix needed
/// to evaluate the first new row after a checkpoint.
#[test]
fn checkpoint_replay_keeps_stateless_operand_prefix_rows() {
    use volas_directive::exec::{execute_with_states_and_anchors, initial_state};
    for text in ["ema:20@(change:)", "ema:20@(roc:7)", "ema:20@(close//open)"] {
        let n = 1200;
        let close: Vec<_> = (0..n).map(|i|100.0+(i as f64*0.7).sin()).collect();
        let open: Vec<_> = (0..n).map(|i|100.0+(i as f64*0.5).cos()).collect();
        let mut df=DataFrame::new(vec!["close".into(),"open".into()],vec![Column::f64(close),Column::f64(open)],None).unwrap();
        let node=parse(text).unwrap();
        let (column,state,before,anchors)=execute_with_states_and_anchors(&df,&node).unwrap();
        df.set_column(text,column).unwrap();
        df.set_computed(text,std::sync::Arc::new((*node).clone()),volas_directive::lookback::reach(&node));
        df.set_computed_states(text,state,before);
        df.keep_computed_anchors(text,anchors);
        let part=df.rows(900,1100,&Replayer);
        let got=Replayer.continuation(&part,text,part.height()).unwrap();
        assert_eq!(format!("{:?}",Some(got.0)),format!("{:?}",initial_state(&df,&node,1100)),"{text}");
        assert_eq!(format!("{:?}",got.1),format!("{:?}",initial_state(&df,&node,1099)),"{text}");
    }
}

#[test]
fn default_series_classification_distinguishes_bound_data_and_explicit_operands() {
    use volas_directive::{bind_frame, parse, uses_default_series};
    for text in ["close", "obv", "ema:2", "atr:14"] {
        assert!(uses_default_series(&parse(text).unwrap()), "{text}");
    }
    for text in ["2", "close+1", "ema:2@close", "ema:2@(close+open)"] {
        assert!(!uses_default_series(&parse(text).unwrap()), "{text}");
    }
    let data = bind_frame(&parse("close").unwrap(), &|name| name == "close", "close").unwrap();
    assert!(!uses_default_series(&data));
}

/// Persisted anchors share framed wire storage, while each borrowed range remains
/// independently resumable after the computing pass's temporary publication is gone.
#[test]
fn persisted_anchor_states_share_one_framed_word_pool() {
    let df=closes(1600);
    for text in ["ema:20", "ma:20", "crsi", "close>ma:20"] {
        let node=parse(text).unwrap();
        let (_,_,_,anchors)=volas_directive::exec::execute_with_states_and_anchors(&df,&node).unwrap();
        assert_eq!(anchors.len(),3);
        for adjacent in anchors.windows(2) {
            let left=adjacent[0].state.as_ref().unwrap();
            let right=adjacent[1].state.as_ref().unwrap();
            assert_eq!(left.as_ptr().wrapping_add(left.len()),right.as_ptr(),"{text}");
        }
        let cp=anchors[0].clone();
        drop(anchors);
        let (_,replayed)=volas_directive::exec::replay(&df,&node,&cp,1200).unwrap();
        let expected=volas_directive::exec::initial_state(&df,&node,1200).unwrap();
        assert_eq!(format!("{:?}",replayed.state.unwrap()),format!("{expected:?}"),"{text}");
    }
}

/// Every raw anchor boundary keeps partial seed rows even before a long recursive
/// kernel can publish a state. Derived frames resume from that same opaque wire.
#[test]
fn anchor_boundaries_before_a_long_seed_remain_resumable() {
    use volas_directive::exec::{execute_with_states_and_anchors, execute_resume, initial_state};
    for origin in [0, 17] {
        for mode in ["clean", "leading", "interior"] {
            let n = 4096;
            let close: Vec<_> = (0..n).map(|i| {
                if mode == "leading" && i < 1300 || mode == "interior" && (i % 7 == 0 || (490..1050).contains(&i)) { f64::NAN }
                else { 100.0 + (i as f64 * 0.7).sin() }
            }).collect();
            let source = DataFrame::new(vec!["close".into()], vec![Column::f64(close)], None).unwrap().with_origin(origin);
            for text in ["ema:1000", "rsi:1000", "crsi:3,2,1000", "crsi:1000,2,100"] {
                let node = parse(text).unwrap();
                let (column, state, before, anchors) = execute_with_states_and_anchors(&source, &node).unwrap();
                let ends: Vec<_> = (512 - origin % 512..n).step_by(512).collect();
                assert_eq!(anchors.iter().map(|cp| cp.at + 1 - origin).collect::<Vec<_>>(), ends, "{text}, {mode}, origin{origin}");
                for cp in &anchors {
                    let expected = initial_state(&source, &node, cp.at + 1 - origin).unwrap();
                    assert_eq!(format!("{:?}", cp.state.as_ref().unwrap()), format!("{expected:?}"), "{text}, {mode}, origin{origin}, at{}", cp.at);
                }
                let expected = execute(&source, &node).unwrap().to_f64_vec();
                let mut cached = source.clone();
                cached.set_column(text, column).unwrap();
                cached.set_computed(text, std::sync::Arc::new((*node).clone()), volas_directive::lookback::reach(&node));
                cached.set_computed_states(text, state, before);
                cached.keep_computed_anchors(text, anchors);
                let mut part = cached.rows(600, 900, &Replayer);
                let (carried, _) = Replayer.continuation(&part, text, part.height()).unwrap();
                assert_eq!(format!("{carried:?}"), format!("{:?}", initial_state(&source, &node, 900).unwrap()), "derived {text}, {mode}, origin{origin}");
                let rows = DataFrame::new(vec!["close".into()], vec![source.column("close").unwrap().slice(900, 1800)], None).unwrap();
                part.append(&rows).unwrap();
                let (tail, next) = execute_resume(&part, &node, &carried, 300, part.height(), part.origin()).unwrap();
                assert!(tail.to_f64_vec().iter().zip(&expected[900..1800]).all(|(a,b)| a.to_bits()==b.to_bits() || a.is_nan() && b.is_nan()), "append {text}, {mode}, origin{origin}");
                assert_eq!(format!("{next:?}"), format!("{:?}", initial_state(&source, &node, 1800).unwrap()), "append state {text}, {mode}, origin{origin}");
            }
        }
    }
}
