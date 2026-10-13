//! Stateless expressions must have identical typed semantics in both continuations.
use volas_core::{Column, DataFrame};
use volas_directive::{parse, exec::{execute, execute_resume, execute_resume_step, initial_state}};

fn last<T: Clone>(base: T, edge: T) -> Vec<T> {
    let mut values = vec![base; 12];
    values[11] = edge;
    values
}

fn values() -> Vec<Column> {
    let ints = [0, 1, -1, 1_i64 << 53, -(1_i64 << 53), (1_i64 << 53) + 1,
        -((1_i64 << 53) + 1), i64::MIN, i64::MAX, i32::MIN as i64, i32::MAX as i64];
    let mut out = Vec::new();
    for i in ints {
        out.push(Column::i64(last(1, i)));
        if let Ok(i) = i32::try_from(i) { out.push(Column::i32(last(1, i))); }
    }
    for x in [0., -0., 1., -1., (1_u64 << 53) as f64, -((1_u64 << 53) as f64),
        i64::MIN as f64, i64::MAX as f64, i32::MIN as f64, i32::MAX as f64,
        ((1_u64 << 53) + 1) as f64, -(((1_u64 << 53) + 1) as f64),
        f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        out.push(Column::f64(last(1.0, x)));
        out.push(Column::f32(last(1.0, x as f32)));
    }
    for b in [Some(false), Some(true), None] {
        out.push(Column::bool_from_options(std::iter::repeat_n(Some(true), 11).chain(std::iter::once(b))));
    }
    for mut col in [Column::i64(vec![1; 11]), Column::i32(vec![1; 11])] {
        col.append(&Column::na_of(col.dtype(), 1)).unwrap();
        out.push(col);
    }
    out
}

fn check(d: &str, a: &Column, b: &Column) {
    let df = DataFrame::new(vec!["close".into(), "a".into(), "b".into()],
        vec![Column::f64((1..=12).map(f64::from).collect()), a.clone(), b.clone()], None).unwrap();
    let node = parse(d).unwrap();
    let full = execute(&df, &node);
    let Some(prev) = initial_state(&df, &node, 11) else { assert!(full.is_err(), "no state: {d}"); return; };
    let mut next = prev.clone();
    let step = execute_resume_step(&df, &node, &prev, (11, 0), &mut next);
    let vector = execute_resume(&df, &node, &prev, 11, 12, 0);
    match full {
        Ok(full) => {
            let want = full.to_f64_vec()[11];
            for got in [step, vector.map(|(v, _)| v.to_f64_vec()[0])] {
                let got = got.unwrap_or_else(|| panic!("declined {d}: {a:?}, {b:?}"));
                assert!(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                    "{d}: {a:?}, {b:?}: {got:?} != {want:?}");
            }
        }
        Err(_) => assert!(step.is_none() && vector.is_none(), "invalid expression accepted: {d}"),
    }
}

#[test]
fn typed_operators_match_full() {
    let values = values();
    for a in &values {
        for unary in ["-", "~"] { check(&format!("ema:2/({unary}a)"), a, a); }
        for b in &values {
            for op in ["+", "-", "*", "/", "<", "<=", "==", "!=", ">=", ">", "&", "|", "^", "//", "\\\\", "><"] {
                let d = if matches!(op, "+" | "-" | "*" | "/") {
                    format!("(a{op}b)/(ema:2)")
                } else { format!("(ema:2>0)&(a{op}b)") };
                check(&d, a, b);
            }
        }
    }
}

#[test]
fn zero_literals_keep_their_integer_or_float_sign_in_continuations() {
    let df = DataFrame::new(vec!["close".into()],
        vec![Column::f64((1..=12).map(f64::from).collect())], None).unwrap();
    for (zero, want) in [("-0.0", f64::NEG_INFINITY), ("-0", f64::INFINITY), ("-(0)", f64::INFINITY)] {
        let node = parse(&format!("ema:2/{zero}")).unwrap();
        let full = execute(&df, &node).unwrap();
        let state = initial_state(&df, &node, 11).unwrap();
        let step = execute_resume_step(&df, &node, &state, (11, 0), &mut state.clone()).unwrap();
        let (vector, _) = execute_resume(&df, &node, &state, 11, 12, 0).unwrap();
        for got in [full.get_f64(11), step, vector.get_f64(0)] {
            assert_eq!(got.to_bits(), want.to_bits(), "{zero}");
        }
    }
}
