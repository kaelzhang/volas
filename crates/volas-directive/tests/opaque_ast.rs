use volas_directive::{bind_frame, parse, stringify, types::{Continuation, Node}};

#[test]
fn bound_tree_and_continuation_are_read_only_views() {
    let ast = parse("ema:2").unwrap();
    let Node::Command(c) = ast.node() else { panic!("command expected") };
    assert_eq!(c.args.values()[0].as_usize(), 2);
    assert!(c.args.bare().is_none());
    assert_eq!(c.args.spec().lookback_of(c.args.values()), 1);
    assert!(matches!(ast.continuation(), Continuation::Own));
    let mut detached = ast.node().clone();
    let Node::Command(c) = &mut detached else { unreachable!() };
    c.series.clear();
    c.name = "untrusted".into();
    assert_eq!(stringify(ast.as_ref()), "ema:2");
}

#[test]
fn binding_decides_the_existing_continuation_cases() {
    for (source, kind) in [
        ("close + 1", 0), ("llv:5", 0), ("ma:20", 1), ("rsi:14", 1),
        ("ema:5@(ema:3)", 2), ("close - ma:5 + obv", 2), ("ema:5@(apo:2,3,1)", 2),
    ] {
        let ast = parse(source).unwrap();
        let actual = match ast.continuation() {
            Continuation::None => 0, Continuation::Own => 1, Continuation::Chain(_) => 2,
        };
        assert_eq!(actual, kind, "{source}");
    }
    let ast = bind_frame(&parse("obv").unwrap(), &|_| false, "obv").unwrap();
    assert!(matches!(ast.continuation(), Continuation::Own));
    let Node::Command(c) = ast.node() else { unreachable!() };
    assert_eq!(c.args.bare(), Some("obv"));
}
