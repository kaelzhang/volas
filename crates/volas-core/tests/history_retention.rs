//! Live bytes, rather than allocation counts: a short derived frame must release
//! unreachable checkpoint banks AND their shared state publication arena.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::{Arc, atomic::{AtomicIsize, Ordering}};
use volas_core::{Checkpoint, Column, DataFrame, History, Reach, Replay, StateWords};

struct Counting;
static LIVE: AtomicIsize = AtomicIsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() { LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed); }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        System.dealloc(ptr, layout);
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
struct NoReplay;
impl Replay for NoReplay {
    fn boundary(&self, _: &DataFrame, _: &str, _: usize) -> Option<Checkpoint> { None }
    fn continuation(&self, _: &DataFrame, _: &str, _: usize) -> Option<(Vec<f64>, Option<Vec<f64>>)> { None }
}

fn retained(rows: usize, middle: bool, chunked: bool) -> isize {
    let before = LIVE.load(Ordering::Relaxed);
    let mut parent = DataFrame::new(vec!["close".into(), "ema".into()],
        vec![Column::f64(vec![1.0; rows]); 2], None).unwrap();
    parent.set_computed("ema", Arc::new(()), Reach {
        lookback: 199, resume: 200, continuation: 200, history: History::Carried { operands: 0 },
    });
    let count = rows / 512 - 1;
    let width = 1200;
    if chunked {
        let mut arena = volas_core::dataframe::StateWordArena::default();
        for _ in 0..count {
            arena.write(width, |words| { words.resize(words.len() + width, 1.0); true });
        }
        parent.keep_computed_anchors("ema", arena.finish().into_iter().enumerate().map(|(k, state)| Checkpoint {
            at: (k + 1) * 512 - 1, state: Some(state), raw: vec![],
        }).collect());
    } else {
        let words = Arc::new(vec![1.0; count * width]);
        parent.keep_computed_anchors("ema", (0..count).map(|k| Checkpoint {
            at: (k + 1) * 512 - 1,
            state: Some(StateWords::from_shared(Arc::clone(&words), k * width..(k + 1) * width).unwrap()),
            raw: vec![],
        }).collect());
        drop(words);
    }
    let end = if middle { rows / 2 + 100 } else { rows };
    let short = parent.rows(end - 5, end, &NoReplay);
    assert!(short.computed_checkpoint("ema").is_some());
    drop(parent);
    let bytes = LIVE.load(Ordering::Relaxed) - before;
    std::hint::black_box(&short);
    drop(short);
    bytes
}

#[test]
fn short_derivations_retain_bytes_independent_of_seed_length() {
    for (chunked, middle) in [(false, false), (false, true), (true, false), (true, true)] {
        let small = retained(16_384, middle, chunked);
        let medium = retained(131_072, middle, chunked);
        let large = retained(1_048_576, middle, chunked);
        println!("chunked={chunked}, middle={middle}: seed16384={small} B, seed131072={medium} B, seed1048576={large} B");
        assert!(medium <= small + 32_768);
        assert!(large <= small + 32_768,
            "five rows retained history proportional to seed: small={small}, large={large}, middle={middle}");
        assert!(large < 128_000, "a bounded checkpoint and lead retained {large} bytes");
    }
}
