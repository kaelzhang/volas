use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use volas_core::{Column, DataFrame};
use volas_directive::{parse, exec::{initial_state, execute_resume_step}};

struct Counter;
static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
#[global_allocator]
static GLOBAL: Counter = Counter;

fn count() {
    if ENABLED.load(Ordering::Relaxed) {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
    }
}

// Forward every allocation unchanged to System; the counter itself never allocates.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count();
        System.realloc(ptr, layout, size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

fn allocations(d: &str, height: usize) -> usize {
    let data: Vec<f64> = (0..height).map(|i| 100.0 + (i as f64 * 0.01).sin()).collect();
    let df = DataFrame::new(
        vec!["close".into(), "high".into(), "low".into(), "volume".into()],
        vec![
            Column::f64(data.clone()),
            Column::f64(data.iter().map(|v| v + 1.0).collect()),
            Column::f64(data.iter().map(|v| v - 1.0).collect()),
            Column::i64(vec![100; height]),
        ],
        None,
    ).unwrap();
    let node = parse(d).unwrap();
    let prev = initial_state(&df, &node, height - 1).unwrap();
    let mut next = prev.clone();
    assert!(execute_resume_step(&df, &node, &prev, (height - 1, 0), &mut next).is_some());
    ALLOCS.store(0, Ordering::Relaxed);
    ENABLED.store(true, Ordering::Relaxed);
    for _ in 0..1000 {
        std::hint::black_box(execute_resume_step(&df, &node, &prev, (height - 1, 0), &mut next));
    }
    ENABLED.store(false, Ordering::Relaxed);
    ALLOCS.load(Ordering::Relaxed)
}

// One test owns the process-wide counter; test harness threads do no measured work.
#[test]
fn single_row_allocation_budget() {
    let own: Vec<_> = ["ema:20", "ma:20", "rsi:14", "atr:14"].into_iter()
        .map(|d| (d, allocations(d, 1000))).collect();
    let short = allocations("ema:5@(ema:3)", 1000);
    let long = allocations("ema:5@(ema:3)", 100_000);
    for (d, count) in &own {
        println!("{d}: {count} allocations / 1000 steps");
    }
    println!("ema:5@(ema:3): {short} / {long} allocations at 1000 / 100000 rows");
    for (d, count) in own {
        assert_eq!(count, 0, "{d}");
    }
    assert_eq!(short, long);
    assert_eq!(short, 0);
    for d in ["close-ma:5+obv", "close>ema:5"] {
        assert_eq!(allocations(d, 1000), 0, "{d}");
    }
}
