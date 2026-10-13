//! Reproducible batch and open-period timings, including allocation requests.
//! Run with `cargo run --release -p volas-time --example cumulate_cost`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::time::Instant;
use volas_core::{Column, DataFrame, Index, Tz};
use volas_time::{cumulate, AggSpec, Cumulator, TimeFrame};

struct Alloc;
static COUNTING: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Alloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Relaxed) {
            CALLS.fetch_add(1, Relaxed);
            BYTES.fetch_add(layout.size(), Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNTING.load(Relaxed) {
            CALLS.fetch_add(1, Relaxed);
            BYTES.fetch_add(size, Relaxed);
        }
        System.realloc(ptr, layout, size)
    }
}

#[global_allocator]
static ALLOC: Alloc = Alloc;

fn measure<T>(name: &str, n: usize, repeat: usize, mut f: impl FnMut() -> T) {
    for _ in 0..3 { black_box(f()); }
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..repeat { black_box(f()); }
        best = best.min(start.elapsed().as_secs_f64() * 1e6 / repeat as f64);
    }
    CALLS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    COUNTING.store(true, Relaxed);
    black_box(f());
    COUNTING.store(false, Relaxed);
    println!("{name},{n},{best:.3},{},{}", CALLS.load(Relaxed), BYTES.load(Relaxed));
}

fn bars(n: usize, step: i64) -> DataFrame {
    let names: Vec<_> = ["open", "high", "low", "close", "volume"].map(String::from).to_vec();
    let columns = names.iter().map(|_| Column::f64((0..n).map(|i| i as f64 + 1.0).collect())).collect();
    DataFrame::new(names, columns, Some(Index::datetime((0..n).map(|i| i as i64 * step).collect(), Tz::Utc))).unwrap()
}

fn main() {
    println!("case,rows,min_us,allocations,bytes");
    for n in [10_000, 100_000, 1_000_000] {
        let fine = bars(n, 60_000_000_000);
        for tf in [TimeFrame::Min1, TimeFrame::Min15, TimeFrame::Day1] {
            measure(&format!("cumulate_{}", tf.label()), n, 1, || cumulate(&fine, tf, &AggSpec::ohlcv()).unwrap());
        }
    }
    for n in [10, 100, 1_000, 10_000] {
        let fine = bars(n, 1_000_000_000);
        let bar = fine.slice(n - 1, n);
        let mut live = Cumulator::new(TimeFrame::Day1, AggSpec::ohlcv());
        live.append(&fine).unwrap();
        measure("resend", n, 100, || live.append(&bar).unwrap());
        measure("last", n, 100, || live.last().unwrap());
    }
}
