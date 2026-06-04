//! Lightweight phase profiler for the directive hot path — no extra deps, builds
//! offline. Times `parse`, `execute`, and `parse+execute` (what `df.exec()` does
//! when it re-parses each call) for every benchmark indicator, so each perf round
//! can see where the Rust-side time goes and compare before/after.
//!
//! Run (release is essential):
//!   cargo run --release --example profile -p volas-directive [-- ITERS N]

use std::time::Instant;

use volas_compute::indicators as ind;
use volas_core::{Column, DataFrame};
use volas_directive::{execute, parse};

const DIRECTIVES: &[&str] = &[
    "ma:20", "ema:12", "macd", "macd.signal", "boll.upper", "bbw", "rsi:14", "atr:14", "llv:10",
    "hhv:10",
];

/// A deterministic pseudo-random OHLCV frame (a random walk), matching the
/// benchmark's 1999-row scale.
fn make_df(n: usize) -> DataFrame {
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut next = || {
        // xorshift64*
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        ((seed.wrapping_mul(0x2545F4914F6CDD1D) >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };
    let mut close = Vec::with_capacity(n);
    let mut x = 100.0;
    for _ in 0..n {
        x += next();
        close.push(x);
    }
    let open: Vec<f64> = close.iter().map(|&c| c - 0.1).collect();
    let high: Vec<f64> = close.iter().map(|&c| c + 0.5).collect();
    let low: Vec<f64> = close.iter().map(|&c| c - 0.5).collect();
    let vol: Vec<f64> = (0..n).map(|i| (i % 1000 + 1) as f64).collect();
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
            Column::f64(vol),
        ],
        None,
    )
    .unwrap()
}

/// Median microseconds per call of `f`, over `iters` timed runs (after warm-up).
fn median_us<T>(iters: usize, mut f: impl FnMut() -> T) -> f64 {
    for _ in 0..(iters / 10).max(50) {
        std::hint::black_box(f());
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        std::hint::black_box(f());
        samples.push(t.elapsed().as_nanos() as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2] / 1000.0
}

fn main() {
    let iters: usize = std::env::args()
        .skip_while(|a| a != "--iters")
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000);
    let df = make_df(1999);

    // raw kernel slices (borrowed, no per-call clone) to isolate kernel cost from
    // the directive layer's overhead (spec validation + input clone + result copy).
    let close = df.column("close").unwrap().as_f64().unwrap().to_vec();
    let high = df.column("high").unwrap().as_f64().unwrap().to_vec();
    let low = df.column("low").unwrap().as_f64().unwrap().to_vec();
    let kernel = |d: &str| -> Option<f64> {
        Some(match d {
            "ma:20" => median_us(iters, || ind::ma(&close, 20)),
            "ema:12" => median_us(iters, || ind::ema(&close, 12)),
            "macd" => median_us(iters, || ind::macd(&close, 12, 26)),
            "macd.signal" => median_us(iters, || ind::macd_signal(&close, 12, 26, 9)),
            "boll.upper" => median_us(iters, || ind::boll_upper(&close, 20, 2.0)),
            "bbw" => median_us(iters, || ind::bbw(&close, 20)),
            "rsi:14" => median_us(iters, || ind::rsi(&close, 14)),
            "atr:14" => median_us(iters, || ind::atr(&high, &low, &close, 14)),
            "llv:10" => median_us(iters, || ind::llv(&low, 10)),
            "hhv:10" => median_us(iters, || ind::hhv(&high, 10)),
            _ => return None,
        })
    };

    println!("phase profile  ({} rows, {} iters, median µs/call)", df.height(), iters);
    println!(
        "{:<14} | {:>7} | {:>8} | {:>8} | {:>12}",
        "directive", "parse", "kernel", "execute", "exec overhead"
    );
    println!("{}", "-".repeat(60));
    for d in DIRECTIVES {
        let node = parse(d).unwrap();
        let p = median_us(iters, || parse(std::hint::black_box(d)).unwrap());
        let e = median_us(iters, || execute(&df, std::hint::black_box(&node)).unwrap());
        let k = kernel(d).unwrap_or(f64::NAN);
        println!("{d:<14} | {p:>7.3} | {k:>8.3} | {e:>8.3} | {:>12.3}", e - k);
    }
}
