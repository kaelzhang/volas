//! Instruction-level probe for the numeric hot kernels (driven by `make asm-diff`).
//!
//! Each `#[no_mangle] #[inline(never)]` wrapper forces the kernel to be emitted as a
//! standalone symbol so `scripts/asm_diff.sh` can disassemble and count its
//! instructions, holding the count byte-stable against `scripts/asm_baseline.txt`.
use std::hint::black_box;
use volas_compute::indicators::{ema, ma};

#[inline(never)]
#[no_mangle]
pub fn probe_ma(close: &[f64], period: usize) -> Vec<f64> {
    ma(close, period)
}

#[inline(never)]
#[no_mangle]
pub fn probe_ema(close: &[f64], period: usize) -> Vec<f64> {
    ema(close, period)
}

fn main() {
    let v: Vec<f64> = (0..64).map(|x| x as f64).collect();
    let a = probe_ma(black_box(&v), black_box(5));
    let b = probe_ema(black_box(&v), black_box(5));
    println!("{} {}", a.len(), b.len());
}
