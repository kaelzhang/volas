//! Instruction-level probe for the string hot kernel (driven by `make asm-diff`).
//!
//! `probe_str_eq_scan` is the shape of the elementwise string compare / filter loop:
//! one `StrBuffer::iter` walk with a per-cell equality. `make asm-diff` holds its
//! instruction count from rising — the per-element string cost must not regress.
use std::hint::black_box;
use volas_core::StrBuffer;

#[inline(never)]
#[no_mangle]
pub fn probe_str_eq_scan(s: &StrBuffer, target: &str) -> usize {
    s.iter().filter(|&x| x == target).count()
}

fn main() {
    let sb = StrBuffer::from_vec(vec!["ab".into(), "cd".into(), "ef".into()]);
    println!("{}", probe_str_eq_scan(black_box(&sb), black_box("cd")));
}
