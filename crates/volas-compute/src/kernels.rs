//! The numeric kernels, with the missing-row rule applied.
//!
//! Every function of [`crate::raw::kernels`] is here under the same name and
//! signature. A rolling-window one is that very function. A recursive one — the
//! SMA-seeded smoothers ([`ema_seeded`], [`wilder`], [`ema_diff_seeded`],
//! [`ewma_with_init`]) and the EMA cascade ([`ema_cascade`], its final state
//! [`ema_cascade_final`] and its step [`ema_cascade_step`]) — runs on the complete rows
//! of its input (see [`crate::missing`]): a missing (NaN) row outputs NaN, and the
//! recursion continues from the last complete row as if that row did not exist; a
//! missing row stepped over leaves the cascade where it was. With no row missing, the
//! input and the output pass through uncopied.

pub use crate::raw::kernels::*;


use crate::missing::{is_missing, on_complete_rows, state_on_complete_rows, Input};
use crate::raw::kernels as raw;

/// [`raw::wilder`] over the complete rows of `data`.
pub fn wilder(data: &[f64], period: usize) -> Vec<f64> {
    on_complete_rows(&[Input::new(data)], |rows| raw::wilder(rows[0], period))
}

/// [`raw::ema_seeded`] over the complete rows of `data`.
pub fn ema_seeded(data: &[f64], period: usize) -> Vec<f64> {
    on_complete_rows(&[Input::new(data)], |rows| raw::ema_seeded(rows[0], period))
}

/// [`raw::ema_diff_seeded`] over the complete rows of `data`.
pub fn ema_diff_seeded(data: &[f64], fast: usize, slow: usize) -> Vec<f64> {
    on_complete_rows(&[Input::new(data)], |rows| raw::ema_diff_seeded(rows[0], fast, slow))
}

/// [`raw::ewma_with_init`] over the complete rows of `data`.
pub fn ewma_with_init(data: &[f64], period: usize, init: f64) -> Vec<f64> {
    on_complete_rows(&[Input::new(data)], |rows| raw::ewma_with_init(rows[0], period, init))
}

/// [`raw::ema_cascade`] over the complete rows of `data`.
pub fn ema_cascade<const S: usize>(data: &[f64], period: usize) -> Vec<f64> {
    on_complete_rows(&[Input::new(data)], |rows| raw::ema_cascade::<S>(rows[0], period))
}

/// [`raw::ema_cascade_final`] over the complete rows of `data`: the cascade as of the
/// last complete row.
pub fn ema_cascade_final<const S: usize>(data: &[f64], period: usize) -> Option<[f64; S]> {
    state_on_complete_rows(&[data], |rows| raw::ema_cascade_final::<S>(rows[0], period))
}

/// [`raw::ema_cascade_step`], except that a missing `x` leaves the cascade `e` where it
/// was.
pub fn ema_cascade_step<const S: usize>(e: &mut [f64; S], x: f64, period: usize) {
    if !is_missing(x) {
        raw::ema_cascade_step(e, x, period);
    }
}
