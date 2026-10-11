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

use ndarray::{Array1, ArrayView1};

use crate::missing::{is_missing, on_complete_rows, state_on_complete_rows, Input};
use crate::raw::kernels as raw;

/// Run the array kernel `kernel` on the complete rows of `data`.
fn on_complete_view(
    data: ArrayView1<f64>,
    kernel: impl FnOnce(ArrayView1<f64>) -> Array1<f64>,
) -> Array1<f64> {
    let owned;
    let values = match data.as_slice() {
        Some(values) => values,
        None => {
            owned = data.to_vec(); // a strided view: one contiguous copy
            &owned
        }
    };
    let out = on_complete_rows(&[Input::new(values)], |rows| {
        kernel(ArrayView1::from(rows[0])).into_raw_vec_and_offset().0
    });
    Array1::from(out)
}

/// [`raw::wilder`] over the complete rows of `data`.
pub fn wilder(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    on_complete_view(data, |x| raw::wilder(x, period))
}

/// [`raw::ema_seeded`] over the complete rows of `data`.
pub fn ema_seeded(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
    on_complete_view(data, |x| raw::ema_seeded(x, period))
}

/// [`raw::ema_diff_seeded`] over the complete rows of `data`.
pub fn ema_diff_seeded(data: ArrayView1<f64>, fast: usize, slow: usize) -> Array1<f64> {
    on_complete_view(data, |x| raw::ema_diff_seeded(x, fast, slow))
}

/// [`raw::ewma_with_init`] over the complete rows of `data`.
pub fn ewma_with_init(data: ArrayView1<f64>, period: usize, init: f64) -> Array1<f64> {
    on_complete_view(data, |x| raw::ewma_with_init(x, period, init))
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
pub fn ema_cascade_step<const S: usize>(e: &mut [f64; S], x: f64, k: f64) {
    if !is_missing(x) {
        raw::ema_cascade_step(e, x, k);
    }
}
