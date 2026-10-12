use ndarray::Array1;

use super::av;
use crate::guard::{state_after, Slide};
use crate::raw::kernels;

// ---------------------------------------------------------------------------
// Support / resistance
// ---------------------------------------------------------------------------

/// Bollinger middle band (= MA).
pub fn boll(close: &[f64], period: usize, origin: usize) -> Vec<f64> {
    kernels::sma(av(close), period, origin).to_vec()
}

/// Bollinger upper band (`ma + times * std`, population std).
pub fn boll_upper(close: &[f64], period: usize, times: f64, origin: usize) -> Vec<f64> {
    let (ma, std) = kernels::rolling_mean_std(av(close), period, 0, origin);
    (&ma + times * &std).to_vec()
}

/// Bollinger lower band (`ma - times * std`, population std).
pub fn boll_lower(close: &[f64], period: usize, times: f64, origin: usize) -> Vec<f64> {
    let (ma, std) = kernels::rolling_mean_std(av(close), period, 0, origin);
    (&ma - times * &std).to_vec()
}

/// Bollinger Band Width (`4 * std / ma`).
pub fn bbw(close: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let (ma, std) = kernels::rolling_mean_std(av(close), period, 0, origin);
    (4.0 * &std / &ma).to_vec()
}

/// Historical Volatility.
pub fn hv(close: &[f64], period: usize, minutes: i64, trading_days: i64, origin: usize) -> Vec<f64> {
    let n = close.len();
    let mut log_return = Array1::from_elem(n, f64::NAN);
    for i in 1..n {
        if close[i - 1] > 0.0 && close[i] > 0.0 {
            log_return[i] = (close[i] / close[i - 1]).ln();
        }
    }
    let std = kernels::rolling_std(log_return.view(), period, 1, origin);
    let day_minutes = 1440.0;
    let annualization = ((trading_days as f64) * day_minutes / (minutes as f64)).sqrt();
    (&std * annualization).to_vec()
}

/// The raw (pre-SMA) Acceleration Band edge for each bar: `high·(1 + 4·(h-l)/(h+l))`
/// for the upper edge, `low·(1 − 4·(h-l)/(h+l))` for the lower. A zero `high+low`
/// falls back to the bare high/low (TA-Lib's guard).
fn accbands_raw(high: &[f64], low: &[f64], upper: bool) -> Vec<f64> {
    (0..high.len()).map(|i| accband_edge(high[i], low[i], upper)).collect()
}

/// One bar's raw Acceleration Band edge (see [`accbands_raw`]).
#[inline]
fn accband_edge(high: f64, low: f64, upper: bool) -> f64 {
    let sum = high + low;
    if sum.abs() < 1e-14 {
        if upper {
            high
        } else {
            low
        }
    } else {
        let factor = 4.0 * (high - low) / sum;
        if upper {
            high * (1.0 + factor)
        } else {
            low * (1.0 - factor)
        }
    }
}

/// Acceleration Bands middle (TA-Lib ACCBANDS middle) — the SMA of close, identical
/// to the Bollinger middle band. Lookback `period-1`.
pub fn accbands_middle(close: &[f64], period: usize, origin: usize) -> Vec<f64> {
    kernels::sma(av(close), period, origin).to_vec()
}

/// Acceleration Bands upper (TA-Lib ACCBANDS upper) — SMA of the raw upper edge.
pub fn accbands_upper(high: &[f64], low: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let raw = accbands_raw(high, low, true);
    kernels::sma(av(&raw), period, origin).to_vec()
}

/// Acceleration Bands lower (TA-Lib ACCBANDS lower) — SMA of the raw lower edge.
pub fn accbands_lower(high: &[f64], low: &[f64], period: usize, origin: usize) -> Vec<f64> {
    let raw = accbands_raw(high, low, false);
    kernels::sma(av(&raw), period, origin).to_vec()
}

// ---------------------------------------------------------------------------
// Resumable forms: the running sums of each full computation, in its exact operation
// order, so a resume of one row at a time is bit-identical to recomputing the column.
// ---------------------------------------------------------------------------

/// Which band a Bollinger-family command emits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BollLine {
    /// `ma + times·std` ([`boll_upper`]).
    Upper,
    /// `ma − times·std` ([`boll_lower`]).
    Lower,
    /// `4·std / ma` ([`bbw`]).
    Width,
}

/// [`boll_upper`] / [`boll_lower`] / [`bbw`]'s state after the rows of `close`:
/// the [`Slide`] of `[Σx, Σx²]` ([`kernels::rolling_mean_std`]).
pub fn boll_band_state(close: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<3>::initial(origin), close.len(), |row, prev, next| {
        boll_band_step(close, period, 2.0, BollLine::Width, row, prev, next)
    })
}

/// One row of the Bollinger `line` (band multiple `times`) at `row`, from `state` into
/// `next`.
pub fn boll_band_step(
    close: &[f64],
    period: usize,
    times: f64,
    line: BollLine,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if period == 0 || row >= close.len() || next.len() != kernels::MEAN_STD_STATE {
        return None;
    }
    let mut slide = Slide::<3>::from_state(state)?;
    let (ma, std) = kernels::mean_std_row(&mut slide, period, 0, row, |j| close[j]);
    slide.write_state(next);
    Some(match line {
        BollLine::Upper => ma + times * std,
        BollLine::Lower => ma - times * std,
        BollLine::Width => 4.0 * std / ma,
    })
}

/// The one-bar log return of `close` at `i` that [`hv`] takes the deviation of.
fn log_return(close: &[f64], i: usize) -> f64 {
    if i > 0 && close[i - 1] > 0.0 && close[i] > 0.0 {
        (close[i] / close[i - 1]).ln()
    } else {
        f64::NAN
    }
}

/// [`hv`]'s state after the rows of `close`: the deviation's [`Slide`] of `[Σr, Σr²]`.
pub fn hv_state(close: &[f64], period: usize, origin: usize) -> Option<Vec<f64>> {
    state_after(Slide::<3>::initial(origin), close.len(), |row, prev, next| {
        hv_step(close, period, 1, 1, row, prev, next)
    })
}

/// One row of [`hv`] at `row`, from `state` into `next`.
pub fn hv_step(
    close: &[f64],
    period: usize,
    minutes: i64,
    trading_days: i64,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if period <= 1 || row >= close.len() || next.len() != kernels::MEAN_STD_STATE {
        return None;
    }
    let mut slide = Slide::<3>::from_state(state)?;
    let (_, std) = kernels::mean_std_row(&mut slide, period, 1, row, |j| log_return(close, j));
    slide.write_state(next);
    let annualization = ((trading_days as f64) * 1440.0 / (minutes as f64)).sqrt();
    Some(std * annualization)
}

/// [`accbands_upper`] / [`accbands_lower`]'s state after the rows: the SMA's of the raw
/// edge (the middle band is [`super::ma_state`]'s).
pub fn accbands_band_state(
    high: &[f64],
    low: &[f64],
    period: usize,
    upper: bool,
    origin: usize,
) -> Option<Vec<f64>> {
    state_after(Slide::<1>::initial(origin), high.len(), |row, prev, next| {
        accbands_band_step(high, low, period, upper, row, prev, next)
    })
}

/// One row of the upper (`upper`) or lower acceleration band at `row`, from `state` into
/// `next`.
pub fn accbands_band_step(
    high: &[f64],
    low: &[f64],
    period: usize,
    upper: bool,
    row: usize,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if row >= high.len() {
        return None;
    }
    kernels::sma_step(period, row, |j| accband_edge(high[j], low[j], upper), state, next)
}
