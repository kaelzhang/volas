//! TA-Lib's smoothing arithmetic — the one owner of how every EMA- and Wilder-family
//! recurrence rounds.
//!
//! Each step is TA-Lib's C source expression (0.6.4) evaluated as written, in plain
//! IEEE double arithmetic and never as a fused multiply-add: the result is the same on
//! every platform and bit-identical to an unfused TA-Lib build (Linux x86-64
//! manylinux). TA-Lib's own result depends on its compiler — clang contracts
//! `x*y + z` into an FMA on arm64 — so "TA-Lib's exact order" means the source order,
//! not a build's contraction of it. Rust never contracts on its own, so these plain
//! expressions compile to the separate multiply and add on every target, and an x86
//! build without a target FMA feature pays no software-`fma` call.
//!
//! Every full computation, carried-state capture and resume of the family (EMA, SMMA,
//! DEMA / TEMA / TRIX / T3, KAMA, MACD, RSI / CMO / StochRSI, ATR / NATR, ADX, ADOSC,
//! MAMA, and the EMA-built EFI / TSI / Mass Index / Keltner / …) steps through here, so
//! the paths cannot drift apart.

/// TA-Lib's EMA factor `PER_TO_K(period) = 2 / (period + 1)`.
#[inline(always)]
pub(crate) fn ema_k(period: usize) -> f64 {
    2.0 / (period as f64 + 1.0)
}

/// One EMA step `((x − prev)·k) + prev` (`ta_EMA.c`; DEMA, TEMA, TRIX, MACD run it
/// through `TA_INT_EMA`, KAMA with its adaptive constant as `k`).
#[inline(always)]
pub(crate) fn ema(prev: f64, x: f64, k: f64) -> f64 {
    (x - prev) * k + prev
}

/// TA-Lib's weighted EMA form `(k·x) + ((1 − k)·prev)` (`ta_T3.c`, `ta_ADOSC.c`,
/// `ta_MAMA.c`) — the same average as [`ema`], rounded differently, so each function
/// keeps the form its C source uses.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Weighted {
    k: f64,
    rest: f64,
}

impl Weighted {
    /// The form for the factor `k`, with TA-Lib's `one_minus_k = 1.0 − k`.
    #[inline(always)]
    pub(crate) fn new(k: f64) -> Self {
        Weighted { k, rest: 1.0 - k }
    }

    /// One step `(k·x) + ((1 − k)·prev)`.
    #[inline(always)]
    pub(crate) fn step(self, prev: f64, x: f64) -> f64 {
        self.k * x + self.rest * prev
    }
}

/// Wilder's average over `period` (`ta_ATR.c`, `ta_NATR.c`, `ta_RSI.c`, `ta_CMO.c`, the
/// ADX average of `ta_ADX.c`): per step `prev *= period − 1; prev += x; prev /= period`.
/// The division stays on the recurrence — `prev·((p − 1)/p) + x/p` rounds differently.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Wilder {
    period: f64,
    prior: f64,
}

impl Wilder {
    /// Wilder's average over `period` bars.
    #[inline(always)]
    pub(crate) fn new(period: usize) -> Self {
        Wilder { period: period as f64, prior: period as f64 - 1.0 }
    }

    /// One step `((prev·(period − 1)) + x) / period`.
    #[inline(always)]
    pub(crate) fn step(self, prev: f64, x: f64) -> f64 {
        (prev * self.prior + x) / self.period
    }

    /// One step of RSI / CMO's average gain and loss over the change `d` (`ta_RSI.c`,
    /// `ta_CMO.c`): a negative change is added (negated) to the loss, any other change to
    /// the gain, both averages stepped. TA-Lib adds nothing to the other side; adding `0`
    /// there is the same bits, as `prev·(period − 1)` is never `−0` (the averages start
    /// at `+0` and only grow by non-negative amounts).
    #[inline(always)]
    pub(crate) fn gain_loss(self, gain: f64, loss: f64, d: f64) -> (f64, f64) {
        let (up, down) = if d < 0.0 { (0.0, -d) } else { (d, 0.0) };
        (self.step(gain, up), self.step(loss, down))
    }
}

/// The seed of RSI / CMO's averages (`ta_RSI.c`, `ta_CMO.c`): the first `period`
/// changes `change(1) ..= change(period)` summed by side in row order, each sum divided
/// by the period. Returns `(gain, loss)`.
#[inline]
pub(crate) fn gain_loss_seed(period: usize, change: impl Fn(usize) -> f64) -> (f64, f64) {
    let (mut gain, mut loss) = (0.0, 0.0);
    for i in 1..=period {
        let d = change(i);
        if d < 0.0 {
            loss -= d;
        } else {
            gain += d;
        }
    }
    (gain / period as f64, loss / period as f64)
}

/// One step of Wilder's running *sum* `s − s/period + term` (`ta_PLUS_DM.c`,
/// `ta_DX.c`, `ta_ADX.c`; `period` as `f64`), and at period 1 the term itself —
/// TA-Lib's one-bar branch has no recurrence. The division stays: `s·(1 − 1/period) +
/// term` rounds differently and keeps an infinite sum infinite where TA-Lib's
/// `inf − inf` is NaN.
#[inline(always)]
pub(crate) fn wilder_sum(s: f64, period: f64, term: f64) -> f64 {
    if period == 1.0 { term } else { s - s / period + term }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The steps are the C expressions unfused: on operands where an FMA rounds
    /// differently, the result is the separately rounded product and sum.
    #[test]
    fn steps_round_the_product_and_the_sum_separately() {
        let (prev, x, k) = (0.1, 0.7, 2.0 / 3.0);
        let product = (x - prev) * k;
        assert_eq!(ema(prev, x, k), product + prev);
        assert_ne!(ema(prev, x, k), (x - prev).mul_add(k, prev), "operands must tell the two apart");
        let w = Weighted::new(k);
        assert_eq!(w.step(prev, x), (k * x) + ((1.0 - k) * prev));
        let avg = Wilder::new(3);
        assert_eq!(avg.step(prev, x), ((prev * 2.0) + x) / 3.0);
        assert_eq!(avg.gain_loss(1.0, 2.0, -0.5), (2.0 / 3.0, 4.5 / 3.0));
        assert_eq!(avg.gain_loss(1.0, 2.0, 0.5), (2.5 / 3.0, 4.0 / 3.0));
        assert_eq!(gain_loss_seed(2, |i| [0.0, 1.5, -0.5][i]), (0.75, 0.25));
        assert_eq!(wilder_sum(6.0, 3.0, 1.0), 5.0);
        assert_eq!(wilder_sum(6.0, 1.0, 1.0), 1.0);
        assert!(wilder_sum(f64::INFINITY, 3.0, 1.0).is_nan()); // inf − inf, as TA-Lib
    }
}
