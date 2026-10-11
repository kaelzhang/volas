//! Exact running sums for sliding-window kernels.
//!
//! A sliding kernel keeps running sums — add the entering row, subtract the leaving
//! one — for ever. Two things go wrong with a plain `f64` accumulator, and this module
//! owns the fix for both:
//!
//! - **Rounding is kept for good.** Every add rounds, and the error stays in the sum
//!   after the row has left: after `1e20` enters and leaves, the window `[1, 2]` reads
//!   `0`, not `3`. A [`Csum`] is compensated (Neumaier's improved Kahan–Babuška): it
//!   carries the rounding error of every add, so a window's sum is its exact sum to
//!   within an ulp, and [`Csum::mean`] divides that exact sum, not its rounded value —
//!   the mean of a flat window is its value exactly.
//! - **A value depends on the whole history.** Even compensated, what the sum holds is
//!   a function of every row it has slid over (the compensation itself rounds once its
//!   bits run out), so a series that differs only long ago — a NaN that has since left
//!   the window, a head dropped by a bounded frame — could still differ in the last
//!   bit. So every running sum **re-anchors**: at each row whose absolute position is
//!   a multiple of the window length `W`, it is recomputed from the window's own rows
//!   (a fresh [`Csum`] over them, oldest first). A value then depends only on the rows
//!   since the anchor before it, less `W` — at most `2W` rows — never on older
//!   history: once a bad row has left the window and an anchor has passed, the values
//!   are bit-identical to clean data's. Re-anchoring costs `O(W)` once per `W` rows,
//!   amortized `O(1)` per row.
//!
//! The absolute position is the row's place in the series the frame continues (see
//! `DataFrame::origin`): a bounded frame that dropped its head, or a contiguous slice,
//! keeps its rows' positions, so its anchors fall where the full series' do. A carried
//! state holds its next row's position, so a resume anchors where a full computation
//! does — every path of a kernel does the same arithmetic on the same rows.

/// A compensated running sum — Neumaier's improved Kahan–Babuška summation. `sum` is
/// the rounded running total and `comp` the accumulated rounding error of every add
/// (each one exact, by Fast2Sum on the larger magnitude first); the sum's value is
/// `sum + comp`. A non-finite term makes the value NaN for good (`comp` takes
/// `inf − inf`), which the careful slides test for.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Csum {
    sum: f64,
    comp: f64,
}

impl Csum {
    /// The length of a compensated sum's state: `[sum, comp]`.
    pub(crate) const STATE: usize = 2;

    /// The compensated sum of `terms`, added in order.
    #[inline]
    pub(crate) fn of(terms: impl IntoIterator<Item = f64>) -> Self {
        let mut s = Csum::default();
        for x in terms {
            s.add(x);
        }
        s
    }

    /// Add `x`.
    #[inline(always)]
    pub(crate) fn add(&mut self, x: f64) {
        let t = self.sum + x;
        self.comp += if self.sum.abs() >= x.abs() {
            (self.sum - t) + x
        } else {
            (x - t) + self.sum
        };
        self.sum = t;
    }

    /// Subtract `x` (add `-x`).
    #[inline(always)]
    pub(crate) fn sub(&mut self, x: f64) {
        self.add(-x);
    }

    /// Add another compensated sum — both its words, so none of its low-order bits is
    /// lost to rounding its value first.
    #[inline(always)]
    pub(crate) fn add_sum(&mut self, other: &Csum) {
        self.add(other.sum);
        self.add(other.comp);
    }

    /// Subtract another compensated sum — both its words.
    #[inline(always)]
    pub(crate) fn sub_sum(&mut self, other: &Csum) {
        self.sub(other.sum);
        self.sub(other.comp);
    }

    /// Add the exact product `a·b` (its rounded value and its rounding error, by FMA),
    /// so a product added here and later removed as a sum of its factor leaves nothing.
    #[inline(always)]
    pub(crate) fn add_prod(&mut self, a: f64, b: f64) {
        let p = a * b;
        self.add(p);
        self.add(a.mul_add(b, -p));
    }

    /// Subtract the exact product `a·b` (see [`add_prod`](Self::add_prod)).
    #[inline(always)]
    pub(crate) fn sub_prod(&mut self, a: f64, b: f64) {
        self.add_prod(-a, b);
    }

    /// The sum's value: `sum + comp`, rounded once.
    #[inline(always)]
    pub(crate) fn value(&self) -> f64 {
        self.sum + self.comp
    }

    /// The sum divided by `n`, from both its words: the quotient of the value, then
    /// corrected by the exact remainder (FMA) and the low word — so a window of `n`
    /// equal values has exactly that value as its mean, where `value() / n` can miss
    /// it by an ulp. The words are normalized first (TwoSum), so the result depends
    /// only on the sum they represent, not on how it was split.
    #[inline(always)]
    pub(crate) fn mean(&self, n: f64) -> f64 {
        let hi = self.sum + self.comp;
        let b = hi - self.sum;
        let lo = (self.sum - (hi - b)) + (self.comp - b);
        let q = hi / n;
        let r = (-q).mul_add(n, hi) + lo;
        q + r / n
    }

    /// A sum from its state `[sum, comp]`; `None` unless exactly two words.
    pub(crate) fn from_state(state: &[f64]) -> Option<Self> {
        let [sum, comp] = *state else { return None };
        Some(Csum { sum, comp })
    }

    /// Write the state `[sum, comp]` into `out[..2]`.
    pub(crate) fn write_state(&self, out: &mut [f64]) {
        out[0] = self.sum;
        out[1] = self.comp;
    }
}

/// Whether the row at absolute position `pos` re-anchors a running sum over a window
/// of `period` rows: its position is a multiple of the window length.
#[inline(always)]
pub(crate) fn anchors_at(pos: usize, period: usize) -> bool {
    period != 0 && pos.is_multiple_of(period)
}

/// The re-anchoring schedule of a running sum over `period`-row windows, row by row
/// from a given absolute position — [`anchors_at`] without a division per row.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Anchors {
    period: usize,
    left: usize,
}

impl Anchors {
    /// The schedule for rows from absolute position `pos` on (`period > 0`).
    #[inline]
    pub(crate) fn from(pos: usize, period: usize) -> Self {
        let r = pos % period;
        Anchors { period, left: if r == 0 { 0 } else { period - r } }
    }

    /// Advance one row: whether it re-anchors ([`anchors_at`] of its position).
    #[inline(always)]
    pub(crate) fn tick(&mut self) -> bool {
        let hit = self.left == 0;
        self.left = if hit { self.period - 1 } else { self.left - 1 };
        hit
    }
}

/// `kernels::sma`'s fast path over a series that arrives one value at a time from its
/// first finite value on — for fused kernels that smooth a line they produce
/// themselves. A `W`-row window: add the new value, drop the one `W` back, re-anchor
/// when told to (the caller's [`Anchors`] tick on a full window, `i + 1 >= W`), and
/// read the mean once `W` values are in. Bit-identical to `sma` over the same series.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StreamMean<const W: usize> {
    ring: [f64; W],
    count: usize,
    sum: Csum,
    clean: bool,
}

impl<const W: usize> StreamMean<W> {
    /// No values yet.
    pub(crate) fn new() -> Self {
        StreamMean { ring: [0.0; W], count: 0, sum: Csum::default(), clean: true }
    }

    /// Take the next value; `anchor` re-sums the window from its own values. The mean
    /// once the window is full.
    #[inline(always)]
    pub(crate) fn push(&mut self, x: f64, anchor: bool) -> Option<f64> {
        let slot = self.count % W;
        self.sum.add(x);
        if self.count >= W {
            self.sum.sub(self.ring[slot]);
        }
        self.ring[slot] = x;
        self.count += 1;
        if anchor {
            self.clean &= self.sum.value().is_finite();
            let m = self.count.min(W);
            self.sum = Csum::of((0..m).map(|k| self.ring[(self.count - m + k) % W]));
        }
        (self.count >= W).then(|| self.sum.mean(W as f64))
    }

    /// Whether the sum never went non-finite (the fast path's validity).
    pub(crate) fn finite(&self) -> bool {
        self.clean && self.sum.value().is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_compensated_sum_forgets_a_spike() {
        let mut s = Csum::default();
        s.add(0.1);
        s.add(1e300);
        s.sub(1e300);
        assert_eq!(s.value(), 0.1);
        let mut st = [0.0; 2];
        s.write_state(&mut st);
        assert_eq!(Csum::from_state(&st), Some(s));
        assert_eq!(Csum::from_state(&[1.0]), None);
        s.add(f64::INFINITY);
        assert!(s.value().is_nan());
    }

    #[test]
    fn a_flat_window_has_its_value_as_mean() {
        for c in [0.1, 100.1, 1234.5678, 1e-7 / 3.0, 7.0 / 3.0] {
            for n in 1..60 {
                let s = Csum::of(std::iter::repeat_n(c, n));
                assert_eq!(s.mean(n as f64), c, "{c} x{n}");
            }
        }
        // the mean does not depend on how the words split the sum
        let a = Csum { sum: 3.0, comp: 0.5 };
        let b = Csum { sum: 3.5, comp: 0.0 };
        assert_eq!(a.mean(7.0), b.mean(7.0));
        let mut p = Csum::default();
        p.add_prod(3.0, 0.1);
        p.sub_sum(&Csum::of([0.1, 0.1, 0.1]));
        assert_eq!(p.value(), 0.0);
        p.add_sum(&Csum::of([1.0]));
        p.sub_prod(2.0, 0.5);
        assert_eq!(p.value(), 0.0);
    }

    #[test]
    fn anchors_fall_on_multiples_of_the_period() {
        for (from, period) in [(0, 3), (5, 3), (6, 3), (7, 1), (11, 4)] {
            let mut a = Anchors::from(from, period);
            for pos in from..from + 20 {
                assert_eq!(a.tick(), anchors_at(pos, period), "{from} {period} {pos}");
            }
        }
        assert!(!anchors_at(5, 0));
    }
}
