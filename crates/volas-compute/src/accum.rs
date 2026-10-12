//! Exact running sums for sliding-window kernels.
//!
//! A sliding kernel keeps running sums — add the entering row, subtract the leaving
//! one — for ever. Two things go wrong with a plain `f64` accumulator, and this module
//! owns the fix for both:
//!
//! - **Rounding is kept for good.** Every add rounds, and the error stays in the sum
//!   after the row has left: after `1e20` enters and leaves, the window `[1, 2]` reads
//!   `0`, not `3`. A window's sum here is **exact**: its value is the correctly rounded
//!   exact sum of the window's terms — what `math.fsum` returns — however the window
//!   was reached.
//! - **A value depends on the whole history.** A running sum's words are a function of
//!   every row it has slid over — but its *value* is not: an exact sum is the window's
//!   own, so every value read from it (`value`, `mean`, the centred moments) is a
//!   function of the window's terms alone, whatever the words. A first-moment slide is
//!   therefore re-summed only where its compensation could not hold the sum. A
//!   **re-anchor** — recomputing the sums from the window's own rows at each row whose
//!   absolute position is a multiple of the window length `W` — is where a shifted
//!   second moment changes its shift (see "second moments" below), and the careful
//!   (row-guarded) slide of [`crate::guard`] re-anchors there on every kernel; either way
//!   a re-anchoring costs `O(W)` once per `W` rows, amortized `O(1)` per row.
//!
//! **How a sum stays exact.** A [`Csum`] is two words, `sum + comp`, updated by
//! error-free transformations: each add splits `sum + x` into its rounded value and
//! its exact rounding error (TwoSum), and the error joins `comp`. The only add that can
//! round is that last one, `comp + e`, and it is **checked** — `(c − comp == e) && (c −
//! e == comp)` holds exactly when `c = comp + e` is exact (one of the two differences
//! is computed exactly by Sterbenz / Fast2Sum, so it equals the other operand only if
//! nothing was lost). So, by induction, while [`Csum::is_exact`] holds, `sum + comp` is
//! the exact sum of every term added and subtracted, and [`Csum::value`] rounds that
//! real number once: the correctly rounded window sum. When a check fails — terms whose
//! magnitudes differ by more than `comp` can hold beside `sum` (`1e35`, `1e20` and
//! `1.3` in one window), or an overflow — the sum is marked inexact, and the kernel
//! **re-anchors at that row**, as at a scheduled anchor: [`Csum::of`] re-sums the
//! window, and when even the fast form cannot hold that sum exactly it falls back to
//! an [`Exact`] expansion (Shewchuk's partials, `fsum`'s algorithm), which never
//! rounds. Its value is rounded once, correctly (with `fsum`'s half-even fix-up), and
//! kept as a canonical pair `(hi, lo)`: `hi` the correctly rounded sum and `lo` the
//! remainder, so the next row's reading and mean depend only on the exact sum. A sum
//! that still needs more than two words stays marked inexact, so every row re-sums
//! its window — `O(W)` per row — until the extreme term has left; ordinary data never
//! takes that path.
//!
//! Every value read is therefore a function of the window's terms alone: the batch
//! kernel, a resume from carried state, the live one-row step and a bounded frame read
//! the same bits. The one residue is outside `fsum`'s own domain: when an add of the
//! window's exact re-sum overflows (oldest term first — exactly where `fsum` raises
//! `OverflowError`), the expansion is scaled by `2⁻⁶⁴` and the sum continues there.
//! That scaling is exact for every term of magnitude at least `2⁻⁹⁵⁷` (their lowest
//! bit is at least `2⁻¹⁰¹⁰`, still on the scaled grid), and then so is the rounded
//! sum; only a nonzero term below `2⁻⁹⁵⁷` beside such an overflow can lose its low
//! bits (each further `2⁻⁶⁴` step moves that bound up by 64 bits). A window whose adds
//! never overflow is summed unscaled, whatever its magnitudes.
//!
//! The absolute position is the row's place in the series the frame continues (see
//! `DataFrame::origin`): a bounded frame that dropped its head, or a contiguous slice,
//! keeps its rows' positions, so its anchors fall where the full series' do. A carried
//! state holds its next row's position, so a resume anchors where a full computation
//! does — every path of a kernel does the same arithmetic on the same rows.

/// Something terms can be summed into: the fast checked [`Csum`] or the exact
/// [`Exact`] expansion. A kernel's re-summing helper is written once, generic over it,
/// and runs the fast form first.
pub(crate) trait Accumulator: Default {
    /// Add `x`.
    fn add(&mut self, x: f64);

    /// Add the exact product `a·b`: its rounded value and its rounding error (by FMA),
    /// so a product added here and later removed leaves nothing.
    #[inline(always)]
    fn add_prod(&mut self, a: f64, b: f64) {
        let p = a * b;
        self.add(p);
        self.add(a.mul_add(b, -p));
    }
}

/// A compensated running sum: two words `sum + comp` updated by error-free transforms,
/// with every rounding of `comp` detected (see the module docs). While
/// [`is_exact`](Self::is_exact) holds, `sum + comp` is the exact sum of every term; a
/// non-finite term or an overflow marks it inexact (and its value non-finite).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Csum {
    sum: f64,
    comp: f64,
    exact: bool,
}

impl Default for Csum {
    #[inline(always)]
    fn default() -> Self {
        Csum { sum: 0.0, comp: 0.0, exact: true }
    }
}

impl Accumulator for Csum {
    #[inline(always)]
    fn add(&mut self, x: f64) {
        Csum::add(self, x);
    }
}

impl Csum {
    /// The length of a compensated sum's state: `[sum, comp]`. A sum that is not exact
    /// is carried as `[sum, NaN]` — it is never read again, the next full window
    /// re-sums itself (see [`from_state`](Self::from_state)).
    pub(crate) const STATE: usize = 2;

    /// The exact sum of `terms`, added in order: the fast form when it holds the sum
    /// exactly (the common case), else the canonical pair of an [`Exact`] expansion
    /// (exact unless the sum needs more than two words — then marked inexact, its
    /// value still the correctly rounded sum).
    #[inline]
    pub(crate) fn of<I>(terms: I) -> Self
    where
        I: IntoIterator<Item = f64>,
        I::IntoIter: Clone,
    {
        let [s] = Csum::of_rows(terms.into_iter().map(|x| [x]));
        s
    }

    /// [`of`](Self::of) for `K` sums at once, from rows of `K` terms each — one pass
    /// over the rows in the common case.
    #[inline]
    pub(crate) fn of_rows<const K: usize, I>(rows: I) -> [Self; K]
    where
        I: IntoIterator<Item = [f64; K]>,
        I::IntoIter: Clone,
    {
        let rows = rows.into_iter();
        let mut sums = [Csum::default(); K];
        for t in rows.clone() {
            for (s, x) in sums.iter_mut().zip(t) {
                s.add(x);
            }
        }
        if !sums.iter().all(Csum::is_exact) {
            resum_exactly(&mut sums, rows);
        }
        sums
    }

    /// The exact sums of a re-summing helper written once over [`Accumulator`]: `fast`
    /// its result with compensated sums, `slow` the same helper over [`Exact`]
    /// expansions — run only when a fast sum could not hold its sum exactly.
    #[inline]
    pub(crate) fn settle<const K: usize>(fast: [Csum; K], slow: impl FnOnce() -> [Exact; K]) -> [Self; K] {
        if fast.iter().all(Csum::is_exact) {
            return fast;
        }
        let slow = slow();
        std::array::from_fn(|k| if fast[k].exact { fast[k] } else { slow[k].to_csum() })
    }

    /// Add `x`: TwoSum of `sum + x`, its exact error into `comp`, that add checked.
    #[inline(always)]
    pub(crate) fn add(&mut self, x: f64) {
        let t = self.sum + x;
        let bv = t - self.sum;
        let e = (self.sum - (t - bv)) + (x - bv);
        let c = self.comp + e;
        self.exact &= (c - self.comp == e) & (c - e == self.comp);
        self.sum = t;
        self.comp = c;
    }

    /// Subtract `x` (add `-x`).
    #[inline(always)]
    pub(crate) fn sub(&mut self, x: f64) {
        self.add(-x);
    }

    /// Add another compensated sum — both its words, so none of its low-order bits is
    /// lost to rounding its value first. Exact only if `other` is.
    #[inline(always)]
    pub(crate) fn add_sum(&mut self, other: &Csum) {
        self.add(other.sum);
        self.add(other.comp);
        self.exact &= other.exact;
    }

    /// Subtract another compensated sum — both its words. Exact only if `other` is.
    #[inline(always)]
    pub(crate) fn sub_sum(&mut self, other: &Csum) {
        self.sub(other.sum);
        self.sub(other.comp);
        self.exact &= other.exact;
    }

    /// Add the exact product `a·b` (see [`Accumulator::add_prod`]).
    #[inline(always)]
    pub(crate) fn add_prod(&mut self, a: f64, b: f64) {
        Accumulator::add_prod(self, a, b);
    }

    /// Subtract the exact product `a·b`.
    #[inline(always)]
    pub(crate) fn sub_prod(&mut self, a: f64, b: f64) {
        self.add_prod(-a, b);
    }

    /// Whether `sum + comp` is the exact sum of every term — else the window must be
    /// re-summed before it is read.
    #[inline(always)]
    pub(crate) fn is_exact(&self) -> bool {
        self.exact
    }

    /// The canonical words `(hi, lo)` of the sum: `hi` its value, `lo` the remainder —
    /// exactly `sum + comp − hi` (TwoSum), so a function of the sum alone, not of how
    /// the two words split it.
    #[inline(always)]
    pub(crate) fn words(&self) -> (f64, f64) {
        two_sum(self.sum, self.comp)
    }

    /// The sum's value: `sum + comp`, rounded once — the correctly rounded exact sum
    /// when [`is_exact`](Self::is_exact) holds or the sum came from [`of`](Self::of).
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
        mean_of_words(self.words(), n)
    }

    /// A sum from its state `[sum, comp]`; `None` unless exactly two words. It is exact
    /// when both words are finite (an inexact sum is written with a NaN low word).
    pub(crate) fn from_state(state: &[f64]) -> Option<Self> {
        let [sum, comp] = *state else { return None };
        Some(Csum { sum, comp, exact: sum.is_finite() && comp.is_finite() })
    }

    /// Write the state `[sum, comp]` into `out[..2]` — `[sum, NaN]` for an inexact sum.
    pub(crate) fn write_state(&self, out: &mut [f64]) {
        out[0] = self.sum;
        out[1] = if self.exact { self.comp } else { f64::NAN };
    }
}

/// [`Csum::of_rows`]'s slow path: re-sum each inexact sum as an [`Exact`] expansion.
#[cold]
#[inline(never)]
fn resum_exactly<const K: usize>(sums: &mut [Csum; K], rows: impl Iterator<Item = [f64; K]> + Clone) {
    for (k, s) in sums.iter_mut().enumerate() {
        if !s.exact {
            let mut e = Exact::default();
            for t in rows.clone() {
                e.add(t[k]);
            }
            *s = e.to_csum();
        }
    }
}

/// `a + b` as its rounded value and its exact rounding error (Knuth's TwoSum).
#[inline(always)]
pub(crate) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bv = s - a;
    (s, (a - (s - bv)) + (b - bv))
}

/// `a·b` as its rounded value and its exact rounding error (by FMA).
#[inline(always)]
pub(crate) fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

// --- second moments from shifted data --------------------------------------------
//
// A variance, covariance or regression slope is a difference of raw moments —
// `Σx² − (Σx)²/n` — that cancels catastrophically when the window's level dwarfs its
// spread (at `12345.67` a variance of `1e-12` is pure rounding noise). The standard cure
// is to shift the data by a value `K` inside the window: the deviations `d = x − K` are
// of the window's own spread, so `Σd² − (Σd)²/n` carries no cancellation beyond a
// factor of the window length. The shift is the value at the window's **anchor row** —
// the row whose absolute position is the multiple of the window length `W` inside the
// window ([`shift_row`]): any `W` consecutive positions hold exactly one, a sliding sum
// re-anchors there already (and so rebuilds its sums with the new shift), and it is a
// function of the absolute position alone, so every path of a kernel shifts by the same
// value. `Σx` itself stays an exact sum (means and bands keep their exact values); the
// linear deviation `Σd = Σx − n·K` is taken from its exact words ([`deviation`]).
//
// Accuracy (`u = 2⁻⁵³`, `n` rows, `R` the window's range, `C` its exact centred sum of
// squares `Σ(x − x̄)²`): each `d` is rounded once (`|δ| ≤ u·R`, as `|x − K| ≤ R`), each
// `d²` once, their sum is exact (then rounded once), and `Σd` is within `3u|Σd| +
// 2u²n|K|` of exact — so the computed `C` is within `≈ 12·n·u·R²` of exact, and since a
// window spanning `R` has `C ≥ R²/2`, within `(24n + 2)·u` **relative** (`R ≥ 2u|K|`
// for any two distinct values). A flat window has `d = 0` and `Σd = 0` exactly: `C` is
// exactly `0`. A covariance is bounded the same way against `sqrt(Cxx·Cyy)`.

/// The row (index into the rows of the series, row `i` at absolute position `pos`) whose
/// value shifts a `period`-row window ending at row `i`: its anchor row, the one whose
/// absolute position is a multiple of `period`. A window that starts before the series
/// does (the warm-up of a series whose first position is not a multiple) uses its first
/// row until the first anchor arrives (a re-anchoring row, which rebuilds the sums).
#[inline(always)]
pub(crate) fn shift_row(i: usize, pos: usize, period: usize) -> usize {
    i.saturating_sub(pos % period)
}

/// `Σx − n·K` from an exact sum `Σx`: its canonical words less the exact product `n·K`
/// ([`two_prod`]) — the deviations' sum without the cancellation of rounding `Σx`
/// first. Exactly `0` for `n` copies of `K`.
#[inline(always)]
pub(crate) fn deviation(sum: &Csum, n: f64, k: f64) -> f64 {
    let (hi, lo) = sum.words();
    let (p, e) = two_prod(n, k);
    (hi - p) + (lo - e)
}

/// The quotient of a sum's canonical words `(hi, lo)` by `n` (see [`Csum::mean`]): the
/// quotient of `hi`, corrected by the exact remainder (FMA) and the low word.
#[inline(always)]
fn mean_of_words((hi, lo): (f64, f64), n: f64) -> f64 {
    // The reciprocal is loop-invariant where `n` is (it hoists out of a kernel's loop):
    // `q` is within an ulp or two of the quotient, the remainder `r` is exact to the
    // last bits (FMA), and `q + r·(1/n)` lands on the correctly rounded quotient but in
    // the rare near-tie — a flat window's `n` equal values give exactly their value.
    let inv = 1.0 / n;
    let q = hi * inv;
    let r = (-q).mul_add(n, hi) + lo;
    q + r * inv
}

/// A window's mean and centred sum of squares `Σ(x − x̄)²` from its exact sums `Σx`
/// and `Σ(x − k)²` — [`Csum::mean`] and [`centred`] from one normalization of `Σx`.
#[inline(always)]
pub(crate) fn spread(n: f64, sum: &Csum, k: f64, sum_sq: &Csum) -> (f64, f64) {
    let words = sum.words();
    let (p, e) = two_prod(n, k);
    let d = (words.0 - p) + (words.1 - e);
    (mean_of_words(words, n), sum_sq.value() - d * d / n)
}

/// The centred cross moment `Σ(a − ā)(b − b̄)` of a window of `n` rows from its exact
/// sums `Σa`, `Σb` (shifted by `ka`, `kb`) and the exact sum of the shifted products
/// `Σ(a − ka)(b − kb)`: `Σdadb − Σda·Σdb/n` (with `a = b`, the centred sum of squares).
#[inline(always)]
pub(crate) fn centred(n: f64, (sa, ka): (&Csum, f64), (sb, kb): (&Csum, f64), sum_dd: &Csum) -> f64 {
    sum_dd.value() - deviation(sa, n, ka) * deviation(sb, n, kb) / n
}

/// `2⁻⁶⁴`: the step an [`Exact`] expansion scales down by when one of its adds
/// overflows.
const RESCALE: f64 = 1.0 / 18_446_744_073_709_551_616.0;

/// An exact sum of any number of `f64` terms: a nonoverlapping expansion (Shewchuk's
/// partials, the algorithm of Python's `math.fsum`), grown by error-free additions so
/// it never rounds. Non-finite terms are summed apart, IEEE-style (`fsum`'s rule:
/// NaN, or ±inf, or NaN for both infinities). Where an add overflows (where `fsum`
/// raises) the expansion is scaled by `2⁻⁶⁴` and continues (see the module docs). The
/// slow, exact form behind [`Csum::of`]; also the running sum of an expanding window,
/// which never drops a term.
#[derive(Clone, Debug, Default)]
pub(crate) struct Exact {
    /// Nonoverlapping words, increasing in magnitude, each scaled by `2^-scale`.
    partials: Vec<f64>,
    /// The sum of the non-finite terms (`0` when there are none).
    special: f64,
    /// The scaling exponent, a multiple of 64.
    scale: i32,
}

impl Accumulator for Exact {
    #[inline]
    fn add(&mut self, x: f64) {
        Exact::add(self, x);
    }
}

/// `2^-scale` for a scale in `0..1023`.
#[inline]
fn pow2_neg(scale: i32) -> f64 {
    f64::from_bits(((1023 - scale) as u64) << 52)
}

impl Exact {
    /// Add `x` exactly.
    pub(crate) fn add(&mut self, x: f64) {
        if !x.is_finite() {
            self.special += x;
            return;
        }
        let x = if self.scale == 0 { x } else { x * pow2_neg(self.scale) };
        if let Err(words) = self.grow(x) {
            // An add overflowed — the point where `fsum` raises. The words still sum
            // exactly to the new total: scale them down and grow them again.
            self.scale += 64;
            self.partials.clear();
            for w in words {
                let scaled = w * RESCALE;
                if scaled != 0.0 {
                    // a few words of magnitude below 2^960 cannot overflow
                    self.grow(scaled).expect("a total scaled by 2^-64 does not overflow");
                }
            }
        }
    }

    /// Grow the expansion by `x` (Shewchuk's `GROW-EXPANSION`, as `fsum`). If an add
    /// overflows, the expansion is left empty and `Err` holds finite words whose exact
    /// sum is the old total plus `x`.
    fn grow(&mut self, mut x: f64) -> Result<(), Vec<f64>> {
        let mut i = 0;
        for j in 0..self.partials.len() {
            let mut y = self.partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            if !hi.is_finite() {
                let mut words: Vec<f64> = self.partials[..i].to_vec();
                words.push(x);
                words.push(y);
                words.extend_from_slice(&self.partials[j + 1..]);
                self.partials.clear();
                return Err(words);
            }
            let lo = y - (hi - x);
            if lo != 0.0 {
                self.partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        self.partials.truncate(i);
        self.partials.push(x);
        Ok(())
    }

    /// `fsum`'s final rounding of the expansion `p` (round-half-even included):
    /// `(hi, n, lo)` with `sum(p) = hi + lo + sum(p[..n])` exactly and `hi` the
    /// correctly rounded sum.
    fn round(p: &[f64]) -> (f64, usize, f64) {
        let Some((&top, below)) = p.split_last() else { return (0.0, 0, 0.0) };
        let mut n = below.len();
        let mut hi = top;
        let mut lo = 0.0;
        while n > 0 {
            let x = hi;
            n -= 1;
            let y = p[n];
            hi = x + y;
            lo = y - (hi - x);
            if lo != 0.0 {
                break;
            }
        }
        // Half-even fix-up: a tie in the last add that the lower words break.
        if n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0)) {
            let y = lo * 2.0;
            let x = hi + y;
            if y == x - hi {
                hi = x;
                lo -= y;
            }
        }
        (hi, n, lo)
    }

    /// The sum as a canonical [`Csum`]: `hi` the correctly rounded sum, `lo` the
    /// remainder rounded to nearest (nudged one ulp toward zero when that rounding
    /// would make `hi + lo` round away from `hi`), exact when `hi + lo` is the sum. A
    /// function of the exact sum alone, so every path that reaches the same window
    /// reads the same bits.
    pub(crate) fn to_csum(&self) -> Csum {
        if self.special != 0.0 {
            return Csum { sum: self.special, comp: 0.0, exact: false };
        }
        let (hi, n, rest) = Exact::round(&self.partials);
        let mut r = Exact::default();
        for &w in self.partials[..n].iter().chain([&rest]) {
            r.add(w);
        }
        let (mut lo, ..) = Exact::round(&r.partials);
        r.add(-lo);
        let exact = self.scale == 0 && r.partials.iter().all(|&w| w == 0.0);
        if hi + lo != hi {
            lo = f64::from_bits(lo.to_bits() - 1); // one ulp toward zero
        }
        if self.scale == 0 {
            Csum { sum: hi, comp: lo, exact }
        } else {
            let up = 1.0 / pow2_neg(self.scale);
            let hi = hi * up;
            Csum { sum: hi, comp: if hi.is_finite() { lo * up } else { 0.0 }, exact }
        }
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
/// when told to (the caller's [`Anchors`] tick on a full window, `i + 1 >= W`) or when
/// the sum went inexact, and read the mean once `W` values are in. Bit-identical to
/// `sma` over the same series.
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
        if anchor || !self.sum.is_exact() {
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
mod tests;
