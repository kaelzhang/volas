//! Window locality for sliding-window kernels.
//!
//! A sliding kernel keeps running sums — add the entering row, subtract the leaving
//! one. A non-finite row breaks that: `inf − inf` and `NaN − NaN` are NaN, so once a
//! bad row has entered, the running sum stays NaN for the rest of the series even
//! though every later window is clean.
//!
//! The careful form keeps the running sums over **finite rows only** and counts the
//! non-finite rows the current window holds. A clean window reads the sums. A window
//! holding a bad row evaluates the kernel's formula directly on its own rows, so it
//! takes the IEEE value of exactly that window — a NaN gives NaN, an infinity gives
//! whatever the formula gives (`±inf` for a sum or mean, NaN for a variance).
//!
//! A kernel writes its loop **once**, generic over [`RowGuard`], and runs it twice at
//! most: first with [`Unguarded`] — every guard call is a constant, so it compiles to
//! the plain sliding loop with no per-row branch — and only if a running sum ended
//! non-finite again with [`NonFiniteRows`]. That check is exact: NaN is absorbing, and
//! an infinity turns into `inf − inf = NaN` when it leaves, so a sum that ever absorbed
//! a bad row is non-finite at the end. Clean data therefore costs nothing extra and is
//! bit-identical to the unguarded loop.
//!
//! Every running sum is an exact one ([`Csum`] — see [`crate::accum`]), so a window's
//! sums are its own: the fast pass, the careful pass and a resume step agree bit for
//! bit, and a bad row stops mattering once it has left the window.

use crate::accum::{anchors_at, Csum};

/// How a sliding kernel treats each row entering and leaving its window.
pub(crate) trait RowGuard: Default {
    /// Admit a row entering the window. `true` when it belongs in the running sums.
    fn enter(&mut self, finite: bool) -> bool;
    /// Retire a row leaving the window. `true` when it was in the running sums.
    fn leave(&mut self, finite: bool) -> bool;
    /// Whether the current window holds no bad row, so the running sums are its value.
    fn clean(&self) -> bool;
    /// Whether a row belongs in the running sums, without admitting it — for
    /// re-anchoring the sums from the window's own rows.
    fn counts(&self, finite: bool) -> bool;
}

/// Trust every row — the fast pass. Compiles to the plain sliding loop.
#[derive(Default)]
pub(crate) struct Unguarded;

impl RowGuard for Unguarded {
    #[inline(always)]
    fn enter(&mut self, _finite: bool) -> bool {
        true
    }
    #[inline(always)]
    fn leave(&mut self, _finite: bool) -> bool {
        true
    }
    #[inline(always)]
    fn clean(&self) -> bool {
        true
    }
    #[inline(always)]
    fn counts(&self, _finite: bool) -> bool {
        true
    }
}

/// Keep non-finite rows out of the running sums and count those inside the window.
#[derive(Default)]
pub(crate) struct NonFiniteRows(usize);

impl RowGuard for NonFiniteRows {
    #[inline]
    fn enter(&mut self, finite: bool) -> bool {
        if !finite {
            self.0 += 1;
        }
        finite
    }
    #[inline]
    fn leave(&mut self, finite: bool) -> bool {
        if !finite {
            self.0 -= 1;
        }
        finite
    }
    #[inline]
    fn clean(&self) -> bool {
        self.0 == 0
    }
    #[inline]
    fn counts(&self, finite: bool) -> bool {
        finite
    }
}

/// How a sliding kernel orders one row's update of its running sums. Each kernel keeps
/// its own order (its parity tests pin the fast loop's), and every path of the kernel —
/// the fast pass, the careful pass, the resume step — follows it, so a value never
/// depends on which path computed it or on rows after it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Order {
    /// Add the entering row, drop the leaving one, then read the window.
    AddDrop,
    /// Drop the leaving row, add the entering one, then read the window.
    DropAdd,
    /// Add the entering row, read the window, then drop the row the next window loses.
    AddReadDrop,
}

/// The window ending at a row, as a [`Slide`] reads it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Window<const K: usize> {
    /// Not full yet.
    Partial,
    /// Full and clean: its sums are the running ones.
    Sums([Csum; K]),
    /// Full and holding a non-finite row: the kernel evaluates it on its own rows.
    Bad,
}

/// The careful slide of `K` running sums of per-row terms (see the module docs): the
/// sums — exact, [`Csum`] — hold finite rows only, and the window's non-finite rows are
/// counted. Its state — the sums, that count and the absolute position of the next row
/// (for re-anchoring) — is everything a resume needs; the rows are read again.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Slide<const K: usize> {
    sums: [Csum; K],
    bad: usize,
    pos: usize,
}

impl<const K: usize> Slide<K> {
    /// The length of a slide's state: each sum's `[sum, comp]`, the count, the position.
    pub(crate) const STATE: usize = K * Csum::STATE + 2;

    /// An empty slide whose first row is at absolute position `origin`.
    pub(crate) fn at(origin: usize) -> Self {
        Slide { sums: [Csum::default(); K], bad: 0, pos: origin }
    }

    /// The state of an empty slide whose first row is at absolute position `origin`.
    pub(crate) fn initial(origin: usize) -> Vec<f64> {
        let mut v = vec![0.0; Self::STATE];
        Self::at(origin).write_state(&mut v);
        v
    }

    /// A slide from its state (`[sum, comp, …, count, position]`); `None` for a
    /// malformed one.
    pub(crate) fn from_state(state: &[f64]) -> Option<Self> {
        if state.len() != Self::STATE {
            return None;
        }
        let whole = |x: f64| (x.fract() == 0.0 && x >= 0.0).then_some(x as usize);
        let bad = whole(state[K * Csum::STATE])?;
        let pos = whole(state[K * Csum::STATE + 1])?;
        let mut sums = [Csum::default(); K];
        for (k, s) in sums.iter_mut().enumerate() {
            *s = Csum::from_state(&state[k * Csum::STATE..(k + 1) * Csum::STATE])?;
        }
        Some(Slide { sums, bad, pos })
    }

    /// Write the state into `out` (`[sum, comp, …, count, position]`).
    pub(crate) fn write_state(&self, out: &mut [f64]) {
        for (k, s) in self.sums.iter().enumerate() {
            s.write_state(&mut out[k * Csum::STATE..]);
        }
        out[K * Csum::STATE] = self.bad as f64;
        out[K * Csum::STATE + 1] = self.pos as f64;
    }

    /// The absolute position of the next row to slide.
    #[inline(always)]
    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    /// Recompute the sums from the window's own rows `rows`, oldest first (the
    /// re-anchoring of [`crate::accum`]); the count of bad rows is unchanged.
    fn reanchor(&mut self, rows: std::ops::RangeInclusive<usize>, term: &impl Fn(usize) -> ([f64; K], bool)) {
        self.sums = Csum::of_rows(rows.filter_map(|j| {
            let (t, finite) = term(j);
            finite.then_some(t)
        }));
    }

    /// Admit a row: its terms join the sums when `finite`, else it is counted.
    #[inline]
    pub(crate) fn enter(&mut self, (t, finite): ([f64; K], bool)) {
        if finite {
            for (s, v) in self.sums.iter_mut().zip(t) {
                s.add(v);
            }
        } else {
            self.bad += 1;
        }
    }

    /// Retire a row admitted earlier with the same terms.
    #[inline]
    pub(crate) fn leave(&mut self, (t, finite): ([f64; K], bool)) {
        if finite {
            for (s, v) in self.sums.iter_mut().zip(t) {
                s.sub(v);
            }
        } else {
            self.bad -= 1;
        }
    }

    /// Whether every running sum is exact (see [`Csum::is_exact`]).
    fn exact(&self) -> bool {
        self.sums.iter().all(Csum::is_exact)
    }

    fn read(&self) -> Window<K> {
        if self.bad == 0 {
            Window::Sums(self.sums)
        } else {
            Window::Bad
        }
    }

    /// Advance over row `i` (rows `0..i` already slid): the window of `period` rows
    /// ending at `i`. `term(j)` is row `j`'s terms and whether they belong in the
    /// running sums (the kernel's own finiteness test, of the row itself). A window at
    /// an anchor position (full or still filling — a shifted kernel's shift changes
    /// there, see [`crate::accum::shift_row`]), or a full one whose sums went inexact, is
    /// re-summed from its own rows before it is read.
    #[inline]
    pub(crate) fn row(
        &mut self,
        order: Order,
        period: usize,
        i: usize,
        term: impl Fn(usize) -> ([f64; K], bool),
    ) -> Window<K> {
        let full = i + 1 >= period;
        let anchor = anchors_at(self.pos, period);
        self.pos += 1;
        let rows = (i + 1).saturating_sub(period)..=i;
        match order {
            Order::AddDrop | Order::DropAdd => {
                if order == Order::DropAdd && i >= period {
                    self.leave(term(i - period));
                }
                self.enter(term(i));
                if order == Order::AddDrop && i >= period {
                    self.leave(term(i - period));
                }
                if anchor || (full && !self.exact()) {
                    self.reanchor(rows, &term);
                }
                if full { self.read() } else { Window::Partial }
            }
            Order::AddReadDrop => {
                self.enter(term(i));
                if anchor || (full && !self.exact()) {
                    self.reanchor(rows, &term);
                }
                if !full {
                    return Window::Partial;
                }
                let w = self.read();
                self.leave(term(i + 1 - period));
                w
            }
        }
    }
}

/// `term`'s values with the default finiteness test: every term finite.
#[inline]
pub(crate) fn all_finite<const K: usize>(t: [f64; K]) -> ([f64; K], bool) {
    (t, t.iter().all(|v| v.is_finite()))
}

/// The careful path of a kernel built on trailing sums of `K` per-row terms (row 0 at
/// absolute position `origin`): slide the running sums over finite rows only, and hand `emit` each full window's sums — the
/// running ones for a clean window, the window's own sums when it holds a row with a
/// non-finite term. A non-finite term therefore reaches exactly the windows holding it.
/// `emit` is called for every `i` in `period-1..n`, in order, after `order`'s update.
pub(crate) fn window_sums<const K: usize>(
    n: usize,
    period: usize,
    order: Order,
    origin: usize,
    term: impl Fn(usize) -> [f64; K],
    mut emit: impl FnMut(usize, [f64; K]),
) {
    if period == 0 || period > n {
        return;
    }
    let mut slide = Slide::<K>::at(origin);
    for i in 0..n {
        match slide.row(order, period, i, |j| all_finite(term(j))) {
            Window::Partial => {}
            Window::Sums(s) => emit(i, s.map(|c| c.value())),
            Window::Bad => emit(i, own_sums(i + 1 - period..=i, &term)),
        }
    }
}

/// The sums of `term` over `rows`, in row order — a window's own value.
pub(crate) fn own_sums<const K: usize>(
    rows: std::ops::RangeInclusive<usize>,
    term: impl Fn(usize) -> [f64; K],
) -> [f64; K] {
    let mut own = [0.0; K];
    for j in rows {
        for (o, v) in own.iter_mut().zip(term(j)) {
            *o += v;
        }
    }
    own
}

/// The state after resuming `step` over rows `0..n` from `init` — the resumable form of
/// a full computation, built from the very step that continues it. `None` if a step
/// declines.
pub(crate) fn state_after(
    init: Vec<f64>,
    n: usize,
    mut step: impl FnMut(usize, &[f64], &mut [f64]) -> Option<f64>,
) -> Option<Vec<f64>> {
    let mut prev = init;
    let mut next = prev.clone();
    for row in 0..n {
        step(row, &prev, &mut next)?;
        std::mem::swap(&mut prev, &mut next);
    }
    Some(prev)
}

/// Shift a ring of the last `ring.len()` values of a stage's output by one row: drop the
/// oldest, append `newest` (the ring as of the next row, written into `next`).
#[inline]
pub(crate) fn ring_push(ring: &[f64], newest: f64, next: &mut [f64]) {
    let shifted = ring.iter().skip(1).chain(std::iter::once(&newest));
    for (slot, &v) in next.iter_mut().zip(shifted) {
        *slot = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unguarded_trusts_every_row() {
        let mut g = Unguarded;
        assert!(g.enter(false) && g.leave(false) && g.clean() && g.counts(false));
        assert!(!NonFiniteRows::default().counts(false));
    }

    #[test]
    fn non_finite_rows_counts_what_the_window_holds() {
        let mut g = NonFiniteRows::default();
        assert!(g.enter(true) && g.clean());
        assert!(!g.enter(false) && !g.clean());
        assert!(!g.enter(false));
        assert!(!g.leave(false) && !g.clean());
        assert!(!g.leave(false) && g.clean());
        assert!(g.leave(true) && g.clean());
    }

    fn sums(xs: &[f64], period: usize) -> Vec<(usize, f64)> {
        let mut out = Vec::new();
        window_sums(xs.len(), period, Order::AddDrop, 0, |i| [xs[i]], |i, [s]| out.push((i, s)));
        out
    }

    #[test]
    fn window_sums_confine_a_bad_term_to_its_windows() {
        let got = sums(&[1.0, 2.0, f64::INFINITY, 4.0, 5.0], 2);
        assert_eq!(got[0], (1, 3.0));
        assert_eq!(got[1], (2, f64::INFINITY));
        assert_eq!(got[2], (3, f64::INFINITY));
        assert_eq!(got[3], (4, 9.0));
        let got = sums(&[1.0, f64::NAN, 3.0, 4.0], 2);
        assert!(got[0].1.is_nan() && got[1].1.is_nan());
        assert_eq!(got[2], (3, 7.0));
    }

    /// A spike that entered and left leaves no trace in a compensated sum; a plain
    /// running sum loses every later window to it.
    #[test]
    fn a_window_sum_forgets_a_spike() {
        let got = sums(&[1e20, 1.0, 2.0, 3.0, 4.0, 5.0], 2);
        let want = [1e20, 3.0, 5.0, 7.0, 9.0];
        assert_eq!(got.iter().map(|g| g.1).collect::<Vec<_>>(), want);
    }

    /// Past an anchor, a slide's sums are the window's own: a series that differs only
    /// before the anchor's window reads the same bits, whatever its origin's phase.
    #[test]
    fn an_anchor_forgets_the_rows_before_its_window() {
        let a: Vec<f64> = (0..40).map(|i| (i as f64 * 0.37).sin() * 1e3 + 1e-7 * i as f64).collect();
        let mut b = a.clone();
        b[3] = 1e25; // inexact even compensated: the low bits do not fit beside it
        b[5] = f64::NAN;
        for origin in [0, 1, 4] {
            let run = |xs: &[f64]| {
                let mut out = Vec::new();
                window_sums(xs.len(), 4, Order::AddReadDrop, origin, |i| [xs[i]], |i, [s]| out.push((i, s)));
                out
            };
            let (ra, rb) = (run(&a), run(&b));
            // rows from the first anchor whose window starts after row 5
            let first = (9..40).find(|&i| (origin + i) % 4 == 0).unwrap();
            assert_eq!(ra[first - 3..], rb[first - 3..], "origin {origin}");
        }
    }

    #[test]
    fn a_slide_state_round_trips_and_rejects_a_malformed_one() {
        let mut slide = Slide::<2>::at(7);
        slide.enter(([1.0, 2.0], true));
        slide.enter(([0.0, 0.0], false));
        let mut st = [0.0; Slide::<2>::STATE];
        slide.write_state(&mut st);
        assert_eq!(Slide::<2>::from_state(&st), Some(slide));
        assert_eq!(Slide::<2>::initial(3), vec![0.0, 0.0, 0.0, 0.0, 0.0, 3.0]);
        assert_eq!(Slide::<2>::from_state(&st[1..]), None);
        for bad in [0.5, -1.0] {
            let mut s = st;
            s[4] = bad;
            assert_eq!(Slide::<2>::from_state(&s), None);
            let mut s = st;
            s[5] = bad;
            assert_eq!(Slide::<2>::from_state(&s), None);
        }
    }

    #[test]
    fn window_sums_emit_nothing_for_an_impossible_window() {
        assert!(sums(&[1.0, 2.0], 3).is_empty());
        assert!(sums(&[1.0, 2.0], 0).is_empty());
    }
}
