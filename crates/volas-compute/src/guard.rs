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

/// How a sliding kernel treats each row entering and leaving its window.
pub(crate) trait RowGuard: Default {
    /// Admit a row entering the window. `true` when it belongs in the running sums.
    fn enter(&mut self, finite: bool) -> bool;
    /// Retire a row leaving the window. `true` when it was in the running sums.
    fn leave(&mut self, finite: bool) -> bool;
    /// Whether the current window holds no bad row, so the running sums are its value.
    fn clean(&self) -> bool;
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
    Sums([f64; K]),
    /// Full and holding a non-finite row: the kernel evaluates it on its own rows.
    Bad,
}

/// The careful slide of `K` running sums of per-row terms (see the module docs): the
/// sums hold finite rows only, and the window's non-finite rows are counted. Its state —
/// the sums and that count — is everything a resume needs; the rows are read again.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Slide<const K: usize> {
    sums: [f64; K],
    bad: usize,
}

impl<const K: usize> Default for Slide<K> {
    fn default() -> Self {
        Slide { sums: [0.0; K], bad: 0 }
    }
}

impl<const K: usize> Slide<K> {
    /// The length of a slide's state: its sums, then the count.
    pub(crate) const STATE: usize = K + 1;

    /// A slide from its state (`[sums.., count]`); `None` for a malformed one.
    pub(crate) fn from_state(state: &[f64]) -> Option<Self> {
        let (sums, rest) = state.split_first_chunk::<K>()?;
        let [bad] = rest else { return None };
        (bad.fract() == 0.0 && *bad >= 0.0).then_some(Slide { sums: *sums, bad: *bad as usize })
    }

    /// Write the state into `out` (`[sums.., count]`).
    pub(crate) fn write_state(&self, out: &mut [f64]) {
        out[..K].copy_from_slice(&self.sums);
        out[K] = self.bad as f64;
    }

    /// Admit a row: its terms join the sums when `finite`, else it is counted.
    #[inline]
    pub(crate) fn enter(&mut self, (t, finite): ([f64; K], bool)) {
        if finite {
            for (s, v) in self.sums.iter_mut().zip(t) {
                *s += v;
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
                *s -= v;
            }
        } else {
            self.bad -= 1;
        }
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
    /// running sums (the kernel's own finiteness test).
    #[inline]
    pub(crate) fn row(
        &mut self,
        order: Order,
        period: usize,
        i: usize,
        term: impl Fn(usize) -> ([f64; K], bool),
    ) -> Window<K> {
        let full = i + 1 >= period;
        match order {
            Order::AddDrop | Order::DropAdd => {
                if order == Order::DropAdd && i >= period {
                    self.leave(term(i - period));
                }
                self.enter(term(i));
                if order == Order::AddDrop && i >= period {
                    self.leave(term(i - period));
                }
                if full { self.read() } else { Window::Partial }
            }
            Order::AddReadDrop => {
                self.enter(term(i));
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

/// The careful path of a kernel built on trailing sums of `K` per-row terms: slide the
/// running sums over finite rows only, and hand `emit` each full window's sums — the
/// running ones for a clean window, the window's own sums when it holds a row with a
/// non-finite term. A non-finite term therefore reaches exactly the windows holding it.
/// `emit` is called for every `i` in `period-1..n`, in order, after `order`'s update.
pub(crate) fn window_sums<const K: usize>(
    n: usize,
    period: usize,
    order: Order,
    term: impl Fn(usize) -> [f64; K],
    mut emit: impl FnMut(usize, [f64; K]),
) {
    if period == 0 || period > n {
        return;
    }
    let mut slide = Slide::<K>::default();
    for i in 0..n {
        match slide.row(order, period, i, |j| all_finite(term(j))) {
            Window::Partial => {}
            Window::Sums(s) => emit(i, s),
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
        assert!(g.enter(false) && g.leave(false) && g.clean());
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
        window_sums(xs.len(), period, Order::AddDrop, |i| [xs[i]], |i, [s]| out.push((i, s)));
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

    #[test]
    fn window_sums_emit_nothing_for_an_impossible_window() {
        assert!(sums(&[1.0, 2.0], 3).is_empty());
        assert!(sums(&[1.0, 2.0], 0).is_empty());
    }
}
