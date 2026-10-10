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

/// The careful path of a kernel built on trailing sums of `K` per-row terms: slide the
/// running sums over finite rows only, and hand `emit` each full window's sums — the
/// running ones for a clean window, the window's own sums when it holds a row with a
/// non-finite term. A non-finite term therefore reaches exactly the windows holding it.
/// `emit` is called for every `i` in `period-1..n`, in order. (Only the careful path;
/// a kernel keeps its own fast loop, whose operation order its parity tests pin.)
pub(crate) fn window_sums<const K: usize>(
    n: usize,
    period: usize,
    term: impl Fn(usize) -> [f64; K],
    mut emit: impl FnMut(usize, [f64; K]),
) {
    if period == 0 || period > n {
        return;
    }
    let finite = |t: &[f64; K]| t.iter().all(|v| v.is_finite());
    let mut guard = NonFiniteRows::default();
    let mut running = [0.0; K];
    for i in 0..n {
        let entering = term(i);
        if guard.enter(finite(&entering)) {
            for (r, v) in running.iter_mut().zip(entering) {
                *r += v;
            }
        }
        if i >= period {
            let leaving = term(i - period);
            if guard.leave(finite(&leaving)) {
                for (r, v) in running.iter_mut().zip(leaving) {
                    *r -= v;
                }
            }
        }
        if i + 1 >= period {
            if guard.clean() {
                emit(i, running);
            } else {
                let mut own = [0.0; K];
                for j in i + 1 - period..=i {
                    for (o, v) in own.iter_mut().zip(term(j)) {
                        *o += v;
                    }
                }
                emit(i, own);
            }
        }
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
        window_sums(xs.len(), period, |i| [xs[i]], |i, [s]| out.push((i, s)));
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
