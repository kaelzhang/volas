//! Simple moving average batch and continuation rows share one guarded slide.
use crate::accum::Csum;
use crate::guard::{Order, Slide, Window};
use ndarray::{Array1, ArrayView1};

/// Simple moving average (a window is valid only when fully populated). Row 0 is at
/// absolute position `origin` (see [`crate::accum`]).
///
/// O(n) sliding running sum: each step adds the entering value and subtracts the
/// leaving one — an exact sum ([`Csum`], re-summed where it could not hold its sum), so each
/// window's mean is that of its own rows however long the slide. A window holding a
/// NaN is NaN.
#[inline]
pub fn sma(data: ArrayView1<f64>, period: usize, origin: usize) -> Array1<f64> {
    sma_pass(data, period, origin, &mut ())
}

/// SMA batch pass with optional terminal-state publication.
#[inline]
pub fn sma_pass<S: crate::pass::StateSink>(data: ArrayView1<f64>, period: usize, origin: usize, states: &mut S) -> Array1<f64> {
    let n = data.len();
    if period == 0 || period > n {
        return Array1::from_elem(n, f64::NAN);
    }
    // Fast path — a clean sliding sum with no per-element NaN bookkeeping, the common
    // OHLCV case. A **leading** NaN prefix is skipped (cascaded indicators — trima,
    // stochastic smoothing — feed an SMA over a series that warms up with NaN),
    // sliding from the first finite value. A later non-finite value poisons the sum
    // (and marks it inexact); that is checked at every re-anchoring and at the end, and
    // then the careful path takes over. A sum that went inexact on finite values is
    // re-summed at that row (see [`crate::accum`]). On the rows it keeps, the fast path does exactly the
    // careful slide's arithmetic ([`sma_row`]), so the two agree bit for bit.
    //
    // Fill the warm-up with NaN, then append each computed mean exactly once.
    if let Some(src) = data.as_slice() {
        let start = src.iter().position(|x| !x.is_nan()).unwrap_or(n);
        let valid_start = (start + period - 1).min(n);
        let mut out = Vec::with_capacity(n); // reserve the full length so push never reallocs
        out.resize(valid_start, f64::NAN);
        let pf = period as f64;
        let mut sum = Csum::default();
        let mut clean = true;
        // One numerical loop owns both paths. Capturing passes stop only at the
        // publication schedule, keeping all callbacks outside the row loop.
        macro_rules! slide_row {
            ($i:ident, $incoming:expr, $drop:block) => {{
                sum.add($incoming);
                $drop
                if !sum.is_exact() && $i + 1 >= period {
                    clean &= sum.value().is_finite();
                    if clean {
                        sum = Csum::of(src[($i + 1 - period).max(start)..=$i].iter().copied());
                    }
                }

            }};
        }
        if S::CAPTURE {
            let mut from = start;
            while from < n {
                let end = states.next_end(from + 1, n);
                // The bounded span proves each incoming load safe once, outside
                // the numerical loop. Outgoing reads keep their original order.
                let warm_end = end.min(valid_start).max(from);
                for (offset, &incoming) in src[from..warm_end].iter().enumerate() {
                    let i = from + offset;
                    slide_row!(i, incoming, {});
                }
                if from <= valid_start && valid_start < end {
                    let i = valid_start;
                    slide_row!(i, src[i], {});
                    out.push(sum.mean(pf));
                }
                let ready = from.max(start + period);
                if ready < end {
                    let incoming = &src[ready..end];
                    let outgoing = &src[ready - period..end - period];
                    out.extend(incoming.iter().zip(outgoing).enumerate().map(|(offset, (&add, &drop))| {
                        let i = ready + offset;
                        slide_row!(i, add, { sum.sub(drop); });
                        sum.mean(pf)
                    }));
                }
                states.record(end - 1, n, |out| Slide::append_components(out, [sum], period.min(end).saturating_sub(end - start), origin + end));
                from = end;
            }
        } else { for i in start..n { slide_row!(i, src[i], { if i >= start + period { sum.sub(src[i - period]); } }); if i + 1 >= start + period { out.push(sum.mean(pf)); } } }
        if clean && sum.value().is_finite() && (!S::CAPTURE || start < n.saturating_sub(1)) {
            return Array1::from_vec(out);
        }
    }
    states.restart();
    // Careful path — window-local (see [`crate::guard`]): the running sum holds only
    // finite rows; a window holding a non-finite row takes its own mean directly
    // (NaN for a NaN, ±inf for an infinity), so a bad row never outlives its window.
    let mut slide = Slide::<1>::at(origin);
    // Keep the value-only iterator free of the capturing closure environment.
    if !S::CAPTURE {
        return (0..n).map(|i| super::sma_row(&mut slide, period, i, |j| data[j])).collect();
    }
    let mut out = Vec::with_capacity(n);
    let mut from = 0;
    while from < n {
        let end = states.next_end(from + 1, n);
        out.extend((from..end).map(|i| super::sma_row(&mut slide, period, i, |j| data[j])));
        states.record(end - 1, n, |out| slide.append_state(out));
        from = end;
    }
    Array1::from_vec(out)
}

/// One row of [`sma`] over the series `term`, resumed from `slide` (rows `0..i` slid):
/// the careful slide, which the fast loop follows on clean windows — so a full [`sma`]
/// and a resume of this one row at a time agree bit for bit.
#[inline]
pub(crate) fn sma_row(
    slide: &mut Slide<1>,
    period: usize,
    i: usize,
    term: impl Fn(usize) -> f64,
) -> f64 {
    match slide.row(Order::AddDrop, period, i, |j| {
        let x = term(j);
        ([x], x.is_finite())
    }) {
        Window::Partial => f64::NAN,
        Window::Sums([sum]) => sum.mean(period as f64),
        Window::Bad => (i + 1 - period..=i).map(&term).sum::<f64>() / period as f64,
    }
}

/// The length of [`sma`]'s resumable state: the [`Slide`] of the sum.
pub(crate) const SMA_STATE: usize = Slide::<1>::STATE;

/// One row of [`sma`] over `term` from `state`, writing the state as of `row` into `next`.
#[inline]
pub(crate) fn sma_step(
    period: usize,
    row: usize,
    term: impl Fn(usize) -> f64,
    state: &[f64],
    next: &mut [f64],
) -> Option<f64> {
    if period == 0 || next.len() != SMA_STATE {
        return None;
    }
    let mut slide = Slide::<1>::from_state(state)?;
    let value = sma_row(&mut slide, period, row, term);
    slide.write_state(next);
    Some(value)
}

