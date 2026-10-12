//! Missing values: what a missing (NaN) input does to every kind of computation.
//!
//! An infinity is a value, not a missing one. A missing value affects exactly the
//! outputs whose definition reads it, by one of three rules owned here:
//!
//! - **A per-row term** (a true range, a signed money flow, an up / down flag, a clamped
//!   difference, a rate of change, …) that reads a missing input is itself missing:
//!   [`term`] / [`terms`]. A comparison, `f64::max` / `f64::min` or a zero-base branch
//!   never steps over a missing input into an ordinary value.
//! - **A finite window** holding a missing row is missing: an output reduced from a
//!   window — a sum, an extremum, a position, a count — is NaN while the missing row is
//!   inside the window, and exactly what clean data gives once it has left. A running
//!   sum gets this from window locality ([`crate::guard`]) once its terms obey the term
//!   rule; a reduction that is not a sum (an extremum or its position, a tracker) is
//!   blanked by [`blank_windows`].
//! - **A recursive computation** (EMA cascade, Wilder smoothing, cumulative sums, the
//!   Hilbert transform, SAR, …) skips a row where any of its inputs is missing: the row
//!   outputs NaN, and the recursion continues from the last complete row as if the row
//!   did not exist — the computation runs, unchanged, on the complete rows only, and its
//!   outputs are scattered back to their original positions. A leading run of missing
//!   rows is the special case of a series that simply starts later.
//!
//! For recursions, [`CompleteRows`] owns the rule for a full computation and its
//! carried-state capture (which therefore describes the last complete row);
//! [`ResumeRows`] owns it for a resume over new rows, which reads only a bounded reach
//! of history. When no row is missing both hand the inputs through untouched. An
//! [`Input`] whose column already knows it holds no missing value (the frame remembers
//! that fact per column) is never scanned, so the common case costs nothing; otherwise
//! it costs one scan.

use std::borrow::Cow;

/// Whether `x` is a missing value. An infinity is a value.
#[inline]
pub fn is_missing(x: f64) -> bool {
    x.is_nan()
}

/// Whether any of `reads` — the inputs one per-row term reads — is missing.
/// Branch-free, so a term's check compiles to a select.
#[inline(always)]
pub fn any_of(reads: &[f64]) -> bool {
    reads.iter().fold(false, |acc, &x| acc | is_missing(x))
}

/// The per-row term rule: `value`, the term computed from `reads`, unless one of those
/// inputs is missing — then the term is missing (NaN) whatever `value` came out as.
#[inline(always)]
pub fn term(reads: &[f64], value: f64) -> f64 {
    if any_of(reads) { f64::NAN } else { value }
}

/// [`term`] for a term of several values (all missing together).
#[inline(always)]
pub fn terms<const K: usize>(reads: &[f64], values: [f64; K]) -> [f64; K] {
    if any_of(reads) { [f64::NAN; K] } else { values }
}

/// Whether any value of `window` is missing — the finite-window rule for one window.
#[inline]
pub fn holds_missing(window: &[f64]) -> bool {
    any_missing(window)
}

/// The finite-window rule over a whole series: set `out[i]` to NaN for every row `i`
/// whose window — the `span` rows of `x` ending at `i` — holds a missing value. For a
/// reduction that is not a running sum (an extremum, its position, a tracker's count):
/// its clean windows are already exact, and this leaves them alone. One scan when `x`
/// has a missing value, otherwise the scan that finds none.
pub fn blank_windows(out: &mut [f64], x: &[f64], span: usize) {
    blank_windows_where(out, x, span, is_missing);
}

/// [`blank_windows`] for a row test of the caller's (`bad`), for a computation whose
/// own rule widens what makes a row unusable (candle patterns: any non-finite price).
pub(crate) fn blank_windows_where(out: &mut [f64], x: &[f64], span: usize, bad: impl Fn(f64) -> bool) {
    let n = out.len().min(x.len());
    let Some(first) = x[..n].iter().position(|&v| bad(v)) else {
        return;
    };
    let mut last = first;
    for i in first..n {
        if bad(x[i]) {
            last = i;
        }
        if i - last < span {
            out[i] = f64::NAN;
        }
    }
}

/// Whether row `i` is complete: every input is present there.
#[inline]
fn complete(inputs: &[&[f64]], i: usize) -> bool {
    inputs.iter().all(|x| !is_missing(x[i]))
}

/// Whether any value of `x` is missing. Branch-free over fixed chunks so the common,
/// fully-present scan vectorises.
fn any_missing(x: &[f64]) -> bool {
    let mut chunks = x.chunks_exact(8);
    for c in &mut chunks {
        if c.iter().fold(false, |acc, &v| acc | is_missing(v)) {
            return true;
        }
    }
    chunks.remainder().iter().any(|&v| is_missing(v))
}

/// One input of a recursive computation: its values and, when its column already
/// knows it, whether any of them is missing. A known `false` skips every scan.
#[derive(Clone, Copy, Debug)]
pub struct Input<'a> {
    values: &'a [f64],
    has_missing: Option<bool>,
}

impl<'a> Input<'a> {
    /// Values with nothing known about them: scanned when it matters.
    pub fn new(values: &'a [f64]) -> Self {
        Input {
            values,
            has_missing: None,
        }
    }

    /// Values whose column knows whether any of them is missing.
    pub fn known(values: &'a [f64], has_missing: bool) -> Self {
        Input {
            values,
            has_missing: Some(has_missing),
        }
    }

    /// The values.
    pub fn values(&self) -> &'a [f64] {
        self.values
    }

    /// Whether any value in `[start..]` is missing: no scan for a known-complete input.
    fn any_missing_from(&self, start: usize) -> bool {
        self.has_missing != Some(false) && any_missing(&self.values[start..])
    }
}

fn values<'a>(inputs: &[Input<'a>]) -> Vec<&'a [f64]> {
    inputs.iter().map(|x| x.values).collect()
}

/// The common length of `inputs` (`0` for none).
fn height(inputs: &[&[f64]]) -> usize {
    inputs.first().map_or(0, |x| x.len())
}

/// The complete rows of a set of equal-length inputs.
#[derive(Clone, Debug, PartialEq)]
pub enum CompleteRows {
    /// Every row is complete: compute on the inputs as they are.
    All,
    /// Only the leading `start` rows are missing: compute on `inputs[start..]`.
    After(usize),
    /// Interior rows are missing: the original positions of the complete rows,
    /// ascending.
    Kept(Vec<usize>),
}

impl CompleteRows {
    /// Classify the rows of `inputs`: no scan at all when every input is known to be
    /// complete, else one pass over each input not known to be.
    pub fn of(inputs: &[Input]) -> CompleteRows {
        if inputs.iter().all(|x| x.has_missing == Some(false)) {
            return CompleteRows::All;
        }
        let vals = values(inputs);
        let n = height(&vals);
        let start = (0..n).find(|&i| complete(&vals, i)).unwrap_or(n);
        if !inputs.iter().any(|x| x.any_missing_from(start)) {
            return if start == 0 {
                CompleteRows::All
            } else {
                CompleteRows::After(start)
            };
        }
        CompleteRows::Kept((start..n).filter(|&i| complete(&vals, i)).collect())
    }

    /// `x` restricted to the complete rows (borrowed unless interior rows are dropped).
    pub fn gather<'a>(&self, x: &'a [f64]) -> Cow<'a, [f64]> {
        match self {
            CompleteRows::All => Cow::Borrowed(x),
            CompleteRows::After(start) => Cow::Borrowed(&x[*start..]),
            CompleteRows::Kept(rows) => Cow::Owned(rows.iter().map(|&i| x[i]).collect()),
        }
    }

    /// How many complete rows come before `row`.
    pub fn before(&self, row: usize) -> usize {
        match self {
            CompleteRows::All => row,
            CompleteRows::After(start) => row.saturating_sub(*start),
            CompleteRows::Kept(rows) => rows.partition_point(|&i| i < row),
        }
    }

    /// The original position of the `i`-th complete row.
    pub fn position(&self, i: usize) -> usize {
        match self {
            CompleteRows::All => i,
            CompleteRows::After(start) => start + i,
            CompleteRows::Kept(rows) => rows[i],
        }
    }

    /// Place `out`, computed over the complete rows, back at their original positions
    /// in a series of `n` rows; every missing row is NaN.
    pub fn scatter(&self, n: usize, out: Vec<f64>) -> Vec<f64> {
        match self {
            CompleteRows::All => out,
            CompleteRows::After(start) => {
                let mut full = Vec::with_capacity(n);
                full.resize(*start, f64::NAN);
                full.extend(out);
                full
            }
            CompleteRows::Kept(rows) => {
                let mut full = vec![f64::NAN; n];
                for (&i, v) in rows.iter().zip(out) {
                    full[i] = v;
                }
                full
            }
        }
    }
}

/// An output computed over the complete rows, placed back on every row (each missing
/// row NaN) — a series, a pair of series, or one that may be absent.
pub trait Scatter {
    /// `self`, computed over `rows` of a series of `n` rows, at their positions.
    fn scatter(self, rows: &CompleteRows, n: usize) -> Self;
}

impl Scatter for Vec<f64> {
    fn scatter(self, rows: &CompleteRows, n: usize) -> Self {
        rows.scatter(n, self)
    }
}

impl Scatter for (Vec<f64>, Vec<f64>) {
    fn scatter(self, rows: &CompleteRows, n: usize) -> Self {
        (rows.scatter(n, self.0), rows.scatter(n, self.1))
    }
}

impl<T: Scatter> Scatter for Option<T> {
    fn scatter(self, rows: &CompleteRows, n: usize) -> Self {
        self.map(|x| x.scatter(rows, n))
    }
}

/// `inputs` restricted to `rows`, borrowed unless interior rows are dropped.
fn gather<'a>(rows: &CompleteRows, inputs: &[&'a [f64]]) -> Vec<Cow<'a, [f64]>> {
    inputs.iter().map(|x| rows.gather(x)).collect()
}

fn views<'a>(gathered: &'a [Cow<[f64]>]) -> Vec<&'a [f64]> {
    gathered.iter().map(|x| x.as_ref()).collect()
}

/// Run `kernel` on the complete rows of `inputs` and scatter its output back: the
/// whole rule for one full computation. An input known complete is not scanned, and
/// with every row complete the inputs and the output pass through uncopied.
pub fn on_complete_rows<T: Scatter>(inputs: &[Input], kernel: impl FnOnce(&[&[f64]]) -> T) -> T {
    let rows = CompleteRows::of(inputs);
    let gathered = gather(&rows, &values(inputs));
    kernel(&views(&gathered)).scatter(&rows, height(&values(inputs)))
}

/// Run `capture` — a carried-state capture — on the complete rows of `inputs`, so the
/// state it returns describes the last complete row.
pub fn state_on_complete_rows<T>(inputs: &[&[f64]], capture: impl FnOnce(&[&[f64]]) -> T) -> T {
    let rows = CompleteRows::of(&unknown(inputs));
    capture(&views(&gather(&rows, inputs)))
}

/// Run `resume` — a resume over the new rows `[from, n)` from `state`, the state as of
/// the last complete row before `from` — on the complete rows of `inputs`: its history
/// is every complete row before `from`, a missing new row is NaN, and with no complete
/// new row the state stays `state`. `resume(inputs, from)` returns the new rows' values
/// and the state after them. Scans the whole history once (a resume reads a bounded
/// reach of it, which this layer does not know).
pub fn resume_on_complete_rows(
    inputs: &[&[f64]],
    from: usize,
    state: &[f64],
    resume: impl FnOnce(&[&[f64]], usize) -> Option<(Vec<f64>, Vec<f64>)>,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n = height(inputs);
    let rows = CompleteRows::of(&unknown(inputs));
    if rows == CompleteRows::All || from >= n {
        return resume(inputs, from);
    }
    let gathered = gather(&rows, inputs);
    let views = views(&gathered);
    let start = rows.before(from);
    if start == height(&views) {
        return Some((vec![f64::NAN; n - from], state.to_vec()));
    }
    let (out, next) = resume(&views, start)?;
    let mut full = vec![f64::NAN; n - from];
    for (i, v) in out.into_iter().enumerate() {
        full[rows.position(start + i) - from] = v;
    }
    Some((full, next))
}

/// Run `step` — a single-row resume at `row` from the state as of the last complete
/// row before it — on the complete rows of `inputs` up to `row`. A missing `row` is
/// `missing()` (its value NaN, the state not advanced); a `row` past the end is left
/// to `step`, which declines it.
pub fn step_on_complete_rows<T>(
    inputs: &[&[f64]],
    row: usize,
    missing: impl FnOnce() -> Option<T>,
    step: impl FnOnce(&[&[f64]], usize) -> Option<T>,
) -> Option<T> {
    if row >= height(inputs) {
        return step(inputs, row);
    }
    if !complete(inputs, row) {
        return missing();
    }
    let upto: Vec<&[f64]> = inputs.iter().map(|x| &x[..=row]).collect();
    let rows = CompleteRows::of(&unknown(&upto));
    if rows == CompleteRows::All {
        return step(inputs, row);
    }
    step(&views(&gather(&rows, &upto)), rows.before(row))
}

fn unknown<'a>(inputs: &[&'a [f64]]) -> Vec<Input<'a>> {
    inputs.iter().map(|x| Input::new(x)).collect()
}

/// The rows a resume over the new rows `[from, n)` runs on, for a recursion whose
/// resume reads at most `reach` rows before its first new row. When [`in_place`]
/// holds for every input the resume reads the inputs as they are; otherwise it runs on
/// these gathered rows.
///
/// [`in_place`]: ResumeRows::in_place
#[derive(Clone, Debug, PartialEq)]
pub struct ResumeRows {
    /// Original positions, ascending: the last `reach` complete rows before the first
    /// new row (fewer when the series has fewer), then the complete new rows.
    pub rows: Vec<usize>,
    /// Where the new rows start in `rows`: the resume starts there.
    pub from: usize,
}

impl ResumeRows {
    /// Whether a resume from `from` can read the input `x` in place: its new rows and
    /// the `reach` rows before them are present. Allocation-free, and free outright for
    /// an input known to be complete — the live single-bar path.
    pub fn in_place(x: Input, from: usize, reach: usize) -> bool {
        from >= x.values.len() || !x.any_missing_from(from.saturating_sub(reach))
    }

    /// The rows a resume of `inputs` from `from` runs on.
    pub fn of(inputs: &[&[f64]], from: usize, reach: usize) -> ResumeRows {
        let n = height(inputs);
        let mut rows: Vec<usize> = (0..from.min(n))
            .rev()
            .filter(|&i| complete(inputs, i))
            .take(reach)
            .collect();
        rows.reverse();
        let start = rows.len();
        rows.extend((from..n).filter(|&i| complete(inputs, i)));
        ResumeRows { rows, from: start }
    }

    /// Whether any new row is complete (else the resume has nothing to run on).
    pub fn has_new(&self) -> bool {
        self.rows.len() > self.from
    }

    /// Place `out`, the resumed values of the gathered new rows, at their positions
    /// among the `n - from` new rows; every missing new row is NaN.
    pub fn scatter(&self, n: usize, from: usize, out: Vec<f64>) -> Vec<f64> {
        let mut full = vec![f64::NAN; n - from];
        for (&i, v) in self.rows[self.from..].iter().zip(out) {
            full[i - from] = v;
        }
        full
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAN: f64 = f64::NAN;

    fn ins<'a>(x: &[&'a [f64]]) -> Vec<Input<'a>> {
        x.iter().map(|v| Input::new(v)).collect()
    }

    /// A column that knows it holds no missing value is taken at its word — no scan
    /// (that is the whole point; the frame keeps the fact honest); one that knows it
    /// does, or does not know, is scanned.
    #[test]
    fn a_known_complete_input_is_not_scanned() {
        let x = [1.0, NAN, 3.0];
        assert_eq!(
            CompleteRows::of(&[Input::known(&x, false)]),
            CompleteRows::All
        );
        assert!(ResumeRows::in_place(Input::known(&x, false), 2, 2));
        assert_eq!(
            CompleteRows::of(&[Input::known(&x, true)]),
            CompleteRows::Kept(vec![0, 2])
        );
        assert!(!ResumeRows::in_place(Input::known(&x, true), 2, 2));
        let y = [1.0, 2.0, 3.0];
        // mixed: only the unknown / missing one is scanned
        assert_eq!(
            CompleteRows::of(&[Input::known(&y, false), Input::new(&x)]),
            CompleteRows::Kept(vec![0, 2])
        );
        assert_eq!(Input::known(&y, false).values(), &y);
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn rows_are_classified_by_every_input() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(CompleteRows::of(&ins(&[&a[..]])), CompleteRows::All);
        let b = [NAN, NAN, 3.0, 4.0];
        assert_eq!(
            CompleteRows::of(&ins(&[&a[..], &b[..]])),
            CompleteRows::After(2)
        );
        let c = [1.0, NAN, 3.0, 4.0];
        assert_eq!(
            CompleteRows::of(&ins(&[&a[..], &c[..]])),
            CompleteRows::Kept(vec![0, 2, 3])
        );
        assert_eq!(
            CompleteRows::of(&ins(&[&b[..], &c[..]])),
            CompleteRows::After(2)
        );
        assert_eq!(
            CompleteRows::of(&ins(&[&[NAN, NAN][..]])),
            CompleteRows::After(2)
        );
        assert_eq!(CompleteRows::of(&ins(&[])), CompleteRows::All);
        // a long input exercises the chunked scan on both sides of a chunk boundary
        let mut long = [1.0; 21];
        assert_eq!(CompleteRows::of(&ins(&[&long[..]])), CompleteRows::All);
        long[3] = NAN;
        long[19] = NAN;
        let kept: Vec<usize> = (0..21).filter(|&i| i != 3 && i != 19).collect();
        assert_eq!(
            CompleteRows::of(&ins(&[&long[..]])),
            CompleteRows::Kept(kept)
        );
    }

    /// A per-row term that reads a missing input is missing, whatever value its formula
    /// came out as (a `max`, a comparison or a zero-base branch would step over it).
    #[test]
    fn a_term_reading_a_missing_input_is_missing() {
        assert!(any_of(&[1.0, NAN]) && !any_of(&[1.0, f64::INFINITY]) && !any_of(&[]));
        assert_eq!(term(&[1.0, 2.0], 3.0), 3.0);
        assert!(term(&[1.0, NAN], NAN.max(1.0)).is_nan());
        assert_eq!(term(&[f64::INFINITY], 0.0), 0.0); // an infinity is a value
        assert_eq!(terms(&[1.0], [1.0, 2.0]), [1.0, 2.0]);
        assert!(terms(&[NAN], [1.0, 2.0]).iter().all(|x| x.is_nan()));
    }

    /// A window holding a missing row is missing for exactly the rows whose window
    /// holds it; the rows before and after are left alone.
    #[test]
    fn a_window_holding_a_missing_row_is_blanked() {
        assert!(holds_missing(&[1.0, NAN]) && !holds_missing(&[1.0, 2.0]));
        let x = [1.0, 2.0, NAN, 4.0, 5.0, 6.0, NAN];
        let mut out = [9.0; 7];
        blank_windows(&mut out, &x, 3);
        assert_eq!(bits(&out), bits(&[9.0, 9.0, NAN, NAN, NAN, 9.0, NAN]));
        // span 1: the row itself only; nothing missing: untouched
        let mut out = [9.0; 7];
        blank_windows(&mut out, &x, 1);
        assert_eq!(bits(&out), bits(&[9.0, 9.0, NAN, 9.0, 9.0, 9.0, NAN]));
        let mut out = [9.0; 3];
        blank_windows(&mut out, &[1.0, 2.0, 3.0], 2);
        assert_eq!(out, [9.0; 3]);
        // a caller's own row test: any non-finite value
        let mut out = [9.0; 4];
        blank_windows_where(&mut out, &[1.0, f64::INFINITY, 3.0, 4.0], 2, |v| !v.is_finite());
        assert_eq!(bits(&out), bits(&[9.0, NAN, NAN, 9.0]));
    }

    #[test]
    fn an_infinity_is_a_value_not_a_missing_row() {
        let x = [f64::INFINITY, 1.0, f64::NEG_INFINITY];
        assert_eq!(CompleteRows::of(&ins(&[&x[..]])), CompleteRows::All);
        assert!(!is_missing(f64::INFINITY));
        assert!(is_missing(NAN));
    }

    #[test]
    fn gather_and_scatter_round_trip() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let all = CompleteRows::All;
        assert!(matches!(all.gather(&x), Cow::Borrowed(_)));
        assert_eq!(all.scatter(4, x.to_vec()), x.to_vec());
        let after = CompleteRows::After(1);
        assert_eq!(&*after.gather(&x), &[2.0, 3.0, 4.0]);
        assert_eq!(
            bits(&after.scatter(4, vec![5.0, 6.0, 7.0])),
            bits(&[NAN, 5.0, 6.0, 7.0])
        );
        let kept = CompleteRows::Kept(vec![0, 3]);
        assert_eq!(&*kept.gather(&x), &[1.0, 4.0]);
        assert_eq!(
            bits(&kept.scatter(4, vec![8.0, 9.0])),
            bits(&[8.0, NAN, NAN, 9.0])
        );
    }

    #[test]
    fn a_kernel_runs_on_the_complete_rows_only() {
        let running_sum = |x: &[&[f64]]| {
            x[0].iter()
                .scan(0.0, |s, v| {
                    *s += v;
                    Some(*s)
                })
                .collect::<Vec<f64>>()
        };
        let x = [1.0, 2.0, NAN, 3.0];
        assert_eq!(
            bits(&on_complete_rows(&[Input::new(&x)], running_sum)),
            bits(&[1.0, 3.0, NAN, 6.0])
        );
        let clean = [1.0, 2.0];
        assert_eq!(on_complete_rows(&[Input::new(&clean)], running_sum), vec![1.0, 3.0]);
        // known complete: taken at its word, never scanned
        assert_eq!(on_complete_rows(&[Input::known(&x, false)], running_sum).len(), 4);
    }

    /// A running sum carried as `[sum]`: the resume of [`a_kernel_runs_on_the_complete_rows_only`]'s kernel.
    fn running_resume(x: &[&[f64]], from: usize, state: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        let mut s = *state.first()?;
        let out: Vec<f64> = x[0][from..].iter().map(|v| {
            s += v;
            s
        })
        .collect();
        Some((out, vec![s]))
    }

    #[test]
    fn a_resume_runs_on_the_complete_rows_of_its_history_and_new_rows() {
        let x = [1.0, NAN, 2.0, 3.0, NAN, 4.0];
        let run = |x: &[&[f64]], from: usize| running_resume(x, from, &[10.0]);
        let (out, next) = resume_on_complete_rows(&[&x], 3, &[10.0], run).unwrap();
        assert_eq!(bits(&out), bits(&[13.0, NAN, 17.0]));
        assert_eq!(next, vec![17.0]);
        // no complete new row: NaN, the state where it was
        let (out, next) = resume_on_complete_rows(&[&x[..5]], 4, &[10.0], run).unwrap();
        assert_eq!(bits(&out), bits(&[NAN]));
        assert_eq!(next, vec![10.0]);
        // complete rows, or nothing new: the resume runs as it is
        let clean = [1.0, 2.0];
        assert_eq!(resume_on_complete_rows(&[&clean], 1, &[0.0], run).unwrap().0, vec![12.0]);
        assert_eq!(resume_on_complete_rows(&[&x], 6, &[0.0], run).unwrap().0, Vec::<f64>::new());
        // a resume that declines is declined
        assert!(resume_on_complete_rows(&[&x], 3, &[], |x, f| running_resume(x, f, &[])).is_none());
    }

    #[test]
    fn a_single_row_resume_skips_a_missing_row() {
        let x = [1.0, NAN, 2.0];
        let step = |x: &[&[f64]], row: usize| x[0].get(row).map(|v| (row, *v));
        assert_eq!(step_on_complete_rows(&[&x], 2, || None, step), Some((1, 2.0)));
        assert_eq!(step_on_complete_rows(&[&x], 1, || Some((9, 9.0)), step), Some((9, 9.0)));
        assert_eq!(step_on_complete_rows(&[&x], 0, || None, step), Some((0, 1.0)));
        assert_eq!(step_on_complete_rows(&[&x], 3, || Some((9, 9.0)), step), None);
    }

    #[test]
    fn a_state_capture_and_every_output_shape_follow_the_complete_rows() {
        let x = [1.0, NAN, 2.0];
        assert_eq!(state_on_complete_rows(&[&x], |x| x[0].to_vec()), vec![1.0, 2.0]);
        let pair = on_complete_rows(&[Input::new(&x)], |x| (x[0].to_vec(), x[0].to_vec()));
        assert_eq!(bits(&pair.1), bits(&[1.0, NAN, 2.0]));
        let some = on_complete_rows(&[Input::new(&x)], |x| Some(x[0].to_vec()));
        assert_eq!(bits(&some.unwrap()), bits(&[1.0, NAN, 2.0]));
        assert!(on_complete_rows(&[Input::new(&x)], |_| None::<Vec<f64>>).is_none());
        let rows = CompleteRows::After(2);
        assert_eq!((rows.before(1), rows.before(3), rows.position(1)), (0, 1, 3));
        assert_eq!((CompleteRows::All.before(2), CompleteRows::All.position(2)), (2, 2));
    }

    #[test]
    fn a_resume_gathers_its_reach_and_the_complete_new_rows() {
        let rows = |rows: Vec<usize>, from: usize| ResumeRows { rows, from };
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(ResumeRows::in_place(Input::new(&x), 4, 2));
        assert!(ResumeRows::in_place(Input::new(&[NAN]), 1, 2)); // no new row
                                                                 // a missing row before the reach is never read
        let y = [NAN, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(ResumeRows::in_place(Input::new(&y), 4, 2));
        // a missing row inside the reach: take the reach from complete rows only
        let z = [1.0, 2.0, NAN, 4.0, 5.0, 6.0];
        assert!(!ResumeRows::in_place(Input::new(&z), 4, 2));
        assert_eq!(ResumeRows::of(&[&z[..]], 4, 2), rows(vec![1, 3, 4, 5], 2));
        // a missing new row
        let w = [1.0, 2.0, 3.0, 4.0, NAN, 6.0];
        assert!(!ResumeRows::in_place(Input::new(&w), 4, 2));
        let gathered = ResumeRows::of(&[&w[..]], 4, 2);
        assert_eq!(gathered, rows(vec![2, 3, 5], 2));
        assert!(gathered.has_new());
        assert_eq!(bits(&gathered.scatter(6, 4, vec![9.0])), bits(&[NAN, 9.0]));
        // a short history gives what it has
        assert_eq!(
            ResumeRows::of(&[&z[..]], 4, 9),
            rows(vec![0, 1, 3, 4, 5], 3)
        );
        // only missing new rows: nothing to run on
        assert!(!ResumeRows::of(&[&[1.0, NAN][..]], 1, 2).has_new());
    }
}
