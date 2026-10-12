# Numerical conventions

How every built-in indicator treats missing and infinite rows, keeps its sliding sums
exact, and takes variances, covariances and correlations. These rules hold on every
path that computes an indicator: a full computation, a resume from carried state, a
live append, a `time_frame` frame's forming bar and a bounded (`window=`) frame. The
indicator index itself is [INDICATORS.md](https://github.com/kaelzhang/volas/blob/main/INDICATORS.md).

## Missing and non-finite inputs

A missing row (NaN) or an infinite one affects only
the outputs whose definition reads it:

- An indicator over a **finite window** (a moving average, a rolling sum / variance /
  regression, an oscillator built from windowed sums, …) gives, for a window holding
  a bad row, the value its formula takes on that window — NaN when the window holds a
  NaN, the mathematical result when it holds an infinity (`±inf` for a sum or mean,
  NaN for a variance). Once that row has left the window, every output is exactly
  what clean data gives.
- A **leading run of missing rows** is not data: every indicator treats the series as
  starting after it, so a series with a missing head matches the same series started
  later.
- A **recursive** indicator (the EMA family and its cascades — DEMA / TEMA / T3 /
  TRIX / MACD —, Wilder smoothing — RSI / ATR / ADX / CMO —, cumulative lines such as
  OBV / AD / PVT / NVI / PVI / WAD / ASI, KAMA, SAR, Supertrend, Keltner, TSI, the
  Hilbert transform and MAMA, and any command whose MA-type argument selects a
  recursive average) skips a missing row: a row where **any** of its inputs is NaN
  outputs NaN, and the recursion continues from the last complete row as if that row
  did not exist. Its output equals the indicator run on the series with those rows
  deleted, NaN put back at their positions. The live paths follow the same rule — an
  append, the forming bar of a `time_frame` frame and a period rollover all leave the
  carried state at the last complete row when a new bar is missing. An infinity is a
  value, not a missing row. This diverges from TA-Lib, which carries a NaN into every
  later output of a recursive indicator.

## Exact window sums

An indicator built on a sliding sum
(a moving average, a rolling sum, variance, regression, correlation, the sums of CCI /
MFI / IMI / ULTOSC / VWMA / CMF / …, KAMA's sum of changes) slides it by adding the
entering row and subtracting the leaving one. volas keeps that sum exact: every window's
sum is the correctly rounded value of the exact sum of its rows — what Python's
`math.fsum` returns — on every path (a full computation, a resume, a live append, a
bounded frame). It carries the rounding error of every step (compensated summation),
detects the step whose error it could not hold — two outliers of different magnitudes
in one window, such as `1e35` and `1e20` beside ordinary prices — and then re-sums the
window exactly; it also recomputes the sum from the window's own rows at every row whose
absolute position in the series is a multiple of the window length. So a window's value
is that of its own rows, however long the history and whatever has left it: after
`close = [1e20, 1, 2, 3, 4, 5]`, `ma:2` is `[nan, 5e19, 1.5, 2.5, 3.5, 4.5]`; after
`[1.1, 1e35, 1e20, 1.3, 2.7, 3.1, 4.9]`, `sum:4` at the last row is `12.0`; a flat
window's `stddev` is exactly `0` and its `ma` exactly its value, after any history.
TA-Lib's plain running sums give `[nan, 5e19, 1, 2, 3, 4]` for the first example (every
window after the spike stays wrong) and drift over a long history; volas follows the
exact values instead, so on such data the two differ. (Where `math.fsum` itself fails —
an intermediate sum of the window's terms, oldest first, overflows `f64` — volas sums
the window at a scale of `2⁻⁶⁴` instead of raising; that is exact unless the same window
also holds a nonzero term smaller than `2⁻⁹⁵⁷` (about `1e-288`), whose lowest bits can
then be lost. A window whose sums never overflow is exact whatever its magnitudes.)
A row's absolute position is kept by a contiguous slice and by a bounded
(`window=`) frame dropping its head, so they compute exactly what the full series does —
the position indicators (`maxindex`, `minindex`, `minmaxindex`) included: they report
the series' own positions, on a slice as on the full frame.

## Centred moments

`var`, `stddev`, `boll`, `correl`, `beta` and the `linearreg` family take each
window's spread as `n·Σab − Σa·Σb` (`n²` times its centred co-moment) from the window's
exact sums — squares and products summed exactly — formed to within an ulp or two (the
high words' difference rounded once by FMA, then the low-order terms) and, near
cancellation, exactly. TA-Lib's one-pass raw moments `Σx²/n − (Σx/n)²` cancel
catastrophically when a window's mean dwarfs its spread: at a price of `12345.67` a
variance of `1e-12` reads `0`, a variance can go negative, a regression slope of `1e-9`
per bar is off by `6e-4`, a correlation exceeds `1`, and a flat window's correlation can
read `-1`. volas gives the exact values: a flat window has exactly zero variance and
slope, a correlation stays in `[-1, 1]`, and a window with a constant series (zero
variance, tested exactly) has `correl` and `beta` `0` — TA-Lib's result there, which
TA-Lib reaches through an absolute `1e-14` threshold on the raw-moment denominator, so
it also returns `0` for a small but genuine variance (e.g. FX-scale returns); volas does
not.

`mfi` likewise gives `0` only for a window whose money flow sums to exactly `0`. TA-Lib's
MFI returns `0` whenever the window's total flow is below `1.0`, an absolute threshold in
price × volume units, so its result changes with the volume's unit (a feed quoting
volume in thousands, or a low-priced instrument, reads `0`); volas's MFI is the same
for the same prices whatever the volume's scale.

