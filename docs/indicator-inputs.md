# Missing values and exact window sums

How every built-in indicator (see [INDICATORS.md](https://github.com/kaelzhang/volas/blob/main/INDICATORS.md))
treats its input data: what a missing (NaN) or infinite input does to the outputs, how
sliding sums stay exact over any history, and how variances and correlations are taken.
These rules hold on every path that computes an indicator: a full computation, a resume
from carried state, a live append, a `time_frame` frame's forming bar and a bounded
(`window=`) frame.

## Missing and non-finite inputs

A missing row (NaN) or an infinite one affects only the outputs whose definition reads
it:

- An indicator over a **finite window** (a moving average, a rolling sum / variance /
  regression, an oscillator built from windowed sums, …) gives, for a window holding
  a bad row, the value its formula takes on that window — NaN when the window holds a
  NaN, the mathematical result when it holds an infinity (`±inf` for a sum or mean,
  NaN for a variance). Once that row has left the window, every output is exactly
  what clean data gives. This holds for every finite-window reduction, not only sums:
  a highest / lowest value and its position (`hhv`, `llv`, `donchian`, `midpoint`,
  `maxindex`, `aroon`, `willr`, `rsv`, `stoch`, `mike`, `ichimoku`, …) and a count
  (`psy`) are NaN while their window holds a NaN — they never step over it. A per-row
  term that reads a missing input is missing too, whatever its formula would make of
  it: a true range (`tr`, and so `chop`, `vortex`, `ultosc`), a signed money flow
  (`mfi`), an up / down split (`vr`, `psy`), a clamped difference (`brar.br`), a rate
  of change over a zero base (`roc`, `rocp`, `rocr`, `rocr100` give `0` over a zero
  base only for a present value), and a missing `mavp` period. A highest / lowest value
  that is zero is `+0.0`, whichever signed zero the window holds, so a later missing
  value never changes an earlier extremum, not even the sign of a zero.
- A **candlestick pattern** (`cdl.<pattern>`) reads whole candles: a bar whose open,
  high, low or close is missing or infinite is a missing candle, and every pattern is
  NaN on the bars whose lookback holds it — its own bars and the bars its body / range
  averages cover — and what clean data gives once it has left (bit for bit: the
  averages are exact window sums, see [Exact window sums](#exact-window-sums)). TA-Lib
  slides those averages as plain running totals, so one bad candle changes every later
  decision.
- A **leading run of missing rows** is not data: every indicator treats the series as
  starting after it, so a series with a missing head matches the same series started
  later.
- A **recursive** indicator (the EMA family and its cascades — DEMA / TEMA / T3 /
  TRIX / MACD —, Wilder smoothing — RSI / ATR / ADX / CMO —, cumulative lines such as
  OBV / AD / PVT / NVI / PVI / WAD / ASI, KAMA, SAR, Supertrend, Keltner, TSI, KDJ,
  Connors RSI, the Hilbert transform and MAMA, and any command whose MA-type argument
  selects a recursive average) skips a missing row: a row where **any** of its inputs is NaN
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
window exactly. So a window's value
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

`var`, `stddev`, `boll`, `hv`, `correl`, `beta` and the `linearreg` family take each
window's spread from **shifted data**; the window API's `var()` / `std()` / `sem()` from
**exact sums of squares** (see [Why two methods](#why-two-methods) below). The
one-pass raw moments `Σx²/n − (Σx/n)²` (TA-Lib's) cancel catastrophically when a
window's level dwarfs its spread: at a price of `12345.67` a variance of `1e-12` reads
`0`, a variance can go negative, a regression slope of `1e-9` per bar is off by `6e-4`,
a correlation exceeds `1`, and a flat window's correlation can read `-1`.

volas subtracts a **shift** `K` that lies inside the window from every value before
squaring: `d = x − K`, and the centred sum of squares is `C = Σd² − (Σd)²/n`. `K` is the
value at the window's anchor row — the row whose absolute position is a multiple of the
window length (any `W` consecutive rows hold exactly one; where a new anchor enters, the
sums are rebuilt from the window's own rows with its value — `O(W)` once per `W` rows).
It is a function of the absolute position alone, so the full computation, a resume, a
live append, a slice and a bounded frame all shift by the same value and agree bit for
bit. `Σx` stays an exact sum (so `boll`'s middle band is `ma`'s exact mean), `Σd²` is
the exact sum of the rounded squares, and `Σd` is taken from `Σx`'s exact value less
the exact product `n·K`. The regression slides the exact sums `Σy` and `Σ age·y` (the
weighted terms are exact products, so no shift is needed to keep them exact) and reads
them on the data shifted by the same anchor value `K`: `Σd = Σy − n·K` and `Σ age·d =
Σ age·y − K·Σage` from their exact words, then `m = (n·Σ age·d − Σage·Σd) / divisor`
and the intercept `K + (Σd − m·Σage) / n`.

**Accuracy.** Let `u = 2⁻⁵³`, `n` the window's rows, `R` its range (max − min), `C*` its
exact centred sum of squares, and assume no deviation, square or product underflows
(every nonzero one is at least `2⁻¹⁰²²` in magnitude; deviations of normal-range prices
are far from that). Every `|x − K| ≤ R` and `|Σd| ≤ n·R`. Rounding each deviation and
each square costs at most `3u·d²`, the exact sum of the squares is rounded once (`u`),
and `Σd` is within `2u·|Σd| + 4u²·n·|K|` of exact; squaring and dividing it adds `2u`.
So the computed `C` is within `10·u·n·R² + 8u²·n·|K|·R + u·|C|` of `C*`. Two distinct
doubles near `K` differ by at least `u·|K|` (a double's spacing exceeds `u` times its
magnitude), so `u·|K| ≤ R`, and a window spanning `R` has `C* ≥ R²/2`; hence

- a variance is within **`(36·n + 2)·2⁻⁵³` relative** of the exact one (a standard
  deviation within half that, plus its square root's rounding) — `≈ 8e-14` for `n =
  20`, whatever the level;
- a correlation is within `(72·n + 6)·2⁻⁵³` absolute of the exact one (its covariance
  is bounded the same way against `sqrt(Cxx·Cyy)`), and is clamped to `[-1, 1]`
  (Cauchy–Schwarz can only be overshot by that rounding);
- a regression slope `m` is within `90·2⁻⁵³·R/(n + 1) + 2⁻⁵³·|m|` absolute of the exact
  one (each shifted sum within `6u` of its bound — `n·R`, `Σage·R` — the numerator
  within `15u·n·Σage·R`, the divisor `n²(n² − 1)/12`);
- a **flat window** has every `d = 0` and `Σd = 0` exactly: its variance, deviation and
  slope are exactly `0`.

### Why two methods

A directive kernel is resumed from carried
state and refreshed live, so it must reach a window by the same arithmetic on every path;
the absolute row position is already part of that state (it is what a bounded frame or
a slice keeps), and shifting by the anchor-row value costs no more than the raw moments.
The window API (`rolling` / `expanding` `var()` / `std()` / `sem()`) is a stateless
snapshot whose contract is that a window's value depends on its values alone — the same
for a series, a slice of it, a bounded frame or a fresh series of those rows — so it
cannot shift by a value chosen by absolute position. It keeps exact `Σx` and an exact
`Σx²` (each square split into its rounded value and its rounding error, summed apart)
and forms `n·Σx² − (Σx)²` from their canonical words: rounded once by FMA with the
low-order terms added, or, within `2⁻⁴⁰` of cancellation, summed exactly. That is within
an ulp or two of the exact variance, exactly `0` for a constant window, and a function of
the window's values only; O(n), one sliding pass.

A window with a constant series (zero variance, tested exactly) has `correl` and `beta`
`0` — TA-Lib's result there, which TA-Lib reaches through an absolute `1e-14` threshold
on the raw-moment denominator, so it also returns `0` for a small but genuine variance
(e.g. FX-scale returns); volas does not.

`mfi` likewise gives `0` only for a window whose money flow sums to exactly `0`. TA-Lib's
MFI returns `0` whenever the window's total flow is below `1.0`, an absolute threshold in
price × volume units, so its result changes with the volume's unit (a feed quoting
volume in thousands, or a low-priced instrument, reads `0`); volas's MFI is the same
for the same prices whatever the volume's scale.

