# Missing values and exact window sums

How every built-in indicator (see [INDICATORS.md](https://github.com/kaelzhang/volas/blob/main/INDICATORS.md))
treats its input data: what a missing (NaN) or infinite input does to the outputs, and how
sliding sums stay exact over any history.

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
  averages cover — and exactly what clean data gives once it has left. TA-Lib slides
  those averages as running totals, so one bad candle changes every later decision.
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
entering row and subtracting the leaving one. volas keeps that sum exact: it carries
the rounding error of every step (compensated summation), and it recomputes the sum from
the window's own rows at every row whose absolute position in the series is a multiple
of the window length. So a window's value is that of its own rows, however long the
history: after `close = [1e20, 1, 2, 3, 4, 5]`, `ma:2` is `[nan, 5e19, 1.5, 2.5, 3.5,
4.5]`; a flat window's `stddev` is exactly `0` and its `ma` exactly its value, after any
history; and once a bad row has left the window and the next anchor has passed, every
output is bit-identical to clean data's. TA-Lib's plain running sums give `[nan, 5e19,
1, 2, 3, 4]` for the example above (every window after the spike stays wrong) and drift
over a long history; volas follows the exact values instead, so on such data the two
differ. A row's absolute position is kept by a contiguous slice and by a bounded
(`window=`) frame dropping its head, so they compute exactly what the full series does —
the position indicators (`maxindex`, `minindex`, `minmaxindex`) included: they report
the series' own positions, on a slice as on the full frame.
