# Benchmark FAQ

The published report is at **https://volas.ost.ai/**. For what it measures and
how to reproduce it, see [benchmarks/README.md](../benchmarks/README.md).

## Is the benchmark fair to TA-Lib?

TA-Lib is used as both an oracle (a correctness reference) and a comparator for
indicators where it has a matching function surface. volas is not claiming to
replace every TA-Lib use case — the comparison is about OHLCV indicator
workloads and append-one-bar refresh. The coverage table is restricted to the
indicators **both** libraries implement, and reports volas-vs-TA-Lib in both the
after-append and batch columns so you can see each separately.

## Why does append-one-bar matter?

In a live system the input grows by one bar (or the forming bar updates) on
every tick. Recomputing the full RSI / MACD / ATR series on each tick repeats
work that did not change. volas caches directive columns on the frame and
refreshes only the stale tail in `O(lookback)`. The append section of the report
isolates exactly this cost.

## Why can the numbers change?

Hardware, CPU frequency governor, library versions, fixture length, and the
metric you pick (mean vs best-of) all move results. That is why the report ships
a `meta.txt` (commit, UTC date, dirty flag, methodology key) and why every
headline names the exact metric — currently **139 / 157 covered indicators by
the default ratio**. If you reproduce different numbers, the fixture, versions,
or machine almost certainly differ; please include them when you report.

## Why not just report one big "X× faster" number?

A single multiplier hides the distribution: volas wins big on some indicators,
modestly on others, and loses on a handful. The report shows the full coverage
table (including the indicators where volas is slower) rather than a cherry-
picked aggregate.

## Where volas does *not* try to win

General-purpose dataframe workloads — joins, group-bys, pivots, arbitrary
reshaping — are out of scope; use pandas or polars. volas is a narrow DataFrame
for live OHLCV indicator pipelines.

## How do I challenge or extend the benchmark?

Open a [benchmark-case issue](https://github.com/kaelzhang/volas/issues/new?template=benchmark_case.yml)
or a PR with a reproducible script (fixture, indicators, scenario, library
versions, hardware). Realistic cases where volas loses are explicitly welcome.
