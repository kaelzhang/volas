# volas benchmarks

The always-current published report lives at **https://volas.ost.ai/** (rebuilt
weekly from `master`). This page explains what it measures and how to reproduce
or extend it.

## What is measured

The benchmark harness (`test/test_benchmark.py`) drives three sections:

1. **Batch indicator computation** (`calc`) — compute an indicator over the full
   history, the way a research notebook would.
2. **Append-one-bar incremental refresh** (`append`) — the live case: a new bar
   arrives and the cached indicator column must update. volas refreshes only the
   stale tail (`O(lookback)`); libraries with no indicator cache recompute the
   series.
3. **Core API** (`api`) — frame construction, indexing, and conversion overhead.

The coverage table compares volas against TA-Lib on every indicator **both**
libraries implement, in two columns: *volas vs TA-Lib (after append)* and
*volas vs TA-Lib* (batch). A ratio > 1.00× means volas is faster.

## What is claimed (and what is not)

- On the current published report, volas beats TA-Lib on **139 / 157** covered
  indicators by the default ratio.
- volas is **not** a general-purpose DataFrame benchmark winner, and is not
  claimed to be faster for non-OHLCV dataframe workloads.
- TA-Lib remains a mature, widely used library; the comparison is about OHLCV
  indicator workloads and append-one-bar refresh, not every TA-Lib use case.
- Numbers shift with hardware, CPU governor, library versions, fixture length,
  and metric choice — so any headline must name the exact metric.

## Reproduce

```bash
pip install -e '.[dev,benchmark]'
make benchmark
```

This archives `benchmark.json`, a `report.html` (the same report published to
the site), and a `meta.txt` recording the commit, UTC date, working-tree dirty
flag, and the methodology key. A single indicator:

```bash
make benchmark INDICATOR=rsi
```

The paired A/B harness gates performance changes:

```bash
make perf-ab BASE=HEAD~1                 # full suite, pass/fail verdict
make perf-ab BASE=HEAD~1 INDICATOR=bop   # one indicator
```

The full gate builds isolated release wheels and runs six balanced AB/BA process
pairs with the current harness. It includes every volas row: coverage, extended
coverage, after-append coverage, API, calc, append, windowed streams, live time
frames, first materialization, and the million-row true-range benchmarks.
Rows are identified by section and full parametrized name, including length and
phase. The median paired HEAD/base ratio is gated at geometric mean ≤ 1.10 and
individual item ≤ 1.60. Rows below the 5 µs base floor are summed within their
section and gated as a section bucket, so small costs still contribute.

Release cost checks also run in the balanced pairs. Their within-run comparison
order alternates; the median of six normalized HEAD cost ratios must satisfy
each original strict bound. BASE cost samples are retained for diagnosis.
Correctness/coverage runs execute these same operations and retain all value
assertions, but do not enforce wall-clock bounds on instrumented code.
Allocation-count assertions remain in the correctness suite.

The preserved bounds are:

| Tests | Bound |
| --- | --- |
| `test_operand_read_cost`: live refresh, twenty directives | large < 8 × small + 20 µs / 100 µs |
| composite capture | 100k rows < 12 × 20k rows |
| first materialization (all directives) | cache < 1.8 × exec + 1 ms |
| `test_cell_write_cost`: hole fill, cell write | large < 5 × small |
| closed-row time-frame write | time-frame < 2 × plain |
| `test_cumulation_cost`: batch, resend | many < 8 × few; large < 5 × small |
| `test_core_io_cost`: integer admission | 30k rows < 30 × 3k rows |
| `test_fillna_cost`: dense fill | 1M rows < 10 × 10k rows |
| `test_append`: amortized append | large < 5 × small |
| `test_fulfill`: incremental refresh | 10 × fulfill < full recompute |

`test_benchmark_true_range` has timings rather than ratio assertions; both its
batch and single-row measurements participate in the full gate. Scoped
`INDICATOR` / `SECTION` runs provide focused measurements; the cost-bound suite
runs with the full gate. Automatic benchmarks have a 0.05 s budget per process;
pedantic append and repeated cost-check samples are spread across pairs
(rounding up, so the total sample budget is preserved). First caching keeps 20
rounds per process: the million-row workloads need that sample count to stabilize
medians under load. The harness reports measurement
and total wall time, paired ratio MAD, and the number of gated cost checks.
Use `PERF_AB_JSON_DIR=/path` to retain raw measurements and cost samples.

## Add a benchmark case

Comment on the standing
[benchmark challenge thread](https://github.com/kaelzhang/volas/issues/13), open
a PR, or open a
[benchmark-case issue](https://github.com/kaelzhang/volas/issues/new?template=benchmark_case.yml)
that includes:

- the fixture (input shape and how it is generated);
- the indicator directive(s);
- the scenario (batch vs append-one-bar);
- the candidate libraries and their versions;
- the hardware / OS and a reproducible script.

Cases where volas **loses** are welcome — they sharpen the claim. See
[docs/benchmark-faq.md](../docs/benchmark-faq.md) for the methodology FAQ.
