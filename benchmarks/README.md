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

The same back-to-back A/B harness gates performance changes:

```bash
make perf-ab BASE=HEAD~1                 # full suite, pass/fail verdict
make perf-ab BASE=HEAD~1 INDICATOR=bop   # one indicator
```

## Add a benchmark case

Open a PR (or a [benchmark-case issue](https://github.com/kaelzhang/volas/issues/new?template=benchmark_case.yml))
that includes:

- the fixture (input shape and how it is generated);
- the indicator directive(s);
- the scenario (batch vs append-one-bar);
- the candidate libraries and their versions;
- the hardware / OS and a reproducible script.

Cases where volas **loses** are welcome — they sharpen the claim. See
[docs/benchmark-faq.md](../docs/benchmark-faq.md) for the methodology FAQ.
