# Developer Guide

## Design notes & non-goals

- **Not a general-purpose DataFrame.** volas models exactly what OHLCV
  quant workflows need; it deliberately omits multi-level indexes,
  heterogeneous per-cell storage, joins, and general reshaping.
- **pandas-independent at runtime.** pandas and TA-Lib are used only as
  test oracles and benchmark comparators, never imported at runtime.
- **External API cleanliness first.** The Python surface is kept clean
  and pandas-shaped; internal layering is secondary to per-bar latency.

## Development

Requires Python >= 3.11 and a Rust toolchain.

```sh
make install        # Rust toolchain + maturin + Python dev deps
make build          # build the Rust extension, install the package in-place
make test           # run the Python test suite
make coverage       # true cargo-test union pytest line coverage
make benchmark      # multi-library benchmark
make build-pkg      # build a release wheel + sdist into dist/
```

`make coverage` delegates to `scripts/coverage.sh`. `make benchmark`
compares pandas, stock-pandas, polars, TA-Lib, and volas where those
benchmark-only dependencies are installed.

### Dependency groups

- **`dev`** (`pip install -e .[dev]`) — everything the test suite needs;
  this is all CI installs. It includes pandas because the parity tests
  use it as an oracle. pandas is test-time only; volas has no pandas
  runtime dependency.
- **`benchmark`** (`pip install -e .[benchmark]`) — extra comparison
  libraries used only by the benchmark. `make benchmark` installs
  `.[dev,benchmark]`; a library that is only needed to benchmark, never
  to test, belongs here so CI test runs stay lean.

### Releasing

Two artifacts ship from one version: the Python wheels (PyPI) and the Rust
crates (crates.io). Both are driven by a git tag.

```sh
make bump TYPE={major|minor|patch}   # bump the workspace version, commit, tag, push
```

Pushing the tag triggers `.github/workflows/release.yml`, which:

- verifies the tag matches the `Cargo.toml` version;
- builds + smoke-tests the wheels and the sdist, then publishes to **PyPI**
  (Trusted Publishing / OIDC);
- publishes the **Rust crates** to crates.io in dependency order (the
  `publish-crates` job runs `make cargo-publish`). `volas-python` is
  `publish = false`; the `volas` facade goes last.

crates.io publishing needs a `CARGO_REGISTRY_TOKEN` repo secret (an API token
with publish scope). To publish the crates by hand instead:

```sh
make cargo-publish          # real publish (run `cargo login` first)
make cargo-publish DRY=1    # dry-run: build + package every crate, upload nothing
```

### Benchmark & web report

`make benchmark` times every candidate on batch indicator computation,
the incremental append-one-bar path, and the full volas-vs-TA-Lib
coverage rows. To optimize one indicator, pass `INDICATOR=<directive>`;
that scoped run prints only that indicator's coverage rows and never
writes the web report:

```sh
make benchmark INDICATOR=roc:10
make benchmark                  # full run; always writes ./benchmark-report.html
make benchmark INDICATOR=bop    # ONE indicator: compact normalized report to
                                # stdout (volas vs talib, batch + after-append +
                                # append overhead); a working probe, NOT archived
make perf-ab BASE=HEAD~1 [INDICATOR=bop]
                                # the standard A/B: build+bench BASE in a temp
                                # worktree, then HEAD (incl. uncommitted changes),
                                # back-to-back on this machine; full suite ->
                                # perf_gate verdict, single indicator -> compact
                                # HEAD/base report. The CI gate's exact method.
# THE RULE: performance questions are answered through these three commands —
# never ad-hoc timing scripts, whose methodology drifts run to run.
# Every full run is also compared against the all-time best run (the
# "performance leader", .benchmarks/LEADER.json) and a leader-report.md is
# written into the run's archive dir. Scoring uses WITHIN-RUN normalized
# ratios (volas/talib measured in the same session), so runs from different
# machines/thermal states compare meaningfully; a run that beats the leader
# becomes the new leader. LEADER.json also carries the forward work queues:
# the top-10 slowest-vs-talib indicators (batch + after-append) and the
# after-append-overhead degrader list.
```

The locally generated `./benchmark-report.html` (and the always-current
published copy at <https://volas.ost.ai>, deployed by the
`pages` workflow) keeps the append and
batch sections as charts, then summarizes full coverage as one row per
TA-Lib indicator. Extra length fixtures and cached append-refresh
comparisons appear as additional `volas vs TA-Lib` columns instead of
duplicate indicator rows.
