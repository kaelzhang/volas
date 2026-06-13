# Contributing to volas

Thanks for helping improve volas. It is a Rust-backed, pandas-shaped DataFrame
for live OHLCV pipelines — contributions that sharpen that focus (indicators,
benchmarks, migration docs, examples) are especially welcome.

## Good first contributions

- Add an example under [`examples/`](examples/).
- Improve the pandas / TA-Lib / stock-pandas migration docs.
- Add a benchmark fixture or a realistic comparison case.
- Add parity tests for an indicator.
- Improve error messages for directive syntax.

## Local setup

volas is a Rust workspace exposed to Python via PyO3 / maturin.

```bash
make install     # dev install (.[dev]) + build the extension
make build       # rebuild the compiled extension
make test        # full suite: cargo ∪ pytest, 100% coverage gate (debug, overflow-checked)
make lint        # ruff + mypy (package) + cargo clippy
make types       # stub == runtime, public API fully typed
make benchmark   # reproduce the benchmark vs pandas / polars / stock-pandas / TA-Lib
```

A change is ready when `make test` and `make types` are green. CI runs the same
gates plus `make lint`; please run them locally before opening a PR.

## Indicator contributions

A new indicator must ship complete. Please include:

- the formula and an authoritative reference;
- the expected input columns and parameter defaults;
- lookback (warm-up) behavior;
- missing-value behavior;
- a source-pinned oracle reference and parity / mutation tests.

See [`INDICATORS.md`](INDICATORS.md) for the directive vocabulary and naming
conventions, and the existing `test/test_oracle*.py` / `test/*_mutation.py`
files for the test shape.

## Benchmark contributions

Benchmarks must be reproducible and must compare libraries **within the same
run** (same machine, same fixture, back-to-back). Extend the standard pipeline
(`make benchmark`) rather than writing ad-hoc timing scripts. When a result
bounds coverage (top-N, sampling, no-retry), say so — silent truncation reads
as "covered everything" when it did not.

PRs that add a realistic case where volas *loses* are welcome and useful.

## Pull requests

- Keep the change focused; one logical change per PR.
- Update the docs and `INDICATORS.md` alongside code when behavior changes.
- Describe what you verified (the commands you ran and their result).

By contributing you agree your contributions are licensed under the project's
[license](LICENSE).
