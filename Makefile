files = volas test *.py
test_files = *
PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
MATURIN ?= $(PYTHON) -m maturin
PY_PREFIX := $(shell $(PYTHON) -c "import sys; print(sys.prefix)")
MATURIN_DEVELOP_ENV := VIRTUAL_ENV="$(PY_PREFIX)" CONDA_PREFIX="$(PY_PREFIX)"

.PHONY: install install-rust build build-pkg build-ext clean test test-quick coverage coverage-html benchmark lint fix fmt check cargo-test upload publish bump dev ci

# Install all dependencies (Python + Rust)
install:
	@echo "\033[1m>> Installing Rust toolchain... <<\033[0m"
	@which rustup > /dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
	@rustup update stable
	@echo "\033[1m>> Installing maturin... <<\033[0m"
	@pip install maturin
	@echo "\033[1m>> Installing coverage tooling (cargo-llvm-cov + llvm-tools)... <<\033[0m"
	@rustup component add llvm-tools-preview
	@cargo install cargo-llvm-cov --locked || true
	@echo "\033[1m>> Installing Python dependencies... <<\033[0m"
	@pip install -e .[dev]

# Install only the Rust toolchain
install-rust:
	@which rustup > /dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
	@rustup update stable

# Build the Rust extension and install the package in-place (development)
build: clean
	@echo "\033[1m>> Building Rust extension... <<\033[0m"
	@$(MATURIN_DEVELOP_ENV) $(MATURIN) develop --release
	@echo "\033[1m>> Build complete! <<\033[0m"

# Build the release package (wheel and sdist) into dist/
build-pkg: clean
	@echo "\033[1m>> Building release package... <<\033[0m"
	@$(MATURIN) build --release --sdist -o dist
	@echo "\033[1m>> Package built in dist/ <<\033[0m"

# Build the Rust extension only (development mode)
build-ext:
	@$(MATURIN_DEVELOP_ENV) $(MATURIN) develop

# Clean build artifacts
clean:
	rm -rf dist build target/wheels
	rm -rf volas/*.so volas_rs*.so
	rm -rf *.egg-info
	rm -rf .eggs
	rm -rf .coverage .coverage.* htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Run the full suite (cargo test + the Python suite) and finish by printing the
# combined Python+Rust line-coverage report. An alias for `coverage` so the
# default test run always ends with the coverage summary. Needs cargo-llvm-cov +
# llvm-tools-preview (installed by `make install`); for a fast inner loop without
# coverage instrumentation use `make test-quick`.
test: coverage

# Fast functional tests only — no coverage, no instrumented rebuild (the dev loop).
# Benchmarks are skipped here (see `make benchmark`); `--ignore` does not exclude an
# explicitly-globbed file, so use `--benchmark-skip`.
test-quick:
	$(PYTEST) -s -v test/test_$(test_files).py --benchmark-skip

# True-union Rust line coverage: `cargo test` ∪ the Python suite exercising the
# compiled extension (see scripts/coverage.sh for why llvm-cov cannot union the
# two builds itself, and why `pytest --cov` is meaningless for a Rust package). Runs
# both suites and prints the combined per-file + total report.
coverage:
	@bash scripts/coverage.sh

# Same, rendered to a browsable HTML report under target/volas-cov/html/.
coverage-html:
	@bash scripts/coverage.sh --html

# Run the multi-library performance benchmark (pandas / stock-pandas / polars /
# TA-Lib / DuckDB / volas), for both batch indicator computation and the
# incremental "append one bar" path. Depends on `build` so the volas extension is
# compiled in release mode, then installs the dev + benchmark comparison libraries
# (read straight from pyproject so the list stays single-sourced; the volas
# extension itself is NOT reinstalled, preserving the release build). A comparison
# library that is absent is skipped by the harness.
#
#   make benchmark                    # full run, archived under .benchmarks/<stamp>/
#   make benchmark INDICATOR=roc:10   # one coverage row only, not archived
#
# Every full run is persisted to .benchmarks/<date>-<short-commit>[-dirty]/ (the
# whole dir is gitignored), each holding benchmark.json + report.html + meta.txt,
# so a given Git commit's performance can be retrieved and compared later (e.g.
#   python scripts/perf_gate.py .benchmarks/<new>/benchmark.json \
#       --base .benchmarks/<old>/benchmark.json
# ). The latest run is also mirrored to .benchmarks/last.json + ./benchmark-report.html.
BENCH_OPTS := --benchmark-only --benchmark-group-by=func,param:indicator \
              --benchmark-columns=mean,median,ops,rounds --benchmark-sort=name
BENCH_STAMP := $(shell date +%Y-%m-%d)-$(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)$(shell git diff --quiet HEAD 2>/dev/null || echo -dirty)
BENCH_DIR := .benchmarks/$(BENCH_STAMP)
benchmark: build
	@echo "\033[1m>> Installing dev + benchmark comparison libraries... <<\033[0m"
	@$(PYTHON) -c "import tomllib; e=tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']; print('\n'.join(e['dev'] + e['benchmark']))" | $(PIP) install -q -r /dev/stdin
ifdef INDICATOR
	@if [ -n "$(WEB_REPORT)" ]; then echo "WEB_REPORT is ignored when INDICATOR is set."; fi
	$(PYTEST) test/test_benchmark.py $(BENCH_OPTS) --volas-benchmark-indicator="$(INDICATOR)"
else
	@mkdir -p $(BENCH_DIR)
	$(PYTEST) test/test_benchmark.py $(BENCH_OPTS) --benchmark-json=$(BENCH_DIR)/benchmark.json
	@$(PYTHON) scripts/benchmark_report.py $(BENCH_DIR)/benchmark.json $(BENCH_DIR)/report.html
	@$(PYTHON) scripts/bench_trim.py $(BENCH_DIR)/benchmark.json   # drop ~1GB of per-round data; keep stats
	@printf 'commit: %s\ndate:   %s\ndirty:  %s\n' "$$(git rev-parse HEAD 2>/dev/null || echo none)" "$$(date -u +%FT%TZ)" "$$(git diff --quiet HEAD 2>/dev/null && echo no || echo yes)" > $(BENCH_DIR)/meta.txt
	@cp $(BENCH_DIR)/benchmark.json .benchmarks/last.json
	@cp $(BENCH_DIR)/report.html benchmark-report.html
	@echo "\033[1m>> Benchmark archived to $(BENCH_DIR)/ <<\033[0m"
endif

# Run linters
lint:
	@echo "\033[1m>> Running ruff... <<\033[0m"
	@ruff check $(files)
	@echo "\033[1m>> Running mypy... <<\033[0m"
	@mypy $(files)
	@echo "\033[1m>> Running cargo check... <<\033[0m"
	@cargo check

# Static type gates — deterministic oracles that prove the shipped typings are correct
# (no human review needed). Needs the extension built so the stub is installed.
#   stubtest      — the .pyi matches the runtime module (signatures, params, attrs)
#   mypy/pyright  — the @overload-driven dynamic surfaces resolve to the right types,
#                   and known-wrong usage is flagged (negative_types.py)
#   --verifytypes — the public API is 100% typed
types: build
	@echo "\033[1m>> stubtest (stub == runtime)... <<\033[0m"
	@$(PYTHON) -m mypy.stubtest volas_rs --allowlist stubtest_allowlist.txt
	@echo "\033[1m>> mypy --strict (assert_type + negative)... <<\033[0m"
	@mypy --strict test/typing/check_types.py test/typing/negative_types.py
	@echo "\033[1m>> pyright (assert_type + negative)... <<\033[0m"
	@pyright test/typing/check_types.py test/typing/negative_types.py
	@echo "\033[1m>> pyright --verifytypes (public-API completeness)... <<\033[0m"
	@pyright --verifytypes volas --ignoreexternal

# Auto-fix lint issues
fix:
	ruff check --fix $(files)
	cargo fmt

# Format Rust code
fmt:
	cargo fmt

# Check Rust code
check:
	cargo check
	cargo clippy

# Run Rust unit tests
cargo-test:
	cargo test

# Upload to PyPI
upload:
	twine upload --config-file ~/.pypirc -r pypi dist/* --verbose

# Publish (build package + upload)
publish:
	make build-pkg
	make upload

# Bump the workspace version, commit, tag (no `v` prefix), then push the commit and
# the tag to origin. The pushed tag triggers the GitHub release workflow, which
# verifies the tag matches the Cargo.toml version, then builds wheels and publishes
# to PyPI via Trusted Publishing. Usage: make bump TYPE={major|minor|patch}
bump:
	@test -n "$(TYPE)" || { echo "TYPE is required: make bump TYPE={major|minor|patch}"; exit 1; }
	@test -z "$$(git status --porcelain)" || { echo "working tree is not clean -- commit or stash first"; exit 1; }
	@next=$$(python3 scripts/bump_version.py --next "$(TYPE)") || exit 1; \
	if git rev-parse -q --verify "refs/tags/$$next" >/dev/null 2>&1; then \
		echo "tag $$next already exists -- aborting"; exit 1; \
	fi; \
	echo "\033[1m>> Bumping version to $$next <<\033[0m"; \
	python3 scripts/bump_version.py "$(TYPE)" >/dev/null || exit 1; \
	git add Cargo.toml; \
	git commit -m "chore(release): $$next" -- Cargo.toml; \
	git tag "$$next"; \
	git push origin HEAD "refs/tags/$$next"; \
	echo "\033[1m>> Released $$next -- pushed commit + tag <<\033[0m"

# Development workflow: build and fast test (no coverage) for a tight inner loop.
dev: build test-quick

# Full CI check: lint, then the full suite with the combined coverage report.
ci: lint test
