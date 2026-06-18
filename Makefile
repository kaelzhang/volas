files = volas test $(wildcard *.py)
test_files = *
# Prefer an explicit VOLAS_PYTHON (the conda-env interpreter) so `make test` /
# `make types` reproduce identically across shells; else the PATH python.
PYTHON ?= $(or $(VOLAS_PYTHON),python)
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
MATURIN ?= $(PYTHON) -m maturin
# Lazy (`=`, not `:=`): the interpreter is probed only when a maturin recipe actually
# expands MATURIN_DEVELOP_ENV — so targets that need no Python (asm-diff, lint) don't
# print `python: command not found` when neither `python` nor VOLAS_PYTHON is set.
PY_PREFIX = $(shell $(PYTHON) -c "import sys; print(sys.prefix)")
MATURIN_DEVELOP_ENV = VIRTUAL_ENV="$(PY_PREFIX)" CONDA_PREFIX="$(PY_PREFIX)"

.PHONY: install install-rust build build-pkg build-ext clean test test-quick count-indicators coverage coverage-html benchmark perf-ab asm-diff asm-diff-update anime-fonts anime lint fix fmt check cargo-test upload publish bump dev ci

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

# Print the built-in indicator count (main commands + sub-command lines + candle
# patterns), derived from the Rust source; `--check` (run by the test suite)
# asserts README / INDICATORS.md cite it.
count-indicators:
	@$(PYTHON) scripts/count_indicators.py

# True-union Rust line coverage: `cargo test` ∪ the Python suite exercising the
# compiled extension (see scripts/coverage.sh for why llvm-cov cannot union the
# two builds itself, and why `pytest --cov` is meaningless for a Rust package). Runs
# both suites and prints the combined per-file + total report.
coverage:
	@VOLAS_PYTHON="$(PYTHON)" bash scripts/coverage.sh

# Same, rendered to a browsable HTML report under target/volas-cov/html/.
coverage-html:
	@VOLAS_PYTHON="$(PYTHON)" bash scripts/coverage.sh --html

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
# Every full run is persisted to .benchmarks/<date>-<time>-<short-commit>[-dirty]/ (the
# whole dir is gitignored), each holding benchmark.json + report.html + meta.txt,
# so a given Git commit's performance can be retrieved and compared later (e.g.
#   python scripts/perf_gate.py .benchmarks/<new>/benchmark.json \
#       --base .benchmarks/<old>/benchmark.json
# ). The latest run is also mirrored to .benchmarks/last.json + ./benchmark-report.html.
BENCH_OPTS := --benchmark-only --benchmark-group-by=func,param:indicator \
              --benchmark-columns=mean,median,ops,rounds --benchmark-sort=name
BENCH_STAMP := $(shell date +%Y-%m-%d-%H%M%S)-$(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)$(shell git diff --quiet HEAD 2>/dev/null || echo -dirty)
BENCH_DIR := .benchmarks/$(BENCH_STAMP)
benchmark: build
	@echo "\033[1m>> Installing dev + benchmark comparison libraries... <<\033[0m"
	@$(PYTHON) -c "import tomllib; e=tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']; print('\n'.join(e['dev'] + e['benchmark']))" | $(PIP) install -q -r /dev/stdin
ifdef INDICATOR
	@$(eval _IND_JSON := $(shell mktemp -t volas-ind-bench).json)
	$(PYTEST) test/test_benchmark.py $(BENCH_OPTS) --volas-benchmark-indicator="$(INDICATOR)" \
	    --benchmark-json=$(_IND_JSON)
	@echo
	@$(PYTHON) scripts/bench_indicator_report.py $(_IND_JSON) "$(INDICATOR)"
	@rm -f $(_IND_JSON)
else
	@mkdir -p $(BENCH_DIR)
	$(PYTEST) test/test_benchmark.py $(BENCH_OPTS) --benchmark-json=$(BENCH_DIR)/benchmark.json
	@$(PYTHON) scripts/benchmark_report.py $(BENCH_DIR)/benchmark.json $(BENCH_DIR)/report.html
	@$(PYTHON) scripts/bench_trim.py $(BENCH_DIR)/benchmark.json   # drop ~1GB of per-round data; keep stats
	@printf 'commit: %s\ndate:   %s\ndirty:  %s\nmethodology: %s\n' "$$(git rev-parse HEAD 2>/dev/null || echo none)" "$$(date -u +%FT%TZ)" "$$(git diff --quiet HEAD 2>/dev/null && echo no || echo yes)" "$$($(PYTHON) -c 'from test.test_benchmark import METHODOLOGY; print(METHODOLOGY)')" > $(BENCH_DIR)/meta.txt
	@$(PYTHON) scripts/bench_leader.py $(BENCH_DIR)/benchmark.json # compare vs the all-time leader; crown if better
	@cp $(BENCH_DIR)/benchmark.json .benchmarks/last.json
	@cp $(BENCH_DIR)/report.html benchmark-report.html
	@echo "\033[1m>> Benchmark archived to $(BENCH_DIR)/ <<\033[0m"
endif

# Standardized local A/B performance comparison (THE pipeline for "did my
# change move performance?"): builds BASE in a temp worktree, benchmarks it,
# rebuilds HEAD (the current tree, incl. uncommitted changes), benchmarks it,
# and prints the verdict — perf_gate for the full suite, a compact normalized
# report for a single indicator. Same harness, same machine, back-to-back
# (exactly the CI gate's method). Nothing is archived.
#
#   make perf-ab BASE=HEAD~1                       # full suite
#   make perf-ab BASE=HEAD~1 INDICATOR=bop         # one indicator
BASE ?= HEAD~1
perf-ab:
	@VOLAS_PYTHON=$(PYTHON) bash scripts/perf_ab.sh "$(BASE)" "$(INDICATOR)"

# Instruction-level regression gate: disassemble the hot-kernel probes and compare
# each instruction count to scripts/asm_baseline.txt (numeric kernels must stay
# byte-identical, the string kernel must not grow its per-element count). Arch-specific;
# no-ops on a host whose arch differs from the baseline's.
#
#   make asm-diff           # check against the committed baseline
#   make asm-diff-update     # refresh the baseline after a reviewed, justified change
asm-diff:
	@bash scripts/asm_diff.sh

asm-diff-update:
	@ASM_UPDATE=1 bash scripts/asm_diff.sh

# Download/check the fonts used by the README / GitHub Pages animated GIFs.
anime-fonts:
	@$(PIP) install -q pillow
	@$(PYTHON) docs/animated_gif/generate_gif.py --ensure-fonts --check-fonts --strict-fonts --no-render

# Generate the README / GitHub Pages animated explainer GIFs locally and open a
# preview when the host has a desktop opener. The GIF files are generated assets
# and are ignored by git; GitHub Pages regenerates them during its own build.
anime: anime-fonts
	@$(PYTHON) docs/animated_gif/generate_gif.py --strict-fonts
	@echo "\033[1m>> Generated docs/animated_gif/after-append-indicator-en.gif <<\033[0m"
	@echo "\033[1m>> Generated docs/animated_gif/after-append-indicator-zh-cn.gif <<\033[0m"
	@if command -v open >/dev/null 2>&1; then \
		open docs/animated_gif/after-append-indicator-en.gif docs/animated_gif/after-append-indicator-zh-cn.gif; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open docs/animated_gif/after-append-indicator-en.gif >/dev/null 2>&1 || true; \
		xdg-open docs/animated_gif/after-append-indicator-zh-cn.gif >/dev/null 2>&1 || true; \
	else \
		echo "Preview manually from docs/animated_gif/"; \
	fi

# Run linters
lint:
	@echo "\033[1m>> Running ruff... <<\033[0m"
	@ruff check $(files)
	@echo "\033[1m>> Running mypy (package)... <<\033[0m"
	@mypy volas
	@echo "\033[1m>> Running cargo clippy (warnings are errors)... <<\033[0m"
	@cargo clippy --workspace --all-targets -- -D warnings

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

# Publish the Rust crates to crates.io in dependency order. `volas-python` is the
# PyO3 cdylib (publish = false) and is skipped; the `volas` facade goes last since
# it depends on all the others. `cargo publish` builds + verifies each crate and
# waits for it to appear in the index before the next dependent crate is uploaded.
# Run after `make bump` so the workspace + dep versions agree. DRY=1 for a dry run.
cargo-publish:
	@flags="--locked"; [ -n "$(DRY)" ] && flags="$$flags --dry-run"; \
	for c in volas-core volas-arrow volas-compute volas-directive volas-time volas-io volas; do \
		echo "\033[1m>> cargo publish $$c $$flags <<\033[0m"; \
		cargo publish -p $$c $$flags || exit 1; \
	done

# Upload to PyPI
upload:
	twine upload --config-file ~/.pypirc -r pypi dist/* --verbose

# Publish (build package + upload)
publish:
	make build-pkg
	make upload

# Bump the workspace version, commit, tag (no `v` prefix), then push the commit and
# the tag to origin. The pushed tag triggers the GitHub release workflow, which
# verifies the tag matches the Cargo.toml version, then publishes the crates to
# crates.io and the wheels to PyPI — both via Trusted Publishing (OIDC, no stored
# tokens). Usage: make bump TYPE={major|minor|patch}
bump:
	@test -n "$(TYPE)" || { echo "TYPE is required: make bump TYPE={major|minor|patch}"; exit 1; }
	@test -z "$$(git status --porcelain)" || { echo "working tree is not clean -- commit or stash first"; exit 1; }
	@next=$$(python3 scripts/bump_version.py --next "$(TYPE)") || exit 1; \
	if git rev-parse -q --verify "refs/tags/$$next" >/dev/null 2>&1; then \
		echo "tag $$next already exists -- aborting"; exit 1; \
	fi; \
	echo "\033[1m>> Bumping version to $$next <<\033[0m"; \
	python3 scripts/bump_version.py "$(TYPE)" >/dev/null || exit 1; \
	git add Cargo.toml Cargo.lock; \
	git commit -m "chore(release): $$next" -- Cargo.toml Cargo.lock; \
	git tag "$$next"; \
	git push origin HEAD "refs/tags/$$next"; \
	echo "\033[1m>> Released $$next -- pushed commit + tag <<\033[0m"

# Development workflow: build and fast test (no coverage) for a tight inner loop.
dev: build test-quick

# Full CI check: lint, then the full suite with the combined coverage report.
ci: lint test
