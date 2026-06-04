files = volas test *.py
test_files = *

.PHONY: install install-rust build build-pkg build-ext clean test coverage coverage-html benchmark lint fix fmt check cargo-test upload publish dev ci

# Install all dependencies (Python + Rust)
install:
	@echo "\033[1m>> Installing Rust toolchain... <<\033[0m"
	@which rustup > /dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
	@rustup update stable
	@echo "\033[1m>> Installing maturin... <<\033[0m"
	@pip install maturin
	@echo "\033[1m>> Installing Python dependencies... <<\033[0m"
	@pip install -e .[dev]

# Install only the Rust toolchain
install-rust:
	@which rustup > /dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
	@rustup update stable

# Build the Rust extension and install the package in-place (development)
build: clean
	@echo "\033[1m>> Building Rust extension... <<\033[0m"
	@maturin develop --release
	@echo "\033[1m>> Build complete! <<\033[0m"

# Build the release package (wheel and sdist) into dist/
build-pkg: clean
	@echo "\033[1m>> Building release package... <<\033[0m"
	@maturin build --release --sdist -o dist
	@echo "\033[1m>> Package built in dist/ <<\033[0m"

# Build the Rust extension only (development mode)
build-ext:
	@maturin develop

# Clean build artifacts
clean:
	rm -rf dist build target/wheels
	rm -rf volas/*.so volas_rs*.so
	rm -rf *.egg-info
	rm -rf .eggs
	rm -rf .coverage .coverage.* htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Run tests (functional; benchmarks are skipped here — see `make benchmark`).
# `--ignore` does not exclude an explicitly-globbed file, so use `--benchmark-skip`.
test:
	pytest -s -v test/test_$(test_files).py --benchmark-skip

# True-union Rust line coverage: `cargo test` ∪ the Python suite exercising the
# compiled extension (see scripts/coverage.sh for why llvm-cov cannot union the
# two builds itself, and why `pytest --cov` is meaningless for a Rust package).
coverage:
	@bash scripts/coverage.sh

# Same, rendered to a browsable HTML report under target/volas-cov/html/.
coverage-html:
	@bash scripts/coverage.sh --html

# Run the 3-way performance benchmark (pandas vs stock-pandas vs volas).
# Depends on `build` so the volas extension is compiled in release mode.
benchmark: build
	pytest test/test_benchmark.py --benchmark-only --benchmark-group-by=param:spec --benchmark-columns=mean,median,ops,rounds --benchmark-sort=name

# Run linters
lint:
	@echo "\033[1m>> Running ruff... <<\033[0m"
	@ruff check $(files)
	@echo "\033[1m>> Running mypy... <<\033[0m"
	@mypy $(files)
	@echo "\033[1m>> Running cargo check... <<\033[0m"
	@cargo check

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

# Development workflow: build and test
dev: build test

# Full CI check
ci: lint cargo-test test
