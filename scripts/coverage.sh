#!/usr/bin/env bash
#
# End-to-end Rust line coverage for volas.
#
# volas's Python layer is a thin re-export over the Rust extension, so
# `pytest --cov volas` only sees volas/__init__.py (4 lines) and reports a
# meaningless "100%". The meaningful coverage is the Rust crates — and most of
# them are exercised through the compiled extension by the Python test suite, not
# by `cargo test` alone. This script measures the *combined* coverage:
#
#   1. instrument the Rust build (incl. the maturin-built .cdylib),
#   2. run the Rust unit tests (cargo test) and the Python suite (pytest),
#   3. merge both profiles and report.
#
# Requires cargo-llvm-cov (`cargo install cargo-llvm-cov`) and llvm-tools-preview
# (`rustup component add llvm-tools-preview`). Run inside the activated Python
# env (so maturin/pytest and CONDA_PREFIX/VIRTUAL_ENV are set).
#
# Usage: scripts/coverage.sh [extra `cargo llvm-cov report` args]
#   scripts/coverage.sh                 # summary table
#   scripts/coverage.sh --html          # HTML report under target/llvm-cov/

set -euo pipefail
cd "$(dirname "$0")/.."

report_args=("$@")
[ ${#report_args[@]} -eq 0 ] && report_args=(--summary-only)

cargo llvm-cov clean --workspace

# Export cargo-llvm-cov's environment (RUSTC_WRAPPER, profile paths, target dir).
# Use `eval "$(...)"` rather than `source <(...)`: the process-substitution FIFO
# can be closed early (SIGPIPE), yielding a partial env where the profile dir and
# the report's search dir disagree.
eval "$(cargo llvm-cov show-env --sh 2>/dev/null)"
# The wrapper alone does not reliably reach maturin's cdylib build, so also force
# instrumentation via RUSTFLAGS — this is what makes the Python suite's exercise
# of the extension show up in the report.
export RUSTFLAGS="${RUSTFLAGS:-} -Cinstrument-coverage"

restore() {
    # cargo-test doctests scatter default_*.profraw into the crate dirs; tidy them.
    find . -name '*.profraw' -not -path './target/*' -delete 2>/dev/null || true
    # Leave the dev env with a normal (fast, non-instrumented) extension.
    echo ">> restoring the normal extension (maturin develop --release)..."
    env -u RUSTFLAGS -u RUSTC_WRAPPER -u CARGO_LLVM_COV -u CARGO_LLVM_COV_SHOW_ENV \
        -u LLVM_PROFILE_FILE maturin develop --release >/dev/null 2>&1 || true
}
trap restore EXIT

echo ">> building instrumented extension..."
maturin develop >/dev/null
echo ">> running Rust unit tests..."
cargo test --workspace --exclude volas-python >/dev/null
echo ">> running Python suite against the instrumented extension..."
pytest test/ --benchmark-skip -q >/dev/null
echo ">> combined coverage:"
cargo llvm-cov report "${report_args[@]}" --ignore-filename-regex 'volas-python'
