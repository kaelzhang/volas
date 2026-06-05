#!/usr/bin/env bash
#
# True-union Rust line coverage for volas (cargo test ∪ pytest).
#
# volas's Python layer is a thin re-export over the Rust extension, so
# `pytest --cov volas` only sees volas/__init__.py and reports a meaningless
# "100%". The meaningful coverage is the Rust crates — and they are exercised
# two complementary ways: directly by `cargo test`, and through the compiled
# `.so` by the Python suite. Each reaches lines the other does not (e.g. parser
# edge cases and `lookback.rs` are pytest-only; the indicator dispatch is
# cargo-only).
#
# We want their UNION: a line counts as covered if it ran in either suite.
# `llvm-cov` cannot compute that directly — the `cargo test` build and the
# maturin cdylib have different coverage-mapping hashes, so it counts each as a
# separate instantiation and reports their *average* (which penalises files one
# suite covers well and the other does not). So we export per-line LCOV from
# each suite independently and union them in `lcov_union.py`.
#
# Requires cargo-llvm-cov (`cargo install cargo-llvm-cov`) and llvm-tools-preview
# (`rustup component add llvm-tools-preview`). Run inside the activated Python
# env (so maturin/pytest and CONDA_PREFIX/VIRTUAL_ENV are set).
#
# Usage: scripts/coverage.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PROF_DIR="target/volas-cov"
rm -rf "$PROF_DIR"
mkdir -p "$PROF_DIR"

# Locate the LLVM tools shipped with the active toolchain (llvm-tools-preview).
HOST="$(rustc -vV | sed -n 's/^host: //p')"
LLVM_BIN="$(rustc --print sysroot)/lib/rustlib/$HOST/bin"
PROFDATA="$LLVM_BIN/llvm-profdata"
COV="$LLVM_BIN/llvm-cov"
if [ ! -x "$PROFDATA" ] || [ ! -x "$COV" ]; then
    echo "error: llvm-tools not found in $LLVM_BIN" >&2
    echo "       run: rustup component add llvm-tools-preview" >&2
    exit 1
fi

restore() {
    find . -name '*.profraw' -not -path './target/*' -delete 2>/dev/null || true
    echo ">> restoring the normal extension (maturin develop --release)..."
    env -u RUSTFLAGS -u LLVM_PROFILE_FILE \
        maturin develop --release >/dev/null 2>&1 || true
}
trap restore EXIT

# --- 1. cargo test side ----------------------------------------------------
# cargo-llvm-cov runs the workspace unit/integration tests under instrumentation
# and emits LCOV in one self-contained step (its own target dir + profiles).
echo ">> cargo test coverage..."
cargo llvm-cov --workspace --exclude volas-python --quiet \
    --lcov --output-path "$PROF_DIR/cargo.lcov" \
    --ignore-filename-regex 'volas-python'

# --- 2. pytest / .so side --------------------------------------------------
# Instrument the cdylib too (RUSTFLAGS, since the wrapper alone does not reach
# maturin's build), run the suite, then export LCOV for the exact .so that ran.
echo ">> building instrumented extension..."
export RUSTFLAGS="${RUSTFLAGS:-} -Cinstrument-coverage"
maturin develop --quiet
# maturin may install volas_rs as a package shim (`from .volas_rs import *`), so
# the native object lives inside the package dir, not at `.origin`.
SO="$(python - <<'PY'
import importlib.util as u, os, glob
s = u.find_spec("volas_rs")
origin = (s.origin or "") if s else ""
if origin.endswith((".so", ".dylib", ".pyd")):
    print(origin)
elif origin:
    d = os.path.dirname(origin)
    hits = glob.glob(os.path.join(d, "**", "*.so"), recursive=True) \
        + glob.glob(os.path.join(d, "**", "*.dylib"), recursive=True)
    print(hits[0] if hits else "")
PY
)"
if [ -z "$SO" ] || [ ! -f "$SO" ]; then
    echo "error: could not locate the volas_rs extension (.so)" >&2
    exit 1
fi

echo ">> running Python suite against the instrumented extension..."
LLVM_PROFILE_FILE="$PROF_DIR/pytest-%p-%m.profraw" \
    pytest test/ --benchmark-skip -q

"$PROFDATA" merge -sparse "$PROF_DIR"/pytest-*.profraw -o "$PROF_DIR/pytest.profdata"
"$COV" export --format=lcov --instr-profile="$PROF_DIR/pytest.profdata" "$SO" \
    --ignore-filename-regex '/\.cargo/|/rustc/|library/(core|std|alloc)|volas-python' \
    > "$PROF_DIR/pytest.lcov"

# --- 3. union + report -----------------------------------------------------
echo ">> combined coverage (true union):"
python scripts/lcov_union.py "$PROF_DIR/cargo.lcov" "$PROF_DIR/pytest.lcov" "$PROF_DIR/union.lcov"

# Optional HTML render (requires lcov's `genhtml`).
if [ "${1:-}" = "--html" ]; then
    if command -v genhtml >/dev/null 2>&1; then
        echo ">> rendering HTML to target/volas-cov/html/ ..."
        genhtml --quiet --output-directory "$PROF_DIR/html" "$PROF_DIR/union.lcov"
        echo "   open $PROF_DIR/html/index.html"
    else
        echo ">> genhtml not found (install lcov); merged LCOV at $PROF_DIR/union.lcov"
    fi
fi
