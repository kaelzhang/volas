#!/usr/bin/env bash
#
# Differential test harness (test/test_harness_*.py + the Rust resume-vs-full
# matrix) against the extension built in a given mode:
#
#   release  the shipped profile (no debug assertions, no overflow checks)
#   checked  release optimisation plus `-C debug-assertions -C overflow-checks`, in
#            its own target dir — turns on the core's structural self-checks
#            (`DataFrame::debug_check`) and traps integer overflow, which the
#            release wheel would let wrap silently
#
# Budget knobs (see test/harness/budget.py): VOLAS_HARNESS_PROFILE=short|long,
# VOLAS_HARNESS_EXAMPLES, VOLAS_HARNESS_STEPS, VOLAS_HARNESS_SECONDS (CPU seconds per
# property test). The normal release extension is restored on exit.
#
# Usage: scripts/audit_harness.sh release|checked

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:?usage: audit_harness.sh release|checked}"
PYTHON="${VOLAS_PYTHON:-${PYTHON:-python}}"
PY_PREFIX="$("$PYTHON" -c 'import sys; print(sys.prefix)')"
export VIRTUAL_ENV="$PY_PREFIX" CONDA_PREFIX="$PY_PREFIX"
TESTS=(test/test_harness_machines.py test/test_harness_schedules.py test/test_harness_defects.py)

case "$MODE" in
    release)
        "$PYTHON" -c 'from scripts.build_release import check_environment; import os; check_environment(os.environ)'
        ;;
    checked)
        export RUSTFLAGS="-C debug-assertions -C overflow-checks"
        export CARGO_TARGET_DIR="${HARNESS_CHECKED_TARGET:-target/harness-checked}"
        restore() {
            echo ">> restoring a normal release through the build owner..."
            env -u RUSTFLAGS -u CARGO_TARGET_DIR "$PYTHON" scripts/build_release.py --mode normal \
                --artifacts "$PWD/target/harness-restore-$(date +%Y%m%d-%H%M%S)" --install
        }
        trap restore EXIT
        ;;
    *)
        echo "unknown mode '$MODE' (release|checked)" >&2
        exit 2
        ;;
esac

echo ">> [$MODE] Rust resume-vs-full matrix..."
cargo test -p volas-directive --test resume_matrix --release
echo ">> [$MODE] building the extension..."
if [ "$MODE" = release ]; then
    "$PYTHON" scripts/build_release.py --mode normal --artifacts "$PWD/target/harness-release-$(date +%Y%m%d-%H%M%S)" --install
else
    "$PYTHON" -m maturin develop --release --quiet
fi
echo ">> [$MODE] Python harness (profile ${VOLAS_HARNESS_PROFILE:-short})..."
"$PYTHON" -m pytest "${TESTS[@]}" --benchmark-skip -q -p no:cacheprovider
