#!/usr/bin/env bash
# Standardized local A/B performance comparison — THE pipeline for "did my
# change move performance?" questions. Never ad-hoc timing scripts: this runs
# the same harness, same machine, back-to-back, exactly like the CI perf gate.
#
#   scripts/perf_ab.sh <base-ref>                     # full suite, perf_gate verdict
#   scripts/perf_ab.sh <base-ref> <indicator>         # one indicator, compact report
#   scripts/perf_ab.sh <base-ref> '' <section>        # one report section, perf_gate verdict
#
# The base ref is built in a TEMPORARY GIT WORKTREE (the working tree is never
# touched — no stashing), benchmarked, then HEAD (the current tree, including
# uncommitted changes) is rebuilt and benchmarked. Nothing is archived.
set -euo pipefail

BASE_REF="${1:?usage: perf_ab.sh <base-ref> [indicator] [section]}"
INDICATOR="${2:-}"
SECTION="${3:-}"
PYTHON="${VOLAS_PYTHON:-python}"
# maturin needs VIRTUAL_ENV / CONDA_PREFIX to find the interpreter — set them from
# the resolved python so this runs from a bare shell (mirrors the Makefile).
PY_PREFIX="$("$PYTHON" -c 'import sys; print(sys.prefix)')"
export VIRTUAL_ENV="$PY_PREFIX" CONDA_PREFIX="$PY_PREFIX"
ROOT="$(git rev-parse --show-toplevel)"
TMP="$(mktemp -d)"
trap 'git -C "$ROOT" worktree remove --force "$TMP/base" 2>/dev/null || true; rm -rf "$TMP"' EXIT

BENCH_OPTS=(--benchmark-only --benchmark-group-by=func,param:indicator --benchmark-sort=name -q)
FILTER=()
[ -n "$INDICATOR" ] && FILTER+=(--volas-benchmark-indicator="$INDICATOR")
[ -n "$SECTION" ] && FILTER+=(-k "test_$SECTION")

echo ">> building + benchmarking BASE ($BASE_REF) in a temp worktree..."
git -C "$ROOT" worktree add --detach "$TMP/base" "$BASE_REF" >/dev/null
# The base build gets its OWN target dir: sharing $ROOT/target between two
# checkouts of the same workspace let cargo hand the HEAD build a stale rlib
# from the base tree (observed: HEAD's volas-directive linked against the
# base's volas-compute, missing a newly added module). A cold build is the
# price of a sound comparison.
( cd "$TMP/base" && CARGO_TARGET_DIR="$TMP/base/target" "$PYTHON" -m maturin develop --release -q )
( cd "$ROOT" && "$PYTHON" -m pytest test/test_benchmark.py "${BENCH_OPTS[@]}" ${FILTER[@]+"${FILTER[@]}"} \
    --benchmark-json="$TMP/base.json" >/dev/null )

echo ">> building + benchmarking HEAD (current tree)..."
( cd "$ROOT" && "$PYTHON" -m maturin develop --release -q )
( cd "$ROOT" && "$PYTHON" -m pytest test/test_benchmark.py "${BENCH_OPTS[@]}" ${FILTER[@]+"${FILTER[@]}"} \
    --benchmark-json="$TMP/head.json" >/dev/null )

echo
if [ -n "$INDICATOR" ]; then
  "$PYTHON" "$ROOT/scripts/bench_indicator_report.py" "$TMP/head.json" "$INDICATOR" --base "$TMP/base.json"
else
  "$PYTHON" "$ROOT/scripts/perf_gate.py" "$TMP/head.json" --base "$TMP/base.json"
fi
