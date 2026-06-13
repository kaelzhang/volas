#!/usr/bin/env bash
# Win-count A/B across commits — the regression-hunting tool.
#
# The per-indicator volas-vs-TA-Lib ratio carries ±10-15% layout noise, and the
# 5-item perf_gate geomean is noisy too. But the WIN COUNT (how many of the ~158
# covered indicators volas beats) is an aggregate over 158 items, so the per-item
# jitter cancels: bench_noise.sh measures its layout-noise floor at ~±2. That makes
# the win count the right metric for detecting a REAL cross-commit regression.
#
# This builds BASE in a temp worktree and HEAD (the current tree) back-to-back on
# the same machine (so thermal cancels), benchmarks the coverage section of each,
# and prints both win counts plus the indicators whose ratio moved most — the
# candidates that actually regressed.
#
#   scripts/bench_wincount_ab.sh <base-ref>
#
# Read Δ against the ~±2 floor: a Δ of -7 is a real regression worth hunting; a Δ
# of ±2 is noise. Needs the comparison libs installed (`make benchmark` once).
set -euo pipefail

BASE_REF="${1:?usage: bench_wincount_ab.sh <base-ref>}"
PYTHON="${VOLAS_PYTHON:-python}"
ROOT="$(git rev-parse --show-toplevel)"
TMP="$(mktemp -d)"
trap 'git -C "$ROOT" worktree remove --force "$TMP/base" 2>/dev/null || true; rm -rf "$TMP"' EXIT
KSEL='test_coverage and not test_coverage_extended and not test_coverage_after_append'

run_bench() { # $1 = json out
  ( cd "$ROOT" && "$PYTHON" -m pytest test/test_benchmark.py -k "$KSEL" --benchmark-only -q \
      --benchmark-json="$1" >/dev/null 2>&1 )
}

echo ">> building + benchmarking BASE ($BASE_REF) in a temp worktree..."
git -C "$ROOT" worktree add --detach "$TMP/base" "$BASE_REF" >/dev/null
( cd "$TMP/base" && CARGO_TARGET_DIR="$TMP/base/target" "$PYTHON" -m maturin develop --release -q )
run_bench "$TMP/base.json"

echo ">> building + benchmarking HEAD (current tree)..."
( cd "$ROOT" && "$PYTHON" -m maturin develop --release -q )
run_bench "$TMP/head.json"

echo
"$PYTHON" - "$TMP/base.json" "$TMP/head.json" "$BASE_REF" <<'PY'
import json, sys
sys.path.insert(0, 'scripts'); import benchmark_report as br
def ratios(p):
    g = br.parse(json.loads(open(p).read()))
    return {ind: m[2] for ind, e in g.get('coverage', {}).items() if (m := br._coverage_ratio(e))}
base, head = ratios(sys.argv[1]), ratios(sys.argv[2])
wb = sum(1 for v in base.values() if v > 1.0)
wh = sum(1 for v in head.values() if v > 1.0)
print(f"=== win-count: BASE({sys.argv[3]}) = {wb}/{len(base)}   HEAD = {wh}/{len(head)}   Δ = {wh-wb:+d} ===")
print("   (layout-noise floor ~±2; |Δ| > 2 is a real change)")
common = [k for k in base if k in head and base[k] > 0]
flipped = [k for k in common if base[k] > 1.0 and head[k] <= 1.0]
print(f"\nwin -> loss between BASE and HEAD ({len(flipped)}): {sorted(flipped)}")
moved = sorted(common, key=lambda k: head[k] / base[k])
print("\nbiggest per-indicator regressions (HEAD/BASE ratio ascending):")
for k in moved[:18]:
    tag = "  <-- flipped to loss" if k in flipped else ""
    print(f"  {k:24s} base={base[k]:.2f}x  head={head[k]:.2f}x  rel={head[k]/base[k]:.3f}{tag}")
PY
