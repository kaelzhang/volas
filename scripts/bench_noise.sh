#!/usr/bin/env bash
# Measure the benchmark's WIN-COUNT NOISE FLOOR.
#
# The "volas beats TA-Lib on N/158" headline is layout-sensitive: changing *any*
# code relays out the whole binary, so an indicator we never touched can run
# ±10-15% faster/slower purely from instruction-cache alignment. Many indicators
# sit right on the 1.00x win/loss line, so that jitter flips several of them every
# build. This script quantifies that floor: it rebuilds HEAD with K *behaviour-
# neutral* layout perturbations (a dead `#[no_mangle]` function that shifts code
# placement but changes no real path), benchmarks the coverage section of each,
# and reports the spread of the win count.
#
# Read the result as: any A/B win-count difference SMALLER than this spread is
# noise, not a real change. (e.g. if the floor is 8, then "153 vs 137" — a gap of
# 16 — is only ~2x the noise, and most of it is layout/thermal, not a regression.)
#
#   scripts/bench_noise.sh [K]      # K layout samples (default 4)
#
# Needs the dev + comparison libs already installed (run `make benchmark` once
# first). Restores the perturbed source on exit, even if interrupted.
set -euo pipefail

K="${1:-4}"
PYTHON="${VOLAS_PYTHON:-python}"
ROOT="$(git rev-parse --show-toplevel)"
# A leaf of the hot crate: appending here shifts the indicator code's layout.
PAD_FILE="$ROOT/crates/volas-compute/src/lib.rs"
TMP="$(mktemp -d)"
trap 'git -C "$ROOT" checkout -- "$PAD_FILE" 2>/dev/null || true; rm -rf "$TMP"' EXIT

# coverage section only (the win-count source); skip the extended / after-append variants.
KSEL='test_coverage and not test_coverage_extended and not test_coverage_after_append'

win_count() {
  "$PYTHON" - "$1" <<'PY'
import json, sys
sys.path.insert(0, 'scripts'); import benchmark_report as br
g = br.parse(json.loads(open(sys.argv[1]).read()))
rows = [m[2] for e in g.get('coverage', {}).values() if (m := br._coverage_ratio(e))]
print(f"{sum(1 for r in rows if r > 1.0)}/{len(rows)}")
PY
}

wins=()
for i in $(seq 1 "$K"); do
  echo ">> [$i/$K] perturb layout -> rebuild --release -> benchmark coverage..."
  git -C "$ROOT" checkout -- "$PAD_FILE"
  # A unique, growing block of dead code: same behaviour, different code placement.
  {
    echo ""
    echo "// bench_noise: behaviour-neutral layout perturbation (sample $i); removed on exit."
    for j in $(seq 1 "$((i * 3))"); do
      echo "#[no_mangle] pub fn _noise_pad_${i}_${j}() -> u64 { ${i} * 1009 + ${j} }"
    done
  } >> "$PAD_FILE"
  ( cd "$ROOT" && "$PYTHON" -m maturin develop --release -q )
  ( cd "$ROOT" && "$PYTHON" -m pytest test/test_benchmark.py -k "$KSEL" --benchmark-only -q \
      --benchmark-json="$TMP/run_$i.json" >/dev/null 2>&1 )
  w="$(win_count "$TMP/run_$i.json")"
  echo "   sample $i win-count = $w"
  wins+=("${w%%/*}")
done
git -C "$ROOT" checkout -- "$PAD_FILE"

echo ""
echo "=== WIN-COUNT NOISE FLOOR (same source, ${K} layout perturbations) ==="
echo "samples: ${wins[*]}"
printf '%s\n' "${wins[@]}" | "$PYTHON" -c "
import sys
v = sorted(int(x) for x in sys.stdin)
n = len(v); mean = sum(v)/n
import statistics
sd = statistics.pstdev(v) if n > 1 else 0.0
print(f'min={v[0]} max={v[-1]} spread={v[-1]-v[0]} median={v[n//2]} mean={mean:.1f} stdev={sd:.1f}')
print(f'>> any win-count change <= {v[-1]-v[0]} indicators is within this layout-noise floor <<')
"
