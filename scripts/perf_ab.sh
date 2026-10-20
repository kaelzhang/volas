#!/usr/bin/env bash
# Standardized local paired A/B performance comparison. Both wheels live in
# isolated sites; identical fresh processes measure AB, BA, AB, BA, ... pairs.
#
#   scripts/perf_ab.sh <base-ref> [indicator] [section]
#   BASE_SITE=/path/to/prebuilt/site-packages scripts/perf_ab.sh <base-ref>
set -euo pipefail

BASE_REF="${1:?usage: perf_ab.sh <base-ref> [indicator] [section]}"
INDICATOR="${2:-}"
SECTION="${3:-}"
PYTHON="${VOLAS_PYTHON:-python}"
ROOT="$(git rev-parse --show-toplevel)"
ROUNDS="${PERF_AB_ROUNDS:-6}"
"$PYTHON" -c 'import sys; sys.path.insert(0, sys.argv[1]); from scripts.perf_pairs import validate_rounds; validate_rounds(int(sys.argv[2]))' "$ROOT" "$ROUNDS"
TMP="$(mktemp -d)"
cleanup() {
  if [ -n "${PERF_AB_JSON_DIR:-}" ]; then
    mkdir -p "$PERF_AB_JSON_DIR"
    cp "$TMP/"*.json "$TMP/"*.log "$PERF_AB_JSON_DIR/" 2>/dev/null || true
    # PGO inputs and installed sites must survive the temporary measurement tree.
    # Cargo caches are reproducible build intermediates, not release evidence.
    for side in base head; do
      if [ -d "$TMP/$side-build" ]; then
        (cd "$TMP" && tar --exclude='*/target' -cf - "$side-build") | \
          (cd "$PERF_AB_JSON_DIR" && tar -xf -)
      fi
    done
  fi
  rm -rf "$TMP"
}
trap cleanup EXIT
# Optional numba-based comparison libraries must not cache into a shared env.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$TMP/numba}"

# Spread the auto-calibrated budget over six pairs instead of repeating the
# default one-second/item budget twelve times. perf_pairs.py spreads the
# pedantic budget across pairs through the current harness's environment.
BENCH_OPTS=(--benchmark-only --benchmark-group-by=func --benchmark-sort=name -q
            --benchmark-max-time=0.05 --benchmark-min-rounds=5)
FILTER=()
[ -n "$INDICATOR" ] && FILTER+=(--volas-benchmark-indicator="$INDICATOR")
KSELECT="${PERF_AB_KSELECT:-}"
if [ -z "$INDICATOR" ] && [ -z "$SECTION" ] && [ -z "$KSELECT" ]; then
  # Every volas section contributes to the full verdict.
  KSELECT='volas or test_tf_live or test_first_cache or test_range'
  FILTER+=(--volas-cost-checks)
fi
[ -n "$SECTION" ] && KSELECT="test_$SECTION${KSELECT:+ and ($KSELECT)}"
[ -n "$KSELECT" ] && FILTER+=(-k "$KSELECT")

build_site() {
  local tree="$1" side="$2"
  echo ">> building $side into an isolated site..."
  local mode="${BUILD_OPT:-normal}"
  if [ "${PERF_COMPARE:-regression}" = optimization ]; then
    if [ "$side" = base ]; then mode=normal; else mode=pgo; fi
  fi
  "$PYTHON" "$ROOT/scripts/build_release.py" --source "$tree" --mode "$mode" \
      --artifacts "$TMP/$side-build" || return 1
  local stage=normal
  [ "$mode" = pgo ] && stage=use
  cp -R "$TMP/$side-build/$stage/site" "$TMP/$side-site"

}

if [ -n "${BASE_SITE:-}" ]; then
  BASE_IMPORT="$(cd "$BASE_SITE" && pwd)"
  echo ">> using prebuilt BASE ($BASE_REF) from $BASE_IMPORT"
else
  mkdir "$TMP/base"
  git -C "$ROOT" archive "$BASE_REF" | tar -x -C "$TMP/base" || exit 2
  # Independent target dirs prevent stale cross-checkout rlibs.
  build_site "$TMP/base" base || exit 2
  BASE_IMPORT="$TMP/base-site"
fi
build_site "$ROOT" head

"$PYTHON" "$ROOT/scripts/build_pair.py" "$BASE_IMPORT" "$TMP/head-site" \
    --comparison "${PERF_COMPARE:-regression}" --output "$TMP/build-pair.json"

"$PYTHON" "$ROOT/scripts/perf_pairs.py" --base-site "$BASE_IMPORT" \
    --head-site "$TMP/head-site" --root "$ROOT" --output "$TMP" --rounds "$ROUNDS" \
    "${BENCH_OPTS[@]}" ${FILTER[@]+"${FILTER[@]}"}
echo ">> total A/B wall time (builds + measurement): ${SECONDS}s"
echo
if [ -n "$INDICATOR" ]; then
  "$PYTHON" "$ROOT/scripts/bench_indicator_report.py" "$TMP/head.json" "$INDICATOR" --base "$TMP/base.json"
else
  "$PYTHON" "$ROOT/scripts/perf_gate.py" "$TMP/head.json" --base "$TMP/base.json"
fi
