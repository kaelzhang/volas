#!/usr/bin/env bash
# Hot-path assembly: the DISCOVERY-and-GATE half of volas's perf pipeline, for the
# BROAD hot-path functions (the per-bar fold, the window kernels) that are too large
# to inline into an `asm-diff` wrapper probe. It regenerates each core crate's release
# asm (`rustc --emit asm`) and counts each hot-path function's OWN symbol directly.
#
# Three modes (1st arg, default `dump`):
#   dump    — extract each function's disassembly (label -> next `.cfi_endproc`) to
#             target/hot-asm/<fn>.s and print an instruction-count table. Read these to
#             spot per-bar heap allocations (`bl _..._alloc`), un-elided bounds-check
#             panics (`b.hs ...; bl ...panic`), broken inlining, spills, or missed SIMD.
#   check   — compare each GATED function's count to scripts/hot_asm_baseline.txt and
#             FAIL if it grew (a `max` gate: a hot path must not get heavier silently).
#   update  — rewrite the baseline (after a reviewed, justified change).
#
# This is the broad-function GATE; `make asm-diff` stays the gate for exact instruction counts for the
# tiny ma/ema/string kernels. Counts are arch- and compiler-specific (the baseline
# records the arch and the gate reports unsupported on a different host).
#
#   make hot-asm                 # dump the whole inventory (FN=combine_at for one)
#   make hot-asm-check            # gate the GATED functions against the baseline
#   make hot-asm-update           # refresh the baseline
#
# INVENTORY (the single source of truth for "what is a hot path"): <crate>|<fn>|<gate>|<why>
#   gate = `loop32` -> whole SIMD scan loop instructions per 32 input rows.
#   gate = `max` -> instruction-count must not increase;  `-` -> dump only (e.g. the
#   indicator `dispatch` switch legitimately grows as commands are added).
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
"${VOLAS_PYTHON:-python3}" -c 'import sys; sys.path.insert(0,"scripts"); from build_release import check_environment; from build_config import check_cargo_config; import os; from pathlib import Path; check_environment(os.environ); check_cargo_config(Path.cwd())'
[ "$(rustc --version | awk '{print $2}')" = 1.96.0 ] || { echo "asm: requires rustc 1.96.0" >&2; exit 2; }
if [ -n "${PGO_SITE:-}" ]; then
  "${VOLAS_PYTHON:-python3}" scripts/pgo_asm.py --site "$PGO_SITE" --output "${PGO_ASM_OUT:?set PGO_ASM_OUT for the separate final-library report}"
fi
ARCH="$(uname -m)"
MODE="${1:-dump}"
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
OUT="$TARGET/hot-asm"
mkdir -p "$TARGET"
ASM_DIR="$(mktemp -d "$TARGET/hot-asm-build.XXXXXX")"
trap 'rm -rf "$ASM_DIR"' EXIT
BASELINE="scripts/hot_asm_baseline.txt"
FN_FILTER="${FN:-}"

INVENTORY="$(cat scripts/hot_asm_inventory.txt)"

# --- emit release asm once per distinct crate (bash-3.2 portable; no assoc arrays) ---
emit_asm() {
  local crates
  crates="$(printf '%s\n' "$INVENTORY" | awk -F'|' -v mode="$MODE" 'NF>=4 && (mode=="dump" || $3=="max" || $3=="loop32"){print $1}' | sort -u)"
  for crate in $crates; do
    echo ">> emitting release asm for $crate ..." >&2
    cargo rustc --release -q -p "$crate" -- \
      --emit "asm=$ASM_DIR/$crate.s" -C "llvm-args=-x86-asm-syntax=intel"
  done
}
asm_of() { [ -s "$ASM_DIR/$1.s" ] && printf '%s\n' "$ASM_DIR/$1.s"; }

# Instruction count of `fn`'s first symbol in `crate`'s asm; also writes its body to
# target/hot-asm/<fn>.s when $1 == keep.
count_fn() { # $1 keep|nokeep  $2 crate  $3 fn  $4 max|loop32|-
  local asm body
  asm="$(asm_of "$2")"
  [ -n "$asm" ] || { echo "hot-asm: missing assembly for $2" >&2; return 2; }
  body="$(awk -v pat="$3" '
    $0 ~ ("[_A-Za-z].*" pat "1[0-9]h[0-9a-f]+E:$") && !f { f = 1 }
    f { print }
    f && /\.cfi_endproc/ { exit }
  ' "$asm")"
  [ -n "$body" ] || { echo "hot-asm: missing symbol $3" >&2; return 2; }
  if [ "$1" = keep ]; then mkdir -p "$OUT"; printf '%s\n' "$body" > "$OUT/$3.s"; fi
  if [ "${4:-max}" = loop32 ]; then
    printf '%s\n' "$body" | "${VOLAS_PYTHON:-python3}" scripts/asm_symbols.py --source-count \
      --comparisons "$([ "$3" = any_missing_f32 ] && echo 8 || echo 16)"
  else
    printf '%s\n' "$body" | "${VOLAS_PYTHON:-python3}" scripts/asm_symbols.py --source-count
  fi
}

baseline_count() { sed -n "s/^$1 ${2:-max} \\([0-9]*\\).*/\\1/p" "$BASELINE" 2>/dev/null | head -1; }

# --- update: write the baseline (gated functions only) ---
if [ "$MODE" = update ]; then
  emit_asm
  {
    echo "# Instruction-level baseline for the broad hot-path functions (arch: $ARCH)."
    echo "# <fn> max <count>: the function's release-asm instruction count must not"
    echo "# increase. Counted from rustc --emit asm. Refresh only after a reviewed change."
    printf '%s\n' "$INVENTORY" | awk -F'|' 'NF>=4 && ($3=="max" || $3=="loop32")' | while IFS='|' read -r crate fn gate why; do
      n="$(count_fn nokeep "$crate" "$fn" "$gate")"
      echo "$fn $gate $n"
    done
  } > "$BASELINE"
  echo "hot-asm: baseline written ($ARCH)"
  exit 0
fi

# --- check: gate gated functions against the baseline ---
if [ "$MODE" = check ]; then
  [ -f "$BASELINE" ] || { echo "hot-asm: no baseline; run 'make hot-asm-update'"; exit 2; }
  base_arch="$(sed -n 's/.*arch: \([^)]*\)).*/\1/p' "$BASELINE" | head -1)"
  if [ -n "$base_arch" ] && [ "$base_arch" != "$ARCH" ]; then
    echo "hot-asm: baseline is for '$base_arch', host is '$ARCH' — unsupported (arch-specific)."
    exit 2
  fi
  emit_asm
  mkdir -p "$OUT"; rm -f "$OUT/.hot_fail"   # the per-regression flag is written here under set -e
  fail=0
  printf '%-20s %8s %8s %8s  %s\n' FUNCTION CURRENT BASELINE DELTA VERDICT
  printf '%s\n' "$INVENTORY" | awk -F'|' 'NF>=4 && ($3=="max" || $3=="loop32")' | while IFS='|' read -r crate fn gate why; do
    cur="$(count_fn nokeep "$crate" "$fn" "$gate")"; [ -n "$cur" ] || exit 2
    base="$(baseline_count "$fn" "$gate")"; [ -n "$base" ] || { echo "hot-asm: missing baseline $fn" >&2; exit 2; }
    verdict=ok
    [ "$cur" -gt "$base" ] && { verdict=REGRESSED; echo REGRESSED > "$OUT/.hot_fail"; }
    printf '%-20s %8s %8s %+8d  %s\n' "$fn" "$cur" "$base" "$((cur - base))" "$verdict"
  done
  if [ -f "$OUT/.hot_fail" ]; then
    rm -f "$OUT/.hot_fail"
    echo "hot-asm: FAIL — a hot-path function's instruction count grew. Review the change;"
    echo "         if intended and justified, refresh with 'make hot-asm-update'."
    exit 1
  fi
  echo "hot-asm: PASS — no gated hot-path function regressed."
  exit 0
fi

# --- dump (default): extract + tabulate, with baseline delta when available ---
emit_asm
rm -rf "$OUT"; mkdir -p "$OUT"
printf '%-20s %-15s %8s %9s  %s\n' FUNCTION CRATE INSTRS VS-BASE HOT-PATH
printf '%s\n' "$INVENTORY" | awk -F'|' 'NF>=4' | while IFS='|' read -r crate fn gate why; do
  [ -n "$FN_FILTER" ] && [[ "$fn" != *"$FN_FILTER"* ]] && continue
  n="$(count_fn keep "$crate" "$fn" "$gate")"; n="$n"
  delta="-"
  if [ "$gate" != - ]; then
    base="$(baseline_count "$fn" "$gate")"
    [ -n "$base" ] && delta="$(printf '%+d' "$((n - base))")"
  fi
  printf '%-20s %-15s %8s %9s  %s\n' "$fn" "$crate" "$n" "$delta" "$why"
done
echo
echo "full disassembly under $OUT/  (arch: $ARCH, release profile)"
