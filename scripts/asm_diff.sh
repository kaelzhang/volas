#!/usr/bin/env bash
# Instruction-level regression gate for the hot kernels.
#
# Builds the `#[no_mangle]` probe wrappers (numeric ma/ema, the string compare scan),
# disassembles each, counts its instructions, and compares to scripts/asm_baseline.txt:
#   exact  — a numeric kernel must stay instruction-identical (any change fails);
#   max    — the string kernel must not increase its per-element instruction count.
#
# This makes the by-hand instruction-level review a permanent, reproducible gate so a
# refactor cannot silently grow a hot loop. The counts are CPU-arch-specific; the
# baseline records its arch and the gate no-ops on a different host.
#
#   bash scripts/asm_diff.sh           # check against the committed baseline
#   ASM_UPDATE=1 bash scripts/asm_diff.sh   # refresh the baseline (reviewed change)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
BASELINE="scripts/asm_baseline.txt"
ARCH="$(uname -m)"
OBJDUMP="${OBJDUMP:-objdump}"

command -v "$OBJDUMP" >/dev/null 2>&1 || { echo "asm-diff: '$OBJDUMP' not found"; exit 2; }

echo ">> building kernel probes (release)..."
cargo build --release -q --example asm_probe_numeric -p volas-compute
cargo build --release -q --example asm_probe_string -p volas-core

NUM=target/release/examples/asm_probe_numeric
STR=target/release/examples/asm_probe_string

# Count the instruction lines of one disassembled function (symbol header to the
# trailing blank line). macOS prefixes a `#[no_mangle]` symbol with one underscore.
count_fn() { # $1 binary  $2 symbol (no leading underscore)
  "$OBJDUMP" -d "$1" 2>/dev/null | awk -v s="<_$2>:" '
    index($0, s) { f = 1; next }
    f && /^$/    { f = 0 }
    f && /^[0-9a-f]+:/ { c++ }
    END { print c + 0 }'
}
# Current instruction count of a probe (kept bash-3.2 portable — no assoc arrays).
cur_count() { # $1 probe
  case "$1" in
    probe_str*) count_fn "$STR" "$1" ;;
    *)          count_fn "$NUM" "$1" ;;
  esac
}

if [[ "${ASM_UPDATE:-0}" == "1" ]]; then
  ma=$(cur_count probe_ma); ema=$(cur_count probe_ema); str=$(cur_count probe_str_eq_scan)
  {
    echo "# Instruction-level baseline for \`make asm-diff\` (arch: $ARCH)."
    echo "# <probe> <kind> <count>: exact = numeric kernel (must stay byte-identical),"
    echo "# max = string kernel (must not increase). Refresh only after a reviewed change."
    echo "probe_ma exact $ma"
    echo "probe_ema exact $ema"
    echo "probe_str_eq_scan max $str"
  } >"$BASELINE"
  echo "asm-diff: baseline written ($ARCH) — ma=$ma ema=$ema str=$str"
  exit 0
fi

[[ -f "$BASELINE" ]] || { echo "asm-diff: no baseline; run 'make asm-diff-update'"; exit 2; }

base_arch="$(sed -n 's/.*arch: \([^)]*\)).*/\1/p' "$BASELINE" | head -1)"
if [[ -n "$base_arch" && "$base_arch" != "$ARCH" ]]; then
  echo "asm-diff: baseline is for '$base_arch', host is '$ARCH' — skipping (counts are arch-specific)."
  exit 0
fi

fail=0
printf '%-20s %8s %8s %8s  %s\n' PROBE CURRENT BASELINE DELTA VERDICT
while read -r probe kind base; do
  [[ "$probe" =~ ^# || -z "$probe" ]] && continue
  cur=$(cur_count "$probe")
  verdict=ok
  if [[ "$kind" == exact && "$cur" != "$base" ]]; then verdict=CHANGED;   fail=1; fi
  if [[ "$kind" == max   && "$cur" -gt "$base" ]]; then verdict=REGRESSED; fail=1; fi
  printf '%-20s %8s %8s %+8d  %s (%s)\n' "$probe" "$cur" "$base" "$((cur - base))" "$verdict" "$kind"
done <"$BASELINE"

if [[ "$fail" == 1 ]]; then
  echo "asm-diff: FAIL — a hot kernel's instruction count moved. Review the change; if"
  echo "          intended and justified, refresh with 'make asm-diff-update'."
  exit 1
fi
echo "asm-diff: PASS — numeric kernels instruction-identical, string kernel not regressed."
