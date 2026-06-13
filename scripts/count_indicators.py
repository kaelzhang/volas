#!/usr/bin/env python3
"""Count the built-in indicators programmatically, from the Rust source of truth.

The total = main commands ∪ sub-command output lines (de-duplicated between the
two) + candlestick patterns. The vocabularies are read from the two enumerable
Rust consts (so a new command / pattern can't drift), and whether a given
(command, sub) form actually exists is decided by the directive validator itself
(volas.directive_stringify, which now rejects anything df[directive] rejects) —
so the count tracks the real command set, not a hand-maintained number.

Usage:
    python scripts/count_indicators.py            # print the count + breakdown
    python scripts/count_indicators.py --check     # also assert README / INDICATORS.md cite it
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import volas

ROOT = Path(__file__).resolve().parent.parent
SPEC_RS = ROOT / "crates/volas-directive/src/spec.rs"
CANDLES_RS = ROOT / "crates/volas-compute/src/indicators/candles/mod.rs"
README = ROOT / "README.md"
README_ZH = ROOT / "README.zh-CN.md"
INDICATORS = ROOT / "INDICATORS.md"

# Canonical sub-command tokens used by the multi-output indicators (after
# canon_sub). The script probes each command with each token and keeps the forms
# the validator accepts, so listing a few extra here is harmless — but a BRAND-NEW
# sub token added to spec.rs must be added here too (the rare case; main commands
# and patterns are read straight from their consts).
SUB_TOKENS = [
    "signal", "histogram", "up", "down", "k", "d", "j", "upper", "lower",
    "quadrature", "leadsine", "fama", "ama", "plus", "minus",
    "r1", "r2", "r3", "s1", "s2", "s3",
    "tenkan", "kijun", "senkou_a", "senkou_b", "chikou", "direction",
    "ar", "br", "ma", "weakr", "midr", "strongr", "weaks", "mids", "strongs",
    "on", "ah", "nh", "nl", "al", "min", "max", "bullish", "bearish",
]


def _const_names(path: Path, const: str) -> list[str]:
    """The string literals of a `pub const <const>: &[&str] = &[ ... ];` block."""
    src = path.read_text()
    m = re.search(rf"pub const {const}\s*:\s*&\[&str\]\s*=\s*&\[(.*?)\];", src, re.S)
    if not m:
        sys.exit(f"error: could not find const {const} in {path}")
    return re.findall(r'"([^"]+)"', m.group(1))


def _canonical(form: str) -> str | None:
    """The canonical name of a directive form, or None if the validator rejects it."""
    try:
        return volas.directive_stringify(form)
    except Exception:
        return None


def count() -> dict:
    commands = _const_names(SPEC_RS, "COMMANDS")
    patterns = _const_names(CANDLES_RS, "CANDLE_PATTERNS")

    mains: set[str] = set()
    subs: set[str] = set()
    pats: set[str] = set()

    for cmd in commands:
        # a main output line exists when the bare command validates (commands that
        # require a sub-command — kdj, stoch, vortex, ... — do not add one).
        c = _canonical(cmd)
        if c is not None:
            mains.add(c)
        for tok in SUB_TOKENS:
            c = _canonical(f"{cmd}.{tok}")
            if c is not None:
                subs.add(c)
    for pat in patterns:
        c = _canonical(f"style.{pat}")
        if c is not None:
            pats.add(c)

    # De-duplicate between mains and subs (a canonical form is counted once).
    all_forms = mains | subs | pats
    return {
        "commands": len(commands),
        "patterns": len(patterns),
        "main_lines": len(mains),
        "sub_lines": len(subs - mains),
        "pattern_lines": len(pats),
        "total": len(all_forms),
        "forms": sorted(all_forms),
    }


def _cites(path: Path, total: int) -> bool:
    """Whether `path` states the indicator total — the number next to the word
    'indicator' (English) or '指标' (Chinese README), so a coincidental number
    elsewhere doesn't count."""
    return re.search(rf"(?<!\d){total}(?!\d)\D{{0,40}}(indicator|指标)", path.read_text(), re.I) is not None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="assert README.md, README.zh-CN.md and INDICATORS.md cite the computed total")
    args = ap.parse_args()

    r = count()
    print(f"commands (main vocab):   {r['commands']}")
    print(f"candle patterns:         {r['patterns']}")
    print(f"  main output lines:     {r['main_lines']}")
    print(f"  sub output lines:      {r['sub_lines']}")
    print(f"  pattern output lines:  {r['pattern_lines']}")
    print(f"TOTAL indicators:        {r['total']}")

    if args.check:
        total = r["total"]
        bad = [p.name for p in (README, README_ZH, INDICATORS) if not _cites(p, total)]
        if bad:
            print(f"\nerror: {', '.join(bad)} do not cite the indicator total {total}; "
                  f"update them (or re-run without --check to see the new number).", file=sys.stderr)
            return 1
        print(f"\nREADME.md, README.zh-CN.md and INDICATORS.md all cite {total} ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
