"""Training-counter coverage and fail-closed use-build diagnostics."""

import re


def canonical(symbol):
    # Preserve Rust disambiguators: different hashes can be distinct monomorphs.
    # LLVM's per-module prefix and Mach-O's leading underscore are not identity.
    return symbol.split(";")[-1].lstrip("_")


def coverage(text):
    result = {}
    name = None
    for line in text.splitlines():
        m = re.fullmatch(r"  (.+):", line)
        if m:
            name = m[1]
        m = re.match(r"    Block counts: \[([0-9, ]*)\]", line)
        if m and name:
            result[name] = max([int(n) for n in m[1].split(",") if n.strip()] or [0])
    if not result or not any(result.values()):
        raise ValueError("empty training function coverage")
    return result


def audit(text, trained):
    if re.search(r"hash mismatch|control flow change", text, re.I):
        raise ValueError("stale profile / CFG mismatch; see build log")
    observed = {canonical(s) for s, count in trained.items() if count > 0}
    known = {canonical(s) for s in trained}
    allowed, failures = [], []
    for line in text.splitlines():
        m = re.search(
            r"no profile data available for function (\S+) Hash = (\d+)", line
        )
        if not m:
            if "no profile data available" in line:
                raise ValueError("unrecognized missing-profile diagnostic")
            continue
        symbol = m[1]
        key = canonical(symbol)
        if key in observed:
            failures.append(symbol)
        else:
            allowed.append(
                {
                    "function": symbol,
                    "cfg_hash": m[2],
                    "diagnostic": line,
                    "training_count": 0,
                    "reason": (
                        "instrumented function has zero training counters"
                        if key in known
                        else "use-only function specialization: absent from generate counter inventory; no trained function identity matches"
                    ),
                }
            )
    if failures:
        raise ValueError(f"trained function missing profile: {failures}")
    return {"allowed_cold_misses": allowed, "trained_functions": len(observed)}
