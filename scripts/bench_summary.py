#!/usr/bin/env python3
"""Summarize the append section of a pytest-benchmark JSON, or compare two runs.

The "append one bar -> updated indicator" times are microsecond-scale for volas
(an O(1) state-carry resume), so their speedup *ratio* over pandas (which does an
O(n) recompute per bar) is inherently noise-sensitive. This tool surfaces, per
indicator, volas's median, its **relative stddev (σ%)** — the noise floor that
decides whether a ratio change is real — and the ratio over each comparison lib.

Usage::

    python scripts/bench_summary.py RUN.json                 # one run
    python scripts/bench_summary.py NEW.json --base OLD.json  # compare (Δ% volas time)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict


def _append_stats(path: str) -> dict[str, dict[str, dict]]:
    """`{indicator: {candidate: stats}}` over the test_append benchmarks."""
    with open(path) as f:
        data = json.load(f)
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for b in data.get("benchmarks", []):
        if "test_append" not in (b.get("name", "") + b.get("fullname", "")):
            continue
        p = b.get("params", {})
        ind, cand = p.get("indicator"), p.get("candidate")
        if ind and cand:
            out[ind][cand] = b["stats"]
    return out


def _ratio(cands: dict, other: str) -> float:
    v = cands.get("volas", {}).get("median")
    o = cands.get(other, {}).get("median")
    return (o / v) if (v and o) else float("nan")


def report(path: str) -> None:
    ap = _append_stats(path)
    print(f"=== append: {path} ===")
    print(f"{'indicator':<14}{'volas µs':>10}{'σ%':>7}{'rounds':>8}{'×pandas':>9}{'×talib':>9}{'×polars':>9}")
    for ind in sorted(ap):
        c = ap[ind]
        v = c.get("volas")
        if not v:
            continue
        sig = (v["stddev"] / v["median"] * 100) if v["median"] else 0.0
        print(f"{ind:<14}{v['median'] * 1e6:>10.3f}{sig:>7.1f}{v['rounds']:>8}"
              f"{_ratio(c, 'pandas'):>9.1f}{_ratio(c, 'talib'):>9.1f}{_ratio(c, 'polars'):>9.1f}")


def compare(new: str, base: str) -> None:
    A, B = _append_stats(new), _append_stats(base)
    print(f"=== append compare:  NEW={new}  vs  BASE={base} ===")
    print(f"{'indicator':<14}{'σ%(new)':>8}{'×pd base':>10}{'×pd new':>9}{'volasµs base':>14}{'volasµs new':>13}{'Δvolas%':>9}")
    for ind in sorted(set(A) & set(B)):
        a, b = A[ind].get("volas"), B[ind].get("volas")
        if not a or not b:
            continue
        va, vb = a["median"] * 1e6, b["median"] * 1e6
        sig = (a["stddev"] / a["median"] * 100) if a["median"] else 0.0
        d = ((va - vb) / vb * 100) if vb else 0.0  # +Δ = new is slower
        print(f"{ind:<14}{sig:>8.1f}{_ratio(B[ind], 'pandas'):>10.1f}{_ratio(A[ind], 'pandas'):>9.1f}"
              f"{vb:>14.3f}{va:>13.3f}{d:>+9.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--base", help="compare NEW (json) against this BASE run")
    a = ap.parse_args()
    compare(a.json, a.base) if a.base else report(a.json)


if __name__ == "__main__":
    main()
