#!/usr/bin/env python3
"""Compact single-indicator performance report (the INDICATOR= pipeline).

Reads a pytest-benchmark JSON produced by `make benchmark INDICATOR=<x>` and
prints the normalized standing — volas vs ta-lib for the batch and the
after-append paths, plus volas's own append/batch overhead factor. Stdout only;
nothing is archived (single-indicator probes are working measurements, not
record runs).

Usage:  python scripts/bench_indicator_report.py <benchmark.json> <indicator>
        python scripts/bench_indicator_report.py <new.json> <indicator> --base <old.json>
"""

from __future__ import annotations

import json
import pathlib
import sys


def _means(path: str) -> dict[str, float]:
    data = json.loads(pathlib.Path(path).read_text())
    return {b["name"]: b["stats"]["mean"] for b in data["benchmarks"]}


def _row(means: dict[str, float], kind: str, ind: str, who: str) -> float | None:
    # exact id or the @n=<len> spelling used by extended coverage rows
    for name, mean in means.items():
        if name.startswith(f"{kind}[{ind}-{who}]") or name.startswith(f"{kind}[{ind}@n="):
            if name.endswith(f"-{who}]"):
                return mean
    return None


def report(means: dict[str, float], ind: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for label, kind in (("batch", "test_coverage"), ("after_append", "test_coverage_after_append")):
        v = _row(means, kind, ind, "volas")
        t = _row(means, kind, ind, "talib")
        out[f"{label}_volas_us"] = v * 1e6 if v else None
        out[f"{label}_talib_us"] = t * 1e6 if t else None
        out[f"{label}_ratio"] = (v / t) if v and t else None
    bv, av = out["batch_volas_us"], out["after_append_volas_us"]
    out["append_overhead"] = (av / bv) if av and bv else None
    return out


def _fmt(r: dict, ind: str, tag: str = "") -> str:
    def f(x, suffix=""):
        return f"{x:8.2f}{suffix}" if x is not None else "       —"
    lines = [f"indicator: {ind}{tag}"]
    for label in ("batch", "after_append"):
        lines.append(
            f"  {label:13} volas {f(r[f'{label}_volas_us'], 'us')}   "
            f"talib {f(r[f'{label}_talib_us'], 'us')}   "
            f"volas/talib {f(r[f'{label}_ratio'], 'x')}"
        )
    lines.append(f"  append/batch overhead (volas): {f(r['append_overhead'], 'x')}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    path, ind = argv[1], argv[2]
    new = report(_means(path), ind)
    print(_fmt(new, ind))
    if "--base" in argv:
        base = report(_means(argv[argv.index("--base") + 1]), ind)
        print()
        print(_fmt(base, ind, "  [BASE]"))
        print()
        for label in ("batch", "after_append"):
            n, b = new[f"{label}_volas_us"], base[f"{label}_volas_us"]
            if n and b:
                print(f"  {label:13} HEAD/base = x{n / b:.3f}  "
                      f"({'regression' if n / b > 1.05 else 'improvement' if n / b < 0.95 else 'noise band'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
