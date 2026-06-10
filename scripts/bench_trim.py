#!/usr/bin/env python3
"""Trim a pytest-benchmark JSON to a small summary (in place, or to argv[2]).

pytest-benchmark stores every individual round time under each benchmark's
``stats.data`` (and ``data``). Across the coverage section that is ~1 GB per run —
far too large to archive per Git commit. The medians / stddev in ``stats`` are all
that scripts/perf_gate.py and scripts/bench_summary.py (and the HTML report) need,
so this drops the raw per-round arrays, shrinking the archive to a few hundred KB
while keeping every comparison-relevant number.
"""
import json
import sys


def main() -> None:
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else src
    with open(src) as f:
        data = json.load(f)
    for b in data.get("benchmarks", []):
        b.pop("data", None)
        if isinstance(b.get("stats"), dict):
            b["stats"].pop("data", None)
    with open(out, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
