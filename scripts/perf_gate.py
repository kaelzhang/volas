#!/usr/bin/env python3
"""Performance-regression gate for CI — compares this commit against its parent.

The benchmark runs twice in the SAME CI job, on the SAME runner: once for the current
commit (HEAD) and once for the base commit (the PR merge-base, or the previous tip on a
push). This gate compares volas's own median times between the two runs.

Because HEAD and base run on one machine, absolute runner speed AND CPU microarchitecture
both cancel out — so the thresholds are architecture-independent, and there is no stored
baseline to drift across machines or to regenerate by hand.

A regression (volas got slower from base to HEAD) is flagged when, over the items whose
base median is at least ``--min-us`` microseconds (faster items are pure timing noise):

* the geometric mean of the per-item HEAD/base ratios exceeds ``--geomean-max`` — a broad
  slowdown across the suite, or
* any single item slows by more than ``--item-max`` — a sharp regression in one indicator.

The gate is fail-safe: a missing or unreadable base file means "no comparison available",
which passes (a regression gate must not block CI on its own plumbing).

Usage::

    python scripts/perf_gate.py HEAD.json --base BASE.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys


def _volas_medians(data: dict) -> dict:
    """``{(section, item): volas median seconds}`` over the coverage + api sections."""
    out: dict[tuple[str, str], float] = {}
    for b in data['benchmarks']:
        name = b['name']
        if name.startswith('test_coverage'):
            section = 'coverage'
        elif name.startswith('test_api'):
            section = 'api'
        else:
            continue
        params = b.get('params') or {}
        cand = params.get('candidate')
        item = params.get('indicator') or params.get('op')
        if cand is None or item is None:  # fall back to parsing "[indicator-candidate]"
            m = re.search(r'\[(.+)\]', name)
            if m:
                inside = m.group(1)
                for c in ('talib', 'volas', 'pandas', 'polars'):
                    if inside.endswith('-' + c):
                        item, cand = inside[: -(len(c) + 1)], c
                        break
        if cand == 'volas' and item:
            out[(section, item)] = b['stats']['median']
    return out


def _load(path: str) -> dict | None:
    try:
        with open(path) as f:
            return _volas_medians(json.load(f))
    except (OSError, ValueError, KeyError):
        return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('head', help='pytest-benchmark JSON for the current commit')
    ap.add_argument('--base', required=True, help='pytest-benchmark JSON for the base commit')
    ap.add_argument('--min-us', type=float, default=5.0,
                    help='only gate items whose base median is at least this many '
                         'microseconds; faster items are timing noise (default 5)')
    ap.add_argument('--geomean-max', type=float, default=1.10,
                    help='max tolerated geometric-mean HEAD/base ratio (default 1.10 = +10%%)')
    ap.add_argument('--item-max', type=float, default=1.60,
                    help='max tolerated single-item HEAD/base ratio (default 1.60 = +60%%)')
    a = ap.parse_args(argv)

    head = _load(a.head)
    base = _load(a.base)
    if head is None:
        print(f'PERF GATE: cannot read HEAD benchmark {a.head}; skipping.')
        return 0
    if base is None:
        print(f'PERF GATE: no readable base benchmark at {a.base} '
              '(first commit, or the base build/benchmark did not run); skipping.')
        return 0

    ratios: dict[tuple[str, str], float] = {}
    sub_floor = 0
    for key, bmed in base.items():
        if key not in head:
            continue
        if bmed * 1e6 < a.min_us:
            sub_floor += 1
            continue
        ratios[key] = head[key] / bmed

    if not ratios:
        print('perf gate: no comparable items above the time floor; skipping.')
        return 0

    geomean = math.exp(sum(math.log(r) for r in ratios.values()) / len(ratios))
    regressions = sorted((kv for kv in ratios.items() if kv[1] > a.item_max),
                         key=lambda kv: -kv[1])
    faster = sum(1 for r in ratios.values() if r < 1.0)

    print(f'perf gate: {len(ratios)} items gated (>= {a.min_us:g}us), {sub_floor} sub-floor '
          f'skipped; geomean HEAD/base = {geomean:.3f} (max {a.geomean_max:.2f}); '
          f'{faster} item(s) faster.')
    for (s, i), r in sorted(ratios.items(), key=lambda kv: -kv[1])[:5]:
        print(f'    biggest mover: [{s}] {i}: x{r:.2f}')

    failed = False
    if geomean > a.geomean_max:
        failed = True
        print(f'FAIL: broad slowdown — geomean HEAD/base {geomean:.3f} > {a.geomean_max:.2f}.')
    if regressions:
        failed = True
        print(f'FAIL: {len(regressions)} item(s) slowed by > {(a.item_max - 1) * 100:.0f}%:')
        for (s, i), r in regressions:
            print(f'    [{s}] {i}: x{r:.2f}  '
                  f'(base {base[(s, i)] * 1e6:.0f}us -> head {head[(s, i)] * 1e6:.0f}us)')

    if failed:
        print('\nThis compares HEAD against its parent on the same runner. If the slowdown is '
              'intentional, say so in the change; otherwise profile the flagged indicator(s).')
        return 1
    print('PASS: no performance regression vs the parent commit.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
