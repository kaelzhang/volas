#!/usr/bin/env python3
"""Performance-regression gate for CI — compares this commit against its parent.

``make perf-ab`` and CI install both builds side by side on the same runner and
measure AB/BA pairs with the same current harness. ``perf_pairs.py`` supplies
comparison-equivalent times from the median of paired HEAD/base ratios, with
between-pair noise stated in the output. Historical single-run benchmark JSON
remains readable; the same thresholds below apply to both input formats.

Because HEAD and base run on one machine, absolute runner speed AND CPU microarchitecture
both cancel out — so the thresholds are architecture-independent, and there is no stored
baseline to drift across machines or to regenerate by hand.

A regression (volas got slower from base to HEAD) is flagged when, over the items whose
base median is at least ``--min-us`` microseconds, plus summed section buckets
for faster items:

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
from statistics import median
import sys

try:
    from scripts.perf_items import volas_key
except ModuleNotFoundError:  # direct script entrypoint
    from perf_items import volas_key


def _volas_medians(data: dict) -> dict:
    """All volas measurements, keyed by section and full parametrized name."""
    return {key: row['stats']['median'] for row in data['benchmarks']
            if (key := volas_key(row)) is not None}


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
                    help='gate individual items above this many microseconds; faster items '
                         'are summed by section (default 5)')
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
    buckets: dict[str, list[float]] = {}
    bucket_keys: dict[str, list[tuple[str, str]]] = {}
    with open(a.head) as stream:
        paired = json.load(stream).get('paired', {})
    observations = {tuple(item['name']): item for item in paired.get('items', [])
                    if 'base_seconds' in item and 'head_seconds' in item}
    for key, bmed in base.items():
        if key not in head:
            continue
        if bmed * 1e6 < a.min_us:
            sub_floor += 1
            totals = buckets.setdefault(key[0], [0.0, 0.0])
            bucket_keys.setdefault(key[0], []).append(key)
            totals[0] += bmed
            totals[1] += head[key]
            continue
        ratios[key] = head[key] / bmed

    for section, (bmed, hmed) in buckets.items():
        keys = bucket_keys[section]
        if observations and all(key in observations for key in keys):
            count = len(observations[keys[0]]['base_seconds'])
            bs = [sum(observations[key]['base_seconds'][r] for key in keys) for r in range(count)]
            hs = [sum(observations[key]['head_seconds'][r] for key in keys) for r in range(count)]
            bmed = median(bs)
            hmed = bmed * median(h / b for b, h in zip(bs, hs))
        key = section, '<sub-floor total>'
        base[key], head[key] = bmed, hmed
        ratios[key] = hmed / bmed

    if not ratios:
        print('perf gate: no comparable items above the time floor; skipping.')
        return 0

    geomean = math.exp(sum(math.log(r) for r in ratios.values()) / len(ratios))
    regressions = sorted((kv for kv in ratios.items() if kv[1] > a.item_max),
                         key=lambda kv: -kv[1])
    faster = sum(1 for r in ratios.values() if r < 1.0)

    print(f'perf gate: {len(ratios)} items gated (>= {a.min_us:g}us), {sub_floor} sub-floor '
          f'aggregated in {len(buckets)} section(s); geomean HEAD/base = {geomean:.3f} (max {a.geomean_max:.2f}); '
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
