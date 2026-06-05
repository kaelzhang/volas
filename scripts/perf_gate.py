#!/usr/bin/env python3
"""Performance-regression gate for CI.

Compares per-item **volas-vs-reference ratios** from a pytest-benchmark JSON against
a committed baseline and exits non-zero on regression. The ratio (volas median /
reference median) is machine-independent — both candidates run in the *same* CI job,
so absolute speed varies with the runner but their ratio does not. Reference =
TA-Lib for the ``coverage`` section, the fastest of pandas / polars for the ``api``
section. Lower ratio = volas faster.

A regression is flagged when an item's ratio grows past ``baseline * (1 + threshold)``
(volas got relatively slower). New items with no baseline are reported, not failed.
The overall "volas wins" count dropping past a small tolerance also fails (catches a
broad regression that no single item trips).

Usage::

    python scripts/perf_gate.py BENCH_JSON [--baseline F] [--threshold 0.25]
    python scripts/perf_gate.py BENCH_JSON --update      # (re)write the baseline
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# Reference candidate per section (best-of for the API plumbing comparison).
_REFERENCE = {'coverage': ('talib',), 'api': ('pandas', 'polars')}


def _ratios(data: dict) -> dict:
    """``{section: {item: volas/reference_ratio}}`` from a pytest-benchmark JSON."""
    cells: dict[tuple[str, str], dict[str, float]] = {}
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
        if cand is None or item is None:  # fall back to parsing name[cand-item]
            m = re.search(r'\[(.+)\]', name)
            if m:
                inside = m.group(1)
                for c in ('talib', 'volas', 'pandas', 'polars'):
                    if inside.startswith(c + '-'):
                        cand, item = c, inside[len(c) + 1:]
                        break
        if cand and item:
            cells.setdefault((section, item), {})[cand] = b['stats']['median']

    out: dict[str, dict[str, float]] = {'coverage': {}, 'api': {}}
    for (section, item), m in cells.items():
        v = m.get('volas')
        if not v:
            continue
        refs = [m[c] for c in _REFERENCE[section] if c in m and m[c] > 0]
        if refs:
            out[section][item] = round(v / min(refs), 4)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bench', help='pytest-benchmark JSON to check')
    ap.add_argument('--baseline', default='scripts/perf_baseline.json')
    ap.add_argument('--threshold', type=float, default=0.25,
                    help='max tolerated relative ratio increase (default 0.25 = 25%%)')
    ap.add_argument('--win-tolerance', type=int, default=2,
                    help='max tolerated drop in the coverage win-count (default 2)')
    ap.add_argument('--update', action='store_true', help='(re)write the baseline and exit')
    a = ap.parse_args(argv)

    ratios = _ratios(json.load(open(a.bench)))

    if a.update:
        with open(a.baseline, 'w') as f:
            json.dump(ratios, f, indent=1, sort_keys=True)
            f.write('\n')
        n = sum(len(v) for v in ratios.values())
        print(f'baseline written to {a.baseline} ({n} items)')
        return 0

    try:
        base = json.load(open(a.baseline))
    except FileNotFoundError:
        print(f'PERF GATE: no baseline at {a.baseline}; run with --update to create it.')
        return 0

    regressions, new_items = [], []
    for section, items in ratios.items():
        for item, r in items.items():
            br = base.get(section, {}).get(item)
            if br is None:
                new_items.append((section, item, r))
            elif r > br * (1 + a.threshold):
                regressions.append((section, item, br, r))

    base_wins = sum(1 for r in base.get('coverage', {}).values() if r <= 1.0)
    now_wins = sum(1 for r in ratios.get('coverage', {}).values() if r <= 1.0)
    win_drop = base_wins - now_wins

    total = sum(len(v) for v in ratios.values())
    print(f'perf gate: {total} items, threshold {a.threshold:.0%}; '
          f'coverage wins {now_wins} (baseline {base_wins}).')
    if new_items:
        print(f'  {len(new_items)} new item(s) without a baseline (not gated): '
              + ', '.join(f'{s}:{i}' for s, i, _ in new_items[:8])
              + (' …' if len(new_items) > 8 else ''))

    failed = False
    if regressions:
        failed = True
        print(f'FAIL: {len(regressions)} item(s) regressed > {a.threshold:.0%} '
              '(volas got relatively slower):')
        for s, i, br, r in sorted(regressions, key=lambda x: -(x[3] / x[2])):
            print(f'    [{s}] {i}: {br:.3f} -> {r:.3f}  (+{r / br - 1:.0%})')
    if win_drop > a.win_tolerance:
        failed = True
        print(f'FAIL: coverage win-count dropped by {win_drop} (> tolerance {a.win_tolerance}).')

    if failed:
        print('\nIf this regression is intentional, regenerate the baseline:\n'
              '  python scripts/perf_gate.py <bench.json> --update  # then commit scripts/perf_baseline.json')
        return 1
    print('PASS: no performance regression beyond threshold.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
