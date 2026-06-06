#!/usr/bin/env python3
"""Performance-regression gate for CI.

Compares per-item **volas-vs-reference ratios** from a pytest-benchmark JSON against a
committed baseline and exits non-zero on regression. The ratio (volas median / reference
median) cancels the runner's absolute speed — both candidates run in the *same* CI job.
Reference = TA-Lib for the ``coverage`` section, the fastest of pandas / polars for the
``api`` section. Lower ratio = volas faster.

A regression is flagged when an item's ratio grows past ``baseline * (1 + threshold)``
(volas got relatively slower). **Two robustness rules keep the gate from firing on noise:**

* **Noise exclusions.** Per-item gating (and the win-count) skip items whose ratio is timing
  noise rather than signal: (a) a *reference* median below ``--min-ref-us`` (the price
  transforms run in ~2 µs, where constant Python/dispatch overhead dominates the ratio), and
  (b) the candlestick family (``cdl.*``) — branch-heavy and data-dependent, so its per-call
  cost, and thus the volas/TA-Lib ratio, swings run-to-run far past any threshold even on the
  same machine (the items that tripped the old gate were always candlesticks). What remains is
  the ~60 straight-line arithmetic indicators, whose ratios are stable.
* **Win-count tolerance (``--win-tolerance``).** The number of ``coverage`` items where volas
  wins (ratio ≤ 1) may drop by up to this many before failing — a broad regression no single
  item trips, with slack for the noisy fast items and for runner-to-runner / cross-architecture
  drift (the baseline is regenerated *in CI* — see ``perf.yml`` ``workflow_dispatch`` — so it
  matches the gate's environment; otherwise a dev-machine baseline adds a systematic offset).

Usage::

    python scripts/perf_gate.py BENCH_JSON [--baseline F] [--threshold 0.25] [--min-ref-us 8]
    python scripts/perf_gate.py BENCH_JSON --update      # (re)write the baseline (do this in CI)
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# Reference candidate per section (best-of for the API plumbing comparison).
_REFERENCE = {'coverage': ('talib',), 'api': ('pandas', 'polars')}


def _ratios(data: dict) -> dict:
    """``{section: {item: (volas/reference ratio, reference median seconds)}}``.

    The reference median is kept so the gate can apply an absolute-time noise floor; the
    committed baseline stores only the ratio (see ``--update``)."""
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

    out: dict[str, dict[str, tuple[float, float]]] = {'coverage': {}, 'api': {}}
    for (section, item), m in cells.items():
        v = m.get('volas')
        if not v:
            continue
        refs = [m[c] for c in _REFERENCE[section] if c in m and m[c] > 0]
        if refs:
            ref = min(refs)
            out[section][item] = (round(v / ref, 4), ref)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('bench', help='pytest-benchmark JSON to check')
    ap.add_argument('--baseline', default='scripts/perf_baseline.json')
    ap.add_argument('--threshold', type=float, default=0.25,
                    help='max tolerated relative ratio increase (default 0.25 = 25%%)')
    ap.add_argument('--min-ref-us', type=float, default=8.0,
                    help='only gate items whose reference median is at least this many '
                         'microseconds; faster items have noise-dominated ratios (default 8)')
    ap.add_argument('--win-tolerance', type=int, default=8,
                    help='max tolerated drop in the coverage win-count (default 8)')
    ap.add_argument('--update', action='store_true', help='(re)write the baseline and exit')
    a = ap.parse_args(argv)

    ratios = _ratios(json.load(open(a.bench)))

    if a.update:
        flat = {s: {i: rr[0] for i, rr in d.items()} for s, d in ratios.items()}
        with open(a.baseline, 'w') as f:
            json.dump(flat, f, indent=1, sort_keys=True)
            f.write('\n')
        n = sum(len(v) for v in flat.values())
        print(f'baseline written to {a.baseline} ({n} items)')
        return 0

    try:
        base = json.load(open(a.baseline))
    except FileNotFoundError:
        print(f'PERF GATE: no baseline at {a.baseline}; run with --update to create it.')
        return 0

    regressions, new_items, skipped = [], [], []
    for section, items in ratios.items():
        for item, (r, ref) in items.items():
            br = base.get(section, {}).get(item)
            if br is None:
                new_items.append((section, item, r))
            elif r > br * (1 + a.threshold):
                # Skip where the ratio is timing noise, not signal: a sub-floor reference
                # (Python/dispatch overhead dominates) or the branch-heavy candlestick family.
                if ref * 1e6 < a.min_ref_us or item.startswith('cdl.'):
                    skipped.append((section, item))
                else:
                    regressions.append((section, item, br, r, ref * 1e6))

    # Win-count over the stable (non-candlestick) coverage items only.
    base_wins = sum(1 for i, r in base.get('coverage', {}).items()
                    if r <= 1.0 and not i.startswith('cdl.'))
    now_wins = sum(1 for i, (r, _) in ratios.get('coverage', {}).items()
                   if r <= 1.0 and not i.startswith('cdl.'))
    win_drop = base_wins - now_wins

    total = sum(len(v) for v in ratios.values())
    print(f'perf gate: {total} items, threshold {a.threshold:.0%}, ref-floor {a.min_ref_us:g}us; '
          f'coverage wins {now_wins} (baseline {base_wins}).')
    if skipped:
        print(f'  {len(skipped)} item(s) over threshold but not gated (timing noise — fast '
              f'reference or candlestick): ' + ', '.join(f'{s}:{i}' for s, i in skipped[:8])
              + (' …' if len(skipped) > 8 else ''))
    if new_items:
        print(f'  {len(new_items)} new item(s) without a baseline (not gated): '
              + ', '.join(f'{s}:{i}' for s, i, _ in new_items[:8])
              + (' …' if len(new_items) > 8 else ''))

    failed = False
    if regressions:
        failed = True
        print(f'FAIL: {len(regressions)} item(s) regressed > {a.threshold:.0%} '
              '(volas got relatively slower):')
        for s, i, br, r, ref_us in sorted(regressions, key=lambda x: -(x[3] / x[2])):
            print(f'    [{s}] {i}: {br:.3f} -> {r:.3f}  (+{r / br - 1:.0%}, ref {ref_us:.0f}us)')
    if win_drop > a.win_tolerance:
        failed = True
        print(f'FAIL: coverage win-count dropped by {win_drop} (> tolerance {a.win_tolerance}).')

    if failed:
        print('\nIf this regression is intentional, regenerate the baseline (in CI, so it matches '
              'the gate environment):\n'
              '  perf.yml -> Run workflow -> update_baseline=true   # commits scripts/perf_baseline.json')
        return 1
    print('PASS: no performance regression beyond threshold.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
