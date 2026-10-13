#!/usr/bin/env python3
"""Run isolated pytest processes in AB/BA pairs and summarize paired ratios.

The report JSON keeps the existing benchmark shape. BASE times are medians
across rounds; HEAD times are BASE times times the median paired HEAD/BASE
ratio (for each statistic). They are comparison-equivalent times, not pooled
HEAD observations. Raw round files and the ratio/MAD metadata remain available
until perf_ab.sh removes its temporary directory.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

try:
    from scripts.perf_costs import COST_TESTS, cost_verdict
    from scripts.perf_items import benchmark_key
except ModuleNotFoundError:
    from perf_costs import COST_TESTS, cost_verdict
    from perf_items import benchmark_key


WORKER = '''
import pathlib, sys
site = pathlib.Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(site))
import volas, volas_rs
for module in (volas, getattr(volas_rs, 'volas_rs', volas_rs)):
    actual = pathlib.Path(module.__file__).resolve()
    if site not in actual.parents:
        raise RuntimeError(f'isolated import failed: {actual} not in {site}')
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
'''


class BenchmarkFailure(RuntimeError):
    """Keep BASE plumbing failures distinct from HEAD failures for CI."""

    def __init__(self, side: str, status: int):
        self.side = side
        super().__init__(f'{side} benchmark failed with exit {status}')


def validate_rounds(rounds: int) -> None:
    if rounds < 6 or rounds % 2:
        raise ValueError('paired rounds must be even and at least 6')


def summarize(base: list[dict], head: list[dict]) -> tuple[dict, dict]:
    """Median of paired ratios; relative MAD describes between-pair noise."""
    validate_rounds(len(base))
    if len(head) != len(base):
        raise ValueError('BASE and HEAD must have equal paired rounds')
    runs = [{benchmark_key(b): b for b in run['benchmarks']} for run in base + head]
    names = runs[0].keys()
    if not names or any(run.keys() != names for run in runs):
        raise ValueError('every paired round must contain the same benchmark items')
    outputs = copy.deepcopy(base[0]), copy.deepcopy(head[0])
    noise = []
    for index, name in enumerate(names):
        bstats, hstats = {}, {}
        stats = ['median', 'mean']
        if all('min' in run[name]['stats'] for run in runs):
            stats.append('min')  # win-count reports keep their existing minimum metric
        for stat in stats:
            bs = [run[name]['stats'][stat] for run in runs[:len(base)]]
            hs = [run[name]['stats'][stat] for run in runs[len(base):]]
            if any(not math.isfinite(x) or x <= 0 for x in bs + hs):
                raise ValueError(f'{name}: {stat} must be positive finite seconds')
            ratios = [h / b for b, h in zip(bs, hs)]
            ratio = median(ratios)
            bstats[stat] = median(bs)
            hstats[stat] = bstats[stat] * ratio
            if stat == 'median':
                noise.append({'name': name, 'median_ratio': ratio,
                              'relative_mad': median(abs(r - ratio) for r in ratios) / ratio,
                              'ratios': ratios, 'base_seconds': bs, 'head_seconds': hs})
        for output, stats in zip(outputs, (bstats, hstats)):
            # Retain only report-consumed statistics; pooled stddev/ops would
            # describe a different estimator and mislead downstream tools.
            output['benchmarks'][index] = copy.deepcopy(runs[0][name])
            output['benchmarks'][index]['stats'] = stats
    outputs[1]['paired'] = {'rounds': len(base), 'items': noise,
                            'estimator': 'median of paired HEAD/BASE ratios'}
    return outputs


def run_pairs(python: str, root: Path, sites: dict[str, str], directory: Path,
              rounds: int, options: list[str]) -> None:
    validate_rounds(rounds)
    samples: dict[str, list[dict]] = {'base': [], 'head': []}
    costs: dict[str, list[dict]] = {'base': [], 'head': []}
    scoped = '-k' in options or any(option.startswith('--volas-benchmark-indicator')
                                    for option in options)
    full_gate = not scoped or '--volas-cost-checks' in options
    options = [option for option in options if option != '--volas-cost-checks']
    started = time.monotonic()
    environment = dict(os.environ, VOLAS_BENCHMARK_PAIRS=str(rounds))
    for pair in range(rounds):
        order = ('base', 'head') if pair % 2 == 0 else ('head', 'base')
        for side in order:
            environment['VOLAS_COST_REVERSE'] = str(pair % 2)
            path = directory / f'{side}-{pair + 1}.json'
            print(f'>> pair {pair + 1}/{rounds}: {side.upper()}', flush=True)
            with (directory / f'{side}-{pair + 1}.log').open('w') as log:
                result = subprocess.run([python, '-c', WORKER, sites[side],
                                         'test/test_benchmark.py', 'test/test_benchmark_true_range.py', *options,
                                         f'--benchmark-json={path}'], cwd=root, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                sys.stderr.write(log.name + ':\n' + Path(log.name).read_text())
                raise BenchmarkFailure(side, result.returncode)
            samples[side].append(json.loads(path.read_text()))
            if full_gate:
                cost_path = directory / f'{side}-cost-{pair + 1}.json'
                with (directory / f'{side}-cost-{pair + 1}.log').open('w') as log:
                    result = subprocess.run([python, '-c', WORKER, sites[side], *COST_TESTS,
                                             '--benchmark-disable', '-q',
                                             f'--volas-cost-json={cost_path}'], cwd=root, env=environment,
                                            stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    sys.stderr.write(Path(log.name).read_text())
                    raise BenchmarkFailure(side, result.returncode)
                costs[side].append(json.loads(cost_path.read_text()))
    if full_gate and not cost_verdict(costs['head']):
        raise BenchmarkFailure('head', 1)
    base, head = summarize(samples['base'], samples['head'])
    for side, data in (('base', base), ('head', head)):
        (directory / f'{side}.json').write_text(json.dumps(data))
    items = head['paired']['items']
    mads = [x['relative_mad'] for x in items]
    print(f'paired A/B: {rounds} AB/BA pairs; median paired HEAD/base; '
          f'relative MAD median {median(mads):.1%}, max {max(mads):.1%}; '
          f'measurement wall time {time.monotonic() - started:.1f}s', flush=True)
    for item in sorted(items, key=lambda x: -x['relative_mad'])[:5]:
        print(f"    noise: {item['name']}: relative MAD {item['relative_mad']:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-site', required=True)
    parser.add_argument('--head-site', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=6)
    args, options = parser.parse_known_args()
    try:
        run_pairs(sys.executable, args.root, {'base': args.base_site, 'head': args.head_site},
                  args.output, args.rounds, options)
    except BenchmarkFailure as error:
        print(error, file=sys.stderr)
        raise SystemExit(2 if error.side == 'base' else 1) from error


if __name__ == '__main__':
    main()
