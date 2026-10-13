"""Release-only cost bounds collected across the same balanced process pairs."""
from __future__ import annotations

import math
import os
from statistics import median

COST_TESTS = [
    'test/test_operand_read_cost.py', 'test/test_cell_write_cost.py',
    'test/test_cumulation_cost.py', 'test/test_core_io_cost.py',
    'test/test_fillna_cost.py', 'test/test_append.py', 'test/test_fulfill.py',
]


def timing_rounds(rounds: int) -> int:
    """Spread repeated timing work across pairs; ordinary tests keep their budget."""
    return max(1, math.ceil(rounds / int(os.environ.get('VOLAS_BENCHMARK_PAIRS', '1'))))


def cost_verdict(samples: list[dict]) -> bool:
    """Keep every original strict bound; one noisy process cannot veto a run."""
    if not samples:
        raise ValueError('cost checks require paired samples')
    names = samples[0].keys()
    if not names or any(sample.keys() != names for sample in samples):
        raise ValueError('every cost round must contain the same nonempty cost checks')
    failed = False
    noise = []
    for name in names:
        ratios = [sample[name] for sample in samples]
        if any(not math.isfinite(r) or r <= 0 for r in ratios):
            raise ValueError(f'{name}: cost ratios must be positive finite')
        ratio = median(ratios)
        noise.append(median(abs(r - ratio) for r in ratios) / ratio)
        if ratio >= 1:
            print(f'FAIL: release cost bound {name}: normalized ratio {ratio:.3f} >= 1')
            failed = True
    print(f'release cost bounds: {len(names)} checks, {len(samples)} paired rounds; '
          f'{"FAIL" if failed else "PASS"}; max normalized ratio '
          f'{max(median(sample[name] for sample in samples) for name in names):.3f}; '
          f'relative MAD median {median(noise):.1%}, max {max(noise):.1%}')
    return not failed
