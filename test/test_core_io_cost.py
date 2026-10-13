"""Multi-row integer admission is linear in incoming rows, including unordered labels."""
import time

import numpy as np
import pytest

from scripts.perf_costs import timing_rounds

import volas


def _append_seconds(n, order):
    labels = np.arange(1, n + 1, dtype=np.int64)
    if order == 'decreasing':
        labels = labels[::-1].copy()
    elif order == 'unordered':
        labels = np.random.default_rng(24).permutation(labels)
    rows = volas.DataFrame({'x': np.ones(n)}, index=labels)
    best = float('inf')
    for _ in range(timing_rounds(3)):
        df = volas.DataFrame({'x': [0.]}, index=[0])
        start = time.perf_counter()
        df.append(rows)
        best = min(best, time.perf_counter() - start)
        assert len(df) == n + 1
    return best


@pytest.mark.parametrize('order', ['increasing', 'decreasing', 'unordered'])
def test_multirow_int_append_is_not_quadratic(order, cost_guard):
    small, large = cost_guard.pair(lambda: _append_seconds(3_000, order),
                                  lambda: _append_seconds(30_000, order))
    cost_guard(large, small * 30)
