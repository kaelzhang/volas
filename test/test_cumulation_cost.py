"""Cumulation costs follow input/output size, not materialized period frames.

Repeatedly correcting the latest fine bar must cost O(columns), independent of
the number of fine bars already held by the open period.
"""

import time

import numpy as np
import pytest

from scripts.perf_costs import timing_rounds

import volas


def fine_bars(n, unit='s'):
    stamps = np.datetime64('2024-01-01', 'ns') + np.arange(n) * np.timedelta64(1, unit)
    values = np.arange(n, dtype=float) + 1.0
    return volas.DataFrame({c: values for c in ('open', 'high', 'low', 'close', 'volume')}, index=stamps)


def best_us(call, repetitions):
    for _ in range(timing_rounds(5)):
        call()
    best = float('inf')
    for _ in range(timing_rounds(5)):
        start = time.perf_counter()
        for _ in range(repetitions):
            call()
        best = min(best, (time.perf_counter() - start) * 1e6 / repetitions)
    return best


def test_batch_cumulation_does_not_materialize_each_period(cost_guard):
    fine = fine_bars(100_000, 'm')
    many, few = cost_guard.pair(lambda: best_us(lambda: fine.cumulate('1m'), 1),
                                lambda: best_us(lambda: fine.cumulate('1d'), 3))
    cost_guard(many, few * 8)


@pytest.mark.parametrize('cached', [False, True])
def test_last_fine_bar_resend_does_not_scale_with_open_period(cached, cost_guard):
    def measure(n):
        fine = fine_bars(n)
        live = fine.cumulate('1d')
        bar = fine.iloc[-1:]
        if cached:
            live['ema:2']

        def resend():
            live.append(bar)
            if cached:
                live['ema:2']

        return best_us(resend, 100)

    small, large = cost_guard.pair(lambda: measure(10), lambda: measure(10_000))
    cost_guard(large, small * 5)
