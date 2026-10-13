"""Cumulation costs follow input/output size, not materialized period frames.

Repeatedly correcting the latest fine bar must cost O(columns), independent of
the number of fine bars already held by the open period.
"""

import time

import numpy as np
import pytest

import volas


def fine_bars(n, unit='s'):
    stamps = np.datetime64('2024-01-01', 'ns') + np.arange(n) * np.timedelta64(1, unit)
    values = np.arange(n, dtype=float) + 1.0
    return volas.DataFrame({c: values for c in ('open', 'high', 'low', 'close', 'volume')}, index=stamps)


def best_us(call, repetitions):
    for _ in range(5):
        call()
    best = float('inf')
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(repetitions):
            call()
        best = min(best, (time.perf_counter() - start) * 1e6 / repetitions)
    return best


def test_batch_cumulation_does_not_materialize_each_period():
    fine = fine_bars(100_000, 'm')
    many = best_us(lambda: fine.cumulate('1m'), 1)
    few = best_us(lambda: fine.cumulate('1d'), 3)
    assert many < few * 8, (many, few)


@pytest.mark.parametrize('cached', [False, True])
def test_last_fine_bar_resend_does_not_scale_with_open_period(cached):
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

    small, large = measure(10), measure(10_000)
    assert large < small * 5, (small, large)
