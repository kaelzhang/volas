"""Dense owned frames reuse their missing-value facts across fill calls."""
import time

from scripts.perf_costs import timing_rounds

import volas


def _dense_fill_seconds(n):
    # Lists give the frame owned storage; ffill learns its no-missing fact.
    df = volas.DataFrame({'x': [1.] * n}).ffill()
    best = float('inf')
    for _ in range(timing_rounds(5)):
        start = time.perf_counter()
        for _ in range(20):
            out = df.fillna(0.)
        best = min(best, (time.perf_counter() - start) / 20)
        assert len(out) == n
    return best


def test_dense_frame_fillna_reuses_the_missing_fact(cost_guard):
    small, large = cost_guard.pair(lambda: _dense_fill_seconds(10_000),
                                  lambda: _dense_fill_seconds(1_000_000))
    cost_guard(large, small * 10)
