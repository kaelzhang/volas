"""Builders for the live time-frame benchmark rows (``test_benchmark.test_tf_live``).

A ``15m`` frame is constructed from the benchmark history (its last row is the
forming bar), an indicator is cached on it, and one warm-up bar is folded in so the
measured call pays the steady-state cost (no one-time capacity growth or
copy-on-write). The measured call appends ONE ``1m`` bar and refreshes the cached
column — split by phase:

* ``forming``  — the bar joins the open period: the forming-row refresh;
* ``rollover`` — the bar opens the next period: the rollover plus the new row's refresh.
"""

import numpy as np

from volas import DataFrame

TF_LIVE_INDICATORS = ['atr:14', 'ema:20', 'rsi:14', 'macd']
TF_LIVE_PHASES = ['forming', 'rollover']

_T0 = np.datetime64('2020-01-01T00:00', 'ns')
_MINUTE = np.timedelta64(1, 'm')
_PERIOD = 15 * _MINUTE


def build(arr, columns, indicator, phase):
    """``(run, setup)`` for ``benchmark.pedantic``: ``setup`` returns a fresh warm
    frame, ``run`` appends the measured bar and fulfills."""
    n = len(arr[columns[0]]) - 2
    index = _T0 + np.arange(n) * _PERIOD
    seed = {c: arr[c][:n] for c in columns}
    step = _MINUTE if phase == 'forming' else _PERIOD

    def bar(i, ts):
        return DataFrame({c: arr[c][i:i + 1] for c in columns}, index=np.array([ts]))

    warm = bar(n, index[-1] + step)
    measured = bar(n + 1, index[-1] + 2 * step)

    def setup():
        df = DataFrame(seed, index=index, time_frame='15m')
        df[indicator]
        df.append(warm)
        df.fulfill()
        return (df,), {}

    def run(df):
        df.append(measured)
        df.fulfill()

    return run, setup
