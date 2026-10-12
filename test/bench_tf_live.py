"""Builders for the live-path benchmark rows (``test_benchmark.test_tf_live``): one bar
arriving on a frame with a cached indicator, then the refresh — the per-tick cost of a
live feed.

A frame is built from the benchmark history (``rows`` bars), an indicator is cached on
it, and one warm-up bar of the same kind is applied, so the measured call pays the
steady-state cost (no one-time capacity growth or copy-on-write). The measured call
applies ONE bar and refreshes the cached column — by phase:

* ``forming``  — a ``1m`` bar joins a ``15m`` frame's open period (the forming row);
* ``rollover`` — a ``1m`` bar opens the next period (rollover + the new row);
* ``resend``   — the newest ``1m`` bar is re-sent with new values (it replaces the
  one already in the open period, and the period is re-cumulated);
* ``plain_append`` — a new bar on a plain frame;
* ``plain_replace`` — a bar re-sent under the plain frame's last label (it replaces
  the last row).

The ``_20k`` phases run on a frame ten times longer: a replacement's cost must not
depend on the history length.
"""

import numpy as np

from volas import DataFrame

TF_LIVE_INDICATORS = ['atr:14', 'ema:20', 'rsi:14', 'macd', 'ma:5', 'boll.upper', 'linearreg:14']
TF_LIVE_PHASES = [
    'forming', 'rollover', 'resend', 'resend_20k', 'plain_append', 'plain_replace',
    'plain_replace_20k',
]

_T0 = np.datetime64('2020-01-01T00:00', 'ns')
_MINUTE = np.timedelta64(1, 'm')
_PERIOD = 15 * _MINUTE


def _history(arr, columns, rows):
    """`rows` bars: the benchmark history, tiled (and drifted) to the length asked."""
    n = len(arr[columns[0]])
    reps = -(-rows // n)
    out = {}
    for c in columns:
        tiled = np.concatenate([arr[c] * (1.0 + 0.001 * k) for k in range(reps)])
        out[c] = tiled[:rows]
    return out


def build(arr, columns, indicator, phase):
    """``(run, setup)`` for ``benchmark.pedantic``: ``setup`` returns a fresh warm
    frame, ``run`` applies the measured bar and fulfills."""
    kind = phase.removesuffix('_20k')
    total = len(arr[columns[0]]) * (10 if phase.endswith('_20k') else 1)
    data = _history(arr, columns, total)
    n = total - 2
    plain = kind.startswith('plain')
    period = _MINUTE if plain else _PERIOD
    index = _T0 + np.arange(n) * period
    seed = {c: data[c][:n] for c in columns}

    def bar(i, ts, scale=1.0):
        return DataFrame({c: data[c][i:i + 1] * scale for c in columns}, index=np.array([ts]))

    last = index[-1]
    if kind == 'forming':
        warm, measured = bar(n, last + _MINUTE), bar(n + 1, last + 2 * _MINUTE)
    elif kind == 'rollover':
        warm, measured = bar(n, last + _PERIOD), bar(n + 1, last + 2 * _PERIOD)
    elif kind == 'resend':
        warm, measured = bar(n, last + _MINUTE), bar(n, last + _MINUTE, 1.0005)
    elif kind == 'plain_append':
        warm, measured = bar(n, last + _MINUTE), bar(n + 1, last + 2 * _MINUTE)
    else:  # plain_replace: the newest bar re-sent under its label
        warm, measured = bar(n, last + _MINUTE), bar(n, last + _MINUTE, 1.0005)

    def setup():
        df = DataFrame(seed, index=index) if plain else DataFrame(seed, index=index, time_frame='15m')
        df[indicator]
        df.append(warm)
        df.fulfill()
        return (df,), {}

    def run(df):
        df.append(measured)
        df.fulfill()

    return run, setup
