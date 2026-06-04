"""rolling_calc: apply an arbitrary 1-D reducer over a rolling window.

This is the one place volas accepts an arbitrary Python callable on a windowed
computation. It is deliberately a standalone function (not a DataFrame method)
operating on array-like input, and is implemented in pure Python/NumPy — it does
NOT enter the Rust kernel, so the typed, callback-free indicator/cumulation
kernels stay clean. For the common typed reductions prefer the directives
(``hhv``/``llv``/``ma``/``sma``/…); reach for ``rolling_calc`` only when you need
a custom reducer.
"""

from typing import Any, Callable

import numpy as np

__all__ = ['rolling_calc']


def rolling_calc(
    values: Any,
    window: int,
    apply: Callable[[np.ndarray], Any],
    forward: bool = False,
    fill: Any = np.nan,
) -> np.ndarray:
    """Apply ``apply`` to each rolling window of ``values``.

    Args:
        values: array-like (a NumPy array or a volas ``Series``).
        window: window size.
        apply: a 1-D reducer, e.g. ``max`` / ``min`` / ``np.std``.
        forward: if False (default) each window *ends* at position ``i`` (the
            last ``window`` values, like ``hhv``/``llv``); if True each window
            *starts* at ``i`` (looks forward).
        fill: value for positions without a full window (default ``NaN``).

    Returns:
        ``np.ndarray`` of the per-window results.
    """
    arr = np.asarray(values)
    n = arr.shape[0]
    out = np.full(n, fill, dtype=float)
    if window <= 0 or window > n:
        return out
    if forward:
        for i in range(n - window + 1):
            out[i] = apply(arr[i:i + window])
    else:
        for i in range(window - 1, n):
            out[i] = apply(arr[i - window + 1:i + 1])
    return out
