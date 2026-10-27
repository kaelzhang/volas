"""Explicit dtype export costs; construct sources outside the measured call."""
import numpy as np

from volas import DataFrame


# A full native scan at this length exceeds perf_gate's 5us item floor.
LENGTH = 1_048_576


def build(entry, source, target, missing, filled):
    values = np.ones(LENGTH, dtype=source)
    if missing:
        # Dense missing complements complete data: temporal object exports use
        # the missing singleton instead of allocating a Timestamp per slot.
        values[:] = np.nan if source == 'float64' else np.datetime64('NaT')
    frame = DataFrame({'x': values})
    obj = frame if entry == 'frame' else frame['x']
    kwargs = {'dtype': target}
    if filled:
        kwargs['na_value'] = 0
    return lambda: obj.to_numpy(**kwargs)
