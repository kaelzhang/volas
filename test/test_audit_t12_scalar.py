"""Systematic audit — T12 (scalar): Timestamp fields / arithmetic / comparison.

Datetime scalars are value-critical for a quant library (P6). The input
representation (I-rep, §4.6) matters: a numpy datetime64 operand takes a
different extraction path than a python/pandas scalar and must agree.

Cell IDs:  T12.field/<f> · T12.cmp/<irep> · T12.arith/<op>
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import volas


def test_fields():
    t = volas.Timestamp("2021-06-15 13:45:30")
    assert (t.year, t.month, t.day) == (2021, 6, 15)
    assert (t.hour, t.minute, t.second) == (13, 45, 30)
    assert t.weekday() == 1                       # 2021-06-15 is a Tuesday
    assert t.strftime("%Y/%m/%d") == "2021/06/15"
    assert t.value == 1623764730000000000         # ns since epoch


def test_comparison_and_arithmetic():
    t1, t2 = volas.Timestamp("2021-06-15"), volas.Timestamp("2021-06-16")
    assert (t1 < t2) is True
    assert (t1 == volas.Timestamp("2021-06-15")) is True
    assert (t2 - t1) == 86400000000000            # one day, in ns
    assert volas.Timestamp("2021-06-15") + 0 == t1


def test_comparison_irep_numpy():
    """A numpy datetime64 operand compares correctly (I-rep parity)."""
    t2 = volas.Timestamp("2021-06-16")
    assert (t2 == np.datetime64("2021-06-16")) is True
    assert (t2 < np.datetime64("2021-06-17")) is True


# F15 (FIXED): pandas / stdlib datetime scalars are recognised datetime
# operands (isoformat round-trip in the parse layer) — the label-KeyError
# leak is gone.
def test_comparison_irep_pandas():
    t2 = volas.Timestamp("2021-06-16")
    assert (t2 == pd.Timestamp("2021-06-16")) is True
