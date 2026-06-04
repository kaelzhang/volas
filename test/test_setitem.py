"""DataFrame column assignment + copy-on-write (audit PD-1)."""

import numpy as np
import pytest
from volas import DataFrame


def make():
    return DataFrame({'open': [1.0, 2.0, 3.0], 'close': [2.0, 3.0, 4.0]})


def test_setitem_array_scalar_series():
    df = make()
    df['x'] = [10.0, 20.0, 30.0]              # list
    assert df['x'].to_numpy().tolist() == [10.0, 20.0, 30.0]
    df['y'] = 7.0                              # scalar broadcast
    assert df['y'].to_numpy().tolist() == [7.0, 7.0, 7.0]
    df['sig'] = df['close'] > df['open']       # a (bool) Series
    assert df['sig'].dtype == 'bool'
    assert 'x' in df.columns and 'y' in df.columns and 'sig' in df.columns


def test_setitem_replaces_existing():
    df = make()
    df['open'] = [9.0, 9.0, 9.0]
    assert df['open'].to_numpy().tolist() == [9.0, 9.0, 9.0]


def test_setitem_length_mismatch_raises():
    df = make()
    with pytest.raises(Exception):
        df['bad'] = [1.0, 2.0]                 # len 2 != height 3


def test_setitem_is_copy_on_write():
    df = make()
    df2 = df.copy()
    df['new'] = [1.0, 2.0, 3.0]
    assert 'new' not in df2.columns            # adding a column doesn't touch the copy
    df['open'] = [9.0, 9.0, 9.0]
    assert df2['open'].to_numpy().tolist() == [1.0, 2.0, 3.0]   # replacing doesn't either
