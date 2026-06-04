"""DataFrame/Series convenience methods (audit PD-4 / PD-5 / PD-6)."""

import numpy as np
from volas import DataFrame


def make():
    return DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'b': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})


def test_contains_and_iter():
    df = make()
    assert 'a' in df and 'z' not in df
    assert list(df) == ['a', 'b']            # iteration yields column names (pandas)
    df.alias('A', 'a')
    assert 'A' in df                         # alias-aware


def test_head_tail():
    df = make()
    assert len(df.head()) == 5               # default n=5 over 6 rows
    assert df.head(3)['a'].to_list() == [1.0, 2.0, 3.0]
    assert df.tail(2)['a'].to_list() == [5.0, 6.0]
    assert len(df.head(100)) == 6            # clamped to height


def test_dtypes():
    df = DataFrame({'a': [1.0, 2.0], 'b': np.array([10, 20], dtype=np.int64),
                    's': ['x', 'y']})
    dt = df.dtypes
    assert dt == {'a': 'float64', 'b': 'int64', 's': 'object'}


def test_to_list():
    df = make()
    assert df['a'].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert df['a'].tolist() == df['a'].to_list()
