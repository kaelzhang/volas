"""Series comparison + logical operators (audit PD-2 / PD-3).

Expected values are hand-computed (pandas/NaN semantics) and inlined, so the
suite is pandas-free; the exhaustive parity vs pandas lives in
``test_pandas_series_ops`` / ``test_pandas_reductions``. The boolean result of a
comparison is usable directly as a row mask.
"""

import math

import numpy as np

from volas import DataFrame

# a = [1.0, nan, 3.0, 4.0], b = [2.0, 2.0, 2.0, 5.0]


def make():
    return DataFrame({'a': [1.0, np.nan, 3.0, 4.0], 'b': [2.0, 2.0, 2.0, 5.0]})


def test_comparison_ops():
    df = make()
    a, b = df['a'], df['b']
    # NaN compares False for every ordered/equality op except !=.
    assert (a < b).to_numpy().tolist() == [True, False, False, True]
    assert (a <= b).to_numpy().tolist() == [True, False, False, True]
    assert (a == b).to_numpy().tolist() == [False, False, False, False]
    assert (a != b).to_numpy().tolist() == [True, True, True, True]
    assert (a >= b).to_numpy().tolist() == [False, False, True, False]
    assert (a > b).to_numpy().tolist() == [False, False, True, False]
    # vs scalar
    assert (a < 3).to_numpy().tolist() == [True, False, False, False]
    # NaN: nan != 2 -> True, nan == 2 / nan < 2 -> False
    assert (a != b).to_numpy().tolist()[1] is True
    assert (a == b).to_numpy().tolist()[1] is False
    # result dtype is bool
    assert (a < b).dtype == 'bool'


def test_logical_ops():
    df = make()
    a, b = df['a'], df['b']
    m1, m2 = (a < b), (b < 5)        # m1 = [T,F,F,T], m2 = [T,T,T,F]
    assert (m1 & m2).to_numpy().tolist() == [True, False, False, False]
    assert (m1 | m2).to_numpy().tolist() == [True, True, True, True]
    assert (m1 ^ m2).to_numpy().tolist() == [False, True, True, True]
    assert (~m1).to_numpy().tolist() == [False, True, True, False]
    assert (m1 & m2).dtype == 'bool'


def test_comparison_used_as_row_mask():
    df = make()
    mask = df['a'] < df['b']      # True at rows 0 (1<2) and 3 (4<5)
    filtered = df[mask]
    assert len(filtered) == 2


def test_reflected_logical_operand():
    df = make()
    m = df['a'] < df['b']
    assert (True & m).to_numpy().tolist() == m.to_numpy().tolist()


def test_reductions():
    s = make()['a']                 # [1.0, nan, 3.0, 4.0] -> non-NaN [1, 3, 4]
    assert s.sum() == 8.0
    assert math.isclose(s.mean(), 8.0 / 3.0)
    assert s.min() == 1.0
    assert s.max() == 4.0
    assert math.isclose(s.var(), 7.0 / 3.0)         # ddof=1 over [1,3,4]
    assert math.isclose(s.std(), math.sqrt(7.0 / 3.0))
    assert s.median() == 3.0                         # median of [1,3,4]


def test_reduction_edge_cases():
    allnan = DataFrame({'x': [np.nan, np.nan]})['x']
    assert allnan.sum() == 0.0                    # pandas: sum of all-NaN is 0
    assert np.isnan(allnan.mean())
    assert np.isnan(allnan.min()) and np.isnan(allnan.max())
    one = DataFrame({'x': [5.0, np.nan]})['x']
    assert np.isnan(one.var()) and np.isnan(one.std())   # ddof=1 needs >=2 values
    # even-length median = mean of the two middle values
    even = DataFrame({'x': [1.0, 2.0, 3.0, 4.0]})['x']
    assert even.median() == 2.5
