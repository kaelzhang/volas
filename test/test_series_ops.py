"""Series comparison + logical operators (audit PD-2 / PD-3).

Verified element-for-element against pandas, including NaN semantics. The result
is a boolean Series usable directly as a row mask.
"""

import numpy as np
import pandas as pd

from volas import DataFrame


def make():
    return DataFrame({'a': [1.0, np.nan, 3.0, 4.0], 'b': [2.0, 2.0, 2.0, 5.0]})


PA = pd.Series([1.0, np.nan, 3.0, 4.0])
PB = pd.Series([2.0, 2.0, 2.0, 5.0])


def test_comparison_ops_match_pandas():
    df = make()
    a, b = df['a'], df['b']
    assert (a < b).to_numpy().tolist() == (PA < PB).tolist()
    assert (a <= b).to_numpy().tolist() == (PA <= PB).tolist()
    assert (a == b).to_numpy().tolist() == (PA == PB).tolist()
    assert (a != b).to_numpy().tolist() == (PA != PB).tolist()
    assert (a >= b).to_numpy().tolist() == (PA >= PB).tolist()
    assert (a > b).to_numpy().tolist() == (PA > PB).tolist()
    # vs scalar
    assert (a < 3).to_numpy().tolist() == (PA < 3).tolist()
    # NaN: nan != 2 -> True, nan == 2 / nan < 2 -> False
    assert (a != b).to_numpy().tolist()[1] is True
    assert (a == b).to_numpy().tolist()[1] is False
    # result dtype is bool
    assert (a < b).dtype == 'bool'


def test_logical_ops_match_pandas():
    df = make()
    a, b = df['a'], df['b']
    m1, m2 = (a < b), (b < 5)
    pm1, pm2 = (PA < PB), (PB < 5)
    assert (m1 & m2).to_numpy().tolist() == (pm1 & pm2).tolist()
    assert (m1 | m2).to_numpy().tolist() == (pm1 | pm2).tolist()
    assert (m1 ^ m2).to_numpy().tolist() == (pm1 ^ pm2).tolist()
    assert (~m1).to_numpy().tolist() == (~pm1).tolist()
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


def test_reductions_match_pandas():
    s = make()['a']                 # [1.0, nan, 3.0, 4.0]
    ps = PA
    assert s.sum() == ps.sum()
    assert s.mean() == ps.mean()
    assert s.min() == ps.min()
    assert s.max() == ps.max()
    assert abs(s.var() - ps.var()) < 1e-12        # ddof=1
    assert abs(s.std() - ps.std()) < 1e-12
    assert s.median() == ps.median()


def test_reduction_edge_cases():
    from volas import DataFrame as DF
    allnan = DF({'x': [np.nan, np.nan]})['x']
    assert allnan.sum() == 0.0                    # pandas: sum of all-NaN is 0
    assert np.isnan(allnan.mean())
    assert np.isnan(allnan.min()) and np.isnan(allnan.max())
    one = DF({'x': [5.0, np.nan]})['x']
    assert np.isnan(one.var()) and np.isnan(one.std())   # ddof=1 needs >=2 values
    # even-length median = mean of the two middle values
    even = DF({'x': [1.0, 2.0, 3.0, 4.0]})['x']
    assert even.median() == 2.5
