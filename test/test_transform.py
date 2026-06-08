"""shift / diff / fillna / dropna / sort_index / reset_index (PD-7/8/9/10).

Expected values are inlined (pandas semantics, hand-computed) so the suite is
pandas-free; the exhaustive vs-pandas parity lives in ``test_pandas_series_methods``.
"""

import numpy as np
from volas import DataFrame


def test_shift_diff():
    # x = [1, 2, 4, 7]
    s = DataFrame({'x': [1.0, 2.0, 4.0, 7.0]})['x']
    nan = float('nan')
    exp = {
        ('shift', 1): [nan, 1.0, 2.0, 4.0],
        ('shift', 2): [nan, nan, 1.0, 2.0],
        ('shift', -1): [2.0, 4.0, 7.0, nan],
        ('diff', 1): [nan, 1.0, 2.0, 3.0],
        ('diff', 2): [nan, nan, 3.0, 5.0],
        ('diff', -1): [-1.0, -2.0, -3.0, nan],   # x - x.shift(-1)
    }
    for (op, k), want in exp.items():
        got = getattr(s, op)(k).to_numpy()
        assert np.allclose(got, want, equal_nan=True), f"{op}({k}): {got.tolist()} != {want}"
    # default n=1 == shift(1)
    assert np.allclose(s.shift().to_numpy(), [nan, 1.0, 2.0, 4.0], equal_nan=True)


def test_fillna_isna_notna():
    s = DataFrame({'x': [1.0, np.nan, 3.0]})['x']
    assert s.fillna(0.0).to_numpy().tolist() == [1.0, 0.0, 3.0]
    assert s.isna().to_numpy().tolist() == [False, True, False]
    assert s.notna().to_numpy().tolist() == [True, False, True]


def test_series_dropna_carries_index():
    s = DataFrame({'x': [1.0, np.nan, 3.0]})['x']
    out = s.dropna()
    assert out.to_numpy().tolist() == [1.0, 3.0]
    assert out.index.tolist() == [0, 2]


def test_df_dropna_how():
    df = DataFrame({'a': [1.0, np.nan, 3.0], 'b': [np.nan, np.nan, 6.0]})
    assert len(df.dropna()) == 1               # 'any' -> only the all-present row
    assert len(df.dropna(how='all')) == 2      # 'all' -> only the all-NaN row dropped


def test_sort_index():
    df = DataFrame({'k': np.array([3, 1, 2], dtype=np.int64),
                    'v': [30.0, 10.0, 20.0]}).set_index('k')
    asc = df.sort_index()
    assert asc.index.tolist() == [1, 2, 3]
    assert asc['v'].to_numpy().tolist() == [10.0, 20.0, 30.0]
    assert df.sort_index(ascending=False).index.tolist() == [3, 2, 1]


def test_reset_index():
    df = DataFrame({'k': np.array([10, 20], dtype=np.int64), 'v': [1.0, 2.0]}).set_index('k')
    r = df.reset_index()
    # reset_index restores the original column label recorded by set_index (pandas parity)
    assert r.columns == ['k', 'v']
    assert r['k'].to_numpy().tolist() == [10, 20]
    assert r.index.tolist() == [0, 1]
    assert df.reset_index(drop=True).columns == ['v']
