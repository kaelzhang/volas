"""shift / diff / fillna / dropna / sort_index / reset_index (PD-7/8/9/10)."""

import numpy as np
import pandas as pd
from volas import DataFrame


def test_shift_diff_match_pandas():
    s = DataFrame({'x': [1.0, 2.0, 4.0, 7.0]})['x']
    ps = pd.Series([1.0, 2.0, 4.0, 7.0])
    for k in (1, 2, -1):
        assert np.allclose(s.shift(k).to_numpy(), ps.shift(k).to_numpy(), equal_nan=True)
        assert np.allclose(s.diff(k).to_numpy(), ps.diff(k).to_numpy(), equal_nan=True)
    assert np.allclose(s.shift().to_numpy(), ps.shift().to_numpy(), equal_nan=True)


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
    assert r.columns == ['index', 'v']
    assert r['index'].to_numpy().tolist() == [10, 20]
    assert r.index.tolist() == [0, 1]
    assert df.reset_index(drop=True).columns == ['v']
