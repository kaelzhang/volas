"""A cell write keeps a column's missing-value fact true to the values it leaves.

A float column remembers whether it holds a NaN; a recursive indicator reads that
fact to decide whether to look for missing rows. A batch that writes a NaN and then a
number at the same position leaves no NaN — the fact must say so, and a recursive
indicator over the column must compute, never raise about a missing row.
"""

import numpy as np
import pytest

import volas

DIRECTIVES = ['ema:3', 'smma:3', 'rsi:3', 'macd', 'atr:3', 'kdj.k']


def _frame():
    n = 40
    rng = np.random.default_rng(7)
    close = 10 + np.cumsum(rng.normal(0, 0.5, n))
    t = np.datetime64('2024-01-01', 'ns') + np.arange(n) * np.timedelta64(1, 'm')
    return volas.DataFrame({
        't': t, 'open': close + 0.1, 'high': close + 1.0, 'low': close - 1.0,
        'close': close, 'volume': np.ones(n),
    }).set_index('t')


def _expected(df, directive):
    fresh = volas.DataFrame({c: df[c].to_numpy() for c in df.columns})
    return fresh[directive].to_numpy()


@pytest.mark.parametrize('via', ['iloc', 'loc'])
@pytest.mark.parametrize('directive', DIRECTIVES)
def test_a_nan_overwritten_in_the_same_batch_leaves_the_column_complete(via, directive):
    df = _frame()
    df['close'].to_numpy()          # the fact is learned: no NaN
    j = list(df.columns).index('close')
    if via == 'iloc':
        df.iloc[[3, 3], j] = [np.nan, 11.0]
    else:
        label = df.index[3]
        df.loc[[label, label], 'close'] = [np.nan, 11.0]
    assert df.iat[3, j] == 11.0 and not np.isnan(df['close'].to_numpy()).any()
    got = df[directive].to_numpy()
    np.testing.assert_array_equal(got, _expected(df, directive))
