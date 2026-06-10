"""`volas.NA` — the canonical missing-value symbol returned by to_list() — must
be assignable / usable wherever `None` is: Series setitem, DataFrame indexers,
boolean-mask assignment, and where/mask `other`. (P2-01)"""

import numpy as np
import pytest
import volas
from volas import DataFrame

NA = volas.NA


def test_series_setitem_volas_na():
    s = DataFrame({'a': [1, 2, 3]})['a']
    s[0] = NA
    assert s.dtype == 'int64' and s.isna().to_list() == [True, False, False]


def test_dataframe_indexer_volas_na():
    df = DataFrame({'a': [1, 2], 'b': [3, 4]})
    df.iat[0, 0] = NA
    df.iloc[1, 1] = NA
    assert df.isna()['a'].to_list() == [True, False]
    assert df.isna()['b'].to_list() == [False, True]
    assert df['a'].dtype == 'int64'


def test_mask_assignment_volas_na():
    df = DataFrame({'a': [1, 2, 3]})
    df[np.array([False, True, False])] = NA
    assert df['a'].dtype == 'int64' and df.isna()['a'].to_list() == [False, True, False]


def test_where_other_volas_na():
    s = DataFrame({'a': [1, 2, 3]})['a']
    mask = DataFrame({'a': [True, False, True]})['a']
    r = s.where(mask, NA)   # keep where True, NA where False
    assert r.dtype == 'int64' and r.isna().to_list() == [False, True, False]
    assert r.to_list()[0] == 1 and r.to_list()[2] == 3
