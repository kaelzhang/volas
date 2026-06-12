"""volas string (object) row-index tests.

The counterpart to the datetime / range cases in ``test_indexing.py``: mirrors
stock-pandas / pandas string-index behaviour — ``set_index`` on a string column,
label lookup, lexicographic ``.loc[a:b]`` slicing, ``.at``, ``drop``, and the
``.index`` object array.
"""

import numpy as np
import pytest

from volas import DataFrame, Series


@pytest.fixture
def df():
    # rows keyed by an ascending string symbol column
    base = DataFrame({
        'sym': ['aa', 'bb', 'cc', 'dd'],
        'open': [1.0, 2.0, 3.0, 4.0],
        'close': [1.5, 2.5, 3.5, 4.5],
    })
    return base.set_index('sym')


def test_set_index_builds_string_index(df):
    # the string column moved out of the columns into the index ...
    assert 'sym' not in df.columns
    assert list(df.index) == ['aa', 'bb', 'cc', 'dd']
    # ... exposed as a NumPy object array (pandas parity)
    assert df.index.dtype == np.dtype('object')


def test_loc_label_lookup(df):
    row = df.loc['cc']
    assert row.name == 'cc'
    assert row['open'] == 3.0


def test_loc_missing_label_raises(df):
    with pytest.raises(KeyError):
        df.loc['zz']


def test_loc_lexicographic_slice(df):
    # inclusive on both ends, like pandas label slicing
    assert list(df.loc['bb':'cc'].index) == ['bb', 'cc']
    # open-ended upper / lower bounds
    assert list(df.loc['cc':].index) == ['cc', 'dd']
    assert list(df.loc[:'bb'].index) == ['aa', 'bb']


def test_at_scalar_by_string_label(df):
    assert df.at['dd', 'close'] == 4.5


def test_drop_by_string_labels(df):
    dropped = df.drop(['bb', 'dd'])
    assert list(dropped.index) == ['aa', 'cc']
    assert len(dropped) == 2


def test_iloc_is_still_positional(df):
    row = df.iloc[1]
    assert row.name == 'bb'
    assert row['open'] == 2.0


def test_series_loc_on_string_index(df):
    s = df['close']
    assert isinstance(s, Series)
    assert s.loc['cc'] == 3.5
    assert list(s.loc['bb':'cc'].index) == ['bb', 'cc']


def test_row_name_round_trips(df):
    row = df.iloc[2]
    assert df.loc[row.name]['open'] == row['open']
