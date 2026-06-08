"""DataFrame(data, columns=...) — strict column projection at construction.

`columns` selects and orders the columns to keep (the same projection as ``df[[...]]``):
a name not present raises ``KeyError``; an empty list or a duplicate name is rejected;
an absent column is never NaN-filled. It works over a dict and over a volas DataFrame —
and a tf-aware frame keeps its ``time_frame``, with the forming-period state projected to
match the kept columns.
"""

import pytest
from volas import DataFrame, to_datetime

D = {'open': [1., 2.], 'high': [3., 4.], 'low': [0., 1.], 'close': [2., 3.], 'volume': [10., 20.]}


# --- dict data -------------------------------------------------------------

def test_dict_subset_and_reorder():
    assert DataFrame(D, columns=['close', 'volume']).columns == ['close', 'volume']
    assert DataFrame(D, columns=['volume', 'open']).columns == ['volume', 'open']


def test_dict_keeps_the_right_values():
    df = DataFrame(D, columns=['volume', 'close'])
    assert df['close'].to_list() == [2., 3.]
    assert df['volume'].to_list() == [10., 20.]


def test_dict_missing_name_raises_keyerror():
    with pytest.raises(KeyError):
        DataFrame(D, columns=['close', 'nope'])


def test_columns_none_is_unchanged():
    assert DataFrame(D).columns == ['open', 'high', 'low', 'close', 'volume']


# --- validation (shared by both data kinds) --------------------------------

def test_empty_columns_raises():
    with pytest.raises(ValueError):
        DataFrame(D, columns=[])


def test_duplicate_column_raises():
    with pytest.raises(ValueError):
        DataFrame(D, columns=['close', 'close'])


# --- DataFrame data --------------------------------------------------------

def test_dataframe_subset_and_reorder():
    src = DataFrame(D)
    assert DataFrame(src, columns=['high', 'low']).columns == ['high', 'low']
    out = DataFrame(src, columns=['volume', 'open'])
    assert out.columns == ['volume', 'open']
    assert out['open'].to_list() == [1., 2.]


def test_dataframe_missing_name_raises_keyerror():
    with pytest.raises(KeyError):
        DataFrame(DataFrame(D), columns=['close', 'nope'])


# --- tf-aware DataFrame: columns keeps the time_frame and folds correctly ---

def _five_min():
    raw = {'t': ['2020-01-01 00:00:00', '2020-01-01 00:05:00'],
           'open': [1., 2.], 'high': [3., 4.], 'low': [0., 1.],
           'close': [2., 3.], 'volume': [10., 20.]}
    f = DataFrame(raw)
    f['t'] = to_datetime(f['t'])
    return DataFrame(f.set_index('t'), time_frame='5m')


def _fine(times, close, volume):
    bar = DataFrame({'t': times, 'close': close, 'volume': volume})
    bar['t'] = to_datetime(bar['t'])
    return bar.set_index('t')


def test_columns_on_tf_aware_keeps_tf_and_folds():
    sub = DataFrame(_five_min(), columns=['close', 'volume'])
    assert sub.columns == ['close', 'volume']
    # two finer bars in the same 5m period fold with the kept per-column aggregators:
    # close -> last, volume -> sum.
    sub.append(_fine(['2020-01-01 00:06:00', '2020-01-01 00:07:00'], [5., 6.], [7., 8.]))
    assert sub.iloc[-1]['close'] == 6.0
    assert sub.iloc[-1]['volume'] == 15.0
