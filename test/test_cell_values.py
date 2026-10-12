"""One classification of an assigned value: an array (any dimension but 0, a list, a
tuple, a Series) is values, never one cell — a cell write (`iat` / `at` / `iloc` /
`loc` of one cell) and a dict bar take one value, as a whole-column assignment does.
And an empty column selection is a frame of no columns over the same rows."""

import warnings

import numpy as np
import pytest

import volas


def frame():
    return volas.DataFrame({'close': [1.0, 2.0], 'n': np.array([1, 2], 'int64')})


@pytest.mark.parametrize('value', [np.array([5.0]), [5.0], (5.0,), np.array([[5.0]])])
def test_a_cell_write_takes_one_value_not_an_array(value):
    df = frame()
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # never through NumPy's deprecated conversion
        with pytest.raises(TypeError, match='a cell takes one value, not an array'):
            df.iat[0, 0] = value
        with pytest.raises(TypeError, match='a cell takes one value, not an array'):
            df.at[0, 'close'] = value
    assert df['close'].to_numpy().tolist() == [1.0, 2.0]


def test_a_scalar_cell_keeps_working():
    df = frame()
    df.iat[0, 0] = np.float64(7.0)
    df.iat[1, 0] = np.array(8.0)  # a 0-d array is one value
    df.iat[0, 1] = np.int32(3)
    assert df['close'].to_numpy().tolist() == [7.0, 8.0]
    assert df['n'].to_numpy().tolist() == [3, 2]


def test_a_dict_bar_takes_one_value_per_column():
    df = volas.DataFrame({'t': volas.to_datetime(['2024-01-01']), 'close': [1.0]}).set_index('t')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(TypeError, match='a cell takes one value, not an array'):
            df.append({'t': '2024-01-02', 'close': np.array([2.0])})
    assert len(df) == 1
    df.append({'t': '2024-01-02', 'close': np.float64(2.0)})
    assert df['close'].to_numpy().tolist() == [1.0, 2.0]


def test_an_empty_selection_is_a_frame_of_no_columns_over_the_rows():
    df = volas.DataFrame({'a': [1.0, 2.0]}, index=[5, 6])
    df['ema:2@a']
    empty = df[[]]
    assert empty.shape == (2, 0)
    assert empty.columns == []
    assert empty.index.tolist() == [5, 6]
    w = volas.DataFrame({'a': [float(i) for i in range(10)]}, window=3, max_lookback=2)
    assert w[[]].shape == (3, 0)
    assert w[[]].index.tolist() == [7, 8, 9]
