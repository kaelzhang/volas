"""Indexer assignment — df.loc/.iloc/.at/.iat = value (audit PD-12)."""

import numpy as np
import pytest
import volas
from volas import DataFrame


def base():
    return DataFrame({'a': [1., 2., 3., 4.], 'b': [10., 20., 30., 40.]})


def test_iat_cell():
    df = base()
    df.iat[1, 0] = 99.
    assert df['a'].to_list() == [1., 99., 3., 4.]


def test_at_cell():
    df = base()
    df.at[2, 'b'] = 88.            # default RangeIndex label == position
    assert df['b'].to_list() == [10., 20., 88., 40.]


def test_iloc_cell_and_column_scalar():
    df = base()
    df.iloc[0, 1] = 7.                 # single cell
    df.iloc[1:3, 0] = 5.               # slice of a column, broadcast scalar
    assert df['b'].to_list() == [7., 20., 30., 40.]
    assert df['a'].to_list() == [1., 5., 5., 4.]


def test_iloc_column_array():
    df = base()
    df.iloc[[0, 2], 0] = np.array([100., 300.])
    assert df['a'].to_list() == [100., 2., 300., 4.]


def test_loc_mask_scalar():
    df = base()
    mask = df['a'] > 2.                # bool Series
    df.loc[mask, 'b'] = 0.
    assert df['b'].to_list() == [10., 20., 0., 0.]


def test_loc_mask_array():
    df = base()
    df.loc[df['a'] >= 3., 'b'] = np.array([-1., -2.])
    assert df['b'].to_list() == [10., 20., -1., -2.]


def test_loc_label_slice():
    df = base()
    df.loc[1:2, 'a'] = 0.             # inclusive label slice on RangeIndex
    assert df['a'].to_list() == [1., 0., 0., 4.]


def test_loc_datetime_index_label():
    df = volas.DataFrame({'t': ['2020-01-01', '2020-01-02', '2020-01-03'], 'c': [1., 2., 3.]})
    df['t'] = volas.to_datetime(df['t'])
    df = df.set_index('t')
    df.at['2020-01-02', 'c'] = 9.
    assert df['c'].to_list() == [1., 9., 3.]


def test_assign_fractional_into_int_raises():
    # A fractional write into an int column is lossy and raises — consistent with
    # the Series path (`s[0] = 1.5` also raises), NOT the old silent widening to
    # float64. The NA model keeps int columns int.
    df = DataFrame({'i': np.array([1, 2, 3], dtype=np.int64)})
    assert str(df.dtypes['i']) == 'int64'
    with pytest.raises(TypeError):
        df.iat[0, 0] = 1.5
    assert str(df.dtypes['i']) == 'int64' and df['i'].to_list() == [1, 2, 3]  # unchanged


def test_assign_nan_into_int_keeps_int_na():
    # A NaN write into an int column keeps int64 and marks the cell NA (Decision 1:
    # no float widening) — again matching the Series path.
    df = DataFrame({'i': np.array([1, 2, 3], dtype=np.int64)})
    df.iat[0, 0] = float('nan')
    assert str(df.dtypes['i']) == 'int64'
    assert df.isna()['i'].to_list() == [True, False, False]
    assert df['i'].to_list()[1:] == [2, 3]


def test_assign_int_stays_int():
    df = DataFrame({'i': np.array([1, 2, 3], dtype=np.int64)})
    df.iat[0, 0] = 7                 # integral write stays int64
    assert str(df.dtypes['i']) == 'int64'
    assert df['i'].to_list() == [7, 2, 3]


def test_assign_drops_computed_then_safe():
    df = DataFrame({'close': [1., 2., 3., 4., 5., 6.]})
    _ = df['ma:2']                    # cache a directive column
    df.loc[df['close'] > 3., 'ma:2'] = -1.   # manual override drops computed
    # a later append must NOT silently clobber the override -> column is plain now
    df.append(DataFrame({'close': [7.], 'ma:2': [0.]}))
    assert df['ma:2'].to_list()[-1] == 0.


def test_setitem_string_value_is_scalar():
    df = DataFrame({'s': ['x', 'y', 'z']})
    df.iloc[1:3, 0] = 'Q'            # a str is a scalar, broadcast
    assert df['s'].to_list() == ['x', 'Q', 'Q']


def test_loc_wrong_length_raises():
    df = base()
    with pytest.raises(ValueError):
        df.loc[df['a'] > 0, 'b'] = np.array([1., 2.])   # 4 rows, 2 values


def test_loc_single_key_errors():
    df = base()
    with pytest.raises(TypeError):
        df.loc[0] = 5.               # whole-row assignment unsupported


def test_setitem_preserves_int_bool_validity():
    # a scalar write keeps other rows' NA (regression: the NA at idx 1 used to
    # become a dense 0 / false). Series and the DataFrame indexers must agree.
    s = DataFrame({'a': [1, None, 3]})['a']
    s[0] = 9
    assert s.dtype == 'int64' and s.to_list() == [9, volas.NA, 3]
    b = DataFrame({'a': [True, None, False]})['a']
    b[0] = True
    assert b.dtype == 'bool' and b.to_list() == [True, volas.NA, False]
    # writing NA (NaN) keeps the int dtype, marking the cell missing (no upcast)
    s2 = DataFrame({'a': [1, 2, 3]})['a']
    s2[1] = float('nan')
    assert s2.dtype == 'int64' and s2.isna().to_list() == [False, True, False]
    # df.iat and df.iloc agree (they share the same column assignment)
    df = DataFrame({'a': [1, None, 3]})
    df.iat[0, 0] = 9
    assert df['a'].to_list() == [9, volas.NA, 3]
    df2 = DataFrame({'a': [1, None, 3]})
    df2.iloc[0, 0] = 9
    assert df2['a'].to_list() == [9, volas.NA, 3]


def test_series_setitem_string_scalar():
    # a str scalar assignment works on the Series surface too (was a TypeError),
    # preserving the existing validity — parity with the DataFrame indexers
    s = DataFrame({'a': ['x', None, 'z']})['a']
    s[0] = 'q'
    assert s.dtype == 'str' and s.to_list() == ['q', volas.NA, 'z']
    # a string into a numeric column is a clear TypeError (not a silent coercion)
    si = DataFrame({'a': [1, 2, 3]})['a']
    with pytest.raises(TypeError):
        si[0] = 'q'


def test_series_setitem_datetime_string_scalar():
    # P2-01 / Decision 2: a parseable datetime string assigns into a datetime
    # Series, including a NaT cell — parity with df.iat (was a TypeError)
    s = DataFrame({'a': np.array(['2021-01-01', 'NaT'], dtype='datetime64[ns]')})['a']
    s[1] = '2021-01-02'
    assert s.to_list()[1] == np.datetime64('2021-01-02') and s.isna().to_list() == [False, False]


def test_series_setitem_none_marks_na():
    # assigning None marks the cell NA, keeping the dtype (uniform across surfaces)
    s = DataFrame({'a': [1, 2, 3]})['a']
    s[1] = None
    assert s.dtype == 'int64' and s.isna().to_list() == [False, True, False]
    sf = DataFrame({'a': [1.0, 2.0, 3.0]})['a']
    sf[0] = None
    assert sf.dtype == 'float64' and sf.isna().to_list() == [True, False, False]


# --- P0-01: writing into an existing NA cell (the validity-aware `scatter`) ----
# A DataFrame indexer write used to mutate the value buffer but NOT the validity,
# so assigning a real value INTO an NA cell silently left it NA (data loss). The
# matrix is surface (iat / iloc / at) x storage dtype, each filling the middle NA
# cell and asserting it becomes present with the value, dtype unchanged.

def _dt(strs):
    return np.array(strs, dtype='datetime64[ns]')


@pytest.mark.parametrize('surface', ['iat', 'iloc', 'at'])
@pytest.mark.parametrize('col,val,want', [
    (np.array([1., 0., 3.], dtype=np.float64), 9., 9.),       # f64
    (np.array([1., 0., 3.], dtype=np.float32), 9., 9.),       # f32
    ([1, None, 3], 9, 9),                                     # i64
    (np.array([1, 0, 3], dtype=np.int32), 9, 9),             # i32
    ([True, None, False], True, True),                        # bool
    (['x', None, 'z'], 'q', 'q'),                             # str
    (_dt(['2021-01-01', 'NaT', '2021-01-03']), '2021-01-02', '2021-01-02'),  # datetime
])
def test_indexer_fills_na_cell(surface, col, val, want):
    # seed the middle cell as NA for the dtypes whose constructor doesn't (f32/i32/
    # f64 numpy arrays carry no None) so every case writes into a real hole. A NaN
    # write keeps f32/f64 (in-band) and marks i32 NA — independent of the surface.
    df = DataFrame({'a': col})
    before_dt = df['a'].dtype
    if df.isna()['a'].to_list()[1] is False:
        df.iloc[1, 0] = float('nan')
    assert df.isna()['a'].to_list()[1] is True  # confirmed a hole before the write
    if surface == 'at':
        df.at[df.index[1], 'a'] = val
    else:
        getattr(df, surface)[1, 0] = val
    assert df.isna()['a'].to_list() == [False, False, False]
    assert df['a'].dtype == before_dt
    got = df['a'].to_list()[1]
    assert (got == np.datetime64(want)) if before_dt == 'datetime64[ns]' else (got == want)


def test_dataframe_iloc_int_nan_keeps_int():
    # P1-01: int + NaN via .iloc keeps int64 + NA (unified with Series / mask),
    # not the old float64 widening. int32 likewise stays int32.
    df = DataFrame({'a': [1, 2, 3]})
    df.iloc[1:3, 0] = float('nan')
    assert df['a'].dtype == 'int64' and df.isna()['a'].to_list() == [False, True, True]
    d32 = DataFrame({'a': np.array([1, 2, 3], dtype=np.int32)})
    d32.iloc[1:3, 0] = float('nan')
    assert d32['a'].dtype == 'int32' and d32.isna()['a'].to_list() == [False, True, True]
