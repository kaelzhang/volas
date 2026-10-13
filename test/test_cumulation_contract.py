"""Regression contracts for cumulation contract."""

import numpy as np
import pandas as pd
import pytest

import volas


def bar(minute, volume):
    return volas.DataFrame({'volume': [float(volume)]},
                           index=[volas.Timestamp(f'2024-01-01 00:{minute:02d}')])



def test_same_time_frame_cumulate_applies_new_cumulators_on_later_folds():
    source = bar(0, 1).cumulate('5m')
    source.append(bar(1, 2))
    source['ema:2@volume']
    result = source.cumulate('5m', cumulators={'volume': 'max'})
    result.append(bar(2, 4))
    assert result['volume'].to_list() == [4.0]
    assert 'ema:2@volume' not in result.columns
    assert source['volume'].to_list() == [3.0]



@pytest.mark.parametrize('spec,error', [
    ({'absent': 'sum'}, 'does not exist'),
    ({'volume': 'unknown'}, 'unknown aggregator'),
])
def test_same_time_frame_cumulate_validates_cumulators(spec, error):
    source = bar(0, 1).cumulate('5m')
    with pytest.raises(ValueError, match=error):
        source.cumulate('5m', cumulators=spec)



@pytest.mark.parametrize('tz', [None, 'UTC', 'Asia/Shanghai', 'America/New_York'])
def test_empty_datetime_cumulate_keeps_schema_dtypes_timezone_and_accepts_bars(tz):
    seed = pd.DataFrame({
        'f32': np.array([], dtype='float32'), 'f64': np.array([], dtype='float64'),
        'i32': np.array([], dtype='int32'), 'i64': np.array([], dtype='int64'),
        'flag': np.array([], dtype='bool'), 'text': pd.Series([], dtype='str'),
        'date': np.array([], dtype='datetime64[ns]'),
    }, index=pd.DatetimeIndex([], tz=tz, name='time'))
    source = volas.DataFrame.from_pandas(seed)
    result = source.cumulate('5m')
    assert result.shape == (0, 7)
    assert result.columns == source.columns
    assert result.dtypes == source.dtypes
    with pytest.raises(ValueError, match='whole multiple'):
        result.cumulate('3m')
    assert result.to_pandas().index.name == 'time'
    assert result.to_pandas().index.equals(seed.index)
    assert result.cumulate('15m').to_pandas().index.equals(seed.index)
    row = pd.DataFrame({
        'f32': np.array([-0.0], dtype='float32'), 'f64': np.array([-0.0], dtype='float64'),
        'i32': np.array([2], dtype='int32'), 'i64': np.array([2**53 + 1], dtype='int64'),
        'flag': [True], 'text': ['a'], 'date': [pd.Timestamp('2024-01-02')],
    }, index=pd.DatetimeIndex(['2024-01-01'], tz=tz, name='time'))
    result.append(volas.DataFrame.from_pandas(row))
    assert result['i64'].to_list() == [2**53 + 1]
    assert result.dtypes == source.dtypes
    for name in ['f32', 'f64']:
        assert np.signbit(result[name].to_numpy()[0])



def test_empty_cumulate_still_validates_index_and_dtype_support():
    with pytest.raises(ValueError, match='DatetimeIndex'):
        volas.DataFrame({'volume': np.array([], dtype='int64')}).cumulate('5m')
    empty = pd.DataFrame({'text': pd.Series([], dtype='str')},
                         index=pd.DatetimeIndex([], tz='UTC'))
    with pytest.raises((TypeError, ValueError), match='cannot sum'):
        volas.DataFrame.from_pandas(empty).cumulate('5m', cumulators={'text': 'sum'})

