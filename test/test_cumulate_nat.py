"""A DatetimeIndex containing NaT (a bar with no timestamp) must not silently
corrupt time-bucketed aggregation, and NaT must not render as a garbage civil
date."""

import numpy as np
import pytest
from volas import DataFrame


def _ohlcv_with_nat():
    return DataFrame({
        'date': ['2021-01-04T14:30:00', None, '2021-01-04T14:31:00'],
        'open': [1.0, 2.0, 3.0], 'high': [1.0, 2.0, 3.0], 'low': [1.0, 2.0, 3.0],
        'close': [1.0, 2.0, 3.0], 'volume': [10.0, 20.0, 30.0],
    }).astype({'date': 'datetime64[ns]'}).set_index('date')


def test_cumulate_rejects_nat_index():
    # P1-03: a NaT index row used to be bucketed as its own period (silent wrong
    # OHLCV, shape (3,5) instead of an aggregated daily bar). It must now raise.
    df = _ohlcv_with_nat()
    with pytest.raises(ValueError):
        df.cumulate('1d')


def test_cumulate_clean_index_still_works():
    # a clean DatetimeIndex still aggregates (regression guard for the new check)
    df = DataFrame({
        'date': ['2021-01-04T14:30:00', '2021-01-04T14:31:00'],
        'open': [1.0, 3.0], 'high': [1.0, 3.0], 'low': [1.0, 3.0],
        'close': [1.0, 3.0], 'volume': [10.0, 30.0],
    }).astype({'date': 'datetime64[ns]'}).set_index('date')
    out = df.cumulate('1d')
    assert out.shape == (1, 5) and out['close'].to_list() == [3.0]


def test_nat_astype_str_is_nat_not_garbage_date():
    # P1-03 (civil_parts/format): NaT must stringify as 'NaT', not 1677-09-21
    s = DataFrame({'d': np.array(['2021-03-15', 'NaT'], dtype='datetime64[ns]')})['d']
    assert s.astype('str').to_list() == ['2021-03-15 00:00:00', 'NaT']
