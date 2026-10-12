"""Datetime COLUMNS hold UTC instants and export UTC-aware.

A volas datetime column stores absolute instants (UTC epoch-ns) and no zone of its
own; every export that can state a zone states UTC — pandas ``datetime64[ns, UTC]``,
Arrow ``timestamp[ns, tz=UTC]``, ``Timestamp`` cells with ``tz='UTC'``, ``.dt.tz`` —
so the instant is never ambiguous. NumPy ``datetime64`` has no zone slot and stays
``datetime64[ns]`` (the same UTC instants). A source column's zone is not stored; it
is restored by converting the UTC instants.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import volas

ZONE = 'Asia/Shanghai'


def _frame():
    t = pd.DatetimeIndex(['2024-01-02 09:30', '2024-01-02 09:31', 'NaT'], tz=ZONE)
    return volas.DataFrame.from_pandas(pd.DataFrame({'t': t, 'c': [1.0, 2.0, 3.0]}))


@pytest.mark.parametrize('backend', ['numpy', 'numpy_nullable'])
def test_to_pandas_exports_datetime_columns_utc_aware(backend):
    out = _frame().to_pandas(dtype_backend=backend)
    assert str(out['t'].dtype) == 'datetime64[ns, UTC]'
    assert out['t'].iloc[0] == pd.Timestamp('2024-01-02 09:30', tz=ZONE)  # same instant
    assert pd.isna(out['t'].iloc[2])
    # the source zone is restored by converting the UTC instants
    assert out['t'].dt.tz_convert(ZONE).iloc[1] == pd.Timestamp('2024-01-02 09:31', tz=ZONE)


def test_naive_source_column_exports_as_utc_instants():
    t = np.array(['2024-01-02T01:30'], 'datetime64[ns]')
    out = volas.DataFrame({'t': t}).to_pandas()
    assert out['t'].iloc[0] == pd.Timestamp('2024-01-02 01:30', tz='UTC')


def test_arrow_exports_datetime_columns_utc_aware():
    df = _frame()
    table = pa.table(df)
    assert table.schema.field('t').type == pa.timestamp('ns', tz='UTC')
    assert table.column('t').null_count == 1
    arr = pa.array(df['t'])
    assert arr.type == pa.timestamp('ns', tz='UTC')
    # and back: an Arrow tz-aware column imports as the same instants
    back = volas.DataFrame.from_arrow(table)
    assert back['t'].to_list()[:2] == df['t'].to_list()[:2]


def test_series_cells_and_accessor_are_utc():
    s = _frame()['t']
    cells = s.to_list()
    assert cells[0].tz == 'UTC'
    assert cells[0] == volas.Timestamp('2024-01-02 09:30', tz=ZONE)
    assert cells[2] is volas.NA
    assert all(v.tz == 'UTC' for v in s.to_dict().values() if v is not volas.NA)
    assert s.dt.tz == 'UTC'
    assert s.dt.strftime('%H:%M%z').to_list()[0] == '01:30+0000'
    obj = s.to_numpy(dtype='object')
    assert obj[0].tz == 'UTC'


@pytest.mark.parametrize('values,want', [
    ([2 ** 62 + 1, None, 3], [2 ** 62 + 1, volas.NA, 3]),   # exact int, never via float
    ([True, None, False], [True, volas.NA, False]),
    (['a', None, 'c'], ['a', volas.NA, 'c']),
])
def test_series_object_export_is_typed_cells(values, want):
    # `dtype='object'` is the lossless export: each cell its own typed value, a
    # missing one volas.NA (or na_value) — the same cells a frame's object export has
    s = volas.DataFrame({'x': values})['x']
    for arr in (s.to_numpy(dtype='object'), np.asarray(s, dtype='object')):
        assert arr.dtype == object
        got = arr.tolist()
        assert [type(v) for v in got] == [type(v) for v in want]
        assert [v is volas.NA if w is volas.NA else v == w for v, w in zip(got, want)] == [True] * 3
    filled = s.to_numpy(dtype='object', na_value=0).tolist()
    assert filled == [0 if w is volas.NA else w for w in want]
    assert s.to_numpy(dtype='object').tolist() == volas.DataFrame({'x': values}).to_numpy(
        dtype='object')[:, 0].tolist()


def test_numpy_export_stays_datetime64_of_the_utc_instants():
    arr = _frame()['t'].to_numpy()
    assert arr.dtype == np.dtype('datetime64[ns]')
    assert arr[0] == np.datetime64('2024-01-02T01:30')


def test_column_zone_is_restored_through_the_index():
    # the documented manual restore: the column's UTC instants become a UTC index,
    # which tz_convert restates in the source zone
    df = _frame().iloc[0:2]
    restored = df.set_index('t').tz_localize('UTC').tz_convert(ZONE)
    assert restored.tz == ZONE
    assert list(restored.to_pandas().index) == list(
        pd.DatetimeIndex(['2024-01-02 09:30', '2024-01-02 09:31'], tz=ZONE))
