"""Explicit Arrow snapshots and copying constructors isolate foreign writes."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from volas import DataFrame, Series
from test.exact import assert_exact


def arrow(values):
    return pa.Array.from_buffers(pa.from_numpy_dtype(values.dtype), len(values),
                                 [None, pa.py_buffer(values)])


class Stream:
    def __init__(self, array):
        self.array = array

    def __arrow_c_stream__(self):
        return pa.RecordBatchReader.from_batches(
            pa.schema([('close', self.array.type)]),
            [pa.record_batch([self.array], names=['close'])]).__arrow_c_stream__()


ENTRY_POINTS = {
    'frame_arrow': lambda a: DataFrame.from_arrow(pa.table({'close': a}), copy=True),
    'series_arrow': lambda a: DataFrame({'close': Series.from_arrow(a, copy=True)}),
    'stream': lambda a: DataFrame.from_arrow(Stream(a), copy=True),
    'frame_column': lambda a: DataFrame({'close': a}),
    'pandas_column': lambda a: DataFrame({'close': pd.Series(a, dtype=pd.ArrowDtype(a.type))}),
    'pandas_arrow': lambda a: DataFrame.from_pandas(
        pd.DataFrame({'close': pd.Series(a, dtype=pd.ArrowDtype(a.type))})),
}


@pytest.mark.parametrize('entry', ENTRY_POINTS)
@pytest.mark.parametrize('changed', [np.nan, 1000.0])
def test_every_foreign_entry_freezes_cache_and_resume(entry, changed):
    source = np.arange(1., 161.)
    snapshot = source.copy()
    frame = ENTRY_POINTS[entry](arrow(source))
    reference = DataFrame({'close': snapshot})
    assert_exact(frame['ema:5'], reference['ema:5'])
    source[100] = changed
    assert_exact(frame['close'], snapshot)
    frame.fulfill()
    assert_exact(frame['ema:5'], reference.exec('ema:5'))
    assert_exact(frame.exec('ema:5'), reference.exec('ema:5'))
    for value in [161., 162.]:
        frame.append({'close': value})
        reference.append({'close': value})
        assert_exact(frame['ema:5'], reference.exec('ema:5'))


@pytest.mark.parametrize('kind', ['series', 'frame', 'stream'])
def test_explicit_immutable_import_borrows(kind):
    source = np.arange(30., dtype='float64')
    a = arrow(source)
    if kind == 'series':
        s = Series.from_arrow(a, copy=False)
    else:
        obj = Stream(a) if kind == 'stream' else pa.table({'close': a})
        s = DataFrame.from_arrow(obj, copy=False)['close']
    assert s.to_arrow().buffers()[1].address == source.ctypes.data
    assert_exact(s, source)


@pytest.mark.parametrize('dtype', [pa.float32(), pa.float64(), pa.int32(), pa.int64(),
                                  pa.int8(), pa.uint16(), pa.bool_(), pa.string(),
                                  pa.large_string(), pa.timestamp('ns'),
                                  pa.timestamp('ms'), pa.date32(), pa.null()])
@pytest.mark.parametrize('nullable', [False, True])
@pytest.mark.parametrize('chunks', [1, 3])
def test_import_layout_matrix(dtype, nullable, chunks):
    if pa.types.is_null(dtype):
        values = [None] * 6
    elif pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        values = ['a', '', 'ω', 'bbb', 'x', 'z']
    elif pa.types.is_boolean(dtype):
        values = [True, False, True, True, False, True]
    elif pa.types.is_floating(dtype):
        values = [1., -0., np.nan, 2., 3., 4.]
    else:
        values = [1, 0, 3, 2, 4, 5]
    if nullable:
        values[4] = None
    a = pa.array(values, type=dtype)
    parts = [a] if chunks == 1 else [a.slice(i, 2) for i in range(0, 6, 2)]
    table = pa.table({'close': pa.chunked_array(parts)})
    owned = DataFrame.from_arrow(table, copy=True)['close']
    borrowed = DataFrame.from_arrow(table, copy=False)['close']
    single = Series.from_arrow(a, copy=True)
    assert owned.dtype == borrowed.dtype == single.dtype
    assert_exact(owned, borrowed)
    assert_exact(owned, single)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
def test_explicit_snapshot_owns_and_default_shares(dtype):
    source = np.arange(30, dtype=dtype)
    a = arrow(source)
    owned = Series.from_arrow(a, copy=True)
    borrowed = Series.from_arrow(a)
    assert owned.to_arrow().buffers()[1].address != source.ctypes.data
    assert borrowed.to_arrow().buffers()[1].address == source.ctypes.data
    del borrowed
    before = source.copy()
    source[10] = 99
    assert_exact(owned, before)


def test_import_precondition_is_public():
    for method in [Series.from_arrow, DataFrame.from_arrow]:
        assert 'never write' in method.__doc__
        assert 'copy=False' in method.__doc__


@pytest.mark.parametrize('changed', [np.nan, 1000.0])
def test_import_snapshot_survives_window_checkpoints(changed):
    source = np.arange(1., 161.)
    frame = DataFrame({'close': Series.from_arrow(arrow(source), copy=True)},
                      window=5, max_lookback=['ema:5'])
    reference = DataFrame({'close': source.copy()})
    source[100] = changed
    for value in range(161, 185):
        frame.append({'close': float(value)})
        reference.append({'close': float(value)})
        assert_exact(frame['ema:5'], reference['ema:5'].to_numpy()[-5:])
    frame.iloc[-1, 0] = 199.
    reference.iloc[-1, 0] = 199.
    assert_exact(frame['ema:5'], reference['ema:5'].to_numpy()[-5:])


@pytest.mark.parametrize('key', [pa.int8(), pa.int16(), pa.int32(), pa.int64(),
                                pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()])
@pytest.mark.parametrize('value_type', [pa.float32(), pa.float64(), pa.int32(), pa.int64(),
                                       pa.int8(), pa.uint64(), pa.bool_(), pa.string(),
                                       pa.large_string(), pa.string_view(), pa.timestamp('ms'),
                                       pa.decimal128(10, 2), pa.decimal256(50, 2)])
def test_dictionary_decode_writes_final_snapshot(key, value_type):
    from decimal import Decimal
    if pa.types.is_string(value_type) or pa.types.is_large_string(value_type) or pa.types.is_string_view(value_type):
        values = ['x', 'ω', None]
    elif pa.types.is_decimal(value_type):
        values = [Decimal('1.50'), Decimal('2.50'), None]
    elif pa.types.is_boolean(value_type):
        values = [True, False, None]
    else:
        values = [1, 2, None]
    values = pa.array(values, type=value_type)
    a = pa.DictionaryArray.from_arrays(pa.array([1, None, 0, 2], type=key), values)
    got = Series.from_arrow(a)
    expected = Series.from_arrow(pa.array([values[1].as_py(), None, values[0].as_py(), None],
                                         type=value_type))
    assert got.dtype == expected.dtype
    assert_exact(got, expected)
    table = pa.table({'close': pa.chunked_array([a, a])})
    assert_exact(DataFrame.from_arrow(table)['close'],
                 DataFrame.from_arrow(table, copy=False)['close'])


def test_zero_copy_timestamp_rejects_a_present_nat_sentinel():
    a = pa.Array.from_buffers(pa.timestamp('ns'), 1,
                             [None, pa.py_buffer(np.array([-(2**63)], dtype='int64'))])
    with pytest.raises(ValueError):
        Series.from_arrow(a, copy=False)


def test_large_decimal256_converts_straight_to_nearest_float():
    from decimal import Decimal
    value = Decimal('10000000000000000000000000000000000000000')
    a = pa.array([value, None], type=pa.decimal256(60, 0))
    assert_exact(Series.from_arrow(a), np.array([float(value), np.nan]))


@pytest.mark.parametrize('dtype', [pa.int32(), pa.int64(), pa.float32(), pa.float64(), pa.timestamp('ns')])
def test_snapshot_isolates_values_and_the_validity_bitmap(dtype):
    numpy_dtype = dtype.to_pandas_dtype() if not pa.types.is_timestamp(dtype) else np.dtype('int64')
    values = np.arange(4, dtype=numpy_dtype)
    mask = np.array([0b1101], dtype='uint8')
    a = pa.Array.from_buffers(dtype, 4, [pa.py_buffer(mask), pa.py_buffer(values)])
    snapshot = Series.from_arrow(a, copy=True)
    expected = snapshot.to_numpy(dtype='object')
    values[:] = 9
    mask[0] = 0b0010
    assert_exact(snapshot.to_numpy(dtype='object'), expected)
