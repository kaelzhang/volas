"""Explicit fills preserve NumPy's resolved units, widths and scalar data."""
import numpy as np
import pytest

from volas import DataFrame


def export(frame, entry, dtype, fill):
    if entry == 'frame':
        result = frame.to_numpy(dtype=dtype, na_value=fill)
        assert result.shape[:2] == (2, 1)
        return result[:, 0]
    if entry == 'series':
        return frame['x'].to_numpy(dtype=dtype, na_value=fill)
    return frame.exec('x', dtype=dtype, na_value=fill)


def assert_metadata_and_data(got, expected):
    assert got.dtype == expected.dtype
    assert got.dtype.str == expected.dtype.str
    assert got.dtype.itemsize == expected.dtype.itemsize
    if got.dtype.kind in 'Mm':
        assert np.datetime_data(got.dtype) == np.datetime_data(expected.dtype)
    if got.dtype.kind in 'MmV':
        assert got.tobytes() == expected.tobytes()
    else:
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize('entry', ['frame', 'series', 'exec'])
@pytest.mark.parametrize('all_missing', [False, True])
@pytest.mark.parametrize('dtype', ['datetime64', 'M', 'M8', 'timedelta64', 'm', 'm8'])
def test_generic_time_retains_source_unit(entry, all_missing, dtype):
    source = np.array([1, 2], dtype='datetime64[ns]')
    mask = np.array([all_missing, True])
    values = source.copy()
    values[mask] = 'NaT'
    fill = np.datetime64(7, 'ns') if np.dtype(dtype).kind == 'M' else np.timedelta64(7, 'ns')
    converted = source[~mask].astype(dtype)
    expected = np.empty(2, dtype=converted.dtype)
    expected[~mask] = converted
    expected[mask] = fill
    got = export(DataFrame({'x': values}), entry, dtype, fill)
    assert_metadata_and_data(got, expected)


@pytest.mark.parametrize('entry', ['frame', 'series', 'exec'])
@pytest.mark.parametrize('all_missing', [False, True])
@pytest.mark.parametrize('source_dtype', ['int64', 'int32', 'float64', 'float32', 'bool', 'datetime64[ns]'])
@pytest.mark.parametrize('dtype', ['V', 'V0', 'void'])
def test_unsized_void_retains_source_itemsize_and_bits(entry, all_missing, source_dtype, dtype):
    source = np.array([1, 0], dtype=source_dtype)
    mask = np.array([all_missing, True])
    values = np.ma.array(source, mask=mask)
    converted = source[~mask].astype(dtype)
    fill = b'x' * converted.dtype.itemsize
    expected = np.empty(2, dtype=converted.dtype)
    expected[~mask] = converted
    expected[mask] = fill
    got = export(DataFrame({'x': values}), entry, dtype, fill)
    assert_metadata_and_data(got, expected)


@pytest.mark.parametrize('entry', ['frame', 'series', 'exec'])
@pytest.mark.parametrize('all_missing', [False, True])
@pytest.mark.parametrize('dtype', ['S', 'S0', 'bytes', 'U', 'U0', 'str'])
@pytest.mark.parametrize('fill', ['x', 'missing value longer than the native integer width'])
def test_unsized_text_retains_resolved_width_and_fits_fill(entry, all_missing, dtype, fill):
    source = np.array([1, 0], dtype='int64')
    mask = np.array([all_missing, True])
    # All explicit entry points retain the same native source array.
    native = source
    converted = native[~mask].astype(dtype)
    resolved = np.result_type(converted.dtype, np.asarray(fill, dtype=dtype).dtype)
    expected = np.empty(2, dtype=resolved)
    expected[~mask] = converted
    expected[mask] = fill
    got = export(DataFrame({'x': np.ma.array(source, mask=mask)}), entry, dtype, fill)
    assert_metadata_and_data(got, expected)


# Sweep every scalar target kind registered by the installed NumPy, plus byte
# order, explicit units, sized flexible types, subarrays and structured records.
TARGETS = sorted({np.dtype(t).str for t in np.sctypeDict.values()} | {
    '>i4', '>f8', '>c16', 'M8[D]', 'm8[us]', 'V2', 'S2', 'U2',
    '(2,)i4', 'i4,f8', 'T',
})


@pytest.mark.parametrize('entry', ['frame', 'series', 'exec'])
@pytest.mark.parametrize('all_missing', [False, True])
@pytest.mark.parametrize('dtype', TARGETS)
def test_numpy_target_kind_resolution_sweep(entry, all_missing, dtype):
    source = np.array([1, 0], dtype='int64')
    mask = np.array([all_missing, True])
    kind = np.dtype(dtype).kind
    native = source
    # The independent oracle performs only valid-slot conversion, then allocates
    # using its resolved dtype. NumPy rejects some source/target combinations.
    try:
        converted = native[~mask].astype(dtype)
        fill = np.zeros((), dtype=converted.dtype)[()] if kind != 'O' else 0
        if kind in 'SU':
            resolved = np.result_type(converted.dtype, np.asarray(fill, dtype=dtype).dtype)
        else:
            resolved = converted.dtype
        expected = np.empty((2,) + converted.shape[1:], dtype=resolved)
        expected[~mask] = converted
        expected[mask] = fill
    except (TypeError, ValueError) as error:
        with pytest.raises(type(error)):
            export(DataFrame({'x': np.ma.array(source, mask=mask)}), entry, dtype, 0)
        return
    got = export(DataFrame({'x': np.ma.array(source, mask=mask)}), entry, dtype, fill)
    assert_metadata_and_data(got, expected)


@pytest.mark.parametrize('integer_dtype,float_dtype', [('int64', 'float64'), ('int32', 'float32'), ('int64', 'float32')])
def test_frame_void_converts_columns_before_numpy_promotion(integer_dtype, float_dtype):
    integer = 2**53 + 1 if integer_dtype == 'int64' else 2**24 + 1
    sources = [np.array([integer, 0], dtype=integer_dtype), np.array([1.5, 0], dtype=float_dtype)]
    frame = DataFrame({name: np.ma.array(values, mask=[False, True])
                       for name, values in zip(['i', 'f'], sources)})
    fill = b'abcd'
    # Convert each native column independently: promotion to a common numeric
    # matrix before the raw-byte cast would lose the integer and/or its itemsize.
    dtype = 'V'
    if sources[0].dtype.itemsize != sources[1].dtype.itemsize:
        # NumPy cannot promote raw voids of different lengths implicitly.
        with pytest.raises(np.exceptions.DTypePromotionError):
            frame.to_numpy(dtype=dtype, na_value=fill)
        dtype = 'V8'
    columns = []
    for source in sources:
        converted = source[:1].astype(dtype)
        column = np.empty(2, dtype=converted.dtype)
        column[:1] = converted
        column[1] = fill
        columns.append(column)
    expected = np.column_stack(columns)
    got = frame.to_numpy(dtype=dtype, na_value=fill)
    assert_metadata_and_data(got, expected)


def test_frame_void_without_fill_keeps_boolean_na_rejection():
    frame = DataFrame({'x': [True, None]})
    with pytest.raises(ValueError, match='na_value'):
        frame.to_numpy(dtype='V')
