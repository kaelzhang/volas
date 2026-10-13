"""NumPy input copies any stride in logical order without changing values."""

import numpy as np
import pandas as pd
import pytest

from volas import DataFrame


@pytest.mark.parametrize('dtype', ['float64', 'float32', 'int64', 'int32', 'bool', 'uint64', 'datetime64[ns]'])
@pytest.mark.parametrize('step', [2, -1])
def test_numpy_strided_input_preserves_values_and_dtype(dtype, step):
    base = np.arange(10).astype(dtype)
    if dtype.startswith('float'):
        base[:] = [-0.0, 0.0, np.nan, 1.0, -2.0, 3.0, 4.0, 5.0, np.inf, 6.0]
    values = base[::step]
    assigned = DataFrame({'x': np.arange(len(values)).astype(dtype)})
    assigned['x'] = values
    for out in [DataFrame({'x': values})['x'].to_numpy(), assigned['x'].to_numpy()]:
        np.testing.assert_array_equal(out, values)
        if dtype.startswith('float'):
            present = ~np.isnan(values)
            np.testing.assert_array_equal(np.isnan(out), np.isnan(values))
            bits = 'u' + str(out.dtype.itemsize)
            np.testing.assert_array_equal(out.view(bits)[present], values.view(bits)[present])
    pdf = pd.DataFrame({'x': base}).iloc[::step]
    # Descending datetime indexes are intentionally rejected; ordinary RangeIndex labels are int.
    np.testing.assert_array_equal(DataFrame.from_pandas(pdf)['x'].to_numpy(), values)


@pytest.mark.parametrize('api', ['frame', 'series', 'setitem'])
def test_strided_boolean_masks_keep_logical_order(api):
    frame = DataFrame({'x': [10, 20, 30]})
    mask = np.array([True, False, False, True, True, False])[::2]
    if api == 'frame':
        assert frame[mask]['x'].to_list() == [10, 30]
    elif api == 'series':
        assert frame['x'][mask].to_list() == [10, 30]
    else:
        frame[mask] = 0
        assert frame['x'].to_list() == [0, 20, 0]


@pytest.mark.parametrize('dtype', ['float64', 'int32', 'bool'])
def test_zero_stride_numpy_input_is_copied_in_logical_order(dtype):
    values = np.broadcast_to(np.array([1], dtype=dtype), (4,))
    np.testing.assert_array_equal(DataFrame({'x': values})['x'].to_numpy(), values)
