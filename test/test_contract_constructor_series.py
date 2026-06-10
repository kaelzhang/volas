"""Contract C1/C2 — DataFrame({'x': volas_series}) must preserve the Series as a
volas column (dtype + volas.NA), not funnel it through Series.__array__ to the
numpy boundary and re-import it as float64. Parity with df['x'] = series."""

import numpy as np
import pytest
import volas
from volas import DataFrame

NA = volas.NA


def _col(values, dtype=None):
    return DataFrame({'a': np.array(values, dtype=dtype) if dtype is not None else values})['a']


def test_constructor_preserves_datetime_series():
    dt = _col(['2024-01-01', '2024-01-02'], 'datetime64[ns]')
    out = DataFrame({'x': dt})['x']
    assert out.dtype == 'datetime64[ns]'                  # was float64 epoch
    assert out.to_list() == dt.to_list()


@pytest.mark.parametrize('values,dtype', [
    ([1, None, 3], None),                  # int64 + NA
    ([True, None, False], None),           # bool + NA
    (['a', None, 'b'], None),              # str + NA
    (['2024-01-01', 'NaT'], 'datetime64[ns]'),   # datetime + NaT
])
def test_constructor_preserves_dtype_and_na(values, dtype):
    s = _col(values, dtype)
    # the already-correct assignment path is the reference
    ref = DataFrame({'a': [0] * len(values)})
    ref['x'] = s
    out = DataFrame({'x': s})['x']
    assert out.dtype == ref['x'].dtype
    assert out.isna().to_list() == ref['x'].isna().to_list()
