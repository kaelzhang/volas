"""The no-override constructor copies once and preserves copy semantics."""
import numpy as np
import pytest

from volas import DataFrame


@pytest.mark.parametrize('kind', ['plain', 'window', 'tf', 'stale'])
def test_no_override_constructor_preserves_snapshot_and_continuation(kind):
    index = np.array([0, 60, 120], dtype='datetime64[s]').astype('datetime64[ns]')
    original = DataFrame({'close': [1., 2., 3.]}, index=index,
                         window=2 if kind == 'window' else None,
                         max_lookback=['ema:2'] if kind == 'window' else None,
                         time_frame='1m' if kind == 'tf' else None)
    original['ema:2']
    if kind == 'stale':
        original.append(DataFrame({'close': [4.]}, index=np.array([180], dtype='datetime64[s]')))
    copied = DataFrame(original)
    reference = original.copy()
    for frame in (copied, reference):
        frame.fulfill()
    assert copied.equals(reference)
    assert copied._physical_height == reference._physical_height
    original['close'] = [9.] * len(original)
    assert copied.equals(reference)
    for frame in (copied, reference):
        frame.append(DataFrame({'close': [5.]}, index=np.array([240], dtype='datetime64[s]')))
        frame.fulfill()
    assert copied.equals(reference)
