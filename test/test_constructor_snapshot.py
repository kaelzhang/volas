"""The no-override constructor copies once and preserves copy semantics."""
from pathlib import Path

import numpy as np
import pytest

from volas import DataFrame


def test_no_override_constructor_returns_before_staging_snapshot():
    # Rust-internal copies do not dispatch through the Python copy method. Guard
    # this allocation budget at its owner, alongside behavioral coverage below.
    source = (Path(__file__).resolve().parents[1] / 'crates' / 'volas-python' / 'src' /
              'frame_construct.rs').read_text()
    constructor = source.split('pub(crate) fn new(', 1)[1].split('/// The DatetimeIndex', 1)[0]
    assert constructor.index('return Ok(other.copy())') < constructor.index('let mut staged')


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
