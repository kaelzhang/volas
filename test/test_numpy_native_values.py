"""Adding a missing slot must not change native valid-value conversion."""
from pathlib import Path

import numpy as np
import pytest

from volas import DataFrame


@pytest.mark.parametrize('entry', ['frame', 'series', 'exec'])
@pytest.mark.parametrize('dtype', ['datetime64[ns]', 'timedelta64[ns]', 'U', 'S', 'int64', 'uint64'])
@pytest.mark.parametrize('filled', [False, True])
def test_na_elsewhere_preserves_lossless_valid_values(entry, dtype, filled):
    rng = np.random.default_rng(20261025)
    values = np.r_[2**53 + 1, 1767225600000000123,
                   rng.integers(2**53, 2**62, size=12)].astype('i8')
    kind = np.dtype(dtype).kind
    fill = np.datetime64('NaT', 'ns') if kind == 'M' else np.timedelta64('NaT', 'ns') if kind == 'm' else 0
    kwargs = {'dtype': dtype, **({'na_value': fill} if filled else {})}

    def export(frame):
        if entry == 'frame':
            return frame.to_numpy(**kwargs)[:, 0]
        if entry == 'series':
            return frame['x'].to_numpy(**kwargs)
        return frame.exec('x', **kwargs)

    dense = DataFrame({'x': values})
    expected = values.astype(dtype)
    np.testing.assert_array_equal(export(dense), expected)
    for missing_row in range(1, len(values)):
        mask = np.arange(len(values)) == missing_row
        sparse = DataFrame({'x': np.ma.array(values, mask=mask)})
        if kind in 'iu' and not filled:
            with pytest.raises(ValueError, match='missing'):
                export(sparse)
        else:
            got = export(sparse)
            np.testing.assert_array_equal(got[~mask], expected[~mask])
            if kind in 'Mm':
                assert got.view('i8')[0] == values[0]


def test_explicit_export_missing_scan_is_lazy():
    # Structural allocation/scan budget: Rust's internal has_missing is not a
    # Python call and timing a fast scan would make this regression noisy.
    source = (Path(__file__).resolve().parents[1] /
              'crates/volas-python/src/convert.rs').read_text()
    owner = source.split('pub(crate) fn columns_to_numpy_with', 1)[1].split(
        '/// Whether `v` is a NumPy bool', 1)[0]
    assert 'let missing = col.has_missing();' not in owner
    assert 'LazyCell::new(|| col.has_missing())' in owner
