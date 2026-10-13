"""Compiled batches retain scalar format semantics and validation timing."""
import numpy as np
import pytest

import volas


@pytest.mark.parametrize('fmt', ['%', '%Q', '%#z', '%F %Z'])
def test_empty_and_missing_batches_defer_format_errors(fmt):
    for values in [[], [''], [' ', None], [None]]:
        strings = volas.DataFrame({'t': np.array(values, dtype=object) if values else np.array([], dtype=str)})['t']
        # Use a typed string column even for all-null input.
        strings = strings.astype('str')
        got = volas.to_datetime(strings, format=fmt)
        assert got.isna().to_list() == [True] * len(values)
        assert got.dt.strftime(fmt).isna().to_list() == [True] * len(values)
    with pytest.raises(ValueError):
        volas.to_datetime(['', '2024-01-01', 'bad'], format=fmt)
    with pytest.raises(ValueError):
        volas.to_datetime(['', '2024-01-01']).dt.strftime(fmt if fmt != '%F %Z' else '%Q')


@pytest.mark.parametrize('fmt', [
    '%Y|%C|%y|%G|%g|%m|%d|%e|%H|%k|%I|%l|%M|%S|%j|%q|%u|%w|%U|%V|%W',
    '%D|%x|%F|%v|%R|%T|%X|%r|%c',
    '%-f|%_f|%0f|%f|%.f|%3f|%6f|%9f|%.3f|%.6f|%.9f',
    '%a|%A|%b|%B|%h|%p|%P|%s|%z|%:z|%::z|%:::z|%Z|%+',
    '日期 %Y%n%t%%',
])
def test_compiled_format_batch_equals_scalar_at_range_and_fraction_edges(fmt):
    values = np.array([np.iinfo(np.int64).min + 1, -1, 0, 1, 123_000_000,
                       123_456_000, 123_456_789, np.iinfo(np.int64).max], dtype=np.int64)
    series = volas.to_datetime(values)
    want = [volas.Timestamp(int(ns), tz='UTC').strftime(fmt) for ns in values]
    assert series.dt.strftime(fmt).to_list() == want


def test_compiled_parse_preserves_first_value_error_and_redundant_fields():
    with pytest.raises(ValueError, match='"bad-first" does not match'):
        volas.to_datetime(['', 'bad-first', 'bad-second'], format='%F %T')
    with pytest.raises(ValueError, match='2024-01-07 2025'):
        volas.to_datetime(['2024-01-07 2024', '2024-01-07 2025'], format='%F %Y')
    expected = volas.to_datetime(['2024-01-07T03:04:05.123456789+05:21:10']).to_numpy()
    got = volas.to_datetime(['2024-01-07 03:04:05.123456789+05:21:10'], format='%F %T%.9f%::z')
    np.testing.assert_array_equal(got.to_numpy(), expected)
