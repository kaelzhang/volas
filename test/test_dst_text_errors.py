"""Text that names no instant: constructing one raises ValueError with the reason; looking
one up is a missing label.

A wall clock a named zone skips (a spring-forward DST gap) or repeats (a fall-back fold)
is not an instant. Building a timestamp from it (`Timestamp(text, tz=...)`) raises
ValueError naming the gap or fold and the zone, as pandas raises (its
NonExistentTimeError / AmbiguousTimeError are ValueErrors); text that is not a datetime
at all is a ValueError wherever an instant is built from it (a Timestamp, a datetime
cell or fill value). Looking such text up as a row label (`loc`) is a missing label, a
KeyError as in pandas, whose message carries the reason.
"""

from __future__ import annotations

import numpy as np
import pytest

import volas

NY = 'America/New_York'
GAP = '2024-03-10 02:30'  # 02:00 -> 03:00 EDT
FOLD = '2024-11-03 01:30'  # 02:00 EDT -> 01:00 EST


@pytest.mark.parametrize('text, reason', [(GAP, 'DST gap'), (FOLD, 'DST fold')])
def test_timestamp_from_a_dst_anomaly_is_a_value_error_naming_it(text, reason):
    with pytest.raises(ValueError, match=f'{reason}') as e:
        volas.Timestamp(text, tz=NY)
    assert not isinstance(e.value, KeyError)
    assert NY in str(e.value) and text in str(e.value)


def test_timestamp_from_non_datetime_text_is_a_value_error():
    with pytest.raises(ValueError, match='could not parse datetime') as e:
        volas.Timestamp('not a date')
    assert not isinstance(e.value, KeyError)


def _aware(start_utc, n=6):
    """Hourly bars from `start_utc`, indexed by `t` in New York."""
    t = [np.datetime64(start_utc, 'ns') + np.timedelta64(h, 'h') for h in range(n)]
    df = volas.DataFrame({'t': volas.to_datetime(np.array(t)), 'close': np.arange(n, dtype=float)})
    return df.set_index('t').tz_localize('UTC').tz_convert(NY)


def test_a_datetime_cell_from_non_datetime_text_is_a_value_error():
    df = volas.DataFrame({'t': volas.to_datetime(volas.DataFrame({'t': ['2024-01-02']})['t'])})
    with pytest.raises(ValueError, match='could not parse datetime') as e:
        df.iat[0, 0] = 'not a date'
    assert not isinstance(e.value, KeyError)


@pytest.mark.parametrize('text, reason', [(GAP, 'DST gap'), (FOLD, 'DST fold'), ('nonsense', 'could not parse')])
def test_looking_up_such_text_is_a_missing_label_with_the_reason(text, reason):
    df = _aware('2024-03-10T05:00') if text == GAP else _aware('2024-11-03T04:00')
    with pytest.raises(KeyError, match=reason):
        df.loc[text]
    with pytest.raises(KeyError, match=reason):
        df['close'].loc[text]


def test_a_value_of_no_timestamp_type_is_a_type_error_to_build_and_a_missing_label():
    with pytest.raises(TypeError, match='cannot convert list to a timestamp'):
        volas.Timestamp([1])
    df = _aware('2024-03-10T05:00')
    with pytest.raises(KeyError, match='cannot convert float to a timestamp'):
        df.loc[1.5]


@pytest.mark.parametrize('text, reason', [(GAP, 'DST gap'), ('nonsense', 'could not parse')])
def test_such_text_cannot_bound_a_datetime_slice(text, reason):
    # pandas 3.0.3: "cannot do slice indexing on DatetimeIndex with these indexers"
    df = _aware('2024-03-10T05:00')
    with pytest.raises(TypeError, match=reason):
        df.loc[text:]
    with pytest.raises(TypeError, match=reason):
        df.loc[:text]


def test_a_datetime_fill_keeps_the_reason():
    s = volas.to_datetime(volas.DataFrame({'t': ['2024-01-02', None]})['t'])
    with pytest.raises(ValueError, match='could not parse datetime "nonsense"'):
        s.fillna('nonsense')
