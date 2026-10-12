"""One rule for what a datetime row label is, read and written back: an aware index
gives aware Timestamps (as `iloc[-1].name`); an aware value is an instant, a naive
Timestamp / datetime / string a wall clock (a naive / aware mix raises, contract #17),
and a raw epoch-ns count — an integer or a NumPy datetime64, the form instants are
exported in — is read in the frame's own zone."""

import datetime as dt

import numpy as np
import pytest

import volas

T = volas.to_datetime


def tokyo():
    return volas.DataFrame({'t': T(['2024-01-01', '2024-01-02']), 'close': [1.0, 2.0]}) \
        .set_index('t').tz_localize('Asia/Tokyo')


def test_an_aware_index_gives_aware_timestamps():
    f = tokyo()
    labels = f.index
    assert labels.dtype == object
    assert labels[-1] == f.iloc[-1].name
    assert repr(labels[-1]) == "Timestamp('2024-01-02 00:00:00', tz='Asia/Tokyo')"
    assert repr(f['close'].index[0]) == "Timestamp('2024-01-01 00:00:00', tz='Asia/Tokyo')"
    # a naive index stays datetime64
    naive = volas.DataFrame({'t': T(['2024-01-01']), 'close': [1.0]}).set_index('t')
    assert naive.index.dtype == np.dtype('datetime64[ns]')


def test_a_label_read_back_is_the_same_label():
    f = tokyo()
    f.append({'t': f.index[-1], 'close': 9.0})  # replaces the last row
    assert f['close'].to_numpy().tolist() == [1.0, 9.0]
    m = volas.DataFrame({'close': [False, True]}, index=f.index)
    assert m.tz == 'Asia/Tokyo'
    f[m] = 0.0  # aligns with the aware frame
    assert f['close'].to_numpy().tolist() == [1.0, 0.0]


@pytest.mark.parametrize('label', [
    np.datetime64('2024-01-05T00:00', 'ns'),
    int(np.datetime64('2024-01-05T00:00', 'ns').astype('int64')),
])
def test_a_raw_epoch_count_is_read_in_the_frame_zone(label):
    f = tokyo()
    f.append({'t': label, 'close': 3.0})
    assert repr(f.index[-1]) == "Timestamp('2024-01-05 09:00:00', tz='Asia/Tokyo')"
    naive = volas.DataFrame({'t': T(['2024-01-01']), 'close': [1.0]}).set_index('t')
    naive.append({'t': label, 'close': 3.0})
    assert naive.index[-1] == np.datetime64('2024-01-05T00:00', 'ns')


@pytest.mark.parametrize('label', [volas.Timestamp('2024-01-05'), '2024-01-05', dt.datetime(2024, 1, 5)])
def test_a_wall_clock_onto_an_aware_frame_raises(label):
    f = tokyo()
    with pytest.raises(ValueError, match='tz-naive DatetimeIndex to a frame with a tz-aware'):
        f.append({'t': label, 'close': 3.0})
    assert len(f) == 2


def test_index_labels_in_one_zone_only():
    f = tokyo()
    utc = volas.Timestamp('2024-01-03').tz_localize('UTC')
    with pytest.raises(ValueError, match=r'two zones \(Asia/Tokyo and UTC\)'):
        volas.DataFrame({'close': [1.0, 2.0]}, index=[f.index[0], utc])
    with pytest.raises(ValueError, match='tz-aware and tz-naive timestamps'):
        volas.DataFrame({'close': [1.0, 2.0]}, index=[f.index[0], volas.Timestamp('2024-01-03')])
    # integers and strings carry no zone of their own
    assert volas.DataFrame({'close': [1.0]}, index=[7]).index.tolist() == [7]
