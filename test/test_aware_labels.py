"""One rule for what a datetime row label is, read and written back. `df.index` is the
row labels as a NumPy array: an aware index exports its UTC instants as `datetime64[ns]`
(NumPy has no zone slot), while a row's `.name` / a cell is an aware `Timestamp`. A label
written follows contract #17 — an aware value is an instant, a naive `Timestamp` /
`datetime` / string a wall clock (a naive / aware mix raises) — and a raw epoch-ns count
(an integer or a NumPy `datetime64`, the form instants are exported in) is read in the
frame's own zone, so a label read back from `.index` is the same label."""

import datetime as dt

import numpy as np
import pytest

import volas

T = volas.to_datetime


def tokyo():
    return volas.DataFrame({'t': T(['2024-01-01', '2024-01-02']), 'close': [1.0, 2.0]}) \
        .set_index('t').tz_localize('Asia/Tokyo')


def test_an_aware_index_exports_utc_instants():
    f = tokyo()
    assert f.index.dtype == np.dtype('datetime64[ns]')
    assert f.index[-1] == np.datetime64('2024-01-01T15:00', 'ns')  # 00:00 Tokyo is 15:00Z
    assert repr(f.iloc[-1].name) == "Timestamp('2024-01-02 00:00:00', tz='Asia/Tokyo')"


def test_a_label_read_back_is_the_same_label():
    f = tokyo()
    f.append({'t': f.index[-1], 'close': 9.0})  # replaces the last row
    assert f['close'].to_numpy().tolist() == [1.0, 9.0]
    f.append({'t': f.iloc[-1].name, 'close': 8.0})
    assert f['close'].to_numpy().tolist() == [1.0, 8.0]


def test_a_mask_over_an_aware_index_restores_its_zone():
    f = tokyo()
    m = volas.DataFrame({'close': [False, True]}, index=f.index)  # naive UTC instants
    assert m.tz is None
    with pytest.raises(ValueError, match=r"tz_localize\('UTC'\)\.tz_convert\(zone\)"):
        f[m] = 0.0
    assert f['close'].to_numpy().tolist() == [1.0, 2.0]
    f[m.tz_localize('UTC').tz_convert('Asia/Tokyo')] = 0.0
    assert f['close'].to_numpy().tolist() == [1.0, 0.0]


@pytest.mark.parametrize('label', [
    np.datetime64('2024-01-05T00:00', 'ns'),
    int(np.datetime64('2024-01-05T00:00', 'ns').astype('int64')),
])
def test_a_raw_epoch_count_is_read_in_the_frame_zone(label):
    f = tokyo()
    f.append({'t': label, 'close': 3.0})
    assert repr(f.iloc[-1].name) == "Timestamp('2024-01-05 09:00:00', tz='Asia/Tokyo')"
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
    first = f.iloc[0].name
    utc = volas.Timestamp('2024-01-03').tz_localize('UTC')
    aware = volas.DataFrame({'close': [1.0, 2.0]}, index=[first, f.iloc[1].name])
    assert aware.tz == 'Asia/Tokyo'  # aware labels keep their zone
    with pytest.raises(ValueError, match=r'two zones \(Asia/Tokyo and UTC\)'):
        volas.DataFrame({'close': [1.0, 2.0]}, index=[first, utc])
    with pytest.raises(ValueError, match='tz-aware and tz-naive timestamps'):
        volas.DataFrame({'close': [1.0, 2.0]}, index=[first, volas.Timestamp('2024-01-03')])
    # integers and strings carry no zone of their own
    assert volas.DataFrame({'close': [1.0]}, index=[7]).index.tolist() == [7]
