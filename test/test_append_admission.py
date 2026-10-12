"""Rows entering a frame: every append form (DataFrame / Row / dict, plain and
time-frame frames, new rows, same-label replacements, forming-bar folds and rollovers)
is checked whole before the first write, so a rejected append leaves the frame exactly
as it was, and a later valid append continues as if the rejected one never happened."""

import numpy as np
import pytest

import volas

T = volas.to_datetime


def bars(times, **cols):
    return volas.DataFrame({'t': T(times), **cols}).set_index('t')


def state(df):
    """Everything a rejected append must leave as it was."""
    view = {c: df.get_column(c).to_numpy().tolist() for c in df.columns if not df.is_computed(c)}
    return (
        list(df.columns), view, df.index.tolist(), str(df.tz), len(df),
        sorted(c for c in df.columns if df.is_computed(c)),
    )


def same(a, b):
    return np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True)


# --- plain frames ---------------------------------------------------------------


def test_a_later_column_that_cannot_take_its_value_grows_no_column():
    df = volas.DataFrame({'open': [1.0, 2.0], 'close': [1.5, 2.5], 'tag': ['a', 'b']})
    before = state(df)
    with pytest.raises(TypeError, match='column "tag"'):
        df.append(volas.DataFrame({'open': [3.0], 'close': [3.5], 'tag': [7]}))
    assert state(df) == before
    df.append(volas.DataFrame({'open': [3.0], 'close': [3.5], 'tag': ['c']}))
    assert df.get_column('open').to_numpy().tolist() == [1.0, 2.0, 3.0]


def test_appended_values_convert_losslessly_to_the_frame_dtype():
    df = volas.DataFrame({'v': np.array([1, 2], 'int64'), 'x': [1.0, 2.0]})
    df.append(volas.DataFrame({'v': [3.0], 'x': np.array([4], 'int64')}))
    assert df.dtypes == {'v': 'int64', 'x': 'float64'}
    assert df.get_column('v').to_numpy().tolist() == [1, 2, 3]
    with pytest.raises(TypeError, match='2.5 is not exactly representable'):
        df.append(volas.DataFrame({'v': [2.5], 'x': [5.0]}))
    assert len(df) == 3


@pytest.mark.parametrize('col, value', [('v', 2.5), ('flag', 0.5), ('name', 1.5)])
def test_a_replacing_row_converts_by_the_same_lossless_rule(col, value):
    df = bars(['2024-01-01', '2024-01-02', '2024-01-03'], v=np.array([10, 20, 30], 'int64'),
              flag=[True, False, True], name=['a', 'b', 'c'])
    before = state(df)
    with pytest.raises(TypeError):
        df.append(bars(['2024-01-03'], **{col: [value]}))
    assert state(df) == before
    # the same row under a new label is judged by the same rule
    with pytest.raises(TypeError):
        df.append(bars(['2024-01-04'], **{col: [value]}))
    assert state(df) == before


def test_a_rejected_replacement_writes_no_column_and_leaves_the_cache_exact():
    df = bars(['2024-01-01', '2024-01-02', '2024-01-03'], close=[1.0, 2.0, 3.0],
              when=T(['2024-01-01', '2024-01-02', '2024-01-03']))
    df['ma:2']
    before = state(df)
    with pytest.raises(TypeError):
        df.append(bars(['2024-01-03'], close=[100.0], when=np.array([5], 'int64')))
    assert state(df) == before
    assert same(df['ma:2'], df.exec('ma:2'))


def test_a_rejected_batch_writes_no_row():
    df = volas.DataFrame({'s': ['a', 'b'], 'close': [1.0, 2.0]}, index=[1, 2])
    before = state(df)
    with pytest.raises(TypeError):
        df.append(volas.DataFrame({'s': [7, 8], 'close': [20.0, 30.0]}, index=[2, 3]))
    assert state(df) == before


def test_an_index_of_another_kind_grows_no_column():
    df = bars(['2024-01-01', '2024-01-02'], close=[1.0, 2.0])
    before = state(df)
    for other in (
        volas.DataFrame({'close': [3.0]}, index=['x']),
        volas.DataFrame({'close': [3.0]}),
        volas.DataFrame({'close': [3.0]}, index=[5]),
        volas.DataFrame({'close': []}),
    ):
        with pytest.raises(ValueError):
            df.append(other)
        assert state(df) == before
    df.append(bars(['2024-01-03'], close=[4.0]))
    assert df.get_column('close').to_numpy().tolist() == [1.0, 2.0, 4.0]


def test_an_empty_positional_frame_takes_no_timestamps():
    df = volas.DataFrame({'close': []})
    with pytest.raises(ValueError, match='RangeIndex'):
        df.append(bars(['2024-01-01'], close=[1.0]))
    assert len(df) == 0


def test_an_empty_range_append_keeps_the_zone():
    df = bars(['2024-01-01', '2024-01-02'], close=[1.0, 2.0]).tz_localize('Asia/Shanghai')
    with pytest.raises(ValueError):
        df.append(volas.DataFrame({'close': []}))
    assert df.tz == 'Asia/Shanghai'
    assert df.loc['2024-01-02', 'close'] == 2.0


def test_the_replacement_path_checks_the_index_kind_and_awareness():
    ints = volas.DataFrame({'close': [1.0, 2.0, 3.0]}, index=[1, 2, 3])
    with pytest.raises(ValueError):
        ints.append(volas.DataFrame({'close': [99.0]}, index=np.array([3], 'datetime64[ns]')))
    assert ints.get_column('close').to_numpy().tolist() == [1.0, 2.0, 3.0]
    naive = bars(['2024-01-01', '2024-01-02'], close=[1.0, 2.0])
    aware = bars(['2024-01-02'], close=[77.0]).tz_localize('UTC')
    with pytest.raises(ValueError, match='tz-aware'):
        naive.append(aware)
    assert naive.get_column('close').to_numpy().tolist() == [1.0, 2.0]


def test_unlabelled_rows_never_repeat_or_overflow_an_integer_label():
    df = volas.DataFrame({'x': [10, 20]}, index=[1, 0])
    with pytest.raises(ValueError, match='already in the index'):
        df.append(volas.DataFrame({'x': [30]}))
    assert df.index.tolist() == [1, 0]
    top = volas.DataFrame({'x': [10]}, index=[2**63 - 1])
    with pytest.raises(OverflowError):
        top.append(volas.DataFrame({'x': [30]}))
    assert len(top) == 1


# --- cached directive columns are never appended -----------------------------------------

R20 = (
    "cannot append columns ['ema:2']: they are cached directives that this DataFrame "
    "computes itself. Select the data columns first, e.g. "
    "df.append(other[['open', 'high', 'low', 'close', 'volume']])."
)


def _ohlcv(times, k=0.0):
    n = len(times)
    return bars(times, open=[1.0 + k] * n, high=[2.0 + k] * n, low=[0.5 + k] * n,
                close=[1.5 + k] * n, volume=[10.0] * n)


@pytest.mark.parametrize('tf', [None, '1d'])
def test_every_append_form_rejects_cached_columns(tf):
    times = ['2024-01-01', '2024-01-02', '2024-01-03']
    df = _ohlcv(times) if tf is None else volas.DataFrame(_ohlcv(times), time_frame=tf)
    df['ema:2']
    other = _ohlcv(['2024-01-04'], 1.0)
    other['ema:2']
    before = state(df)
    forms = [other, other.iloc[-1],
             {'t': volas.Timestamp('2024-01-04'), 'open': 2.0, 'high': 3.0, 'low': 1.5,
              'close': 2.5, 'volume': 10.0, 'ema:2': 9.0}]
    for form in forms:
        with pytest.raises(ValueError) as err:
            df.append(form)
        assert str(err.value) == R20
        assert state(df) == before
    df.append(other[['open', 'high', 'low', 'close', 'volume']])
    assert same(df['ema:2'], df.exec('ema:2'))


# --- time-frame frames ------------------------------------------------------------------


def _tf(volume_dtype='int64'):
    seed = bars(['2024-01-01 00:00', '2024-01-01 01:00'], high=[10.0, 11.0], close=[10.0, 11.0],
                volume=np.array([1, 1], volume_dtype))
    return volas.DataFrame(seed, time_frame='1h',
                           cumulators={'high': 'max', 'close': 'last', 'volume': 'sum'})


def _bar(t, high, close, volume):
    return {'t': volas.Timestamp(t), 'high': high, 'close': close, 'volume': volume}


def test_a_fold_that_overflows_changes_nothing():
    df = _tf()
    df.append(_bar('2024-01-01 02:00', 12.0, 12.0, 5 * 10**18))
    df['ma:2@close']
    before = state(df)
    with pytest.raises(OverflowError):
        df.append(_bar('2024-01-01 02:30', 50.0, 50.0, 5 * 10**18))
    assert state(df) == before
    assert same(df['ma:2@close'], df.exec('ma:2@close'))
    df.append(_bar('2024-01-01 02:45', 13.0, 13.0, 5))
    df.fulfill()
    assert df.iloc[-1]['high'] == 13.0
    assert df.iloc[-1]['volume'] == 5 * 10**18 + 5
    assert same(df['ma:2@close'], df.exec('ma:2@close'))


def test_a_resent_bar_that_overflows_changes_nothing():
    df = _tf()
    df.append(_bar('2024-01-01 02:00', 12.0, 12.0, 5 * 10**18))
    df.append(_bar('2024-01-01 02:10', 13.0, 13.0, 1))
    before = state(df)
    with pytest.raises(OverflowError):
        df.append(_bar('2024-01-01 02:10', 60.0, 60.0, 5 * 10**18))  # re-sent, overflows
    assert state(df) == before
    df.append(_bar('2024-01-01 02:10', 14.0, 14.0, 2))  # a valid re-send refolds the period
    assert (df.iloc[-1]['high'], df.iloc[-1]['volume']) == (14.0, 5 * 10**18 + 2)


def test_a_rejected_resend_leaves_the_open_period_as_it_was():
    seed = bars(['2024-01-01 00:00', '2024-01-01 00:05'], open=[1.0, 2.0], high=[1.5, 2.5],
                low=[0.5, 1.5], close=[1.2, 2.2], volume=np.array([10, 20], 'int64'))

    def fine(t, high, close, volume):
        return bars([t], open=[2.2], high=[high], low=[2.1], close=[close], volume=[volume])

    live = volas.DataFrame(seed, time_frame='5m')
    live['ma:2@high']
    live.append(fine('2024-01-01 00:06', 2.6, 2.3, 5))
    before = state(live)
    with pytest.raises(TypeError):
        live.append(fine('2024-01-01 00:06', 99.0, 2.3, 5.75))  # lossy volume
    assert state(live) == before
    live.append(fine('2024-01-01 00:07', 2.7, 2.4, 6))
    live.append(fine('2024-01-01 00:07', 2.8, 2.5, 7))  # a valid re-send refolds the open bars
    ref = volas.DataFrame(seed, time_frame='5m')
    ref.append(fine('2024-01-01 00:06', 2.6, 2.3, 5))
    ref.append(fine('2024-01-01 00:07', 2.8, 2.5, 7))
    assert state(live)[1] == state(ref)[1]
    assert same(live['ma:2@high'], ref['ma:2@high'])


def test_a_rejected_rollover_keeps_the_closed_row_closed():
    df = _tf('float64')
    with pytest.raises(TypeError):
        df.append(bars(['2024-01-01 02:00'], high=[20.0], close=[20.0], volume=['x']))
    for t, v in [('2024-01-01 02:10', 30.0), ('2024-01-01 02:20', 40.0)]:
        df.append(bars([t], high=[v], close=[v], volume=np.array([7], 'int64')))
    assert df.index.tolist() == T(['2024-01-01 00:00', '2024-01-01 01:00',
                                   '2024-01-01 02:00']).to_numpy().tolist()
    assert df.iloc[1]['high'] == 11.0 and df.iloc[1]['volume'] == 1.0
    assert (df.iloc[2]['high'], df.iloc[2]['volume']) == (40.0, 14.0)


def test_a_multi_bar_fold_is_all_or_nothing():
    df = _tf()
    df['ma:2@close']
    before = state(df)
    batch = bars(['2024-01-01 02:00', '2024-01-01 02:10', '2024-01-01 03:00', '2024-01-01 03:10'],
                 high=[12.0, 13.0, 14.0, 15.0], close=[12.0, 13.0, 14.0, 15.0],
                 volume=np.array([1, 2, 5 * 10**18, 5 * 10**18], 'int64'))
    with pytest.raises(OverflowError):
        df.append(batch)
    assert state(df) == before
    df.append(batch.iloc[:3])
    assert len(df) == 4
    assert same(df['ma:2@close'], df.exec('ma:2@close'))


def test_a_time_frame_bar_must_match_the_frame_awareness():
    df = _tf()
    before = state(df)
    with pytest.raises(ValueError, match='tz-aware'):
        df.append(bars(['2024-01-01 02:00'], high=[1.0], close=[1.0],
                       volume=np.array([1], 'int64')).tz_localize('UTC'))
    assert state(df) == before
