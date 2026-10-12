"""One freshness rule for every reader (`PyDataFrame::readable`): while an `append`
has left a cached directive column stale, a column read refreshes the columns it reads
(`df[d]`, `df[[...]]`, `get_column`, the constructor's `columns=`), and every other read
or frame operand raises — none returns, derives from or launders a stale placeholder,
and none sees a bounded-window frame's hidden margin."""

import numpy as np
import pytest

import volas

rng = np.random.default_rng(11)
N = 30
CLOSE = 100 + np.cumsum(rng.standard_normal(N + 1))
OPEN = CLOSE + rng.standard_normal(N + 1)


def frame(a, b):
    return volas.DataFrame({'open': OPEN[a:b], 'close': CLOSE[a:b]})


def stale():
    df = frame(0, N)
    df['ema:3']
    df.append(frame(N, N + 1))
    return df


def full(directive):
    return frame(0, N + 1).exec(directive)


STALE = 'stale computed'
T0 = np.datetime64('2026-01-01T00:00', 'ns')


def test_get_column_refreshes_a_stale_cached_column():
    df = stale()
    got = df.get_column('ema:3')
    assert np.array_equal(got.to_numpy(), full('ema:3'), equal_nan=True)
    with pytest.raises(KeyError):
        df.get_column('ma:5')  # never computes


def test_dlpack_of_a_column_read_is_fresh():
    df = stale()
    s = df.get_column('ema:3')
    assert np.array_equal(np.from_dlpack(s), full('ema:3'), equal_nan=True)


@pytest.mark.parametrize('read', [
    lambda df: df[1:5],
    lambda df: df[df['close'] > 0],
    lambda df: df[np.ones(N + 1, dtype=bool)],
    lambda df: df[[True] * (N + 1)],
    lambda df: df.isna(),
    lambda df: df.notna(),
    lambda df: df.rolling(2).mean(),
    lambda df: df.tz_localize('UTC'),
    lambda df: df.where(df == df),
    lambda df: volas.DataFrame(df, index=list(range(100, 100 + N + 1))),
    lambda df: volas.DataFrame(df, dtype='float32'),
])
def test_every_other_read_of_a_stale_frame_raises(read):
    df = stale()
    with pytest.raises(ValueError, match=STALE):
        read(df)


def test_a_stale_operand_raises():
    fresh = frame(0, N + 1)
    fresh['ema:3']
    with pytest.raises(ValueError, match=STALE):
        fresh == stale()  # noqa: B015
    s = stale()
    with pytest.raises(ValueError, match=STALE):
        fresh.where(s)
    with pytest.raises(ValueError, match=STALE):
        fresh[s] = 0.0


def test_the_constructor_projection_reads_like_a_column_projection():
    df = stale()
    sub = volas.DataFrame(df, columns=['close', 'ema:3'])
    assert np.array_equal(sub['ema:3'].to_numpy(), full('ema:3'), equal_nan=True)
    assert np.array_equal(sub['ema:3'].to_numpy(), df[['close', 'ema:3']]['ema:3'].to_numpy(),
                          equal_nan=True)


def test_windowed_derivations_see_the_visible_rows_only():
    w = volas.DataFrame({'close': CLOSE[:N]}, window=5, max_lookback=3,
                        index=np.datetime64('2026-01-01', 'ns') + np.arange(N) * np.timedelta64(1, 'm'))
    assert len(w.tz_localize('UTC')) == 5
    assert len(w.tz_localize('UTC').tz_convert('Asia/Shanghai')) == 5
    assert len(volas.DataFrame(w, columns=['close'])) == 5
    other = volas.DataFrame({'close': CLOSE[:N]}, window=5, max_lookback=3,
                            index=np.datetime64('2026-01-01', 'ns') + np.arange(N) * np.timedelta64(1, 'm'))
    assert (w == other).shape == (5, 1)


def test_a_stale_windowed_frame_raises_on_a_bulk_read():
    w = volas.DataFrame({'close': CLOSE[:N]}, window=1, max_lookback=['ma:10', 'ma:20'])
    w['ma:10']
    w.append(volas.DataFrame({'close': CLOSE[N:N + 1]}))
    with pytest.raises(ValueError, match=STALE):
        w.rolling(1).count()
    with pytest.raises(ValueError, match=STALE):
        w.to_numpy()


def test_a_cell_mask_pairs_columns_by_name():
    df = volas.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
    cond = volas.DataFrame({'b': [True, False], 'a': [False, False]})
    df[cond] = 0.0
    assert df.get_column('a').to_numpy().tolist() == [1.0, 2.0]
    assert df.get_column('b').to_numpy().tolist() == [0.0, 4.0]
    with pytest.raises(ValueError, match='by name'):
        df[volas.DataFrame({'a': [True, False], 'c': [False, False]})] = 0.0
    with pytest.raises(ValueError, match='different indexes'):
        df[volas.DataFrame({'a': [True, False], 'b': [False, False]}, index=[5, 6])] = 0.0


def test_a_series_value_is_aligned_by_the_index_rule():
    df = volas.DataFrame({'a': [1.0, 2.0, 3.0]}, index=[10, 20, 30])
    df['b'] = df['a'] * 2
    assert df.get_column('b').to_numpy().tolist() == [2.0, 4.0, 6.0]
    with pytest.raises(ValueError, match='different indexes'):
        df['c'] = volas.DataFrame({'x': [1.0, 2.0, 3.0]})['x']


def test_a_scalar_column_keeps_its_exact_type():
    df = volas.DataFrame({'a': [1.0, 2.0]})
    df['n'] = 2**53 + 1
    df['m'] = 5
    df['f'] = 2.5
    df['t'] = True
    df['s'] = 'x'
    df['i32'] = np.int32(7)
    df['f32'] = np.float32(0.5)
    df['b0'] = np.array(True)
    df['n0'] = np.array(2**60 + 1)
    assert df.dtypes == {'a': 'float64', 'n': 'int64', 'm': 'int64', 'f': 'float64',
                         't': 'bool', 's': 'str', 'i32': 'int32', 'f32': 'float32',
                         'b0': 'bool', 'n0': 'int64'}
    assert df.get_column('n').to_numpy().tolist() == [2**53 + 1] * 2
    assert df.get_column('n0').to_numpy().tolist() == [2**60 + 1] * 2


def test_a_one_element_array_is_an_array_not_a_scalar():
    one = volas.DataFrame({'a': [1.0]})
    one['m'] = np.array([2**60 + 1])
    assert one.dtypes['m'] == 'int64'
    assert one.get_column('m').to_numpy().tolist() == [2**60 + 1]
    two = volas.DataFrame({'a': [1.0, 2.0]})
    with pytest.raises(ValueError):
        two['m'] = np.array([7])  # one value for two rows: a length error, not a broadcast
    with pytest.raises(ValueError):
        two['m'] = [7]


def test_a_mask_write_is_in_place_and_keeps_the_cache():
    df = volas.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0]})
    df['ema:2']
    df.append(volas.DataFrame({'close': [6.0]}, index=[5]))
    for write in (lambda: df.__setitem__(df['close'] > 100, 0.0),
                  lambda: df.__setitem__(volas.DataFrame({'close': [False] * 6, 'ema:2': [False] * 6}), 0.0)):
        with pytest.raises(ValueError, match=STALE):
            write()
    assert df.is_computed('ema:2')
    df.fulfill()
    df[df['close'] > 100] = 0.0  # selects nothing: changes nothing
    assert df.is_computed('ema:2')
    assert np.array_equal(df['ema:2'].to_numpy(),
                          volas.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}).exec('ema:2'),
                          equal_nan=True)
    df[df['close'] > 5] = 0.0  # every column of the row, the cached one too: now data
    assert not df.is_computed('ema:2')
    assert df.get_column('ema:2').to_numpy()[-1] == 0.0
    assert df['ema:4'].to_numpy()[-1] == volas.DataFrame(
        {'close': [1.0, 2.0, 3.0, 4.0, 5.0, 0.0]}).exec('ema:4')[-1]


def test_a_list_projection_validates_every_entry_before_caching_any():
    df = frame(0, N)
    with pytest.raises(volas.DirectiveError):
        df[['ma:2', 'bad:2']]
    assert 'ma:2' not in df.columns


@pytest.mark.parametrize('target', ['datetime64[s]', 'float64'])
def test_astype_is_a_new_series_without_cached_columns(target):
    # one rule for every astype (R15): the cached columns are not carried
    df = volas.DataFrame({'close': [1.0, 2.0, 3.0, 4.0], 'ts': np.array([10, 20, 30, 40], 'int64')})
    df['ma:2@ts']
    df['ema:2']
    a = df.astype({'ts': target})
    assert a.columns == ['close', 'ts']
    a.fulfill()
    a.to_numpy()
    assert np.array_equal(a['ema:2'].to_numpy(), df['ema:2'].to_numpy(), equal_nan=True)


# --- labels on a windowed frame (REQUIREMENTS R26) -----------------------------------------


def _labelled(labels, values):
    return volas.DataFrame({'k': np.array(labels, 'int64'), 'close': values}).set_index('k')


def test_a_label_retained_in_the_hidden_margin_is_still_unique():
    w = volas.DataFrame(_labelled([10, 20, 30], [1.0, 2.0, 3.0]), window=2, max_lookback=1)
    w.append(_labelled([40], [4.0]))  # retained: 20 (margin), 30, 40; 10 is past retention
    assert w.index.tolist() == [30, 40]
    with pytest.raises(ValueError) as err:
        w.append(_labelled([20], [9.0]))
    assert str(err.value) == ("cannot append label 20: an earlier row with that label is "
                              "still retained in the window's hidden margin (max_lookback)")
    with pytest.raises(ValueError, match='an earlier row has that label'):
        w.append(_labelled([30], [9.0]))


def test_retention_is_the_last_window_plus_margin_rows_not_the_buffer():
    # the physical buffer still holds 10 (compaction has not run), but it is past the
    # retained window + max_lookback rows: accepted, whenever compaction happens
    w = volas.DataFrame(_labelled([10, 20, 30], [1.0, 2.0, 3.0]), window=2, max_lookback=1)
    w.append(_labelled([40], [4.0]))
    assert w._physical_height == 4
    w.append(_labelled([10], [9.0]))
    assert w.index.tolist() == [40, 10]


def test_unlabelled_rows_continue_unique_over_the_retained_rows_only():
    # 11 continues 10, and is held only by a row past retention (still in the buffer)
    w = volas.DataFrame(_labelled([11, 20, 30], [1.0, 2.0, 3.0]), window=2, max_lookback=1)
    w.append(_labelled([10], [4.0]))  # retained: 20 (margin), 30, 10
    assert w._physical_height == 4
    w.append(volas.DataFrame({'close': [5.0]}))
    assert w.index.tolist() == [10, 11]
    # a continued label a retained row still holds raises, frame unchanged
    w.append(_labelled([9], [6.0]))  # retained: 10 (margin), 11, 9
    with pytest.raises(ValueError, match='the next label 10 is already in the index'):
        w.append(volas.DataFrame({'close': [7.0]}))
    assert w.index.tolist() == [11, 9]


def test_a_label_no_longer_retained_is_accepted():
    w = volas.DataFrame(_labelled([10, 20, 30], [1.0, 2.0, 3.0]), window=2, max_lookback=1)
    for label in (40, 50, 60):
        w.append(_labelled([label], [float(label)]))
    assert w._physical_height <= 6
    w.append(_labelled([10], [99.0]))  # 10 left the retained rows
    assert w.index.tolist()[-1] == 10


def test_the_gate_says_what_left_the_frame_stale():
    df = stale()
    with pytest.raises(ValueError, match='after an append'):
        df.to_numpy()
    df.fulfill()
    df['close'] = df['close'] * 2  # a column write: every cache is recomputed
    with pytest.raises(ValueError, match='after a column write'):
        df.to_numpy()


def test_a_time_frame_built_from_a_stale_frame_reads_it_through_the_gate():
    df = volas.DataFrame({'t': T0 + np.arange(N) * np.timedelta64(1, 'm'), 'close': CLOSE[:N]})
    df = df.set_index('t')
    df['ema:3']
    df.append(volas.DataFrame({'t': [T0 + np.timedelta64(N, 'm')], 'close': [1.0]}).set_index('t'))
    with pytest.raises(ValueError, match=STALE):
        volas.DataFrame(df, time_frame='5m')
