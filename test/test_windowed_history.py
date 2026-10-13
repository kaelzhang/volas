"""A bounded window keeps the history of what it computes (docs "Bounded rolling
window"): every declared directive and every directive it admits later equals the
unbounded history bit for bit — through appends, re-sent last rows, missing runs,
writes and the periodic drop of its head — and anything it cannot compute exactly
raises, naming the correct paths."""

import re

import numpy as np
import pytest

import volas

MIN = np.timedelta64(1, 'm')
T0 = np.datetime64('2024-01-01T00:00', 'ns')


def F(xs, ids=None, ohlcv=False):
    data = {'close': [float(x) for x in xs]}
    if ohlcv:
        data.update(high=[2.0] * len(xs), low=[0.0] * len(xs), volume=[1.0] * len(xs))
    if ids is None:
        return volas.DataFrame(data)
    return volas.DataFrame(data, index=(T0 + np.array(list(ids)) * MIN).astype('datetime64[ns]'))


def pair(xs, d, tf=None, ohlcv=False, window=1):
    """A windowed frame declaring ``d`` and its unbounded twin, fed the same rows."""
    kw = {} if tf is None else {'time_frame': tf}
    seed = F(xs, range(len(xs)), ohlcv)
    w = volas.DataFrame(seed, window=window, max_lookback=[d], **kw)
    u = volas.DataFrame(seed, **kw)
    u[d]
    return w, u


def feed(w, u, xs, ids, d, ohlcv=False):
    bar = F(xs, ids, ohlcv)
    w.append(bar)
    u.append(bar)
    return w[d].to_list(), u[d].to_list()[-len(w):]


def same(a, b):
    return np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True)


def bars(n, seed=0, gaps=0.0):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    cols = {'open': c + rng.normal(0, .5, n), 'high': c + 1 + rng.random(n), 'low': c - 1 - rng.random(n),
            'close': c.copy(), 'volume': rng.uniform(1, 1000, n)}
    for k in cols:
        cols[k][rng.random(n) < gaps] = np.nan
    return cols


def rows(cols, a, b):
    return volas.DataFrame({k: v[a:b] for k, v in cols.items()},
                           index=(T0 + np.arange(a, b) * MIN).astype('datetime64[ns]'))


# --- RC-28: a multi-row refresh leaves both states --------------------------------------


@pytest.mark.parametrize('tf', [None, '1m'])
def test_a_multi_row_refresh_keeps_the_state_before_the_last_row(tf):
    # two rows refreshed at once, then the last re-sent: the replacement resumes from
    # the state before it (the windowed frame dropped the rows a recompute would need)
    d = 'adosc:2,3'
    xs = [0., 2., 0., 2., 0., 2.]
    w, u = pair(xs[:3], d, tf, True)
    for i in [3, 4, 5]:
        feed(w, u, [xs[i]], [i], d, True)
    a, b = feed(w, u, [0., 2.], [6, 7], d, True)
    assert same(a, b)
    a, b = feed(w, u, [1.], [7], d, True)
    assert same(a, b)


# --- RC-29: the checkpoint of every declared recursion -----------------------------------


def test_a_missing_run_across_a_drop_resumes_from_the_checkpoint():
    # RSI reads the last complete close before a new row: a run of missing rows lets
    # the drop of the head take it, and the checkpoint supplies it
    d = 'rsi:2'
    w, u = pair([1., 3., 2.], d)
    for i, x in enumerate([float('nan')] * 6 + [4., 3., 5.], start=3):
        a, b = feed(w, u, [x], [i], d)
        assert same(a, b), i


_DECLARED = ['ema:5', 'kama:5', 'macd.signal:5,8,3', 'kdj.j', 'rsi:6', 'stochrsi.d', 'trix:4', 'obv', 'ad',
             'adosc:3,6', 'mass_index', 'tsi:5,3', 'keltner.upper', 'wad', 'asi', 'supertrend', 'natr:5',
             'adxr:5', 'sar', 'sarext', 'ht_trendline', 'ht_sine', 'mama.fama', 'rsi:5@open',
             'ema:4@(high + low)', 'rsi:4@(ma:3)', 'ma:5', 'boll.upper:5', 'cci:5', 'maxindex:4']


@pytest.mark.parametrize('tf', [None, '1m'])
@pytest.mark.parametrize('window', [1, 3])
def test_declared_directives_equal_the_unbounded_history(tf, window):
    # appends, re-sent last rows, several rows between reads, missing values, and the
    # head dropped many times; the seed is longer than twice the retention
    rng = np.random.default_rng(window)
    cols = bars(700, seed=window, gaps=0.04)
    kw = {} if tf is None else {'time_frame': tf}
    seed = rows(cols, 0, 150)
    w = volas.DataFrame(seed, window=window, max_lookback=_DECLARED, **kw)
    u = volas.DataFrame(seed, **kw)
    names = [u[d].name for d in _DECLARED]
    assert all(w.is_computed(n) for n in names)  # declared: cached at construction
    i = 150
    while i < 700:
        k = int(rng.integers(1, 4))
        for j in range(i, min(i + k, 700)):
            bar = rows(cols, j, j + 1)
            w.append(bar)
            u.append(bar)
            if rng.random() < 0.3:  # the row re-sent with other values
                again = rows(bars(j + 1, seed=j), j, j + 1)
                w.append(again)
                u.append(again)
        i += k
        for n in names:
            got = w[n].to_numpy(dtype='float64')
            assert same(got, u[n].to_numpy(dtype='float64')[-len(got):]), (n, i)
    assert w._physical_height < 2 * (max(window, 2) + volas.directive_lookback('ht_trendline'))


def test_a_write_replays_each_declared_recursion_from_its_checkpoint():
    cols = bars(300, seed=7, gaps=0.05)
    w = volas.DataFrame(rows(cols, 0, 20), window=4, max_lookback=['ema:6', 'atr:5', 'ma:4'])
    u = volas.DataFrame(rows(cols, 0, 20))
    for d in ['ema:6', 'atr:5', 'ma:4']:
        u[d]
    for i in range(20, 300):
        w.append(rows(cols, i, i + 1))
        u.append(rows(cols, i, i + 1))
        if i % 17 == 0:
            w.fulfill()
            u.fulfill()
            w.iat[1, 3] = 1.0 + i
            u.iat[len(u) - 3, 3] = 1.0 + i
        for d in ['ema:6', 'atr:5', 'ma:4']:
            assert same(w[d], u[d].to_numpy()[-4:]), (d, i)


@pytest.mark.parametrize('d', ['psy', 'psy:5', 'minindex:4', 'maxindex:4'])
def test_a_window_directive_reads_its_lookback_rows_alone(d):
    # psy's first window starts at the first difference, and a tied extreme reports its
    # latest position: each value reads its lookback's rows only, so a write that makes
    # the frame recompute from the rows it keeps gives the unbounded history's values
    cols = bars(260, seed=13)
    cols['close'] = np.round(cols['close'])  # ties
    w = volas.DataFrame(rows(cols, 0, 30), window=3, max_lookback=[d])
    u = volas.DataFrame(rows(cols, 0, 30))
    u[d]
    for i in range(30, 260):
        w.append(rows(cols, i, i + 1))
        u.append(rows(cols, i, i + 1))
        if i % 5 == 0:
            w.fulfill()
            u.fulfill()
            w.iat[0, 0] = float(i)  # open: no input of d, every cache recomputes
            u.iat[len(u) - 3, 0] = float(i)
        assert same(w[d], u[d].to_numpy()[-3:]), i


def test_declared_directives_are_computed_over_the_whole_seed():
    cols = bars(500, seed=3)
    w = volas.DataFrame(rows(cols, 0, 500), window=5, max_lookback=['ema:20', 'ma:10'])
    u = rows(cols, 0, 500)
    assert {'ema:20', 'ma:10'} <= set(w.columns)
    assert w._physical_height < 500  # the head beyond twice the retention is dropped
    assert same(w['ema:20'], u['ema:20'].to_numpy()[-5:])
    assert same(w['ma:10'], u['ma:10'].to_numpy()[-5:])


def test_a_recursion_requested_late_raises_naming_the_two_paths():
    w = volas.DataFrame(rows(bars(40), 0, 40), window=5, max_lookback=['ma:20'])
    for read in (lambda: w['ema:10'], lambda: w[['close', 'ema:10']], lambda: w.exec('ema:10')):
        with pytest.raises(ValueError, match=r"cannot compute 'ema:10' on this windowed frame: each of its "
                                             r"values depends on every row before it.*max_lookback=\[\.\.\., "
                                             r"'ema:10'\].*or compute it on an unbounded frame"):
            read()
    assert 'ema:10' not in w.columns
    # an int margin declares nothing: every recursion is late
    w = volas.DataFrame(rows(bars(10), 0, 10), window=3, max_lookback=30)
    with pytest.raises(ValueError, match='cannot compute'):
        w['obv']


def test_a_window_directive_past_the_margin_raises():
    w = volas.DataFrame(rows(bars(40), 0, 40), window=5, max_lookback=['ma:10'])
    for read in (lambda: w['mom:10'], lambda: w.exec('mom:10')):
        with pytest.raises(ValueError, match=r"'mom:10' .* reads 10 rows before each row, more than the 9 "
                                             r"it keeps \(max_lookback\)"):
            read()


def test_a_class_1_directive_within_the_margin_is_computed_late_exactly():
    # a class-1 directive (a trailing window, no running accumulator) is computed on request
    cols = bars(400, seed=11)
    w = volas.DataFrame(rows(cols, 0, 30), window=4, max_lookback=['ma:12'])
    u = volas.DataFrame(rows(cols, 0, 30))
    for i in range(30, 200):
        w.append(rows(cols, i, i + 1))
        u.append(rows(cols, i, i + 1))
    late = ['maxindex:8', 'minindex:5', 'llv:6', 'hhv:9', 'roc:7', 'mom:3', 'psy:6', 'tr', 'bop',
            'willr:9', 'close > open']
    for d in late:
        assert same(w.exec(d, dtype='float64'), u.exec(d, dtype='float64')[-4:]), d
        assert same(w[d].to_numpy(dtype='float64'), u[d].to_numpy(dtype='float64')[-4:]), d
    for i in range(200, 400):
        w.append(rows(cols, i, i + 1))
        u.append(rows(cols, i, i + 1))
        for d in late:
            assert same(w[d].to_numpy(dtype='float64'), u[d].to_numpy(dtype='float64')[-4:]), (d, i)


@pytest.mark.parametrize('d, why', [
    ('ma:5', 'it carries a running accumulator \\(a class-2 directive\\)'),
    ('close > boll.upper:5', 'it carries a running accumulator \\(a class-2 directive\\)'),
    ('wma:4', 'it carries a running accumulator \\(a class-2 directive\\)'),
    ('increase:3', 'it counts a run of rows \\(a class-3 directive\\)'),
])
def test_class_2_and_3_must_be_declared(d, why):
    # owner rule R13: only class 1 is computed on request; classes 2 and 3 are declared
    w = volas.DataFrame(rows(bars(40), 0, 40), window=5, max_lookback=['ma:20'])
    for read in (lambda: w[d], lambda: w.exec(d)):
        with pytest.raises(ValueError, match=f"cannot compute '{re.escape(d)}' on this windowed frame: {why}. "
                                             "Declare it when constructing"):
            read()
    declared = volas.DataFrame(rows(bars(40), 0, 40), window=5, max_lookback=['ma:20', d])
    assert same(declared[d].to_numpy(dtype='float64'), rows(bars(40), 0, 40)[d].to_numpy(dtype='float64')[-5:])
    # exec of a declared directive points to its column
    with pytest.raises(ValueError, match=r"It is declared — read its column df\['"):
        declared.exec(d)


@pytest.mark.parametrize('d', ['ema:3@(rsi:5)', 'ma:3@(ema:2)', 'ema:3 - ema:5', 'close > ema:5',
                               'crsi', 'apo:5,9,1', 'stoch.k:5,3,1'])
def test_a_recursion_no_state_carries_is_refused(d):
    seed = rows(bars(10), 0, 10)
    with pytest.raises(ValueError, match=f"cannot keep '{re.escape(d)}' exact on a windowed frame"):
        volas.DataFrame(seed, window=3, max_lookback=[d])
    w = volas.DataFrame(seed, window=3, max_lookback=60)
    for read in (lambda: w[d], lambda: w.exec(d)):
        with pytest.raises(ValueError, match="cannot keep"):
            read()


@pytest.mark.parametrize('d', ['kcw', 'kcw:7,4,1.5', 'ma:5,1', 'ma:4,3', 'ma:4,4', 'ma:4,6', 'ma:4,7', 'ma:4,8'])
def test_a_recursion_carrying_its_own_state_is_kept_by_the_window(d):
    # kcw continues the Keltner band's EMA + ATR pair; `ma` at a recursive MA type runs
    # as that average: both carry a state, so a window keeps them exact once declared
    cols = bars(300, seed=5)
    w = volas.DataFrame(rows(cols, 0, 60), window=4, max_lookback=[d])
    u = volas.DataFrame(rows(cols, 0, 60))
    u[d]
    for i in range(60, 300):
        w.append(rows(cols, i, i + 1))
        u.append(rows(cols, i, i + 1))
        if i % 37 == 0:
            assert same(w[d].to_numpy(dtype='float64'), u[d].to_numpy(dtype='float64')[-4:]), (d, i)
    assert w._physical_height < 300
    assert same(w[d].to_numpy(dtype='float64'), u[d].to_numpy(dtype='float64')[-4:]), d


@pytest.mark.parametrize('d', ['obv', 'ad', 'adosc', 'mfi:14', 'vwma:20', 'ema:20@(volume)'])
def test_an_int64_volume_window_equals_the_unbounded_history(d):
    # an int64 input converted over the rows a continuation reads, across compactions:
    # the declared column equals the unbounded frame's bit for bit
    rng = np.random.default_rng(1)
    n = 2600
    c = 100 + np.cumsum(rng.standard_normal(n))
    o = c + 0.3 * rng.standard_normal(n)
    data = {'open': o, 'high': np.maximum(o, c) + .5, 'low': np.minimum(o, c) - .5, 'close': c,
            'volume': rng.integers(10, 1000, n).astype(np.int64)}
    frame = lambda lo, hi: volas.DataFrame({k: v[lo:hi] for k, v in data.items()})
    w = volas.DataFrame({k: v[:2000] for k, v in data.items()}, window=50, max_lookback=[d])
    u = frame(0, 2000)
    u[d]
    for i in range(2000, n):
        w.append(frame(i, i + 1))
        u.append(frame(i, i + 1))
    assert same(w[d].to_numpy(dtype='float64'), u[d].to_numpy(dtype='float64')[-50:])


def test_a_copied_cached_column_must_be_kept_by_the_window():
    u = rows(bars(30), 0, 30)
    u['ema:3 - ema:5']
    with pytest.raises(ValueError, match="cannot keep 'ema:3-ema:5' exact"):
        volas.DataFrame(u, window=3, max_lookback=10)
    v = rows(bars(30), 0, 30)
    v['ma:20']
    with pytest.raises(ValueError, match=r"cannot keep 'ma:20' on a windowed frame with max_lookback=5: it "
                                         r"reads 19 rows"):
        volas.DataFrame(v, window=3, max_lookback=5)
    v['ema:4']
    w = volas.DataFrame(v, window=3, max_lookback=['ma:20'])  # carried over, kept exactly
    for i in range(30, 120):
        bar = rows(bars(120), i, i + 1)
        w.append(bar)
        v.append(bar)
    assert same(w['ema:4'], v['ema:4'].to_numpy()[-3:])


def test_a_slice_cannot_recompute_a_recursion_after_a_write():
    # a slice continues a recursion from its carried state; a write discards it, and the
    # slice does not hold the rows the history started on
    df = rows(bars(40), 0, 40)
    df['ema:5']
    s = df.iloc[-10:]
    s.iat[0, 3] = 1.0
    with pytest.raises(ValueError, match="cannot recompute 'ema:5' exactly"):
        s.fulfill()


@pytest.mark.parametrize('declared', ['macd:', 'MACD:12,26'])
def test_a_declaration_named_like_a_data_column_raises(declared):
    # the declared column would be cached under 'macd', a data column: the declaration
    # cannot be kept, so construction refuses it rather than drop it
    data = {'close': [1., 2.], 'macd': [99., 99.]}
    with pytest.raises(ValueError) as raised:
        volas.DataFrame(data, window=2, max_lookback=[declared])
    assert str(raised.value) == (
        f"cannot declare '{declared}': its column name 'macd' is a data column of this DataFrame. "
        "Rename the data column — df.rename(columns={'macd': 'macd_data'}) — or declare another "
        "directive.")


def test_an_empty_declaration_raises():
    with pytest.raises(ValueError, match=r'max_lookback=\[\] declares no directive'):
        volas.DataFrame(F([1.0]), window=1, max_lookback=[])


# --- RC-30: retention covers a replaced row's resume ------------------------------------


@pytest.mark.parametrize('tf', [None, '1m'])
@pytest.mark.parametrize('d', ['ma:3', 'boll.upper:3', 'wma:3', 'linearreg:3', 'tsf:3'])
def test_a_window_of_one_keeps_a_replaced_rows_reach(tf, d):
    # a margin of 2 behind a window of 1 retains 2 + 2 rows after a drop: the replaced
    # last row resumes from the row before it, which reads 3 rows behind
    w, u = pair([1., 2., 3.], d, tf)
    drops = 0
    for i in range(3, 12):
        before = w._physical_height
        assert same(*feed(w, u, [float(i + 1)], [i], d))
        if w._physical_height < before:
            drops += 1
            assert w._physical_height == 4
        assert same(*feed(w, u, [7.], [i], d))  # the last row replaced
    assert drops


@pytest.mark.parametrize('d, xs, last', [('maxindex:3', [0., 0., 100., 1., 2., 3.], 0.),
                                         ('minindex:3', [100., 100., 0., 3., 2., 1.], 4.)])
def test_an_extreme_left_with_the_head_is_not_kept(d, xs, last):
    w, u = pair(xs[:3], d)
    for i in [3, 4, 5]:
        feed(w, u, [xs[i]], [i], d)
    assert same(*feed(w, u, [last], [5], d))


# --- RC-31: the retention is counted exactly ------------------------------------------


@pytest.mark.parametrize('window, margin', [(2, 2**64 - 2), (2**63, 0), (2**62, 2**62)])
def test_a_retention_too_large_to_count_raises_before_any_data(window, margin):
    src = F([1., 2., 3.], range(3))
    src['ema:2']
    with pytest.raises(ValueError, match='is too large: the frame could not count the rows it retains'):
        volas.DataFrame(src, window=window, max_lookback=margin)
    assert src['close'].to_list() == [1., 2., 3.]


# --- RC-32: a drop of the head never refreshes ----------------------------------------


def test_dropping_the_head_leaves_stale_columns_stale():
    # window 2 + margin 2 retains 4 rows, dropped back to when 8 are held
    cols = bars(100)
    w = volas.DataFrame(rows(cols, 0, 5), window=2, max_lookback=['ema:3'])
    i = 5
    while w._physical_height < 7:  # up to the row before the drop
        w.append(rows(cols, i, i + 1))
        i += 1
    w.append(rows(cols, i, i + 1))  # this row drops the head
    assert w._physical_height == 4
    with pytest.raises(ValueError, match='stale computed'):
        w.to_numpy()
    assert same(w['ema:3'], rows(cols, 0, i + 1)['ema:3'].to_numpy()[-2:])


def test_a_column_that_cannot_refresh_never_blocks_an_append_or_the_drop():
    w = volas.DataFrame({'v': np.array([1, 2, 3], dtype='int64')}, window=2, max_lookback=['ema:2@v'])
    w.append(volas.DataFrame({'v': np.array([2**60], dtype='int64')}))
    for k in range(40):
        w.append(volas.DataFrame({'v': np.array([k], dtype='int64')}))
    assert w._physical_height <= 8
    with pytest.raises((ValueError, OverflowError)):
        w['ema:2@v']


# --- RC-33: a whole-column assignment writes the visible rows only ---------------------


def test_a_scalar_assignment_keeps_the_hidden_history():
    u = F([float(i) for i in range(1, 11)])
    w = volas.DataFrame(u, window=2, max_lookback=['ma:3'])
    w['close'] = 100.0
    c = u['close'].to_numpy().copy()
    c[-2:] = 100.0
    assert w['ma:3'].to_list() == F(c)['ma:3'].to_list()[-2:]


def test_an_array_assignment_keeps_the_hidden_history():
    u = F([float(i) for i in range(1, 11)])
    w = volas.DataFrame(u, window=2, max_lookback=['ma:3'])
    w['close'] = [100.0, 101.0]
    c = u['close'].to_numpy().copy()
    c[-2:] = [100.0, 101.0]
    assert w['ma:3'].to_list() == F(c)['ma:3'].to_list()[-2:]
    w['close'] = w['close'] * 2  # a Series aligned with the visible rows
    c[-2:] = [200.0, 202.0]
    assert w['ma:3'].to_list() == F(c)['ma:3'].to_list()[-2:]


def test_a_new_column_has_no_hidden_history():
    w = volas.DataFrame(F([float(i) for i in range(1, 11)]), window=2, max_lookback=['ma:3'])
    w['x'] = [5.0, 7.0]
    assert np.isnan(w['mom:1@x'].to_list()[0]) and w['mom:1@x'].to_list()[1] == 2.0


def test_an_assignment_converts_the_hidden_rows_exactly_or_raises():
    w = volas.DataFrame({'n': np.arange(6, dtype='int64'), 'f': [0.5] * 6}, window=2, max_lookback=2)
    w['n'] = [1.5, 2.5]  # int hidden rows convert to float exactly
    assert w.dtypes['n'] == 'float64' and w['n'].to_list() == [1.5, 2.5]
    with pytest.raises(TypeError, match="cannot assign int64 values to column 'f' of a windowed frame"):
        w['f'] = np.array([1, 2], dtype='int64')
    assert w['f'].to_list() == [0.5, 0.5]
    with pytest.raises(ValueError, match=r'length of values \(3\) does not match the window length \(2\)'):
        w['f'] = [1.0, 2.0, 3.0]


def test_a_declared_count_folds_exactly_and_a_late_one_raises():
    # a 5m frame by cumulate, window 1: increase is class 3 — refused on request (with
    # any margin), and declared it stays exact through a new period and re-folds of it
    bar0 = {'open': [1.0], 'high': [2.0], 'low': [0.5], 'close': [1.5], 'volume': [3.0]}
    base = volas.DataFrame(bar0, index=np.array([T0])).cumulate('5m')
    for margin in (0, 1):
        w = volas.DataFrame(base, window=1, max_lookback=margin)
        with pytest.raises(ValueError, match=r"'INCREASE@open' on this windowed frame: it counts a run"):
            w['INCREASE@open']
    w = volas.DataFrame(base, window=1, max_lookback=['INCREASE@open'])
    u = volas.DataFrame(base)
    u['INCREASE@open']
    for k, o in enumerate([2.0, 0.5, 3.0]):
        bar = volas.DataFrame({'open': [o], 'high': [4.0], 'low': [0.1], 'close': [2.0], 'volume': [1.0]},
                              index=np.array([T0 + (5 + k) * MIN]))
        w.append(bar)
        u.append(bar)
        assert same(w['INCREASE@open'].to_numpy(dtype='float64'),
                    u['INCREASE@open'].to_numpy(dtype='float64')[-1:]), k


# --- R18: `ready` --------------------------------------------------------------------


@pytest.mark.parametrize('d, window', [('ema:5', 3), ('ma:5', 3), ('rsi:5', 2)])
def test_ready_counts_the_complete_rows_a_declared_directive_reads(d, window):
    # leading missing closes delay the warm-up: ready turns True exactly when the first
    # visible row of the declared directive holds a value in the unbounded history
    cols = bars(40, seed=2)
    cols['close'][:2] = np.nan
    w = volas.DataFrame(rows(cols, 0, 1), window=window, max_lookback=[d])
    u = volas.DataFrame(rows(cols, 0, 1))
    for i in range(1, 40):
        w.append(rows(cols, i, i + 1))
        u.append(rows(cols, i, i + 1))
        w.fulfill()
        first_visible = u[d].to_numpy()[len(u) - len(w)]
        assert w.ready == (not np.isnan(first_visible)), i


# --- a windowed frame built from a tail that carries a recursion's state ---------------


@pytest.mark.parametrize('read_every', [1, 7, 1000])
@pytest.mark.parametrize('d', ['ema:5', 'rsi:5', 'obv', 'atr:5'])
def test_a_tail_carried_into_a_window_keeps_its_recursion(d, read_every):
    cols = bars(400, seed=5)
    base = rows(cols, 0, 200)
    base[d]
    w = volas.DataFrame(base.iloc[150:], window=5, max_lookback=20)
    u = rows(cols, 0, 200)
    u[d]
    for i in range(200, 300):
        w.append(rows(cols, i, i + 1))
        u.append(rows(cols, i, i + 1))
        if i % read_every == 0 or i == 299:
            assert same(w[d], u[d].to_numpy()[-5:]), i


def test_a_tail_carried_into_a_window_cannot_recompute_after_a_visible_write():
    # its checkpoint is the tail's last row, which a visible write may follow: the rows up
    # to it are no longer ones the frame vouches for
    cols = bars(300, seed=5)
    base = rows(cols, 0, 200)
    base['ema:5']
    w = volas.DataFrame(base.iloc[150:], window=5, max_lookback=20)
    w.iat[1, 3] = 1.0
    with pytest.raises(ValueError, match="cannot recompute 'ema:5' exactly"):
        w['ema:5']


# --- a whole-column assignment keeps the margin exactly ----------------------------------


def test_a_narrowing_assignment_must_hold_the_hidden_rows_exactly():
    w = volas.DataFrame(rows(bars(100), 0, 100), window=4, max_lookback=['ema:5', 'ma:5'])
    v32 = np.array([101.1, 101.2, 101.3, 101.4], dtype=np.float32)
    with pytest.raises(TypeError, match="cannot assign float32 values to column 'close'"):
        w['close'] = v32
    assert w.dtypes['close'] == 'float64'
    # hidden values a float32 holds exactly are kept
    exact = volas.DataFrame({'close': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]}, window=2, max_lookback=['ma:3'])
    exact['close'] = np.array([7.5, 8.5], dtype=np.float32)
    assert exact['close'].to_list() == [7.5, 8.5]
    assert exact['ma:3'].to_list() == [(3.5 + 4.5 + 7.5) / 3, (4.5 + 7.5 + 8.5) / 3]


def test_an_input_out_of_the_float_domain_is_reported_as_it_is():
    w = volas.DataFrame({'v': np.array([1, 2, 3, 4], dtype='int64')}, window=2, max_lookback=['ema:2@v'])
    for k in range(20):
        w.append(volas.DataFrame({'v': np.array([2**60 if k == 3 else k], dtype='int64')}))
    # the row went with the dropped head: the reason its history could not be kept stays
    with pytest.raises((ValueError, OverflowError), match='2\\^53') as raised:
        w['ema:2@v']
    assert 'cannot recompute' not in str(raised.value)
