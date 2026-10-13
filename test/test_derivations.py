"""Every derivation of a frame against a from-scratch computation of the same accepted
events (docs/dataframe-api.md "Derived frames"):

- a contiguous row selection, however it is written, continues each cached column: its
  values, an append and a re-sent last row equal the full history's, bit for bit — a
  middle selection replays its recursions once, a short one keeps the rows before it its
  columns read, a recursion that carries no state keeps its whole history;
- a projection keeps the rows and continues the listed cached columns;
- every other derivation is a new series: cached columns dropped, recomputed from the new
  frame's own rows on request;
- a derivation of a stale frame raises (the read gate), a copy proceeds.
"""

import numpy as np
import pytest

import volas
from volas import DataFrame

N = 60
rng = np.random.default_rng(11)
_c = 100 + np.cumsum(rng.normal(0, 1, N + 4))
COLS = {'open': _c + rng.normal(0, .3, N + 4), 'high': _c + 1 + rng.random(N + 4),
        'low': _c - 1 - rng.random(N + 4), 'close': _c,
        'volume': rng.integers(1, 1000, N + 4).astype(float)}
DATA = list(COLS)

# one directive per history class and shape: a window, a running accumulator, a count, a
# recursion carried by its state (plain, reading past its lookback, over an operand), one
# that carries none (a command, a composite), an absolute position, a boolean composite
DIRECTIVES = ['ma:5', 'boll.upper:10', 'increase:3', 'ema:12', 'sar', 'ema:5@(ma:3)', 'macd.signal',
              'kcw:5', 'close+kama:3@open', 'maxindex:5', 'close > ema:6']


def frame(a, b, cols=COLS):
    return DataFrame({k: v[a:b] for k, v in cols.items()}, index=list(range(a, b)))


def bars(a, b, salt=0.0):
    return {k: v[a:b] * (1 + salt) for k, v in COLS.items()}


def cached(directives=DIRECTIVES, a=0, b=N):
    df = frame(a, b)
    for d in directives:
        df[d]
    return df


def same(got, want):
    got, want = np.asarray(got, dtype=object), np.asarray(want, dtype=object)
    norm = [lambda v: None if v is volas.NA or (isinstance(v, float) and v != v) else v] * 2
    return [norm[0](v) for v in got] == [norm[1](v) for v in want]


def full(directive, rows):
    """The directive over the full history of `rows` (a dict of columns)."""
    return DataFrame(rows).exec(directive, dtype='object')


def history(b, extra=None):
    """The accepted rows: the first `b` rows, then `extra` (a dict of columns)."""
    rows = {k: v[:b] for k, v in COLS.items()}
    if extra:
        rows = {k: np.concatenate([rows[k], extra[k]]) for k in rows}
    return rows


# --- contiguous row selections ------------------------------------------------------------

CONTIGUOUS = {
    'iloc': lambda df, a, b: df.iloc[a:b],
    'iloc_2d': lambda df, a, b: df.iloc[a:b, :],
    'getitem': lambda df, a, b: df[a:b],
    'loc': lambda df, a, b: df.loc[a:b - 1],
    'loc_2d': lambda df, a, b: df.loc[a:b - 1, :],
    'mask': lambda df, a, b: df[np.arange(len(df)) >= a][:b - a],
    'bool_list': lambda df, a, b: df[[a <= i < b for i in range(len(df))]],
    'iloc_list': lambda df, a, b: df.iloc[list(range(a, b))],
    'drop': lambda df, a, b: df.drop([i for i in range(len(df)) if not a <= i < b]),
    'head_tail': lambda df, a, b: df.head(b).tail(b - a),
}
RANGES = {'tail': (N - 7, N), 'short_tail': (N - 2, N), 'one_row': (N - 1, N), 'middle': (17, 41),
          'short_middle': (30, 32), 'head': (0, 25), 'first_row': (0, 1), 'whole': (0, N)}


@pytest.mark.parametrize('rng_name', RANGES)
@pytest.mark.parametrize('form', CONTIGUOUS)
def test_a_contiguous_selection_continues_every_cached_column(form, rng_name):
    a, b = RANGES[rng_name]
    df = cached()
    sub = CONTIGUOUS[form](df, a, b)
    assert list(sub.index) == list(range(a, b))
    for d in DIRECTIVES:
        name = volas.directive_stringify(d)
        assert sub.is_computed(name), (form, d)
        assert same(sub[name].to_numpy(dtype='object'), full(d, history(b))[a:b]), (form, d)
    # an append continues every column as the full history would
    new = bars(b, b + 2, 0.01)
    sub.append(DataFrame(new, index=[b, b + 1]))
    sub.fulfill()
    for d in DIRECTIVES:
        name = volas.directive_stringify(d)
        assert same(sub[name].to_numpy(dtype='object'), full(d, history(b, new))[a:]), (form, d, 'append')
    # a re-sent last row replaces it, every column as the full history would
    resent = {k: v[-1:] * 1.003 for k, v in new.items()}
    sub.append(DataFrame(resent, index=[b + 1]))
    sub.fulfill()
    rows = {k: np.concatenate([new[k][:1], resent[k]]) for k in new}
    for d in DIRECTIVES:
        name = volas.directive_stringify(d)
        assert same(sub[name].to_numpy(dtype='object'), full(d, history(b, rows))[a:]), (form, d, 'resend')


@pytest.mark.parametrize('rng_name', ['tail', 'short_tail', 'middle'])
def test_a_selection_of_a_selection_continues_as_the_first_one(rng_name):
    a, b = RANGES[rng_name]
    once = cached().iloc[a - 1 if a else 0:]
    twice = once.iloc[1 if a else 0:b - a + (1 if a else 0)]
    new = bars(b, b + 1, -0.02)
    twice.append(DataFrame(new, index=[b]))
    twice.fulfill()
    for d in DIRECTIVES:
        assert same(twice[volas.directive_stringify(d)].to_numpy(dtype='object'),
                    full(d, history(b, new))[a:]), d


def test_a_write_on_a_selection_recomputes_its_columns_over_their_history():
    # a window column, and a recursion that carries no state (whose whole history the
    # selection keeps), recompute over the rows the selection keeps before it — and so
    # does a recursion carried by its state once those rows hold its whole history
    rows = history(N)
    rows['close'] = rows['close'].copy()
    rows['close'][N - 3] = 99.0
    tail = cached(['ma:5', 'kcw:5', 'ema:12']).iloc[N - 3:]
    tail.iat[0, DATA.index('close')] = 99.0
    for d in ['ma:5', 'kcw:5', 'ema:12']:
        assert same(tail[d].to_numpy(dtype='object'), full(d, rows)[N - 3:]), d
    # alone, a recursion carried by its state replays from the checkpoint of its history
    # the selection keeps at its boundary
    tail = cached(['ema:12']).iloc[N - 3:]
    tail.iat[0, DATA.index('close')] = 99.0
    assert same(tail['ema:12'].to_numpy(dtype='object'), full('ema:12', rows)[N - 3:])


def test_selections_of_a_long_history_replay_from_its_anchors():
    # past the anchor spacing, a selection's edge replays from the nearest anchor; every
    # later selection reads the same anchors
    n = 10_000
    c = 100 + np.cumsum(np.random.default_rng(3).normal(0, 1, n + 1))
    df = DataFrame({'close': c[:n]})
    for d in ('ema:12', 'ema:5@(ma:3)'):
        df[d]
    for a, b in [(n - 5, n), (4100, 4200), (9000, 9001), (n - 3, n)]:
        sub = df.iloc[a:b]
        sub.append(DataFrame({'close': c[n:n + 1]}, index=[b]))
        for d in ('ema:12', 'ema:5@(ma:3)'):
            want = DataFrame({'close': np.append(c[:b], c[n])}).exec(d)[a:]
            assert np.array_equal(sub[d].to_numpy(), want, equal_nan=True), (a, b, d)


def test_a_selection_of_a_stale_frame_raises_and_a_copy_proceeds():
    df = cached(['ema:12'])
    df.append(frame(N, N + 1))
    for form in CONTIGUOUS.values():
        with pytest.raises(ValueError, match='stale'):
            form(df, N - 5, N)
    copy = df.copy()
    copy.fulfill()
    assert same(copy['ema:12'].to_numpy(dtype='object'), full('ema:12', history(N + 1)))


# --- projections ---------------------------------------------------------------------------

PROJECTIONS = {
    'getitem': lambda df, names: df[names],
    'constructor': lambda df, names: DataFrame(df, columns=names),
    'loc_2d': lambda df, names: df.loc[:, names],
}


@pytest.mark.parametrize('form', PROJECTIONS)
def test_a_projection_keeps_its_listed_cached_columns_as_data(form):
    # the listed columns only, their values as the source shows them; a listed cached
    # column is data from then on — never continued, even with its inputs alongside
    df = cached(['ema:12', 'ma:5']).iloc[N - 4:]
    p = PROJECTIONS[form](df, DATA + ['ema:12'])
    assert p.columns == DATA + ['ema:12'] and not p.is_computed('ema:12')
    assert same(p['ema:12'].to_numpy(dtype='object'), df['ema:12'].to_numpy(dtype='object'))
    p.append(DataFrame(bars(N, N + 1), index=[N]))  # a bar without it: missing there
    assert np.isnan(p['ema:12'].to_numpy()[-1]) and not p.is_computed('ema:12')
    q = PROJECTIONS[form](df, ['close', 'ema:12'])
    q.append(DataFrame({'close': [1.0], 'ema:12': [5.0]}, index=[N]))  # a bar with it: its value
    assert q['ema:12'].to_numpy()[-1] == 5.0


def test_a_projection_names_a_directive_by_its_canonical_column():
    df = cached(['ma:5'])
    assert df[['close', 'ma:05']].columns == ['close', 'ma:5']
    with pytest.raises(ValueError, match='duplicate column name'):
        df[['ma:5', 'ma:05']]


# --- new series --------------------------------------------------------------------------

NEW_SERIES = {
    'mask_gap': lambda df: df[np.arange(len(df)) % 2 == 0],
    'iloc_list_gap': lambda df: df.iloc[[0, 2, 5, 9]],
    'reversed': lambda df: df.sort_index(ascending=False),
    'astype': lambda df: df.astype({'volume': 'float32'}),
    'rename': lambda df: df.rename({'volume': 'vol'}),
    'reset_index': lambda df: df.reset_index(),
    'reset_index_drop': lambda df: df.reset_index(drop=True),
    'set_index': lambda df: df.reset_index().set_index('index'),
    'relabel': lambda df: DataFrame(df, index=list(range(100, 100 + len(df)))),
    'retype': lambda df: DataFrame(df, dtype='float64'),
}


@pytest.mark.parametrize('form', NEW_SERIES)
def test_every_other_derivation_is_a_new_series_without_the_cached_columns(form):
    df = cached()
    new = NEW_SERIES[form](df)
    assert not any(new.is_computed(c) for c in new.columns)
    assert not set(new.columns) & {volas.directive_stringify(d) for d in DIRECTIVES}
    # a directive requested again is computed from the new frame's own rows
    own = DataFrame({c: new[c].to_numpy() for c in new.columns if c in DATA})
    for d in ['ma:5', 'ema:12', 'maxindex:5']:
        assert same(new[d].to_numpy(dtype='object'), own.exec(d, dtype='object')), (form, d)


VALUE_TRANSFORMS = {
    'fillna': lambda df: df.fillna(0.0), 'ffill': lambda df: df.ffill(), 'bfill': lambda df: df.bfill(),
    'isna': lambda df: df.isna(), 'notna': lambda df: df.notna(), 'round': lambda df: df.round(2),
    'abs': lambda df: df.abs(), 'clip': lambda df: df.clip(0.0, 1e9), 'cumsum': lambda df: df.cumsum(),
    'cummax': lambda df: df.cummax(), 'cummin': lambda df: df.cummin(), 'cumprod': lambda df: df.cumprod(),
    'diff': lambda df: df.diff(), 'shift': lambda df: df.shift(1), 'isin': lambda df: df.isin([1.0]),
    'replace': lambda df: df.replace(1.0, 2.0), 'interpolate': lambda df: df.interpolate(),
    'rank': lambda df: df.rank(), 'eq': lambda df: df == 0.0, 'ne': lambda df: df != 0.0,
    'where': lambda df: df.where(df.notna()), 'mask': lambda df: df.mask(df.isna()),
    'rolling': lambda df: df.rolling(3).mean(), 'expanding': lambda df: df.expanding().sum(),
    'ewm': lambda df: df.ewm(span=3).mean(),
}


@pytest.mark.parametrize('form', VALUE_TRANSFORMS)
def test_a_value_transform_keeps_the_cached_columns_as_data(form):
    # every column transformed, a cached one as the plain data of its transformed values
    df = cached(['ma:5', 'ema:12'])
    new = VALUE_TRANSFORMS[form](df)
    assert new.columns == df.columns, form
    assert not any(new.is_computed(c) for c in new.columns)


def test_a_value_transform_of_a_cached_column_transforms_its_values():
    df = cached(['ma:5'])
    filled = df.fillna(0.0)
    want = np.nan_to_num(df['ma:5'].to_numpy(), nan=0.0)
    assert np.array_equal(filled['ma:5'].to_numpy(), want) and not filled.is_computed('ma:5')


def test_a_cached_column_read_as_a_series_becomes_a_data_column():
    df = cached(['ma:5'])
    for frame_ in (df['ma:5'].to_frame(), df['ma:5'].copy().to_frame()):
        assert frame_.columns == ['ma:5'] and not frame_.is_computed('ma:5')
        assert same(frame_['ma:5'].to_numpy(dtype='object'), df['ma:5'].to_numpy(dtype='object'))
    assert df['ma:5'].reset_index().columns == ['index', 'ma:5']


def test_a_new_series_of_a_stale_frame_raises():
    df = cached(['ema:12'])
    df.append(frame(N, N + 1))
    for form in NEW_SERIES.values():
        with pytest.raises(ValueError, match='stale'):
            form(df)


def test_two_d_rows_that_are_not_contiguous_do_not_carry_a_cached_column():
    df = cached(['ma:5'])
    assert df.loc[[0, 2], ['close']].columns == ['close']
    with pytest.raises(KeyError, match="'ma:5' is a cached directive column"):
        df.loc[[0, 2], ['close', 'ma:5']]
    assert same(df.loc[[4, 6], 'ma:5'].to_list(), df['ma:5'].to_list()[4:7:2])  # a read


# --- windowed frames -------------------------------------------------------------------------

def test_a_selection_of_a_windowed_frame_continues_its_declared_columns():
    seed = {k: v[:N] for k, v in COLS.items()}
    wf = DataFrame(seed, window=6, max_lookback=['ema:12', 'ma:5'])
    for i in range(N, N + 3):
        wf.append(DataFrame({k: v[i:i + 1] for k, v in COLS.items()}))
    wf.fulfill()
    for a, b in [(0, 6), (2, 5), (5, 6)]:
        sub = wf.iloc[a:b]
        assert sub.window is None if hasattr(sub, 'window') else True
        end = N + 3 - 6 + b
        new = bars(end, end + 1, 0.02)
        sub.append(DataFrame(new))
        sub.fulfill()
        for d in ['ema:12', 'ma:5']:
            want = full(d, history(end, new))[N + 3 - 6 + a:]
            assert same(sub[d].to_numpy(dtype='object'), want), (a, b, d)


@pytest.mark.parametrize('batches', [[7], [1] * 7, [3, 4]])
def test_a_compaction_before_a_refresh_keeps_a_recursion_s_checkpoint(batches):
    # the rows a checkpoint covers decide whether it is kept, never the output cursor
    d = 'ema:3@(ma:3)'
    f = DataFrame({'close': np.arange(20.0)}, window=3, max_lookback=[d])
    at = 20
    for k in batches:
        f.append(DataFrame({'close': np.arange(float(at), float(at + k))}))
        at += k
    whole = DataFrame({'close': np.arange(float(at))})
    assert f[d].to_list() == whole.exec(d).tolist()[-3:]


def test_a_windowed_frame_copied_by_the_constructor_keeps_its_window():
    seed = {k: v[:N] for k, v in COLS.items()}
    wf = DataFrame(seed, window=5, max_lookback=['ema:12'])
    for other in (DataFrame(wf), wf.copy()):
        assert len(other) == 5 and other.ready == wf.ready
        assert same(other['ema:12'].to_numpy(dtype='object'), wf['ema:12'].to_numpy(dtype='object'))
        other.append(DataFrame({k: v[N:N + 1] for k, v in COLS.items()}))
        assert len(other) == 5
        assert same(other['ema:12'].to_numpy(dtype='object'), full('ema:12', history(N + 1))[-5:])


# --- time-frame frames ------------------------------------------------------------------------

T0 = np.datetime64('2024-01-02T09:30', 'ns')
MIN = np.timedelta64(1, 'm')


def fine(minutes, closes):
    return DataFrame({'close': np.asarray(closes, float), 'volume': np.ones(len(closes))},
                     index=T0 + np.asarray(minutes) * MIN)


def tf_frame():
    df = fine(range(12), np.arange(12) + 1.0).cumulate('5m')
    df['ema:2']
    return df


def rows_of(df):
    return df.to_numpy(dtype='object').tolist()


@pytest.mark.parametrize('derive', [
    lambda df: df.iloc[1:], lambda df: df[1:], lambda df: df.tail(2), lambda df: df.loc[df.index[1]:],
    lambda df: df.rename({'volume': 'vol'}), lambda df: df.astype({'volume': 'float32'}),
    lambda df: df.tz_localize('UTC'), lambda df: df.tz_localize('UTC').tz_convert('Asia/Shanghai'),
    lambda df: df[['close', 'volume']],
])
def test_a_time_frame_frame_derived_into_the_same_bars_keeps_folding(derive):
    df = tf_frame()
    got = derive(df)
    bar = fine([12], [100.0]).rename({'volume': 'vol'} if 'vol' in got.columns else {})
    got.append(bar if got.tz is None else bar.tz_localize('UTC'))
    want = tf_frame()
    want.append(fine([12], [100.0]))
    assert len(got) == len(derive(tf_frame()))  # folded into the forming bar, no new row
    assert got['close'].to_list()[-1] == 100.0
    vol = [c for c in got.columns if c in ('volume', 'vol')][0]
    assert got[vol].to_list()[-1] == want['volume'].to_list()[-1]


def test_a_time_frame_frame_whose_bars_leave_the_grid_is_plain():
    day = fine([0, 1], [1.0, 2.0]).tz_localize('Asia/Shanghai').cumulate('1d')
    utc = day.tz_convert('UTC')  # Shanghai days start at 16:00 UTC: not UTC days
    n = len(utc)
    utc.append(fine([3], [5.0]).tz_localize('UTC'))
    assert len(utc) == n + 1


def test_a_middle_selection_of_a_time_frame_frame_is_the_frame_of_its_bars():
    df = tf_frame()
    mid = df.iloc[:2]  # the closed bars 09:30 and 09:35; the last one becomes forming
    mid.append(fine([9], [50.0]))  # a finer bar of 09:35 folds into it
    assert len(mid) == 2 and mid['close'].to_list()[-1] == 50.0


def test_a_rename_keeps_each_aggregator_and_ignores_an_absent_source():
    # a source that is not a column is no rename: volume keeps its sum
    t = np.datetime64('2024-01-01', 'ns') + np.arange(3) * np.timedelta64(1, 'm')
    f = DataFrame({'volume': [1.0, 1.0]}, index=t[:2]).cumulate('5m')
    s = f.rename({'Volume': 'volume'})
    s.append(DataFrame({'volume': [3.0]}, index=t[2:]))
    assert s['volume'].to_list() == [5.0]
    r = f.rename({'volume': 'vol'})
    r.append(DataFrame({'vol': [3.0]}, index=t[2:]))
    assert r['vol'].to_list() == [5.0]


def test_a_constructor_re_typed_projection_casts_its_listed_columns_as_data():
    df = cached(['ema:3'])
    got = DataFrame(df, columns=['close', 'ema:3'], dtype='float32')
    assert got.columns == ['close', 'ema:3'] and not got.is_computed('ema:3')
    assert [str(got[c].dtype) for c in got.columns] == ['float32', 'float32']


def test_an_empty_range_is_the_same_selection_in_every_form():
    df = cached(['ema:3'])
    for sub in (df.iloc[:0], df.iloc[:0, :], df.loc[N:], df.loc[N:, :]):
        assert sub.shape == (0, len(DATA) + 1) and sub.is_computed('ema:3')


def test_cumulators_without_a_time_frame_raise_on_a_copy_too():
    with pytest.raises(ValueError, match='cumulators requires time_frame'):
        DataFrame(cached(['ema:3']), cumulators={'close': 'sum'})


def test_a_windowed_source_made_unbounded_gives_its_visible_rows():
    seed = {k: v[:N] for k, v in COLS.items()}
    wf = DataFrame(seed, window=3, max_lookback=['ma:3'], index=T0 + np.arange(N) * MIN)
    tf = DataFrame(wf, time_frame='1m')
    assert len(tf) == 3 and tf['close'].to_list() == wf['close'].to_list()


def test_the_constructor_projects_a_cached_column_by_any_spelling():
    df = cached(['ema:3'])
    p = DataFrame(df, columns=['close', 'ema:03'])
    assert p.columns == ['close', 'ema:3'] and not p.is_computed('ema:3')


def test_a_period_key_selects_the_same_rows_in_one_and_two_dimensions():
    df = DataFrame({'close': np.arange(40.0)}, index=DAYS)
    df['ema:3']
    assert df.loc['2024-02', :].shape == df.loc['2024-02'].shape == (29, 2)
    assert df.loc['2024-02', :].is_computed('ema:3')


def test_dropping_a_column_keeps_the_others_continuing():
    # a drop removes the columns it names, everything else unchanged — not a projection:
    # kcw carries no state, so a slice keeps its whole history; once it is dropped, ema
    # stays cached and continues bit for bit, and only what ema needs stays
    df = cached(['ema:3', 'kcw:5'])
    tail = df.iloc[N - 10:]
    kept = tail.drop(['kcw:5'], axis=1)
    assert kept.columns == DATA + ['ema:3'] and kept.is_computed('ema:3')
    assert kept.iloc[5:].shape == (5, len(DATA) + 1) and kept.iloc[5:].is_computed('ema:3')
    kept.append(DataFrame(bars(N, N + 1), index=[N]))
    assert same(kept['ema:3'].to_numpy(dtype='object'), full('ema:3', history(N, bars(N, N + 1)))[N - 10:])


# --- labels -----------------------------------------------------------------------------------

@pytest.mark.parametrize('minutes', [[0, 1, 1, 2], [0, 1, 0, 2]])
def test_a_datetime_index_never_repeats_a_label(minutes):
    with pytest.raises(ValueError, match='duplicate labels'):
        DataFrame({'close': [1.0, 2.0, 3.0, 4.0]}, index=T0 + np.array(minutes) * MIN)
    with pytest.raises(ValueError, match='duplicate labels'):
        DataFrame({'close': [1.0, 2.0, 3.0, 4.0], 't': T0 + np.array(minutes) * MIN}).set_index('t')
    pdf = pytest.importorskip('pandas').DataFrame({'close': [1.0, 2.0, 3.0, 4.0]},
                                                    index=T0 + np.array(minutes) * MIN)
    with pytest.raises(ValueError, match='duplicate labels'):
        DataFrame.from_pandas(pdf)
    nat = np.array(['NaT', 'NaT', '2024-01-01'], dtype='datetime64[ns]')
    assert len(DataFrame({'close': [1.0, 2.0, 3.0]}, index=nat)) == 3  # NaT is no label


DAYS = np.datetime64('2024-01-30', 'ns') + np.arange(40) * np.timedelta64(1, 'D')


@pytest.mark.parametrize('lo,hi,first,last', [
    ('2024-01', '2024-02', '2024-01-30', '2024-02-29'),
    ('2024-02', None, '2024-02-01', '2024-03-09'),
    (None, '2024-01', '2024-01-30', '2024-01-31'),
    ('2024-02-28', '2024-03-01', '2024-02-28', '2024-03-01'),
    ('2024', '2024', '2024-01-30', '2024-03-09'),
])
def test_partial_date_strings_select_whole_periods(lo, hi, first, last):
    df = DataFrame({'close': np.arange(40.0)}, index=DAYS)
    df['ema:3']
    got = df.loc[lo:hi]
    assert str(got.index[0])[:10] == first and str(got.index[-1])[:10] == last
    assert got.is_computed('ema:3')  # a contiguous selection
    assert len(df[lo:hi]) == len(got)


def test_a_partial_date_key_selects_its_period_as_a_contiguous_selection():
    df = DataFrame({'close': np.arange(40.0)}, index=DAYS)
    df['ema:3']
    feb = df.loc['2024-02']
    assert len(feb) == 29 and feb.is_computed('ema:3')
    hourly = DataFrame({'close': np.arange(48.0)}, index=T0 + np.arange(48) * np.timedelta64(1, 'h'))
    assert len(hourly.loc['2024-01-02 10':'2024-01-02 12']) == 3


# --- a long history: the anchors its computation keeps ------------------------------------

LONG = 2600
_lc = 100 + np.cumsum(np.random.default_rng(5).normal(0, 1, LONG))
LONG_COLS = {'open': _lc + .2, 'high': _lc + 1, 'low': _lc - 1, 'close': _lc}
RECURSIONS = ('ema:12', 'ema:5@(ma:3)', 'sar', 'atr:14', 'macd')


def long_cached():
    df = DataFrame(dict(LONG_COLS))
    for d in RECURSIONS:
        df[d]
    return df


def written(lo, hi, at):
    """Each recursion over the long history with `close` at row `at` replaced, rows
    [lo, hi)."""
    cols = {k: v.copy() for k, v in LONG_COLS.items()}
    cols['close'][at] = 99.0
    return {d: DataFrame(cols).exec(d)[lo:hi] for d in RECURSIONS}


@pytest.mark.parametrize('lo,hi', [(LONG - 100, LONG), (1200, 1300), (1023, 1100), (1024, 1100), (LONG - 3, LONG)])
def test_a_selection_of_a_long_history_recomputes_a_written_row_exactly(lo, hi):
    # the computation keeps anchors of each recursion's history; a selection keeps the
    # nearest one before it with the rows after it, and a write to its rows recomputes them
    # from there — itself, a selection of it, and a window built from it
    df = long_cached()
    sub = df.iloc[lo:hi]
    for frame_, first in ((sub, lo), (sub.iloc[1:], lo + 1)):
        f = frame_.copy()
        f.iat[1, DATA.index('close')] = 99.0
        want = written(first, hi, first + 1)
        for d in RECURSIONS:
            assert np.array_equal(f[d].to_numpy(), want[d], equal_nan=True), (lo, hi, d)


def test_a_window_from_a_selection_of_a_long_history_recomputes_a_visible_write():
    df = long_cached()
    w = DataFrame(df.iloc[LONG - 60:], window=5, max_lookback=40)
    w.iat[1, DATA.index('close')] = 99.0
    want = written(LONG - 5, LONG, LONG - 4)
    for d in RECURSIONS:
        assert np.array_equal(w[d].to_numpy(), want[d], equal_nan=True), d


def test_a_selection_past_the_anchors_of_an_appended_history_replays_and_keeps_them():
    # rows appended after the computation have no anchors: the first selection past them
    # replays from the last one, forming those it passes, and a later one keeps them
    df = DataFrame({k: v[:600] for k, v in LONG_COLS.items()})
    for d in RECURSIONS:
        df[d]
    df.append(DataFrame({k: v[600:] for k, v in LONG_COLS.items()}, index=list(range(600, LONG))))
    df.fulfill()
    for lo in (LONG - 50, LONG - 40):
        f = df.iloc[lo:].copy()
        f.iat[1, DATA.index('close')] = 99.0
        want = written(lo, LONG, lo + 1)
        for d in RECURSIONS:
            assert np.array_equal(f[d].to_numpy(), want[d], equal_nan=True), (lo, d)
