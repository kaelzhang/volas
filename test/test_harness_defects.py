"""Minimal reproductions of the open defects the differential harness steps around.

Each test asserts the contract (see ``harness/contracts.py``) and is a strict xfail
naming its defect: when a fix lands the test XPASSes, which fails the suite until the
xfail and the matching ``OPEN_DEFECTS`` / ``DIVERGENT_FAMILIES`` entry are removed —
so the machines start hunting that defect class again.
"""

import numpy as np
import pytest

import volas

from .harness import build
from .harness.data import EPOCHS, NS, Schema, make_rows
from .harness.ledger import Ledger
from .harness.schedules import OHLCV


def _xfail(defect, why):
    return pytest.mark.xfail(strict=True, reason=f'{defect}: {why}')


def _walk(n, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.standard_normal(n))
    o = c + 0.3 * rng.standard_normal(n)
    spread = np.abs(rng.standard_normal(n)) + 0.1
    return {'open': o, 'high': np.maximum(o, c) + spread, 'low': np.minimum(o, c) - spread, 'close': c,
            'volume': rng.uniform(1, 100, n)}


def _frame(cols, a, b, index=None):
    data = {k: v[a:b] for k, v in cols.items()}
    if index is None:
        return volas.DataFrame(data)
    return volas.DataFrame({'time': index[a:b], **data}).set_index('time')


def _same(a, b):
    return np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True)


# --- atomicity (I8) -----------------------------------------------------------------


@_xfail('M1 append-atomicity', 'a dtype error on a later column leaves earlier columns grown')
def test_a_rejected_append_leaves_the_frame_unchanged():
    df = volas.DataFrame({'open': [1.0, 2.0], 'close': [1.5, 2.5], 'volume': [10.0, 20.0]})
    with pytest.raises(TypeError):
        df.append(volas.DataFrame({'open': [3.0], 'close': [3.5], 'volume': np.array([30], 'int64')}))
    assert [len(df.get_column(c)) for c in df.columns] == [2, 2, 2]


# --- name binding (I9) ---------------------------------------------------------------


@_xfail('M2 stale-name-binding', 'a composite cached over a stale bare-named cached column keeps its NaN')
def test_a_composite_never_reads_a_stale_cached_column():
    cols = _walk(30)
    df = _frame(cols, 0, 28)
    df['obv']
    df.append(_frame(cols, 28, 29))
    df['close - obv']
    df.append(_frame(cols, 29, 30))
    df.fulfill()
    assert _same(df['close - obv'], _frame(cols, 0, 30)['close - obv'])


@_xfail('H1 stringify-column-name', 'directive_stringify keeps argument spellings the column name normalizes')
@pytest.mark.parametrize('directive', ['ma:05', 'sar:0.020', 'stddev:2,1.50', 'boll.upper:20,2.0'])
def test_stringify_is_the_cached_column_name(directive):
    df = _frame(_walk(30), 0, 30)
    before = set(df.columns)
    df[directive]
    assert set(df.columns) - before == {volas.directive_stringify(directive)}


# --- freshness of read surfaces (I10) ----------------------------------------------------


@_xfail('M3 unguarded-stale-read', "exec of a bare cached name returns the stale placeholder")
def test_exec_is_fresh_on_a_stale_frame():
    cols = _walk(30)
    df = _frame(cols, 0, 29)
    df['obv']
    df.append(_frame(cols, 29, 30))
    assert _same(df.exec('obv'), _frame(cols, 0, 30).exec('obv'))


@_xfail('M3 unguarded-stale-read', 'get_column of a stale computed column returns the placeholder')
def test_get_column_of_a_stale_computed_column_raises():
    cols = _walk(30)
    df = _frame(cols, 0, 29)
    df['ema:3']
    df.append(_frame(cols, 29, 30))
    with pytest.raises(ValueError, match='stale'):
        df.get_column('ema:3')


@_xfail('C5 stale-compare-rhs', 'a comparison consumes a stale right-hand frame')
def test_a_comparison_rejects_a_stale_right_hand_frame():
    cols = _walk(30)
    a = _frame(cols, 0, 29)
    a['ema:3']
    b = a.copy()
    b.append(_frame(cols, 29, 30))
    a.append(_frame(cols, 29, 30))
    a.fulfill()
    with pytest.raises(ValueError, match='stale'):
        a == b  # noqa: B015


@_xfail('H2 stale-append-source', 'append copies a stale source frame placeholders as data')
def test_appending_a_stale_frame_raises():
    cols = _walk(30)
    src = _frame(cols, 20, 29)
    src['ema:3']
    src.append(_frame(cols, 29, 30))
    df = _frame(cols, 0, 20)
    df['ema:3']
    with pytest.raises(ValueError, match='stale'):
        df.append(src)


@_xfail('C4 windowed-source-margin', "a windowed source contributes its hidden margin rows")
def test_a_windowed_source_contributes_only_its_visible_rows():
    cols = _walk(20)
    src = volas.DataFrame({k: v[:10] for k, v in cols.items()}, window=3, max_lookback=4)
    df = _frame(cols, 0, 2)
    df.append(src)
    assert len(df) == 2 + len(src)


@_xfail('H3 windowed-stale-logical-read', 'a stale windowed frame reads through its visible slice')
def test_a_stale_windowed_frame_rejects_a_bulk_read():
    cols = _walk(40)
    df = volas.DataFrame({k: v[:30] for k, v in cols.items()}, window=1, max_lookback=19)
    df['accbands']
    df.append(_frame(cols, 30, 31))
    with pytest.raises(ValueError, match='stale'):
        df.rolling(1).count()


# --- append index contract (I12) -----------------------------------------------------------

_T = np.datetime64('2026-01-01T00:00', 'ns') + np.arange(4) * np.timedelta64(60, 's')


def _dated():
    return volas.DataFrame({'time': _T[:3], 'close': [1.0, 2.0, 3.0]}).set_index('time')


@_xfail('M6 append-index-coercion', 'a RangeIndex bar turns a DatetimeIndex into int labels')
def test_a_range_index_bar_onto_a_datetime_frame_raises():
    with pytest.raises(ValueError):
        _dated().append(volas.DataFrame({'close': [4.0]}))


@_xfail('M6 append-index-coercion', 'a naive bar onto an aware frame is taken as UTC')
def test_a_naive_bar_onto_an_aware_frame_raises():
    df = _dated().tz_localize('Asia/Shanghai')
    with pytest.raises(ValueError):
        df.append(volas.DataFrame({'time': _T[3:], 'close': [4.0]}).set_index('time'))


@_xfail('M6 append-index-coercion', 'a NaT-labelled bar is accepted by a plain frame')
def test_a_nat_bar_onto_a_plain_frame_raises():
    bar = volas.DataFrame({'close': [4.0]}, index=np.array(['NaT'], 'datetime64[ns]'))
    with pytest.raises(ValueError):
        _dated().append(bar)


@_xfail('C6 localize-subsecond', 'tz_localize drops the sub-second part')
def test_tz_localize_keeps_subseconds():
    t = np.array(['2026-03-08T06:59:59.123456789'], 'datetime64[ns]')
    df = volas.DataFrame({'time': t, 'c': [1.0]}).set_index('time').tz_localize('UTC')
    assert np.asarray(df.index)[0] == t[0]


# --- numeric resume / kernels (I1 / I6) ----------------------------------------------------


@_xfail('H4 resume-warmup-capture', 'a state captured inside the warm-up resumes where the full is NaN')
@pytest.mark.parametrize('directive,rows', [('adosc', 1), ('ht_trendline', 62), ('ht_dcperiod', 31),
                                            ('mama', 31)])
def test_a_column_cached_inside_its_warm_up_continues_exactly(directive, rows):
    cols = _walk(90)
    df = _frame(cols, 0, rows)
    df[directive]
    for i in range(rows, 90):
        df.append(_frame(cols, i, i + 1))
        df.fulfill()
    assert _same(df[directive], _frame(cols, 0, 90)[directive])


@_xfail('H5 fast-above-slow', 'macd with fast > slow panics instead of raising')
def test_macd_with_fast_above_slow_raises_a_directive_error():
    with pytest.raises(volas.DirectiveError):
        _frame(_walk(80), 0, 80)['macd:30,5']


@_xfail('H5 fast-above-slow', 'apo / ppo / dma with fast > slow resume differently from the full compute')
@pytest.mark.parametrize('directive', ['apo:10,2', 'ppo:10,2', 'dma:10,1'])
def test_fast_above_slow_resumes_like_the_full_compute(directive):
    cols = _walk(60, 2)
    df = _frame(cols, 0, 5)
    df[directive]
    for i in range(5, 60):
        df.append(_frame(cols, i, i + 1))
    df.fulfill()
    assert _same(df[directive], _frame(cols, 0, 60)[directive])


@_xfail('H6 sliding-sum-magnitude', 'a 1e300 input outside the window still skews the full kernel')
def test_a_huge_value_outside_the_window_does_not_skew_linearreg():
    cols = _walk(30)
    cols['close'] = cols['close'].copy()
    cols['close'][0] = 1e300
    full = np.asarray(_frame(cols, 0, 30)['linearreg:2'])
    assert _same(full[-5:], np.asarray(_frame(cols, 10, 30)['linearreg:2'])[-5:])


@_xfail('H7 candle-nan-memory', 'one NaN input suppresses a candlestick pattern far beyond its lookback')
@pytest.mark.parametrize('pattern', ['style.doji', 'style.hammer', 'style.longline'])
def test_a_nan_input_leaves_a_candle_pattern_after_its_lookback(pattern):
    ledger = Ledger(OHLCV)
    rows = make_rows(OHLCV, ('walk', 0), 60, None)
    rows[11]['high'] = None
    ledger.append(rows, [EPOCHS[0] + i * 60 * NS for i in range(60)])
    start = 12 + volas.directive_lookback(pattern) + 2
    full = np.asarray(ledger.frame()[pattern])[start:]
    after = np.asarray(ledger.frame(start=12)[pattern])[start - 12:]
    assert _same(full, after)


@_xfail('H8 comoment-cancellation', 'correl:2 of two points leaves [-1, 1]')
def test_correl_stays_within_unit_range():
    cols = _walk(200, seed=7)
    values = np.asarray(_frame(cols, 0, 200)['correl:2@close,open'])
    finite = values[np.isfinite(values)]
    assert np.all(np.abs(finite) <= 1.0)


@_xfail('H9 vr-cancellation', 'vr over a window without down volume is a huge ratio of either sign')
def test_vr_matches_a_window_recompute():
    rows = make_rows(OHLCV, ('walk', 11), 64, None)
    ledger = Ledger(OHLCV)
    ledger.append(rows, [EPOCHS[1] + i * 60 * NS for i in range(64)])
    full = np.asarray(ledger.frame()['vr:3'])
    tail = np.asarray(ledger.frame(start=30)['vr:3'])
    assert _same(full[40:], tail[10:])


@_xfail('known stochrsi-ulp', 'stochrsi.d resume drifts from the full kernel in the last ULP')
def test_stochrsi_d_resume_is_bit_exact():
    drift = []
    for seed in range(12):
        cols = _walk(40, seed)
        df = _frame(cols, 0, 21)
        df['stochrsi.d']
        df.append(_frame(cols, 21, 22))
        df.fulfill()
        df.append(_frame(cols, 22, 40))
        got, want = np.asarray(df['stochrsi.d']), np.asarray(_frame(cols, 0, 40)['stochrsi.d'])
        if not np.array_equal(got, want, equal_nan=True):
            drift.append(seed)
    assert not drift


# --- tf fold (I13) --------------------------------------------------------------------------


@_xfail('M7 int-fold-na', 'an int fold reads the NA placeholder (0) as a value')
def test_an_int_max_fold_skips_na():
    schema = Schema('float64', 'float64', (('n', 'int64'),), 'datetime', None)
    t0 = EPOCHS[1]
    a = {'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1.0, 'n': -5}
    b = dict(a, n=None)
    df = build.frame(schema, [a], [t0]).cumulate('5m', cumulators={'n': 'max'})
    df.append(build.frame(schema, [b], [t0 + 60 * NS]))
    assert df.get_column('n').to_numpy(dtype='object').tolist() == [-5]


@_xfail('known all-nan-fold-extreme', 'an all-NaN period folds high / low to -inf / +inf')
def test_an_all_nan_period_folds_to_nan():
    schema = Schema('float64', 'float64', (), 'datetime', None)
    row = {'open': None, 'high': None, 'low': None, 'close': None, 'volume': None}
    df = build.frame(schema, [row], [EPOCHS[1]]).cumulate('5m')
    assert np.isnan(df.get_column('high').to_numpy()[0])


@_xfail('M5 windowed-drift', 'a windowed recursive column is not bit-exact with the unbounded frame')
def test_a_windowed_recursive_column_is_bit_exact():
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.standard_normal(200))
    v = rng.uniform(1, 100, 200)
    cols = {'open': c + 0.3, 'high': c + 1, 'low': c - 1, 'close': c, 'volume': v}
    w = volas.DataFrame({k: x[:40] for k, x in cols.items()}, window=5, max_lookback=['ema:10'])
    w['ema:10']
    for i in range(40, 200):
        w.append(_frame(cols, i, i + 1))  # no read until the end
    assert np.array_equal(np.asarray(w['ema:10']), np.asarray(_frame(cols, 0, 200)['ema:10'])[-5:])


@_xfail('H10 resume-zero-prior', 'the trix resume divides by a zero prior where the full kernel yields 0')
def test_trix_resume_matches_on_zero_prices():
    vals = [0.0, 0.0, 1.0, 2.0]
    df = volas.DataFrame({'close': vals[:1]})
    df['trix:1']
    for i in range(1, 4):
        df.append(volas.DataFrame({'close': vals[i:i + 1]}))
        df.fulfill()
    assert _same(df['trix:1'], volas.DataFrame({'close': vals})['trix:1'])


@_xfail('H19 slice-recursive-rewarm', 'a recursive composite re-warms on a slice instead of continuing')
def test_a_recursive_composite_continues_across_a_slice():
    cols = _walk(40, 3)
    df = _frame(cols, 0, 30)
    df['(close)+(kama:3@open)']
    df = df.iloc[1:]
    df.append(_frame(cols, 30, 31))
    df.fulfill()
    assert np.array_equal(np.asarray(df['close+kama:3@open'])[-1:],
                          np.asarray(_frame(cols, 0, 31)['close+kama:3@open'])[-1:])


@_xfail('H20 variance-cancellation', 'running sum-of-squares variance drifts; sqrt amplifies it')
def test_a_one_sample_bollinger_band_is_the_value_itself():
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.standard_normal(500))
    band = np.asarray(volas.DataFrame({'close': c})['boll.lower:1'], float)
    assert np.nanmax(np.abs(band - c)) <= 1e-9 * np.max(np.abs(c))


@_xfail('H22 ultosc-period-order', 'ultosc with periods out of order refreshes differently from its full compute')
@pytest.mark.parametrize('directive', ['ultosc:29', 'ultosc:7,30,10', 'ultosc:20,5,3'])
def test_ultosc_with_unordered_periods_refreshes_like_the_full_compute(directive):
    cols = _walk(80, 0)
    df = _frame(cols, 0, 40)
    df[directive]
    for i in range(40, 80):
        df.append(_frame(cols, i, i + 1))
        df.fulfill()
    assert _same(df[directive], _frame(cols, 0, 80)[directive])
