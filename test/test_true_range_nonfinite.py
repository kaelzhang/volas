"""TR and its smoothers on non-finite prices, against TA-Lib 0.6.4.

Non-finite inputs follow the missing-row rule: recursive oracles delete
those rows; per-term and finite-window oracles substitute NaN. NATR(1) retains
volas's documented normalization, and DI(1) its documented percentage scale.
"""

from itertools import combinations, product

import numpy as np
import pytest
import talib

import volas
from test import oracle_reference as native
from test.test_directional_talib_order import _adx_reference, _CONTRACTS

from test.exact import assert_exact


def _bits(got, want):
    assert_exact(got, want)


_VALUES = [0.0, -0.0, 3.0, -2.0, np.inf, -np.inf, np.nan]


@pytest.mark.parametrize('high,low,prev_close', product(_VALUES, repeat=3))
def test_tr_uses_talibs_ordered_comparisons(high, low, prev_close):
    h, l, c = np.array([5., high]), np.array([1., low]), np.array([prev_close, 2.])
    want = talib.TRANGE(h, l, c)
    if (~np.isfinite([high, low, prev_close])).any():
        want[1] = np.nan  # a term reading a missing operand is missing
    _bits(volas.DataFrame({'high': h, 'low': l, 'close': c}).exec('tr'), want)


_COLUMNS = [cols for n in (1, 2, 3) for cols in combinations(('high', 'low', 'close'), n)]


def _data(columns, value, row, n=80):
    rng = np.random.default_rng(231)
    c = 100. + np.cumsum(rng.normal(size=n))
    d = {'high': c + rng.random(n), 'low': c - rng.random(n), 'close': c}
    for col in columns:
        d[col][row:row + 2] = value
    return d


def _frame(d, start=0, end=None):
    return volas.DataFrame({k: v[start:end] for k, v in d.items()},
                           index=np.arange(len(d['close']))[start:end])


def _complete_oracle(d, fn):
    keep = np.logical_and.reduce([np.isfinite(v) for v in d.values()])
    out = np.full(len(keep), np.nan)
    with np.errstate(all='ignore'):
        out[keep] = fn(*(d[k][keep] for k in ('high', 'low', 'close')))
    return out


def _atr_oracle(d, period, normalized):
    def run(h, l, c):
        if not normalized:
            return talib.ATR(h, l, c, period)
        if period > 1:
            return talib.NATR(h, l, c, period)
        # NATR's documented percentage scale, including at period 1.
        return talib.NATR(h, l, c, 1) / c * 100.
    return _complete_oracle(d, run)


def _live_paths(d, directive, want):
    # Multi-row resume, starting before warm-up and crossing both the seed and bad rows.
    live = _frame(d, 0, 1)
    live[directive]
    for start, end in ((1, 4), (4, 12), (12, 40), (40, len(want))):
        live.append(_frame(d, start, end))
        _bits(live[directive], want[:end])
    # The allocation-free single-row path, with a same-label re-send of each bar.
    live = _frame(d, 0, 1)
    live[directive]
    for row in range(1, len(want)):
        bar = _frame(d, row, row + 1)
        live.append(bar)
        _bits(live[directive], want[:row + 1])
        live.append(bar)
        _bits(live[directive], want[:row + 1])


@pytest.mark.parametrize('columns', _COLUMNS)
@pytest.mark.parametrize('value', [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize('row', [0, 1, 5, 35])
@pytest.mark.parametrize('period', [1, 2, 7])
@pytest.mark.parametrize('name', ['atr', 'natr'])
def test_atr_full_and_live_match_talib(columns, value, row, period, name):
    d = _data(columns, value, row)
    directive = f'{name}:{period}'
    want = _atr_oracle(d, period, name == 'natr')
    _bits(_frame(d).exec(directive), want)
    _live_paths(d, directive, want)


@pytest.mark.parametrize('period', [1, 2, 7])
@pytest.mark.parametrize('name', ['atr', 'natr'])
def test_atr_with_opposite_infinities_in_the_same_bar(name, period):
    d = _data((), np.inf, 35)
    d['high'][35:39] = [np.inf, -np.inf, np.inf, -np.inf]
    d['low'][35:39] = [-np.inf, np.inf, np.inf, -np.inf]
    d['close'][35:39] = [np.inf, -np.inf, -np.inf, np.inf]
    directive = f'{name}:{period}'
    want = _atr_oracle(d, period, name == 'natr')
    _bits(_frame(d).exec(directive), want)
    _live_paths(d, directive, want)


@pytest.mark.parametrize('period', [1, 7])
@pytest.mark.parametrize('name', ['atr', 'natr'])
@pytest.mark.parametrize('windowed', [False, True])
def test_atr_forming_and_window_checkpoints_preserve_nonfinite_state(name, period, windowed):
    d = _data(('high', 'low'), np.inf, 35)
    for v in d.values():
        v[45] = np.nan
    directive = f'{name}:{period}'
    want = _atr_oracle(d, period, name == 'natr')
    index = np.datetime64('2024-01-01', 'ns') + np.arange(len(want)) * np.timedelta64(1, 'm')

    def frame(start, end, **kwargs):
        return volas.DataFrame({k: v[start:end] for k, v in d.items()},
                               index=index[start:end], **kwargs)

    options = {'window': 8, 'max_lookback': [directive]} if windowed else {}
    live = frame(0, 20, time_frame='1m', **options)
    live[directive]
    for row in range(20, len(want)):
        bar = frame(row, row + 1)
        for _ in range(2):  # rollover, then replacement of the forming bar
            live.append(bar)
            start = max(0, row + 1 - 8) if windowed else 0
            _bits(live[directive], want[start:row + 1])


@pytest.mark.parametrize('name', ['plus_dm', 'minus_dm', 'plus_di', 'minus_di', 'dx', 'adx', 'adxr'])
@pytest.mark.parametrize('period', [2, 7])
@pytest.mark.parametrize('columns', _COLUMNS)
@pytest.mark.parametrize('value', [np.inf, -np.inf, np.nan])
def test_directional_ranges_match_talib(name, period, columns, value):
    d = _data(columns, value, 35)
    if name.endswith('_dm'):
        # DM does not read close, so a missing close does not delete that row.
        d['close'] = np.full(len(d['close']), 100.)
        fn = lambda h, l, c: getattr(talib, name.upper())(h, l, period)
    elif name in ('adx', 'adxr') and _CONTRACTS:
        fn = lambda h, l, c: _adx_reference(h, l, c, period, name == 'adxr')
        installed = _complete_oracle(d, lambda h, l, c: getattr(talib, name.upper())(h, l, c, period))
        np.testing.assert_allclose(_complete_oracle(d, fn), installed,
                                   rtol=1e-12, atol=1e-12, equal_nan=True)
    else:
        fn = lambda h, l, c: getattr(talib, name.upper())(h, l, c, period)
    want = _complete_oracle(d, fn)
    directive = f'{name}:{period}'
    _bits(_frame(d).exec(directive), want)
    _live_paths(d, directive, want)


@pytest.mark.parametrize('name', ['plus_dm', 'minus_dm', 'plus_di', 'minus_di'])
@pytest.mark.parametrize('value', [np.inf, -np.inf, np.nan])
def test_directional_period_one_has_no_smoothing(name, value):
    d = _data(('high', 'low'), value, 35)
    def fn(h, l, c):
        if name.endswith('_dm'):
            return getattr(talib, name.upper())(h, l, 1)
        return getattr(talib, name.upper())(h, l, c, 1) * 100.
    want = _complete_oracle(d, fn)
    directive = f'{name}:1'
    _bits(_frame(d).exec(directive), want)
    _live_paths(d, directive, want)


@pytest.mark.parametrize('first', [1., -0., np.inf, -np.inf, np.nan])
def test_smma_period_one_skips_nonfinite_inputs(first):
    c = np.array([first, np.inf, 2., np.nan, -np.inf, -0., 3., 0.])
    d = {'close': c}
    want = np.where(np.isfinite(c), c, np.nan)
    _bits(_frame(d).exec('smma:1'), want)
    live = _frame(d, 0, 2)
    live['smma:1']
    for row in range(2, len(c)):
        live.append(_frame(d, row, row + 1))
        _bits(live['smma:1'], want[:row + 1])


@pytest.mark.parametrize('name', ['sar', 'sarext'])
@pytest.mark.parametrize('columns', [('high',), ('low',), ('high', 'low')])
@pytest.mark.parametrize('value', [np.inf, -np.inf, np.nan])
def test_sar_range_limits_do_not_discard_arithmetic_nan(name, columns, value):
    # Binary-exact factors and integer prices avoid the installed build's FMA
    # differences: this tests the C comparison sequence, including its NaN mask.
    c = 100. + np.arange(80) % 9
    d = {'high': c + 2., 'low': c - 1., 'close': c}
    for col in columns:
        d[col][35:37] = value
    if name == 'sar':
        directive = 'sar:0.5,0.5'
        fn = lambda h, l, c: talib.SAR(h, l, .5, .5)
    else:
        directive = 'sarext:0,0,0.5,0.5,0.5,0.5,0.5,0.5'
        fn = lambda h, l, c: talib.SAREXT(h, l, 0, 0, .5, .5, .5, .5, .5, .5)
    want = _complete_oracle(d, fn)
    _bits(_frame(d).exec(directive), want)
    _live_paths(d, directive, want)


@pytest.mark.parametrize('period', [1, 7])
@pytest.mark.parametrize('row', [1, 35])
@pytest.mark.parametrize('value', [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize('line', ['keltner.upper', 'keltner.lower', 'kcw',
                                  'supertrend', 'supertrend.direction'])
def test_native_atr_consumers_share_the_range_and_identity(line, value, row, period):
    d = _data(('high', 'low'), value, row)
    h, l, c = (d[k] for k in ('high', 'low', 'close'))
    if line.startswith('supertrend'):
        directive = f'{line}:{period},2'
        with np.errstate(all='ignore'):
            want = _complete_oracle(d, lambda h, l, c: native.supertrend(
                c, h, l, c, c, period, 2.,
                'direction' if line.endswith('direction') else 'line'))
    else:
        directive = f'{line}:5,{period},2'
        def fn(h, l, c):
            a, e = talib.ATR(h, l, c, period), native._ema(c, 5).to_numpy()
            if line == 'kcw':
                return 4. * a / e
            return e + (2. if line.endswith('upper') else -2.) * a
        want = _complete_oracle(d, fn)
    got = np.asarray(_frame(d).exec(directive))
    # Native EMA reference uses a mathematically equivalent expression; compare
    # rounding within 1e-12, but always compare every NaN and infinity too.
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, equal_nan=True)
    _live_paths(d, directive, got)


@pytest.mark.parametrize('periods', [(1, 1, 1), (2, 3, 7)])
@pytest.mark.parametrize('columns', _COLUMNS)
@pytest.mark.parametrize('value', [np.inf, -np.inf, np.nan])
def test_ultosc_uses_talib_terms_with_window_local_sums(periods, columns, value):
    d = _data(columns, value, 35)
    n, width = len(d['close']), max(periods)
    want = np.full(n, np.nan)
    for row in range(width, n):
        h, l, c = (d[k][row - width:row + 1].copy() for k in ('high', 'low', 'close'))
        # Only the prior close is read on the extra bar; avoid the Python
        # wrapper treating an irrelevant missing high/low as a leading gap.
        h[0], l[0] = 1., 0.
        if any((~np.isfinite(x)).any() for x in (h, l, c)):
            continue
        if width == 1:
            # TA_ULTOSC_Lookback(1,1,1) is SMA_Lookback(1)+1 == 0:
            # SMA rejects 1, so that build reads close[-1] at its first bar.
            # Use its TRANGE term and the defined one-bar C formula instead.
            with np.errstate(all='ignore'):
                tr = talib.TRANGE(h, l, c)[-1]
                ratio = (c[-1] - min(l[-1], c[-2])) / tr
                want[row] = 0. if abs(tr) < 1e-14 else 100. * ((4. * ratio + 2. * ratio + ratio) / 7.)
        else:
            want[row] = talib.ULTOSC(h, l, c, *periods)[-1]
    directive = 'ultosc:' + ','.join(map(str, periods))
    got = np.asarray(_frame(d).exec(directive))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, equal_nan=True)
    _live_paths(d, directive, got)


@pytest.mark.parametrize('directive', ['chop:7', 'vortex.plus:7', 'vortex.minus:7',
                                      'ttm_squeeze.on:7'])
@pytest.mark.parametrize('value', [np.inf, -np.inf, np.nan])
def test_finite_range_consumers_recover_after_a_bad_window(directive, value):
    d = _data(('high', 'low'), value, 35)
    got = np.asarray(_frame(d).exec(directive))
    assert np.isnan(got[35:43]).all()
    clean = _data((), value, 35)
    recovered = 43
    if directive.startswith('vortex'):
        # VM also reads the prior high/low: its last affected term is bar 37,
        # one later than TR's. At bar 43 the TR sum is finite but VM is not.
        assert np.isnan(got[43])
        recovered = 44
    _bits(got[recovered:], _frame(clean).exec(directive)[recovered:])
    _live_paths(d, directive, got)
