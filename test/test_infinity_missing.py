"""Non-finite indicator inputs have exactly the existing missing-row semantics."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

import volas
from test.exact import assert_exact
from test.test_missing_rows import BASE, MATYPE, OPERANDS, _production_lines
from test.harness.inventory import sample_value


@pytest.fixture(autouse=True)
def _restore_hypothesis_settings(request):
    # The pinned pytest plugin adds a settings(parent=previous) to a reused
    # parametrized function at every call. Restore the original after each case
    # so thousands of cases do not form a recursive fallback chain. No data,
    # examples, assertions or health checks are changed here.
    function = request.node.obj
    original = getattr(function, '_hypothesis_internal_use_settings', None)
    yield
    if original is not None:
        function._hypothesis_internal_use_settings = original


DIRECTIVES = sorted({line.form + line.series_suffix for line in _production_lines()} | set(MATYPE) | set(OPERANDS))
# Exercise all nine MA types, including finite SMA / WMA / TRIMA selectors.
for _line in _production_lines():
    _slots = [i for i, arg in enumerate(_line.args) if arg.kind == 'int' and arg.lo == 0 and arg.hi == 8]
    for _kind in range(9) if _slots else []:
        _values = [str(_kind) if i in _slots else sample_value(arg, 5) for i, arg in enumerate(_line.args)]
        DIRECTIVES.append(_line.head + ':' + ','.join(_values) + _line.series_suffix)
DIRECTIVES = sorted(set(DIRECTIVES))
MODES = ['plain', 'append', 'tf', 'windowed', 'windowed_tf']


def _data(seed):
    rng = np.random.default_rng(seed)
    infinite = {k: v.copy() for k, v in BASE.items()}
    missing = {k: v.copy() for k, v in BASE.items()}
    for k in infinite:
        rows = np.unique(np.r_[0, 129, 130, 131, 259, rng.choice(260, 8, replace=False)])
        infinite[k][rows] = rng.choice([np.inf, -np.inf], len(rows))
        missing[k][rows] = np.nan
    return infinite, missing


def _frame(data, a, b, directive, mode, initial=False):
    # Same-period snapshots exercise the forming path without changing R14 aggregation.
    ts = np.datetime64('2026-01-01', 'ns') + np.arange(a, b) * np.timedelta64(1, 'm')
    kw = {}
    if initial and 'tf' in mode:
        kw['time_frame'] = '1m'
    if initial and 'windowed' in mode:
        kw.update(window=30, max_lookback=[directive])
    return volas.DataFrame({k: v[a:b] for k, v in data.items()}, index=ts, **kw)


@pytest.mark.parametrize('directive', DIRECTIVES)
@pytest.mark.parametrize('mode', MODES)
@settings(max_examples=2, deadline=None, derandomize=True)
@given(seed=st.integers(0, 2**16))
def test_infinities_equal_nan_on_every_execution_path(directive, mode, seed):
    inf, nan = _data(seed)
    if mode == 'plain':
        a = _frame(inf, 0, 260, directive, mode)
        b = _frame(nan, 0, 260, directive, mode)
        assert_exact(a.exec(directive, dtype='float64'), b.exec(directive, dtype='float64'), directive)
        assert_exact(a.exec(directive, dtype='float64'), b.exec(directive, dtype='float64'), directive + ' remembered')
        return
    a = _frame(inf, 0, 130, directive, mode, True)
    b = _frame(nan, 0, 130, directive, mode, True)
    assert_exact(a[directive].to_numpy(dtype='float64'), b[directive].to_numpy(dtype='float64'), directive + ' seed')
    full = _frame(nan, 0, 130, directive, 'plain').exec(directive, dtype='float64')
    assert_exact(b[directive].to_numpy(dtype='float64'), full[-len(b):], directive + ' seed/full')
    start = 130
    for end in [131, 134, 135, 160, 200, 260]:
        a.append(_frame(inf, start, end, directive, mode))
        b.append(_frame(nan, start, end, directive, mode))
        assert_exact(a[directive].to_numpy(dtype='float64'), b[directive].to_numpy(dtype='float64'), f'{directive} {mode} @{end}')
        full = _frame(nan, 0, end, directive, 'plain').exec(directive, dtype='float64')
        assert_exact(a[directive].to_numpy(dtype='float64'), full[-len(a):], f'{directive} {mode}/full @{end}')
        if 'tf' in mode:
            a.append(_frame(inf, end - 1, end, directive, mode))
            b.append(_frame(nan, end - 1, end, directive, mode))
            assert_exact(a[directive].to_numpy(dtype='float64'), full[-len(a):], directive + ' forming/full')
            assert_exact(a[directive].to_numpy(dtype='float64'), b[directive].to_numpy(dtype='float64'), directive + ' forming')
        start = end


WINDOW_METHODS = ['count', 'nunique', 'sum', 'mean', 'min', 'max', 'median', 'var', 'std',
                  'sem', 'skew', 'kurt', 'quantile', 'rank', 'first', 'last', 'cov', 'corr']
EWM_METHODS = ['sum', 'mean', 'var', 'std', 'cov', 'corr']


def _reduce(window, method, other):
    args = [other] if method in ['cov', 'corr'] else [0.3] if method == 'quantile' else []
    return getattr(window, method)(*args)


@pytest.mark.parametrize('family,method', [(f, m) for f in ['rolling', 'expanding'] for m in WINDOW_METHODS]
                         + [('ewm', m) for m in EWM_METHODS])
@settings(max_examples=3, deadline=None, derandomize=True)
@given(seed=st.integers(0, 2**16))
def test_window_infinities_equal_nan_and_pandas(family, method, seed):
    assert pd.__version__ == '3.0.3'
    rng = np.random.default_rng(seed)
    x = rng.integers(-8, 9, 40).astype(float)
    y = rng.integers(-8, 9, 40).astype(float)
    x[rng.choice(40, 8, replace=False)] = rng.choice([np.inf, -np.inf], 8)
    y[rng.choice(40, 8, replace=False)] = rng.choice([np.inf, -np.inf], 8)
    xn, yn = np.where(np.isfinite(x), x, np.nan), np.where(np.isfinite(y), y, np.nan)
    for center in ([False, True] if family == 'rolling' else [False]):
        for min_periods in [0, 1, 4]:
            opts = dict(window=7, min_periods=min_periods, center=center) if family == 'rolling' else dict(min_periods=min_periods)
            if family == 'ewm':
                opts.update(alpha=0.3, adjust=True, ignore_na=False)
            series = [volas.DataFrame({'x': z})['x'] for z in [x, y, xn, yn]]
            a, b, c, d = series
            got = _reduce(getattr(a, family)(**opts), method, b).to_numpy(dtype='float64')
            clean = _reduce(getattr(c, family)(**opts), method, d).to_numpy(dtype='float64')
            assert_exact(got, clean, f'{family}.{method}')
            p, q, pn, qn = [pd.Series(z) for z in [x, y, xn, yn]]
            oracle = _reduce(getattr(p, family)(**opts), method, q).to_numpy()
            normalized = _reduce(getattr(pn, family)(**opts), method, qn).to_numpy()
            # pandas count builds a notna mask before preparing window values, so it
            # counts infinities. sem inherits that denominator via std/sqrt(count).
            # The unified rule excludes infinities in both methods.
            if method not in ['count', 'sem']:
                assert_exact(oracle, normalized, f'pandas {family}.{method}')
            if family == 'rolling' and method == 'rank' and center:
                # pandas's documented clipped-tail rank bug keeps departed rows.
                # Check every clipped window independently, and retain the pandas
                # comparison wherever its forward edge is not clipped.
                back, fwd = opts['window'] // 2, opts['window'] - opts['window'] // 2 - 1
                ref = []
                for i in range(len(xn)):
                    values = xn[max(0, i - back):min(len(xn), i + fwd + 1)]
                    ref.append(pd.Series(values).rank().iloc[-1]
                               if np.isfinite(values).sum() >= max(1, min_periods) else np.nan)
                assert_exact(got, np.array(ref), 'centered rank clipped-window oracle')
                np.testing.assert_allclose(got[:-fwd], normalized[:-fwd],
                                           rtol=1e-10, atol=1e-10, equal_nan=True)
            else:
                np.testing.assert_allclose(got, normalized, rtol=1e-10, atol=1e-10, equal_nan=True)


@pytest.mark.parametrize('bad', [np.inf, -np.inf])
def test_ema_continues_from_the_last_complete_row(bad):
    got = volas.DataFrame({'close': [1., bad, 3., 4., 5.]}).exec('ema:3')
    want = np.full(5, np.nan)
    want[[0, 2, 3, 4]] = volas.DataFrame({'close': [1., 3., 4., 5.]}).exec('ema:3')
    assert want[3] == 8. / 3. and np.isfinite(want[3:]).all()
    assert_exact(got, want)


def test_directive_elements_treat_infinite_inputs_as_missing():
    df = volas.DataFrame({'close': [np.inf, -np.inf, 3.], 'open': [1., 1., 1.]})
    assert_exact(df.exec('close * 2'), np.array([np.nan, np.nan, 6.]))
    assert_exact(df.exec('close > open', dtype='float64'), np.array([np.nan, np.nan, 1.]))


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('bad', [np.inf, -np.inf])
def test_remembered_input_facts_follow_infinity_writes(dtype, bad):
    x = np.arange(10., 30., dtype=dtype)
    df = volas.DataFrame({'close': x})
    clean = df.exec('ema:3')
    assert_exact(df.exec('ema:3'), clean)
    df.iloc[8, 0] = bad
    x[8] = np.nan
    want = volas.DataFrame({'close': x}).exec('ema:3')
    assert_exact(df.exec('ema:3'), want)
    assert_exact(df.exec('ema:3'), want)
    df.iloc[8, 0] = 18.
    assert_exact(df.exec('ema:3'), clean)
