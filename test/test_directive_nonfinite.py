"""Every directive consumer treats non-finite numeric inputs as missing."""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from volas import DataFrame
from test.exact import assert_exact


OPS = ['<', '<=', '==', '!=', '>=', '>', '//', r'\\', '><', '&', '|', '^', '+', '-', '*', '/']
EXPRESSIONS = [f'close{op}open' for op in OPS] + ['~close', '-close']
SIGNALS = [f'close{op}open' for op in OPS[:12]] + ['~close', 'close']
DIRECTIVES = EXPRESSIONS + [f'repeat:3@({d})' for d in SIGNALS] + [
    'repeat:1@close', 'repeat:3@(~(close>open))',
    'repeat:3@(repeat:2@(close>open))',
    'increase:3@close', 'increase:3,-1@close',
    'style.bullish', 'style.bearish',
    # Stateful operands exercise the plan's allocation-free scalar remainder
    # and scalar operand reader, as well as the vector continuation.
    'ema:3@(close>open)', 'ema:3@(~close)', 'ema:3@(close&open)',
    'ema:3@(close+open)', 'ema:3@(-close)',
    '(ema:3@close)>open', '(ema:3@close)//open',
    'repeat:3@((ema:3@close)>open)',
]
DIRECTIVES = sorted(set(DIRECTIVES + [f'ema:3@({d})' for d in EXPRESSIONS]
                        + [f'(ema:3@close){op}open' for op in OPS]
                        + [f'repeat:3@((ema:3@close){op}open)' for op in OPS[:12]]
                        + ['~(ema:3@close)', '-(ema:3@close)']))
MODES = ['plain', 'append', 'tf', 'windowed', 'windowed_tf']


@pytest.fixture(autouse=True)
def _restore_hypothesis_settings(request):
    function = request.node.obj
    original = getattr(function, '_hypothesis_internal_use_settings', None)
    yield
    if original is not None:
        function._hypothesis_internal_use_settings = original


@pytest.mark.parametrize('bad', [np.inf, -np.inf])
def test_repeat_comparison_nonfinite_is_missing(bad):
    data = {'close': [3., 3., 3., bad, 3., 3., 3.], 'open': [2.] * 7}
    got = DataFrame(data).exec('repeat:3@(close>open)', dtype='float64')
    assert_exact(got, np.array([np.nan, np.nan, 1., np.nan, np.nan, np.nan, 1.]))


def _frame(data, start, end, directive, mode, initial=False):
    ts = np.datetime64('2026-01-01', 'ns') + np.arange(start, end) * np.timedelta64(1, 'm')
    options = {}
    if initial and 'tf' in mode:
        options['time_frame'] = '1m'
    if initial and 'windowed' in mode:
        options.update(window=9, max_lookback=[directive])
    return DataFrame({k: v[start:end] for k, v in data.items()}, index=ts, **options)


@pytest.mark.parametrize('directive', DIRECTIVES)
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@settings(max_examples=3, deadline=None, derandomize=True)
@given(seed=st.integers(0, 2**16))
def test_directive_nonfinite_paths_equal_nan_and_full(directive, mode, dtype, seed):
    rng = np.random.default_rng(seed)
    data = {k: rng.integers(-3, 4, 48).astype(dtype) for k in ['close', 'open']}
    for k, values in data.items():
        rows = np.unique(np.r_[0, 1, 15, 16, 17, 20, 21, 32, 47, rng.choice(48, 6, replace=False)])
        values[rows] = rng.choice([np.inf, -np.inf, np.nan], len(rows))
    nan = {k: np.where(np.isfinite(v), v, np.nan) for k, v in data.items()}

    def check(a, b, end):
        expected = _frame(nan, 0, end, directive, 'plain').exec(directive, dtype='float64')
        actual = a[directive].to_numpy(dtype='float64')
        assert_exact(actual, b[directive].to_numpy(dtype='float64'), directive + ' inf/nan')
        assert_exact(actual, expected[-len(a):], directive + ' full')
        # A remembered fresh cache and the uncached evaluator must agree too.
        assert_exact(a[directive].to_numpy(dtype='float64'), actual, directive + ' remembered')
        if mode == 'plain':
            assert_exact(a.exec(directive, dtype='float64'), expected, directive + ' exec')

    if mode == 'plain':
        check(_frame(data, 0, 48, directive, mode), _frame(nan, 0, 48, directive, mode), 48)
        return
    a = _frame(data, 0, 16, directive, mode, True)
    b = _frame(nan, 0, 16, directive, mode, True)
    check(a, b, 16)
    start = 16
    for end in [17, 20, 21, 22, 32, 33, 48]:
        a.append(_frame(data, start, end, directive, mode))
        b.append(_frame(nan, start, end, directive, mode))
        check(a, b, end)
        # Replace the forming row with a finite value and then a missing value,
        # so state/validity must be rolled back rather than advanced twice.
        for replacement in [1., -np.inf]:
            changed = {k: v.copy() for k, v in data.items()}
            changed['close'][end - 1] = replacement
            normalized = {k: np.where(np.isfinite(v), v, np.nan) for k, v in changed.items()}
            a.append(_frame(changed, end - 1, end, directive, mode))
            b.append(_frame(normalized, end - 1, end, directive, mode))
            expected = _frame(normalized, 0, end, directive, 'plain').exec(directive, dtype='float64')
            assert_exact(a[directive].to_numpy(dtype='float64'), expected[-len(a):], directive + ' forming/full')
            assert_exact(a[directive].to_numpy(dtype='float64'), b[directive].to_numpy(dtype='float64'), directive + ' forming inf/nan')
        # Restore the original last row before continuing the independent ledger.
        a.append(_frame(data, end - 1, end, directive, mode))
        b.append(_frame(nan, end - 1, end, directive, mode))
        check(a, b, end)
        start = end
