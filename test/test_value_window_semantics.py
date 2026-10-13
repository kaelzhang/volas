"""Regression coverage for numeric matching, missing ranks and window arithmetic."""
import numpy as np
import pandas as pd
import pytest

import volas


def series(values):
    return volas.DataFrame({'x': values})['x']


def bits_equal(actual, expected):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_array_equal(a[~np.isnan(a)].view('u8'), b[~np.isnan(b)].view('u8'))


@pytest.mark.parametrize('dtype', [np.float64, np.float32, np.int64, np.int32])
def test_numeric_membership_and_replace_cross_dtype(dtype):
    s = series(np.array([1, 2], dtype=dtype))
    probe = 1.0 if np.issubdtype(dtype, np.integer) else 1
    actual = (s.isin([probe]).to_list(), s.replace(probe, 9).to_list())
    assert actual == ([True, False], [9, 2])


def test_numeric_membership_preserves_large_integer_identity():
    assert series([2**53, 2**53 + 1, 2**63 - 1]).isin([float(2**53), float(2**63)]).to_list() == [True, False, False]


def test_frame_replace_surfaces_matched_column_type_error_atomically():
    f = volas.DataFrame({'a': [1, 2], 'b': ['one', 'two']})
    before = f.to_pandas()
    with pytest.raises(TypeError):
        f.replace(1, 'bad')
    pd.testing.assert_frame_equal(f.to_pandas(), before)


@pytest.mark.parametrize('values', [[np.nan, np.nan], [1., 1.]])
def test_frame_mode_pads_all_missing_columns(values):
    f = volas.DataFrame({'a': values, 'b': [np.nan, np.nan]})
    pd.testing.assert_frame_equal(f.mode().to_pandas(), f.to_pandas().mode())


@pytest.mark.parametrize('method', ['average', 'min', 'max', 'first', 'dense'])
@pytest.mark.parametrize('na_option', ['keep', 'top', 'bottom'])
@pytest.mark.parametrize('ascending', [True, False])
@pytest.mark.parametrize('pct', [True, False])
@pytest.mark.parametrize('values', [[3., np.nan, 1., np.nan, 3.], [np.nan, np.nan], ['b', None, 'a', None, 'b']])
def test_rank_missing_ties_and_percentile(method, na_option, ascending, pct, values):
    kw = dict(method=method, na_option=na_option, ascending=ascending, pct=pct)
    bits_equal(series(values).rank(**kw).to_list(), pd.Series(values).rank(**kw).tolist())


@pytest.mark.parametrize('scale', [1e100, 1e-100])
@pytest.mark.parametrize('adjust', [True, False])
def test_ewm_self_correlation_avoids_variance_product_range_loss(scale, adjust):
    s = series(np.array([1., 2., 3.]) * scale)
    bits_equal(s.ewm(span=3, adjust=adjust).corr(s).to_list(), [np.nan, 1., 1.])


def test_ewm_large_halflife_retains_small_positive_alpha():
    actual = series([0., 1e20]).ewm(halflife=1e20, adjust=False).mean().to_list()
    bits_equal(actual, [0., -np.expm1(-np.log(2.) / 1e20) * 1e20])


@pytest.mark.parametrize('kind', ['rolling', 'expanding'])
@pytest.mark.parametrize('method', ['sum', 'nunique'])
def test_empty_windows_with_zero_min_periods_return_identity(kind, method):
    s = series([np.nan, np.nan, 1.])
    w = s.rolling(2, min_periods=0) if kind == 'rolling' else s.expanding(min_periods=0)
    actual = getattr(w, method)().to_list()
    if method == 'sum':
        bits_equal(actual, [0., 0., 1.])
    else:
        assert actual == [0, 0, 1]


@pytest.mark.parametrize('method', ['mean', 'sum', 'var', 'std', 'cov', 'corr'])
@pytest.mark.parametrize('adjust', [True, False])
@pytest.mark.parametrize('ignore_na', [True, False])
def test_ewm_infinities_are_missing_and_state_recovers(method, adjust, ignore_na):
    if method == 'sum' and not adjust:
        with pytest.raises(ValueError, match='adjust=False'):
            series([1., np.inf]).ewm(alpha=.5, adjust=False).sum()
        return
    values = [1., np.inf, 3., -np.inf, 5., np.nan, 7.]
    vs, ps = series(values), pd.Series(values)
    kw = dict(alpha=.5, adjust=adjust, ignore_na=ignore_na)
    vw, pw = vs.ewm(**kw), ps.ewm(**kw)
    actual = getattr(vw, method)(vs) if method in ('cov', 'corr') else getattr(vw, method)()
    expected = getattr(pw, method)(ps) if method in ('cov', 'corr') else getattr(pw, method)()
    np.testing.assert_allclose(np.asarray(actual.to_list(), dtype=float), expected.to_numpy(), rtol=1e-15, equal_nan=True)


@pytest.mark.parametrize('kind,method,expected,pandas_expected', [
    ('rolling', 'max', [1., np.inf, np.inf, 3., 5.], [1., 1., 3., 3., 5.]),
    ('rolling', 'median', [1., np.inf, np.inf, np.nan, np.nan], [1., 1., 3., 3., 5.]),
    ('rolling', 'rank', [1., 2., 1., 1., 2.], [1., np.nan, 1., np.nan, 1.]),
    ('expanding', 'max', [1., np.inf, np.inf, np.inf, np.inf], [1., 1., 3., 3., 5.]),
    ('expanding', 'median', [1., np.inf, 3., 2., 3.], [1., 1., 2., 2., 3.]),
    ('expanding', 'rank', [1., 2., 2., 1., 4.], [1., np.nan, 2., np.nan, 3.]),
])
def test_window_infinity_value_policy(kind, method, expected, pandas_expected):
    values = [1., np.inf, 3., -np.inf, 5.]
    vs, ps = series(values), pd.Series(values)
    vw = vs.rolling(2, min_periods=1) if kind == 'rolling' else vs.expanding(1)
    pw = ps.rolling(2, min_periods=1) if kind == 'rolling' else ps.expanding(1)
    bits_equal(getattr(vw, method)().to_list(), expected)
    bits_equal(getattr(pw, method)().tolist(), pandas_expected)


def test_frame_replace_leaves_unmatched_dtypes_unchanged():
    f = volas.DataFrame({'a': [1, 2], 'b': ['one', 'two']})
    assert f.replace(1, 9)['b'].to_list() == ['one', 'two']
    assert f.replace('one', 'new')['a'].to_list() == [1, 2]
    assert f.replace(7, 'bad')['a'].to_list() == [1, 2]
    assert series([1., np.nan]).replace(np.nan, 9).to_list() == [1., 9.]


@pytest.mark.parametrize('sign', [1., -1.])
def test_ewm_correlation_of_oppositely_scaled_columns(sign):
    x, y = series([1e100, 2e100, 3e100]), series(np.array([1e-100, 2e-100, 3e-100]) * sign)
    np.testing.assert_allclose(np.asarray(x.ewm(span=3).corr(y).to_list(), dtype=float), [np.nan, sign, sign], rtol=1e-15, atol=0, equal_nan=True)


@pytest.mark.parametrize('kind', ['rolling', 'expanding'])
@pytest.mark.parametrize('method', ['mean', 'max', 'median', 'rank', 'var', 'std'])
def test_empty_windows_without_identity_stay_missing(kind, method):
    s = series([np.nan, np.nan])
    w = s.rolling(2, min_periods=0) if kind == 'rolling' else s.expanding(min_periods=0)
    bits_equal(getattr(w, method)().to_list(), [np.nan, np.nan])


@pytest.mark.parametrize('values', [[np.inf, -np.inf, np.nan, 2., 4.], [np.inf, -np.inf, np.nan]])
@pytest.mark.parametrize('min_periods', [0, 2, 4])
def test_ewm_infinities_do_not_count_toward_min_periods(values, min_periods):
    kw = dict(alpha=.25, min_periods=min_periods)
    bits_equal(series(values).ewm(**kw).mean().to_list(), pd.Series(values).ewm(**kw).mean().tolist())
