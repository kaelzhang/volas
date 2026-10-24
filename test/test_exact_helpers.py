"""Every exact-comparison entry point must preserve zero signs and integer bits."""
import importlib

import numpy as np
import pytest


# (module, helper, returns a predicate, takes a diagnostic context)
HELPERS = [
    ('test_derivations', 'same', True, False),
    ('harness.observe', 'same_indicator', False, True),
    ('harness.observe', 'same_data', False, True),
    ('test_centred_moments', '_bits', False, True),
    ('test_centred_moments', '_bits_eq', False, True),
    ('test_resume_guards', '_bits', False, True),
    ('test_windowed_history', 'same', True, False),
    ('test_resume_extreme_values', '_bits', True, False),
    ('test_exact_window_sums', '_bits', False, True),
    ('test_window_recompute', '_bit_exact', False, True),
    ('test_missing_rows', '_bit_exact', False, True),
    ('test_forming_resume', '_bit_exact', False, True),
    ('test_tf_forming_states', '_bit_exact', False, True),
    ('test_warmup_boundary', '_bit_exact', False, True),
    ('test_append_replace', '_same', False, True),
    ('test_wrappers', '_bits_equal', False, False),
    ('test_harness_defects', '_same', True, False),
    ('test_mutation_parity', '_eq', False, False),
    ('test_typed_continuation', '_bits_equal', True, False),
    ('test_true_range_nonfinite', '_bits', False, False),
    ('test_smoothing_talib_order', '_assert_bits', False, True),
    ('test_directional_talib_order', '_assert_bits', False, True),
    ('test_foreign_memory', '_bit_exact', False, False),
    ('test_formula_conventions', '_same', False, False),
    ('test_audit_t14_window', '_exact', False, False),
    ('test_append_admission', 'same', True, False),
    ('test_directive_cache_name', '_same', True, False),
    ('test_directive_zero_sign', '_same_bits', True, False),
    ('test_continuation_reach', '_same', True, False),
    ('test_windowed_indexing', '_eq_values', False, False),
    ('test_talib_parity', '_ht', False, True),
    ('test_talib_parity', '_candle_pat', False, True),
    ('test_round_exact', '_bits_equal', True, False),
]


def _accepts(module, name, predicate, context, a, b):
    helper = getattr(importlib.import_module('test.' + module), name)
    args = [a, b]
    if context:
        args.append(0 if module == 'test_talib_parity' else 'helper self-test')
    kwargs = {'exact': True} if module == 'test_formula_conventions' else {}
    if module == 'test_audit_t14_window':
        import volas
        args = [volas.DataFrame({'x': a})['x'], volas.DataFrame({'x': b})['x']]
    mod = importlib.import_module('test.' + module)
    contraction = getattr(mod, '_TALIB_CONTRACTS', None)
    if module == 'test_talib_parity':
        mod._TALIB_CONTRACTS = False  # Exercise the promised exact branch.
    try:
        result = helper(*args, **kwargs)
    except AssertionError:
        return False
    finally:
        if module == 'test_talib_parity':
            mod._TALIB_CONTRACTS = contraction
    return bool(result) if predicate else True


@pytest.mark.parametrize('module,name,predicate,context', HELPERS,
                         ids=[m + '.' + n for m, n, _, _ in HELPERS])
@pytest.mark.parametrize('case', ['zero64', 'zero32', 'missing', 'integer', 'integer_same', 'bool_same', 'bool_different'])
def test_exact_helper_contract(module, name, predicate, context, case):
    if case.startswith('zero'):
        dtype = 'float32' if case == 'zero32' else 'float64'
        a, b, expected = np.array([0.0], dtype=dtype), np.array([-0.0], dtype=dtype), False
    elif case == 'missing':
        a, b, expected = np.array([np.nan]), np.array([np.nan]), True
    elif case.startswith('bool'):
        a, b = np.array([True]), np.array([case == 'bool_same'])
        expected = case == 'bool_same'
    else:
        a, b = np.array([2**53 + 1]), np.array([2**53 + (case == 'integer_same')])
        expected = case == 'integer_same'
    assert _accepts(module, name, predicate, context, a, b) == expected


@pytest.mark.parametrize('dtype', ['float16', 'float32', 'float64', 'longdouble'])
def test_native_float_bits(dtype):
    from test.exact import assert_exact, same_exact
    a = np.array([0.0, 1.0, np.nan, np.inf], dtype=dtype)
    b = a.copy()
    b[0] = -0.0
    assert not same_exact(a, b)
    with pytest.raises(AssertionError, match='bits differ'):
        assert_exact(a, b)
    assert_exact(a[::2], a.copy()[::2])


def test_masks_shapes_and_integer_promotions():
    from test.exact import assert_exact, same_exact
    with pytest.raises(AssertionError, match='missing masks differ'):
        assert_exact([np.nan, 0.0], [0.0, np.nan])
    assert not same_exact([1.0], [[1.0]])
    with pytest.raises(AssertionError, match='shape'):
        assert_exact([1.0], [[1.0]])
    assert not same_exact(np.array([2**63 + 1], dtype='uint64'), np.array([2**63 - 1], dtype='int64'))
    assert_exact(np.array([2**53 + 1], dtype='uint64'), np.array([2**53 + 1], dtype='int64'))
    assert_exact(np.array([True, False]), np.array([True, False]))
    assert not same_exact(np.array([True]), np.array([False]))
    a = np.array([np.nan, 1.0])
    b = a.copy()
    b.view('uint64')[0] = 0xfff8000000000001
    assert_exact(a, b)  # Payload and sign of a missing NaN do not matter.


def test_harness_tolerance_keeps_zero_signs_and_integer_precision():
    from test.harness.observe import Mismatch, same_indicator
    for a, b in [(np.array([0.0]), np.array([-0.0])),
                 (np.array([2**53 + 1]), np.array([2**53]))]:
        with pytest.raises(Mismatch):
            same_indicator(a, b, 'tolerance', tolerance=1e-9)
    same_indicator(np.array([1.0]), np.array([1.0 + 1e-10]), 'tolerance', tolerance=1e-9)


def test_object_missing_cells_keep_present_float_bits():
    import volas
    from test.exact import assert_exact, same_exact
    a = np.array([None, volas.NA, np.nan, 0.0, 2**53 + 1], dtype=object)
    b = np.array([np.nan, None, volas.NA, -0.0, 2**53 + 1], dtype=object)
    assert not same_exact(a, b)
    b[3] = 0.0
    assert_exact(a, b)
    b[4] = 2**53
    assert not same_exact(a, b)


@pytest.mark.parametrize('dtype,values', [('bool', [True, None, False]),
                                       ('int64', [2**53 + 1, None, 0])])
def test_missing_typed_series_preserve_their_cells(dtype, values):
    import volas
    from test.exact import assert_exact, same_exact
    a = volas.DataFrame({'x': values}, dtype=dtype)['x']
    b = volas.DataFrame({'x': values}, dtype=dtype)['x']
    assert_exact(a, b)
    changed = list(values)
    changed[0] = False if dtype == 'bool' else 2**53
    b = volas.DataFrame({'x': changed}, dtype=dtype)['x']
    assert not same_exact(a, b)
