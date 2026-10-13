"""Release bounds for arithmetic and compiled datetime batch owners."""
import numpy as np
import pytest

import volas
from test.test_cumulation_cost import best_us


@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_homogeneous_arithmetic_uses_a_single_output_loop(dtype, cost_guard):
    values = np.arange(300_000, dtype=dtype) + 1
    a, b = volas.DataFrame({'v': values})['v'], volas.DataFrame({'v': values + 1})['v']
    add, div = cost_guard.pair(lambda: best_us(lambda: a + b, 5),
                               lambda: best_us(lambda: a / b, 5))
    cost_guard(add, div * 3)


def test_explicit_parse_compiles_the_format_once(cost_guard):
    text = volas.DataFrame({'t': np.full(50_000, '2024-01-07 03:04:05')})['t']
    explicit, inferred = cost_guard.pair(
        lambda: best_us(lambda: volas.to_datetime(text, format='%F %T'), 1),
        lambda: best_us(lambda: volas.to_datetime(text), 1))
    cost_guard(explicit, inferred * 3)


def test_numeric_formatting_has_no_per_field_strings(cost_guard):
    series = volas.to_datetime(np.arange(50_000, dtype=np.int64) * 60_000_000_000)
    fields, literal = cost_guard.pair(
        lambda: best_us(lambda: series.dt.strftime('%F %T'), 1),
        lambda: best_us(lambda: series.dt.strftime('1970-01-01 00:00:00'), 1))
    cost_guard(fields, literal * 5)
