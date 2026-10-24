"""Non-finite aggregation inputs are missing; raw storage and IEEE sums survive."""
import numpy as np
import pandas as pd
import pytest
import volas

from .exact import assert_exact


def frame(values, start=0, dtype='float64', **kwargs):
    index = pd.date_range('2024-01-01', periods=start + len(values), freq='min')[start:]
    return volas.DataFrame({'value': np.asarray(values, dtype=dtype)}, index=index, **kwargs)


@pytest.mark.parametrize('dtype', ['float64', 'float32'])
@pytest.mark.parametrize('agg', ['first', 'last', 'max', 'min', 'sum'])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('position', [0, 1, 2])
def test_every_aggregation_path_treats_nonfinite_inputs_as_missing(dtype, agg, bad, position):
    values = [-0.0, 2.0, 3.0]
    values[position] = bad
    spec = {'value': agg}
    raw = frame(values, dtype=dtype)
    # Raw storage remains an exact snapshot, including infinities and signed zeros.
    assert_exact(raw['value'].to_numpy(), np.asarray(values, dtype=dtype))
    assert np.isnan(raw.cumulate('5m', cumulators=spec)['value'].to_numpy()[0])
    live = frame([], dtype=dtype).cumulate('5m', cumulators=spec)
    for i, value in enumerate(values):
        live.append(frame([value], i, dtype))
        expected = raw.iloc[:i + 1].cumulate('5m', cumulators=spec)['value'].to_numpy()
        assert_exact(live['value'].to_numpy(), expected)
        # Re-send the current fine bar, then restore: exercises both checkpoints.
        live.append(frame([bad], i, dtype))
        assert np.isnan(live['value'].to_numpy()[0])
        live.append(frame([value], i, dtype))
        assert_exact(live['value'].to_numpy(), expected)
    live.append(frame([4.0], 5, dtype))
    assert_exact(live['value'].to_numpy(), np.asarray([np.nan, 4.0], dtype=dtype))


@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_finite_sum_overflow_is_an_ieee_intermediate(dtype):
    maximum = np.finfo(dtype).max
    spec = {'value': 'sum'}
    raw = frame([maximum, maximum, -maximum], dtype=dtype)
    live = raw.iloc[:2].cumulate('5m', cumulators=spec)
    live.append(frame([-maximum], 2, dtype))
    assert np.isposinf(live['value'].to_numpy()[0])
    assert_exact(live['value'].to_numpy(), raw.cumulate('5m', cumulators=spec)['value'].to_numpy())
    live.append(frame([np.inf], 2, dtype))
    assert np.isnan(live['value'].to_numpy()[0])
    live.append(frame([-maximum], 2, dtype))
    assert np.isposinf(live['value'].to_numpy()[0])


@pytest.mark.parametrize('agg', ['first', 'last', 'max', 'min', 'sum'])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('position', [0, 1, 2])
def test_independent_aggregation_oracle_uses_the_nonfinite_input_contract(agg, bad, position):
    from .harness.ledger import _aggregate
    values = [1.0, 2.0, 3.0]
    values[position] = float(bad)
    assert _aggregate(agg, values) is None


def test_independent_aggregation_oracle_keeps_ieee_intermediates():
    from .harness.ledger import _aggregate
    maximum = float(np.finfo('float64').max)
    assert np.isposinf(_aggregate('sum', [maximum, maximum, -maximum]))


@pytest.mark.parametrize('dtype', ['float64', 'float32'])
@pytest.mark.parametrize('agg', ['first', 'last', 'max', 'min', 'sum'])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('entry', ['seed', 'write'])
@pytest.mark.parametrize('copy', [False, True])
def test_unreduced_tf_input_enters_the_aggregation_owner(dtype, agg, bad, entry, copy):
    spec = {'value': agg}
    if entry == 'seed':
        live = frame([bad], dtype=dtype, time_frame='5m', cumulators=spec)
    else:
        live = frame([1.0], dtype=dtype).cumulate('5m', cumulators=spec)
        live.iloc[0, 0] = bad
    if copy:
        live = live.cumulate('5m', cumulators=spec)
    raw = frame([bad, 2.0], dtype=dtype)
    live.append(frame([2.0], 1, dtype))
    expected = raw.cumulate('5m', cumulators=spec)['value'].to_numpy()
    assert_exact(live['value'].to_numpy(), expected)
    live.append(frame([2.0], 1, dtype))
    assert_exact(live['value'].to_numpy(), expected)
    # Correct the initial input before appending: a one-bar resend replaces it.
    repaired = frame([bad], dtype=dtype, time_frame='5m', cumulators=spec)
    repaired.append(frame([-0.0], 0, dtype))
    repaired.append(frame([2.0], 1, dtype))
    assert_exact(repaired['value'].to_numpy(),
                 frame([-0.0, 2.0], dtype=dtype).cumulate('5m', cumulators=spec)['value'].to_numpy())


@pytest.mark.parametrize('dtype', ['float64', 'float32'])
@pytest.mark.parametrize('entry', ['seed', 'write'])
@pytest.mark.parametrize('copy', [False, True])
def test_unreduced_finite_seed_keeps_sum_overflow(dtype, entry, copy):
    maximum = np.finfo(dtype).max
    spec = {'value': 'sum'}
    if entry == 'seed':
        live = frame([maximum], dtype=dtype, time_frame='5m', cumulators=spec)
    else:
        live = frame([1.0], dtype=dtype).cumulate('5m', cumulators=spec)
        live.iloc[0, 0] = float(maximum)
    if copy:
        live = live.cumulate('5m', cumulators=spec)
    for minute, value in [(1, maximum), (2, -maximum), (2, -maximum)]:
        live.append(frame([value], minute, dtype))
        raw = frame([maximum, maximum, -maximum][:minute + 1], dtype=dtype)
        assert_exact(live['value'].to_numpy(), raw.cumulate('5m', cumulators=spec)['value'].to_numpy())
        assert np.isposinf(live['value'].to_numpy()[0])
