import numpy as np
import pytest

from volas import DataFrame


@pytest.mark.parametrize('data', [[], [1., 3.]])
@pytest.mark.parametrize('center', [False, True])
def test_window_parameters_are_checked_when_no_window_can_emit(data, center):
    rolling = DataFrame({"x": np.asarray(data, dtype=float)})["x"].rolling(200, min_periods=200, center=center)
    with pytest.raises(ValueError, match='quantile q must be in'):
        rolling.quantile(float('inf'))
    with pytest.raises(ValueError, match='interpolation must be one of'):
        rolling.quantile(0.5, interpolation='bogus')
    with pytest.raises(ValueError, match='rank method must be'):
        rolling.rank(method='bogus')
