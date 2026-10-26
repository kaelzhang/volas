"""A source cache is admitted as values when the target owns that data column."""
import numpy as np
import pytest
from volas import DataFrame


@pytest.mark.parametrize("bulk", [False, True])
def test_cached_source_values_are_plain_admitted_bars(bulk):
    start = np.datetime64("2020-01-01", "ns")
    source = DataFrame({"close": [10.0, 20.0, 30.0]},
                       index=start + np.array([15, 16, 17]) * np.timedelta64(1, "m"))
    source["ma:2"]
    before = source.to_numpy().copy()
    target = DataFrame({"close": [1.0], "ma:2": [2.0]},
                       index=np.array([start]), time_frame="15m")
    incoming = [source.iloc[1:3]] if bulk else [source.iloc[1:2], source.iloc[2:3]]
    for bars in incoming:
        target.append(bars)
    np.testing.assert_array_equal(target.to_numpy(), [[1.0, 2.0], [30.0, 25.0]])
    np.testing.assert_array_equal(source.to_numpy(), before)
