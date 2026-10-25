"""Checked index derivations surface invalid labels as Python errors."""

import pytest

import volas


def _dated():
    labels = volas.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"])
    return volas.DataFrame({"value": [30, 20, 10]}, index=labels.to_numpy())


@pytest.mark.parametrize("series,operation", [(False, "slice"), (True, "slice"),
                                               (True, "sort_values"), (False, "sort_index"),
                                               (True, "sort_index")])
def test_datetime_derivation_rejects_decreasing_labels(series, operation):
    frame = _dated()
    source = frame["value"] if series else frame
    with pytest.raises(ValueError, match="monotonic"):
        if operation == "slice":
            source.iloc[::-1]
        elif operation == "sort_values":
            source.sort_values()
        else:
            source.sort_index(ascending=False)
    assert frame["value"].to_list() == [30, 20, 10]


@pytest.mark.parametrize("labels", [[1, 2, 3], ["a", "b", "c"]])
def test_frame_gather_rejects_duplicate_labels(labels):
    frame = volas.DataFrame({"value": [30, 20, 10]}, index=labels)
    with pytest.raises(ValueError, match="duplicate labels"):
        frame.iloc[[0, 0]]
