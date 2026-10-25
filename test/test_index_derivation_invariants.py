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


@pytest.mark.parametrize("method,values,expected", [
    ("nlargest", [30, 10, 20], [30]),
    ("nsmallest", [30, 10, 20], [10]),
])
@pytest.mark.parametrize("n", [0, 1])
def test_series_top_n_checks_only_final_datetime_selection(method, values, expected, n):
    labels = volas.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"])
    s = volas.DataFrame({"value": values}, index=labels.to_numpy())["value"]
    result = getattr(s, method)(n)
    assert result.to_list() == (expected if n else [])
    assert result.dtype == s.dtype


@pytest.mark.parametrize("method,values,expected", [
    ("nlargest", [40, 30, 10, 20], [40, 30]),
    ("nsmallest", [10, 20, 40, 30], [10, 20]),
])
def test_series_top_n_legal_prefix_of_illegal_full_sort(method, values, expected):
    labels = volas.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"])
    s = volas.DataFrame({"value": values}, index=labels.to_numpy())["value"]
    result = getattr(s, method)(2)
    assert result.to_list() == expected
    assert result.index.tolist() == s.index[:2].tolist()
    with pytest.raises(ValueError, match="monotonic"):
        getattr(s, method)(4)


@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_series_top_n_drops_missing_before_truncation(method):
    s = volas.DataFrame({"value": [None, 2, None, 1]})["value"]
    assert getattr(s, method)(10).to_list() == ([2, 1] if method == "nlargest" else [1, 2])
