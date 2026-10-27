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
@pytest.mark.parametrize("values", [[3.0, float("nan"), 1.0], [None, 2, None, 1], [None, None], [float("nan"), float("nan")]])
@pytest.mark.parametrize("n", [1, 3, 10])
def test_series_top_n_keeps_missing_last(method, values, n):
    import pandas as pd

    s = volas.DataFrame({"value": values})["value"]
    result = getattr(s, method)(n)
    expected = getattr(pd.Series(values, dtype="Float64"), method)(n)
    assert result.dtype == s.dtype
    assert result.index.tolist() == expected.index.tolist()
    pd.testing.assert_series_equal(pd.Series([None if value is volas.NA else value for value in result.to_list()], dtype="Float64"), expected.reset_index(drop=True))


def test_readme_grouped_reductions_describe_rowgroups():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("README.md", "README.zh-CN.md", "crates/volas/README.md", "crates/volas/README.zh-CN.md"):
        text = (root / name).read_text()
        assert "RowGroups::new" in text and "reduce_with_spare" in text
        assert "repeatable (`Clone`) group iterators" not in text
        assert "分组迭代器须可重复遍历" not in text


def test_range_append_capacity_error_keeps_index_columns_and_cache():
    import pandas as pd

    source = volas.DataFrame.from_pandas(pd.DataFrame(index=pd.RangeIndex(1 << 60)))
    target = volas.DataFrame({"x": [1.0]})
    target["ma:1@x"]
    before = target.copy()
    with pytest.raises(OverflowError, match="capacity"):
        target.append(source)
    assert target.shape == before.shape == (1, 2)
    assert target.equals(before)
    assert target.index.tolist() == [0]
    assert target["x"].to_list() == [1.0]
    assert target["ma:1@x"].to_list() == [1.0]
    with pytest.raises(KeyError):
        target["x"].loc[1]


@pytest.mark.parametrize("method", ["nlargest", "nsmallest"])
def test_top_n_datetime_keeps_stable_all_missing_rows(method):
    labels = volas.to_datetime(["2021-01-01", "2021-01-02"])
    s = volas.DataFrame({"value": [None, None]}, index=labels.to_numpy())["value"]
    result = getattr(s, method)(10)
    assert result.index.tolist() == s.index.tolist()
    assert result.isna().to_list() == [True, True]


def test_top_n_checks_final_missing_row_labels_too():
    labels = volas.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"])
    s = volas.DataFrame({"value": [3, None, 1]}, index=labels.to_numpy())["value"]
    assert s.nlargest(2).to_list() == [3, 1]
    with pytest.raises(ValueError, match="monotonic"):
        s.nlargest(3)
