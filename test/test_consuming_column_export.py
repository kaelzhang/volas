"""All consuming exports share the Column owner's typed dispatch."""
import numpy as np
import pyarrow as pa
import pytest
import volas


@pytest.mark.parametrize("dtype,values,expected", [
    (pa.float64(), [1.0, 2.0], [1.0, 2.0]),
    (pa.float32(), [1.0, 2.0], [1.0, 2.0]),
    (pa.int64(), [1, 2], [1, 2]),
    (pa.int32(), [1, 2], [1, 2]),
    (pa.int64(), [1, None], [1.0, np.nan]),
    (pa.int32(), [1, None], [1.0, np.nan]),
    (pa.bool_(), [True, False], [True, False]),
    (pa.string(), ["x", None], ["x", None]),
    (pa.timestamp("ns"), [1, None], np.array([1, "NaT"], dtype="datetime64[ns]")),
])
def test_exec_consuming_export_preserves_values_and_missing(dtype, values, expected):
    frame = volas.DataFrame.from_arrow(pa.table({"x": pa.array(values, type=dtype)}))
    actual = frame.exec("x")
    if dtype == pa.string():
        assert actual.tolist() == expected
    else:
        np.testing.assert_array_equal(actual, expected)
    assert frame.shape == (2, 1)


def test_exec_consuming_export_rejects_unknown_boolean_truth():
    frame = volas.DataFrame.from_arrow(pa.table({"x": pa.array([True, None], type=pa.bool_())}))
    with pytest.raises(ValueError, match="missing"):
        frame.exec("x")
