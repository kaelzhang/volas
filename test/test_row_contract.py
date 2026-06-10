"""Row contract (R2 / V5, R6 / V7): Row.to_numpy() is only valid for an
all-numeric row (a str/datetime cell can't go to float64 without a silent NaN);
an index-label scalar under a DatetimeIndex is a typed volas.Timestamp.
(R5 / V6 — the Row repr printing no dtype line — is covered separately.)"""

import numpy as np
import pytest
from volas import DataFrame


def test_row_to_numpy_mixed_raises():
    # V5: a row with a str column must not silently export str -> float64 NaN
    df = DataFrame({'close': [1.0, 2.0], 'sym': ['a', 'b']})
    with pytest.raises(Exception):
        df.iloc[0].to_numpy()


def test_row_to_numpy_all_numeric_works():
    df = DataFrame({'open': [1.0, 2.0], 'close': [3.0, 4.0], 'vol': [10, 20]})
    arr = df.iloc[0].to_numpy()
    assert np.asarray(arr).ravel().tolist() == [1.0, 3.0, 10.0]


def test_index_label_scalar_is_timestamp_on_datetime_index():
    # V7 / R6: row.name / idxmax / idxmin return a volas.Timestamp (not a str)
    # under a DatetimeIndex, so .loc/.at round-trip is typed
    df = DataFrame({'close': [3.0, 1.0, 2.0],
                    't': np.array(['2021-01-01', '2021-01-02', '2021-01-03'],
                                  dtype='datetime64[ns]')}).set_index('t')
    assert type(df.iloc[1].name).__name__ == 'Timestamp'
    assert type(df['close'].idxmax()).__name__ == 'Timestamp'
    assert type(df['close'].idxmin()).__name__ == 'Timestamp'
