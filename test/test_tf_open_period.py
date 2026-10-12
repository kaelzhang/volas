"""The open period of a time-frame frame: the small frame of fine bars whose
cumulation is the forming row.

* It holds one bar per timestamp, in increasing order, however it was built — a
  re-sent bar in the bars handed to ``cumulate`` (or carried by ``copy`` /
  ``DataFrame(other)``) is never counted again when a later bar re-cumulates it.
* Every incoming bar is normalized once to the frame's data columns: a column the bar
  leaves out is a missing value, the same in every append form and on every path
  (fast fold, refold, rollover), identical to a batch ``cumulate``.
* A write that reaches the forming row (``df[col] = ...``, ``iloc`` / ``loc`` /
  ``iat`` / ``at``, a mask write, a cached column overwritten) makes the written row
  the period's bar of record: the open period's single entry, stamped with the
  period's first bar time. Re-labelled or re-typed copies (``DataFrame(df, index=...,
  dtype=...)``) are the time-frame frames of their new rows.
"""

import numpy as np
import pytest

import volas
from volas import DataFrame

T0 = np.datetime64("2024-01-01T09:00", "ns")
MINUTE = np.timedelta64(1, "m")
OHLCV = ("open", "high", "low", "close", "volume")


def frame(minutes, **columns):
    return DataFrame(columns, index=T0 + np.asarray(minutes) * MINUTE)


def bar(minute, o, h, l, c, v):
    return frame([minute], open=[o], high=[h], low=[l], close=[c], volume=[v])


def rows(df):
    return df.to_numpy().tolist()


def batch(ledger, tf="5m"):
    """The batch cumulation of an accepted ledger of fine bars ``[(minute, o, h, l, c, v)]``
    (a re-sent bar replaces the bar it repeats)."""
    kept = {}
    for b in ledger:
        kept[b[0]] = b
    bars = [kept[m] for m in sorted(kept)]
    cols = {name: [b[k + 1] for b in bars] for k, name in enumerate(OHLCV)}
    return rows(frame([b[0] for b in bars], **cols).cumulate(tf))


# --- one bar per timestamp ----------------------------------------------------------


def test_a_resend_after_cumulate_never_recounts_a_superseded_bar():
    ledger = [(0, 10.0, 12.0, 9.0, 11.0, 10.0), (1, 11.0, 30.0, 10.0, 12.0, 5.0),
              (1, 11.0, 13.0, 10.0, 12.5, 7.0)]
    fine = frame([b[0] for b in ledger], **{n: [b[k + 1] for b in ledger] for k, n in enumerate(OHLCV)})
    df = fine.cumulate("5m")
    assert rows(df) == [[10.0, 13.0, 9.0, 12.5, 17.0]]
    df.append(bar(2, 12.5, 13.0, 12.0, 12.8, 1.0))
    ledger.append((2, 12.5, 13.0, 12.0, 12.8, 1.0))
    assert rows(df) == batch(ledger) == [[10.0, 13.0, 9.0, 12.8, 18.0]]
    df.append(bar(2, 12.5, 13.0, 12.0, 12.9, 2.0))  # re-sent 09:02: refolds the period
    ledger.append((2, 12.5, 13.0, 12.0, 12.9, 2.0))
    assert rows(df) == batch(ledger) == [[10.0, 13.0, 9.0, 12.9, 19.0]]


@pytest.mark.parametrize("derive", ["cumulate", "copy", "constructor"])
def test_every_carrier_of_the_open_period_keeps_one_bar_per_timestamp(derive):
    # the re-sent bar is the last fed one, then the first, then re-sent again
    for minutes, volumes, want in (([0, 1, 1], [1.0, 10.0, 2.0], 3.0), ([0, 0, 1], [1.0, 10.0, 2.0], 12.0)):
        df = frame(minutes, volume=volumes).cumulate("5m")
        df = {"cumulate": df, "copy": df.copy(), "constructor": DataFrame(df)}[derive]
        assert df["volume"].to_list() == [want]
        df.append(frame([1], volume=[2.0]))  # an identical re-send of 09:01
        assert df["volume"].to_list() == [want]


# --- one normalized bar --------------------------------------------------------------


def _seeded(volume=10.0):
    return DataFrame(bar(0, 10.0, 12.0, 9.0, 11.0, volume), time_frame="5m")


@pytest.mark.parametrize("form", ["frame", "row", "dict"])
def test_a_bar_leaving_out_a_data_column_folds_it_as_missing_in_every_form(form):
    df = DataFrame(volas.DataFrame(
        {"t": [T0], "open": [10.0], "high": [12.0], "low": [9.0], "close": [11.0], "volume": [10.0]}
    ).set_index("t"), time_frame="5m")
    late = frame([1], open=[11.0], high=[13.0], low=[10.0], close=[12.0])
    other = {"frame": late, "row": late.iloc[0],
             "dict": {"t": T0 + MINUTE, "open": 11.0, "high": 13.0, "low": 10.0, "close": 12.0}}[form]
    df.append(other)
    # any missing input makes the aggregate missing, as in the batch cumulate
    assert rows(df)[0][:4] == [10.0, 13.0, 9.0, 12.0] and np.isnan(rows(df)[0][4])
    df.append(late)  # an identical re-send changes nothing
    assert np.isnan(rows(df)[0][4]) and rows(df)[0][:4] == [10.0, 13.0, 9.0, 12.0]


def test_a_rollover_bar_leaving_out_a_column_does_not_fix_the_period_schema():
    df = DataFrame(frame([0], close=[1.0], volume=[10.0]), time_frame="5m")
    df.append(frame([5], close=[2.0]))  # rollover bar without volume
    assert np.isnan(rows(df)[1][1])
    df.append(frame([5], close=[3.0], volume=[20.0]))  # its corrected re-send
    assert rows(df) == [[1.0, 10.0], [3.0, 20.0]]
    high = DataFrame(frame([0], high=[1.0], close=[1.0]), time_frame="5m")
    high.append(frame([5], close=[2.0]))
    high.append(frame([5], high=[20.0], close=[3.0]))
    high.append(frame([5], high=[5.0], close=[3.0]))  # a corrected re-send replaces 20
    assert rows(high) == [[1.0, 1.0], [5.0, 3.0]]


@pytest.mark.parametrize("tf", [None, "5m"])
def test_the_dict_form_pads_a_left_out_column_like_every_other_form(tf):
    base = volas.DataFrame({"t": [T0], "close": [1.0], "volume": [10.0]}).set_index("t")
    df = DataFrame(base, time_frame=tf) if tf else base
    df.append({"t": T0 + 5 * MINUTE, "close": 2.0})
    got = rows(df)
    assert got[0] == [1.0, 10.0] and got[1][0] == 2.0 and np.isnan(got[1][1])


# --- writes reaching the forming row ---------------------------------------------------


def test_an_added_column_reaches_the_open_period():
    df = _seeded()
    df["signal"] = 0.0
    df.append(frame([1], open=[11.0], high=[13.0], low=[10.0], close=[12.0], volume=[5.0], signal=[1.0]))
    assert rows(df) == [[10.0, 13.0, 9.0, 12.0, 15.0, 1.0]]
    df.append(frame([1], open=[11.0], high=[13.0], low=[10.0], close=[12.0], volume=[6.0], signal=[2.0]))
    assert rows(df) == [[10.0, 13.0, 9.0, 12.0, 16.0, 2.0]]


def test_a_replaced_column_is_not_undone_by_a_resend():
    df = _seeded()
    df["volume"] = [100.0]
    df.append(bar(1, 11.0, 13.0, 10.0, 12.0, 5.0))
    assert df["volume"].to_list() == [105.0]
    df.append(bar(1, 11.0, 13.0, 10.0, 12.0, 5.0))  # an identical re-send
    assert df["volume"].to_list() == [105.0]


@pytest.mark.parametrize("via", ["iloc", "loc", "iat", "at", "mask"])
def test_a_cell_write_to_the_forming_row_reaches_the_open_period(via):
    df = _seeded()
    if via == "mask":
        df[df.isna() == df.isna()] = 100.0
    else:
        key = (0, 4) if via in ("iloc", "iat") else (T0, "volume")
        getattr(df, via)[key] = 100.0
    df.append(bar(1, 11.0, 13.0, 10.0, 12.0, 5.0))
    df.append(bar(1, 11.0, 13.0, 10.0, 12.0, 5.0))
    assert df["volume"].to_list() == [105.0]


def test_a_write_to_a_closed_row_keeps_the_open_period_exact():
    df = DataFrame(frame([0, 5], close=[1.0, 2.0], volume=[10.0, 20.0]), time_frame="5m")
    df.append(frame([6], close=[3.0], volume=[1.0]))
    df.iat[0, 1] = 99.0  # the closed 09:00 row
    df.append(frame([6], close=[3.0], volume=[4.0]))  # re-send of 09:06 still replaces it
    assert rows(df) == [[1.0, 99.0], [3.0, 24.0]]


def test_a_forming_write_over_several_fine_bars_becomes_the_period_bar_of_record():
    df = DataFrame(frame([0], close=[1.0], volume=[10.0]), time_frame="5m")
    df.append(frame([6], close=[2.0], volume=[1.0]))
    df.append(frame([7], close=[3.0], volume=[2.0]))
    df["volume"] = [10.0, 50.0]
    # the written row is the open period's single entry, at the period's first bar time
    # (09:06): a re-send of 09:06 replaces it whole, a later bar folds into it
    df.append(frame([8], close=[4.0], volume=[5.0]))
    assert rows(df) == [[1.0, 10.0], [4.0, 55.0]]
    df.append(frame([8], close=[4.5], volume=[6.0]))
    assert rows(df) == [[1.0, 10.0], [4.5, 56.0]]


def test_a_cached_column_overwritten_becomes_data_of_the_open_period():
    df = DataFrame(frame([0, 5], close=[1.0, 2.0], volume=[1.0, 2.0]), time_frame="5m")
    df["ma:2"]
    df.iat[0, 2] = -1.0  # a closed cell of the cached column: the column becomes data
    assert not df.is_computed("ma:2")
    df.append(frame([6], close=[3.0], volume=[1.0], **{"ma:2": [7.0]}))
    assert rows(df) == [[1.0, 1.0, -1.0], [3.0, 3.0, 7.0]]


def test_a_write_the_aggregator_cannot_fold_is_refused_unchanged():
    df = DataFrame(frame([0], close=[1.0], x=[5.0]), time_frame="5m", cumulators={"x": "max"})
    with pytest.raises(TypeError, match="cannot max the str column"):
        df["x"] = ["a"]
    with pytest.raises(TypeError, match="cannot max the str column"):
        df["x"] = "a"
    assert rows(df) == [[1.0, 5.0]]
    df.append(frame([1], close=[2.0], x=[3.0]))
    df.append(frame([1], close=[2.0], x=[7.0]))
    assert rows(df) == [[2.0, 7.0]]


def test_a_column_added_to_an_empty_frame_is_in_its_first_bar():
    df = DataFrame(frame([], close=np.array([], dtype=float)), time_frame="5m")
    df["x"] = np.array([], dtype=float)
    df.append(frame([1], close=[2.0], x=[3.0]))
    df.append(frame([2], close=[4.0]))
    assert df.columns == ["close", "x"]
    assert rows(df)[0][0] == 4.0 and np.isnan(rows(df)[0][1])


# --- derived frames ---------------------------------------------------------------------


def test_a_retyped_copy_is_the_time_frame_frame_of_its_rows():
    int_volume = DataFrame(frame([0], close=[1.0], volume=np.array([10], dtype=np.int64)), time_frame="5m")
    df = DataFrame(int_volume, dtype="float64")
    df.append(frame([1], close=[2.0], volume=[5.5]))
    assert rows(df) == [[2.0, 15.5]]
    with pytest.raises(TypeError, match="cannot max the str column"):
        DataFrame(DataFrame(frame([0], close=[1.0]), time_frame="5m", cumulators={"close": "max"}),
                  dtype="str")


def test_a_relabelled_copy_is_the_time_frame_frame_of_its_rows():
    df = DataFrame(DataFrame(frame([0], close=[1.0], volume=[10.0]), time_frame="5m"),
                   index=np.array(["2024-01-01T10:00"], dtype="datetime64[ns]"))
    df.append(frame([61], close=[2.0], volume=[5.0]))
    df.append(frame([61], close=[2.0], volume=[5.0]))
    assert np.asarray(df.index).astype(str).tolist() == ["2024-01-01T10:00:00.000000000"]
    assert rows(df) == [[2.0, 15.0]]


def test_a_projection_keeps_the_fine_bars_of_its_data_columns():
    df = DataFrame(frame([0], close=[1.0], high=[1.0], volume=[10.0]), time_frame="5m")
    df.append(frame([1], close=[2.0], high=[5.0], volume=[1.0]))
    df.append(frame([2], close=[3.0], high=[2.0], volume=[2.0]))
    p = DataFrame(df, columns=["close", "high"])
    p.append(frame([2], close=[3.0], high=[1.0]))  # a re-send of 09:02 refolds every bar
    assert rows(p) == [[3.0, 5.0]]
    # a cached column projected becomes data the fine bars never carried: the forming
    # row (refreshed) is the open period's bar of record, at the period's first bar time
    two = DataFrame(frame([0, 5], close=[1.0, 3.0]), time_frame="5m")
    two["ma:2"]
    two.append(frame([6], close=[5.0]))
    q = DataFrame(two, columns=["close", "ma:2"])
    assert not q.is_computed("ma:2") and rows(q)[1] == [5.0, 3.0]
    q.append(frame([7], close=[6.0], **{"ma:2": [8.0]}))
    assert rows(q)[1] == [6.0, 8.0]

