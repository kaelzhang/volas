"""The open period of a time-frame frame: the small frame of fine bars whose
cumulation is the forming row.

* It holds one bar per timestamp, in increasing order, however it was built — a
  re-sent bar in the bars handed to ``cumulate`` (or carried by ``copy`` /
  ``DataFrame(other)``) is never counted again when a later bar re-cumulates it.
* Every incoming bar is normalized once to the frame's data columns: a column the bar
  leaves out is a missing value, the same in every append form and on every path
  (fast fold, refold, rollover), identical to a batch ``cumulate``.
* A write never yields a wrong number: a column that becomes data (added, or a cached
  column overwritten) enters the fine bars as missing, so the next fold makes its
  forming cell missing; a new forming value of a column the bars carry rewrites the
  period's bar when it holds one, and is refused when it holds several; writes to
  closed rows leave the open period as it is. Re-labelled or re-typed copies
  (``DataFrame(df, index=..., dtype=...)``) are the time-frame frames of their new rows.
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


def test_a_bar_leaving_out_a_column_is_completed_around_cached_columns():
    df = DataFrame(frame([0, 5], close=[1.0, 2.0], volume=[1.0, 2.0]), time_frame="5m")
    assert df["ma:2"].to_list()[1] == 1.5
    df.append(frame([6], close=[4.0]))  # no volume; the cached ma:2 is not data
    assert df["ma:2"].to_list()[1] == 2.5
    assert rows(df)[1][0] == 4.0 and np.isnan(rows(df)[1][1])


@pytest.mark.parametrize("tf", [None, "5m"])
def test_the_dict_form_pads_a_left_out_column_like_every_other_form(tf):
    base = volas.DataFrame({"t": [T0], "close": [1.0], "volume": [10.0]}).set_index("t")
    df = DataFrame(base, time_frame=tf) if tf else base
    df.append({"t": T0 + 5 * MINUTE, "close": 2.0})
    got = rows(df)
    assert got[0] == [1.0, 10.0] and got[1][0] == 2.0 and np.isnan(got[1][1])


# --- writes reaching the forming row ---------------------------------------------------


def _live():
    """A 5m frame whose open period holds the three fine bars 09:05, 09:06, 09:07."""
    df = DataFrame(frame([0], close=[1.0], volume=[10.0]), time_frame="5m")
    for m, c, v in ((5, 2.0, 1.0), (6, 3.0, 2.0), (7, 4.0, 3.0)):
        df.append(frame([m], close=[c], volume=[v]))
    return df


def test_an_added_column_enters_the_fine_bars_as_missing():
    # a derived column on a cumulate() result (several fine bars in the open period)
    df = _live()
    df["signal"] = [0.5, 0.7]
    assert rows(df)[1] == [4.0, 6.0, 0.7]  # as written, until the next fold
    df.append(frame([8], close=[5.0], volume=[4.0], signal=[0.9]))
    # the earlier fine bars never carried it: the forming value is missing, never 0.9
    assert rows(df)[1][:2] == [5.0, 10.0] and np.isnan(rows(df)[1][2])
    df.append(frame([8], close=[5.0], volume=[4.0], signal=[0.9]))  # an identical re-send
    assert rows(df)[1][:2] == [5.0, 10.0] and np.isnan(rows(df)[1][2])
    assert rows(df)[0] == [1.0, 10.0, 0.5]


def test_an_added_column_keeps_its_value_in_a_row_that_closes_unfolded():
    df = _live()
    df["signal"] = [0.5, 0.7]
    df.append(frame([10], close=[6.0], volume=[1.0], signal=[0.2]))  # rolls over
    assert rows(df) == [[1.0, 10.0, 0.5], [4.0, 6.0, 0.7], [6.0, 1.0, 0.2]]


def test_a_value_written_over_one_bar_rewrites_it_and_a_resend_replaces_it():
    df = DataFrame(frame([0], close=[1.0], volume=[10.0]), time_frame="5m")
    df.append(frame([6], close=[2.0], volume=[1.0]))  # the period's one bar: 09:06
    df["volume"] = [10.0, 50.0]
    df.append(frame([7], close=[3.0], volume=[5.0]))
    assert rows(df) == [[1.0, 10.0], [3.0, 55.0]]
    df.append(frame([7], close=[3.5], volume=[6.0]))  # a re-send: no double count
    assert rows(df) == [[1.0, 10.0], [3.5, 56.0]]


def test_a_rewritten_bar_is_replaced_by_its_resend():
    df = DataFrame(frame([0], close=[1.0], volume=[10.0]), time_frame="5m")
    df.append(frame([6], close=[2.0], volume=[1.0]))
    df.iat[1, 1] = 50.0
    df.append(frame([6], close=[2.0], volume=[3.0]))  # the exchange re-sends 09:06
    assert rows(df) == [[1.0, 10.0], [2.0, 3.0]]  # replaced, not 50 + 3


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


@pytest.mark.parametrize("via", ["setitem", "iat", "loc", "mask"])
def test_a_value_written_over_several_fine_bars_is_refused_unchanged(via):
    df = _live()
    before = rows(df)
    write = {
        "setitem": lambda: df.__setitem__("volume", [10.0, 50.0]),
        "iat": lambda: df.iat.__setitem__((1, 1), 50.0),
        "loc": lambda: df.loc.__setitem__((T0 + 5 * MINUTE, "volume"), 50.0),
        "mask": lambda: df.__setitem__(df["close"] > 0, 7.0),
    }[via]
    with pytest.raises(ValueError, match="cumulation of the 3 fine bars"):
        write()
    assert rows(df) == before
    df.append(frame([7], close=[4.5], volume=[2.5]))  # a re-send still replaces exactly
    assert rows(df) == [[1.0, 10.0], [4.5, 5.5]]


def test_writes_that_change_no_forming_value_are_allowed_over_several_bars():
    df = _live()
    df["volume"] = [99.0, 6.0]  # the forming volume unchanged; the closed one rewritten
    df.iat[1, 0] = 4.0  # the same forming close
    df.append(frame([8], close=[5.0], volume=[4.0]))
    assert rows(df) == [[1.0, 99.0], [5.0, 10.0]]


def test_a_cached_column_overwritten_becomes_data_missing_in_the_fine_bars():
    df = DataFrame(frame([0, 5], close=[1.0, 2.0], volume=[1.0, 2.0]), time_frame="5m")
    df["ma:2"]
    df.iat[0, 2] = -1.0  # a closed cell of the cached column: the column becomes data
    assert not df.is_computed("ma:2")
    df.append(frame([6], close=[3.0], volume=[1.0], **{"ma:2": [7.0]}))
    got = rows(df)
    assert got[0] == [1.0, 1.0, -1.0] and got[1][:2] == [3.0, 3.0] and np.isnan(got[1][2])


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
    # a cached column projected becomes data the fine bars never carried: they hold it
    # as missing, so the next fold makes its forming cell missing
    two = DataFrame(frame([0, 5], close=[1.0, 3.0]), time_frame="5m")
    two["ma:2"]
    two.append(frame([6], close=[5.0]))
    q = DataFrame(two, columns=["close", "ma:2"])
    assert not q.is_computed("ma:2") and rows(q)[1] == [5.0, 3.0]
    q.append(frame([7], close=[6.0], **{"ma:2": [8.0]}))
    assert rows(q)[1][0] == 6.0 and np.isnan(rows(q)[1][1])


def test_cumulate_does_not_carry_a_cached_column_as_data():
    fine = frame([0, 1, 5, 6], close=[1.0, 3.0, 5.0, 7.0])
    fine["ma:2"]
    coarse = fine.cumulate("5m")
    assert coarse.columns == ["close"]
    assert coarse["ma:2"].to_list()[1] == 5.0 and np.isnan(coarse["ma:2"].to_list()[0])


# --- review: every write judged by the forming cell it leaves ----------------------------


@pytest.mark.parametrize("via", ["iloc", "loc"])
def test_a_duplicate_position_write_is_judged_by_the_value_it_leaves(via):
    df = _live()
    before = rows(df)
    with pytest.raises(ValueError, match="cumulation of the 3 fine bars"):
        if via == "iloc":
            df.iloc[[1, 1], 1] = [6.0, 50.0]  # the last write wins: 50, not 6
        else:
            df.loc[[T0 + 5 * MINUTE, T0 + 5 * MINUTE], "volume"] = [6.0, 50.0]
    assert rows(df) == before
    df.iloc[[1, 1], 1] = [50.0, 6.0]  # leaves the forming volume as it was
    df.append(frame([8], close=[5.0], volume=[4.0]))
    assert rows(df) == [[1.0, 10.0], [5.0, 10.0]]


def test_a_duplicate_position_write_over_one_bar_rewrites_it_with_the_last_value():
    df = DataFrame(frame([0], close=[1.0], volume=[1.0]), time_frame="5m")
    df.iloc[[-1, 0], 1] = [3.0, 50.0]
    df.append(frame([1], close=[2.0], volume=[3.0]))
    assert rows(df) == [[2.0, 53.0]]


def test_a_column_retyped_with_the_same_values_keeps_the_fine_bars():
    df = DataFrame(frame([0], close=[1.0], volume=np.array([10], dtype=np.int64)), time_frame="5m")
    df.append(frame([1], close=[2.0], volume=np.array([5], dtype=np.int64)))
    df.append(frame([2], close=[3.0], volume=np.array([7], dtype=np.int64)))
    df["volume"] = np.array([22.0])  # float64, the same value
    df.append(frame([2], close=[3.0], volume=[8.0]))  # a re-send of 09:02 refolds every bar
    assert rows(df) == [[3.0, 23.0]]


def test_a_relabelled_copy_is_checked_like_a_time_frame_seed():
    base = DataFrame(frame([0, 5], close=[1.0, 2.0]), time_frame="5m")
    with pytest.raises(ValueError, match="monotonic"):
        DataFrame(base, index=np.array(["2024-01-01T10:00", "2024-01-01T09:00"], dtype="datetime64[ns]"))
    with pytest.raises(ValueError, match="without NaT"):
        DataFrame(base, index=np.array(["2024-01-01T10:00", "NaT"], dtype="datetime64[ns]"))
    with pytest.raises(ValueError, match="requires a DatetimeIndex"):
        DataFrame(base, index=[1, 2])


# --- a write that cannot change the forming row leaves the open period as it is -----


@pytest.mark.parametrize("via", ["iloc", "loc", "iat", "at", "mask", "rowmask", "cellmask"])
def test_a_write_to_closed_rows_only_leaves_a_several_bar_period_exact(via):
    df = _live()  # 09:00 closed; 09:05-09:07 forming
    write = {
        "iloc": lambda: df.iloc.__setitem__((0, 1), 99.0),
        "loc": lambda: df.loc.__setitem__((T0, "volume"), 99.0),
        "iat": lambda: df.iat.__setitem__((0, 1), 99.0),
        "at": lambda: df.at.__setitem__((T0, "volume"), 99.0),
        "mask": lambda: df.loc.__setitem__((np.array([True, False]), "volume"), 99.0),
        "rowmask": lambda: df.__setitem__(np.array([True, False]), 99.0),
        "cellmask": lambda: df.__setitem__(df == 1.0, 99.0),  # the closed close only
    }[via]
    write()
    closed_row = {"rowmask": [99.0, 99.0], "cellmask": [99.0, 10.0]}.get(via, [1.0, 99.0])
    assert rows(df) == [closed_row, [4.0, 6.0]]
    df.append(frame([7], close=[4.5], volume=[2.5]))  # a re-send still replaces exactly
    df.append(frame([8], close=[5.0], volume=[1.0]))
    assert rows(df) == [closed_row, [5.0, 6.5]]


def test_a_closed_write_to_a_cached_column_still_turns_it_into_data():
    # the batch writes no forming cell, but a cached column: the data columns change, so
    # the next fold makes its forming cell missing, as for any column that becomes data
    df = DataFrame(frame([0, 5], close=[1.0, 2.0], volume=[1.0, 2.0]), time_frame="5m")
    df.append(frame([6], close=[4.0], volume=[3.0]))  # the forming period: two fine bars
    df["ma:2"]
    df.loc[np.array([True, False]), "ma:2"] = -1.0
    assert not df.is_computed("ma:2")
    df.append(frame([7], close=[5.0], volume=[1.0], **{"ma:2": [7.0]}))
    got = rows(df)
    assert got[0] == [1.0, 1.0, -1.0] and got[1][:2] == [5.0, 6.0] and np.isnan(got[1][2])


def test_a_windowed_closed_write_leaves_the_open_period_like_an_unbounded_one():
    seed = frame(list(range(0, 50, 5)), close=np.arange(1.0, 11.0), volume=np.ones(10))
    unb = DataFrame(seed, time_frame="5m")
    win = DataFrame(seed, time_frame="5m", window=4, max_lookback=2)
    for df in (unb, win):
        df.append(frame([46], close=[20.0], volume=[2.0]))
        df.iat[-2, 1] = 77.0  # the closed row before the forming one
        df.append(frame([47], close=[21.0], volume=[3.0]))
        df.append(frame([47], close=[21.5], volume=[2.5]))  # a re-send replaces exactly
    assert rows(win) == rows(unb)[-4:]
    assert rows(win)[-2:] == [[9.0, 77.0], [21.5, 5.5]]
