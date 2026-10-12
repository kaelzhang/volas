"""The open period of a time-frame frame: the small frame of fine bars whose
cumulation is the forming row.

It holds one bar per timestamp, in increasing order, however it was built — a re-sent
bar in the bars handed to ``cumulate`` (or carried by ``copy`` / ``DataFrame(other)``)
is never counted again when a later bar re-cumulates it.
"""

import numpy as np
import pytest

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
