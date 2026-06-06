"""P2: pandas-style calendar fields / strftime on volas.Timestamp.

Fields read the **local wall-clock** in the timestamp's timezone (matching
pandas); weekday is Monday=0 .. Sunday=6.
"""

import pytest

from volas import Timestamp


def test_date_and_time_fields_utc():
    t = Timestamp("2020-03-15 04:05:06")
    assert (t.year, t.month, t.day) == (2020, 3, 15)
    assert (t.hour, t.minute, t.second) == (4, 5, 6)


@pytest.mark.parametrize(
    "date,wd",
    [
        ("2020-03-16", 0),  # Monday
        ("2020-03-17", 1),
        ("2020-03-20", 4),
        ("2020-03-21", 5),  # Saturday
        ("2020-03-15", 6),  # Sunday
    ],
)
def test_weekday_monday_is_zero(date, wd):
    assert Timestamp(date).weekday() == wd


def test_strftime_common_codes():
    t = Timestamp("2020-03-15 04:05:06")
    assert t.strftime("%Y-%m-%d") == "2020-03-15"
    assert t.strftime("%H:%M:%S") == "04:05:06"
    assert t.strftime("%A") == "Sunday"
    assert t.strftime("a %% literal and %Y") == "a % literal and 2020"


def test_strftime_invalid_format_raises():
    with pytest.raises(ValueError, match="invalid strftime"):
        Timestamp("2020-03-15").strftime("%Q")


def test_tz_aware_fields_read_local_wallclock():
    # A naive string in a zone is that zone's wall clock; the fields read it back.
    t = Timestamp("2020-03-15 23:30:45", tz="America/New_York")
    assert (t.year, t.month, t.day) == (2020, 3, 15)
    assert (t.hour, t.minute, t.second) == (23, 30, 45)


def test_tz_keeps_local_calendar_day():
    t = Timestamp("2020-01-01 00:30:00", tz="Asia/Tokyo")
    assert (t.year, t.month, t.day, t.hour) == (2020, 1, 1, 0)
    assert t.strftime("%Y-%m-%d %H:%M") == "2020-01-01 00:30"
