"""Named zones follow their rules past the compiled transition table.

tzdb expands rules in force indefinitely into explicit transitions only through 2099; a
TZif file carries them on in its POSIX footer, which Python's `zoneinfo` applies. volas
must give the same offsets, the same wall-clock → instant answers (gaps and folds
included) and the same period starts for every instant of the datetime64[ns] axis
(1677-2262), so these properties are checked against `zoneinfo` from 1900 to the end of
the axis, with transitions sampled densely after 2099.
"""

from __future__ import annotations

import datetime as dt
import zlib
from zoneinfo import ZoneInfo

import numpy as np
import pytest

import volas

ZONES = ['America/New_York', 'Europe/London', 'Australia/Sydney', 'Australia/Lord_Howe',
         'America/Santiago', 'Asia/Tehran', 'Asia/Shanghai']
UTC = dt.timezone.utc
NS = 1_000_000_000
LO = int(dt.datetime(1900, 1, 1, tzinfo=UTC).timestamp())
HI = int(dt.datetime(2262, 4, 1, tzinfo=UTC).timestamp())
FAR = int(dt.datetime(2100, 1, 1, tzinfo=UTC).timestamp())


def _offset(zone, sec):
    return dt.datetime.fromtimestamp(sec, tz=ZoneInfo(zone)).utcoffset().total_seconds()


def _transitions(zone, lo, hi):
    """The instants (seconds) in [lo, hi) where the zone's offset changes, to the minute."""
    out, step = [], 6 * 3600
    for a in range(lo, hi, step):
        if _offset(zone, a) != _offset(zone, a + step):
            x, y = a, a + step
            while y - x > 60:
                m = (x + y) // 2
                x, y = (m, y) if _offset(zone, m) == _offset(zone, x) else (x, m)
            out.append(y)
    return out


def _localize(zone, wall):
    """zoneinfo's reading of a naive wall clock: ('ok', seconds) / ('gap',) / ('fold',)."""
    z = ZoneInfo(zone)
    first, second = wall.replace(tzinfo=z), wall.replace(tzinfo=z, fold=1)
    a, b = first.timestamp(), second.timestamp()
    if dt.datetime.fromtimestamp(a, tz=z).replace(tzinfo=None) != wall:
        return ('gap',)
    return ('fold',) if a != b else ('ok', int(a))


def _volas_localize(zone, wall):
    naive = np.array([np.datetime64(wall, 'ns')])
    try:
        return ('ok', int(volas.DataFrame({'c': [1.0]}, index=naive).tz_localize(zone).index.view('i8')[0]) // NS)
    except ValueError as e:
        return ('gap',) if 'DST gap' in str(e) else ('fold',) if 'DST fold' in str(e) else e


@pytest.mark.parametrize('zone', ZONES)
def test_offsets_match_zoneinfo(zone):
    rng = np.random.default_rng(zlib.crc32(zone.encode()))
    secs = np.r_[rng.integers(LO, HI, 600), rng.integers(FAR, HI, 600)]
    for s in map(int, secs):
        got = volas.Timestamp(s * NS, tz=zone)
        want = dt.datetime.fromtimestamp(s, tz=ZoneInfo(zone))
        assert got.utcoffset() == want.utcoffset(), f'{zone} {want}'
        assert str(got)[:19] == want.replace(tzinfo=None).isoformat(sep=' '), f'{zone} {want}'


@pytest.mark.parametrize('zone', ZONES)
def test_wall_clocks_localize_like_zoneinfo(zone):
    rng = np.random.default_rng(7)
    years = sorted(set(map(int, rng.integers(2100, 2262, 6))) | {2100, 2261})
    walls = []
    for y in years:
        lo = int(dt.datetime(y, 1, 1, tzinfo=UTC).timestamp())
        for t in _transitions(zone, lo, lo + 366 * 86400):
            for off in (-5400, -3600, -1800, -60, 0, 60, 1800, 3600, 5400):
                walls.append(dt.datetime.fromtimestamp(t + off, tz=UTC).replace(tzinfo=None)
                             + dt.timedelta(seconds=_offset(zone, t - 1)))
        walls += [dt.datetime(y, 1, 1) + dt.timedelta(seconds=int(r)) for r in rng.integers(0, 365 * 86400, 40)]
    assert zone in ('Asia/Tehran', 'Asia/Shanghai') or len(walls) > 40 * len(years)
    for w in walls:
        assert _volas_localize(zone, w) == _localize(zone, w), f'{zone} {w}'


def _period_start(zone, sec, step):
    """The latest period boundary at or before `sec` on the zone's local grid of `step`
    seconds: an instant whose wall clock is a grid point, or the end of a gap that skipped
    one (days: the first instant of the local date)."""
    z = ZoneInfo(zone)
    wall = lambda t: dt.datetime.fromtimestamp(t, tz=z).replace(tzinfo=None)  # noqa: E731
    if step == 86400:
        d = wall(sec).date()
        lo, hi = sec - 2 * 86400, sec  # earliest instant whose local date is d
        while hi - lo > 1:
            m = (lo + hi) // 2
            lo, hi = (lo, m) if wall(m).date() >= d else (m, hi)
        return hi
    t = sec - sec % 60
    while True:
        w, before = wall(t), wall(t - 60)
        on_grid = (w - dt.datetime(w.year, w.month, w.day)).total_seconds() % step == 0
        skipped = before + dt.timedelta(seconds=60) != w and any(
            (g - dt.datetime(g.year, g.month, g.day)).total_seconds() % step == 0
            for g in (before + dt.timedelta(seconds=k) for k in range(60, int((w - before).total_seconds()), 60)))
        if on_grid or skipped:
            return t
        t -= 60


@pytest.mark.parametrize('freq,step', [('1d', 86400), ('1h', 3600)])
@pytest.mark.parametrize('zone', ZONES)
def test_period_starts_match_zoneinfo(zone, freq, step):
    secs = []
    for y in (2101, 2150, 2230, 2261):
        lo = int(dt.datetime(y, 1, 1, tzinfo=UTC).timestamp())
        for t in _transitions(zone, lo, lo + 366 * 86400):
            secs += [t + k * 900 for k in range(-8, 9)]
        secs.append(lo + 200 * 86400 + 4321)
    secs = sorted(set(secs))
    ns = np.array(secs, dtype='int64') * NS
    df = volas.DataFrame({'close': np.arange(len(secs), dtype=float)},
                         index=ns.astype('datetime64[ns]')).tz_localize('UTC').tz_convert(zone)
    got = list(df.cumulate(freq).index.view('i8') // NS)
    want = list(dict.fromkeys(_period_start(zone, s, step) for s in secs))
    assert got == want
