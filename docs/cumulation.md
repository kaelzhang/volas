# Cumulation, time frames and timezones

## Cumulation and DatetimeIndex

Suppose we have a csv file containing kline data of a stock in the 1-minute time
frame:

```py
csv = read_csv(csv_path)

print(csv)
```

```
                   date   open   high    low  close    volume
0   2020-01-01 00:00:00  329.4  331.6  327.6  328.8  14202519
1   2020-01-01 00:01:00  330.0  332.0  328.0  331.0  13953191
2   2020-01-01 00:02:00  332.8  332.8  328.4  331.0  10339120
3   2020-01-01 00:03:00  332.0  334.2  330.2  331.0   9904468
4   2020-01-01 00:04:00  329.6  330.2  324.9  324.9  13947162
5   2020-01-01 00:04:00  329.6  330.2  324.8  324.8  13947163    <- an update of
                                                                    2020-01-01 00:04:00
...
19  2020-01-01 00:19:00  327.0  327.2  322.0  323.0  15086985
```

> Note that duplicated records of the same timestamp are not cumulated. All
> records except the latest one are discarded.

Read the same csv, but parse the `date` column into a `DatetimeIndex`:

```py
df = read_csv(
    csv_path,
    parse_dates=['date'],
    index_col='date'
)

print(df)
```

```
                      open   high    low  close    volume
2020-01-01 00:00:00  329.4  331.6  327.6  328.8  14202519
2020-01-01 00:01:00  330.0  332.0  328.0  331.0  13953191
...
2020-01-01 00:19:00  327.0  327.2  322.0  323.0  15086985
```

You must have figured it out that the data frame now has a
[`DatetimeIndex`](https://pandas.pydata.org/docs/reference/api/pandas.DatetimeIndex.html).

But it will not become a 5-minute kline unless we cumulate it:

```py
df_5m = df.cumulate('5m')

print(df_5m)
```

Now we get a 5-minute kline:

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  334.2  324.8  324.8  62346461.0
2020-01-01 00:05:00  325.0  327.8  316.2  322.0  82176419.0
2020-01-01 00:10:00  323.0  327.8  314.6  327.6  74409815.0
2020-01-01 00:15:00  330.0  335.2  322.0  323.0  82452902.0
```

`cumulate` defaults to OHLCV semantics — `open`=first, `high`=max, `low`=min,
`close`=last, `volume`=sum — and **any other column falls back to `last`**. Pass
`cumulators=` to override a column's aggregator; the common case is a non-OHLCV
column that should be summed, such as a turnover (`amount`) column that would
otherwise default to `last`:

```py
df.cumulate('1h', cumulators={'amount': 'sum'})
```

The supported aggregators are `first`, `max`, `min`, `last` and `sum`. Each keeps
its column's dtype, and each dtype supports:

| dtype | `first` / `last` | `max` / `min` | `sum` |
|---|---|---|---|
| `float64` / `float32` / `int64` / `int32` | ✓ | ✓ | ✓ (an integer sum is exact and raises `OverflowError` rather than wrap) |
| `datetime64[ns]` | ✓ | ✓ | — (`TypeError`) |
| `bool` / `str` | ✓ | — (`TypeError`) | — (`TypeError`) |

**A missing input is never skipped.** A period's `first` / `last` / `max` / `min` /
`sum` is missing as soon as **any** of the period's values is missing — a `NaN`
float, an int / bool / str `volas.NA`, a datetime `NaT` — whatever value a missing
cell happens to carry, and an all-missing period is missing too (never `±inf`). One
`NaN` high inside a 15-minute bar thus gives that bar a `NaN` high (which the
indicators then treat as a missing row) instead of the maximum of the known highs,
which could understate the true range with no sign that data was missing. The live
fold (`append` on a `time_frame` frame) and the batch `cumulate` run the same
aggregation, so they always agree. pandas `resample` skips missing values instead
(see [pandas compatibility](pandas-compatibility.md)).

The `time_frame` may be a string label or a `TimeFrame` constant — see
[TimeFrame](#timeframe) for the full list.

### Bar labels are the period start

Every time frame lies on a **fixed grid**, and a cumulated bar is labelled with
its period's grid **start** — even when the first raw bar arrives mid-period. A
bar that opens with a `09:07` tick on a `15m` frame is labelled `09:00`, never
`09:07`, so volas bars line up exactly with exchange klines and with pandas
`resample` (`label='left'`).

The grid origins per frame: intraday frames anchor at **midnight** of the
index's (timezone-aware) trading day — a `15m` bar starts at `:00`/`:15`/`:30`/
`:45`, a `4h` bar at `00:00`/`04:00`/…; `1d` starts at midnight; `1w` on
**Monday**; `3d` is a continuous grid from the Unix epoch; `1M` / `1y` on the
calendar month / year.

Across a daylight-saving transition, **sub-day** frames bucket by real instants
on that local grid: a wall-clock hour that a fall-back repeats is two bars with
distinct instant labels (on 2026-11-01 in `America/New_York` a `1h` frame has a
`01:00-04:00` bar and a `01:00-05:00` bar, exactly as pandas `resample('1h')`),
and a grid point that a spring-forward gap skips starts its bar at the instant
the gap ends. A fall-back that does not repeat a grid point stays inside one bar
(the `2h` bar of `00:00` spans three real hours that day). **Day-and-coarser**
frames stay on the local civil calendar: the 25-hour fall-back day is one `1d`
bar, labelled at its local midnight (or, where midnight itself is skipped, at
the first real instant of the day). Live folding and `cumulate` bucket
identically.

The rows given to `DataFrame(data, time_frame=...)` follow the same rule: volas
reads each row's label as its period start (it cannot check this), so data
labelled by its period **end** must be shifted back one period first.

For **live** streaming you do not re-cumulate the whole history on every tick —
you keep the current 5-minute bar *forming* and update it as each finer bar
arrives. A **tf-aware DataFrame** does exactly that: it stays an ordinary
DataFrame (read columns, run directives, slice it), except `append` **folds**
each finer bar into the bar currently forming instead of adding a row. You make
one with `df.cumulate('5m')` or `DataFrame(data, time_frame='5m')`, and the live
loop is then just:

| step                           | call                      |
| ------------------------------ | ------------------------- |
| make a `5m` frame              | `cum = df.cumulate('5m')` |
| feed it the next finer bar     | `cum.append(bar)`         |
| read the current forming bar   | `cum.iloc[-1]`            |
| read an indicator over it      | `cum['macd']`             |

### Watch the forming bar grow

Build the 5-minute frame from the 1-minute `df` above one bar at a time. Seed it
with the `00:00` bar (`DataFrame(df.iloc[0:1], time_frame='5m')` seeds it the same
way: its last row is the forming bar), then fold in `00:01`. Both fall in the same `00:00`–`00:05`
window, so the frame still holds **one** row — the forming bar — now updated
(`high` rose to `332.0`, `close` to `331.0`, `volume` summed):

```py
cum = df.iloc[0:1].cumulate('5m')   # seed the 5m frame with the 00:00 bar
cum.append(df.iloc[1:2])            # fold in 00:01 (same 5m window)

print(cum)
```

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  332.0  327.6  331.0  28155710.0
```

Fold in `00:02`, `00:03` and `00:04` and the window fills up. That single forming
row is now the **finished** first 5-minute bar — identical to the first row of
the one-shot `df.cumulate('5m')` printed earlier:

```py
for i in range(2, 5):
    cum.append(df.iloc[i:i + 1])

print(cum)
```

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  334.2  324.8  324.8  62346461.0
```

Now fold in `00:05`. It opens the **next** window, so the `00:00` bar is finalized
and a fresh forming bar starts; the frame grows to two rows and `cum.iloc[-1]` is
the new, still-forming `00:05` bar:

```py
cum.append(df.iloc[5:6])

print(cum)
```

```
                      open   high    low  close      volume
2020-01-01 00:00:00  329.4  334.2  324.8  324.8  62346461.0   <- finalized
2020-01-01 00:05:00  325.0  327.8  324.8  327.6  10448427.0   <- still forming
```

Two properties make this safe for a live feed:

- **Indicators are lazy, and fresh on read.** `append` does not recompute
  anything — it only flags the dependent directive columns as stale (their
  valid-row cursor now lags the frame height). The recompute happens when you
  **read** `cum['ema:9']` (or any directive): only the stale tail is refreshed —
  `O(lookback)`, not the whole column — over the frame *including* the forming
  row, bit-identical to a one-shot cumulate-then-compute. (A bulk read such as
  `to_numpy()` does not auto-refresh; call `cum.fulfill()` first, or just read
  the directive.)
- **Re-sent bars do not double-count.** The forming period keeps its bars, and
  folding a bar whose timestamp is already among them **replaces** that bar — the
  same dedup rule shown at the top of this section — matching exchanges that
  revise their most recent bar. Re-sending a bar of the frame's own time frame
  (`5m` here, stamped `00:05`) replaces the whole forming row.

See [Live cumulation](dataframe-api.md#live-cumulation--a-tf-aware-dataframe) for the API summary.

### Every bar is complete

Each bar folded into a time-frame frame is first made a complete bar of the frame's
data columns (every column except the cached indicator ones), the same way in every
append form — a `DataFrame`, a `Row` or a scalar dict: a column the bar leaves out is
a missing value of that bar. The open period, the in-place fold and the re-cumulation
after a re-sent bar all read that one bar, so leaving `volume` out of a bar gives the
period a missing `volume` (a missing input is never skipped), exactly as a batch
`cumulate` of the same bars does, and a re-send of the bar with its `volume` restores
it.

### Writing to a time-frame frame

The forming row is the cumulation of the open period's bars, so a write to a
time-frame frame — `df[col] = values`, a cell write such as `df.iloc[-1, j] = v` (or
`loc` / `iat` / `at`, or a mask write) — is taken only where the open period can
follow it exactly; it never leaves a value that a later bar would count twice or
silently drop:

- A write that reaches only **closed** rows leaves the open period as it is: a bar
  re-sent afterwards still replaces its earlier version.
- A column that **becomes data** — a new column (`df['signal'] = values`, say on a
  `cumulate` result), or a cached indicator column overwritten — was never carried by
  the period's fine bars, so they hold it as missing. The forming cell keeps the
  written value until the next bar of the period arrives; that fold makes it missing
  (a missing input is never skipped), so the value is visibly stale, never wrong. A
  forming row that closes without another bar keeps the written value. Later bars
  that carry the column fold it as usual.
- A **new forming value** of a column the bars carry is exact while the period holds
  one bar — right after the frame is built with `time_frame=`, or when only the first
  bar of a new period has arrived: that bar is rewritten, the next bar folds into it,
  and a re-send of it replaces it whole. A forming bar folded from several fine bars
  has no single bar to rewrite, so the write raises `ValueError` and the frame is
  unchanged — write to closed rows only, or append a corrected fine bar (a bar with
  the same timestamp replaces the one it repeats). A write is judged by the value it
  leaves in the forming cell — `df.iloc[[-1, -1], j] = [a, b]` leaves `b` — and writing
  a forming cell its current value is always allowed. A column replaced by the same
  values in another dtype (an integer `volume` by floats) keeps its bars, converted
  losslessly; bars that cannot take the new dtype exactly count as a new value.
- A column added or replaced must be one its aggregator folds: with
  `cumulators={'tag': 'max'}`, `df['tag'] = ['a']` raises `TypeError` and the frame is
  unchanged.
- `DataFrame(df, index=...)` and `DataFrame(df, dtype=...)` re-label or re-type the
  rows: the result is the time-frame frame of its new rows, the last one its forming
  bar, exactly as `DataFrame(rows, time_frame=...)` builds it. `DataFrame(df,
  columns=[...])` keeps the open period's bars of the data columns it keeps; a cached
  column it turns into data enters them as missing.

## TimeFrame

A `TimeFrame` names a bar interval. It is accepted anywhere volas resamples —
`df.cumulate` and the `time_frame` DataFrame argument — either as a `TimeFrame`
constant or as its equivalent **string label**. There is no `TimeFrame(...)`
constructor — use one of the constants below or a label string. The `hv`
indicator takes the label inside its directive; see its
[reference entry](https://github.com/kaelzhang/volas/blob/main/INDICATORS.md#user-content-indicator-hv)
for how each frame is annualized.

```py
TimeFrame.m5            # the 5-minute frame
'5m'                    # the equivalent label string, accepted everywhere too

df.cumulate(TimeFrame.m5)     # identical to df.cumulate('5m')
```

Supported frames (constant ⇄ label):

| Constant | Label | Alignment |
| --- | --- | --- |
| `TimeFrame.s1` | `'1s'` | Civil second. |
| `TimeFrame.m1` | `'1m'` | Civil minute. |
| `TimeFrame.m3` | `'3m'` | Minute-of-hour buckets starting at `00`, `03`, `06`, ... |
| `TimeFrame.m5` | `'5m'` | Minute-of-hour buckets starting at `00`, `05`, `10`, ... |
| `TimeFrame.m15` | `'15m'` | Minute-of-hour buckets starting at `00`, `15`, `30`, `45`. |
| `TimeFrame.m30` | `'30m'` | Minute-of-hour buckets starting at `00` and `30`. |
| `TimeFrame.H1` | `'1h'` | Civil hour. |
| `TimeFrame.H2` | `'2h'` | Hour-of-day buckets starting at `00`, `02`, `04`, ... |
| `TimeFrame.H4` | `'4h'` | Hour-of-day buckets starting at `00`, `04`, `08`, ... |
| `TimeFrame.H6` | `'6h'` | Hour-of-day buckets starting at `00`, `06`, `12`, `18`. |
| `TimeFrame.H8` | `'8h'` | Hour-of-day buckets starting at `00`, `08`, `16`. |
| `TimeFrame.H12` | `'12h'` | Hour-of-day buckets starting at `00` and `12`. |
| `TimeFrame.D1` | `'1d'` | Civil day in the frame timezone. |
| `TimeFrame.D3` | `'3d'` | Continuous 3-day buckets anchored to the Unix epoch; they do not reset at month boundaries. |
| `TimeFrame.W1` | `'1w'` | Continuous Monday-start weeks, including runs that cross month boundaries. |
| `TimeFrame.M1` | `'1M'` | Civil calendar month in the frame timezone. |
| `TimeFrame.Y1` | `'1y'` | Civil calendar year in the frame timezone. |

Every bucket is aligned in the **frame timezone's local wall-clock** while storage
stays UTC: the hour-of-day frames (`2h`/`4h`/`6h`/`8h`/`12h`) start at local `00`
and step in local hours; `3d` counts continuous 3-local-civil-day buckets keyed
from the Unix epoch day in that zone (not reset at month boundaries); `1w` is
Monday-start in local civil time. So a daily/weekly bar follows the local trading
day, and a named zone makes the hour buckets DST-aware (see
[Bar labels are the period start](#bar-labels-are-the-period-start) for the
repeated and skipped hours). The grid always belongs to the **frame's** zone: a
bar appended to a tf-aware frame is placed by its instant on that grid, whatever
zone the bar's own index carries.

## Timezones

Storage is always **UTC epoch-nanoseconds** — the universal axis on which US,
HK and A-share equity frames coexist and align on the absolute instant. A
`DatetimeIndex` additionally carries a **per-frame timezone** that governs how
those instants render, how bare-string labels match, and how `cumulate` aligns
day-and-coarser buckets. A timezone is either a **fixed offset** (`'+08:00'`,
cheap; A-share / HK equities) or a **named IANA zone** (`'America/New_York'`).
volas reads named-zone TZif bytes from the running interpreter's `zoneinfo.TZPATH`
directories in order, then the `tzdata` package, and evaluates them in Rust across
the full `datetime64[ns]` range, including future POSIX footer rules. `tzdata` is a
runtime dependency. The first successful initialization establishes one immutable
rules object per named key and process. All later timestamps, frames, and open cumulation
periods using that key share the same immutable rules without reading files or
recompiling. Named Python exports use the interpreter's standard `ZoneInfo(key)`;
volas retains that object so its identity remains stable for pandas interop.
A supplied keyed tzinfo selects these canonical key rules, including an object
constructed with `ZoneInfo.from_file`; its private rule table is not imported.
An aware datetime still supplies its own UTC offset when determining its instant.
For an unkeyed tzinfo passed as `tz=`, volas uses its whole-second
`utcoffset(None)` as a fixed offset. An aware datetime uses that fixed offset
when available; if its unkeyed tzinfo provides no fixed offset, volas captures
its offset at that instant instead. Sub-second offsets are rejected; no dynamic
custom rule table is imported.

Here is the whole picture. Build a `DatetimeIndex` by parsing a column with
`to_datetime`, promoting it with `set_index`, then tagging the display zone with
`tz_localize` (reinterpret a naive wall-clock *as* that zone — the instant moves)
or `tz_convert` (keep the instant, restate the zone). A US exchange opens at 09:30
local on 2021-01-04, held as a naive local string:

```py
from volas import DataFrame, to_datetime, Timestamp

# Parse the naive 't' strings to UTC instants and make them the index, then read
# the wall-clock *as New York local time* with tz_localize. The instant is stored
# UTC (14:30Z), but the index renders and matches in New York.
df = DataFrame({'t': ['2021-01-04 09:30:00'], 'close': [100.0]})
df['t'] = to_datetime(df['t'])
df = df.set_index('t').tz_localize('America/New_York')
df.tz       # 'America/New_York'
df.index    # ['2021-01-04T14:30:00.000000000']  (raw .index is UTC, matching pandas .values)

# The tz is what lets a bare local string match the right row — it is parsed in df.tz:
df.at['2021-01-04 09:30:00', 'close']   # 100.0

# A Timestamp is a typed, cross-tz label. The SAME instant in Shanghai is
# 22:30+08:00, and it still matches, regardless of df.tz:
ts = Timestamp('2021-01-04 22:30:00', tz='+08:00')   # == 09:30 New York
df.at[ts, 'close']                       # 100.0
ts.value                                 # its UTC epoch-nanoseconds (int)
ts.tz                                    # '+08:00'

# Integer epochs: to_datetime(unit=...) reads the unit. An epoch is *absolute* —
# anchor it as UTC, then restate the zone for display. 1609770600000 ms == 14:30Z:
e = DataFrame({'t': [1609770600000], 'close': [100.0]})
e['t'] = to_datetime(e['t'], unit='ms')
e.set_index('t').tz_localize('UTC').tz_convert('America/New_York').index
# ['2021-01-04T14:30:00.000000000']

# An offset-aware string is already absolute too — to_datetime resolves the offset:
o = DataFrame({'t': ['2021-01-04T09:30:00+08:00'], 'close': [1.0]})
o['t'] = to_datetime(o['t'])
o.set_index('t').index
# ['2021-01-04T01:30:00.000000000']  (09:30+08:00 == 01:30Z)
```

A frame's time axis is in one of two states (the pandas model): **naive** (an
unanchored wall-clock, `df.tz is None`) or **tz-aware** (anchored, `df.tz` names
the zone — `'UTC'` included). `tz_localize` anchors a naive axis (the instant
moves to match the wall-clock in that zone); `tz_convert` restates an aware
axis in another zone (the instant is unchanged). Each refuses the other state —
converting an unanchored clock or re-anchoring an anchored one would silently
shift instants:

```py
naive = df                                   # df.tz is None
aware = naive.tz_localize('America/New_York')   # anchor: instants move, wall-clock kept
aware.tz_convert('+08:00')                   # restate: instants kept, wall-clock moves
naive.tz_convert('+08:00')                   # TypeError — anchor with tz_localize first
aware.tz_localize('UTC')                     # TypeError — already anchored; use tz_convert
```

### Reproducible timezone deployment

Pin `tzdata` to an exact release in your deployment lockfile and set
`PYTHONTZPATH=""` before starting Python. This makes every host resolve zones from
that package instead of a host-specific system or Conda directory. volas does
not bundle or download a second timezone database. Configure `TZPATH` before any
timezone use, and restart the process after updating timezone data. Changing
files, resetting `TZPATH`, or clearing `ZoneInfo` caches after a key has been used
is not a supported way to update that key within a running process.

```sh
PYTHONTZPATH="" python your_service.py
```

Assert the chosen source and IANA version at startup:

```py
import volas

# The IANA release selected by this deployment's pinned tzdata package.
expected_iana_version = "2026b"
assert volas.tz_database() == {"source": "tzdata", "version": expected_iana_version}
```

`volas.tz_database(key="UTC")` returns `{"source": str, "version": str | None}`.
`source` is the selected `TZPATH` directory, or `"tzdata"` for
the package. `version` is the `# version` entry in that directory's `tzdata.zi`,
or `tzdata.IANA_VERSION`; it is `None` when the source does not identify a release.
Pass a specific IANA key to inspect its source when directories contain only
some zones, for example `volas.tz_database("America/New_York")`. This initializes
an unused key; later calls report its first-use source and version. The default
key is `UTC`. String `"UTC"` and numeric UTC offsets remain fixed specifications.

`cumulate` to a daily (or coarser) bar aligns buckets to the frame's local
trading day — DST-aware for a named zone — while the raw `.index` numpy export
stays UTC (matching pandas `.values`).
