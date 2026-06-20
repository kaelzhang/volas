"""The append-freshness contract (Policy D).

After a lazy `append` leaves cached directive columns stale, every public read
falls into exactly one of two buckets:

* **Column projection** — `df['ma:3']` and `df[['ma:3', ...]]` — AUTO-REFRESHES
  the stale tail (O(lookback)) and returns fresh values. The single- and
  multi-column forms must behave identically.
* **Everything else** — positional/label indexing, conversions, reductions,
  transforms, display — FAILS LOUD with a clear ValueError telling the caller to
  `fulfill()` first. It must never silently return stale / NaN values.

This test pins both buckets so a newly added read method cannot silently
reintroduce the stale-read bug.
"""

import numpy as np
import pytest

from volas import DataFrame


def _stale():
    """A frame with two cached directives and one un-fulfilled appended bar."""
    n = 50
    close = 100.0 + np.cumsum(np.full(n, 0.3))
    b = DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": np.full(n, 1_000.0),
        }
    )
    b["ma:3"]
    b["rsi:14"]
    b.append(
        DataFrame({"open": [115.0], "high": [116.0], "low": [114.5], "close": [115.5], "volume": [1500.0]})
    )
    return b


def _norm(x):
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    try:
        return np.array2string(np.asarray(x).astype(float), precision=6)
    except (TypeError, ValueError):
        return repr(x)


# --- bucket 1: column projection auto-refreshes (single == multi) ----------------

AUTO_REFRESH = {
    "single": lambda b: b["ma:3"],
    "list-one": lambda b: b[["ma:3"]],
    "list-mixed": lambda b: b[["close", "ma:3", "rsi:14"]],
}


@pytest.mark.parametrize("name", AUTO_REFRESH)
def test_column_projection_auto_refreshes(name):
    """A stale read equals the same read after an explicit fulfill()."""
    fn = AUTO_REFRESH[name]
    stale = fn(_stale())
    ful = _stale()
    ful.fulfill()
    assert _norm(stale) == _norm(fn(ful))


def test_single_and_multi_projection_agree():
    """`df['ma:3']` and `df[['ma:3']]` return the same fresh ma:3 after append."""
    b = _stale()
    one = b["ma:3"].to_numpy()
    two = b[["ma:3"]].to_numpy()[:, 0]
    np.testing.assert_array_equal(one, two)
    assert not np.isnan(one[-1])  # the appended bar's value is computed, not NaN


# --- bucket 2: every other read fails loud (never silent stale) ------------------

FAIL_LOUD = {
    "to_numpy": lambda b: b.to_numpy(),
    "iloc-row": lambda b: b.iloc[-1],
    "at": lambda b: b.at[50, "ma:3"],
    "to_csv": lambda b: b.to_csv(),
    "repr": lambda b: repr(b),
    "sum": lambda b: b.sum(),
    "mean": lambda b: b.mean(),
    "max": lambda b: b.max(),
    "min": lambda b: b.min(),
    "count": lambda b: b.count(),
    "std": lambda b: b.std(),
    "var": lambda b: b.var(),
    "median": lambda b: b.median(),
    "quantile": lambda b: b.quantile(0.5),
    "prod": lambda b: b.prod(),
    "sem": lambda b: b.sem(),
    "skew": lambda b: b.skew(),
    "kurt": lambda b: b.kurt(),
    "nunique": lambda b: b.nunique(),
    "any": lambda b: b.any(),
    "all": lambda b: b.all(),
    "idxmax": lambda b: b.idxmax(),
    "idxmin": lambda b: b.idxmin(),
    "describe": lambda b: b.describe(),
    "head": lambda b: b.head(),
    "tail": lambda b: b.tail(1),
    "fillna": lambda b: b.fillna(0),
    "dropna": lambda b: b.dropna(),
    "round": lambda b: b.round(2),
    "rename": lambda b: b.rename(columns={"open": "o2"}),
    "drop": lambda b: b.drop(["open"], axis=1),
    "nlargest": lambda b: b.nlargest(3, "ma:3"),
    "nsmallest": lambda b: b.nsmallest(3, "ma:3"),
    "mode": lambda b: b.mode(),
    "replace": lambda b: b.replace(0.0, 1.0),
    "interpolate": lambda b: b.interpolate(),
    "isin": lambda b: b.isin([0.0]),
    "equals": lambda b: b.equals(b.copy()),
    "eq": lambda b: b == b.copy(),
    "drop_duplicates": lambda b: b.drop_duplicates(),
    "duplicated": lambda b: b.duplicated(),
    "value_counts": lambda b: b.value_counts(),
    "sort_index": lambda b: b.sort_index(),
    "reset_index": lambda b: b.reset_index(),
    "set_index": lambda b: b.set_index("open"),
    "astype": lambda b: b.astype({"open": "float32"}),
}


@pytest.mark.parametrize("name", FAIL_LOUD)
def test_bulk_read_fails_loud(name):
    """A stale bulk/aggregate read raises, naming fulfill() — never silent stale."""
    with pytest.raises(ValueError, match="stale computed|fulfill"):
        FAIL_LOUD[name](_stale())


def test_fulfill_unblocks_bulk_reads():
    """After fulfill(), the same reads succeed and reflect the appended bar."""
    b = _stale()
    b.fulfill()
    assert b.max()["ma:3"] == pytest.approx(115.06666666, rel=1e-9)
    assert int(b.count()["ma:3"]) == 49
