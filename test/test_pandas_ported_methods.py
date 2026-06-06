"""Spec-to-spec ports of pandas's own method tests, restricted to the float64 /
bool, default- or datetime-indexed Series that volas models.

Each block cites the upstream pandas test module. Cases that exercise a feature
volas deliberately omits (object / nullable / Categorical dtypes, index-label
*alignment*, array-valued clip bounds, non-linear quantile interpolation,
multi-quantile lists, a Float64 index) are intentionally not ported; where a
result is identical but the spelling differs (volas uses ``s.round(n)`` rather
than the ``round(s)`` builtin), the volas spelling is used.
"""

import numpy as np
import pytest

import volas
from volas import DataFrame


def _s(values):
    return DataFrame({"a": list(values)})["a"]


nan = float("nan")


# === pandas/tests/series/methods/test_round.py ==============================

class TestRound:
    def test_round_numpy(self):
        # GH#12600
        np.testing.assert_array_equal(_s([1.53, 1.36, 0.06]).round(0).to_numpy(), [2.0, 1.0, 0.0])

    def test_round_numpy_with_nan(self):
        # GH#14197 — NaN is preserved
        np.testing.assert_array_equal(
            _s([1.53, nan, 0.06]).round().to_numpy(), [2.0, nan, 0.0]
        )

    def test_round_builtin(self):
        # volas spelling: s.round(n) (pandas also supports the round() builtin)
        np.testing.assert_array_equal(_s([1.123, 2.123, 3.123]).round(0).to_numpy(), [1, 2, 3])
        np.testing.assert_array_equal(
            _s([1.123, 2.123, 3.123]).round(2).to_numpy(), [1.12, 2.12, 3.12]
        )

    @pytest.mark.parametrize(
        "data,decimals,expected",
        [
            ([0.2], 0, [0.0]),
            ([1.234, 2.567], 1, [1.2, 2.6]),
            ([1.234, 2.567], 2, [1.23, 2.57]),
            ([1.234, 2.567], 0, [1.0, 3.0]),
        ],
    )
    def test_round_data(self, data, decimals, expected):
        np.testing.assert_array_equal(_s(data).round(decimals).to_numpy(), expected)

    def test_round_empty_series(self):
        assert _s([]).round(4).to_numpy().tolist() == []


# === pandas/tests/series/methods/test_clip.py ==============================

class TestClip:
    def test_clip(self):
        s = _s([5.0, 1.0, -3.0, 2.0, 4.0])
        val = s.median()
        assert s.clip(lower=val).min() == val
        assert s.clip(upper=val).max() == val
        np.testing.assert_array_equal(
            s.clip(-0.5, 0.5).to_numpy(), np.clip(s.to_numpy(), -0.5, 0.5)
        )

    def test_clip_types_and_nulls(self):
        s = _s([nan, 1.0, 2.0, 3.0])
        thresh = 2.0  # s[2]
        lower = s.clip(lower=thresh)
        upper = s.clip(upper=thresh)
        assert np.nanmin(lower.to_numpy()) == thresh
        assert np.nanmax(upper.to_numpy()) == thresh
        # NaN positions are preserved on both sides
        assert np.isnan(lower.to_numpy()[0])
        assert np.isnan(upper.to_numpy()[0])

    def test_clip_with_na_args_is_noop(self):
        # GH#17276 — a NaN bound behaves as None (no clipping)
        s = _s([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(s.clip(nan).to_numpy(), [1, 2, 3])
        np.testing.assert_array_equal(s.clip(upper=nan, lower=nan).to_numpy(), [1, 2, 3])


# === pandas/tests/series/methods/test_quantile.py ==========================

class TestQuantile:
    @pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_quantile_matches_numpy_percentile_linear(self, q):
        data = list(np.random.default_rng(2).standard_normal(50))
        assert _s(data).quantile(q) == pytest.approx(np.percentile(data, q * 100))

    def test_quantile_invalid_raises(self):
        for invalid in [-1.0, 2.0]:
            with pytest.raises(ValueError, match=r"\[0, 1\]"):
                _s([1.0, 2.0, 3.0]).quantile(invalid)

    def test_quantile_nan(self):
        # GH#13098 — NaN is skipped
        assert _s([1.0, 2.0, 3.0, 4.0, nan]).quantile(0.5) == 2.5
        # all-NaN / empty -> NaN
        assert np.isnan(_s([nan, nan]).quantile(0.5))
        assert np.isnan(_s([]).quantile(0.5))


# === pandas/tests/series/test_cumulative.py ================================

class TestCumsum:
    def test_cumsum_matches_numpy(self):
        data = list(np.random.default_rng(2).standard_normal(20))
        np.testing.assert_allclose(_s(data).cumsum().to_numpy(), np.cumsum(data))

    def test_cumsum_skips_missing(self):
        # ts[::2] = nan; cumsum(ts)[1::2] == cumsum(ts.dropna())
        odd = [1.0, 2.0, 3.0, 4.0, 5.0]
        ts = []
        for v in odd:
            ts += [nan, v]
        got = _s(ts).cumsum().to_numpy()[1::2]
        np.testing.assert_allclose(got, np.cumsum(odd))


# === pandas/tests/reductions/test_reductions.py ============================

class TestAnyAll:
    def test_all_any(self):
        # ts = arange(10); bool = ts > 0
        bool_series = _s(list(range(10))) > 0
        assert bool_series.all() is False
        assert bool_series.any() is True

    def test_all_any_numeric_skipna(self):
        # volas (float model): NaN is skipped, like pandas skipna=True
        assert _s([nan, 1.0]).all() is True
        assert _s([nan, 0.0]).any() is False


class TestIdxMinMax:
    def test_idxmin_idxmax_with_nans(self):
        # Series(range(20), float64) with [5:15] = NaN
        vals = [nan if 5 <= i < 15 else float(i) for i in range(20)]
        s = _s(vals)
        assert s.to_numpy()[s.idxmin()] == np.nanmin(s.to_numpy())
        assert s.to_numpy()[s.idxmax()] == np.nanmax(s.to_numpy())

    def test_idxmin_idxmax_all_na_raises(self):
        with pytest.raises(ValueError, match="Encountered all NA values"):
            _s([nan, nan, nan]).idxmin()
        with pytest.raises(ValueError, match="Encountered all NA values"):
            _s([nan, nan, nan]).idxmax()

    def test_idxmin_idxmax_over_datetime_values(self):
        # Series(date_range("20130102", periods=6)) -> RangeIndex labels
        d = DataFrame(
            {"t": [f"2013-01-0{i}" for i in range(2, 8)], "k": list(range(6))}
        )
        s = volas.to_datetime(d["t"])
        assert s.idxmin() == 0
        assert s.idxmax() == 5


# === pandas/tests/tools/test_to_datetime.py ================================

class TestToDatetimeFormat:
    @pytest.mark.parametrize(
        "arg,expected,fmt",
        [
            ("1/1/2000", "2000-01-01", "%d/%m/%Y"),
            ("1/1/2000", "2000-01-01", "%m/%d/%Y"),
            ("1/2/2000", "2000-02-01", "%d/%m/%Y"),  # day/month -> Feb 1
            ("1/2/2000", "2000-01-02", "%m/%d/%Y"),  # month/day -> Jan 2
            ("1/3/2000", "2000-03-01", "%d/%m/%Y"),
            ("1/3/2000", "2000-01-03", "%m/%d/%Y"),
        ],
    )
    def test_format_disambiguates(self, arg, expected, fmt):
        # test_to_datetime_format_scalar — an explicit format fixes the d/m order.
        got = volas.to_datetime(DataFrame({"t": [arg]})["t"], format=fmt)
        assert got.to_numpy()[0] == np.datetime64(expected, "ns")


# === pandas/tests/io/formats/test_to_csv.py ================================

class TestToCsvFloatFormat:
    def test_printf_float_format(self):
        # printf-style "%.2f" (volas does not implement pandas's new "{:.2f}" form)
        df = DataFrame({"a": [0.123456, 1.0]})
        assert df.to_csv(index=False, float_format="%.2f") == "a\n0.12\n1.00\n"
