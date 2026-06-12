"""Contract O5 (resolved -> reject) — a boolean mask / condition carrying any
volas.NA is rejected at every masking surface, instead of silently reading the
NA as False. A missing condition is an *unknown* signal; treating it as a
deliberate False would drop a row (filtering) or fill it in the False direction
(where / mask) — in a live system that turns a data gap into a trade signal. The
user must fill or drop the NA first. A dense bool mask is unaffected."""

import pytest
import volas
from volas import DataFrame

NA = volas.NA


def _na_mask():
    return DataFrame({"m": [True, NA, False]})["m"]


def _na_cond_frame():
    return DataFrame({"x": [True, NA, False]})


def _s():
    return DataFrame({"x": [1, 2, 3]})["x"]


def _df():
    return DataFrame({"x": [10, 20, 30]})


def test_series_getitem_na_mask_rejected():
    with pytest.raises(ValueError):
        _s()[_na_mask()]


def test_dataframe_getitem_na_mask_rejected():
    with pytest.raises(ValueError):
        _df()[_na_mask()]


def test_series_where_na_cond_rejected():
    with pytest.raises(ValueError):
        _s().where(_na_mask(), 0)


def test_series_mask_na_cond_rejected():
    with pytest.raises(ValueError):
        _s().mask(_na_mask(), 0)


def test_dataframe_where_na_cond_rejected():
    with pytest.raises(ValueError):
        _df().where(_na_cond_frame(), 0)


def test_dataframe_mask_na_cond_rejected():
    with pytest.raises(ValueError):
        _df().mask(_na_cond_frame(), 0)


def test_row_mask_assignment_na_rejected():
    df = _df()
    with pytest.raises(ValueError):
        df[_na_mask()] = 0


def test_cell_mask_assignment_na_rejected():
    df = _df()
    with pytest.raises(ValueError):
        df[_na_cond_frame()] = 0


def test_dataframe_iloc_na_mask_rejected():
    # DataFrame.iloc accepts a bool mask (Series.iloc is int/slice only); it too
    # rejects an NA mask via the shared as_bool_mask chokepoint.
    with pytest.raises(ValueError):
        _df().iloc[_na_mask()]


# --- a dense (no-NA) bool mask is unaffected --------------------------------

def test_dense_bool_mask_still_works():
    s, df = _s(), _df()
    good = DataFrame({"m": [True, False, True]})["m"]
    good_f = DataFrame({"x": [True, False, True]})
    assert s[good].to_list() == [1, 3]
    assert df[good].shape == (2, 1)
    assert s.where(good, 0).to_list() == [1, 0, 3]
    assert df.where(good_f, 0)["x"].to_list() == [10, 0, 30]
