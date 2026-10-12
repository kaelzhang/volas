"""Observation without refreshing the frame under test.

Every check here reads either through a documented non-refreshing path
(``columns`` / ``dtypes`` / ``shape`` / ``index`` / ``is_computed`` /
``get_column``) or through a ``copy()`` taken for validation and refreshed there —
never by refreshing the frame itself, so the op schedule under test stays exactly
the generated one.
"""

from __future__ import annotations

import math

import numpy as np

import volas

from . import contracts

STALE = 'stale computed'


class Mismatch(AssertionError):
    pass


def is_stale(df: volas.DataFrame) -> bool:
    """Whether a bulk read raises the staleness error (a non-refreshing probe)."""
    try:
        df.iloc[0:0]
    except ValueError as e:
        if STALE in str(e):
            return True
        raise
    return False


def structure(df: volas.DataFrame) -> None:
    """§5.4 structural assertions observable from Python: every column, the index and
    every validity mask have the frame's length; the name map, dtype map, iteration
    order and membership agree; the physical height bounds the visible one."""
    cols = df.columns
    n = len(df)
    assert len(set(cols)) == len(cols), f'duplicate column names {cols}'
    assert df.shape == (n, len(cols)), f'shape {df.shape} != ({n}, {len(cols)})'
    assert list(df) == cols, 'iteration order differs from columns'
    assert list(df.dtypes) == cols, f'dtypes keys {list(df.dtypes)} != columns {cols}'
    assert len(df.index) == n, f'index length {len(df.index)} != height {n}'
    if hasattr(df, '_physical_height'):
        assert df._physical_height >= n, 'physical height below the visible height'
    for c in cols:
        assert c in df, f'{c!r} listed but not a member'
        s = df.get_column(c)
        assert len(s) == n, f'column {c!r} has {len(s)} rows, frame {n}'
        assert len(s.isna().to_numpy()) == n, f'validity of {c!r} has the wrong length'
        assert s.dtype == df.dtypes[c], f'{c!r} dtype {s.dtype} != frame dtype {df.dtypes[c]}'
        if hasattr(df, 'is_computed'):  # absent before the computed-column API existed
            df.is_computed(c)
        representation(s, c)


def representation(s: volas.Series, c: str) -> None:
    """The NA model at the NumPy boundary (README "series.to_numpy"): a dense int
    column exports exactly as int64 (its validity holds no stale "has nulls" mark), and
    a bool column with a missing cell exports as an object array of its cells."""
    missing = s.isna().to_numpy()
    if s.dtype in ('int64', 'int32') and not missing.any():
        exported = s.to_numpy()
        assert exported.dtype == np.dtype(s.dtype), \
            f'dense int column {c!r} exported as {exported.dtype} (a stale "has nulls" mark)'
    if s.dtype == 'bool' and missing.any():
        out = s.to_numpy()
        assert out.dtype == object, f'bool column {c!r} with NA exported as {out.dtype}'
        assert all(isinstance(v, float) and math.isnan(v) for v in out[missing].tolist()), \
            f'bool column {c!r}: a missing cell did not export as NaN'


def raw_state(df: volas.DataFrame) -> dict:
    """Everything observable without refreshing: schema, index, cache flags, staleness,
    and every column's stored cells (stale placeholders included) and NA mask."""
    state = {
        'columns': tuple(df.columns),
        'dtypes': dict(df.dtypes),
        'shape': df.shape,
        'physical': getattr(df, '_physical_height', None),
        'index': _bytes(np.asarray(df.index)),
        'tz': df.tz,
        'computed': tuple(df.is_computed(c) for c in df.columns),
        'stale': is_stale(df),
    }
    for c in df.columns:
        s = df.get_column(c)
        state[c] = (_bytes(s.to_numpy(dtype='object') if s.dtype == 'str' else s.to_numpy()),
                    _bytes(s.isna().to_numpy()))
    return state


def fresh_view(df: volas.DataFrame) -> dict:
    """Every column's values after refreshing a *copy* (the frame is untouched)."""
    view = df.copy()
    view.fulfill()
    return {c: _bytes(_column(view, c)) for c in view.columns}


def _column(df: volas.DataFrame, c: str) -> np.ndarray:
    s = df.get_column(c)
    return s.to_numpy(dtype='object') if s.dtype in ('str', 'bool', 'int64', 'int32', 'datetime64[ns]') \
        else s.to_numpy()


def _bytes(a: np.ndarray):
    if a.dtype == object:
        return tuple(_cell(v) for v in a.tolist())
    return (a.dtype.str, a.tobytes())


def _cell(v):
    if v is volas.NA or v is None:
        return ('NA',)
    if isinstance(v, float) and math.isnan(v):
        return ('nan',)
    if isinstance(v, volas.Timestamp):
        return ('ts', v.value)
    return (type(v).__name__, v)


def diff_state(before: dict, after: dict) -> list[str]:
    return [k for k in before if before[k] != after.get(k)] + [k for k in after if k not in before]


# --- value comparison --------------------------------------------------------------


def column_values(df: volas.DataFrame, c: str) -> np.ndarray:
    """A column of an already-fresh frame as comparable values (NA -> None / NaN)."""
    s = df.get_column(c)
    if s.dtype in ('float64', 'float32'):
        return s.to_numpy()
    # the exact typed cells (an int past 2^53 stays exact, a missing cell is NA)
    return np.array([None if (v is volas.NA or v is None) else _scalar(v)
                     for v in s.to_list()], dtype=object)


def column_values_series(s: volas.Series) -> np.ndarray:
    """A directive result Series as comparable values: bool / float arrays as they
    are, any other dtype through :func:`column_values`' cell form."""
    if s.dtype in ('float64', 'float32', 'bool') and not (s.dtype == 'bool' and s.isna().any()):
        return s.to_numpy()
    return np.array([None if (v is volas.NA or v is None) else _scalar(v)
                     for v in s.to_numpy(dtype='object').tolist()], dtype=object)


def _scalar(v):
    if isinstance(v, float) and math.isnan(v):
        return None  # a non-float column's NA exported through the float channel
    if isinstance(v, volas.Timestamp):
        return v.value
    if isinstance(v, np.datetime64):
        return None if np.isnat(v) else int(v.astype('datetime64[ns]').astype('int64'))
    if isinstance(v, (np.integer, np.bool_)):
        return v.item()
    return v


def expected_values(values: np.ndarray, dtype: str) -> np.ndarray:
    """Ledger values in :func:`column_values`' representation."""
    if dtype in ('float64', 'float32'):
        return np.asarray(values, dtype=dtype)
    return np.array([None if v is None else (bool(v) if dtype == 'bool' else v) for v in values], dtype=object)


def same_data(got: np.ndarray, want: np.ndarray, what: str) -> None:
    if got.shape != want.shape:
        raise Mismatch(f'{what}: shape {got.shape} != {want.shape}')
    if got.dtype == object or want.dtype == object:
        bad = [i for i, (a, b) in enumerate(zip(got.tolist(), want.tolist())) if a != b]
    else:
        bad = np.flatnonzero(~((got == want) | (np.isnan(got) & np.isnan(want)))).tolist()
    if bad:
        i = bad[0]
        raise Mismatch(f'{what}: {len(bad)} cell(s) differ, first at row {i}: got {got[i]!r} want {want[i]!r}')


def same_indicator(got: np.ndarray, want: np.ndarray, what: str, tolerance: float = 0.0,
                   excused: np.ndarray | None = None) -> None:
    """Bit-exact (NaN == NaN) unless ``tolerance`` > 0, then relative to the column's
    finite scale; rows in ``excused`` (a documented tie) are not judged."""
    if got.shape != want.shape:
        raise Mismatch(f'{what}: shape {got.shape} != {want.shape}')
    if got.dtype == np.bool_ or want.dtype == np.bool_:
        ok = got == want
    else:
        g, w = got.astype(float), want.astype(float)
        ok = (g == w) | (np.isnan(g) & np.isnan(w))
        if tolerance > 0:
            finite = np.isfinite(w) & np.isfinite(g)
            scale = np.max(np.abs(w[finite]), initial=1.0)
            with np.errstate(invalid='ignore', over='ignore'):
                ok |= finite & (np.abs(np.where(finite, g - w, 0.0)) <= tolerance * scale)
    if excused is not None:
        ok |= excused
    if not ok.all():
        i = int(np.flatnonzero(~ok)[0])
        raise Mismatch(f'{what}: {int((~ok).sum())} row(s) differ, first at row {i}: '
                       f'got {got[i]!r} want {want[i]!r}')


def tie_rows(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Rows where a comparison's operands are equal within the ULP-drift tolerance."""
    lhs, rhs = np.asarray(lhs, float), np.asarray(rhs, float)
    finite = np.isfinite(lhs) & np.isfinite(rhs)
    scale = np.max(np.abs(np.concatenate([lhs[finite], rhs[finite]])), initial=1.0)
    return finite & (np.abs(lhs - rhs) <= contracts.ULP_DRIFT_TOLERANCE * scale)
