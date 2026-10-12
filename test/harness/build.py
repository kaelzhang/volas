"""Turn ledger rows into volas objects — frames, bar frames, Rows and bar dicts.

Typed columns go through Arrow (``DataFrame.from_arrow``) so a missing cell keeps its
dtype (a Python ``[None]`` would build a float column); float-only bars may instead
take the NumPy constructor path, and a bar may be a slice of a larger frame — so
appended data arrives in owned, foreign and sliced storage.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

import volas

from .data import Schema

INDEX_NAME = 'time'

_ARROW = {
    'float64': pa.float64(), 'float32': pa.float32(), 'int64': pa.int64(), 'int32': pa.int32(),
    'bool': pa.bool_(), 'str': pa.string(), 'datetime64[ns]': pa.timestamp('ns'),
}


def _arrow_column(values: list, dtype: str) -> pa.Array:
    if dtype in ('float64', 'float32'):
        # A float NA is NaN in volas; Arrow nulls would also be accepted, but NaN is
        # the representation the frame itself stores.
        return pa.array([np.nan if v is None else v for v in values], type=_ARROW[dtype])
    return pa.array(values, type=_ARROW[dtype])


def frame(schema: Schema, rows: list[dict], index: list[int] | None, columns=None,
          storage: str = 'arrow', dtypes: dict[str, str] | None = None) -> volas.DataFrame:
    """A frame of ``rows`` (``columns`` defaults to the schema's) indexed by the
    epoch-ns ``index`` (datetime schema) or a RangeIndex (``index=None``);
    ``dtypes`` overrides the schema's dtype of some columns."""
    columns = tuple(columns or schema.columns)
    dtype = {c: (dtypes or {}).get(c) or schema.dtype(c) for c in columns}
    if storage == 'numpy' and all(dtype[c] in ('float64', 'float32') for c in columns):
        data = {c: np.array([np.nan if r[c] is None else r[c] for r in rows], dtype=dtype[c])
                for c in columns}
        if index is None:
            return volas.DataFrame(data)
        data = {INDEX_NAME: np.array(index, dtype='datetime64[ns]'), **data}
        return _tz(volas.DataFrame(data).set_index(INDEX_NAME), schema)
    arrays = {c: _arrow_column([_cast(r[c], dtype[c]) for r in rows], dtype[c]) for c in columns}
    if index is None:
        return volas.DataFrame.from_arrow(pa.table(arrays))
    arrays = {INDEX_NAME: pa.array(index, type=pa.timestamp('ns')), **arrays}
    return _tz(volas.DataFrame.from_arrow(pa.table(arrays)).set_index(INDEX_NAME), schema)


def _cast(value, dtype: str):
    if value is None:
        return None
    if dtype in ('int64', 'int32') and isinstance(value, float):
        bound = 2**31 - 1 if dtype == 'int32' else 2**62
        return int(np.clip(value, -bound, bound)) if np.isfinite(value) else None
    if dtype in ('float64', 'float32') and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _tz(df: volas.DataFrame, schema: Schema) -> volas.DataFrame:
    # `_set_index_tz` tags the absolute (UTC) instants with the zone without moving
    # them (tz_localize would reinterpret them as wall times).
    return df._set_index_tz(schema.tz) if schema.tz else df


def bar_dict(schema: Schema, row: dict, ts: int | None, extra: tuple[str, ...] = (),
             omit: tuple[str, ...] = ()) -> dict:
    """The scalar-bar dict form of one row: every data column but ``omit`` present (the
    ``extra`` plain columns as NA)."""
    out: dict = {}
    if ts is not None:
        out[INDEX_NAME] = volas.Timestamp(ts, tz='UTC').tz_convert(schema.tz) if schema.tz else np.datetime64(ts, 'ns')
    for c in extra:
        out[c] = None
    for c in schema.columns:
        if c in omit:
            continue
        v = row[c]
        if v is not None and schema.dtype(c) == 'datetime64[ns]':
            v = np.datetime64(v, 'ns')
        out[c] = v
    return out


def bars(schema: Schema, rows: list[dict], index: list[int] | None, form: str, pad: int,
         extra: tuple[str, ...] = (), omit: tuple[str, ...] = ()):
    """``rows`` in the requested append form: 'arrow' / 'numpy' frame, 'sliced' (a view
    into a larger frame), 'row' (a ``Row``, one bar), or 'dict' (one bar; ``extra``
    names plain columns the frame has beyond the schema); the data columns ``omit``
    left out."""
    columns = tuple(c for c in schema.columns if c not in omit)
    if form in ('arrow', 'numpy'):
        return frame(schema, rows, index, columns=columns, storage=form)
    if form == 'sliced':
        filler = rows[:1] * pad
        fidx = None if index is None else [index[0] - (pad - i) for i in range(pad)]
        big = frame(schema, filler + rows + filler, None if index is None else fidx + index + [
            index[-1] + 1 + i for i in range(pad)], columns=columns)
        return big.iloc[pad:pad + len(rows)]
    if form == 'row':
        return frame(schema, rows[:1], None if index is None else index[:1], columns=columns).iloc[0]
    return bar_dict(schema, rows[0], None if index is None else index[0], extra, omit)
