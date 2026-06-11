"""Lazy pandas interop. pandas is imported only when these are called, so volas
stays pandas-free at import (`to_pandas` is a DataFrame method, also lazy)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from volas_rs import DataFrame


def from_pandas(pdf: Any) -> DataFrame:
    """Build a volas ``DataFrame`` from a ``pandas.DataFrame``.

    Numeric / bool columns are carried natively, with a pandas **nullable** column
    (``Int64`` / ``boolean`` / ``string``) read back as the volas dtype + ``volas.NA``.
    A pandas ``object`` column of plain strings is read as a ``str`` column; volas has
    **no** ``object`` dtype, so an object column mixing strings with other values
    (e.g. ``np.nan``) is rejected rather than silently widened. Datetime columns and a
    datetime *index* are carried natively as ``datetime64[ns]`` instants (no string
    round-trip), and a tz-aware index keeps its zone for display. pandas is imported
    lazily, so volas stays pandas-free at import.

    Args:
        pdf (pandas.DataFrame): the source frame.

    Usage::

        import pandas as pd, volas
        vdf = volas.from_pandas(pd.read_csv('ohlcv.csv', index_col='time',
                                            parse_dates=['time']))

    Returns:
        DataFrame: the equivalent volas frame (the inverse of ``df.to_pandas()``).
    """
    import pandas as pd  # noqa: PLC0415  (intentional lazy import)
    from volas_rs import DataFrame

    def to_values(s: Any) -> Any:
        # datetime (naive or tz-aware) -> native UTC datetime64[ns]; no strftime round-trip.
        if pd.api.types.is_datetime64_any_dtype(s.dtype):
            return s.to_numpy(dtype='datetime64[ns]')
        # A nullable masked int / boolean / string column passes through the typed
        # masked channel `(values, missing_mask, dtype)`, so volas keeps the
        # DECLARED dtype + volas.NA even when the column is all-NA or empty (F9)
        # and an Int32 stays int32 (F10) — the faithful inverse of
        # to_pandas(dtype_backend='numpy_nullable') and the pandas 3.0 str dtype.
        if pd.api.types.is_extension_array_dtype(s.dtype) and (
            pd.api.types.is_integer_dtype(s.dtype)
            or pd.api.types.is_bool_dtype(s.dtype)
            or pd.api.types.is_string_dtype(s.dtype)
        ):
            mask = s.isna().tolist()
            if pd.api.types.is_integer_dtype(s.dtype):
                volas_dtype = 'int32' if str(s.dtype) == 'Int32' else 'int64'
                filler: Any = 0
            elif pd.api.types.is_bool_dtype(s.dtype):
                volas_dtype, filler = 'bool', False
            else:
                volas_dtype, filler = 'str', ''
            values = [filler if m else v for v, m in zip(s.tolist(), mask)]
            return (values, mask, volas_dtype)
        if s.dtype == object:
            return s.tolist()
        return s.to_numpy()

    data = {str(c): to_values(pdf[c]) for c in pdf.columns}

    idx = pdf.index
    if isinstance(idx, pd.RangeIndex):
        return DataFrame(data)

    name = str(idx.name) if idx.name is not None else 'index'
    if pd.api.types.is_datetime64_any_dtype(idx.dtype):
        # Carry the absolute instants natively (UTC), set them as the index, then re-attach
        # the source zone for display (a tz-naive index stays UTC-default).
        tz = getattr(idx, 'tz', None)
        merged = {name: idx.to_numpy(dtype='datetime64[ns]'), **data}
        df = DataFrame(merged).set_index(name)
        if tz is None:
            return df
        # pandas renders a fixed offset as 'UTC+08:00'; volas's tz name for that is '+08:00'.
        tzname = str(tz)
        if tzname.startswith(('UTC+', 'UTC-')):
            tzname = tzname[3:]
        # the imported instants are already true UTC — tag the zone directly
        # (tz_convert would hit the naive-axis guard; localize would shift).
        return df._set_index_tz(tzname)
    merged = {name: idx.to_numpy(), **data}
    return DataFrame(merged).set_index(name)
