"""Lazy pandas interop. pandas is imported only when these are called, so volas
stays pandas-free at import (`to_pandas` is a DataFrame method, also lazy)."""

import numpy as np

_FMT = '%Y-%m-%d %H:%M:%S'


def from_pandas(pdf):
    """Build a volas ``DataFrame`` from a ``pandas.DataFrame``.

    Numeric / bool columns are carried natively; string / object columns become
    string columns; datetime columns are carried as formatted strings (a datetime
    *index* is parsed back to a DatetimeIndex). pandas is imported lazily.
    """
    import pandas as pd  # noqa: PLC0415  (intentional lazy import)
    from volas_rs import DataFrame

    def to_values(s):
        if np.issubdtype(s.dtype, np.datetime64):
            return s.dt.strftime(_FMT).tolist()
        if s.dtype == object:
            return s.tolist()
        return s.to_numpy()

    data = {str(c): to_values(pdf[c]) for c in pdf.columns}

    idx = pdf.index
    if isinstance(idx, pd.RangeIndex):
        return DataFrame(data)

    name = str(idx.name) if idx.name is not None else 'index'
    if np.issubdtype(idx.dtype, np.datetime64):
        merged = {name: idx.strftime(_FMT).tolist(), **data}
        return DataFrame(merged, date_col=name)
    merged = {name: idx.to_numpy(), **data}
    return DataFrame(merged).set_index(name)
