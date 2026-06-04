"""volas - a high-performance, Rust-backed columnar kernel for stock /
candlestick (OHLCV) time-series data.
"""

from volas_rs import (
    DataFrame, Series, Row, read_csv, TimeFrame, Cumulator,
    DirectiveError, DirectiveSyntaxError, DirectiveValueError,
)

from ._rolling import rolling_calc

from importlib.metadata import version as _get_version

__version__ = _get_version('volas')

__all__ = [
    'DataFrame', 'Series', 'Row', 'read_csv', 'TimeFrame', 'Cumulator',
    'rolling_calc',
    'DirectiveError', 'DirectiveSyntaxError', 'DirectiveValueError',
    '__version__',
]
