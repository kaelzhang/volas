"""volas - a high-performance, Rust-backed columnar kernel for stock /
candlestick (OHLCV) time-series data.
"""

from volas_rs import (
    DataFrame, Series, Row, Timestamp, read_csv, to_datetime, TimeFrame,
    directive_stringify, directive_lookback,
    DirectiveError, DirectiveSyntaxError, DirectiveValueError,
    NA,
)

from importlib.metadata import version as _get_version

__version__ = _get_version('volas')

__all__ = [
    'DataFrame', 'Series', 'Row', 'Timestamp', 'read_csv', 'to_datetime', 'TimeFrame',
    'directive_stringify', 'directive_lookback',
    'NA',
    'DirectiveError', 'DirectiveSyntaxError', 'DirectiveValueError',
    '__version__',
]
