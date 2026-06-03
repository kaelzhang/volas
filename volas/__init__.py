"""volas - a high-performance, Rust-backed columnar kernel for stock /
candlestick (OHLCV) time-series data.
"""

from volas_rs import DataFrame, Series

from importlib.metadata import version as _get_version

__version__ = _get_version('volas')

__all__ = ['DataFrame', 'Series', '__version__']
