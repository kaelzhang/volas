"""Exhaustive source × composition × target × fill × entry export contract.

Oracle: NumPy astype + result_type on the SAME per-column arrays, never on
volas exports. Deliberate policies are cited in docs/interop.md#numpy-export-rules
and docs/series-api.md#series-to-numpy. Naive / aware sources share UTC ns
storage per docs/interop.md#datetime-columns-are-utc-instants.
"""
from dataclasses import dataclass
from datetime import timezone, timedelta
from itertools import product
import warnings

import numpy as np
import pytest

from volas import DataFrame, NA, Timestamp


@dataclass
class Source:
    name: str
    native: np.ndarray
    mask: np.ndarray
    input: object

    @property
    def kind(self):
        return self.native.dtype.kind

    def objects(self, default_string=False):
        result = self.native.astype(object)
        if self.kind == 'M':
            result = np.array([NA if m else Timestamp(int(v), tz='UTC')
                               for v, m in zip(self.native.view('i8'), self.mask)], dtype=object)
        elif self.kind in 'iUb':
            result[self.mask] = None if default_string and self.kind == 'U' else NA
        return result

    def default(self):
        if self.kind == 'b' and self.mask.any():
            raise ValueError('bool NA needs a meaning')
        if self.kind == 'i' and self.mask.any():
            result = self.native.astype('f8')
            result[self.mask] = np.nan
            return result
        if self.kind == 'U':
            return self.objects(default_string=True)
        return self.native.copy()


def sources():
    result = []
    for dtype in ('f4', 'f8', 'i4', 'i8', 'bool', 'str', 'str-number', 'str-date', 'str-time', 'naive', 'aware'):
        for state in ('dense', 'partial', 'empty-valid'):
            mask = np.array([False, state != 'dense', state == 'empty-valid'])
            if state == 'empty-valid':
                mask[:] = True
            if dtype in ('naive', 'aware'):
                native = np.array(['2026-01-01T00:00:00.000000123',
                                   '2000-02-29T12:00:00.000000456',
                                   '1970-01-01T00:00:00.000000007'], dtype='M8[ns]')
                native[mask] = 'NaT'
                values = native.copy()
                if dtype == 'aware':
                    # UTC+8 datetime values ingest as the same UTC instants.
                    zone = timezone(timedelta(hours=8))
                    values = [None if m else Timestamp(int(v), tz='UTC').to_pydatetime().astimezone(zone)
                              for v, m in zip(native.view('i8'), mask)]
                    # Python datetime has microsecond resolution; pin the independent
                    # source to those exact instants before constructing the frame.
                    native = np.array([np.datetime64('NaT', 'ns') if v is None else
                                       np.datetime64(v.astimezone(timezone.utc).replace(tzinfo=None), 'ns')
                                       for v in values])
                    if mask.all():
                        values = native.copy()  # all-None has no inferable dtype
            else:
                kind = np.dtype('str' if dtype.startswith('str') else dtype).kind
                cells = [1.5, -0., 2.25] if kind == 'f' else [2**53 + 1, 0, -3] if dtype == 'i8' else [1, 0, -3]
                if dtype == 'str':
                    cells = ['1', 'é', 'long text']
                elif dtype == 'str-number':
                    cells = ['1', '0', '-3']
                elif dtype == 'str-date':
                    cells = ['2026-01-01', '2000-02-29', '1970-01-01']
                elif dtype == 'str-time':
                    cells = ['2026-01-01T00:00:00.000000123',
                             '2000-02-29T12:00:00.000000456',
                             '1970-01-01T00:00:00.000000007']
                native = np.array(cells, dtype='str' if dtype.startswith('str') else dtype)
                values = np.ma.array(native, mask=mask)
                if kind == 'f':
                    native = native.copy()
                    native[mask] = np.nan
            result.append(Source(f'{dtype}-{state}', native, mask, values))
    return result


SOURCES = sources()
# Every numeric kind/width supported by this NumPy (including long double and
# complex), every time unit, flexible sized/unsized types; extra layouts guard
# against losing subarray trailing dimensions or record fields.
TARGETS = [None] + sorted({np.dtype(t).str for t in np.sctypeDict.values()
                          if np.dtype(t).kind in 'biufc'} | {
    'datetime64', 'timedelta64', 'U', 'U0', 'U2', 'U80', 'S', 'S0', 'S2', 'S80',
    'V', 'V0', 'V2', 'V8', 'V16', 'object', '>i4', '>f8', '>c16',
    '(2,)i4', 'i4,f8', 'T',
} | {f'{endian}{kind}8[{unit}]' for endian, kind, unit in product(('', '>'), 'Mm',
      ('Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as'))})
COMPOSITIONS = [(s,) for s in SOURCES] + list(product(SOURCES, repeat=2)) + [tuple(SOURCES)]


def fill_for(dtype):
    kind = np.dtype(dtype).kind if dtype is not None else 'O'
    if kind == 'M':
        return np.datetime64('2000-01-01', 'ns')
    if kind == 'm':
        return np.timedelta64(7, 'ns')
    if kind in 'US':
        return 'missing value longer than native numeric widths'
    if kind == 'V' and np.dtype(dtype).fields is None and np.dtype(dtype).subdtype is None:
        return b'ab'
    return 0


def default_column(source, fill, filled):
    if source.kind == 'b' and source.mask.any() and filled:
        result = source.objects()
    else:
        result = source.default()
    if filled and source.mask.any():
        result[source.mask] = fill
    return result


def reference(columns, dtype, filled, frame):
    fill = fill_for(dtype)
    if dtype is None:
        if not frame:
            return default_column(columns[0], fill, filled)
        kinds = {s.kind for s in columns}
        if kinds <= set('ifb'):
            arrays = [s.native.astype('f8') for s in columns]
            for a, s in zip(arrays, columns):
                a[s.mask] = np.nan
        elif kinds == {'M'}:
            arrays = [s.native.copy() for s in columns]
        else:
            arrays = [s.objects() for s in columns]
        if filled:
            for a, s in zip(arrays, columns):
                a[s.mask] = fill
        return np.column_stack(arrays)

    target = np.dtype(dtype)
    if frame and target.kind in 'biufcMm' and any(s.kind == 'U' for s in columns):
        raise ValueError('string column has no numeric meaning')
    arrays, holes = [], []
    for s in columns:
        if not filled and s.mask.any():
            if target.kind in 'biu' and not (frame and s.kind == 'M'):
                raise ValueError('NA has no integer or boolean representation')
            if s.kind == 'b' and target.kind not in 'fcO':
                raise ValueError('bool NA needs a meaning')
        # The oracle converts native valid slots; it never calls default() for
        # explicit targets. Missing representations are independent source data.
        native = s.objects() if target.kind == 'O' else s.native
        if s.kind == 'U' and target.kind != 'O':
            native = native.astype(object)  # storage is variable-width text
        arrays.append(native[~s.mask] if s.mask.any() else native)
        if filled:
            holes.append(None)
        elif target.kind == 'O':
            holes.append(s.objects()[s.mask])
        elif s.kind in 'ib' or (frame and s.kind == 'M' and target.kind in 'fc'):
            holes.append(np.full(s.mask.sum(), np.nan))
        elif s.kind == 'U':
            holes.append(np.full(s.mask.sum(), None, dtype=object))
        else:
            holes.append(s.native[s.mask])

    # NumPy determines source-dependent target metadata; combine metadata before
    # converting any numeric epoch count into the agreed time unit. Subarray
    # astype expands dimensions, so its requested descriptor must remain intact.
    if target.itemsize == 0 or (target.kind in 'Mm' and np.datetime_data(target)[0] == 'generic'):
        resolved = np.result_type(*(a.astype(target).dtype for a in arrays))
        if target.kind in 'US' and any(s.mask.any() for s in columns):
            descriptors = [np.asarray(fill, dtype=target).dtype] if filled else [
                h.astype(target).dtype for h, s in zip(holes, columns) if s.mask.any()]
            resolved = np.result_type(resolved, *descriptors)
    else:
        resolved = target
    outputs = []
    for a, h, s in zip(arrays, holes, columns):
        cast = a.astype(resolved)
        if s.mask.any():
            output = np.empty((len(s.mask),) + cast.shape[1:], dtype=cast.dtype)
            output[~s.mask] = cast
            output[s.mask] = fill if filled else h.astype(resolved)
        else:
            output = cast
        outputs.append(output)
    # stack(axis=1), not column_stack: preserve expanded subarray axes.
    return np.stack(outputs, axis=1, dtype=outputs[0].dtype) if frame else outputs[0]


def assert_export(got, expected):
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    assert got.dtype.str == expected.dtype.str
    assert got.dtype.itemsize == expected.dtype.itemsize
    if got.dtype.kind in 'Mm':
        assert np.datetime_data(got.dtype) == np.datetime_data(expected.dtype)
    if got.dtype.kind in 'MmV':
        assert got.tobytes() == expected.tobytes()
    elif got.dtype.kind == 'O':
        for g, e in zip(got.flat, expected.flat):
            if e is NA or e is None:
                assert g is e
            elif isinstance(e, float) and np.isnan(e):
                assert isinstance(g, float) and np.isnan(g)
            else:
                assert type(g) is type(e) and g == e
    else:
        np.testing.assert_array_equal(got, expected)
        if got.dtype.kind in 'fc':
            # Values include signed zero. Ignore platform-specific longdouble
            # padding bytes while checking every numeric component's sign.
            np.testing.assert_array_equal(np.signbit(got.real), np.signbit(expected.real))
            if got.dtype.kind == 'c':
                np.testing.assert_array_equal(np.signbit(got.imag), np.signbit(expected.imag))


@pytest.mark.parametrize('dtype', TARGETS, ids=lambda d: str(d))
@pytest.mark.parametrize('filled', [False, True], ids=['default-na', 'explicit-na'])
def test_numpy_export_matrix(dtype, filled):
    failures = []
    for columns in COMPOSITIONS:
        frame = DataFrame({f'x{i}': s.input for i, s in enumerate(columns)})
        entries = [('frame', columns, frame.to_numpy)]
        for i, s in enumerate(columns):
            entries.append((f'series{i}', (s,), frame[f'x{i}'].to_numpy))
            entries.append((f'exec{i}', (s,), lambda i=i, **kw: frame.exec(f'x{i}', **kw)))
        for entry, cols, call in entries:
            kwargs = {'dtype': dtype}
            if filled:
                kwargs['na_value'] = fill_for(dtype)
            with warnings.catch_warnings(), np.errstate(all='raise'):
                warnings.simplefilter('error')
                try:
                    try:
                        expected = reference(cols, dtype, filled, entry == 'frame')
                    except (TypeError, ValueError, OverflowError, FloatingPointError, RuntimeError) as error:
                        with pytest.raises(type(error)):
                            call(**kwargs)
                    else:
                        assert_export(call(**kwargs), expected)
                except (Exception, pytest.fail.Exception) as error:
                    failures.append(f'{tuple(s.name for s in columns)} {entry}: {type(error).__name__}: {error}')
    assert not failures, f'{len(failures)} mismatches:\n' + '\n'.join(failures[:40])
