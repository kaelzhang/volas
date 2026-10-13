"""One owner for exact test comparisons: missing masks, then native value bits.

NaNs are missing regardless of payload. Present floats retain their dtype and
zero sign; integers never pass through floating-point conversion.
"""
import numpy as np


def _array(values):
    # NumPy has no native missing bool/int. Keep those Series' typed cells and
    # explicit missing markers instead of exporting them through float64.
    if hasattr(values, 'to_numpy') and str(getattr(values, 'dtype', '')) in ('bool', 'int32', 'int64'):
        if values.isna().any():
            return values.to_numpy(dtype='object')
    return np.asarray(values)


def _missing(a):
    if a.dtype.kind in 'fc':
        return np.isnan(a)
    if a.dtype.kind in 'mM':
        return np.isnat(a)
    if a.dtype.kind == 'O':
        import volas
        return np.array([x is None or x is volas.NA or
                         (isinstance(x, (float, np.floating)) and np.isnan(x))
                         for x in a.flat], dtype=bool).reshape(a.shape)
    return np.zeros(a.shape, dtype=bool)


def exact_mask(got, want):
    """Per-cell equality; arrays must have equal shapes (never broadcast)."""
    got, want = _array(got), _array(want)
    if got.shape != want.shape:
        raise AssertionError(f'shape {got.shape} != {want.shape}')
    gm, wm = _missing(got), _missing(want)
    ok = np.array(gm & wm, dtype=bool)
    present = ~(gm | wm)
    g, w = got[present], want[present]
    if got.dtype.kind == 'O' or want.dtype.kind == 'O':
        ok[present] = [same_exact(a, b) if isinstance(a, (float, np.floating)) or
                       isinstance(b, (float, np.floating)) else a == b
                       for a, b in zip(g, w)]
    elif got.dtype.kind in 'fc' or want.dtype.kind in 'fc':
        if got.dtype == want.dtype:
            # A byte view works for every float width, strided arrays and scalars.
            gb = np.ascontiguousarray(g).view(np.uint8).reshape(-1, got.dtype.itemsize)
            wb = np.ascontiguousarray(w).view(np.uint8).reshape(-1, want.dtype.itemsize)
            ok[present] = (gb == wb).all(axis=1)
    elif got.dtype == want.dtype:
        ok[present] = g == w
    else:
        # Object comparison prevents NumPy promoting mixed signed/unsigned ints
        # to float64 and collapsing integers above 2**53.
        ok[present] = g.astype(object) == w.astype(object)
    return ok


def same_exact(got, want):
    got, want = _array(got), _array(want)
    if got.shape != want.shape:
        return False
    return bool(exact_mask(got, want).all())


def assert_exact(got, want, err_msg=''):
    got, want = _array(got), _array(want)
    gm, wm = _missing(got), _missing(want)
    assert got.shape == want.shape, f'{err_msg}: shape {got.shape} != {want.shape}'
    assert np.array_equal(gm, wm), f'{err_msg}: missing masks differ'
    ok = exact_mask(got, want)
    if not ok.all():
        i = int(np.flatnonzero(~ok)[0])
        raise AssertionError(f'{err_msg}: cell {i}: got {got.flat[i]!r} want {want.flat[i]!r} '
                             f'(dtype {got.dtype}/{want.dtype}, bits differ)')
