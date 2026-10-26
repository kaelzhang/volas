"""The frozen evaluation inputs are executable and setup never shares state."""

import numpy as np
import pytest

from test.bench_holdout import build, SHAPES


@pytest.mark.parametrize("kind", ["batch", "fold", "interop"])
@pytest.mark.parametrize("shape", SHAPES)
def test_holdout_factories_are_deterministic(kind, shape):
    run, setup = build(kind, shape=shape)
    args, kwargs = setup()
    result = run(*args, **kwargs)
    args2, kwargs2 = setup()
    if args:
        assert args[0] is not args2[0]
    again = run(*args2, **kwargs2)
    if kind == "interop":
        for a, b in zip(result, again):
            np.testing.assert_array_equal(a, b)
    else:
        np.testing.assert_array_equal(result, again)
