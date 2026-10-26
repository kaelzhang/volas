"""Runtime metadata budget: shape storage lives in its managed tensor block."""
import ctypes

import numpy as np
import pytest
from volas import DataFrame

from .test_arrow_interop import _DLManagedTensorVersioned, _DLTensor


class LegacyTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p),
                ("deleter", ctypes.c_void_p)]


@pytest.mark.parametrize("versioned,copy", [(False, None), (True, False), (True, True)])
def test_dlpack_metadata_uses_one_allocation(versioned, copy):
    source = DataFrame({"x": [1.0, 2.0, 3.0]})["x"]
    capsule = source.__dlpack__(max_version=(1, 0) if versioned else None, copy=copy)
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.py_object, ctypes.c_char_p]
    address = get(capsule, b"dltensor_versioned" if versioned else b"dltensor")
    tensor_type = _DLManagedTensorVersioned if versioned else LegacyTensor
    tensor = tensor_type.from_address(address)
    shape = ctypes.addressof(tensor.dl_tensor.shape.contents)
    # The only tail is the keepalive's two pointer words followed by inline shape.
    # A separately allocated shape cannot satisfy this allocation budget.
    assert shape == address + ctypes.sizeof(tensor_type) + 2 * ctypes.sizeof(ctypes.c_void_p)
    assert tensor.dl_tensor.shape[0] == 3
    del source
    assert tensor.dl_tensor.shape[0] == 3


def test_dlpack_consumer_keeps_inline_shape_and_values_alive():
    array = np.from_dlpack(DataFrame({"x": [2.0, 4.0, 6.0]})["x"])
    np.testing.assert_array_equal(array, [2.0, 4.0, 6.0])
