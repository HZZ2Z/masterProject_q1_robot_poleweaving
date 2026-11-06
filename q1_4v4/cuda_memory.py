"""Release CUDA graph allocations independently of Torch/Warp allocator caches."""
import ctypes
from functools import lru_cache
import torch


@lru_cache(maxsize=1)
def _driver():
    driver=ctypes.CDLL('libcuda.so.1')
    driver.cuDeviceGet.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int]
    driver.cuDeviceGet.restype=ctypes.c_int
    driver.cuDeviceGraphMemTrim.argtypes=[ctypes.c_int]
    driver.cuDeviceGraphMemTrim.restype=ctypes.c_int
    return driver


def trim_graph_memory(device):
    """Call after graph destruction and synchronization, only at scene rebuilds.

    Graph allocation nodes reserve a separate pool that empty_cache cannot trim.
    https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html
    """
    device=torch.device(device)
    if device.type!='cuda':raise ValueError('Graph memory trimming requires a CUDA device')
    driver=_driver();handle=ctypes.c_int()
    ordinal=torch.cuda.current_device() if device.index is None else device.index
    status=driver.cuDeviceGet(ctypes.byref(handle),ordinal)
    if status:raise RuntimeError(f'cuDeviceGet failed with CUDA status {status}')
    status=driver.cuDeviceGraphMemTrim(handle.value)
    if status:raise RuntimeError(f'cuDeviceGraphMemTrim failed with CUDA status {status}')
