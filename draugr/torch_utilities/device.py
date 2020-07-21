#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 15/11/2019
           """

DEVICE = None

__all__ = [
    "global_torch_device",
    "select_cuda_device",
    "get_gpu_usage_mb",
    "auto_select_available_cuda_device",
    "set_global_torch_device",
]


def global_torch_device(
    cuda_if_available: bool = None, override: torch.device = None, verbose: bool = False
) -> torch.device:
    """

first time call stores to device for global reference, later call must manually override

  :param verbose:
  :type verbose:
:param cuda_if_available:
:type cuda_if_available:
:param override:
:type override:
:return:
:rtype:
"""
    global DEVICE

    if override is not None:
        DEVICE = override
        if verbose:
            print(f"Overriding global torch device to {override}")
    elif cuda_if_available is not None:
        d = torch.device(
            "cuda" if torch.cuda.is_available() and cuda_if_available else "cpu"
        )
        if DEVICE is None:
            DEVICE = d
        return d
    elif DEVICE is None:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() and True else "cpu")

    return DEVICE


def set_global_torch_device(device: torch.device) -> None:
    global DEVICE
    DEVICE = device


def select_cuda_device(cuda_device_idx: int) -> torch.device:
    """

:param cuda_device_idx:
:type cuda_device_idx:
:return:
:rtype:
"""
    num_cuda_device = torch.cuda.device_count()
    assert num_cuda_device > 0
    assert cuda_device_idx < num_cuda_device
    if 0 <= cuda_device_idx < num_cuda_device:
        return torch.device(f"cuda:{cuda_device_idx}")


def get_gpu_usage_mb():
    """

:return:
:rtype:
"""

    import subprocess

    """Get the current gpu usage.

Returns
-------
usage: dict
Keys are device ids as integers.
Values are memory usage as integers in MB.
"""
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
    ).decode("utf-8")
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def destroy() -> None:
    r"""**Destroy cuda state by emptying cache and collecting IPC.**

   Consecutively calls `torch.cuda.empty_cache()` and `torch.cuda.ipc_collect()`.
  """

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def auto_select_available_cuda_device(
    expected_memory_usage_mb: int = 1024
) -> torch.device:
    r"""
    Auto selects the device with highest compute capability and with the requested memory available

:param expected_memory_usage_mb:
:type expected_memory_usage_mb:
:return:
:rtype:
"""

    num_cuda_device = torch.cuda.device_count()
    assert num_cuda_device > 0
    """
print(torch.cuda.cudart())
print(torch.cuda.memory_snapshot())
torch.cuda.memory_cached(dev_idx),
torch.cuda.memory_allocated(dev_idx),
torch.cuda.max_memory_allocated(dev_idx),
torch.cuda.max_memory_cached(dev_idx),
torch.cuda.get_device_name(dev_idx),
torch.cuda.get_device_properties(dev_idx),
torch.cuda.memory_stats(dev_idx)
"""
    preferred_idx = None
    highest_capab = 0
    for dev_idx, usage in enumerate(get_gpu_usage_mb().values()):
        cuda_capab = float(
            ".".join([str(x) for x in torch.cuda.get_device_capability(dev_idx)])
        )
        if expected_memory_usage_mb:
            total_mem = (
                torch.cuda.get_device_properties(dev_idx).total_memory // 1000 // 1000
            )
            if expected_memory_usage_mb < total_mem - usage:
                if cuda_capab > highest_capab:
                    highest_capab = cuda_capab
                    preferred_idx = dev_idx
        else:
            if cuda_capab > highest_capab:
                highest_capab = cuda_capab
                preferred_idx = dev_idx

    return select_cuda_device(preferred_idx)


if __name__ == "__main__":

    def stest_override():
        print(global_torch_device(verbose=True))
        print(
            global_torch_device(
                override=global_torch_device(cuda_if_available=False, verbose=True),
                verbose=True,
            )
        )
        print(global_torch_device(verbose=True))
        print(global_torch_device(cuda_if_available=True))
        print(global_torch_device())
        print(
            global_torch_device(
                override=global_torch_device(cuda_if_available=True, verbose=True)
            )
        )
        print(global_torch_device())

    def a():
        print(global_torch_device())
        print(auto_select_available_cuda_device())

    stest_override()
