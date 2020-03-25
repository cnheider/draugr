#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 15/11/2019
           """

device = None

__all__ = [
    "global_torch_device",
    "select_cuda_device",
    "get_gpu_usage_mb",
    "auto_select_available_cuda_device",
]


def global_torch_device(
    prefer_cuda: bool = True, override: torch.device = None
) -> torch.device:
    """

  :param prefer_cuda:
  :type prefer_cuda:
  :param override:
  :type override:
  :return:
  :rtype:
  """
    global device
    if override is not None:
        device = override
    elif not device:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and prefer_cuda else "cpu"
        )
    return device


def select_cuda_device(gpuidx: int) -> torch.device:
    """

  :param gpuidx:
  :type gpuidx:
  :return:
  :rtype:
  """
    num_cuda_device = torch.cuda.device_count()
    assert num_cuda_device > 0
    assert gpuidx < num_cuda_device
    if 0 <= gpuidx < num_cuda_device:
        return torch.device(f"cuda:{gpuidx}")


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


def auto_select_available_cuda_device(
    expected_memory_usage_mb: int = 1024
) -> torch.device:
    r"""

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
    print(global_torch_device())
    print(auto_select_available_cuda_device())
