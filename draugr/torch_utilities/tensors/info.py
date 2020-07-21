#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/06/2020
           """

import itertools
import multiprocessing
import platform
import sys
import typing

import torch

__all__ = ["size_of_tensor", "system_info", "cuda_info"]


def size_of_tensor(obj) -> int:
    r"""**Get size in bytes of Tensor, torch.nn.Module or standard object.**

  Specific routines are defined for torch.tensor objects and torch.nn.Module
  objects. They will calculate how much memory in bytes those object consume.

  If another object is passed, `sys.getsizeof` will be called on it.

  This function works similarly to C++'s sizeof operator.


  Parameters
  ----------
  obj
          Object whose size will be measured.

  Returns
  -------
  int
          Size in bytes of the object

  """
    if torch.is_tensor(obj):
        return obj.element_size() * obj.numel()

    elif isinstance(obj, torch.nn.Module):
        return sum(
            size_of_tensor(tensor)
            for tensor in itertools.chain(obj.buffers(), obj.parameters())
        )
    else:
        return sys.getsizeof(obj)


def system_info() -> str:
    """

  :return:
  :rtype:
  """
    return "\n".join(
        [
            f"Python version: {platform.python_version()}",
            f"Python implementation: {platform.python_implementation()}",
            f"Python compiler: {platform.python_compiler()}",
            f"PyTorch version: {torch.__version__}",
            f"System: {platform.system() or 'Unable to determine'}",
            f"System version: {platform.release() or 'Unable to determine'}",
            f"Processor: {platform.processor() or 'Unable to determine'}",
            f"Number of CPUs: {multiprocessing.cpu_count()}",
        ]
    )


def cuda_info() -> str:
    """

  :return:
  :rtype:
  """

    def _cuda_devices_formatting(
        info_function: typing.Callable,
        formatting_function: typing.Callable = None,
        mapping_function: typing.Callable = None,
    ):
        def _setup_default(function):
            return (lambda arg: arg) if function is None else function

        formatting_function = _setup_default(formatting_function)
        mapping_function = _setup_default(mapping_function)

        return " | ".join(
            mapping_function(
                [
                    formatting_function(info_function(i))
                    for i in range(torch.cuda.device_count())
                ]
            )
        )

    def _device_properties(attribute):
        return _cuda_devices_formatting(
            lambda i: getattr(torch.cuda.get_device_properties(i), attribute),
            mapping_function=lambda in_bytes: map(str, in_bytes),
        )

    cuda_cap = _cuda_devices_formatting(
        torch.cuda.get_device_capability,
        formatting_function=lambda capabilities: ".".join(map(str, capabilities)),
    )
    return "\n".join(
        [
            f"Available CUDA devices count: {torch.cuda.device_count()}",
            f"CUDA devices names: {_cuda_devices_formatting(torch.cuda.get_device_name)}",
            f"Major.Minor CUDA capabilities of devices: {cuda_cap}",
            f"Device total memory (bytes): {_device_properties('total_memory')}",
            f"Device multiprocessor count: {_device_properties('multi_processor_count')}",
        ]
    )


if __name__ == "__main__":

    module = torch.nn.Linear(20, 20)
    bias = 20 * 4  # in bytes
    weights = 20 * 20 * 4  # in bytes
    assert size_of_tensor(module) == bias + weights

    print(system_info())
    print(cuda_info())
