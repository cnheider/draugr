#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 02-02-2021
           """

import numpy
import torch

__all__ = [
    "numpy_to_torch_dtype_dict",
    "torch_to_numpy_dtype_dict",
    "numpy_to_torch_dtype",
    "torch_to_numpy_dtype",
]

numpy_to_torch_dtype_dict = (
    {  # Dict of NumPy dtype -> torch dtype (when the correspondence exists)
        bool: torch.bool,
        numpy.uint8: torch.uint8,
        numpy.int8: torch.int8,
        numpy.int16: torch.int16,
        numpy.int32: torch.int32,
        numpy.int64: torch.int64,
        numpy.float16: torch.float16,
        numpy.float32: torch.float32,
        numpy.float64: torch.float64,
        numpy.complex64: torch.complex64,
        numpy.complex128: torch.complex128,
    }
)

torch_to_numpy_dtype_dict = {
    value: key for (key, value) in numpy_to_torch_dtype_dict.items()
}  # Dict of torch dtype -> NumPy dtype


def numpy_to_torch_dtype(numpy_dtype: numpy.dtype) -> torch.dtype:
    """ """
    return numpy_to_torch_dtype_dict[numpy_dtype.type]


def torch_to_numpy_dtype(torch_dtype: torch.dtype) -> numpy.dtype:
    """ """
    return torch_to_numpy_dtype_dict[torch_dtype]


if __name__ == "__main__":

    def iusahdu() -> None:
        """
        :rtype: None
        """
        a = numpy.zeros((1, 1))
        print(a.dtype)
        b = numpy_to_torch_dtype(a.dtype)
        print(b)
        print(type(b))
        c = torch_to_numpy_dtype(b)
        print(c)
        print(type(c))

    iusahdu()
