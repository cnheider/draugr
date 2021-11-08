#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/06/2020
           """

import itertools
import sys

import torch

__all__ = ["size_of_tensor"]


def size_of_tensor(obj: torch.Tensor) -> int:
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
      Size in bytes of the object"""
    if torch.is_tensor(obj):
        return obj.element_size() * obj.numel()

    elif isinstance(obj, torch.nn.Module):
        return sum(
            size_of_tensor(tensor)
            for tensor in itertools.chain(obj.buffers(), obj.parameters())
        )
    else:
        return sys.getsizeof(obj)


if __name__ == "__main__":
    module = torch.nn.Linear(20, 20)
    bias = 20 * 4  # in bytes
    weights = 20 * 20 * 4  # in bytes
    assert size_of_tensor(module) == bias + weights
