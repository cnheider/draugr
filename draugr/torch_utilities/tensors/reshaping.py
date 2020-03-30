#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/02/2020
           """
__all__ = ["flatten_tn_dim"]

import torch

from draugr.python_utilities import prod


def flatten_tn_dim(_tensor: torch.tensor) -> torch.tensor:
    """

:param _tensor:
:return:
"""
    T, N, *r = _tensor.size()
    return _tensor.view(T * N, *r)


if __name__ == "__main__":
    shape = (2, 3, 4, 5)
    t = torch.reshape(torch.arange(0, prod(shape)), shape)

    f = flatten_tn_dim(t)
    tf = t.flatten(0, 1)
    print(t, f, tf)
    print(t.shape, f.shape, tf.shape)
