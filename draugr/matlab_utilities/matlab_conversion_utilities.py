#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 14-02-2021
           """

from typing import Sequence

import numpy

__all__ = ["get_strides_f", "get_strides_c"]


def get_strides_f(shape: Sequence) -> object:
    """Get strides of a F like array, for numpy array need to multiply by itemsize

    Parameters
    ----------
    shape : tuple of int or iterable
      shape of the array
    Returns
    -------
    s :  tuple of int or iterable
      strides of the array

    Examples
    --------
    >>> get_strides_f((2, 3, 3))
    [1, 2, 6]

    References
    ----------
    https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html"""

    # $s_k = \Prod_{j=0}^{k-1} d_j$
    s = numpy.cumprod((1,) + shape[0:-1]).tolist()

    return s


def get_strides_c(shape):
    """Get strides of a C like array. For numpy array need to be multiply by itemsize

    Parameters
    ----------
    shape : tuple of int or iterable
      shape of the array
    Returns
    -------
    s :  tuple of int or iterable
      strides of the array

    Examples
    --------
    >>> get_strides_c((2, 3, 3))
    [9, 3, 1]
    References
    ----------
    https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html"""

    n = len(shape)

    # $s_k = \Prod_{j=k+1}^{N-1} d_j$ with shape[j] <=> $d_j$
    s = [0] * n
    for k in reversed(range(0, n)):
        if k == n - 1:
            s[k] = 1
        else:
            s[k] = shape[k + 1] * s[k + 1]

    return s
