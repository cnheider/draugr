#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17-09-2020
           """

from typing import Any

import matlab
import numpy

__all__ = ["ndarray_to_matlab", "matlab_to_ndarray"]


def ndarray_to_matlab(x: numpy.ndarray) -> Any:
    """

    :param x:
    :return:
    """
    return matlab.double(x.tolist())


def matlab_to_ndarray(x: Any) -> numpy.ndarray:
    """

    :param x:
    :return:
    """
    return numpy.asarray(x)
