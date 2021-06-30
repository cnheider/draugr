#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17-12-2020
           """

import numpy

__all__ = ["mean_square", "root_mean_square"]


def mean_square(x: numpy.ndarray) -> numpy.ndarray:
    """Mean Square value of signal `x`.
    input
    -----
    * x: signal vector
    output
    ------
    * Mean square of `x`."""

    return numpy.mean(numpy.abs(x) ** 2)


def root_mean_square(x: numpy.ndarray) -> numpy.ndarray:
    """Root Mean Square value of signal `x`.
    input
    -----
    * x: signal vector"""

    return numpy.sqrt(mean_square(x))  # faster than **.5
