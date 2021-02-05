#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import numba
import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """

__all__ = ["hwc_to_chw", "chw_to_hwc", "uint_hwc_to_chw_float", "float_chw_to_hwc_uint"]


# @numba.njit()
def hwc_to_chw(inp: numpy.ndarray) -> numpy.ndarray:
    """

    :param inp:
    :type inp:
    :return:
    :rtype:"""
    return inp.transpose((2, 0, 1))


# @numba.njit()
def chw_to_hwc(inp: numpy.ndarray) -> numpy.ndarray:
    """

    :param inp:
    :type inp:
    :return:
    :rtype:"""
    return inp.transpose((1, 2, 0))


# @numba.njit()
def uint_hwc_to_chw_float(
    inp: numpy.ndarray, *, normalise: bool = True
) -> numpy.ndarray:
    """

    :param inp:
    :type inp:
    :param normalise:
    :type normalise:
    :return:
    :rtype:"""
    if normalise:
        inp /= 255.0
        inp = numpy.clip(inp, 0, 1)
    return hwc_to_chw(inp)


# @numba.njit()
def float_chw_to_hwc_uint(
    inp: numpy.ndarray, *, unnormalise: bool = True
) -> numpy.ndarray:
    """

    :param inp:
    :type inp:
    :param unnormalise:
    :type unnormalise:
    :return:
    :rtype:"""
    inp = chw_to_hwc(inp)
    if unnormalise:
        inp *= 255.0
        inp = numpy.clip(inp, 0, 255)
    return inp.astype(numpy.uint8)
