#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25/03/2020
           """

import numpy

__all__ = [
    "gamma_correct_byte",
    "gamma_correct_fast_to_byte",
    "gamma_correct_float_to_byte",
    "linear_correct_byte",
    "linear_correct_float_to_byte",
]


def gamma_correct_fast_to_byte(image: numpy.ndarray) -> numpy.ndarray:
    """

:param image:
:type image:
:return:
:rtype:
"""
    return ((image ** 0.454545) * 255).astype(numpy.uint8)


def gamma_correct_float_to_byte(
    image: numpy.ndarray, gamma: float = 2.2
) -> numpy.ndarray:
    """

:param image:
:type image:
:param gamma:
:type gamma:
:return:
:rtype:
"""
    return ((image ** (1.0 / gamma)) * 255).astype(numpy.uint8)


def linear_correct_float_to_byte(
    image: numpy.ndarray, gamma: float = 2.2
) -> numpy.ndarray:
    """

:param image:
:type image:
:param gamma:
:type gamma:
:return:
:rtype:
"""
    return ((image ** gamma) * 255).astype(numpy.uint8)


def linear_correct_byte(image: numpy.ndarray, gamma: float = 2.2) -> numpy.ndarray:
    """

:param image:
:type image:
:param gamma:
:type gamma:
:return:
:rtype:
"""
    return gamma_correct_float_to_byte(image / 255, gamma)


def gamma_correct_byte(image: numpy.ndarray, gamma: float = 2.2) -> numpy.ndarray:
    """

:param image:
:type image:
:param gamma:
:type gamma:
:return:
:rtype:
"""
    return gamma_correct_float_to_byte(image / 255, gamma)
