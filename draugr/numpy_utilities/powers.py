#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 07-01-2021
           """

import numpy

from warg import Number

__all__ = [
    "next_pow_numpy",
    "next_pow_2_numpy",
    "next_pow_10_numpy",
    "next_power_2_numpy",
    "next_power_10_numpy",
]


def next_pow_numpy(x: Number, n: int) -> int:
    """Calculates the next power of n of a number."""

    return int(pow(n, numpy.ceil(numpy.log(x) / numpy.log(n))))


def next_pow_2_numpy(x: Number) -> int:
    """Calculates the next power of 2 of a number."""

    return int(pow(2, numpy.ceil(numpy.log2(x))))


next_power_2_numpy = next_pow_2_numpy


def next_pow_10_numpy(x: Number) -> int:
    """Calculates the next power of 10 of a number."""

    return int(pow(10, numpy.ceil(numpy.log10(x))))


next_power_10_numpy = next_pow_10_numpy

if __name__ == "__main__":
    for i in range(1, 11 + 1):
        print(next_pow_2_numpy(i))
        print(next_pow_numpy(i, 2))
        # print(next_pow(i, 3))
        # print(next_pow_10(i))

    print(next_pow_2_numpy(next_pow_2_numpy(next_pow_2_numpy(2))))
