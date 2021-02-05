#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 13-01-2021
           """

from math import ceil, log, log10, log2, floor

__all__ = [
    "next_pow",
    "next_power",
    "next_pow_2",
    "next_pow_10",
    "next_power_2",
    "next_power_10",
    "prev_pow",
    "prev_pow_10",
    "prev_pow_2",
]

from warg import Number


def next_pow(x: Number, n: int = None) -> int:
    """
    If the base(n) not specified, returns the natural logarithm (base e) of x.

    @param x:
    @param n:
    @return:
    """
    return int(pow(n, ceil(log(x, n))))


def prev_pow(x: Number, n: int = None) -> int:
    """
    If the base(n) not specified, returns the natural logarithm (base e) of x.

    @param x:
    @param n:
    @return:
    """
    return int(pow(n, floor(log(x, n))))


def next_pow_2(x: Number) -> int:
    return int(pow(2, ceil(log2(x))))


def prev_pow_2(x: Number) -> int:
    return int(pow(2, floor(log2(x))))


def next_pow_10(x: Number) -> int:
    return int(pow(10, ceil(log10(x))))


def prev_pow_10(x: Number) -> int:
    return int(pow(10, floor(log10(x))))


next_power = next_pow
next_power_2 = next_pow_2
next_power_10 = next_pow_10

if __name__ == "__main__":
    print(next_pow(17, 5))
    print(prev_pow(17, 5))
