#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 18-01-2021
           """

import numpy

from warg import next_pow_2

__all__ = ["zero_pad_to_power_2", "zero_pad_to"]


def zero_pad_to(signal: numpy.ndarray, length: int) -> numpy.ndarray:
    """ """
    return numpy.pad(
        signal, (0, length - len(signal)), "constant", constant_values=(0, 0)
    )


def zero_pad_to_power_2(signal: numpy.ndarray) -> numpy.ndarray:
    """ """
    return zero_pad_to(signal, next_pow_2(len(signal)))


if __name__ == "__main__":
    aasd = numpy.arange(8 + 1)
    print(aasd, aasd.shape)
    padded = zero_pad_to_power_2(aasd)
    print(padded, padded.shape)
