#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17/07/2020
           """

__all__ = ["recursive_flatten_numpy"]

from typing import Sequence

import numpy


def recursive_flatten_numpy(nested: Sequence) -> numpy.ndarray:
    return numpy.array(nested).ravel()


if __name__ == "__main__":

    print(recursive_flatten_numpy([[numpy.zeros((2, 2))], [numpy.zeros((2, 2))]]))
    print(recursive_flatten_numpy(numpy.zeros((2, 2))))
