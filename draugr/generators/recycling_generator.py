#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from typing import Iterable, Sized

import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/10/2019
           """

__all__ = ["recycle", "batched_recycle"]


def recycle(iterable: Iterable):
    if not isinstance(iterable, Sized):
        iterable = list(iterable)
    while True:
        for element in random.sample(iterable, len(iterable)):
            yield element


def batched_recycle(sized: Sized, batch_size: int = 32):
    """Batches and re-cycles an array with a different permutation"""
    if isinstance(sized, numpy.ndarray):
        while True:
            yield sized[numpy.random.choice(sized.shape[0], batch_size, replace=False)]
    else:
        if not isinstance(sized, Sized):
            sized = [*sized]
        while True:
            yield random.sample(sized, batch_size)


if __name__ == "__main__":
    a = [0, 1.0, 3.0, 2.2]
    for b in zip(range(3), batched_recycle(a, 2)):
        print(b)

    for b in zip(range(9), recycle(a)):
        print(b)
