#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from typing import Iterable, Sized

import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/10/2019
           """


def recycle(sized: Iterable):
    if not isinstance(sized, Sized):
        sized = list(sized)
    while True:
        for element in random.sample(sized, len(sized)):
            yield element


def batched_recycle(sized: Sized, n: int = 32):
    """Batches and re-cycles an array with a different permutation"""
    if isinstance(sized, numpy.ndarray):
        while True:
            yield sized[numpy.random.choice(sized.shape[0], n, replace=False)]
    else:
        if not isinstance(sized, Sized):
            sized = [*sized]
        while True:
            yield random.sample(sized, n)
