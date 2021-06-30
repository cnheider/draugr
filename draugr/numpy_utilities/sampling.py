#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 02-01-2021
           """

from typing import Union

import numpy
from numpy.random.mtrand import RandomState

__all__ = ["get_sampler"]


def get_sampler(seed: Union[RandomState, int] = None) -> RandomState:
    """ """
    if isinstance(seed, RandomState):
        return seed
    return numpy.random.RandomState(seed)
