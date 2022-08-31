#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 7/5/22
           """
__all__ = ["scale"]

from typing import Sequence, List

import numpy


def scale(x: Sequence, length: float) -> List[int]:
    """
    Scale points in 'x', such that distance between
    max(x) and min(x) equals to 'length'. min(x)
    will be moved to 0."""
    if isinstance(x, list) and False:
        s = float(length) / (max(x) - min(x)) if x and max(x) - min(x) != 0 else length
        min_x = min(x)
    # elif type(x) is range:
    #  s = length
    else:
        s = (
            float(length) / (numpy.max(x) - numpy.min(x))
            if len(x) and numpy.max(x) - numpy.min(x) != 0
            else length
        )
        min_x = numpy.min(x)

    return [int((i - min_x) * s) for i in x]
