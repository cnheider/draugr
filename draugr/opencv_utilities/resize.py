#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any

import cv2
import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 01/12/2019
           """

__all__ = ["cv2_resize"]


def cv2_resize(x: Any, target_size: tuple, interpolation=cv2.INTER_LINEAR):
    """

:param interpolation:
:param x:
:param target_size: proper (width, height) shape, no cv craziness
:return:
"""
    if x.shape != target_size:
        x = cv2.resize(x, target_size[::-1], interpolation=interpolation)
    return x


if __name__ == "__main__":
    a = numpy.zeros((50, 50))
    a = cv2_resize(numpy.zeros((100, 100)), a.shape)
    print(a.shape)
