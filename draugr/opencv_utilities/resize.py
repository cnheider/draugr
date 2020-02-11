#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy

import cv2

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 01/12/2019
           """

__all__ = ["resize_image_cv"]


def resize_image_cv(x, target_size: tuple, interpolation=cv2.INTER_LINEAR):
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
    a = resize_image_cv(numpy.zeros((100, 100)), a.shape)
    print(a.shape)
