#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 14-03-2021
           """

from typing import Tuple

import numpy

from warg import Number

__all__ = ["circular_mask"]


def circular_mask(
    h: Number, w: Number, center: Tuple[Number, Number] = None, radius: Number = None
) -> numpy.ndarray:
    """

    :param h:
    :param w:
    :param center:
    :param radius:
    :return:"""
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    y, x = numpy.ogrid[:h, :w]
    dist_from_center = numpy.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    return dist_from_center <= radius


if __name__ == "__main__":
    print(circular_mask(9, 9))
