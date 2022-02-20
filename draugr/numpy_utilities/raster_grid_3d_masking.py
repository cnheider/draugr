#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 14-03-2021
           """

from typing import Tuple

import numpy

from warg import Number

__all__ = ["sphere_mask"]


def sphere_mask(
    h: Number,
    w: Number,
    d: Number,
    center: Tuple[Number, Number, Number] = None,
    radius: Number = None,
) -> numpy.ndarray:
    """

    :param d:
    :param h:
    :param w:
    :param center:
    :param radius:
    :return:"""
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2), int(d / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(
            center[0], center[1], center[2], w - center[0], h - center[1], d - center[2]
        )

    y, x, z = numpy.ogrid[:h, :w, :d]
    dist_from_center = numpy.sqrt(
        (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    )

    return dist_from_center <= radius


if __name__ == "__main__":
    print(sphere_mask(9, 9, 9))
