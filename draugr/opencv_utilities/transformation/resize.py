#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from enum import Enum
from typing import Any

import cv2
import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 01/12/2019
           """

__all__ = ["cv2_resize", "InterpolationEnum"]


class InterpolationEnum(Enum):
    nearest = cv2.INTER_NEAREST  # a nearest-neighbor interpolation
    linear = cv2.INTER_LINEAR  # a bilinear interpolation (used by default)
    area = (
        cv2.INTER_AREA
    )  # resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
    cubic = cv2.INTER_CUBIC  # a bicubic interpolation over 4×4 pixel neighborhood
    lanczos4 = cv2.INTER_LANCZOS4  # a Lanczos interpolation over 8×8 pixel neighborhood


def cv2_resize(
    x: Any,
    target_size: tuple,
    interpolation: InterpolationEnum = InterpolationEnum.linear,
) -> Any:
    """

    :param interpolation:
    :param x:
    :param target_size: proper (width, height) shape, no cv craziness
    :return:"""
    interpolation = InterpolationEnum(interpolation)
    if x.shape != target_size:
        x = cv2.resize(x, target_size[::-1], interpolation=interpolation.value)
    return x


if __name__ == "__main__":
    a = numpy.zeros((50, 50))
    a = cv2_resize(numpy.zeros((100, 100)), a.shape)
    print(a.shape)
