#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/03/2020
           """

from typing import Tuple

__all__ = ["xywh_to_minmax", "minmax_to_xywh"]

Quadruple = Tuple[float, float, float, float]


def xywh_to_minmax(box: Quadruple) -> Quadruple:
    """

    :param box:
    :type box:
    :return:
    :rtype:
    """
    x1, y1, w, h = box
    return x1, y1, x1 + w, y1 + h


def minmax_to_xywh(boxes: Quadruple) -> Quadruple:
    """

    :param boxes:
    :type boxes:
    :return:
    :rtype:
    """
    xmin, ymin, xmax, ymax = boxes
    return xmin, ymin, xmax - xmin, ymax - ymin


if __name__ == "__main__":
    quad = (2, 2, 3, 4)
    mm = xywh_to_minmax(quad)
    xywh = minmax_to_xywh(mm)
    print(quad, mm, xywh)
