#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/03/2020
           """

from warg import QuadNumber

__all__ = ["xywh_to_minmax", "minmax_to_xywh"]


def xywh_to_minmax(box: QuadNumber) -> QuadNumber:
    """

    :param box:
    :type box:
    :return:
    :rtype:"""
    x1, y1, w, h = box
    return x1, y1, x1 + w, y1 + h


def minmax_to_xywh(boxes: QuadNumber) -> QuadNumber:
    """

    :param boxes:
    :type boxes:
    :return:
    :rtype:"""
    x_min, y_min, x_max, y_max = boxes
    return x_min, y_min, x_max - x_min, y_max - y_min


if __name__ == "__main__":
    quad = (2, 2, 3, 4)
    mm = xywh_to_minmax(quad)
    xywh = minmax_to_xywh(mm)
    print(quad, mm, xywh)
