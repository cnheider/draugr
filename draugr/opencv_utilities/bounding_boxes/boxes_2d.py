#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/03/2020
           """

from typing import Tuple

__all__ = ["xywh_to_minmax", "minmax_to_xywh"]


def xywh_to_minmax(
    box: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    x1, y1, w, h = box
    return x1, y1, x1 + w, y1 + h


def minmax_to_xywh(
    boxes: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = boxes
    return xmin, ymin, xmax - xmin, ymax - ymin
