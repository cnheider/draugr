#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17/07/2020
           """

from functools import reduce

import numpy
from matplotlib import patches, pyplot


__all__ = [
    "denormalise_minusoneone",
    "matplotlib_bounding_box",
    "plot_img_array",
    "plot_side_by_side",
]


def denormalise_minusoneone(T, coords):
    return 0.5 * ((coords + 1.0) * T)


def matplotlib_bounding_box(x, y, size, color="w"):
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False
    )
    return rect


def plot_img_array(img_array: numpy.ndarray, n_col: int = 3) -> None:
    """

:param img_array:
:type img_array:
:param n_col:
:type n_col:
:return:
:rtype:
"""
    n_row = len(img_array) // n_col

    f, plots = pyplot.subplots(
        n_row, n_col, sharex="all", sharey="all", figsize=(n_col * 4, n_row * 4)
    )

    for i in range(len(img_array)):
        plots[i // n_col, i % n_col].imshow(img_array[i])


def plot_side_by_side(img_arrays) -> None:
    flatten_list = reduce(lambda x, y: x + y, zip(*img_arrays))

    plot_img_array(numpy.array(flatten_list), n_col=len(img_arrays))
