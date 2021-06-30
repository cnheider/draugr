#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17-02-2021
           """

from functools import reduce

import numpy
from matplotlib import pyplot

__all__ = [
    "plot_img_array",
    "plot_side_by_side",
]


def plot_img_array(img_array: numpy.ndarray, n_col: int = 3) -> None:
    """

    :param img_array:
    :type img_array:
    :param n_col:
    :type n_col:
    :return:
    :rtype:"""
    n_row = len(img_array) // n_col

    f, plots = pyplot.subplots(
        n_row, n_col, sharex="all", sharey="all", figsize=(n_col * 4, n_row * 4)
    )

    for i in range(len(img_array)):
        plots[i // n_col, i % n_col].imshow(img_array[i])


def plot_side_by_side(img_arrays: numpy.ndarray) -> None:
    """ """
    flatten_list = reduce(lambda x, y: x + y, zip(*img_arrays))

    plot_img_array(numpy.array(flatten_list), n_col=len(img_arrays))
