#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17-02-2021
           """

from matplotlib import cycler, pyplot, rcParams
from matplotlib.axes import Axes

from .styles import simple_hatch_cycler

__all__ = ["fix_edge_gridlines", "auto_post_print_dpi", "auto_post_hatch"]

from warg import Number


def fix_edge_gridlines(
    ax: Axes = None,
) -> None:  # TODO: make a wrapper version of the function
    """
    Fixes gridlines when using zero margins round_number number ticks

    :param ax:
    :return:"""
    if ax is None:
        ax = pyplot.gca()
    ax.xaxis.get_gridlines()[-1].set_clip_on(False)  # last gridline
    ax.yaxis.get_gridlines()[0].set_clip_on(False)  # first gridline


def auto_post_print_dpi(scalar: Number = 5) -> None:
    """
    auto scale dpi of lines for print

    :param scalar:
    :return:"""
    dpi = rcParams["figure.dpi"]
    line_width = 1.0 / dpi * scalar
    rcParams["hatch.linewidth"] = line_width * 2
    rcParams["grid.linewidth"] = line_width


def auto_post_hatch(
    ax: Axes = None, hatch_cycler: cycler = simple_hatch_cycler
) -> None:
    """
    Auto hatches patch-types because matplotlib's prop-cycler does not iterate hatch_props

    :param ax:
    :param hatch_cycler:
    :return:"""
    if ax is None:
        ax = pyplot.gca()
    for p, d in zip(ax.patches, hatch_cycler):
        p.set_hatch(next(iter(d.values())))
