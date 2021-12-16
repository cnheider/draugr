#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17-02-2021
           """

from matplotlib import cycler, pyplot, rcParams
from matplotlib.axes import Axes

from draugr.visualisation.matplotlib_utilities.styles.cyclers import simple_hatch_cycler

__all__ = [
    "fix_edge_gridlines",
    "auto_post_print_dpi",
    "auto_post_hatch",
    "scatter_auto_mark",
]

from warg import Number, drop_unused_kws, passes_kws_to
import numpy


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


@drop_unused_kws
@passes_kws_to(pyplot.scatter)
def scatter_auto_mark(x, y, c, ax=None, m=("|", "_"), fillstyle="none", **kw):
    """
    TODO:Quick hack, can be generalised further
    :param x:
    :param y:
    :param c:
    :param ax:
    :param m:
    :param fillstyle:
    :param kw:
    :return:
    """
    import matplotlib.markers as mmarkers

    if not ax:
        ax = pyplot.gca()
    sc = ax.scatter(x, y, c=c, **kw)
    if m is not None and len(m) == len(x):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker, fillstyle=fillstyle)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    elif (
        c is not None and isinstance(c[0], (int, numpy.ndarray)) and len(c) == len(x)
    ):  # TODO: HANDLE numpy ndarray
        # better
        paths = []
        for c_ in c:
            if isinstance(c_, numpy.ndarray):
                c_ = c_.item()
            if isinstance(m[c_], mmarkers.MarkerStyle):
                marker_obj = m[c_]
            else:
                marker_obj = mmarkers.MarkerStyle(m[c_], fillstyle=fillstyle)
            paths.append(marker_obj.get_path().transformed(marker_obj.get_transform()))
        sc.set_paths(paths)
    else:
        pass
        # raise NotImplemented
    return sc
