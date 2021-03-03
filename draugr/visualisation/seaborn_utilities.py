#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 18-02-2021
           """

import seaborn

__all__ = ["despine_all", "set_y_log_scale"]

from matplotlib import pyplot
from matplotlib.axes import Axes


def despine_all(ax: Axes = None) -> None:
    if ax is None:
        ax = pyplot.gca()

    seaborn.despine(
        ax=ax, top=True, right=True, left=True, bottom=True, offset=None, trim=False
    )


def set_y_log_scale(ax: Axes = None) -> None:
    if ax is None:
        ax = pyplot.gca()

    ax.set(yscale="log")


if __name__ == "__main__":

    def stest_box_plot_props():
        PROPS = {
            "boxprops": {"facecolor": "none", "edgecolor": "red"},
            "medianprops": {"color": "green"},
            "whiskerprops": {"color": "blue"},
            "capprops": {"color": "yellow"},
        }

        seaborn.boxplot(
            x="variable", y="value", data=[], showfliers=False, linewidth=0.75, **PROPS
        )
