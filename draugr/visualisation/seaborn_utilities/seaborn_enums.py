#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 26-01-2021
           """

from enum import Enum
from typing import Tuple

import numpy
from matplotlib import patheffects, pyplot

__all__ = [
    "plot_median_labels",
    "show_values_on_bars",
    "VisualisationErrorStyle",
]

from draugr.visualisation.matplotlib_utilities.styles.annotation import (
    semi_opaque_round_tight_bbox,
)


class VisualisationErrorStyle(Enum):
    band = "band"
    bar = "bars"


def plot_median_labels(
    ax: pyplot.Axes,
    *,
    has_fliers: bool = False,
    # text_size: int = 10,
    # text_weight: str = "normal",
    stroke_width: int = 0,
    precision: int = 3,
    color: str = "black",
    edgecolor: str = "black",  # also the stroke color
    ha: str = "center",
    va: str = "center",  # bottom
    bbox: Tuple = semi_opaque_round_tight_bbox,
) -> None:
    """ """
    lines = ax.get_lines()
    # depending on fliers, toggle between 5 and 6 lines per box
    lines_per_box = 5 + int(has_fliers)
    # iterate directly over all median lines, with an interval of lines_per_box
    # this enables labeling of grouped data without relying on tick positions
    for median_line in lines[4 : len(lines) : lines_per_box]:
        # get center of median line
        mean_x = sum(median_line._x) / len(median_line._x)
        mean_y = sum(median_line._y) / len(median_line._y)

        text = ax.text(
            mean_x,
            mean_y,
            f"{round(mean_y, precision)}",
            ha=ha,
            va=va,
            # fontweight=text_weight,
            # size=text_size,
            color=color,
            # edgecolor=edgecolor
            bbox=bbox,
        )  # print text to center coordinates

        if stroke_width:
            # create small black border around white text
            # for better readability on multi-colored boxes
            text.set_path_effects(
                [
                    patheffects.Stroke(linewidth=stroke_width, foreground=edgecolor),
                    patheffects.Normal(),
                ]
            )


def show_values_on_bars(axs: pyplot.Axes, h_v: str = "v", space: float = 0.4) -> None:
    """ """

    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center")
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, numpy.ndarray):
        for idx, ax in numpy.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
