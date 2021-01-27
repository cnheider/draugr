#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 26-01-2021
           """

import numpy
from matplotlib import patheffects, pyplot

__all__ = ["plot_median_labels", "show_values_on_bars"]


def plot_median_labels(
    ax: pyplot.Axes,
    has_fliers: bool = False,
    text_size: int = 10,
    text_weight: str = "normal",
    stroke_width: int = 2,
    precision: int = 3,
) -> None:
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
            ha="center",
            va="center",
            fontweight=text_weight,
            size=text_size,
            color="white",
        )  # print text to center coordinates

        # create small black border around white text
        # for better readability on multi-colored boxes
        text.set_path_effects(
            [
                patheffects.Stroke(linewidth=stroke_width, foreground="black"),
                patheffects.Normal(),
            ]
        )


def show_values_on_bars(axs: pyplot.Axes, h_v="v", space=0.4) -> None:
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
