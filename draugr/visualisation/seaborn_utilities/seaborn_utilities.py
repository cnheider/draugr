#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 18-02-2021
           """

from typing import Iterable, List

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from matplotlib.axes import Axes

from warg import Number

__all__ = [
    "despine_all",
    "set_y_log_scale",
    "exponential_moving_average",
]


def despine_all(ax: Axes = None) -> None:
    """

    :param ax:
    """
    if ax is None:
        ax = pyplot.gca()

    seaborn.despine(
        ax=ax, top=True, right=True, left=True, bottom=True, offset=None, trim=False
    )


def set_y_log_scale(ax: Axes = None) -> None:
    """

    :param ax:
    """
    if ax is None:
        ax = pyplot.gca()

    ax.set(yscale="log")


def exponential_moving_average(
    scalars: Iterable[Number], decay: float = 0.4
) -> List[Number]:
    """
    Like is usual in tensorboard visual rep just weight is inverse

    :param decay:
    :param scalars:
    :return:
    """
    if isinstance(scalars, numpy.ndarray):
        assert len(scalars.shape) <= 1, "only support one dimensional series"

    last = next(iter(scalars))
    smoothed = list()
    for new in scalars:
        # 1st-order IIR low-pass filter to attenuate the higher-
        # frequency components of the time-series.
        smoothed_point = last * decay + new * (1.0 - decay)
        smoothed.append(smoothed_point)
        last = smoothed_point

    return smoothed


if __name__ == "__main__":

    def stest_box_plot_props() -> None:
        """
        :rtype: None
        """
        props = {
            "boxprops": {"facecolor": "none", "edgecolor": "red"},
            "medianprops": {"color": "green"},
            "whiskerprops": {"color": "blue"},
            "capprops": {"color": "yellow"},
        }

        seaborn.boxplot(
            x="variable",
            y="value",
            data=pandas.DataFrame(
                [range(3), range(3)], columns=("variable", "value", "value2")
            ),
            showfliers=False,
            linewidth=0.75,
            **props
        )
        pyplot.show()

    stest_box_plot_props()
