#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04-03-2021
           """

from typing import Tuple

from draugr.scipy_utilities import mag_decimation_subsample

__all__ = [
    "annotate_lines",
    "default_index_decimator",
]

from warg import GDKC, passes_kws_to

from matplotlib.pyplot import Axes

default_index_decimator = GDKC(
    mag_decimation_subsample, decimation_factor=5, return_indices=True
)  # finds interesting features?


@passes_kws_to(Axes.annotate)
def annotate_lines(
    ax_: Axes,
    num_lines: int = 1,  # None for all
    index_decimator: callable = default_index_decimator,
    color: str = "k",  # None for auto color
    xycoords: Tuple[str, str] = (
        "data",
        # 'axes fraction',
        "data",
    ),  # TODO: NOT DONE! Where to place annotation, use 'axes fraction' for along axes'
    ha: str = "left",
    va: str = "center",
    **kwargs,
) -> None:
    """

    :param ax_:
    :param num_lines:
    :param index_decimator:
    :param color:
    :param xycoords:
    :param ha:
    :param va:
    :param kwargs:
    """
    lines = ax_.lines
    if not num_lines:
        num_lines = len(lines)
    for l, _ in zip(lines, range(num_lines)):
        y = l.get_ydata()
        x = l.get_xdata()

        if not color:
            color = l.get_color()

        if index_decimator:
            mag_y = index_decimator(y)
        else:
            mag_y = list(range(len(y)))

        for x_, y_ in zip(x[mag_y], y[mag_y]):
            ax_.annotate(
                f"{y_:.2f}",
                xy=(x_, y_),  # ( 1, y_) axes fraction'
                xycoords=xycoords,
                ha=ha,
                va=va,
                color=color,
                **kwargs,
            )


if __name__ == "__main__":

    def hsdh() -> None:
        """
        :rtype: None
        """
        from matplotlib import pyplot

        a = [*range(0, 10), *range(10, -10, -1), *range(-10, 0)]
        ax_ = pyplot.plot(a)
        annotate_lines(pyplot.gca(), index_decimator=default_index_decimator)
        pyplot.show()

    hsdh()
