#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sequence, Tuple, Sized

import matplotlib

from draugr.drawers.drawer import Drawer

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/11/2019

           """

from matplotlib import pyplot

import numpy

__all__ = ["SeriesScrollPlot"]


class SeriesScrollPlot(Drawer):
    def __init__(
        self,
        window_length: int = None,
        title: str = "",
        time_label: str = "Time",
        data_label: str = "Magnitude",
        reverse: bool = False,
        overwrite: bool = False,
        placement: Tuple = (0, 0),
        render: bool = True,
    ):
        self.fig = None

        if not render:
            return

        self.overwrite = overwrite
        self.reverse = reverse
        self.window_length = window_length
        self.n = 0

        if window_length:
            assert window_length > 3
            self.fig = pyplot.figure(figsize=(4, 4))
        else:
            self.fig = pyplot.figure(figsize=(4, 4))

        self.placement = placement

        self.im, = pyplot.plot([], [])

        if window_length:
            self.im.set_xdata(range(window_length))
            self.im.set_ydata([numpy.nan] * window_length)
            pyplot.xlim(0, window_length)

        pyplot.xlabel(time_label)
        pyplot.ylabel(data_label)

        pyplot.title(title)
        pyplot.tight_layout()

    @staticmethod
    def move_figure(figure: pyplot.Figure, x: int = 0, y: int = 0):
        """Move figure's upper left corner to pixel (x, y)"""
        backend = matplotlib.get_backend()
        if hasattr(figure.canvas.manager, "window"):
            window = figure.canvas.manager.window
            if backend == "TkAgg":
                window.wm_geometry("+%d+%d" % (x, y))
            elif backend == "WXAgg":
                window.SetPosition((x, y))
            else:
                # This works for QT and GTK
                # You can also use window.setGeometry
                window.move(x, y)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fig:
            pyplot.close(self.fig)

    def draw(self, data: Sized, delta: float = 1.0 / 120.0):
        """

:param data:
:param delta: 1 / 60 for 60fps
:return:
"""
        if not isinstance(data, Sized):
            data = [data]

        ts = self.im.get_xdata()
        array = self.im.get_ydata()

        if self.window_length:
            if not self.overwrite:
                if not self.reverse:
                    array = numpy.concatenate((array[1:], data))
                else:
                    array = numpy.concatenate((data, array[:-1]))
            else:
                if not self.reverse:
                    array[self.n % self.window_length] = data[0]
                else:
                    array[self.window_length - 1 - self.n % self.window_length] = data[
                        0
                    ]
        else:
            if not self.reverse:
                ts = numpy.concatenate((ts, [self.n]))
                array = numpy.concatenate((array, data))
            else:
                ts = numpy.concatenate(([self.n], ts))
                array = numpy.concatenate((data, array))
            pyplot.xlim(0, self.n + 1)

        self.im.set_xdata(ts)
        self.im.set_ydata(array)
        min_, max_ = numpy.nanmin(array), numpy.nanmax(array)

        if min_ == max_:
            max_ += 1

        pyplot.ylim(min_, max_)

        pyplot.draw()

        if self.n <= 1:
            self.move_figure(self.fig, *self.placement)
        self.n += 1

        if delta:
            pyplot.pause(delta)


if __name__ == "__main__":
    delta = 1.0 / 60.0

    s = SeriesScrollPlot(100, reverse=False, overwrite=False)
    for LATEST_GPU_STATS in range(1000):
        s.draw(LATEST_GPU_STATS % 10, delta)
