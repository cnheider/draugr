#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import cos, sin
from typing import Sequence, Sized, Tuple, Union

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
    """

    """

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

        self.im = pyplot.plot([], [])

        if window_length:
            for i in range(len(self.im)):
                self.im[i].set_xdata(range(window_length))
                self.im[i].set_ydata([numpy.nan] * window_length)
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
                window.wm_geometry(f"+{x:d}+{y:d}")
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

    def draw(self, data: Union[Sized, int, float, complex], delta: float = 1.0 / 120.0):
        """

:param data:
:param delta: 1 / 60 for 60fps
:return:
"""
        if not isinstance(data, Sequence):
            data = [data]

        num_figures = len(self.im)
        num_series = len(data)

        min_min, max_max = None, None

        if num_figures != num_series:
            # print('Reinstantiating figures')
            self.im = pyplot.plot(*[[] for _ in range(num_series)] * 2)
            if self.window_length:
                for i in range(len(self.im)):
                    self.im[i].set_xdata(range(self.window_length))
                    self.im[i].set_ydata([numpy.nan] * self.window_length)

        for i in range(num_figures):
            time_points = self.im[i].get_xdata()
            data_points = self.im[i].get_ydata()

            if self.window_length:
                if not self.overwrite:
                    if not self.reverse:
                        data_points = numpy.concatenate(
                            (data_points[1:], [data[i]]), axis=0
                        )
                    else:
                        data_points = numpy.concatenate(
                            ([data[i]], data_points[:-1]), axis=0
                        )
                else:
                    if not self.reverse:
                        data_points[self.n % self.window_length] = data[i]
                    else:
                        data_points[
                            self.window_length - 1 - self.n % self.window_length
                        ] = data[i]
            else:
                if not self.reverse:
                    time_points = numpy.concatenate((time_points, [self.n]), axis=0)
                    data_points = numpy.concatenate((data_points, [data[i]]), axis=0)
                else:
                    time_points = numpy.concatenate(([self.n], time_points), axis=0)
                    data_points = numpy.concatenate(([data[i]], data_points), axis=0)
                pyplot.xlim(0, self.n + 1)

            self.im[i].set_xdata(time_points)
            self.im[i].set_ydata(data_points)
            min_, max_ = numpy.nanmin(data_points), numpy.nanmax(data_points)
            if min_min is None or min_ < min_min:
                min_min = min_
            if max_max is None or max_ > max_max:
                max_max = max_

        if min_min == max_max:
            max_max += 1

        pyplot.ylim(min_min, max_max)

        pyplot.draw()

        if self.n <= 1:
            self.move_figure(self.fig, *self.placement)
        self.n += 1

        if delta:
            pyplot.pause(delta)


if __name__ == "__main__":

    def multi_series():
        """

        """
        s = SeriesScrollPlot(200, reverse=False, overwrite=False)
        for i in range(1000):
            s.draw([sin(i / 100) * 2, cos(i / 10)], 1.0 / 60.0)

    def single_series():
        """

        """
        s = SeriesScrollPlot(200, reverse=False, overwrite=False)
        for i in range(1000):
            s.draw([sin(i / 20)], 1.0 / 60.0)

    def single_series_no_wrap():
        """

        """
        s = SeriesScrollPlot(200, reverse=True, overwrite=False)
        for i in range(1000):
            s.draw(sin(i / 20), 1.0 / 60.0)

    def single_series_no_wrap_rescale():
        """

        """
        s = SeriesScrollPlot(100, reverse=True, overwrite=False)
        for i in range(1000):
            s.draw(sin(i / 100), 1.0 / 60.0)

    multi_series()
