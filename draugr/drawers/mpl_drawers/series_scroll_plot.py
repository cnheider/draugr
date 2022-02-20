#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Sequence, Union

from draugr.drawers.mpl_drawers.mpldrawer import MplDrawer
from draugr.numpy_utilities import recursive_flatten_numpy

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/11/2019

           """

from matplotlib import pyplot

import numpy

__all__ = ["SeriesScrollPlot"]

from warg import passes_kws_to


class SeriesScrollPlot(MplDrawer):
    """ """

    @passes_kws_to(MplDrawer.__init__)
    def __init__(
        self,
        window_length: int = None,
        title: str = "",
        time_label: str = "Time",
        data_label: str = "Magnitude",
        reverse: bool = False,
        overwrite: bool = False,
        render: bool = True,
        **kws
    ):
        """

        :param window_length:
        :param title:
        :param time_label:
        :param data_label:
        :param reverse:
        :param overwrite:
        :param placement:
        :param render:"""

        super().__init__(render=render, **kws)
        if not render:
            return

        self.overwrite = overwrite
        self.reverse = reverse
        self.window_length = window_length

        if window_length:
            assert window_length > 3

        self.fig = pyplot.figure(figsize=(4, 4))
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

    def _draw(self, data: Union[Sequence, int, float, complex]):
        """
        SHOULD NOT BE CALLED DIRECTLY!

        :param data:
        :return:"""
        if not isinstance(data, Sequence):
            data = [data]

        data = recursive_flatten_numpy(data)

        num_images = len(self.im)
        num_series = len(data)

        min_min, max_max = None, None

        if num_images != num_series:
            # print('Reinstantiating figures')
            self.im = pyplot.plot(*[[] for _ in range(num_series)] * 2)
            if self.window_length:
                for i in range(len(self.im)):
                    self.im[i].set_xdata(range(self.window_length))
                    self.im[i].set_ydata([numpy.nan] * self.window_length)

        for i in range(num_images):
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


if __name__ == "__main__":

    def asidjas() -> None:
        """
        :rtype: None
        """
        from math import cos, sin

        def multi_series():
            """ """
            s = SeriesScrollPlot(
                window_length=200, reverse=False, overwrite=False, default_delta=None
            )
            for i in range(1000):
                s.draw([sin(i / 100) * 2, cos(i / 10)])

        def single_series():
            """ """
            s = SeriesScrollPlot(window_length=200, reverse=False, overwrite=False)
            for i in range(1000):
                s.draw([sin(i / 20)], 1.0 / 60.0)

        def single_series_no_wrap():
            """ """
            s = SeriesScrollPlot(window_length=200, reverse=True, overwrite=False)
            for i in range(1000):
                s.draw(sin(i / 20), 1.0 / 60.0)

        def single_series_no_wrap_rescale():
            """ """
            s = SeriesScrollPlot(window_length=100, reverse=True, overwrite=False)
            for i in range(1000):
                s.draw(sin(i / 100), 1.0 / 60.0)

        multi_series()

    asidjas()
