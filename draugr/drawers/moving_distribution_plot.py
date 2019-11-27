#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sequence, Tuple, Sized

import matplotlib

from draugr.drawers.drawer import Drawer
from warg import passes_kws_to

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/11/2019

           """

from matplotlib import pyplot

import numpy

__all__ = ["MovingDistributionPlot"]


class MovingDistributionPlot(Drawer):
    @passes_kws_to(pyplot.hist)
    def __init__(
        self,
        window_length: int = None,
        title: str = "",
        data_label: str = "Magnitude",
        reverse: bool = False,
        overwrite: bool = False,
        placement: Tuple = (0, 0),
        render: bool = True,
        **kwargs
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

        self.array = []
        self.hist_kws = kwargs
        pyplot.hist(self.array, **self.hist_kws)

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

        self.array.extend(data)
        pyplot.cla()
        pyplot.hist(self.array, **self.hist_kws)

        pyplot.draw()

        if self.n <= 1:
            self.move_figure(self.fig, *self.placement)
        self.n += 1

        if delta:
            pyplot.pause(delta)


if __name__ == "__main__":
    delta = 1.0 / 60.0

    s = MovingDistributionPlot()
    for LATEST_GPU_STATS in range(100):
        s.draw(numpy.random.sample(), delta)
