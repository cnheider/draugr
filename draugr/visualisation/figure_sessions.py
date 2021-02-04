#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 15/06/2020
           """

from typing import Iterable, Sequence, Tuple

from matplotlib import pyplot
from warg import AlsoDecorator, passes_kws_to

__all__ = ["FigureSession", "SubplotSession"]


class FigureSession(AlsoDecorator):
    @passes_kws_to(pyplot.figure)
    def __init__(self, **kws):
        self.fig = pyplot.figure(**kws)

    def __enter__(self) -> pyplot.Figure:
        return self.fig

    def __exit__(self, exc_type, exc_val, exc_tb):
        pyplot.cla()
        pyplot.close(self.fig)
        pyplot.clf()


class SubplotSession(AlsoDecorator):
    @passes_kws_to(pyplot.subplots)
    def __init__(self, **kws):
        self.fig, axs = pyplot.subplots(**kws)
        if not isinstance(axs, Iterable):
            axs = (axs,)
        self.axs = axs

    def __enter__(self) -> Tuple[pyplot.Figure, Sequence[pyplot.Axes]]:
        return self.fig, self.axs

    def __exit__(self, exc_type, exc_val, exc_tb):
        pyplot.cla()
        pyplot.close(self.fig)
        pyplot.clf()


if __name__ == "__main__":

    def deiajsd():
        for a in range(100):
            with SubplotSession() as a:
                fig, (ax1,) = a
                ax1.set_ylabel("test")

    deiajsd()
