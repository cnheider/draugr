#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04-01-2021
           """

import numpy

__all__ = ["DisplaySampler"]


class DisplaySampler(object):
    """
    # A class that will downsample the data and recompute when zoomed."""

    def __init__(self, x_data, y_data, verbose: bool = False):
        self._y_data = y_data
        self._x_data = x_data
        self.max_points = 50
        self.delta = x_data[-1] - x_data[0]
        self.verbose = verbose

    def downsample(self, xstart, xend):
        """ """
        mask = (self._x_data > xstart) & (
            self._x_data < xend
        )  # get the points in the view range

        mask = numpy.convolve([1, 1], mask, mode="same").astype(
            bool
        )  # dilate the mask by one to catch the points just outside
        # of the view range to not truncate the line

        ratio = max(
            numpy.sum(mask) // self.max_points, 1
        )  # sort out how many points to drop

        x_data, y_data = (
            self._x_data[mask][::ratio],
            self._y_data[mask][::ratio],
        )  # mask data and downsample data

        if self.verbose:
            print(f"using {len(y_data)} of {numpy.sum(mask)} visible points")

        return x_data, y_data

    def update(self, ax):
        """Update the line"""
        limits = ax.viewLim
        if numpy.abs(limits.width - self.delta) > 1e-8:
            self.delta = limits.width
            self.line.set_data(*self.downsample(*limits.intervalx))
            ax.figure.canvas.draw_idle()


if __name__ == "__main__":

    def asdsad() -> None:
        """
        :rtype: None
        """
        from matplotlib import pyplot

        # Create a signal
        xdata = numpy.linspace(16, 365, (365 - 16) * 4)
        ydata = numpy.sin(2 * numpy.pi * xdata / 153) + numpy.cos(
            2 * numpy.pi * xdata / 127
        )

        sampler_display = DisplaySampler(xdata, ydata)

        fig, ax = pyplot.subplots()

        # Hook up the line
        (sampler_display.line,) = ax.plot(xdata, ydata, "o-")
        ax.set_autoscale_on(False)  # Otherwise, infinite loop

        # Connect for changing the view limits
        ax.callbacks.connect("xlim_changed", sampler_display.update)
        ax.set_xlim(16, 365)

    asdsad()
