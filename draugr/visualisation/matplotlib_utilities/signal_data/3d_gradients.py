#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20-01-2021
           """

import numpy
from matplotlib import cm, pyplot

__all__ = ["plot_gradients_3d"]


def plot_gradients_3d() -> pyplot.Figure:
    """ """
    fig = pyplot.figure()
    ax = fig.gca(projection="3d")
    x = numpy.arange(-5, 5, 0.25)
    y = numpy.arange(-5, 5, 0.25)
    x, y = numpy.meshgrid(x, y)
    r = numpy.sqrt(x**2 + y**2)
    z = numpy.sin(r)
    gx, gy = numpy.gradient(z)  # gradients with respect to x and y
    g = (gx**2 + gy**2) ** 0.5  # gradient magnitude
    n = g / g.max()  # normalize 0..1
    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        facecolors=cm.jet(n),
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    return fig


if __name__ == "__main__":
    plot_gradients_3d()
    pyplot.show()
