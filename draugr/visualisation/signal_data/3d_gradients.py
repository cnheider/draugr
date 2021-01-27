#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20-01-2021
           """

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, pyplot
import numpy

__all__ = ["plot_gradients_3d"]


def plot_gradients_3d() -> pyplot.Figure:
    fig = pyplot.figure()
    ax = fig.gca(projection="3d")
    X = numpy.arange(-5, 5, 0.25)
    Y = numpy.arange(-5, 5, 0.25)
    X, Y = numpy.meshgrid(X, Y)
    R = numpy.sqrt(X ** 2 + Y ** 2)
    Z = numpy.sin(R)
    Gx, Gy = numpy.gradient(Z)  # gradients with respect to x and y
    G = (Gx ** 2 + Gy ** 2) ** 0.5  # gradient magnitude
    N = G / G.max()  # normalize 0..1
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        facecolors=cm.jet(N),
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    return fig


if __name__ == "__main__":
    plot_gradients_3d()
    pyplot.show()
