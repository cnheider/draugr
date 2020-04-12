#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04/04/2020
           """

from matplotlib import pyplot
import numpy

with plt.xkcd():
    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-30, 10])

    data = numpy.ones(100)
    data[70:] -= numpy.arange(30)

    ax.annotate(
        r"""THE DAY I REALIZED
I COULD COOK BACON
WHENEVER I WANTED""",
        xy=(70, 1),
        arrowprops=dict(arrowstyle="->"),
        xytext=(15, -10),
    )

    ax.plot(data)

    ax.set_xlabel("time")
    ax.set_ylabel("my overall health")

    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax.bar([0, 1], [0, 100], 0.25)
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CONFIRMED BY\nEXPERIMENT", "REFUTED BY\nEXPERIMENT"])
    ax.set_xlim([-0.5, 1.5])
    ax.set_yticks([])
    ax.set_ylim([0, 110])

    ax.set_title("CLAIMS OF SUPERNATURAL POWERS")

plt.show()
