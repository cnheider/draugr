#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import queue
import threading

import matplotlib
from matplotlib import animation

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

if __name__ == "__main__":

    from matplotlib import pyplot
    import numpy as np
    import pandas as pd

    x = np.arange(100)
    y = np.random.rand(100)
    df = pd.DataFrame({"x": x, "y": y})
    df2 = df[0:0]

    pyplot.ion()
    fig, ax = pyplot.subplots()
    i = 0
    while i < len(df):
        df2 = df2.append(df[i : i + 1])
        ax.clear()
        df2.plot(x="x", y="y", ax=ax)
        pyplot.draw()
        pyplot.pause(0.2)
        i += 1
    pyplot.show()
