#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 9/2/19
           """

from draugr.visualisation import activation_scroll_plot, terminal_plot


import numpy


def test_timeseries_scroll_plot():
    gen = iter(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]
    )
    activation_scroll_plot(gen, labels=("a", "b", "c"))
    assert True


if __name__ == "__main__":
    test_timeseries_scroll_plot()
