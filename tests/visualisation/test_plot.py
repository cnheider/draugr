#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from draugr.visualisation import scroll_plot, terminal_plot

__author__ = "cnheider"

import numpy


def test_scroll_plot():
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
    scroll_plot(gen, labels=("a", "b", "c"))
    assert True


def test_terminal_plot():
    terminal_plot(numpy.tile(range(9), 4), plot_character="o")
    assert True


if __name__ == "__main__":
    test_terminal_plot()
    test_scroll_plot()
