#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "cnheider"

import numpy as np

from draugr import scroll_plot, terminal_plot


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


def test_plot():
    terminal_plot(np.tile(range(9), 4), plot_character="o")
    assert True


if __name__ == "__main__":
    test_plot()
