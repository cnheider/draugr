#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from draugr import discrete_scroll_plot

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 9/2/19
           """


def test_moving_distribution_plot():
    data_generator = iter(
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
    discrete_scroll_plot(data_generator, labels=("a", "b", "c"))
    assert True


if __name__ == "__main__":
    test_moving_distribution_plot()
