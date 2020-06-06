#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from draugr.drawers import discrete_scroll_plot

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 9/2/19
           """


def test_activation_scroll_plot():
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
    discrete_scroll_plot(gen, labels=("a", "b", "c"))
    assert True


if __name__ == "__main__":
    test_activation_scroll_plot()
