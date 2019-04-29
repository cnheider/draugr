#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "cnheider"

import numpy as np

from draugr import terminal_plot


def test_plot():
    terminal_plot(np.tile(range(9), 4), plot_character="o")
    assert True


if __name__ == "__main__":
    test_plot()
