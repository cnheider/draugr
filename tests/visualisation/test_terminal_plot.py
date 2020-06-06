#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from draugr.writers import terminal_plot

__author__ = "Christian Heider Nielsen"

import numpy


def test_terminal_plot():
    terminal_plot(numpy.tile(range(9), 4), plot_character="o")
    assert True


if __name__ == "__main__":
    test_terminal_plot()
