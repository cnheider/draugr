#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path


from warg import ensure_in_sys_path, find_nearest_ancestral_relative

ensure_in_sys_path(find_nearest_ancestral_relative("draugr").parent)

from draugr.drawers.terminal import terminal_plot

__author__ = "Christian Heider Nielsen"

import numpy


def test_terminal_plot():
    terminal_plot(numpy.tile(range(9), 4), plot_character="o")
    assert True


if __name__ == "__main__":
    test_terminal_plot()
