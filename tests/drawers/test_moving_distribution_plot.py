#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import pytest

from draugr.drawers.mpl_drawers.distribution_plot import DistributionPlot

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 9/2/19
           """


@pytest.mark.skip
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
    # moving_distribution_plot(data_generator, labels=("a", "b", "c"))
    delta = 1.0 / 60.0

    s = DistributionPlot()
    for LATEST_GPU_STATS in range(100):
        s.draw(numpy.random.sample(), delta)
    assert True


if __name__ == "__main__":
    test_moving_distribution_plot()
