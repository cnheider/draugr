#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sequence

from draugr.drawers.mpl_drawers.mpldrawer import MplDrawer
from warg import passes_kws_to

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 22/11/2019

           """

from matplotlib import pyplot

import numpy

__all__ = ["DistributionPlot"]


class DistributionPlot(MplDrawer):
    """ """

    @passes_kws_to(MplDrawer.__init__)
    @passes_kws_to(pyplot.hist)
    def __init__(
        self,
        window_length: int = None,
        title: str = "",
        data_label: str = "Magnitude",
        reverse: bool = False,
        overwrite: bool = False,
        render: bool = True,
        **kwargs
    ):
        super().__init__(render=render, **kwargs)

        if not render:
            return

        self.fig = pyplot.figure(figsize=(4, 4))

        self.overwrite = overwrite
        self.reverse = reverse
        self.window_length = window_length

        self.array = []
        self.hist_kws = kwargs
        pyplot.hist(self.array, **self.hist_kws)

        pyplot.ylabel(data_label)

        pyplot.title(title)
        pyplot.tight_layout()

    def _draw(self, data: Sequence):
        """

        :param data:
        :return:"""
        if not isinstance(data, Sequence):
            data = [data]

        self.array.extend(data)
        pyplot.cla()
        pyplot.hist(self.array, **self.hist_kws)


if __name__ == "__main__":
    delta = 1.0 / 60.0

    s = DistributionPlot()
    for _ in range(100):
        s.draw(numpy.random.sample(), delta)
