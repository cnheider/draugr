#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Iterator, Sequence, Tuple

from matplotlib import animation

from draugr.drawers.mpl_drawers.mpldrawer import MplDrawer
from draugr.numpy_utilities import recursive_flatten_numpy

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

__all__ = ["DiscreteScrollPlot", "discrete_scroll_plot"]

from matplotlib import pyplot

import numpy
from warg import passes_kws_to


class DiscreteScrollPlot(MplDrawer):
    """
    Waterfall plot
    only supports a single trajectory at a time, do not supply parallel trajectories to draw method, will get truncated to num actions, effectively dropping actions for other envs than the first."""

    @passes_kws_to(MplDrawer.__init__)
    def __init__(
        self,
        num_bins: int,
        window_length: int = None,
        labels: Sequence = None,
        title: str = "",
        time_label: str = "Time",
        data_label: str = "Bin Index",
        vertical: bool = True,
        reverse: bool = True,
        overwrite: bool = False,
        render: bool = True,
        figure_size: Tuple[int, int] = None,
        **kwargs
    ):

        super().__init__(render=render, figure_size=figure_size, **kwargs)
        if not render:
            return

        self._num_actions = num_bins

        if not window_length:
            window_length = num_bins * 20

        array = numpy.zeros((window_length, num_bins))

        self.vertical = vertical
        self.overwrite = overwrite
        self.reverse = reverse
        self.window_length = window_length

        if not figure_size:
            if vertical:
                self.fig = pyplot.figure(figsize=(window_length / 10, 2))
                extent = [-window_length, 0, 0, num_bins]
            else:
                self.fig = pyplot.figure(figsize=(2, window_length / 10))
                extent = [num_bins, 0, 0, -window_length]
        else:
            self.fig = pyplot.figure(figsize=figure_size)
            extent = [figure_size[0], 0, 0, -figure_size[1]]

        if vertical:
            array = array.T

        self.im = pyplot.imshow(
            array,
            cmap="gray",
            aspect="auto",
            interpolation="none",
            vmin=0.0,
            vmax=1.0,
            extent=extent,
        )

        b = numpy.arange(0.5, num_bins + 0.5, 1)
        if not labels:
            labels = numpy.arange(0, num_bins, 1)

        if vertical:
            pyplot.yticks(b, labels, rotation=45)
        else:
            pyplot.xticks(b, labels, rotation=45)

        if vertical:
            pyplot.xlabel(time_label)
            pyplot.ylabel(data_label)
        else:
            pyplot.xlabel(data_label)
            pyplot.ylabel(time_label)

        pyplot.title(title)
        pyplot.tight_layout()

    def _draw(self, data: Sequence[int]):
        """

        :param data:
        :return:"""

        array = self.im.get_array()

        data = recursive_flatten_numpy(data)
        # assert isinstance(data[0], int), f'Data was {data}'
        """
if not isinstance(data[0], int):
data = data[0]
"""

        data = data[: self._num_actions]

        if self.vertical:
            array = array.T

        if not self.overwrite:
            if not self.reverse:
                array = numpy.vstack((array[1:], data))
            else:
                array = numpy.vstack((data, array[:-1]))
        else:
            array[self.n % self.window_length] = data

        if self.vertical:
            array = array.T

        self.im.set_array(array)


def discrete_scroll_plot(
    vector_provider: Iterator,
    delta: float = 1 / 60,
    window_length: int = None,
    labels: Sequence = None,
    title: str = "",
    time_label: str = "Time",
    data_label: str = "Action Index",
    vertical: bool = True,
    reverse: bool = True,
    overwrite: bool = False,
):
    """ """
    d = vector_provider.__next__()
    num_actions = len(d)
    if not window_length:
        window_length = num_actions * 20

    if not overwrite:
        array = numpy.zeros((window_length - 1, num_actions))
        if not reverse:
            array = numpy.vstack((array, d))
        else:
            array = numpy.vstack((d, array))
    else:
        array = numpy.zeros((window_length, num_actions))
        if not reverse:
            array[0] = d
        else:
            array[-1] = d

    def update_fig(n):
        """ """
        data = vector_provider.__next__()
        array = im.get_array()

        if vertical:
            array = array.T

        if not overwrite:
            if not reverse:
                array = numpy.vstack((array[1:], data))
            else:
                array = numpy.vstack((data, array[:-1]))
        else:
            array[n % window_length] = data

        if vertical:
            array = array.T

        im.set_array(array)

        return im

    if vertical:
        fig = pyplot.figure(figsize=(window_length / 10, 2))
        extent = [-window_length, 0, 0, num_actions]
    else:
        fig = pyplot.figure(figsize=(2, window_length / 10))
        extent = [num_actions, 0, 0, -window_length]

    if vertical:
        array = array.T

    im = pyplot.imshow(
        array,
        cmap="gray",
        aspect="auto",
        interpolation="none",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
    )

    b = numpy.arange(0.5, num_actions + 0.5, 1)
    if not labels:
        labels = numpy.arange(0, num_actions, 1)

    if vertical:
        pyplot.yticks(b, labels, rotation=45)
    else:
        pyplot.xticks(b, labels, rotation=45)

    if vertical:
        pyplot.xlabel(time_label)
        pyplot.ylabel(data_label)
    else:
        pyplot.xlabel(data_label)
        pyplot.ylabel(time_label)

    pyplot.title(title)
    pyplot.tight_layout()

    anim = animation.FuncAnimation(
        fig, update_fig, blit=False, fargs=(), interval=delta
    )

    return anim


if __name__ == "__main__":

    def siajdisajd() -> None:
        """
        :rtype: None
        """
        import queue
        import threading

        def ma():
            """ """
            data = queue.Queue(100)

            class QueueGen:
                def __iter__(self):
                    return self

                def __next__(self):
                    return self.get()

                def __call__(self, *args, **kwargs):
                    return self.__next__()

                def add(self, a):
                    """ """
                    return data.put(a)

                def get(self):
                    """ """
                    return data.get()

            def get_sample(num_actions=3):
                """ """
                a = numpy.zeros(num_actions)
                a[numpy.random.randint(0, num_actions)] = 1.0
                return a

            class MyDataFetchClass(threading.Thread):
                """ """

                def __init__(self, data):
                    threading.Thread.__init__(self)

                    self._data = data

                def run(self):
                    """ """
                    while True:
                        self._data.add(get_sample())

            d = QueueGen()

            MyDataFetchClass(d).start()

            anim = discrete_scroll_plot(iter(d), labels=("a", "b", "c"))

            try:
                pyplot.show()
            except:
                print("Plot Closed")

        def asda():
            """ """
            s = DiscreteScrollPlot(3, default_delta=None)
            for _ in range(100):
                s.draw(numpy.random.rand(3))

        asda()

    siajdisajd()
