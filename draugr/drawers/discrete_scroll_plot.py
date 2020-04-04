#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import queue
import threading
from typing import Iterator, Sequence, Sized, Tuple

import matplotlib
from matplotlib import animation

from draugr.drawers.drawer import Drawer

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

__all__ = ["DiscreteScrollPlot", "discrete_scroll_plot"]

from matplotlib import pyplot

import numpy


class DiscreteScrollPlot(Drawer):
    """
Waterfall plot

"""

    def __init__(
        self,
        num_actions: int,
        window_length: int = None,
        labels: Sequence = None,
        title: str = "",
        time_label: str = "Time",
        data_label: str = "Action Index",
        vertical: bool = True,
        reverse: bool = True,
        overwrite: bool = False,
        placement: Tuple = (0, 0),
        render: bool = True,
    ):
        self.fig = None
        if not render:
            return
        self._num_actions = num_actions

        if not window_length:
            window_length = num_actions * 20

        array = numpy.zeros((window_length, num_actions))

        self.vertical = vertical
        self.overwrite = overwrite
        self.reverse = reverse
        self.window_length = window_length
        self.n = 0

        if vertical:
            self.fig = pyplot.figure(figsize=(window_length / 10, 2))
            extent = [-window_length, 0, 0, num_actions]
        else:
            self.fig = pyplot.figure(figsize=(2, window_length / 10))
            extent = [num_actions, 0, 0, -window_length]

        """
fig_manager = pyplot.get_current_fig_manager()
geom = fig_manager.window.geometry()
x, y, dx, dy = geom.getRect()
fig_manager.window.setGeometry(*placement, dx, dy)
fig_manager.window.SetPosition((500, 0))
"""
        self.placement = placement

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

    @staticmethod
    def move_figure(figure: pyplot.Figure, x: int = 0, y: int = 0):
        """Move figure's upper left corner to pixel (x, y)"""
        backend = matplotlib.get_backend()
        if hasattr(figure.canvas.manager, "window"):
            window = figure.canvas.manager.window
            if backend == "TkAgg":
                window.wm_geometry(f"+{x:d}+{y:d}")
            elif backend == "WXAgg":
                window.SetPosition((x, y))
            else:
                # This works for QT and GTK
                # You can also use window.setGeometry
                window.move(x, y)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fig:
            pyplot.close(self.fig)

    def draw(self, data: Sized, delta: float = 1 / 120):
        """

:param data:
:param delta: 1 / 60 for 60fps
:return:
"""

        array = self.im.get_array()

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
        pyplot.draw()
        if self.n <= 1:
            self.move_figure(self.fig, *self.placement)
        self.n += 1
        if delta:
            pyplot.pause(delta)


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

    def ma():
        data = queue.Queue(100)

        class QueueGen:
            def __iter__(self):
                return self

            def __next__(self):
                return self.get()

            def __call__(self, *args, **kwargs):
                return self.__next__()

            def add(self, a):
                return data.put(a)

            def get(self):
                return data.get()

        def get_sample(num_actions=3):
            a = numpy.zeros(num_actions)
            a[numpy.random.randint(0, num_actions)] = 1.0
            return a

        class MyDataFetchClass(threading.Thread):
            def __init__(self, data):

                threading.Thread.__init__(self)

                self._data = data

            def run(self):

                while True:
                    self._data.add(get_sample())

        d = QueueGen()

        MyDataFetchClass(d).start()

        anim = discrete_scroll_plot(iter(d), labels=("a", "b", "c"))

        try:
            pyplot.show()
        except:
            print("Plot Closed")

    delta = 1 / 60

    s = DiscreteScrollPlot(3)
    for _ in range(100):
        s.draw(numpy.random.rand(3))
