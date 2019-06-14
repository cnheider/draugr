#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import queue
import threading

import matplotlib
import numpy
from matplotlib import animation

__author__ = "cnheider"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

import matplotlib.pyplot as plt

import numpy as np


class scroll_plot_class(object):
    def __init__(
        self,
        num_actions,
        window_length=None,
        labels=None,
        title="",
        time_label="Time",
        data_label="Action Index",
        vertical=True,
        reverse=True,
        overwrite=False,
        placement=(0, 0),
        render=True,
    ):
        self.fig = None
        if not render:
            return
        self._num_actions = num_actions

        if not window_length:
            window_length = num_actions * 20

        array = np.zeros((window_length, num_actions))

        self.vertical = vertical
        self.overwrite = overwrite
        self.reverse = reverse
        self.window_length = window_length
        self.n = 0

        if vertical:
            self.fig = plt.figure(figsize=(window_length / 10, 2))
            extent = [-window_length, 0, 0, num_actions]
        else:
            self.fig = plt.figure(figsize=(2, window_length / 10))
            extent = [num_actions, 0, 0, -window_length]

        """
fig_manager = plt.get_current_fig_manager()
geom = fig_manager.window.geometry()
x, y, dx, dy = geom.getRect()
fig_manager.window.setGeometry(*placement, dx, dy)
fig_manager.window.SetPosition((500, 0))
"""
        self.placement = placement

        if vertical:
            array = array.T

        self.im = plt.imshow(
            array,
            cmap="gray",
            aspect="auto",
            interpolation="none",
            vmin=0.0,
            vmax=1.0,
            extent=extent,
        )

        b = np.arange(0.5, num_actions + 0.5, 1)
        if not labels:
            labels = np.arange(0, num_actions, 1)

        if vertical:
            plt.yticks(b, labels, rotation=45)
        else:
            plt.xticks(b, labels, rotation=45)

        if vertical:
            plt.xlabel(time_label)
            plt.ylabel(data_label)
        else:
            plt.xlabel(data_label)
            plt.ylabel(time_label)

        plt.title(title)
        plt.tight_layout()

    @staticmethod
    def move_figure(figure: plt.Figure, x=0, y=0):
        """Move figure's upper left corner to pixel (x, y)"""
        backend = matplotlib.get_backend()
        if hasattr(figure.canvas.manager, "window"):
            window = figure.canvas.manager.window
            if backend == "TkAgg":
                window.wm_geometry("+%d+%d" % (x, y))
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
            plt.close(self.fig)

    def draw(self, data, delta=1 / 120):
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
                striped = np.delete(array, 0, 0)
                array = np.vstack((striped, data))
            else:
                striped = np.delete(array, -1, 0)
                array = np.vstack((data, striped))
        else:
            array[self.n % self.window_length] = data

        if self.vertical:
            array = array.T

        self.im.set_array(array)
        plt.draw()
        if self.n <= 1:
            self.move_figure(self.fig, *self.placement)
        self.n += 1
        if delta:
            plt.pause(delta)


def scroll_plot(
    vector_provider,
    delta=1 / 60,
    window_length=None,
    labels=None,
    title="",
    time_label="Time",
    data_label="Action Index",
    vertical=True,
    reverse=True,
    overwrite=False,
):
    d = vector_provider.__next__()
    num_actions = len(d)
    if not window_length:
        window_length = num_actions * 20

    if not overwrite:
        array = np.zeros((window_length - 1, num_actions))
        if not reverse:
            array = np.vstack((array, d))
        else:
            array = np.vstack((d, array))
    else:
        array = np.zeros((window_length, num_actions))
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
                striped = np.delete(array, 0, 0)
                array = np.vstack((striped, data))
            else:
                striped = np.delete(array, -1, 0)
                array = np.vstack((data, striped))
        else:
            array[n % window_length] = data

        if vertical:
            array = array.T

        im.set_array(array)

        return im

    if vertical:
        fig = plt.figure(figsize=(window_length / 10, 2))
        extent = [-window_length, 0, 0, num_actions]
    else:
        fig = plt.figure(figsize=(2, window_length / 10))
        extent = [num_actions, 0, 0, -window_length]

    if vertical:
        array = array.T

    im = plt.imshow(
        array,
        cmap="gray",
        aspect="auto",
        interpolation="none",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
    )

    b = np.arange(0.5, num_actions + 0.5, 1)
    if not labels:
        labels = np.arange(0, num_actions, 1)

    if vertical:
        plt.yticks(b, labels, rotation=45)
    else:
        plt.xticks(b, labels, rotation=45)

    if vertical:
        plt.xlabel(time_label)
        plt.ylabel(data_label)
    else:
        plt.xlabel(data_label)
        plt.ylabel(time_label)

    plt.title(title)
    plt.tight_layout()

    anim = animation.FuncAnimation(
        fig, update_fig, blit=False, fargs=(), interval=delta
    )

    return anim


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
        a = np.zeros(num_actions)
        a[np.random.randint(0, num_actions)] = 1.0
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

    anim = scroll_plot(iter(d), labels=("a", "b", "c"))

    try:
        plt.show()
    except:
        print("Plot Closed")


if __name__ == "__main__":
    delta = 1 / 60

    s = scroll_plot_class(3)
    for LATEST_GPU_STATS in range(100):
        s.draw(numpy.random.rand(3))
