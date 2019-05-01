#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import queue
import threading

from matplotlib import animation

__author__ = "cnheider"
__doc__ = ""

import matplotlib.pyplot as plt

import numpy as np


def scroll_plot(
    vector_provider,
    delta=1 / 24,
    window_length=None,
    labels=None,
    title="",
    x_label="Time",
    y_label="Action Index",
):
    d = vector_provider()
    num_actions = len(d)
    if not window_length:
        window_length = num_actions * 20

    array = np.zeros((num_actions, window_length - 1))
    array = np.hstack((array, d))

    def update_fig(n):
        data = vector_provider()
        im_data = np.hstack((np.delete(im.get_array(), 0, 1), data))
        im.set_array(im_data)

        return im

    fig = plt.figure(figsize=(window_length / 10, 2))

    im = plt.imshow(
        array,
        cmap="gray",
        aspect="auto",
        interpolation="none",
        vmin=0.0,
        vmax=1.0,
        extent=[-window_length, 0, 0, num_actions],
    )

    b = np.arange(0.5, num_actions + 0.5, 1)
    if not labels:
        labels = np.arange(0, num_actions, 1)
    plt.yticks(b, labels, rotation=45)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()

    anim = animation.FuncAnimation(
        fig, update_fig, blit=False, fargs=(), interval=delta
    )

    try:
        plt.show()
    except:
        print("Plot Closed")


if __name__ == "__main__":

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
        a = np.zeros((num_actions, 1))
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

    scroll_plot(iter(d), labels=("a", "b", "c"))
