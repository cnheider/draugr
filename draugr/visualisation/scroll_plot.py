#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import queue
import threading

from matplotlib import animation

__author__ = "cnheider"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

import matplotlib.pyplot as plt

import numpy as np


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
