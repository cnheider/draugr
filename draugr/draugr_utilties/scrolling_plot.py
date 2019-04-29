#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib import animation

__author__ = "cnheider"
__doc__ = ""

import matplotlib.pyplot as plt

import numpy as np


def main():
    delta = 0.01
    window_length = 80
    bins = 10

    array = np.random.random((bins, window_length))

    def get_sample():
        a = np.zeros((bins, 1))
        a[np.random.randint(0, bins)] = 255.0
        return a

    def update_fig(n):
        data = get_sample()

        im_data = im.get_array()[:, 1:]
        im_data = np.hstack((im_data, data))

        im.set_array(im_data)

        return im

    fig = plt.figure()
    im = plt.imshow(array, aspect="auto", interpolation="none")

    plt.xlabel("Step")
    plt.ylabel("Action Index")
    plt.title("Episode")

    anim = animation.FuncAnimation(
        fig, update_fig, blit=False, fargs=(), interval=delta
    )

    try:
        plt.show()
    except:
        print("Plot Closed")


if __name__ == "__main__":
    main()
