#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"

from pathlib import Path

import numpy
import scipy.misc
from matplotlib import pyplot
from matplotlib.image import imread

__all__ = ["display_depth_map_3d"]


def display_depth_map_3d(
    data_set_directory: Path = Path.home()
    / "Datasets"
    / "neodroid"
    / "depth"
    / "80.png",
):
    """

    :param data_set_directory:
    :type data_set_directory:"""
    # lena = scipy.misc.ascent()

    # downscaling has a 'smoothing' effect
    # lena = scipy.misc.imresize(lena, 0.15, interp='cubic')

    img = imread(data_set_directory)
    img = img[..., 0]

    # def ivas(x, cam_ang):
    #  return x * math.cos(math.radians(90 - cam_ang))

    # image_length = img.shape[0]
    # camera_angle = 45.
    # ys = numpy.array([ivas(x, camera_angle) / 255 for x in range(0, image_length)])
    # ys = numpy.repeat(ys, img.shape[0], axis=0).reshape((img.shape[0], img.shape[0]))
    # img = img - ys

    img = scipy.misc.imresize(img, 0.2, interp="cubic")

    # img = img/255.

    # create the x and y coordinate arrays (here we just use pixel indices)
    d0 = img.shape[0]
    d1 = img.shape[1] / 2
    # xx, yy = numpy.mgrid[-d0:d0, -d1:d1]
    xx, yy = numpy.mgrid[0:d0, -d1:d1]

    fig = pyplot.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(xx, yy, img, rstride=1, cstride=1, cmap=pyplot.cm.gray, linewidth=0)

    pyplot.show()


if __name__ == "__main__":
    display_depth_map_3d()
