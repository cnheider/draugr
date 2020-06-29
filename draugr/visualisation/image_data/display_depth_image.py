#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"

from pathlib import Path

import matplotlib.image as mpimg
from matplotlib import pyplot

__all__ = ["display_depth_map"]


def display_depth_map(
    data_set_directory=Path.home() / "Datasets" / "neodroid" / "depth" / "80.png"
):
    """

    :param data_set_directory:
    :type data_set_directory:
    """
    # img = Image.open(data_set_directory + file_name).convert('LA')
    # img_array =numpy.asarray(img)
    # print(img_array.shape)

    img = mpimg.imread(data_set_directory)
    img = img[..., 0]
    img = img / 255

    pyplot.imshow(img, cmap=pyplot.get_cmap("gray"))
    pyplot.show()


if __name__ == "__main__":
    display_depth_map()
