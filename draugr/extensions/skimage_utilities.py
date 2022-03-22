#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04-04-2021
           """

import warnings
from typing import Any

import numpy
from skimage import color, img_as_ubyte

__all__ = ["rgb_to_grayscale"]

from draugr.numpy_utilities.mixing import mix_channels


def rgb_to_grayscale(obs: Any) -> Any:
    """Convert a 3-channel color observation image to grayscale and uint8.

    Args:
       obs (numpy.ndarray): Observation array, conforming to observation_space

    Returns:
       numpy.ndarray: 1-channel grayscale version of obs, represented as uint8

    """
    with warnings.catch_warnings():
        # Suppressing warning for possible precision loss when converting
        # from float64 to uint8
        warnings.simplefilter("ignore")
        return img_as_ubyte(color.rgb2gray(obs))


if __name__ == "__main__":

    def asuijhd():
        a = numpy.expand_dims(numpy.eye(5), -1)
        b = numpy.expand_dims(numpy.flip(numpy.eye(5), 0), -1)
        c = numpy.expand_dims(numpy.ones((5, 5)), -1)
        stacked = numpy.swapaxes(numpy.array((a, b, c)), 0, -1)
        print(stacked.shape)
        print(mix_channels(stacked))

    asuijhd()
