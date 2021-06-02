#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 04-04-2021
           """

import warnings
from typing import Any

from skimage import color, img_as_ubyte

__all__ = ["rgb_to_grayscale"]


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
