#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 01/02/2022
           """

import numpy

__all__ = ["mix_channels"]


def mix_channels(raster: numpy.ndarray) -> numpy.ndarray:
    # TODO: MAYBE ASSERT SHAPE?
    num_channels = raster.shape[-1]
    return numpy.dot(
        raster[..., :num_channels],
        numpy.ones(num_channels) / num_channels,
    )
