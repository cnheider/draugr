#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib import pyplot

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/10/2019
           """


def horizontal_imshow(images, columns=4, figsize=(20, 10), **kwargs):
    """Small helper function for creating horizontal subplots with pyplot"""
    pyplot.figure(figsize=figsize)
    for i, image in enumerate(images):
        pyplot.subplot(len(images) / columns + 1, columns, i + 1)
        pyplot.imshow(image, **kwargs)
