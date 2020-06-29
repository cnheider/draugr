#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 19/03/2020
           """

from typing import Tuple

import cv2
import numpy
from PIL import Image

from draugr.opencv_utilities.bounding_boxes.colors import compute_color_for_labels
from draugr.python_utilities.colors import RGB

__all__ = ["find_contours", "draw_masks"]


def find_contours(*args, **kwargs):
    """
Wraps cv2.findContours to maintain compatibility between versions 3 and 4
Returns:
contours, hierarchy
"""
    if cv2.__version__.startswith("4"):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith("3"):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError("cv2 must be either version 3 or 4 to call this method")
    return contours, hierarchy


def draw_masks(
    image,
    masks,
    labels=None,
    border=True,
    border_width: float = 2,
    border_color: Tuple = RGB(255, 255, 255),
    alpha: float = 0.5,
    color: Tuple = None,
) -> numpy.ndarray:
    """
Args:
image: numpy array image, shape should be (height, width, channel)
masks: (N, 1, Height, Width)
labels: mask label
border: draw border on mask
border_width: border width
border_color: border color
alpha: mask alpha
color: mask color
Returns:
numpy.ndarray
"""
    if isinstance(image, Image.Image):
        image = numpy.array(image)
    assert isinstance(image, numpy.ndarray)
    masks = numpy.array(masks)
    for i, mask in enumerate(masks):
        mask = mask.squeeze()[..., None].astype(numpy.bool)

        label = labels[i] if labels is not None else 1
        _color = compute_color_for_labels(label) if color is None else tuple(color)

        image = numpy.where(
            mask, mask * numpy.array(_color) * alpha + image * (1 - alpha), image
        )
        if border:
            contours, hierarchy = find_contours(
                mask.astype(numpy.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(
                image,
                contours,
                -1,
                border_color,
                thickness=border_width,
                lineType=cv2.LINE_AA,
            )
    return image.astype(numpy.uint8)
