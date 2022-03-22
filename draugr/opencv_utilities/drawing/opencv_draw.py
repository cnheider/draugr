#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 19/03/2020
           """

from typing import Sequence, Tuple, Union

import cv2
import numpy
from PIL import Image

from draugr.opencv_utilities.namespaces.enums import (
    LineTypeEnum,
    ContourRetrievalModeEnum,
)
from draugr.python_utilities.colors import RGB, compute_color_for_labels

__all__ = ["find_contours", "draw_masks"]


def find_contours(*args, **kwargs) -> Tuple:
    """
    Wraps cv2.findContours to maintain compatibility between versions 3 and 4
    Returns:
    contours, hierarchy"""
    if cv2.__version__.startswith("4"):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith("3"):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError("cv2 must be either version 3 or 4 to call this method")
    return contours, hierarchy


def draw_masks(
    image: Union[Image.Image, numpy.ndarray],
    masks: Union[Image.Image, numpy.ndarray],
    *,
    labels: Sequence = None,
    border: bool = True,
    border_width: int = 1,  # If it is negative, the contour interiors are drawn.
    border_color: Tuple = RGB(255, 255, 255),
    alpha: float = 0.5,
    color: Tuple = None,
    line_type: LineTypeEnum = LineTypeEnum.anti_aliased
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
    numpy.ndarray"""

    line_type = LineTypeEnum(line_type)
    if isinstance(image, Image.Image):
        image = numpy.array(image)
    assert isinstance(image, numpy.ndarray)
    if isinstance(masks, Image.Image):
        masks = numpy.array(masks)
    assert isinstance(masks, numpy.ndarray)
    # TODO: ASSERT 3/4 CHANNELS!
    if labels is None:
        labels = list(range(masks.shape[0]))
    for i, mask in enumerate(masks):
        mask = mask.squeeze()[..., None].astype(numpy.bool)

        label = labels[i]
        mask_color = compute_color_for_labels(label) if color is None else tuple(color)

        image = numpy.where(
            mask, mask * numpy.array(mask_color) * alpha + image * (1 - alpha), image
        )
        if border:
            contours, hierarchy = find_contours(
                mask.astype(numpy.uint8),
                ContourRetrievalModeEnum.tree.value,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            image = cv2.drawContours(
                image,
                contours,
                -1,
                border_color,
                thickness=border_width,
                lineType=line_type.value,
            )
    return image.astype(numpy.uint8)


if __name__ == "__main__":
    pass
