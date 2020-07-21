#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17/07/2020
           """

import numpy
from PIL import Image


__all__ = ["pil_merge_images", "pil_img_to_np_array", "np_array_to_pil_img"]


def pil_img_to_np_array(data_path, desired_size=None, expand=False, view=False):
    """
  Util function for loading RGB image into a numpy array.

  Returns array of shape (1, H, W, C).
  """
    img = Image.open(data_path)
    img = img.convert("RGB")
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    if view:
        img.show()
    x = numpy.asarray(img, dtype="float32")
    if expand:
        x = numpy.expand_dims(x, axis=0)
    x /= 255.0
    return x


def np_array_to_pil_img(x):
    """
  Util function for converting anumpy array to a PIL img.

  Returns PIL RGB img.
  """
    x = numpy.asarray(x)
    x = x + max(-numpy.min(x), 0)
    x_max = numpy.max(x)
    if x_max != 0:
        x /= x_max
    x *= 255
    return Image.fromarray(x.astype("uint8"), "RGB")


def pil_merge_images(image1, image2):
    """Merge two images into one, displayed side by side.
  """
    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new("RGB", (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result
