#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17/07/2020
           """

from enum import Enum
from typing import Tuple, Union

import numpy

__all__ = [
    "pil_merge_images",
    "pil_img_to_np_array",
    "np_array_to_pil_img",
    "pil_image_to_byte_array",
    "byte_array_to_pil_image",
]

import io

from PIL import Image
from sorcery import assigned_names


class PilModesEnum(Enum):
    """
      PIL pixel formats:

    RGB 24bits per pixel, 8-bit-per-channel RGB), 3 channels
    RGBA (8-bit-per-channel RGBA), 4 channels
    RGBa (8-bit-per-channel RGBA, remultiplied alpha), 4 channels
    1 - 1bpp, often for masks, 1 channel
    L - 8bpp, grayscale, 1 channel
    P - 8bpp, paletted, 1 channel
    I - 32-bit integers, grayscale, 1 channel
    F - 32-bit floats, grayscale, 1 channel
    CMYK - 8 bits per channel, 4 channels
    YCbCr - 8 bits per channel, 3 channels
    """

    OneBpp = "1"
    CMYK, F, HSV, I, L, LAB, P, RGB, RGBA, RGBX, YCbCr = assigned_names()


"""
1 (1-bit pixels, black and white, stored with one pixel per byte)

L (8-bit pixels, black and white)

P (8-bit pixels, mapped to any other mode using a color palette)

RGB (3x8-bit pixels, true color)

RGBA (4x8-bit pixels, true color with transparency mask)

CMYK (4x8-bit pixels, color separation)

YCbCr (3x8-bit pixels, color video format)

Note that this refers to the JPEG, and not the ITU-R BT.2020, standard

LAB (3x8-bit pixels, the L*a*b color space)

HSV (3x8-bit pixels, Hue, Saturation, Value color space)

I (32-bit signed integer pixels)

F (32-bit floating point pixels)

# Pillow also provides limited support for a few additional modes, including:

LA (L with alpha)

PA (P with alpha)

RGBX (true color with padding)

RGBa (true color with premultiplied alpha)

La (L with premultiplied alpha)

I;16 (16-bit unsigned integer pixels)

I;16L (16-bit little endian unsigned integer pixels)

I;16B (16-bit big endian unsigned integer pixels)

I;16N (16-bit native endian unsigned integer pixels)

BGR;15 (15-bit reversed true colour)

BGR;16 (16-bit reversed true colour)

BGR;24 (24-bit reversed true colour)

BGR;32 (32-bit reversed true colour)
"""


def pil_image_to_byte_array(image: Image.Image, *, coding: str = "PNG") -> bytes:
    """
    PNG encoded by default
    :param coding:
    :param image:
    :return:"""
    buffer = io.BytesIO()
    image.save(buffer, coding)
    return buffer.getvalue()


def byte_array_to_pil_image(byte_array: bytes) -> Image.Image:
    """

    :param byte_array:
    :return:"""
    return Image.open(io.BytesIO(byte_array))


def pil_img_to_np_array(
    data_path: Union[str, Image.Image],
    *,
    desired_size: Tuple[int, int] = None,
    expand: int = False
) -> numpy.ndarray:
    """
    Util function for loading RGB image into a numpy array.

    Returns array of shape (1, H, W, C)."""
    img = Image.open(data_path)
    img = img.convert("RGB")
    if desired_size:
        img = img.resize((desired_size[1], desired_size[0]))
    x = numpy.asarray(img, dtype="float32")
    if expand:
        x = numpy.expand_dims(x, axis=0)
    return x / 255.0


def np_array_to_pil_img(x: numpy.ndarray) -> Image.Image:
    """
    Util function for converting a numpy array to a PIL img.

    Returns PIL RGB img."""
    x = numpy.asarray(x)
    x = x + max(-numpy.min(x), 0)
    x_max = numpy.max(x)
    if x_max != 0:
        x /= x_max
    return Image.fromarray((x * 255.0).astype("uint8"), "RGB")


def pil_merge_images(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Merge two images into one, displayed side by side."""
    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new("RGB", (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result
