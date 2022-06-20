import warnings

import cv2

from draugr.opencv_utilities.namespaces.color_conversion_enum import (
    ColorConversionEnum,
)

__all__ = ["num_channels", "is_singular_channel", "to_color"]


def num_channels(image) -> int:
    """

    :param image:
    :type image:
    :return:
    :rtype:
    """
    return image.shape[-1] if image.ndim == 3 else 1


def is_singular_channel(image) -> bool:
    """

    :param image:
    :type image:
    :return:
    :rtype:
    """
    return num_channels(image) == 1


def to_color(image, conversion: ColorConversionEnum = ColorConversionEnum.gray2bgr):
    """
    convert from the default bgr cv2 format to gray, using a single component

    :param conversion:
    :type conversion:
    :param image:
    :type image:
    :return:
    :rtype:
    """

    if is_singular_channel(image):
        conversion = ColorConversionEnum(conversion)
        image = cv2.cvtColor(image, conversion)
    else:
        warnings.warn(f"expected singular image was shape {image.shape}")
    return image
