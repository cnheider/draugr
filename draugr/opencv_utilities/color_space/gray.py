from enum import Enum

import cv2
from sorcery import assigned_names

from draugr.opencv_utilities.namespaces.color_conversion_enum import (
    ColorConversionCodesEnum,
)

__all__ = ["ToGrayMethodEnum", "to_gray"]


class ToGrayMethodEnum(Enum):
    gray, rgb, hsv, ycrcb, yuv, lab, luv = assigned_names()


def to_gray(
    image,
    *,
    component: int = 0,
    to_gray_method: ToGrayMethodEnum = ToGrayMethodEnum.gray
):
    """
    convert from the default bgr cv2 format to gray, using a single component

    :param image:
    :type image:
    :param component:
    :type component:
    :param to_gray_method:
    :type to_gray_method:
    :return:
    :rtype:
    """
    to_gray_method = ToGrayMethodEnum(to_gray_method)
    if to_gray_method == to_gray_method.gray:
        components = (cv2.cvtColor(image, ColorConversionCodesEnum.bgr2gray.value),)
    elif to_gray_method == ToGrayMethodEnum.rgb:
        components = cv2.split(
            cv2.cvtColor(image, ColorConversionCodesEnum.bgr2rgb.value)
        )
    elif to_gray_method == ToGrayMethodEnum.hsv:
        components = cv2.split(
            cv2.cvtColor(image, ColorConversionCodesEnum.bgr2hsv.value)
        )
    elif to_gray_method == ToGrayMethodEnum.ycrcb:
        components = cv2.split(
            cv2.cvtColor(image, ColorConversionCodesEnum.bgr2ycrcb.value)
        )
    elif to_gray_method == ToGrayMethodEnum.yuv:
        components = cv2.split(
            cv2.cvtColor(image, ColorConversionCodesEnum.bgr2yuv.value)
        )
    elif to_gray_method == to_gray_method.lab:
        components = cv2.split(
            cv2.cvtColor(image, ColorConversionCodesEnum.bgr2lab.value)
        )
    elif to_gray_method == to_gray_method.luv:
        components = cv2.split(
            cv2.cvtColor(image, ColorConversionCodesEnum.bgr2luv.value)
        )
    else:
        raise NotImplementedError
    return components[component]
