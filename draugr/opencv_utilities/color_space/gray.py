from enum import Enum

import cv2
from sorcery import assigned_names

from draugr.opencv_utilities.color_space.color import is_singular_channel
from draugr.opencv_utilities.namespaces.color_conversion_enum import (
    ColorConversionEnum,
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
    if not is_singular_channel(image):
        to_gray_method = ToGrayMethodEnum(to_gray_method)
        if to_gray_method == to_gray_method.gray:
            # Weighted sum of the three channels
            # 0.299*R + 0.587*G + 0.114*B
            components = (cv2.cvtColor(image, ColorConversionEnum.bgr2gray.value),)
        elif to_gray_method == ToGrayMethodEnum.rgb:  # Red Green Blue
            # Pick a single channel
            components = cv2.split(
                cv2.cvtColor(image, ColorConversionEnum.bgr2rgb.value)
            )
        elif (
            to_gray_method == ToGrayMethodEnum.hsv
        ):  # Hue ( Dominant Wavelength ) S – Saturation ( Purity / shades of the color ) V – Value ( Intensity )
            # The H Component is very similar in both the images which indicates the color information is intact even under illumination changes.
            # The S component is also very similar in both images.
            # The V Component captures the amount of light falling on it thus it changes due to illumination changes.
            # There is drastic difference between the values of the red piece of outdoor and Indoor image. This is because Hue is represented as a circle and red is at the starting angle. So, it may take values between [300, 360] and again [0, 60].
            components = cv2.split(
                cv2.cvtColor(image, ColorConversionEnum.bgr2hsv.value)
            )
        elif (
            to_gray_method == ToGrayMethodEnum.ycrcb
        ):  # Y – Luma ( Luminance ) Cb – Chroma Blue ( Color ) Cr – Chroma Red ( Color )
            # Y – Luminance or Luma component obtained from RGB after gamma correction.
            # Cr = R – Y ( how far is the red component from Luma ).
            # Cb = B – Y ( how far is the blue component from Luma ).
            # Similar observations as LAB can be made for Intensity and color components with regard to Illumination changes.
            # Perceptual difference between Red and Orange is less even in the outdoor image as compared to LAB.
            # White has undergone change in all 3 components.
            components = cv2.split(
                cv2.cvtColor(image, ColorConversionEnum.bgr2ycrcb.value)
            )
        elif to_gray_method == ToGrayMethodEnum.yuv:
            # Y, a measure of overall brightness or luminance. U and V are computed as scaled differences between Y′ and the B and R values.
            components = cv2.split(
                cv2.cvtColor(image, ColorConversionEnum.bgr2yuv.value)
            )
        elif (
            to_gray_method == to_gray_method.lab
        ):  # L – Lightness ( Brightness ) a – Green-Magenta ( Green - Red ) b – Blue-Yellow ( Blue - Red )
            # Illumination has mostly affected the L component.
            # The A and B components which contain the color information did not undergo massive changes.
            # The respective values of Green, Orange and Red ( which are the extremes of the A Component ) has not changed in the B Component and similarly the respective values of Blue and Yellow ( which are the extremes of the B Component ) has not changed in the A component.
            components = cv2.split(
                cv2.cvtColor(image, ColorConversionEnum.bgr2lab.value)
            )
        elif to_gray_method == to_gray_method.luv:
            # LUV decouple the "color" (chromaticity, the UV part) and "lightness" (luminance, the L part)
            components = cv2.split(
                cv2.cvtColor(image, ColorConversionEnum.bgr2luv.value)
            )
        else:
            raise NotImplementedError
        return components[component]
    return image
