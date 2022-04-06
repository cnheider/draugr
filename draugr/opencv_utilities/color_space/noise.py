from enum import Enum
from typing import Any

import cv2
from sorcery import assigned_names

from warg import ceil_odd

__all__ = ["noise_filter", "NoiseFilterMethodEnum"]


class NoiseFilterMethodEnum(Enum):
    none, median_blur, bilateral_filter, gaussian_blur = assigned_names()


def noise_filter(
    img: Any,
    method: NoiseFilterMethodEnum = NoiseFilterMethodEnum.bilateral_filter,
    **kwargs
) -> Any:
    method = NoiseFilterMethodEnum(method)

    if method == NoiseFilterMethodEnum.none:
        return img

    ksize = kwargs.get("ksize", max(ceil_odd(max(*(img.shape)) // 100), 5))
    if method == NoiseFilterMethodEnum.median_blur:
        return cv2.medianBlur(img, ksize=ksize)
    elif method == NoiseFilterMethodEnum.bilateral_filter:
        return cv2.bilateralFilter(
            img,
            d=kwargs.get("d", 11),
            sigmaColor=kwargs.get("sigmaColor", 17),
            sigmaSpace=kwargs.get("sigmaSpace", 17),
        )
    elif method == NoiseFilterMethodEnum.gaussian_blur:

        return cv2.GaussianBlur(
            img,
            ksize=(ksize, ksize),
            sigmaX=kwargs.get("sigmaX", 5),
            borderType=kwargs.get("borderType", None),
        )

    raise NoiseFilterMethodEnum
