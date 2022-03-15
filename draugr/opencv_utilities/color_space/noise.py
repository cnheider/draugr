import cv2
from enum import Enum
from sorcery import assigned_names
from typing import Any


__all__ = ["noise_filter", "NoiseFilterMethodEnum"]


class NoiseFilterMethodEnum(Enum):
    none, median_blur, bilateral_filter, gaussian_blur = assigned_names()


def noise_filter(
    img: Any, method: NoiseFilterMethodEnum = NoiseFilterMethodEnum.bilateral_filter
) -> Any:
    method = NoiseFilterMethodEnum(method)
    if method == NoiseFilterMethodEnum.none:
        return img
    elif method == NoiseFilterMethodEnum.median_blur:
        return cv2.medianBlur(img, 3)
    elif method == NoiseFilterMethodEnum.bilateral_filter:
        return cv2.bilateralFilter(img, 11, 2, 220)
    elif method == NoiseFilterMethodEnum.gaussian_blur:
        return cv2.GaussianBlur(img, (5, 5), 1.4)

    raise NoiseFilterMethodEnum
