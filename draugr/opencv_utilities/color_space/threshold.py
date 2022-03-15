import cv2
from enum import Enum
from sorcery import assigned_names


__all__ = ["threshold_gray", "ThresholdMethodEnum"]

from draugr.opencv_utilities.namespaces.flags import ThresholdTypeFlag


class ThresholdMethodEnum(Enum):
    simple, adaptive = assigned_names()


def threshold_gray(gray, method: ThresholdMethodEnum = ThresholdMethodEnum.simple):
    method = ThresholdMethodEnum(method)

    if method == ThresholdMethodEnum.simple:
        return cv2.threshold(
            gray,
            120,
            255,
            ThresholdTypeFlag.otsu.value + ThresholdTypeFlag.to_zero.value,
        )[1]
    elif method == ThresholdMethodEnum.adaptive:
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    raise NotImplementedError
