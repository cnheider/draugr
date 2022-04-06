from enum import Enum

import cv2
from sorcery import assigned_names

from draugr.opencv_utilities.namespaces.enums import (
    MorphShapeEnum,
    MorphTypeEnum,
    BorderTypeEnum,
)

__all__ = ["clean_up", "CleanUpMethod"]


class CleanUpMethod(Enum):
    close, open, nclose, nopen, erode, dilate, none = assigned_names()


def clean_up(img, method: CleanUpMethod = CleanUpMethod.open, **kwargs):
    filter_size = (
        max(int(img.shape[1] * 0.01), kwargs.get("min_filter_size", 3)),
        max(int(img.shape[0] * 0.01), kwargs.get("min_filter_size", 3)),
    )
    morph_shape = MorphShapeEnum.rect.value
    kernel = cv2.getStructuringElement(morph_shape, filter_size)

    if method == CleanUpMethod.close:
        img = cv2.morphologyEx(
            img,
            MorphTypeEnum.close.value,
            kernel,
            iterations=kwargs.get("iterations", 1),
            borderType=kwargs.get("borderType", None),  # BorderTypeEnum.constant.value,
            borderValue=kwargs.get("borderValue", 0),  # 255
        )
    elif method == CleanUpMethod.open:
        img = cv2.morphologyEx(
            img,
            MorphTypeEnum.open.value,
            kernel,
            iterations=kwargs.get("iterations", 1),
            borderType=kwargs.get("borderType", None),  # BorderTypeEnum.constant.value,
            borderValue=kwargs.get("borderValue", 0),  # 255
        )
    elif method == CleanUpMethod.nclose:
        img = cv2.morphologyEx(
            img,
            MorphTypeEnum.dilate.value,
            kernel,
            None,
            None,
            iterations=kwargs.get("iterations", 2),
            borderType=kwargs.get("borderType", BorderTypeEnum.reflect101.value),
        )
        img = cv2.morphologyEx(
            img,
            MorphTypeEnum.erode.value,
            kernel,
            None,
            None,
            iterations=kwargs.get("iterations", 2),
            borderType=kwargs.get("borderType", BorderTypeEnum.reflect101.value),
        )
    elif method == CleanUpMethod.erode:
        img = cv2.dilate(
            img,
            kernel,
            iterations=kwargs.get("iterations", 1),
        )
    elif method == CleanUpMethod.dilate:
        img = cv2.erode(
            img,
            kernel,
            iterations=kwargs.get("iterations", 1),
        )

    return img
