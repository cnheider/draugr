import cv2

from draugr.opencv_utilities.windows.image import show_image
from draugr.opencv_utilities.namespaces.enums import MorphShapesEnum, MorphTypesEnum

__all__ = ["clean_up"]


def clean_up(img, method, debug: bool = False):
    if False:
        img = cv2.morphologyEx(
            img,
            MorphTypesEnum.open.value,
            cv2.getStructuringElement(MorphShapesEnum.cross.value, (20, 20)),
            iterations=1,
        )
        if debug:
            show_image(img, "open")

    if False:
        filter_size = (
            max(int(img.shape[1] * 0.001), 5),
            max(int(img.shape[0] * 0.001), 5),
        )
        img = cv2.dilate(
            img,
            cv2.getStructuringElement(MorphShapesEnum.ellipse.value, filter_size),
            iterations=2,
        )
        if debug:
            show_image(img, "dilate")

    if False:
        filter_size = (
            max(int(img.shape[1] * 0.001), 5),
            max(int(img.shape[0] * 0.001), 5),
        )
        img = cv2.morphologyEx(
            img,
            MorphTypesEnum.close.value,
            cv2.getStructuringElement(MorphShapesEnum.ellipse.value, filter_size),
            iterations=3,
        )
        if debug:
            show_image(img, "close")

    return img
