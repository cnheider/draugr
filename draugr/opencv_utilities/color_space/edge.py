from enum import Enum

import cv2
from sorcery import assigned_names

from draugr.opencv_utilities.namespaces.enums import MorphShapeEnum, MorphTypeEnum
from warg import next_odd

__all__ = ["to_edge", "ToEdgeMethodEnum", "CannyApertureSize"]


class ToEdgeMethodEnum(Enum):
    canny, laplacian, sobel_vh, sobel_h, sobel_v, morph_gradient = assigned_names()


class CannyApertureSize(Enum):
    a3, a5, a7 = 3, 5, 7


def to_edge(img, method: ToEdgeMethodEnum = ToEdgeMethodEnum.canny, **kwargs):
    method = ToEdgeMethodEnum(method)
    if method == ToEdgeMethodEnum.canny:
        return cv2.Canny(
            img,
            threshold1=kwargs.get("threshold1", 60),
            threshold2=kwargs.get("threshold2", 180),
            apertureSize=kwargs.get("apertureSize", CannyApertureSize.a3).value,
            L2gradient=kwargs.get("L2gradient", None),
        )

    ksize = kwargs.get("ksize", max(next_odd(max(*(img.shape)) // 100), 5))
    if method == ToEdgeMethodEnum.morph_gradient:

        return cv2.morphologyEx(
            img,
            MorphTypeEnum.gradient.value,
            cv2.getStructuringElement(MorphShapeEnum.rect.value, ksize=(ksize, ksize)),
        )
    elif method == ToEdgeMethodEnum.laplacian:
        return cv2.Laplacian(
            img, cv2.CV_8UC1, ksize=ksize
        )  # ,cv2.CV_16UC1, #cv2.CV_16S, # cv2.CV_64F
    elif method == ToEdgeMethodEnum.sobel_h:
        return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    elif method == ToEdgeMethodEnum.sobel_v:
        return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    elif method == ToEdgeMethodEnum.sobel_vh:
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        return sobelx + sobely

    raise NotImplementedError


if __name__ == "__main__":

    def aushd():
        import numpy
        from draugr.opencv_utilities import show_image

        a = numpy.zeros((50, 50))
        a[:, 25] = 1
        a[25, :] = 1

        show_image(to_edge(a, ToEdgeMethodEnum.sobel_vh), wait=True)

    aushd()
