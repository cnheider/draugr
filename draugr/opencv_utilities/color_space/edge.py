from enum import Enum

import cv2
from sorcery import assigned_names
from draugr.opencv_utilities.namespaces.enums import MorphShapesEnum

__all__ = ["to_edge", "ToEdgeMethodEnum"]


class ToEdgeMethodEnum(Enum):
    canny, laplacian, sobel_vh, sobel_h, sobel_v, morph_gradient = assigned_names()


def to_edge(img, method: ToEdgeMethodEnum = ToEdgeMethodEnum.canny):
    method = ToEdgeMethodEnum(method)
    if method == ToEdgeMethodEnum.canny:
        return cv2.Canny(img, 40, 180, 5)
    elif method == ToEdgeMethodEnum.morph_gradient:
        return cv2.morphologyEx(
            img,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(MorphShapesEnum.rect.value, (7, 4)),
        )
    elif method == ToEdgeMethodEnum.laplacian:
        return cv2.Laplacian(
            img, cv2.CV_8UC1, ksize=3  # ,cv2.CV_16UC1, #cv2.CV_16S, # cv2.CV_64F
        )
    elif method == ToEdgeMethodEnum.sobel_h:
        return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    elif method == ToEdgeMethodEnum.sobel_v:
        return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    elif method == ToEdgeMethodEnum.sobel_vh:
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
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
