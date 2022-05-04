import cv2
import numpy
from typing import Tuple, Union, Sequence

__all__ = ["circle_crop"]


def circle_crop(image: numpy.ndarray, center: Sequence, radius: int) -> numpy.ndarray:
    mask = numpy.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    cropped = masked[
        center[1] - radius : center[1] + radius + 1,
        center[0] - radius : center[0] + radius + 1,
    ]

    return cropped
