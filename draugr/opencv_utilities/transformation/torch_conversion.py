import types
from typing import Any, List, Tuple

import cv2
import numpy
import torch
from PIL.ImageTransform import Transform
from numpy import random

__all__ = [
    "CV2ToImage",
    "CV2ToTensor",
]


class CV2ToImage(object):
    def __call__(
        self,
        tensor: torch.Tensor,
        boxes: numpy.ndarray = None,
        labels: numpy.ndarray = None,
    ) -> Tuple:
        return (
            tensor.cpu().numpy().astype(numpy.float32).transpose((1, 2, 0)),
            boxes,
            labels,
        )


class CV2ToTensor(object):
    def __call__(
        self,
        cvimage: numpy.ndarray,
        boxes: numpy.ndarray = None,
        labels: numpy.ndarray = None,
    ) -> Tuple:
        return (
            torch.from_numpy(cvimage.astype(numpy.float32)).permute(2, 0, 1),
            boxes,
            labels,
        )
