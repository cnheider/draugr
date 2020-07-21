#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17/07/2020
           """

from typing import Tuple

import numpy

from draugr.visualisation.pillow_utilities import np_array_to_pil_img

__all__ = ["resize_array"]


def resize_array(x: numpy.ndarray, size: Tuple[int, int]) -> numpy.ndarray:
    assert x.ndim in [3, 4], "Only 3D and 4D Tensors allowed!"

    if x.ndim == 4:  # 4D Tensor
        res = []
        for i in range(x.shape[0]):
            img = np_array_to_pil_img(x[i])
            img = img.resize((size, size))
            img = numpy.asarray(img, dtype="float32")
            img = numpy.expand_dims(img, axis=0)
            img /= 255.0
            res.append(img)
        res = numpy.concatenate(res)
        res = numpy.expand_dims(res, axis=1)
        return res
    else:  # 3D Tensor

        img = np_array_to_pil_img(x)
        img = img.resize((size, size))
        res = numpy.asarray(img, dtype="float32")
        res = numpy.expand_dims(res, axis=0)
        res /= 255.0
        return res
