#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sized

import numpy
import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """

__all__ = [
    "hwc_to_chw",
    "chw_to_hwc",
    "uint_hwc_to_chw_float",
    "float_chw_to_hwc_uint",
    "rgb_drop_alpha",
    "uint_hwc_to_chw_float_batch",
    "float_chw_to_hwc_uint_batch",
    "rgb_drop_alpha_batch",
    "torch_vision_normalize_chw",
    "reverse_torch_vision_normalize_chw",
]


def hwc_to_chw(inp: numpy.ndarray) -> numpy.ndarray:
    return inp.transpose((2, 0, 1))


def chw_to_hwc(inp: numpy.ndarray) -> numpy.ndarray:
    return inp.transpose((1, 2, 0))


def uint_hwc_to_chw_float(
    inp: numpy.ndarray, *, normalise: bool = True
) -> numpy.ndarray:
    if normalise:
        inp = inp / 255.0
        inp = numpy.clip(inp, 0, 1)
    return hwc_to_chw(inp)


def float_chw_to_hwc_uint(
    inp: numpy.ndarray, *, unnormalise: bool = True
) -> numpy.ndarray:
    inp = chw_to_hwc(inp)
    if unnormalise:
        inp = inp * 255.0
        inp = numpy.clip(inp, 0, 255)
    return inp.astype(numpy.uint8)


def rgb_drop_alpha(inp: torch.Tensor) -> torch.Tensor:
    return inp[:, :, :3]


def uint_hwc_to_chw_float_batch(
    inp: torch.Tensor, *, normalise: bool = True
) -> torch.Tensor:
    if normalise:
        inp = inp / 255.0
        inp = inp.clamp(0, 1)
    inp = inp.transpose(-1, 1)
    return inp


def float_chw_to_hwc_uint_batch(
    inp: torch.Tensor, *, unnormalise: bool = True
) -> torch.Tensor:
    inp = inp.transpose(-1, 1)
    if unnormalise:
        inp = inp * 255.0
        inp = inp.clamp(0, 255)
    return inp.to(dtype=torch.uint8)


def rgb_drop_alpha_batch(inp: torch.Tensor) -> torch.Tensor:
    return inp[:, :, :, :3]


def torch_vision_normalize_chw(inp: Sized) -> Sized:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    inp[:, 0] = (inp[:, 0] - mean[0]) / std[0]
    inp[:, 1] = (inp[:, 1] - mean[1]) / std[1]
    inp[:, 2] = (inp[:, 2] - mean[2]) / std[2]

    return inp


def reverse_torch_vision_normalize_chw(inp: Sized) -> Sized:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    inp[:, 0] = inp[:, 0] * std[0] + mean[0]
    inp[:, 1] = inp[:, 1] * std[1] + mean[1]
    inp[:, 2] = inp[:, 2] * std[2] + mean[2]

    return inp
