#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 15/04/2020
           """

from typing import Sequence


__all__ = [
    "rgb_drop_alpha_hwc",
    "rgb_drop_alpha_batch_nhwc",
    "torch_vision_normalize_batch_nchw",
    "reverse_torch_vision_normalize_batch_nchw",
]


def rgb_drop_alpha_hwc(inp: Sequence) -> Sequence:
    """

    :param inp:
    :type inp:
    :return:
    :rtype:
    """
    return inp[..., :3]


def rgb_drop_alpha_batch_nhwc(inp: Sequence) -> Sequence:
    """

    :param inp:
    :type inp:
    :return:
    :rtype:
    """
    return inp[..., :3]


def torch_vision_normalize_batch_nchw(inp: Sequence) -> Sequence:
    """

    :param inp:
    :type inp:
    :return:
    :rtype:
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    inp[:, 0] = (inp[:, 0] - mean[0]) / std[0]
    inp[:, 1] = (inp[:, 1] - mean[1]) / std[1]
    inp[:, 2] = (inp[:, 2] - mean[2]) / std[2]

    return inp


def reverse_torch_vision_normalize_batch_nchw(inp: Sequence) -> Sequence:
    """

    :param inp:
    :type inp:
    :return:
    :rtype:
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    inp[:, 0] = inp[:, 0] * std[0] + mean[0]
    inp[:, 1] = inp[:, 1] * std[1] + mean[1]
    inp[:, 2] = inp[:, 2] * std[2] + mean[2]

    return inp
