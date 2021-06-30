#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

          A few low dimensional orders common in raster-grids

           Created on 28-03-2021
           """

import torch

__all__ = [
    "hwc_to_chw_tensor",
    "chw_to_hwc_tensor",
    "nhwc_to_nchw_tensor",
    "nchw_to_nhwc_tensor",
    "nthwc_to_ntchw_tensor",
    "ntchw_to_nthwc_tensor",
]


def hwc_to_chw_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """

    :param tensor:
    :return:
    :rtype:"""
    assert len(tensor.shape) == 3
    return tensor.permute(2, 0, 1)


def chw_to_hwc_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """

    :param tensor:
    :return:
    :rtype:"""
    assert len(tensor.shape) == 3
    return tensor.permute(1, 2, 0)


def nhwc_to_nchw_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """

    :param tensor:
    :return:
    :rtype:"""
    assert len(tensor.shape) == 4
    return tensor.permute(0, 3, 1, 2)


def nchw_to_nhwc_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """

    :param tensor:
    :return:
    :rtype:"""
    assert len(tensor.shape) == 4
    return tensor.permute(0, 2, 3, 1)


def nthwc_to_ntchw_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """

    :param tensor:
    :return:
    :rtype:"""
    assert len(tensor.shape) == 5
    return tensor.permute(0, 1, 4, 2, 3)


def ntchw_to_nthwc_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """

    :param tensor:
    :return:
    :rtype:"""
    assert len(tensor.shape) == 5
    return tensor.permute(0, 1, 3, 4, 2)
