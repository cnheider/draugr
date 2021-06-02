#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """

__all__ = [
    "uint_nhwc_to_nchw_float_batch",
    "float_nchw_to_nhwc_uint_batch",
    "float_chw_to_hwc_uint_tensor",
    "uint_hwc_to_chw_float_tensor",
]

from draugr.torch_utilities.tensors.dimension_order import (
    chw_to_hwc_tensor,
    hwc_to_chw_tensor,
    nchw_to_nhwc_tensor,
    nhwc_to_nchw_tensor,
)


def uint_hwc_to_chw_float_tensor(
    tensor: torch.Tensor, *, normalise: bool = True
) -> torch.Tensor:
    """

    :param tensor:
    :type tensor:
    :param normalise:
    :type normalise:
    :return:
    :rtype:"""
    if normalise:
        tensor = (tensor / 255.0).clamp(0, 1)
    return hwc_to_chw_tensor(tensor)


def float_chw_to_hwc_uint_tensor(
    tensor: torch.Tensor, *, unnormalise: bool = True
) -> torch.Tensor:
    """

    :param tensor:
    :type tensor:
    :param unnormalise:
    :type unnormalise:
    :return:
    :rtype:"""
    tensor = chw_to_hwc_tensor(tensor)
    if unnormalise:
        tensor = (tensor * 255.0).clamp(0, 255)
    return tensor.to(dtype=torch.uint8)


def uint_nhwc_to_nchw_float_batch(
    tensor: torch.Tensor, *, normalise: bool = True
) -> torch.Tensor:
    """

    :param tensor:
    :type tensor:
    :param normalise:
    :type normalise:
    :return:
    :rtype:"""
    if normalise:
        tensor = (tensor / 255.0).clamp(0, 1)
    return nhwc_to_nchw_tensor(tensor)


def float_nchw_to_nhwc_uint_batch(
    tensor: torch.Tensor, *, unnormalise: bool = True
) -> torch.Tensor:
    """

    :param tensor:
    :type tensor:
    :param unnormalise:
    :type unnormalise:
    :return:
    :rtype:"""
    tensor = nchw_to_nhwc_tensor(tensor)
    if unnormalise:
        tensor = (tensor * 255.0).clamp(0, 255)
    return tensor.to(dtype=torch.uint8)


if __name__ == "__main__":
    hw = 2
    a = torch.ones(3, hw, hw)
    print(a)
    b = float_chw_to_hwc_uint_tensor(a)
    print(b)
    c = uint_hwc_to_chw_float_tensor(b)
    print(c)
    d = chw_to_hwc_tensor(c)
    assert (
        d.shape == c.T.shape
    )  # only work h and w is same size, mind that semantically not the same as transpose will be (whc)
    print(c.shape)
    print(d.shape)
