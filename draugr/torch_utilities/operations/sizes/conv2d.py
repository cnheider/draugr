#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/07/2020
           """

import math
from typing import Tuple, Union

from draugr import replicate

__all__ = ["conv2d_padding_size", "conv2d_hw_shape"]


def conv2d_hw_shape(
    h_w: Union[int, Tuple[int, int]],
    kernel_size: Union[int, Tuple[int, int]] = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    pad: Union[int, Tuple[int, int], Tuple[int, int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
) -> Tuple[int, int]:
    """

    :param h_w:
    :type h_w:
    :param kernel_size:
    :type kernel_size:
    :param stride:
    :type stride:
    :param pad:
    :type pad:
    :param dilation:
    :type dilation:
    :return:
    :rtype:"""
    (h_w, kernel_size, stride, pad, dilation) = (
        replicate(h_w),
        replicate(kernel_size),
        replicate(stride),
        replicate(pad),
        replicate(dilation),
    )
    pad = (replicate(pad[0]), replicate(pad[1]))

    h = math.floor(
        (h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    )
    w = math.floor(
        (h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    )

    return h, w


def conv2d_padding_size(
    h_w_in: Union[int, Tuple[int, int]],
    h_w_out: Union[int, Tuple[int, int]],
    kernel_size: int = 1,
    stride: int = 1,
    dilation: int = 1,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """

    :param h_w_in:
    :type h_w_in:
    :param h_w_out:
    :type h_w_out:
    :param kernel_size:
    :type kernel_size:
    :param stride:
    :type stride:
    :param dilation:
    :type dilation:
    :return:
    :rtype:"""
    (h_w_in, h_w_out, kernel_size, stride, dilation) = (
        replicate(h_w_in),
        replicate(h_w_out),
        replicate(kernel_size),
        replicate(stride),
        replicate(dilation),
    )

    p_h = (
        (h_w_out[0] - 1) * stride[0]
        - h_w_in[0]
        + dilation[0] * (kernel_size[0] - 1)
        + 1
    )
    p_w = (
        (h_w_out[1] - 1) * stride[1]
        - h_w_in[1]
        + dilation[1] * (kernel_size[1] - 1)
        + 1
    )

    return (
        (math.floor(p_h / 2), math.ceil(p_h / 2)),
        (math.floor(p_w / 2), math.ceil(p_w / 2)),
    )


if __name__ == "__main__":
    print(conv2d_hw_shape(105, (0, 1, 2, 3)))
