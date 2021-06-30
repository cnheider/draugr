#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/07/2020
           """

import math
from typing import Tuple, Union

__all__ = ["transp_conv2d_output_shape", "transp_conv2d_padding_sizes"]

from draugr import replicate


def transp_conv2d_output_shape(
    h_w: Union[int, Tuple[int, int]],
    kernel_size: Union[int, Tuple[int, int]] = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    pad: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    out_pad: Union[int, Tuple[int, int]] = 0,
) -> Tuple[int, int]:
    """ """
    (h_w, kernel_size, stride, pad, dilation, out_pad) = (
        replicate(h_w),
        replicate(kernel_size),
        replicate(stride),
        replicate(pad),
        replicate(dilation),
        replicate(out_pad),
    )
    pad = (replicate(pad[0]), replicate(pad[1]))

    h = (
        (h_w[0] - 1) * stride[0]
        - sum(pad[0])
        + dilation[0] * (kernel_size[0] - 1)
        + out_pad[0]
        + 1
    )
    w = (
        (h_w[1] - 1) * stride[1]
        - sum(pad[1])
        + dilation[1] * (kernel_size[1] - 1)
        + out_pad[1]
        + 1
    )

    return h, w


def transp_conv2d_padding_sizes(
    h_w_in: Union[int, Tuple[int, int]],
    h_w_out: Union[int, Tuple[int, int]],
    kernel_size: Union[int, Tuple[int, int]] = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    out_pad: Union[int, Tuple[int, int]] = 0,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """ """
    (h_w_in, h_w_out, kernel_size, stride, dilation, out_pad) = (
        replicate(h_w_in),
        replicate(h_w_out),
        replicate(kernel_size),
        replicate(stride),
        replicate(dilation),
        replicate(out_pad),
    )

    p_h = (
        -(
            h_w_out[0]
            - 1
            - out_pad[0]
            - dilation[0] * (kernel_size[0] - 1)
            - (h_w_in[0] - 1) * stride[0]
        )
        / 2
    )
    p_w = (
        -(
            h_w_out[1]
            - 1
            - out_pad[1]
            - dilation[1] * (kernel_size[1] - 1)
            - (h_w_in[1] - 1) * stride[1]
        )
        / 2
    )

    return (
        (math.floor(p_h / 2), math.ceil(p_h / 2)),
        (math.floor(p_w / 2), math.ceil(p_w / 2)),
    )


if __name__ == "__main__":
    print(transp_conv2d_output_shape(105, 10))
