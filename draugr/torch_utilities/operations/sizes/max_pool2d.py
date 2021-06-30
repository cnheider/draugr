#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/07/2020
           """

import math
from typing import Tuple, Union

from draugr import replicate

__all__ = ["max_pool2d_hw_shape"]


def max_pool2d_hw_shape(
    h_w: Union[int, Tuple[int, int]],
    pool_size: Union[int, Tuple[int, int]] = 2,
    stride: Union[int, Tuple[int, int]] = None,
    pad: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
) -> Tuple[int, int]:
    r"""

    .. math::
    H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]}
          \times (\text{kernel\_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

    .. math::
    W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]}
          \times (\text{kernel\_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

    :param pool_size:
    :type pool_size:
    :param h_w:
    :type h_w:
    :param stride:
    :type stride:
    :param pad:
    :type pad:
    :param dilation:
    :type dilation:
    :return:
    :rtype:"""

    if stride is None:
        stride = pool_size
    (h_w, pool_size, stride, pad, dilation) = (
        replicate(h_w),
        replicate(pool_size),
        replicate(stride),
        replicate(pad),
        replicate(dilation),
    )
    pad = (replicate(pad[0]), replicate(pad[1]))

    h = math.floor(
        (h_w[0] + sum(pad[0]) - dilation[0] * (pool_size[0] - 1) - 1) / stride[0] + 1
    )
    w = math.floor(
        (h_w[1] + sum(pad[1]) - dilation[1] * (pool_size[1] - 1) - 1) / stride[1] + 1
    )

    return h, w


if __name__ == "__main__":
    print(max_pool2d_hw_shape(12, 2))
