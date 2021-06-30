#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/07/2020
           """

from typing import Sequence, Tuple, Union

from draugr import replicate

__all__ = ["pad2d_hw_shape"]


def pad2d_hw_shape(
    h_w: Union[int, Tuple[int, int]],
    pad_size: Union[int, Tuple[int, int], Tuple[int, int, int, int]] = 1,
) -> Tuple[int, int]:
    """

    quad(left,right,top,bottom) torch quad definition
    double(h,w)
    single(all)

      :param pad_size:
    :param h_w:
    :type h_w:
    :return:
    :rtype:"""
    h_w = replicate(h_w)
    if isinstance(pad_size, Sequence) and len(pad_size) == 4:
        pad = (pad_size[2:], pad_size[:2])  # NOTE: permuted torch quadruple pad order
    else:
        pad_size = replicate(pad_size)
        pad = (replicate(pad_size[0]), replicate(pad_size[1]))

    h = h_w[0] + sum(pad[0])  # Sum across pad[0] (top,bottom)
    w = h_w[1] + sum(pad[1])  # Sum across pad[1] (left,right)

    return h, w


if __name__ == "__main__":
    print(pad2d_hw_shape(5, 1))
    print(pad2d_hw_shape(5, (10, 10)))
    print(pad2d_hw_shape(5, (10, 10, 3, 2)))
