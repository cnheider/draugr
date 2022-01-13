#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 15/04/2020
           """

__all__ = [
    "rgb_drop_alpha_hwc",
    "rgb_drop_alpha_batch_nhwc",
    "torch_vision_normalize_batch_nchw",
    "reverse_torch_vision_normalize_batch_nchw",
]

# from numba import jit

from warg.typing_extension import StrictNumbers


# @jit(nopython=True, fastmath=True)
def rgb_drop_alpha_hwc(inp: StrictNumbers) -> StrictNumbers:
    """

    :param inp:
    :type inp:
    :return:
    :rtype:"""
    assert len(inp[-1, -1]) >= 3, f"not enough channels, only had {len(inp[-1, -1])}"
    return inp[..., :3]


# @jit(nopython=True, fastmath=True)
def rgb_drop_alpha_batch_nhwc(inp: StrictNumbers) -> StrictNumbers:
    """

    :param inp:
    :type inp:
    :return:
    :rtype:"""
    assert (
        len(inp[-1, -1, -1]) >= 3
    ), f"not enough channels, only had {len(inp[-1, -1, -1])}"
    return inp[..., :3]


# @jit(nopython=True, fastmath=True)
def torch_vision_normalize_batch_nchw(inp: StrictNumbers) -> StrictNumbers:
    """

      WARNING INPLACE!

    :param inp:
    :type inp:
    :return:
    :rtype:"""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    assert len(inp[-1]) == 3, f"was {len(inp[-1])}"

    inp[:, 0] = (inp[:, 0] - mean[0]) / std[0]
    inp[:, 1] = (inp[:, 1] - mean[1]) / std[1]
    inp[:, 2] = (inp[:, 2] - mean[2]) / std[2]

    return inp


# @jit(nopython=True, fastmath=True)
def reverse_torch_vision_normalize_batch_nchw(inp: StrictNumbers) -> StrictNumbers:
    """

    :param inp:
    :type inp:
    :return:
    :rtype:"""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    assert len(inp[-1]) == 3, f"was {len(inp[-1])}"

    inp[:, 0] = inp[:, 0] * std[0] + mean[0]
    inp[:, 1] = inp[:, 1] * std[1] + mean[1]
    inp[:, 2] = inp[:, 2] * std[2] + mean[2]

    return inp


if __name__ == "__main__":
    import numpy

    def asda() -> None:
        """
        :rtype: None
        """
        a = numpy.ones((1, 4, 4, 4))
        b = numpy.ones((1, 4, 4, 3))
        c = numpy.ones((4, 4, 3))
        d = numpy.ones((1, 4, 4, 2))

        rgb_drop_alpha_batch_nhwc(a)
        rgb_drop_alpha_batch_nhwc(b)
        try:
            rgb_drop_alpha_batch_nhwc(c)
        except:
            pass
        rgb_drop_alpha_hwc(c)
        try:
            rgb_drop_alpha_batch_nhwc(d)
        except:
            pass
        try:
            rgb_drop_alpha_hwc(d)
        except:
            pass

    def asbsdfdsa() -> None:
        """
        :rtype: None
        """
        a = numpy.ones((1, 3, 4, 4))
        ba = torch_vision_normalize_batch_nchw(a)
        print(ba)
        ca = reverse_torch_vision_normalize_batch_nchw(ba)
        print(ca)

    asda()
    asbsdfdsa()
