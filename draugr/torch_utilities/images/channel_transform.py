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
    "hwc_to_chw_tensor",
    "chw_to_hwc_tensor",
    "float_chw_to_hwc_uint_tensor",
    "uint_hwc_to_chw_float_tensor",
]


def hwc_to_chw_tensor(inp: torch.Tensor) -> torch.Tensor:
    """

  :param inp:
  :type inp:
  :return:
  :rtype:
  """
    assert len(inp.shape) == 3
    return inp.permute(2, 0, 1)


def chw_to_hwc_tensor(inp: torch.Tensor) -> torch.Tensor:
    """

  :param inp:
  :type inp:
  :return:
  :rtype:
  """
    assert len(inp.shape) == 3
    return inp.permute(1, 2, 0)


def uint_hwc_to_chw_float_tensor(
    inp: torch.Tensor, *, normalise: bool = True
) -> torch.Tensor:
    """

  :param inp:
  :type inp:
  :param normalise:
  :type normalise:
  :return:
  :rtype:
  """
    if normalise:
        inp = (inp / 255.0).clamp(0, 1)
    return hwc_to_chw_tensor(inp)


def float_chw_to_hwc_uint_tensor(
    inp: torch.Tensor, *, unnormalise: bool = True
) -> torch.Tensor:
    """

  :param inp:
  :type inp:
  :param unnormalise:
  :type unnormalise:
  :return:
  :rtype:
  """
    inp = chw_to_hwc_tensor(inp)
    if unnormalise:
        inp = (inp * 255.0).clamp(0, 255)
    return inp.to(dtype=torch.uint8)


def uint_nhwc_to_nchw_float_batch(
    inp: torch.Tensor, *, normalise: bool = True
) -> torch.Tensor:
    """

  :param inp:
  :type inp:
  :param normalise:
  :type normalise:
  :return:
  :rtype:
  """
    assert len(inp.shape) == 4
    if normalise:
        inp = (inp / 255.0).clamp(0, 1)
    return inp.permute(0, 3, 1, 2)


def float_nchw_to_nhwc_uint_batch(
    inp: torch.Tensor, *, unnormalise: bool = True
) -> torch.Tensor:
    """

  :param inp:
  :type inp:
  :param unnormalise:
  :type unnormalise:
  :return:
  :rtype:
  """
    assert len(inp.shape) == 4
    inp = inp.permute(0, 3, 1, 2)
    if unnormalise:
        inp = (inp * 255.0).clamp(0, 255)
    return inp.to(dtype=torch.uint8)


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
