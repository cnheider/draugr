#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/02/2020
           """

from typing import Union

import numpy
import torch

__all__ = ["standardise"]

from warg import Number


def minus_one_one_unnormalise(
    x: Union[Number, torch.Tensor, numpy.ndarray], low: float = 0, high: float = 1
) -> torch.tensor:
    """ """
    act_k = (high - low) / 2.0
    act_b = (high + low) / 2.0
    return act_k * x + act_b


def minus_one_one_normalise(
    x: Union[Number, torch.Tensor, numpy.ndarray], low: float = 0, high: float = 1
) -> torch.tensor:
    """ """
    act_k_inv = 2.0 / (high - low)
    act_b = (high + low) / 2.0
    return act_k_inv * (x - act_b)


def standardise(x: torch.Tensor, eps: float = 1e-6) -> torch.tensor:
    """

    :param eps:
    :param x:
    :return:"""
    x -= x.mean()
    x /= x.std() + eps
    return x


if __name__ == "__main__":
    print(standardise(torch.ones(10)))
    print(standardise(torch.ones((10, 1))))
    print(standardise(torch.ones((1, 10))))

    print(standardise(torch.diag(torch.ones(3))))

    print(standardise(torch.ones((1, 10)) * torch.rand((1, 10))))

    print(standardise(torch.rand((1, 10))))

    print(minus_one_one_normalise(7, 0, 10))
    print(minus_one_one_unnormalise(0.4, 0, 10))
    print(minus_one_one_normalise(minus_one_one_unnormalise(3.4, 3, 4), 3, 4))
