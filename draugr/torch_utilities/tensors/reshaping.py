#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/02/2020
           """
__all__ = ["flatten_tn_dim", "flatten_keep_batch", "safe_concat"]

import torch


def flatten_tn_dim(_tensor: torch.Tensor) -> torch.tensor:
    """

    :param _tensor:
    :return:"""
    t, n, *r = _tensor.size()
    return _tensor.reshape(t * n, *r)


def flatten_keep_batch(t: torch.Tensor) -> torch.Tensor:
    """

    :param t:
    :return:"""
    return t.reshape(t.shape[0], -1)


def safe_concat(arr: torch.Tensor, el: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """

    :param arr:
    :param el:
    :param dim:
    :return:"""
    if arr is None:
        return el
    return torch.cat((arr, el), dim=dim)


if __name__ == "__main__":

    def a() -> None:
        """
        :rtype: None
        """
        shape = (2, 3, 4, 5)
        from warg import prod

        t = torch.reshape(torch.arange(0, prod(shape)), shape)

        f = flatten_tn_dim(t)
        tf = t.flatten(0, 1)
        print(t, f, tf)
        print(t.shape, f.shape, tf.shape)

    a()
