#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/10/2019
           """

__all__ = ["normal_init_weights"]


def normal_init_weights(m, mean: float = 0.0, std: float = 0.1) -> None:
    """

      :param mean:
      :param std:
    :param m:
    :type m:"""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=mean, std=std)
        nn.init.constant_(m.bias, std)
