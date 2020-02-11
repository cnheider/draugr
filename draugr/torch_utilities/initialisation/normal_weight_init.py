#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torch import nn

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/10/2019
           """

__all__ = ["init_weights"]


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.1)
        nn.init.constant_(m.bias, 0.1)
