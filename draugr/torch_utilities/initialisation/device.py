#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 15/11/2019
           """

device = None

__all__ = ["global_torch_device"]


def global_torch_device(prefer_cuda: bool = True):
    global device
    if not device:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and prefer_cuda else "cpu"
        )
    return device
