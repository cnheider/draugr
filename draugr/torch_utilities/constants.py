#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
from draugr.torch_utilities.to_tensor import to_tensor

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """


def torch_pi(device="cpu"):
    return to_tensor([numpy.math.pi], device=device)
