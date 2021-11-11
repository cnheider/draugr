#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

import numpy
import torch

from draugr.torch_utilities.tensors import to_tensor

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
__all__ = ["torch_pi"]


def torch_pi(device: Union[str, torch.device] = "cpu") -> torch.tensor:
    """
    Returns numpy.pi as a tensor

    :param device:
    :type device:
    :return:
    :rtype:"""
    return to_tensor([numpy.math.pi], device=device)
