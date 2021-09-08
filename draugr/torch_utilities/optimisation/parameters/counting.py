#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27/06/2020
           """

__all__ = ["get_num_parameters"]

from torch import nn

from draugr.torch_utilities.optimisation.parameters.trainable import (
    trainable_parameters,
)


def get_num_parameters(model: nn.Module, *, only_trainable: bool = False) -> int:
    """
    Returns the number of parameters of a model (trainable or all)

    :param only_trainable:
    :type only_trainable:
    :param model:
    :type model:
    :return:
    :rtype:"""
    if only_trainable:
        return sum(p.numel() for p in trainable_parameters(model))
    return sum(param.numel() for param in model.parameters())
