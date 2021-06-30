#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 13/11/2019
           """

from typing import Dict, List

import torch
from torch import nn
from torch.nn import Module

__all__ = [
    "set_all_parameter_requires_grad",
    "set_first_n_parameter_requires_grad",
    "trainable_parameters",
    "named_trainable_parameters",
    "trainable_parameters_iterator",
    "named_trainable_parameters_iterator",
]


def set_all_parameter_requires_grad(model: nn.Module, bo: bool = False) -> None:
    """

    :param model:
    :type model:
    :param bo:
    :type bo:"""
    for param in model.parameters():
        param.requires_grad = bo


def set_first_n_parameter_requires_grad(
    model: nn.Module, n: int = 6, bo: bool = False
) -> None:
    """

    :param model:
    :type model:
    :param n:
    :type n:
    :param bo:
    :type bo:"""
    for i, child in enumerate(model.children()):
        if i <= n:
            set_all_parameter_requires_grad(child, bo)


def trainable_parameters(model: Module) -> List[torch.nn.Parameter]:
    """

    :param model:
    :type model:
    :return:
    :rtype:"""
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    return params_to_update


def named_trainable_parameters(model: Module) -> Dict[str, torch.nn.Parameter]:
    """

    :param model:
    :type model:
    :return:
    :rtype:"""
    params_to_update = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update[name] = param
    return params_to_update


def trainable_parameters_iterator(model: Module) -> List[torch.nn.Parameter]:
    """ """
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield param


def named_trainable_parameters_iterator(model: Module) -> Dict[str, torch.nn.Parameter]:
    """

    :param model:
    :type model:
    :return:
    :rtype:"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield {name: param}
