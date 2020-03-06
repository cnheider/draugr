#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 13/11/2019
           """

from typing import List

import torch
from torch import nn
from torch.nn import Module


def set_all_parameter_requires_grad(model: nn.Module, bo: bool = False) -> None:
    for param in model.parameters():
        param.requires_grad = bo


def set_first_n_parameter_requires_grad(
    model: nn.Module, n: int = 6, bo: bool = False
) -> None:
    for i, child in enumerate(model.children()):
        if i <= n:
            set_all_parameter_requires_grad(child, bo)


def get_trainable_parameters(model: Module) -> List[torch.nn.Parameter]:
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    return params_to_update
