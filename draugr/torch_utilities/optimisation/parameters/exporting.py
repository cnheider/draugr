#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 9/20/22
           """

__all__ = ["named_parameters_numpy"]

from typing import Dict

import torch
from torch.nn import Module


def named_parameters_numpy(model: Module) -> Dict[str, torch.nn.Parameter]:
    """

    :param model:
    :type model:
    :return:
    :rtype:"""
    params_to_export = {}
    for name, param in model.named_parameters():
        params_to_export[name] = param.detach().cpu().numpy()
    return params_to_export
