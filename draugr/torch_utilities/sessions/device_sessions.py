#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11/05/2020
           """

import torch
from torch.nn import Module

from draugr.torch_utilities import global_torch_device
from warg.decorators.kw_passing import AlsoDecorator

__all__ = ["TorchCpuSession", "TorchCudaSession"]


class TorchCudaSession(AlsoDecorator):
    """
Sets global torch devices to cuda if available
"""

    def __init__(self, model: Module = None, no_side_effect: bool = True):
        self._model = model
        self._no_side_effect = no_side_effect
        if no_side_effect:
            self.prev_dev = global_torch_device()

    def __enter__(self):
        device = global_torch_device(override=torch.device("cuda"))
        if self._model:
            self._model.to(device)

        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._no_side_effect:
            device = global_torch_device(override=self.prev_dev)
        else:
            device = global_torch_device(override=torch.device("cpu"))
        if self._model:
            self._model.to(device)


class TorchCpuSession(AlsoDecorator):
    """
Sets global torch devices to cpu

"""

    def __init__(self, model: Module = None, no_side_effect: bool = True):
        self._model = model
        self._no_side_effect = no_side_effect
        if no_side_effect:
            self.prev_dev = global_torch_device()

    def __enter__(self):
        device = global_torch_device(override=torch.device("cpu"))
        if self._model:
            self._model.to(device)
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._no_side_effect:
            device = global_torch_device(override=self.prev_dev)
        else:
            device = global_torch_device(override=torch.device("cuda"))
        if self._model:
            self._model.to(device)
