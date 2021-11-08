#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11/05/2020
           """

import torch
from torch.nn import Module

from draugr.torch_utilities import global_torch_device
from warg import AlsoDecorator

__all__ = ["TorchCpuSession", "TorchCudaSession", "TorchDeviceSession"]


class TorchCudaSession(AlsoDecorator):
    """
    Sets global torch devices to cuda if available"""

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
        return False


class TorchCpuSession(AlsoDecorator):
    """
    Sets global torch devices to cpu"""

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
        return False


class TorchDeviceSession(AlsoDecorator):
    """
    Sets global torch devices to cpu"""

    def __init__(
        self, device: torch.device, model: Module = None, no_side_effect: bool = True
    ):
        self._model = model
        self._no_side_effect = no_side_effect
        self._device = device
        if no_side_effect:
            self.prev_dev = global_torch_device()

    def __enter__(self):
        device = global_torch_device(override=self._device)
        if self._model:
            self._model.to(device)
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._no_side_effect:
            device = global_torch_device(override=self.prev_dev)
            if self._model:
                self._model.to(device)
        return False


if __name__ == "__main__":
    with TorchDeviceSession(global_torch_device()):
        pass
