#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/03/2020
           """

import torch

from draugr.torch_utilities import global_torch_device

__all__ = [
    "TorchCacheSession",
    "TorchEvalSession",
    "TorchTrainSession",
    "TorchCudaSession",
    "TorchCpuSession",
]

from warg.decorators.kw_passing import AlsoDecorator


class TorchCacheSession(AlsoDecorator):
    """
# speed up evaluating after training finished
# NOTE: HAS THE SIDE EFFECT OF CLEARING CACHE, NON RECOVERABLE

"""

    def __init__(self, using_cuda: bool = global_torch_device().type == "cuda"):
        self.using_cuda = using_cuda

    def __enter__(self):
        if self.using_cuda:
            torch.cuda.empty_cache()
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.using_cuda:
            torch.cuda.empty_cache()


class TorchEvalSession(AlsoDecorator):
    """
# speed up evaluating after training finished

"""

    def __init__(self, model: torch.nn.Module, no_side_effect: bool = True):
        self.model = model
        self._no_side_effect = no_side_effect
        if no_side_effect:
            self.prev_state = model.training

    def __enter__(self):
        # self.model.eval()
        self.model.train(False)
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._no_side_effect:
            self.model.train(self.prev_state)
        else:
            self.model.train(True)


class TorchTrainSession(AlsoDecorator):
    """
# speed up evaluating after training finished

"""

    def __init__(self, model: torch.nn.Module, no_side_effect: bool = True):
        self.model = model
        self._no_side_effect = no_side_effect
        if no_side_effect:
            self.prev_state = model.training

    def __enter__(self):
        self.model.train(True)
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._no_side_effect:
            self.model.train(self.prev_state)
        else:
            self.model.train(False)


class TorchCudaSession(AlsoDecorator):
    """
  Sets global torch devices to cuda if available
  """

    def __init__(self, no_side_effect: bool = True):
        self._no_side_effect = no_side_effect
        if no_side_effect:
            self.prev_dev = global_torch_device()

    def __enter__(self):
        global_torch_device(override=torch.device("cuda"))
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._no_side_effect:
            global_torch_device(override=self.prev_dev)
        else:
            global_torch_device(override=torch.device("cpu"))


class TorchCpuSession(AlsoDecorator):
    """
  Sets global torch devices to cpu

"""

    def __init__(self, no_side_effect: bool = True):
        self._no_side_effect = no_side_effect
        if no_side_effect:
            self.prev_dev = global_torch_device()

    def __enter__(self):
        global_torch_device(override=torch.device("cpu"))
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._no_side_effect:
            global_torch_device(override=self.prev_dev)
        else:
            global_torch_device(override=torch.device("cuda"))


if __name__ == "__main__":

    def a():
        print(torch.cuda.memory_cached(global_torch_device()))
        with TorchCacheSession():
            torch.tensor([0.0], device=global_torch_device())
            print(torch.cuda.memory_cached(global_torch_device()))
        print(torch.cuda.memory_cached(global_torch_device()))

    def b():
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(0.1))
        print(model.training)
        with TorchEvalSession(model):
            print(model.training)
        print(model.training)

    def c():
        model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(0.1))
        model.eval()
        print(model.training)
        with TorchTrainSession(model):
            print(model.training)
        print(model.training)

    def d():
        print(
            global_torch_device(override=global_torch_device(cuda_if_available=False))
        )
        print(global_torch_device())
        with TorchCudaSession():
            print(global_torch_device())
        print(global_torch_device())

    def e():
        print(global_torch_device(override=global_torch_device(cuda_if_available=True)))
        print(global_torch_device())
        with TorchCpuSession():
            print(global_torch_device())
        print(global_torch_device())

    # a()
    # b()
    # c()
    d()
    e()
