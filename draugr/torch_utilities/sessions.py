#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/03/2020
           """

import torch

from draugr.torch_utilities import global_torch_device

__all__ = ["TorchCacheSession", "TorchEvalSession", "TorchTrainSession"]


class TorchCacheSession:
    """
  # speed up evaluating after training finished

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


class TorchEvalSession:
    """
  # speed up evaluating after training finished

  """

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def __enter__(self):
        # self.model.eval()
        self.model.train(False)
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.train(True)


class TorchTrainSession:
    """
  # speed up evaluating after training finished

  """

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def __enter__(self):
        self.model.train(True)
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.train(False)


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

    a()
    b()
    c()
