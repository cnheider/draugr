#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import functools
from typing import Any, Callable

import numpy
import torch

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
__all__ = ["torch_seed"]

from torch import Tensor


def torch_seed(s: int = 72163) -> torch.Generator:
    """
    seeding for reproducibility"""
    generator = torch.manual_seed(s)
    if False:  # Disabled for now
        torch.set_deterministic(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
    return generator


class Seed:
    r"""**Seed PyTorch and numpy.**

    This code is based on PyTorch's reproducibility guide: https://pytorch.org/docs/stable/notes/randomness.html
    Can be used as standard seeding procedure, context manager (seed will be changed only within block) or function decorator.

    **Standard seed**::

          torchfunc.Seed(0) # no surprises I guess

    **Used as context manager**::

      with Seed(1):
          ... # your operations

      print(torch.initial_seed()) # Should be back to seed pre block

    **Used as function decorator**::

      @Seed(1) # Seed only within function
      def foo():
          return 42

    **Important:** It's impossible to put original `numpy` seed after context manager
    or decorator, hence it will be set to original PyTorch's seed.

    Parameters
    ----------
    value: int
          Seed value used in numpy.random_seed and torch.manual_seed. Usually int is provided
    cuda: bool, optional
          Whether to set PyTorch's cuda backend into deterministic mode (setting cudnn.benchmark to `False`
          and cudnn.deterministic to `True`). If `False`, consecutive runs may be slightly different.
          If `True`, automatic autotuning for convolutions layers with consistent input shape will be turned off.
          Default: `False`
    """

    def __init__(self, value: int, cuda: bool = False):
        self.value = value
        self.cuda = cuda

        self.no_side_effect = False
        if self.no_side_effect:
            self._last_seed = torch.initial_seed()
        numpy.random.seed(self.value)
        torch.manual_seed(self.value)
        if False:  # Disabled for now
            torch.set_deterministic(True)
        if self.cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        if self.no_side_effect:
            torch.manual_seed(self._last_seed)
            numpy.random.seed(self._last_seed)
        return False

    def __call__(self, function: Callable) -> callable:
        @functools.wraps(function)
        def decorated(*args, **kwargs) -> Any:
            """ """
            value = function(*args, **kwargs)
            self.__exit__()
            return value

        return decorated


if __name__ == "__main__":

    @Seed(1)  # Seed only within function
    def foo() -> Tensor:
        """
        :rtype: None
        """
        return torch.randint(5, (2, 2))

    def bar() -> Tensor:
        """
        :rtype: None
        """
        with Seed(1):
            return torch.randint(5, (2, 2))

    def buzz() -> Tensor:
        """
        :rtype: None
        """
        Seed(1)
        return torch.randint(5, (2, 2))

    for f in [foo, bar, buzz]:
        print(f())
