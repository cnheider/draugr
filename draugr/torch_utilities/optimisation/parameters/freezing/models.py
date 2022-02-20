#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 02-12-2020
           """

from contextlib import contextmanager
from itertools import tee

from torch.nn import Module

from draugr.torch_utilities.optimisation.parameters.freezing.parameters import (
    freeze_parameters,
)

__all__ = ["freeze_model", "frozen_model"]


def freeze_model(model: Module, value: bool = None, recurse: bool = True) -> None:
    """

    :param model:
    :type model:
    :param recurse:
    :param value:
    :return:"""
    freeze_parameters(model.parameters(recurse), value)


@contextmanager
def frozen_model(model: Module, recurse: bool = True, enabled: bool = True) -> None:
    """

    :param enabled:
    :type enabled:
    :param model:
    :param recurse:
    :return:"""
    params_1, params_2 = tee(model.parameters(recurse))
    if enabled:
        freeze_parameters(params_1, True)
    yield True
    if enabled:
        freeze_parameters(params_2, False)


if __name__ == "__main__":
    from torch import nn

    def asda() -> None:
        """
        :rtype: None
        """
        a = nn.Linear(10, 5)
        print(a.weight.requires_grad)
        with frozen_model(a):
            print(a.weight.requires_grad)
        print(a.weight.requires_grad)

    asda()
