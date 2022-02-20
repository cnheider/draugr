#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 10/06/2020
           """

import torch
from torch import jit

from warg import AlsoDecorator

__all__ = ["TorchJitSession", "TorchIgnoreJitSession"]


class TorchIgnoreJitSession(AlsoDecorator):
    """
    # Disable torch jit tracing"""

    def __init__(self, no_side_effect: bool = True):
        self._no_side_effect = no_side_effect
        if no_side_effect:
            self.prev_state = jit._enabled

    def __enter__(self):
        jit._enabled = False
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._no_side_effect:
            jit._enabled = self.prev_state
        else:
            jit._enabled = True


class TorchJitSession(AlsoDecorator):
    """
    # Disable torch jit tracing"""

    def __init__(self, enabled=False, no_side_effect: bool = True):
        self._no_side_effect = no_side_effect
        self._effect = enabled
        if no_side_effect:
            self.prev_state = jit._enabled

    def __enter__(self):
        jit._enabled = self._effect
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._no_side_effect:
            jit._enabled = self.prev_state
        else:
            jit._enabled = True


if __name__ == "__main__":

    def a() -> None:
        """
        :rtype: None
        """

        @torch.jit.script
        def scripted_fn(x: torch.Tensor):
            """ """
            for i in range(12):
                x = x + x

            return x

        def fn(x):
            """ """
            x = torch.neg(x)
            # import pdb
            # pdb.set_trace()
            return scripted_fn(x)

        traced_fn = torch.jit.trace(fn, (torch.rand(4, 5),))
        traced_fn(torch.rand(3, 4))

        print(type(traced_fn))  # torch.jit.ScriptFuntcion

        if isinstance(traced_fn, torch.jit.ScriptFunction):
            # See the compiled graph as Python code
            print(traced_fn.code)

    with TorchIgnoreJitSession():
        a()

    a()
