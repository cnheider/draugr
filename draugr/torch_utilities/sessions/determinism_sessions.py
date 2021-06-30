#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28-10-2020
           """

import torch

from warg import AlsoDecorator

__all__ = ["TorchDeterministicSession"]


class TorchDeterministicSession(AlsoDecorator):
    """
    # Disable torch jit tracing"""

    def __init__(self, no_side_effect: bool = True):
        self._no_side_effect = no_side_effect
        if no_side_effect:
            self.prev_state = torch.is_deterministic()

    def __enter__(self):
        torch.set_deterministic(True)
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._no_side_effect:
            torch.set_deterministic(self.prev_state)
        else:
            torch.set_deterministic(False)
