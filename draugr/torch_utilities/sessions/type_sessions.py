#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 15/06/2020
           """

import torch

from warg import AlsoDecorator

__all__ = ["DefaultTypeSession"]


# torch.set_default_tensor_type('torch.cuda.FloatTensor') # Legacy


class DefaultTypeSession(AlsoDecorator):
    """
    # speed up evaluating after training finished"""

    def __init__(self, dtype: torch.dtype = torch.float32, no_side_effect: bool = True):
        self._dtype = dtype
        self._no_side_effect = no_side_effect
        if no_side_effect:
            self.prev_state = torch.get_default_dtype()

    def __enter__(self):
        torch.set_default_dtype(self._dtype)
        return True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._no_side_effect:
            torch.set_default_dtype(self.prev_state)
        else:
            torch.set_default_dtype(torch.float32)


if __name__ == "__main__":
    pass
