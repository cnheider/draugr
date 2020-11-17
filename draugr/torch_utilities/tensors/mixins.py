#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from draugr.torch_utilities.tensors import to_tensor

__author__ = "Christian Heider Nielsen"
__all__ = ["TensoriseMixin"]


class TensoriseMixin(object):
    """
    Tensorise attributes at set"""

    device = "cpu"
    dtype = torch.float  # Default values may be monkey patched for other types

    def __setattr__(self, key, value):
        super().__setattr__(key, to_tensor(value, dtype=self.dtype, device=self.device))
