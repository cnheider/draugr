#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import torch

from draugr.torch_utilities.to_tensor import to_tensor

__author__ = "Christian Heider Nielsen"
__all__ = ["TensoriseMixin"]


class TensoriseMixin(object):
    device = "cpu"
    dtype = torch.float

    def __setattr__(self, key, value):
        super().__setattr__(key, to_tensor(value, dtype=self.dtype, device=self.device))
