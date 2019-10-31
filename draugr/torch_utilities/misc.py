#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import torch

from draugr.torch_utilities.to_tensor import to_tensor

__author__ = "Christian Heider Nielsen"


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def identity(x):
    return x


class TensoriseMixin(object):
    device = "cpu"
    dtype = torch.float

    def __setattr__(self, key, value):
        super().__setattr__(key, to_tensor(value, dtype=self.dtype, device=self.device))
