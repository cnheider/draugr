#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import torch

from draugr import torch_pi

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
__all__ = [
    "shannon_entropy",
    "log_shannon_entropy",
    "normal_entropy",
    "differential_entropy_gaussian",
    "normal_log_density",
]


def shannon_entropy(prob: torch.Tensor):
    return -torch.sum(prob * torch.log2(prob), -1)


def log_shannon_entropy(log_prob: torch.Tensor):
    return -torch.sum(torch.pow(2, log_prob) * log_prob, -1)
    # return - torch.sum(torch.exp(log_prob) * log_prob, -1)


def normal_entropy(std: torch.Tensor):
    var = std.pow(2)
    ent = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return ent.sum(dim=-1, keepdim=True)


def differential_entropy_gaussian(std: torch.Tensor):
    return torch.log(std * torch.sqrt(2 * torch_pi())) + 0.5


def normal_log_density(x: torch.Tensor, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)
