#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

import numpy
import torch
from torch.nn import Conv2d, Linear, Module
from torch.nn.init import calculate_gain, constant_, uniform_, xavier_uniform_

__all__ = ["fan_in_init", "xavier_init", "constant_init", "normal_init"]


def fan_in_init(model: Module):
    """

    :param model:
    :type model:"""
    for m in model.modules():
        if isinstance(m, (Conv2d, Linear)):
            fan_in = m.weight.size(1)
            v = 1.0 / numpy.sqrt(fan_in)
            uniform_(m.weight, -v, v)
            constant_(m.bias, 0)


def xavier_init(model: Module, activation="relu"):
    """

    :param model:
    :type model:
    :param activation:
    :type activation:"""
    gain = calculate_gain(activation)
    for m in model.modules():
        if isinstance(m, (Conv2d, Linear)):
            xavier_uniform_(m.weight, gain=gain)
            constant_(m.bias, 0)


def constant_init(model: Module, constant: float = 1):
    """

    :param model:
    :type model:
    :param constant:
    :type constant:"""
    for m in model.modules():
        if isinstance(m, (Conv2d, Linear)):
            constant_(m.weight, constant)
            constant_(m.bias, constant)


def normal_init(model: Module, mean: float = 0, std: float = 1.0):
    """

    :param model:
    :param mean:
    :param std:
    :return:
    """
    for m in model.modules():
        if isinstance(m, (Conv2d, Linear)):
            torch.nn.init.normal_(m.weight, mean, std)
            torch.nn.init.normal_(m.bias, mean, std)


if __name__ == "__main__":
    a = Linear(3, 3)
    fan_in_init(a)
    xavier_init(a)
    constant_init(a)
