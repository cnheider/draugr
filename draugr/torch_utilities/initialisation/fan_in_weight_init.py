#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

import numpy
from torch.nn import Module, Conv2d, Linear
from torch.nn.init import uniform_, constant_, calculate_gain, xavier_uniform_

__all__ = ["fan_in_init", "xavier_init", "constant_init"]


def fan_in_init(model: Module):
    for m in model.modules():
        if isinstance(m, (Conv2d, Linear)):
            fan_in = m.weight.size(1)
            v = 1.0 / numpy.sqrt(fan_in)
            uniform_(m.weight, -v, v)
            constant_(m.bias, 0)


def xavier_init(model: Module, activation="relu"):
    gain = calculate_gain(activation)
    for m in model.modules():
        if isinstance(m, (Conv2d, Linear)):
            xavier_uniform_(m.weight, gain=gain)
            constant_(m.bias, 0)


def constant_init(model: Module, constant: float = 1):
    for m in model.modules():
        if isinstance(m, (Conv2d, Linear)):
            constant_(m.weight, constant)
            constant_(m.bias, constant)


if __name__ == "__main__":
    a = Linear(3, 3)
    fan_in_init(a)
    xavier_init(a)
    constant_init(a)
