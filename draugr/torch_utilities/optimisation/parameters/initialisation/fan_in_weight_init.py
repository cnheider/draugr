#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

import numpy
import torch

from torch.nn import (
    Conv2d,
    Linear,
    Module,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    BatchNorm2d,
    BatchNorm1d,
    BatchNorm3d,
    Dropout,
    Softmax,
    Softplus,
    SiLU,
    ELU,
    Identity,
    Conv1d,
    Conv3d,
)
from torch.nn.init import calculate_gain, constant_, uniform_, xavier_uniform_
from enum import Enum


__all__ = ["fan_in_init", "xavier_init", "constant_init", "normal_init", "ortho_init"]


def fan_in_init(model: Module) -> None:
    """

    :param model:
    :type model:"""
    for m in model.modules():
        if isinstance(m, (Conv2d, Linear)):
            fan_in = m.weight.size(1)
            v = 1.0 / numpy.sqrt(fan_in)
            uniform_(m.weight, -v, v)
            constant_(m.bias, 0)


NonLinearityEnum = {
    Sigmoid: "sigmoid",
    torch.sigmoid: "sigmoid",
    Tanh: "tanh",
    torch.tanh: "tanh",
    ReLU: "relu",
    torch.relu: "relu",
    LeakyReLU: "leaky_relu",
    ELU: "elu",
    SiLU: "silu",
    Softplus: "softplus",
    Softmax: "softmax",
    torch.softmax: "softmax",
    Linear: "linear",
    Identity: "identity",
    torch.conv1d: "conv1d",
    Conv1d: "conv1d",
    torch.conv2d: "conv2d",
    Conv2d: "conv2d",
    torch.conv3d: "conv3d",
    Conv3d: "conv3d",
    torch.conv_transpose1d: "conv_transpose1d",
    torch.nn.ConvTranspose1d: "conv_transpose1d",
    torch.conv_transpose2d: "conv_transpose2d",
    torch.nn.ConvTranspose2d: "conv_transpose2d",
    torch.conv_transpose3d: "conv_transpose3d",
    torch.nn.ConvTranspose3d: "conv_transpose3d",
}


def xavier_init(model: Module, activation: NonLinearityEnum = "relu") -> None:
    """

    :param model:
    :type model:
    :param activation:
    :type activation:"""

    if not isinstance(activation, str):
        activation = NonLinearityEnum[activation].value

    gain = calculate_gain(activation)
    for m in model.modules():
        if isinstance(m, (Conv2d, Linear)):
            xavier_uniform_(m.weight, gain=gain)
            constant_(m.bias, 0)


def constant_init(model: Module, constant: float = 1) -> None:
    """

    :param model:
    :type model:
    :param constant:
    :type constant:"""
    for m in model.modules():
        if isinstance(m, (Conv2d, Linear)):
            constant_(m.weight, constant)
            constant_(m.bias, constant)


def normal_init(model: Module, mean: float = 0, std: float = 1.0) -> None:
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


def ortho_init(module: Module, gain: float = 1) -> None:
    """
    Initialize the weights of a module with orthogonal initialization.
    """
    for m in module.modules():
        if isinstance(m, (Linear, Conv2d)):
            torch.nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


if __name__ == "__main__":
    a = Linear(3, 3)
    fan_in_init(a)
    xavier_init(a)
    constant_init(a)
    ortho_init(a)
