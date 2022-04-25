#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy

import torch
import pytest

__author__ = "Christian Heider Nielsen"
__doc__ = ""

from torch import nn

from draugr.torch_utilities import to_tensor, MLP, constant_init
from warg import prod


def test_single_dim():
    pos_size = (4,)
    a_size = (1,)
    model = MLP(input_shape=pos_size, output_shape=a_size)

    pos_1 = to_tensor(
        numpy.random.rand(64, pos_size[0]), device="cpu", dtype=torch.float
    )
    print(model(pos_1))


def test_hidden_dim():
    pos_size = (4,)
    hidden_size = (2, 3)
    a_size = (2,)
    model = MLP(input_shape=pos_size, hidden_layers=hidden_size, output_shape=a_size)

    pos_1 = to_tensor(
        numpy.random.rand(64, pos_size[0]), device="cpu", dtype=torch.float
    )
    print(model(pos_1))


@pytest.mark.skip
def test_multi_dim():
    """
    TODO: BROKEN!
    """

    pos_size = (2, 3, 2)  # two, 2d tensors, expected flatten
    a_size = (2, 4, 5)
    model = MLP(input_shape=pos_size, output_shape=a_size)

    pos_1 = to_tensor(
        numpy.random.rand(64, numpy.prod(pos_size[1:])), device="cpu", dtype=torch.float
    )
    pos_2 = to_tensor(
        numpy.random.rand(64, numpy.prod(pos_size[1:])), device="cpu", dtype=torch.float
    )
    print(model(pos_1, pos_2))


def test_single_dim2():
    """ """
    pos_size = (4,)
    a_size = (1,)
    model = MLP(input_shape=pos_size, output_shape=a_size)

    pos_1 = to_tensor(
        numpy.random.rand(64, pos_size[0]), device="cpu", dtype=torch.float
    )
    print(model(pos_1)[0].shape)


def test_hidden_dim2():
    """ """
    pos_size = (3,)
    hidden_size = list(range(6, 10))
    a_size = (4,)
    model = MLP(
        input_shape=pos_size,
        hidden_layers=hidden_size,
        output_shape=a_size,
        hidden_layer_activation=torch.nn.Tanh(),
        default_init=None,
    )

    model2 = nn.Sequential(
        *[
            nn.Linear(3, 6),
            nn.Tanh(),
            nn.Linear(6, 7),
            nn.Tanh(),
            nn.Linear(7, 8),
            nn.Tanh(),
            nn.Linear(8, 9),
            nn.Tanh(),
            nn.Linear(9, 4),
        ]
    )
    model3 = nn.Sequential(
        *[
            nn.Linear(3, 6),
            nn.Tanh(),
            nn.Linear(6, 7),
            nn.Tanh(),
            nn.Linear(7, 8),
            nn.Tanh(),
            nn.Linear(8, 9),
            nn.Tanh(),
            nn.Linear(9, 4),
        ]
    )
    constant_init(model, 0.142)
    constant_init(model2, 0.142)
    constant_init(model3, 0.142)
    print(model, model2, model3)

    pos_1 = to_tensor(
        numpy.random.rand(64, pos_size[0]), device="cpu", dtype=torch.float
    )
    print(model(pos_1)[0].shape)
    print(model2(pos_1).shape)
    print(model3(pos_1).shape)


def test_multi_dim_in():
    """ """
    pos_size = (2, 3, 2)
    a_size = (2, 4, 5)
    model = MLP(input_shape=pos_size, output_shape=a_size)

    pos_1 = to_tensor(
        numpy.random.rand(64, pos_size[0]), device="cpu", dtype=torch.float
    )
    pos_2 = to_tensor(
        numpy.random.rand(64, pos_size[1]), device="cpu", dtype=torch.float
    )
    pos_3 = to_tensor(
        numpy.random.rand(64, pos_size[2]), device="cpu", dtype=torch.float
    )
    heads = model(pos_1, pos_2, pos_3)
    for h in heads:
        print(h.shape)


def test_multi_dim_out():
    """ """
    pos_size = (10,)
    a_size = (2, 1)
    model = MLP(input_shape=pos_size, hidden_layers=(100,), output_shape=a_size)

    pos_1 = to_tensor(numpy.random.rand(64, *pos_size), device="cpu", dtype=torch.float)
    res = model(pos_1)
    print(model)
    print(len(res), res[0].shape, res[1].shape)


def test_multi_dim_both():
    """ """
    pos_size = (2, 3)
    a_size = (2, 4, 5)
    model = MLP(input_shape=pos_size, output_shape=a_size)

    pos_1 = to_tensor(
        numpy.random.rand(64, pos_size[0]), device="cpu", dtype=torch.float
    )
    pos_2 = to_tensor(
        numpy.random.rand(64, pos_size[1]), device="cpu", dtype=torch.float
    )
    res = model(pos_1, pos_2)
    print(model)
    print(len(res), res[0].shape, res[1].shape, res[2].shape)


def test_auto():
    """ """
    pos_size = (4,)
    a_size = (2,)
    model = MLP(input_shape=pos_size, output_shape=a_size)

    pos_1 = to_tensor(
        numpy.random.rand(64, pos_size[0]), device="cpu", dtype=torch.float
    )
    res = model(pos_1)
    print(model)
    print(len(res), res[0].shape)


if __name__ == "__main__":
    test_single_dim()
    test_hidden_dim()
    # test_multi_dim()
