#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy
import torch

from draugr.torch_utilities.tensors.to_tensor import to_tensor

__author__ = "Christian Heider Nielsen"
__doc__ = ""


def test_to_tensor_none():
    try:
        tensor = to_tensor(None)
    except:
        return
    assert False


def test_to_tensor_empty_list():
    try:
        tensor = to_tensor([])
    except:
        return
    assert False


def test_to_tensor_empty_tuple():
    try:
        tensor = to_tensor(())
    except:
        return
    assert False


def test_to_tensor_list():
    ref = [0]
    tensor = to_tensor(ref, device="cpu")
    assert tensor.equal(torch.FloatTensor([0]))


def test_to_tensor_multi_list():
    ref = [[0], [1]]
    tensor = to_tensor(ref, device="cpu")
    assert tensor.equal(torch.FloatTensor([[0], [1]]))


def test_to_tensor_tuple():
    ref = (0,)
    tensor = to_tensor(ref, device="cpu")
    assert tensor.equal(torch.FloatTensor([0]))


def test_to_tensor_multi_tuple():
    ref = ([0], [1])
    tensor = to_tensor(ref, device="cpu")
    assert tensor.equal(torch.FloatTensor([[0], [1]]))


def test_to_tensor_from_numpy_tensor():
    ref = torch.from_numpy(numpy.random.sample((1, 2)))
    tensor = to_tensor(ref, dtype=torch.double, device="cpu")
    assert tensor.equal(ref)


def test_to_tensor_float_tensor():
    ref = torch.FloatTensor([0])
    tensor = to_tensor(ref, device="cpu")
    assert tensor.equal(ref)


def test_generator_to_float_tensor():
    s = range(9)
    ref = torch.FloatTensor([*s])
    tensor = to_tensor(s, device="cpu")
    assert tensor.equal(ref)


if __name__ == "__main__":
    pass
