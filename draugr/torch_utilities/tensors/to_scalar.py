#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Iterable, Sequence, Union

import numpy
import torch

__author__ = "Christian Heider Nielsen"
__doc__ = ""
__all__ = ["to_scalar"]

from draugr.torch_utilities.tensors.to_tensor import to_tensor
from warg import Number


def to_scalar(
    obj: Union[torch.Tensor, numpy.ndarray, Iterable, Sequence, int, float],
    device: Union[str, torch.device] = "cpu",
    aggregation: callable = torch.mean,
) -> Number:
    """
    Always detaches from computation graph

    default behaviour is obj.cpu().mean().item()

    :param aggregation:
    :param obj:
    :param device:
    :return:"""

    if not torch.is_tensor(obj):
        obj = to_tensor(obj)
    else:
        obj = obj.detach()

    if aggregation:
        obj = aggregation(obj)

    return obj.to(device=device).item()


if __name__ == "__main__":
    print(to_scalar(0))
    print(to_scalar(2.5))
    print(to_scalar(to_tensor(1).item()))
    print(to_scalar(to_tensor(1)))
    print(to_scalar(to_tensor(2.0)))

    print(to_scalar(to_tensor([0.5, 0.5])))
    print(to_scalar(to_tensor([[0.5, 0.5]])))
    print(to_scalar(to_tensor((0.5, 0.5))))
    print(to_scalar(to_tensor(range(10))))
    print(to_scalar(to_tensor(torch.from_numpy(numpy.array([0.5, 0.5])))))
    a = torch.arange(0, 10)
    print(to_scalar(to_tensor(a)))
    print(to_scalar(to_tensor([a, a])))
    print(to_scalar(to_tensor((a, a))))
