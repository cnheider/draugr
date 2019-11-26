#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Iterable, Sequence, Union

import numpy
import torch

from draugr.torch_utilities.initialisation.device import global_torch_device

__author__ = "Christian Heider Nielsen"
__doc__ = ""
__all__ = ["to_tensor"]

# @passes_kws_to(torch.Tensor.to)
def to_tensor(
    obj: Union[torch.Tensor, numpy.ndarray, Iterable, int, float],
    dtype=torch.float,
    device=global_torch_device(),
    **kwargs
):
    if torch.is_tensor(obj):
        return obj.to(dtype=dtype, device=device, **kwargs)

    if isinstance(obj, numpy.ndarray):
        if torch.is_tensor(obj[0]) and len(obj[0].size()) > 0:
            return torch.stack(obj.tolist()).to(dtype=dtype, device=device, **kwargs)
        return torch.from_numpy(obj).to(dtype=dtype, device=device, **kwargs)

    if not isinstance(obj, Sequence):
        obj = [obj]
    elif not isinstance(obj, list) and isinstance(obj, Iterable):
        obj = [*obj]

    if isinstance(obj, list):
        if torch.is_tensor(obj[0]) and len(obj[0].size()) > 0:
            return torch.stack(obj).to(dtype=dtype, device=device, **kwargs)

    return torch.tensor(obj, dtype=dtype, device=device, **kwargs)


if __name__ == "__main__":
    print(to_tensor(1))
    print(to_tensor(2.0))
    print(to_tensor([0.5, 0.5]))
    print(to_tensor([[0.5, 0.5]]))
    print(to_tensor((0.5, 0.5)))
    print(to_tensor(range(10)))
    print(to_tensor(torch.from_numpy(numpy.array([0.5, 0.5]))))
    a = torch.arange(0, 10)
    print(to_tensor(a))
    print(to_tensor([a, a]))
    print(to_tensor((a, a)))
