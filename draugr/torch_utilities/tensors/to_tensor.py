#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Iterable, Sequence, Union

import numpy
import torch
import torchvision
from PIL.Image import Image

__author__ = "Christian Heider Nielsen"
__doc__ = ""
__all__ = ["to_tensor"]

from draugr.torch_utilities.tensors.types import numpy_to_torch_dtype


# @passes_kws_to(torch.Tensor.to)
def to_tensor(
    obj: Union[torch.Tensor, numpy.ndarray, Iterable, Sequence, int, float],
    dtype: Union[
        torch.dtype, object
    ] = None,  # if None, torch.float or equivalent numpy.dtype is used, can be used to force dtype
    device: Union[str, torch.device] = "cpu",
    **kwargs
) -> torch.Tensor:
    """

    :param obj:
    :param dtype:
    :param device:
    :param kwargs:
    :return:"""

    if dtype is None:
        use_dtype = torch.float
    else:
        if dtype == float:
            dtype = torch.float  # TODO: LOOKUP TABLE OF TYPES
        use_dtype = dtype

    # torch.as_tensor()
    if torch.is_tensor(obj):
        if dtype is None:
            use_dtype = obj.dtype
        return obj.to(dtype=use_dtype, device=device, **kwargs)

    if isinstance(obj, Image):
        return torchvision.transforms.functional.to_tensor(obj)

    if isinstance(obj, numpy.ndarray):
        if dtype is None:
            use_dtype = numpy_to_torch_dtype(obj.dtype)
        if torch.is_tensor(obj[0]) and len(obj[0].size()) > 0:
            return torch.stack(obj.tolist()).to(
                dtype=use_dtype, device=device, **kwargs
            )
        return torch.from_numpy(obj).to(dtype=use_dtype, device=device, **kwargs)

    if not isinstance(obj, Sequence):
        if isinstance(obj, set):
            obj = [*obj]
        else:
            obj = [obj]
    elif not isinstance(obj, list) and isinstance(obj, Iterable):
        obj = [*obj]

    if isinstance(obj, list):
        if torch.is_tensor(obj[0]) and len(obj[0].size()) > 0:
            return torch.stack(obj).to(dtype=use_dtype, device=device, **kwargs)
        elif isinstance(obj[0], list):
            obj = [to_tensor(o) for o in obj]
            return torch.stack(obj).to(dtype=use_dtype, device=device, **kwargs)

    return torch.tensor(obj, dtype=use_dtype, device=device, **kwargs)


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

    print(to_tensor((torch.zeros((2, 2)), (torch.zeros((2, 2))))).shape)
    print(to_tensor(([torch.zeros((2, 2)), torch.zeros((2, 2))])).shape)
    print(to_tensor(([torch.zeros((2, 2))], [torch.zeros((2, 2))])).shape)
    print(to_tensor(([[[torch.zeros((2, 2))]]], [[[torch.zeros((2, 2))]]])).shape)
    print(to_tensor(([[torch.zeros((2, 2))], [torch.zeros((2, 2))]])).shape)
    print(to_tensor([(torch.zeros((2, 2))), (torch.zeros((2, 2)))]).shape)
    print(to_tensor({(torch.zeros((2, 2))), (torch.zeros((2, 2)))}).shape)
    print(to_tensor(((torch.zeros((2, 2))), (torch.zeros((2, 2))))).shape)
    print(to_tensor([[[[torch.zeros((2, 2))]], [[torch.zeros((2, 2))]]]]).shape)

    print(
        to_tensor(
            (
                numpy.zeros((2, 2)),
                numpy.zeros((2, 2)),
                numpy.zeros((2, 2)),
                numpy.zeros((2, 2)),
            )
        )
    )

    print(
        (
            *[
                to_tensor(a, device="cpu")
                for a in [
                    numpy.zeros((2, 2)),
                    numpy.zeros((2, 2)),
                    numpy.zeros((2, 2)),
                    numpy.zeros((2, 2)),
                ]
            ],
        )
    )

    print(
        to_tensor(
            [
                to_tensor(numpy.zeros((2, 2))),
                to_tensor(numpy.zeros((2, 2))),
                to_tensor(numpy.zeros((2, 2))),
                to_tensor(numpy.zeros((2, 2))),
            ]
        )
    )

    print(
        (
            *[
                to_tensor(a, device="cpu")
                for a in [
                    to_tensor(numpy.zeros((2, 2))),
                    to_tensor(numpy.zeros((2, 2))),
                    to_tensor(numpy.zeros((2, 2))),
                    to_tensor(numpy.zeros((2, 2))),
                ]
            ],
        )
    )
