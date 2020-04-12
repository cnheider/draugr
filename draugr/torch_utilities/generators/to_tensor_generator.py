#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Iterable

from torch.utils.data import DataLoader

from draugr.torch_utilities.datasets import NonSequentialDataset
from draugr.torch_utilities.tensors import to_tensor
from warg import passes_kws_to
import numpy

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/10/2019
           """
__all__ = [
    "to_tensor_generator",
    "batch_generator_torch",
    "to_device_tensor_iterator_shitty",
]


@passes_kws_to(to_tensor)
def to_tensor_generator(iterable: Iterable, preload_next: bool = False, **kwargs):
    """

:param iterable:
:param preload_next:
:param kwargs:
:return:
"""
    if preload_next:
        iterable_iter = iter(iterable)
        current = to_tensor(next(iterable_iter), **kwargs)
        kwargs["non_blocking"] = True
        while current is not None:
            next_ = to_tensor(next(iterable_iter), **kwargs)
            yield current
            current = next_
    else:
        for a in iterable:
            yield to_tensor(a, **kwargs)
    return


def to_device_tensor_iterator_shitty(data_iterator, device):
    while True:
        yield (to_tensor(i, device=device) for i in next(data_iterator))


@passes_kws_to(DataLoader)
def batch_generator_torch(
    sized: numpy.ndarray, mini_batches: int = 10, shuffle: bool = True, **kwargs
) -> DataLoader:
    """

:param dataset:
:param mini_batches:
:param shuffle:
:param kwargs:
:return:
"""

    dataset = NonSequentialDataset(sized)
    return DataLoader(
        dataset, batch_size=len(dataset) // mini_batches, shuffle=shuffle, **kwargs
    )


if __name__ == "__main__":
    from torchvision.transforms import transforms
    import numpy
    from draugr.generators.recycling_generator import batched_recycle
    from draugr import inner_map

    def s():

        a = iter(numpy.random.sample((5, 5, 5)))
        for a in to_device_tensor_iterator_shitty(a, "cpu"):
            d, *_ = a
            print(d)
            print(type(d))

    a_transform = transforms.Compose(
        [
            transforms.ToPILImage("RGB"),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    channels_in = 3
    channels_out = 3

    samples = 10
    device = "cuda"
    batches = 3
    batch_size = 32
    data_shape = (batches * batch_size, 256, 256, channels_in)

    generator = to_tensor_generator(
        inner_map(
            a_transform, batched_recycle(numpy.random.sample(data_shape), batch_size)
        ),
        device=device,
    )

    for i, a in enumerate(generator):
        print(a)
        break
