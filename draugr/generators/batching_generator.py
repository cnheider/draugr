#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Sized, Iterable, Sequence

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/10/2019
           """

import numpy
from torch.utils.data import DataLoader, Dataset

from draugr.torch_utilities.datasets.non_sequential_dataset import NonSequentialDataset
from warg import passes_kws_to

__all__ = [
    "sized_batch",
    "shuffled_batches",
    "random_batches",
    "batch_generator_torch",
    "generator_batch",
]


def sized_batch(sized: Iterable, n: int = 32, drop_not_full: bool = True):
    """

:param sized:
:param n:
:param drop_not_full:
:return:
"""
    if not isinstance(sized, Sized):
        sized = list(sized)
    l = len(sized)
    for ndx in range(0, l, n):
        if drop_not_full and ndx + n > l - 1:
            return
        yield sized[ndx : min(ndx + n, l)]


def random_batches(*args, size, batch_size) -> Sequence:
    for _ in range(size // batch_size):
        rand_ids = numpy.random.randint(0, size, batch_size)
        yield [a[rand_ids] for a in args]


def shuffled_batches(*args, size, batch_size) -> Sequence:
    permutation = numpy.random.permutation(size)
    r = size // batch_size
    for i in range(r):
        perm = permutation[i * batch_size : (i + 1) * batch_size]
        yield [a[perm] for a in args]


@passes_kws_to(DataLoader)
def batch_generator_torch(
    sized: Sized, mini_batches: int = 10, shuffle: bool = True, **kwargs
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


def generator_batch(iterable: Iterable, n: int = 32, drop_not_full: bool = True):
    """

:param iterable:
:param n:
:param drop_not_full:
:return:
"""
    b = []
    i = 0
    for a in iterable:
        b.append(a)
        i += 1
        if i >= n:
            yield b
            b.clear()
            i = 0
    if drop_not_full:
        return
    return b


if __name__ == "__main__":
    arg_num = 4
    size = 12
    mini_batch_size = 5
    a = numpy.ones((arg_num, size))
    for a in shuffled_batches(*a, size=size, batch_size=mini_batch_size):
        print(list(a))
