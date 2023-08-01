#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from warg import ensure_in_sys_path, find_nearest_ancestral_relative

ensure_in_sys_path(find_nearest_ancestral_relative("draugr").parent)
from draugr.python_utilities import batched_recycle

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/10/2019
           """


def test_batch_generator1():
    a = range(9)
    batch_size = 3
    for i, b in zip(range(18), batched_recycle(a, batch_size)):
        assert [b_ in a for b_ in b]
    assert i == 17


def test_batch_with_label():
    import numpy

    channels_in = 3
    batches = 3
    batch_size = 32
    data_shape = (batches * batch_size, 256, 256, channels_in)

    generator = batched_recycle(
        zip(numpy.random.sample(data_shape), numpy.random.sample(data_shape[0])),
        batch_size,
    )

    for i, a in enumerate(generator):
        print(a)
        break
