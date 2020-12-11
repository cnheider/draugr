#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time

import numpy
import pytest
import torch
from draugr import batched_recycle
from draugr.torch_utilities import to_tensor_generator

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/10/2019
           """


@pytest.mark.skip
def test_d1():
    channels_in = 3
    channels_out = 3

    samples = 4
    device = "cuda"
    batches = 10
    batch_size = 32
    data_shape = (batches * batch_size, channels_in, 512, 512)

    model = torch.nn.Sequential(
        torch.nn.Conv2d(channels_in, channels_out, (3, 3)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(channels_out, channels_out, (3, 3)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(channels_out, channels_out, (3, 3)),
        torch.nn.ReLU(),
    ).to("cuda")

    for _ in range(samples):
        s1 = time.time()
        for _, a in zip(
            range(batches),
            to_tensor_generator(
                batched_recycle(numpy.random.sample(data_shape), batch_size),
                device=device,
                preload_next=False,
            ),
        ):

            model(a)

        s2 = time.time()
        for _, a in zip(
            range(batches),
            to_tensor_generator(
                batched_recycle(numpy.random.sample(data_shape), batch_size),
                device=device,
            ),
        ):
            model(a)

        s3 = time.time()

        print(s2 - s1)
        print(s3 - s2)
