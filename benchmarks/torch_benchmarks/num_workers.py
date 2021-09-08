#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 08-12-2020
           """

import time

import numpy
import torch

from draugr import WorkerSession, batched_recycle
from draugr.torch_utilities import to_tensor_generator


def test_d3():
    channels_in = 3
    channels_out = 3

    samples = 10
    device = "cuda"
    batches = 3
    batch_size = 32
    data_shape = (batches * batch_size, channels_in, 512, 512)

    model = torch.nn.Sequential(
        torch.nn.Conv2d(channels_in, channels_out, (3, 3)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(channels_out, channels_out, (3, 3)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(channels_out, channels_out, (3, 3)),
        torch.nn.ReLU(),
    ).to(device)

    generator = to_tensor_generator(
        batched_recycle(numpy.random.sample(data_shape), batch_size), device=device
    )

    with WorkerSession(0.3) as num_workers:
        dataloader = torch.utils.data.DataLoader(
            numpy.random.sample(data_shape),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        for _ in range(samples):
            s1 = time.time()
            for _, a in zip(range(batches), dataloader):
                model(a.to(device, dtype=torch.float))

            s2 = time.time()
            for _, a in zip(range(batches), generator):
                model(a)

            s3 = time.time()

            print(f"dataloader: {s2 - s1}")
            print(f"generator: {s3 - s2}")


if __name__ == "__main__":
    test_d3()
