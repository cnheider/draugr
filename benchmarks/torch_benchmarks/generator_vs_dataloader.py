#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time

import numpy
import torch
from torch.utils.data import Dataset

from draugr import batched_recycle
from draugr.torch_utilities import to_tensor_generator

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 28/10/2019
           """


def test_d1():
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
            torch.utils.data.DataLoader(
                numpy.random.sample(data_shape),
                batch_size=batch_size,
                shuffle=True,
                num_workers=1,
                pin_memory=False,
            ),
        ):
            model(a.to(device, dtype=torch.float))

        s3 = time.time()

        print(f"generator: {s2 - s1}")
        print(f"dataloader: {s3 - s2}")


def test_d2():
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
        batched_recycle(numpy.random.sample(data_shape), batch_size),
        device=device,
        preload_next=False,
    )

    dataloader = torch.utils.data.DataLoader(
        numpy.random.sample(data_shape),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
    )

    for _ in range(samples):
        s1 = time.time()
        for _, a in zip(range(batches), generator):
            model(a)

        s2 = time.time()
        for _, a in zip(range(batches), dataloader):
            model(a.to(device, dtype=torch.float))
        s3 = time.time()

        print(f"generator: {s2 - s1}")
        print(f"dataloader: {s3 - s2}")


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

    dataloader = torch.utils.data.DataLoader(
        numpy.random.sample(data_shape),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
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


def test_d4():
    from torchvision.transforms import transforms
    import numpy
    from draugr import inner_map

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
    batch_shape = torch.Size([batch_size, channels_in, 224, 224])

    model = torch.nn.Sequential(
        torch.nn.Conv2d(channels_in, channels_out, (3, 3)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(channels_out, channels_out, (3, 3)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(channels_out, channels_out, (3, 3)),
        torch.nn.ReLU(),
    ).to(device)

    class RandomDataset(Dataset):
        """ """

        def __init__(self):
            self.d = numpy.random.sample(data_shape)

        def __len__(self):
            return len(self.d)

        def __getitem__(self, item):
            return a_transform(self.d[item])

    dataloader = torch.utils.data.DataLoader(
        RandomDataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
    )

    generator = to_tensor_generator(
        inner_map(
            a_transform, batched_recycle(numpy.random.sample(data_shape), batch_size)
        ),
        device=device,
    )

    for _ in range(samples):
        s1 = time.time()
        for _, a in zip(range(batches), dataloader):
            assert batch_shape == a.shape, a.shape
            model(a.to(device, dtype=torch.float))

        s2 = time.time()
        for _, a in zip(range(batches), generator):
            assert batch_shape == a.shape, a.shape
            model(a)

        s3 = time.time()

        print(f"dataloader: {s2 - s1}")
        print(f"generator: {s3 - s2}")


def test_d5():
    from torchvision.transforms import transforms
    import numpy
    from draugr import inner_map

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
    batch_shape = torch.Size([batch_size, channels_in, 224, 224])

    class RandomDataset(Dataset):
        """ """

        def __init__(self):
            self.d = numpy.random.sample(data_shape)

        def __len__(self):
            return len(self.d)

        def __getitem__(self, item):
            return a_transform(self.d[item])

    dataloader = torch.utils.data.DataLoader(
        RandomDataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
    )

    generator = to_tensor_generator(
        inner_map(
            a_transform, batched_recycle(numpy.random.sample(data_shape), batch_size)
        ),
        device=device,
    )

    for _ in range(samples):
        s1 = time.time()
        for _, a in zip(range(batches), generator):
            assert batch_shape == a.shape, a.shape

        s2 = time.time()

        for _, a in zip(range(batches), dataloader):
            assert batch_shape == a.shape, a.shape
        s3 = time.time()

        print(f"generator: {s2 - s1}")
        print(f"dataloader: {s3 - s2}")


def test_d6():
    from torchvision.transforms import transforms
    import numpy
    from draugr import inner_map

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
    batch_shape = torch.Size([batch_size, channels_in, 224, 224])

    class RandomDataset(Dataset):
        """ """

        def __init__(self):
            self.d = numpy.random.sample(data_shape)

        def __len__(self):
            return len(self.d)

        def __getitem__(self, item):
            return a_transform(self.d[item])

    dataloader = torch.utils.data.DataLoader(
        RandomDataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    generator = to_tensor_generator(
        inner_map(
            a_transform, batched_recycle(numpy.random.sample(data_shape), batch_size)
        ),
        device=device,
        preload_next=True,
    )

    for _ in range(samples):
        s1 = time.time()
        for _, a in zip(range(batches), generator):
            assert batch_shape == a.shape, a.shape

        s2 = time.time()

        for _, a in zip(range(batches), dataloader):
            assert batch_shape == a.shape, a.shape
        s3 = time.time()

        print(f"generator: {s2 - s1}")
        print(f"dataloader: {s3 - s2}")


def test_d7():
    import numpy

    channels_in = 3

    samples = 10
    device = "cuda"
    batches = 3
    batch_size = 32
    data_shape = (batches * batch_size, 256, 256, channels_in)
    batch_shape = torch.Size([batch_size, 256, 256, channels_in])
    dtype = torch.float

    class RandomDataset(Dataset):
        """ """

        def __init__(self):
            self.d = numpy.random.sample(data_shape)

        def __len__(self):
            return len(self.d)

        def __getitem__(self, item):
            return self.d[item]

    dataloader = torch.utils.data.DataLoader(
        RandomDataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    generator = to_tensor_generator(
        batched_recycle(numpy.random.sample(data_shape), batch_size),
        device=device,
        preload_next=True,
        dtype=dtype,
    )

    for _ in range(samples):
        s1 = time.time()
        for _, a in zip(range(batches), generator):
            assert batch_shape == a.shape, a.shape

        s2 = time.time()

        for _, a in zip(range(batches), dataloader):
            a = a.to(device, dtype=dtype)
            assert batch_shape == a.shape, a.shape
        s3 = time.time()

        print(f"generator: {s2 - s1}")
        print(f"dataloader: {s3 - s2}")


if __name__ == "__main__":
    test_d7()
