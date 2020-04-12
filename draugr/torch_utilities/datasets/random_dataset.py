#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Sized

import torch
from torch.utils.data import Dataset

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """

__all__ = ["RandomDataset"]


class RandomDataset(Dataset):
    """

    """

    def __init__(self, size: Sized, length: int):
        self.len = length
        self.data = torch.randn((length, *size))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


if __name__ == "__main__":
    print(RandomDataset((5, 5), 10)[0])
