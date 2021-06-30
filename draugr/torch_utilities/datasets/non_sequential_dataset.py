#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List

import numpy
from torch.utils.data import Dataset

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 09/10/2019
           """
__all__ = ["NonSequentialDataset"]


class NonSequentialDataset(Dataset):
    """
    * ``N`` - number of parallel environments
    * ``T`` - number of time steps explored in environments

    Dataset that flattens ``N*T*...`` arrays into ``B*...`` (where ``B`` is equal to ``N*T``) and returns
    such rows
    one by one. So basically we loose information about sequence order and we return
    for example one state, action and reward per row.

    It can be used for ``Model``'s that does not need to keep the order of events like MLP models.

    For ``LSTM`` use another implementation that will slice the dataset differently"""

    def __init__(self, *arrays: numpy.ndarray) -> None:
        """
        :param arrays: arrays to be flattened from ``N*T*...`` to ``B*...`` and returned in each call to get
        item"""
        super().__init__()
        self.arrays = [array.reshape(-1, *array.shape[2:]) for array in arrays]

    def __getitem__(self, index: int) -> List:
        return [array[index] for array in self.arrays]

    def __len__(self):
        return len(self.arrays[0])
