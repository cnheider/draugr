#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 01/03/2020
           """

from abc import abstractmethod
from typing import Sequence

from draugr.torch_utilities.datasets.supervised.supervised_dataset import (
    SupervisedDataset,
)

__all__ = ["CategoricalDataset"]

from warg import OrderedSet


class CategoricalDataset(SupervisedDataset):
    """
    Categorical Dataset for discrete learning problems."""

    @property
    @abstractmethod
    def categories(self) -> OrderedSet[str]:
        """ """
        raise NotImplementedError

    def idx_to_str(self, idx: Sequence[int]) -> Sequence[str]:
        """ """
        return self.categories[idx]

    def str_to_idx(self, s: str) -> int:
        """ """
        return self.categories.index(s)
