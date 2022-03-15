#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 25/03/2020
           """

import collections
import hashlib
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, OrderedDict, Sequence

import numpy

__all__ = ["SplitEnum", "SplitIndexer", "train_valid_test_split", "select_split"]

from sorcery import assigned_names


class SplitEnum(Enum):
    """
    Split Enum class for selecting splits"""

    (training, validation, testing) = assigned_names()


class SplitIndexer:
    """Splits dataset in to 3 parts based on percentages, returns indices for the data set sequence"""

    default_split_names = {i: i.value for i in SplitEnum}

    def __init__(
        self,
        dataset_length: int,
        training: float = 0.7,
        validation: float = 0.2,
        testing: float = 0.1,
    ):
        self.total_num = dataset_length
        splits = numpy.array([training, validation, testing])
        self.normalised_split = splits / sum(splits)
        (
            self.training_percentage,
            self.validation_percentage,
            self.testing_percentage,
        ) = self.normalised_split
        self.training_num, self.validation_num, self.testing_num = self.unnormalised(
            dataset_length
        )

    def shuffled_indices(self) -> dict:
        """ """
        split_indices = numpy.random.permutation(self.total_num).tolist()

        return {
            SplitEnum.training: self.select_train_indices(split_indices),
            SplitEnum.validation: self.select_validation_indices(split_indices),
            SplitEnum.testing: self.select_testing_indices(split_indices),
        }

    def select_train_indices(self, ind: Sequence) -> Sequence:
        """ """
        return ind[: self.training_num]

    def select_validation_indices(self, ind: Sequence) -> Sequence:
        """ """
        if self.validation_num:
            if self.testing_num:
                return ind[self.training_num : -self.testing_num]
            return ind[self.training_num :]
        return []

    def select_testing_indices(self, ind: Sequence) -> Sequence:
        """ """
        if self.testing_num:
            return ind[-self.testing_num :]
        return []

    def unnormalised(self, num: int, floored: bool = True) -> numpy.ndarray:
        """

        :param num:
        :type num:
        :param floored:
        :type floored:
        :return:
        :rtype:"""
        unnorm = self.normalised_split * num
        if floored:
            unnorm = numpy.floor(unnorm)
        return unnorm.astype(int)

    def __repr__(self) -> str:
        return str(
            {k: n for k, n in zip(self.default_split_names, self.normalised_split)}
        )

    def select_shuffled_split_indices(self, split: SplitEnum, seed: int = 0) -> object:
        """ """
        numpy.random.seed(seed)
        split_indices = numpy.random.permutation(self.total_num).tolist()

        if split == SplitEnum.training:
            return self.select_train_indices(split_indices)
        elif split == SplitEnum.validation:
            return self.select_validation_indices(split_indices)
        elif split == SplitEnum.testing:
            return self.select_testing_indices(split_indices)
        elif split is None:
            return split_indices
        raise NotImplementedError


def train_valid_test_split(
    categories: Dict[str, Iterable[Path]],
    *,
    validation_percentage: float = 15,  # TODO: ACCEPT AND SQUEEZE ZERO-HUNDRED TO ZERO-ONE range!
    testing_percentage: float = 0,
    verbose: bool = False,
) -> OrderedDict:
    """
    Magic hashing

    :param verbose:
    :type verbose:
    :param categories:
    :param testing_percentage:
    :param validation_percentage:
    :return:"""
    result = collections.OrderedDict()

    if verbose:
        print(categories)

    for c, vs in categories.items():
        training_images = []
        testing_images = []
        validation_images = []

        for file_name in vs:
            b_rep = bytes(re.sub(r"_nohash_.*$", "", f"{c}{file_name.name}"), "utf8")
            percentage_hash = (
                int(hashlib.sha1(b_rep).hexdigest(), 16) % (sys.maxsize + 1)
            ) * (100.0 / sys.maxsize)
            if percentage_hash < validation_percentage + testing_percentage:
                if percentage_hash < testing_percentage:
                    testing_images.append(file_name)
                else:
                    validation_images.append(file_name)
            else:
                training_images.append(file_name)

        result[c] = {
            SplitEnum.training: training_images,
            SplitEnum.validation: validation_images,
            SplitEnum.testing: testing_images,
        }

    return result


def select_split(
    data_cat_split: Dict[Any, Dict[SplitEnum, Sequence]],
    split: SplitEnum,
    verbose: bool = False,
) -> Dict[Any, Sequence]:
    """

    :param verbose:
    :type verbose:
    :param data_cat_split:
    :type data_cat_split:
    :param split:
    :type split:
    :return:
    :rtype:"""
    data = {k: [] for k in data_cat_split.keys()}
    if verbose:
        print(data_cat_split)
    for k, v in data_cat_split.items():
        if verbose:
            print(v[split])
        for item in v[split]:
            data[k].append(item)
    return data


if __name__ == "__main__":

    def asd():
        split_by_p = SplitIndexer(100)
        print(split_by_p.default_split_names)
        print(split_by_p.shuffled_indices())
        print(split_by_p.select_shuffled_split_indices(SplitEnum.training))
        a = split_by_p.select_shuffled_split_indices(None)
        print(a, len(a))

    def uihsad():
        a = None
        if a:
            a = SplitEnum(a)
        print(a)

    uihsad()
