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
from typing import Dict, Iterable, OrderedDict, Sequence

import numpy

__all__ = ["Split", "SplitByPercentage", "train_valid_test_split"]


class Split(Enum):
    """

"""

    Training = "training"
    Validation = "validation"
    Testing = "testing"


class SplitByPercentage:
    """

"""

    default_split_names = {i: i.value for i in Split}

    def __init__(self, dataset_length: int, training=0.7, validation=0.2, testing=0.1):
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

    def unnormalised(self, num: int, floored: bool = True) -> numpy.ndarray:
        """

:param num:
:type num:
:param floored:
:type floored:
:return:
:rtype:
"""
        unnorm = self.normalised_split * num
        if floored:
            unnorm = numpy.floor(unnorm)
        return unnorm.astype(int)

    def __repr__(self) -> str:
        return str(
            {k: n for k, n in zip(self.default_split_names, self.normalised_split)}
        )


def train_valid_test_split(
    categories: Dict[str, Iterable[Path]],
    *,
    validation_percentage: float = 15,
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
:return:
"""
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
            Split.Training: training_images,
            Split.Validation: validation_images,
            Split.Testing: testing_images,
        }

    return result


def select_split(
    data_cat_split, split: Split, verbose: bool = False
) -> Dict[str, Sequence]:
    """

:param verbose:
:type verbose:
:param data_cat_split:
:type data_cat_split:
:param split:
:type split:
:return:
:rtype:
"""
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
    print(SplitByPercentage(9).default_split_names)
