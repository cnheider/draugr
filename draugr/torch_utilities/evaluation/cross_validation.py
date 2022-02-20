#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 29/07/2020
           """

__all__ = ["cross_validation_generator"]

from typing import Tuple

import torch
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset

from draugr.torch_utilities.tensors.to_tensor import to_tensor


def cross_validation_generator(
    *datasets: Dataset, n_splits: int = 10
) -> Tuple[Subset, Subset]:
    """
    Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set"""

    cum = ConcatDataset(datasets)
    for train_index, val_index in KFold(n_splits=n_splits).split(cum):
        yield Subset(cum, train_index), Subset(cum, val_index)


if __name__ == "__main__":

    def asdasidoj() -> None:
        """
        :rtype: None
        """
        X = to_tensor([torch.diag(torch.arange(i, i + 2)) for i in range(200)])
        x_train = TensorDataset(X[:100])
        x_val = TensorDataset(X[100:])

        for train, val in cross_validation_generator(x_train, x_val):
            print(len(train), len(val))
            print(train[0], val[0])

    asdasidoj()
