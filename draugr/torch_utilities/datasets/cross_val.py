#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 29/07/2020
           """

__all__ = []

import torch
from sklearn.model_selection import KFold

"""
You can merge the fixed train/val/test folds you currently have using data.ConcatDataset into a single Dataset. Then you can use data.Subset to randomly split the single dataset into different folds over and over.

"""


def a():
    x_train = torch.rand((20, 2))
    y_train = torch.rand((20, 1))

    kfold = KFold(n_splits=10)

    for train_index, val_index in kfold.split(x_train, y_train):
        print(train_index, val_index)

        x_train_fold = x_train[train_index]
        y_train_fold = y_train[train_index]
        x_test_fold = x_train[val_index]
        y_test_fold = y_train[val_index]


if __name__ == "__main__":
    a()
