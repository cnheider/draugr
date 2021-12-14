#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/07/2020
           """

import random
from pathlib import Path
from typing import Iterable, Tuple

from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset  # TODO: Do not need to be images

__all__ = ["DictDatasetFolder", "SplitDictDatasetFolder"]

from torchvision.datasets.folder import has_file_allowed_extension

from draugr.numpy_utilities import (
    SplitEnum,
    build_flat_dataset,
    build_shallow_categorical_dataset,
    select_split,
)


class SplitDictDatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

    root/class_x/xxx.ext
    root/class_x/xxy.ext
    root/class_x/xxz.ext

    root/class_y/123.ext
    root/class_y/nsdf3.ext
    root/class_y/asd932_.ext

    Args:
    root (string): Root directory path.
    loader (callable): A function to load a sample given its path.
    extensions (tuple[string]): A list of allowed extensions.
        both extensions and is_valid_file should not be passed.
    transform (callable, optional): A function/transform that takes in
        a sample and returns a transformed version.
        E.g, ``transforms.RandomCrop`` for images.
    target_transform (callable, optional): A function/transform that takes
        in the target and transforms it.
    is_valid_file (callable, optional): A function that takes path of a file
        and check if the file is a valid file (used to check of corrupt files)
        both extensions and is_valid_file should not be passed.

    Attributes:
    _categories (list): List of the class names sorted alphabetically.
    _data_categories (list): List of (sample path, class_index) tuples"""

    def __init__(
        self,
        root: Path,
        loader: DataLoader,
        extensions: Iterable = None,
        transform: callable = None,
        target_transform: callable = None,
        split: SplitEnum = SplitEnum.training,
        valid_percentage: float = 15,
        test_percentage: float = 0,
        is_valid_file: callable = has_file_allowed_extension,
    ):
        super().__init__(
            str(root), transform=transform, target_transform=target_transform
        )
        # TODO: merge Split and non split common in a base class
        self._data_categories = select_split(
            build_shallow_categorical_dataset(
                self.root,
                extensions=extensions,
                testing_percentage=test_percentage,
                validation_percentage=valid_percentage,
            ),
            split,
        )

        if len(self._data_categories) == 0:
            msg = f"Found 0 categories in sub-folders of: {self.root}\n"
            if extensions is not None:
                msg += f"Supported extensions are: {','.join(extensions)}"
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.category_sizes = {k: len(v) for k, v in self._data_categories.items()}
        for cat, cl in self.category_sizes.items():
            if cl == 0:
                print(f"Warning category {cat} has {cl} samples")
        self.category_names = (*self.category_sizes.keys(),)

    def __getitem__(self, index) -> Tuple:
        """

        Non-pure implementation! Index maybe not map to the same item as target randomly sampled

        Args:
        index (int): Index

        Returns:
        tuple: (sample, target) where target is class_index of the target class."""
        target = random.choice(self.category_names)
        return self.sample(target, index)

    def sample(self, target, index) -> Tuple:
        """ """
        sample = self.loader(
            self._data_categories[target][index % self.category_sizes[target]]
        )
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return sum(list(self.category_sizes.values()))


class DictDatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

    root/class_x/xxx.ext
    root/class_x/xxy.ext
    root/class_x/xxz.ext

    root/class_y/123.ext
    root/class_y/nsdf3.ext
    root/class_y/asd932_.ext

    Args:
    root (string): Root directory path.
    loader (callable): A function to load a sample given its path.
    extensions (tuple[string]): A list of allowed extensions.
        both extensions and is_valid_file should not be passed.
    transform (callable, optional): A function/transform that takes in
        a sample and returns a transformed version.
        E.g, ``transforms.RandomCrop`` for images.
    target_transform (callable, optional): A function/transform that takes
        in the target and transforms it.
    is_valid_file (callable, optional): A function that takes path of a file
        and check if the file is a valid file (used to check of corrupt files)
        both extensions and is_valid_file should not be passed.

    Attributes:
    _categories (list): List of the class names sorted alphabetically.
    _data (list): List of (sample path, class_index) tuples"""

    def __init__(
        self,
        root: Path,
        loader: DataLoader,
        extensions: Iterable = None,
        transform: callable = None,
        target_transform: callable = None,
        is_valid_file: callable = has_file_allowed_extension,
    ):
        super().__init__(
            str(root), transform=transform, target_transform=target_transform
        )
        # TODO: merge Split and non split common in a base class
        self._data = build_flat_dataset(
            self.root, extensions=extensions, is_valid_file=is_valid_file
        )

        if len(self._data) == 0:
            msg = f"Found 0 files in sub-folders of: {self.root}\n"
            if extensions is not None:
                msg += f"Supported extensions are: {','.join(extensions)}"
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.category_sizes = {k: len(v) for k, v in self._data.items()}
        self.category_names = (*self.category_sizes.keys(),)

    def __getitem__(self, index) -> Tuple:
        """

        Non-pure implementation! Index maybe not map to the same item as target randomly sampled

        Args:
        index (int): Index

        Returns:
        tuple: (sample, target) where target is class_index of the target class."""
        target = random.choice(self.category_names)
        return self.sample(target, index)

    def sample(self, target, index) -> Tuple:
        """ """
        sample = self.loader(self._data[target][index % self.category_sizes[target]])
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return sum(list(self.category_sizes.values()))


if __name__ == "__main__":
    pass
