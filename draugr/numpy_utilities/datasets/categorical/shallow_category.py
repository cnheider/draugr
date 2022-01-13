#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os

__author__ = "Christian Heider Nielsen"

from pathlib import Path
from typing import Dict, Sequence, Iterable, Union

from draugr.numpy_utilities.datasets.splitting import (
    SplitEnum,
    train_valid_test_split,
)
from draugr.numpy_utilities.datasets.defaults import DEFAULT_ACCEPTED_FILE_FORMATS
from warg import drop_unused_kws

__all__ = ["build_shallow_categorical_dataset"]


@drop_unused_kws
def build_shallow_categorical_dataset(
    directory: Union[Path, str],
    *,
    validation_percentage: float = 15,
    testing_percentage: float = 0,
    extensions: Iterable = DEFAULT_ACCEPTED_FILE_FORMATS,
    is_valid_file: callable = None,
    verbose: bool = False,
) -> Dict[str, Dict[SplitEnum, Sequence]]:
    """
    Returns:
    An OrderedDict containing an entry for each label subfolder, with images
    split into training, testing, and validation sets within each label.
    The order of items defines the class indices.
      :param is_valid_file:
      :param directory:
      :param validation_percentage:
      :param testing_percentage:
      :param extensions:
      :param verbose:
      :return:"""

    if not isinstance(directory, Path):
        directory = Path(directory)

    if not extensions:
        extensions = DEFAULT_ACCEPTED_FILE_FORMATS

    if not directory.exists():
        logging.error(f"directory {directory} not found.")
        raise FileNotFoundError(f"directory {directory} not found.")

    categories_dict = {category: [] for category in next(os.walk(str(directory)))[1]}
    logging.info(f"Found categories {categories_dict.keys()}")

    for c in categories_dict.keys():
        for sub_directory in sorted([Path(x[0]) for x in os.walk(str(directory / c))]):
            logging.info(f"Looking for samples in {sub_directory}")
            for extension in sorted(set(os.path.normcase(ext) for ext in extensions)):
                extension = extension.lstrip(".")
                files = list(sub_directory.glob(f"*.{extension}"))
                logging.info(
                    f"Found {len(files)} samples of type {extension} for category {c}"
                )
                categories_dict[c].extend(files)

    if verbose:
        print(categories_dict)

    return train_valid_test_split(
        categories_dict,
        testing_percentage=testing_percentage,
        validation_percentage=validation_percentage,
    )


if __name__ == "__main__":

    def asd() -> None:
        """
        :rtype: None
        """
        from draugr.visualisation import indent_lines
        from draugr.numpy_utilities.datasets.splitting import SplitEnum

        a = build_shallow_categorical_dataset(
            Path.home() / "Data" / "mnist_png" / "training", testing_percentage=0
        )

        for k in a.keys():
            total = (
                len(a[k][SplitEnum.training])
                + len(a[k][SplitEnum.validation])
                + len(a[k][SplitEnum.testing])
            )
            print(f"\n{k}:")
            print(indent_lines(len(a[k][SplitEnum.training]) / total))
            print(indent_lines(len(a[k][SplitEnum.validation]) / total))
            print(indent_lines(len(a[k][SplitEnum.testing]) / total))

    asd()
