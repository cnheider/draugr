#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os

__author__ = "Christian Heider Nielsen"

from pathlib import Path

from draugr.numpy_utilities.datasets.splitting import train_valid_test_split
from draugr.numpy_utilities.datasets.defaults import DEFAULT_ACCEPTED_FILE_FORMATS
from typing import Iterable, Union
from warg import drop_unused_kws

__all__ = ["build_deep_categorical_dataset"]


@drop_unused_kws
def build_deep_categorical_dataset(
    directory: Union[Path, str],
    *,
    validation_percentage: float = 15,
    testing_percentage: float = 0,
    extensions: Iterable = DEFAULT_ACCEPTED_FILE_FORMATS,
    is_valid_file: callable = None,
) -> dict:
    """
    Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
    image_directory: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

    Returns:
    An OrderedDict containing an entry for each label subfolder, with images
    split into training, testing, and validation sets within each label.
    The order of items defines the class indices.
      :param is_valid_file:
      :param directory:
      :param validation_percentage:
      :param testing_percentage:
    :param extensions:
    :type extensions:"""

    if not isinstance(directory, Path):
        directory = Path(directory)

    if not directory.exists():
        logging.error(f"Image directory {directory} not found.")
        raise FileNotFoundError(f"Image directory {directory} not found.")

    b = [path for path, sub_dirs, files in os.walk(str(directory)) if len(files) > 0]

    categories_dict = {label.split("/")[-1]: [] for label in b}

    for label, path in {label.split("/")[-1]: label for label in b}.items():
        for sub_directory in sorted([Path(x[0]) for x in os.walk(str(path))]):
            logging.info(f"Looking for images in {sub_directory}")
            for extension in sorted(set(os.path.normcase(ext) for ext in extensions)):
                extension = extension.lstrip(".")
                categories_dict[label].extend(sub_directory / f"*.{extension}")

    return train_valid_test_split(
        categories_dict,
        testing_percentage=testing_percentage,
        validation_percentage=validation_percentage,
    )


if __name__ == "__main__":

    def aiusdj() -> None:
        """
        :rtype: None
        """
        from draugr.visualisation import indent_lines
        from draugr.numpy_utilities.datasets.splitting import SplitEnum

        a = build_deep_categorical_dataset(
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

    aiusdj()
