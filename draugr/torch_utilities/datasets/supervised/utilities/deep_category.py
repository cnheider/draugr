#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os

__author__ = "Christian Heider Nielsen"

from pathlib import Path

from draugr.torch_utilities.datasets.supervised.splitting import train_valid_test_split
from draugr.torch_utilities.datasets.supervised.utilities.defaults import (
    ACCEPTED_IMAGE_FORMATS,
)
from warg import drop_unused_kws

__all__ = ["build_deep_categorical_dataset"]


@drop_unused_kws
def build_deep_categorical_dataset(
    image_directory: Path,
    *,
    validation_percentage: float = 15,
    testing_percentage: float = 0,
    extensions=ACCEPTED_IMAGE_FORMATS,
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
  :param image_directory:
  :param validation_percentage:
  :param testing_percentage:
:param extensions:
:type extensions:
"""

    if not image_directory.exists():
        logging.error(f"Image directory {image_directory} not found.")
        raise FileNotFoundError(f"Image directory {image_directory} not found.")

    b = [
        path
        for path, sub_dirs, files in os.walk(str(image_directory))
        if len(files) > 0
    ]

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
