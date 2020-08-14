#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os

__author__ = "Christian Heider Nielsen"

from pathlib import Path
from typing import Dict, Sequence

from draugr import indent_lines
from draugr.torch_utilities.datasets.supervised.splitting import (
    Split,
    train_valid_test_split,
)
from draugr.torch_utilities.datasets.supervised.utilities.defaults import (
    ACCEPTED_IMAGE_FORMATS,
)
from warg import drop_unused_kws

__all__ = ["build_shallow_categorical_dataset"]


@drop_unused_kws
def build_shallow_categorical_dataset(
    image_directory: Path,
    *,
    validation_percentage: float = 15,
    testing_percentage: float = 0,
    extensions=ACCEPTED_IMAGE_FORMATS,
    verbose: bool = False,
) -> Dict[str, Dict[Split, Sequence]]:
    """
Returns:
An OrderedDict containing an entry for each label subfolder, with images
split into training, testing, and validation sets within each label.
The order of items defines the class indices.
"""

    if not extensions:
        extensions = ACCEPTED_IMAGE_FORMATS

    if not image_directory.exists():
        logging.error(f"Image directory {image_directory} not found.")
        raise FileNotFoundError(f"Image directory {image_directory} not found.")

    categories_dict = {
        category: [] for category in next(os.walk(str(image_directory)))[1]
    }
    logging.info(f"Found categories {categories_dict.keys()}")

    for c in categories_dict.keys():
        for sub_directory in sorted(
            [Path(x[0]) for x in os.walk(str(image_directory / c))]
        ):
            logging.info(f"Looking for images in {sub_directory}")
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
    a = build_shallow_categorical_dataset(
        Path.home() / "Data" / "mnist_png" / "training", testing_percentage=0
    )

    for k in a.keys():
        total = (
            len(a[k][Split.Training])
            + len(a[k][Split.Validation])
            + len(a[k][Split.Testing])
        )
        print(f"\n{k}:")
        print(indent_lines(len(a[k][Split.Training]) / total))
        print(indent_lines(len(a[k][Split.Validation]) / total))
        print(indent_lines(len(a[k][Split.Testing]) / total))
