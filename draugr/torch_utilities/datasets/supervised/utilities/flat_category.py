#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/07/2020
           """

import os
from pathlib import Path
from typing import Sequence, Tuple, Union

from torchvision.datasets.folder import (
    has_file_allowed_extension,
)  # TODO: allow all types not only images

__all__ = ["build_flat_dataset"]


def build_flat_dataset(
    directory: Path, extensions: Tuple = None, is_valid_file: Sequence = None
) -> dict:
    """

    :param directory:
    :param extensions:
    :param is_valid_file:
    :return:
    """
    categories = [d.name for d in directory.iterdir() if d.is_dir()]
    instances = {k: [] for k in categories}
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )

    if extensions is not None:

        def is_valid_file(x: Union[Path, str]) -> bool:
            return has_file_allowed_extension(str(x), extensions)

    elif is_valid_file is None:
        is_valid_file = lambda a: a is not None

    for target_class in sorted(categories):
        target_dir = directory.expanduser() / target_class
        if not target_dir.is_dir():
            continue
        for root, _, fnames in sorted(os.walk(str(target_dir), followlinks=True)):
            for fname in sorted(fnames):
                path = Path(root) / fname
                if is_valid_file(path):
                    instances[target_class].append(path)
    return instances
