#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 07-05-2021
           """

import os
from pathlib import Path

__all__ = ["latest_file"]


def latest_file(
    directory: Path,
    extension: str = "",
    *,
    recurse: bool = False,
    raise_on_failure: bool = True,
) -> Path:
    """

    :param directory:
    :param extension:
    :param recurse:
    :param raise_on_failure:
    :return:
    """
    a = f"*{extension}"
    if recurse:
        path_gen = directory.rglob(a)
    else:
        path_gen = directory.glob(a)
    list_of_files = list(path_gen)
    if len(list_of_files) == 0:
        msg = f"Found no previous files with extension {extension} in {directory}"
        if raise_on_failure:
            raise FileNotFoundError(msg)
        print(f"{msg}, returning None!")
        return None
    return max(list_of_files, key=os.path.getctime)  # USES CREATION TIME


if __name__ == "__main__":
    print(latest_file(Path(__file__).parent, recurse=True))
