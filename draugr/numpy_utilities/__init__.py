#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 26/03/2020
           """

from pathlib import Path

with open(Path(__file__).parent / "README.md", "r") as this_init_file:
    __doc__ += this_init_file.read()

try:
    from .channel_transform import *
    from .manipulation import *
    from .resize import *
    from .datasets import *
    from .sampling import *
    from .powers import *
    from .signal_utilities import *
    from .raster_grid_2d_masking import *
except ImportError as ix:
    this_package_name = Path(__file__).parent.name
    this_package_reqs = (
        Path(__file__).parent.parent.parent
        / "requirements"
        / f"requirements_{this_package_name}.txt"
    )
    print(
        f"Make sure requirements is installed for {this_package_name}, see {this_package_reqs}"
    )  # TODO: PARSE WHAT is missing and print
    raise ix
