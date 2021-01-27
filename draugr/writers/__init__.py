#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = """
Created on 27/04/2019

@author: cnheider
"""

import pathlib

with open(pathlib.Path(__file__).parent / "README.md", "r") as this_init_file:
    __doc__ += this_init_file.read()

try:
    from .csv_writer import *
    from .log_writer import *
    from .mixins import *
    from .mock_writer import *
    from .terminal import *
    from .writer import *
    from .standard_tags import *
except ImportError as ix:
    print(
        f"Make sure requirements is installed for {pathlib.Path(__file__).parent.name}"
    )
    raise ix
