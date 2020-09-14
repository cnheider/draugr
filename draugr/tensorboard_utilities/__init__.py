#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""

import pathlib

with open(pathlib.Path(__file__).parent / "README.md", "r") as this_init_file:
    __doc__ += this_init_file.read()

from .event_export import *

if __name__ == "__main__":
    print(__doc__)
