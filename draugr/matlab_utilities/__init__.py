#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 17-09-2020
           """

from .. import INCLUDE_PROJECT_READMES

if INCLUDE_PROJECT_READMES:
    import pathlib

    with open(pathlib.Path(__file__).parent / "README.md", "r") as this_init_file:
        __doc__ += this_init_file.read()

from .conversion import *
from .singleton_engine import *
