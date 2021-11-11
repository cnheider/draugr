#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 03/08/2020
           """

from pathlib import Path

from draugr import INCLUDE_PROJECT_READMES

if INCLUDE_PROJECT_READMES:
    with open(Path(__file__).parent / "README.md", "r") as this_init_file:
        __doc__ += this_init_file.read()

# from .gtk_settings import *
from .theme_preferences import *
