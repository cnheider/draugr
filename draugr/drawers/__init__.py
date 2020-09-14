#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

__doc__ = r"""
"""
import pathlib

with open(pathlib.Path(__file__).parent / "README.md", "r") as this_init_file:
    __doc__ += this_init_file.read()

from .discrete_scroll_plot import *
from .mpldrawer import *
from .series_scroll_plot import *
from .spectral import *
