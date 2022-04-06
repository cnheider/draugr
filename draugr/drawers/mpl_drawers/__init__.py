#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

__doc__ = r"""
"""

from pathlib import Path

with open(Path(__file__).parent / "README.md", "r") as this_init_file:
    __doc__ += this_init_file.read()

try:
    from .spectral import *
    from .mpldrawer import *
    from .discrete_scroll_plot import *
    from .series_scroll_plot import *
    from .distribution_plot import *
    from .image_stream_plot import *
except ImportError as ix:
    print(f"Make sure requirements is installed for {Path(__file__).parent.name}")
    raise ix
