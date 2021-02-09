#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""

from pathlib import Path

with open(Path(__file__).parent / "README.md", "r") as this_init_file:
    __doc__ += this_init_file.read()

from .truncation import *
from .signal_statistics import *
from .signal_mixing import *
from .signal_projection import *
from .padding import *
