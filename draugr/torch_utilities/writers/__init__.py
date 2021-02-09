#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"

__doc__ = r"""

           Created on 18/03/2020
           """

from pathlib import Path

with open(Path(__file__).parent / "README.md", "r") as this_init_file:
    __doc__ += this_init_file.read()

from .tensorboard import *
from .torch_module_writer import *
