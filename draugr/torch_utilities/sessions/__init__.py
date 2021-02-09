#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 11/05/2020
           """

from pathlib import Path

with open(Path(__file__).parent / "README.md", "r") as this_init_file:
    __doc__ += this_init_file.read()

from .cache_sessions import *
from .device_sessions import *
from .jit_sessions import *
from .model_sessions import *
from .type_sessions import *
from .determinism_sessions import *
