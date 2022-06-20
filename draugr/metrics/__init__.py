#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"

__doc__ = r"""
"""

from pathlib import Path

with open(Path(__file__).parent / "README.md", "r") as this_init_file:
    __doc__ += this_init_file.read()
# #del Path # find a way to sanitise namespace

from .accumulation import *
from .meters import *
from .metric_aggregator import *
from .metric_collection import *
from .metric_summary import *

""" Sanitize namespace # https://stackoverflow.com/questions/61278110/excluding-modules-when-importing-everything-in-init-py
import types
__all__ = [name for name, thing in globals().items()
          if not (name.startswith('_') or isinstance(thing, types.ModuleType))]
del types
"""
