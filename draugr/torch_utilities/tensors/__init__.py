#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 21/02/2020
           """

from pathlib import Path

with open(Path(__file__).parent / "README.md", "r") as this_init_file:
    __doc__ += this_init_file.read()

from .info import *
from .mixins import *
from .normalise import *
from .reshaping import *
from .tensor_container import *
from .to_scalar import *
from .to_tensor import *
from .dimension_order import *
from .types import *
