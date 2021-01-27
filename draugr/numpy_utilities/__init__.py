#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 26/03/2020
           """

import pathlib

with open(pathlib.Path(__file__).parent / "README.md", "r") as this_init_file:
    __doc__ += this_init_file.read()

from .channel_transform import *
from .manipulation import *
from .resize import *
from .datasets import *
from .sampling import *
from .powers import *
from .signal_utilities import *
